# server.py — FastAPI backend for iTranslator
import os
import sys
import json
import uuid
import shutil
import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ---------- UTF-8 everywhere ----------
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

# Import translation engine pieces
from languages import LANGUAGE_CODE_TO_NAME, get_language_name
from main import (
    load_supported_languages,
    remove_nontranslatables,
    chunk_resources,
    build_llm,
    translate_one_chunk,
    protect_specials,
    restore_placeholders,
    escape_apostrophes_outside_tags,
    normalize_unicode,
    safe_encode_decode,
    remove_invalid_xml_chars,
    num_tokens_from_string,
    strip_code_fences,
    empty_resources_skeleton,
    PREFER_NUMERIC_ENTITIES,
    ESCAPE_APOSTROPHES,
    CHUNK_TOKEN_LIMIT,
    CHUNK_CONCURRENCY,
    LANG_CONCURRENCY,
    RES_PATTERN,
)

app = FastAPI(title="iTranslator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory job store ───
jobs: dict = {}

class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# ─── Models ───
class TranslationConfig(BaseModel):
    app_name: str = "Android"
    provider: str = "openai"
    languages: list[str] = []

# ─── Routes ───

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/languages")
async def get_languages():
    """Return all available languages with their codes."""
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in LANGUAGE_CODE_TO_NAME.items()
        ]
    }


@app.get("/api/config")
async def get_config():
    """Return current .env config."""
    return {
        "app_name": os.getenv("APP_NAME", "Android"),
        "provider": os.getenv("TRANSLATION_PROVIDER", "openai"),
        "supported_languages": load_supported_languages(),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "has_gemini_key": bool(os.getenv("GEMINI_API_KEY", "").strip()),
    }


@app.post("/api/translate")
async def start_translation(
    file: UploadFile = File(...),
    app_name: str = Form("Android"),
    provider: str = Form("openai"),
    languages: str = Form("[]"),
    openai_api_key: str = Form(""),
    gemini_api_key: str = Form(""),
):
    """Upload strings.xml and start translation job."""
    # Parse languages
    try:
        lang_list = json.loads(languages)
    except json.JSONDecodeError:
        lang_list = [l.strip() for l in languages.split(",") if l.strip()]

    if not lang_list:
        lang_list = load_supported_languages()

    # Save uploaded file
    job_id = str(uuid.uuid4())
    job_dir = Path(tempfile.gettempdir()) / "itranslator" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    source_path = job_dir / "strings.xml"
    content = await file.read()
    source_path.write_bytes(content)

    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Create language folders
    for lc in lang_list:
        (output_dir / f"values-{lc}").mkdir(exist_ok=True)

    jobs[job_id] = {
        "id": job_id,
        "status": JobStatus.PENDING,
        "app_name": app_name,
        "provider": provider,
        "languages": lang_list,
        "total_languages": len(lang_list),
        "completed_languages": 0,
        "current_language": None,
        "errors": [],
        "progress": [],
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "job_dir": str(job_dir),
        "openai_api_key": openai_api_key.strip() if openai_api_key else "",
        "gemini_api_key": gemini_api_key.strip() if gemini_api_key else "",
    }

    # Start background translation
    asyncio.create_task(_run_translation(job_id))

    return {"job_id": job_id, "total_languages": len(lang_list)}


async def _run_translation(job_id: str):
    """Background task that translates all languages."""
    job = jobs[job_id]
    job["status"] = JobStatus.RUNNING

    try:
        source_path = job["source_path"]
        output_dir = job["output_dir"]
        lang_list = job["languages"]
        app_name = job["app_name"]

        # Read source
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        original_xml = None
        for enc in encodings:
            try:
                with open(source_path, "r", encoding=enc, errors="replace") as f:
                    original_xml = f.read()
                break
            except Exception:
                continue
        if original_xml is None:
            job["status"] = JobStatus.FAILED
            job["errors"].append("Could not read source file")
            return

        original_xml = normalize_unicode(original_xml)
        original_xml = safe_encode_decode(original_xml)
        filtered_xml, removed = remove_nontranslatables(original_xml)

        job["progress"].append(f"Excluded {removed} non-translatable item(s).")

        # Count total string entries for info
        total_entries = len(RES_PATTERN.findall(filtered_xml))
        job["total_entries"] = total_entries

        # Temporarily set env vars for provider and API keys
        old_provider = os.environ.get("TRANSLATION_PROVIDER")
        old_openai_key = os.environ.get("OPENAI_API_KEY")
        old_gemini_key = os.environ.get("GEMINI_API_KEY")
        os.environ["TRANSLATION_PROVIDER"] = job["provider"]
        if app_name:
            os.environ["APP_NAME"] = app_name
        # Override keys if user provided them
        if job.get("openai_api_key"):
            os.environ["OPENAI_API_KEY"] = job["openai_api_key"]
        if job.get("gemini_api_key"):
            os.environ["GEMINI_API_KEY"] = job["gemini_api_key"]

        # Process languages with concurrency
        sem = asyncio.Semaphore(LANG_CONCURRENCY)

        async def translate_lang(lc: str):
            async with sem:
                try:
                    job["current_language"] = lc
                    lang_name = get_language_name(lc)
                    job["progress"].append(f"Starting {lang_name} ({lc})...")

                    import re
                    if not re.search(r'<\s*(string|string-array|plurals)\b', filtered_xml, flags=re.IGNORECASE):
                        folder = os.path.join(output_dir, f"values-{lc}")
                        out_file = os.path.join(folder, "strings.xml")
                        with open(out_file, "w", encoding="utf-8", newline="", errors="replace") as f:
                            f.write(empty_resources_skeleton(original_xml))
                        job["completed_languages"] += 1
                        job["progress"].append(f"✓ {lang_name} ({lc}) — empty skeleton")
                        return

                    provider_name, llm = build_llm()
                    chunks = chunk_resources(filtered_xml, "cl100k_base", CHUNK_TOKEN_LIMIT)
                    job["progress"].append(f"[{lc}] {len(chunks)} chunk(s)")

                    chunk_sem = asyncio.Semaphore(CHUNK_CONCURRENCY)

                    async def sem_chunk(c):
                        async with chunk_sem:
                            return await translate_one_chunk(
                                c, lc, llm, provider_name, PREFER_NUMERIC_ENTITIES, app_name
                            )

                    translated_chunks = await asyncio.gather(*[sem_chunk(c) for c in chunks])

                    body = "\n".join(translated_chunks)
                    import re as _re
                    prolog_m = _re.search(r'^\s*<\?xml[^>]*\?>', original_xml, flags=_re.IGNORECASE | _re.MULTILINE)
                    prolog = prolog_m.group(0) if prolog_m else '<?xml version="1.0" encoding="utf-8"?>'
                    m = _re.search(r'<\s*resources\b[^>]*>', original_xml, flags=_re.IGNORECASE)
                    start_tag = m.group(0) if m else "<resources>"
                    final_xml = f"{prolog}\n{start_tag}\n{body}\n</resources>\n"

                    if ESCAPE_APOSTROPHES:
                        final_xml = escape_apostrophes_outside_tags(final_xml)
                    final_xml = normalize_unicode(final_xml)
                    final_xml = safe_encode_decode(final_xml)
                    final_xml = remove_invalid_xml_chars(final_xml)

                    folder = os.path.join(output_dir, f"values-{lc}")
                    os.makedirs(folder, exist_ok=True)
                    out_file = os.path.join(folder, "strings.xml")
                    with open(out_file, "w", encoding="utf-8-sig", newline="", errors="replace") as f:
                        content = final_xml.lstrip("\ufeff")
                        f.write(content)

                    job["completed_languages"] += 1
                    job["progress"].append(f"✓ {lang_name} ({lc}) completed")

                except Exception as e:
                    job["errors"].append(f"Error translating {lc}: {str(e)}")
                    job["completed_languages"] += 1
                    job["progress"].append(f"✗ {lc} failed: {str(e)}")

        await asyncio.gather(*[translate_lang(lc) for lc in lang_list])

        # Restore env
        if old_provider is not None:
            os.environ["TRANSLATION_PROVIDER"] = old_provider
        if old_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = old_openai_key
        elif job.get("openai_api_key"):
            os.environ.pop("OPENAI_API_KEY", None)
        if old_gemini_key is not None:
            os.environ["GEMINI_API_KEY"] = old_gemini_key
        elif job.get("gemini_api_key"):
            os.environ.pop("GEMINI_API_KEY", None)

        # Create ZIP
        zip_path = shutil.make_archive(str(Path(job["job_dir"]) / "output"), "zip", output_dir)
        job["zip_path"] = zip_path
        job["status"] = JobStatus.COMPLETED
        job["current_language"] = None
        job["progress"].append("🎉 All translations completed!")

    except Exception as e:
        job["status"] = JobStatus.FAILED
        job["errors"].append(str(e))
        job["progress"].append(f"❌ Job failed: {str(e)}")


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "total_languages": job["total_languages"],
        "completed_languages": job["completed_languages"],
        "current_language": job["current_language"],
        "errors": job["errors"],
        "progress": job["progress"],
        "total_entries": job.get("total_entries", 0),
    }


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """SSE endpoint for real-time progress."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_idx = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                break

            # Send new progress messages
            if len(job["progress"]) > last_idx:
                for msg in job["progress"][last_idx:]:
                    yield f"data: {json.dumps({'type': 'progress', 'message': msg, 'completed': job['completed_languages'], 'total': job['total_languages'], 'status': job['status']})}\n\n"
                last_idx = len(job["progress"])

            if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED):
                yield f"data: {json.dumps({'type': 'done', 'status': job['status'], 'errors': job['errors']})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download the translated ZIP."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    zip_path = job.get("zip_path")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="translations.zip",
    )


@app.get("/api/jobs/{job_id}/preview/{lang_code}")
async def preview_translation(job_id: str, lang_code: str):
    """Preview a single language translation."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    output_dir = job["output_dir"]
    file_path = os.path.join(output_dir, f"values-{lang_code}", "strings.xml")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Translation not found")
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        content = f.read()
    return {"lang_code": lang_code, "content": content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
