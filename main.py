# translate_strings.py
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import sys
import json
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import unicodedata

# ---------- UTF-8 everywhere (fixes typical "utf8" errors) ----------
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors='replace')
    sys.stderr.reconfigure(encoding="utf-8", errors='replace')
except Exception:
    pass

# Windows: prefer selector policy for broad compatibility with run_in_executor
if os.name == "nt":
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
# -------------------------------------------------------------------

load_dotenv()

# ------------------------- config toggles ---------------------------
def _truthy(s: str) -> bool:
    return str(s).strip().lower() not in ("0", "false", "no", "")

# If 1, restore masked specials as numeric entities like &#x1F525;; if 0, restore as literal glyphs
PREFER_NUMERIC_ENTITIES = _truthy(os.getenv("PREFER_NUMERIC_ENTITIES", "0"))

# If 1 (default), escape apostrophes in text nodes as \' (outside tags/comments/CDATA)
ESCAPE_APOSTROPHES = _truthy(os.getenv("ESCAPE_APOSTROPHES", "1"))

# Token budget per chunk (rough); actual tokenization best-effort fallback
CHUNK_TOKEN_LIMIT = int(os.getenv("CHUNK_TOKEN_LIMIT", "4000"))

# Per-language concurrency for chunk translation (prevents rate-limit spikes)
CHUNK_CONCURRENCY = max(1, int(os.getenv("CHUNK_CONCURRENCY", "8")))

# Cross-language concurrency (how many languages processed in parallel)
LANG_CONCURRENCY = max(1, int(os.getenv("MAX_CONCURRENT_TRANSLATIONS", "50")))
# -------------------------------------------------------------------

# -------------------------- languages list --------------------------
def load_supported_languages():
    try:
        langs_raw = os.getenv("SUPPORTED_LANGUAGES", "[]")
        SUPPORTED_LANGUAGES = json.loads(langs_raw)
        if not isinstance(SUPPORTED_LANGUAGES, list) or not SUPPORTED_LANGUAGES:
            raise ValueError("SUPPORTED_LANGUAGES is empty")
        return [str(lang).strip().strip('"') for lang in SUPPORTED_LANGUAGES]
    except json.JSONDecodeError:
        SUPPORTED_LANGUAGES = [
            lang.strip() for lang in os.getenv("SUPPORTED_LANGUAGES", "").split(",") if lang.strip()
        ]
        if not SUPPORTED_LANGUAGES:
            raise ValueError("SUPPORTED_LANGUAGES environment variable is not set or is empty")
        return [lang.strip('"') for lang in SUPPORTED_LANGUAGES]

SUPPORTED_LANGUAGES = load_supported_languages()
# -------------------------------------------------------------------

# ----------------------- directory management -----------------------
def create_directory_structure():
    base_dir = 'output'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    for lang_code in SUPPORTED_LANGUAGES:
        folder_name = f'values-{lang_code}'
        os.makedirs(os.path.join(base_dir, folder_name), exist_ok=True)
    return base_dir
# -------------------------------------------------------------------

# -------------------------- token counting --------------------------
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    except Exception:
        # conservative fallback
        return max(1, len(string) // 4)
# -------------------------------------------------------------------

# ------------------- code fences / skeleton utils -------------------
def strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()

def empty_resources_skeleton(original_xml: str) -> str:
    prolog_m = re.search(r'^\s*<\?xml[^>]*\?>', original_xml, flags=re.IGNORECASE | re.MULTILINE)
    prolog = prolog_m.group(0) if prolog_m else '<?xml version="1.0" encoding="utf-8"?>'
    m = re.search(r'<\s*resources\b[^>]*>', original_xml, flags=re.IGNORECASE)
    start = m.group(0) if m else '<resources>'
    return f'{prolog}\n{start}\n</resources>\n'
# -------------------------------------------------------------------

# -------------------- Unicode normalization & cleanup ---------------
def normalize_unicode(text: str) -> str:
    """
    Normalize unicode and remove/replace problematic characters that cause �
    """
    # First, normalize to NFC (canonical composition)
    text = unicodedata.normalize('NFC', text)
    
    # Remove replacement characters if they already exist
    text = text.replace('\ufffd', '')
    
    # Replace control characters (except tab, newline, carriage return)
    cleaned = []
    for char in text:
        code = ord(char)
        # Keep printable chars, tabs, newlines
        if char in ('\t', '\n', '\r') or not unicodedata.category(char).startswith('C'):
            cleaned.append(char)
        # Replace other control chars with space
        elif unicodedata.category(char).startswith('C'):
            cleaned.append(' ')
        else:
            cleaned.append(char)
    
    return ''.join(cleaned)

def safe_encode_decode(text: str) -> str:
    """
    Safely handle encoding/decoding to prevent � characters
    """
    try:
        # Encode to UTF-8 and decode back, replacing errors
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    except Exception:
        return text

def remove_invalid_xml_chars(text: str) -> str:
    """
    Remove characters that are invalid in XML 1.0
    Valid ranges: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    """
    def is_valid_xml_char(char):
        code = ord(char)
        return (
            code == 0x9 or code == 0xA or code == 0xD or
            (0x20 <= code <= 0xD7FF) or
            (0xE000 <= code <= 0xFFFD) or
            (0x10000 <= code <= 0x10FFFF)
        )
    
    return ''.join(char for char in text if is_valid_xml_char(char))
# -------------------------------------------------------------------

# --------------------- remove non-translatables ---------------------
_NONTRANS_BLOCK = re.compile(
    r'<(?P<tag>string|plurals|string-array)\b[^>]*?\btranslatable\s*=\s*(["\'])false\2[^>]*>'
    r'[\s\S]*?'
    r'</(?P=tag)\s*>',
    flags=re.IGNORECASE
)
_NONTRANS_SELF = re.compile(
    r'<(?:string|plurals|string-array)\b[^>]*?\btranslatable\s*=\s*(["\'])false\1[^>]*/\s*>',
    flags=re.IGNORECASE
)

def remove_nontranslatables(xml_text: str):
    filtered, n1 = _NONTRANS_SELF.subn('', xml_text)
    filtered, n2 = _NONTRANS_BLOCK.subn('', filtered)
    removed = n1 + n2
    filtered = re.sub(r'\n[ \t]*\n[ \t]*\n+', '\n\n', filtered)
    return filtered, removed
# -------------------------------------------------------------------

# ------------- masking: icons / emoji / entities protection ----------
HEX_ENTITY_RE = re.compile(r'&\s*#\s*x\s*([0-9A-Fa-f]{2,6})\s*;')  # normalize sloppy hex
DEC_ENTITY_RE = re.compile(r'&\s*#\s*([0-9]{2,7})\s*;')

PUA_RANGES = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)
EMOJI_PICTO_RANGES = (
    (0x1F000, 0x1FFFF),
    (0x2600,  0x26FF),
    (0x2700,  0x27BF),
    (0x2B00,  0x2BFF),
)
EXTRA_EMOJI_RANGES = (
    (0x1F1E6, 0x1F1FF),
    (0x1F3FB, 0x1F3FF),
)
VARIATION_SELECTOR_RANGES = (
    (0xFE00, 0xFE0F),
    (0xE0100, 0xE01EF),
)
TAG_RANGES = (
    (0xE0000, 0xE007F),
)
EMOJI_HELPERS = {0x200D}  # ZWJ

EMOJI_SINGLETONS = {
    0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D, 0x3297, 0x3299,
    0x24C2, 0x231A, 0x231B, 0x2328, 0x23CF, 0x23E9, 0x23EA, 0x23EB, 0x23EC, 0x23ED,
    0x23EE, 0x23EF, 0x23F0, 0x23F1, 0x23F2, 0x23F3, 0x23F8, 0x23F9, 0x23FA, 0x2B50, 0x2B55, 0x20E3,
}

def _in_ranges(cp: int, ranges) -> bool:
    for a, b in ranges:
        if a <= cp <= b:
            return True
    return False

def _is_special_cp(cp: int) -> bool:
    return (
        _in_ranges(cp, PUA_RANGES)
        or _in_ranges(cp, EMOJI_PICTO_RANGES)
        or _in_ranges(cp, EXTRA_EMOJI_RANGES)
        or _in_ranges(cp, VARIATION_SELECTOR_RANGES)
        or _in_ranges(cp, TAG_RANGES)
        or (cp in EMOJI_HELPERS)
        or (cp in EMOJI_SINGLETONS)
    )

def protect_specials(text: str, prefer_numeric: bool = True):
    """
    Replace:
      - literal PUA/emoji/pictograph/special symbols with placeholders __UXXXX__/__U1F9XX__
      - any existing numeric entities (hex/dec) with placeholders
    Return (masked_text, mapping).
    """
    mapping = {}

    def _hex_repl(m):
        cp = int(m.group(1), 16)
        key = f"__HEXU{cp:X}__"
        mapping[key] = f"&#x{cp:X};"
        return key
    s = HEX_ENTITY_RE.sub(_hex_repl, text)

    def _dec_repl(m):
        cp = int(m.group(1))
        key = f"__DECU{cp:X}__"
        mapping[key] = f"&#{cp};"
        return key
    s = DEC_ENTITY_RE.sub(_dec_repl, s)

    out = []
    for ch in s:
        cp = ord(ch)
        if _is_special_cp(cp):
            key = f"__U{cp:X}__"
            if key not in mapping:
                mapping[key] = f"&#x{cp:X};" if prefer_numeric else ch
            out.append(key)
        else:
            out.append(ch)
    return "".join(out), mapping

def restore_placeholders(text: str, mapping: dict) -> str:
    text = HEX_ENTITY_RE.sub(lambda m: f"&#x{m.group(1)};", text)
    text = DEC_ENTITY_RE.sub(lambda m: f"&#{m.group(1)};", text)
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text
# -------------------------------------------------------------------

# --------------- apostrophe escaping (outside of tags) --------------
_TAG_OR_CDATA_OR_COMMENT = re.compile(r'(<!\[CDATA\[.*?\]\]>|<!--.*?-->|<[^>]+>)', re.DOTALL)
_APOS_NEEDS_ESCAPE = re.compile(r"(?<!\\)'")  # bare ' not already escaped

def escape_apostrophes_outside_tags(xml_text: str) -> str:
    parts = _TAG_OR_CDATA_OR_COMMENT.split(xml_text)
    out = []
    for part in parts:
        if not part:
            continue
        if _TAG_OR_CDATA_OR_COMMENT.match(part):
            out.append(part)
        else:
            out.append(_APOS_NEEDS_ESCAPE.sub(r"\\'", part))
    return "".join(out)
# -------------------------------------------------------------------

# ---------------------- chunking the source XML ---------------------
RES_PATTERN = re.compile(
    r'(<plurals\b[^>]*?>[\s\S]*?<\/plurals>|<string-array\b[^>]*?>[\s\S]*?<\/string-array>|<string\b[^>]*?>[\s\S]*?<\/string>)',
    re.IGNORECASE
)

def chunk_resources(xml_text: str, encoding_name: str = "cl100k_base", token_limit: int = CHUNK_TOKEN_LIMIT):
    """
    Extract <string>, <string-array>, <plurals> blocks and pack them into chunks by token limit.
    """
    resources = RES_PATTERN.findall(xml_text)
    if not resources:
        return []

    chunks, current, cur_tokens = [], [], 0
    for res in resources:
        rt = num_tokens_from_string(res, encoding_name)
        if cur_tokens and cur_tokens + rt > token_limit:
            chunks.append("\n".join(current))
            current, cur_tokens = [], 0
        current.append(res)
        cur_tokens += rt
    if current:
        chunks.append("\n".join(current))
    return chunks
# -------------------------------------------------------------------

# -------------------------- LLM helpers -----------------------------
def build_llm():
    provider = os.getenv("TRANSLATION_PROVIDER", "openai").lower()
    if provider == "openai":
        return "openai", OpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "16000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.5")),  
            timeout=float(os.getenv("OPENAI_TIMEOUT", "360")),
        )
    elif provider == "gemini":
        return "gemini", ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "65536")),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
            model_kwargs={
                "generation_config": {
                    "thinking_config": {"thinking_budget": 0}
                }
            },
        )
    else:
        raise ValueError(f"Unknown TRANSLATION_PROVIDER: {provider}")

def llm_translate_chunk(masked_chunk: str, target_language: str, app_name: str, llm, provider: str) -> str:
    prompt = f"""You are an expert localization specialist for mobile applications. Translate this Android strings XML snippet to ISO language code: {target_language} for the "{app_name}" application.

CRITICAL TRANSLATION GUIDELINES:

1. QUALITY STANDARDS:
   - Use professional, native-level language appropriate for the target locale
   - Maintain consistent terminology throughout all translations
   - Adapt idioms and expressions naturally to the target culture
   - Consider context and intended user audience (formal vs casual tone)
   - Use proper grammar, spelling, and punctuation conventions for the target language

2. TECHNICAL PRESERVATION (DO NOT MODIFY):
   - Preserve ALL XML tags, attributes, and structure EXACTLY as provided
   - Keep all named entities unchanged: &appname;, &author;, etc.
   - Maintain all placeholders EXACTLY: __U[0-9A-F]+__, __HEXU[0-9A-F]+__, __DECU[0-9A-F]+__
   - Keep printf-style format specifiers in correct order: %s, %1$s, %2$d, etc.
   - Preserve escaped characters: \\n, \\t, \\', \\"
   - Keep inline HTML tags intact: <b>, <i>, <u>, <a href="">, etc.

3. TRANSLATION SCOPE:
   - ONLY translate text nodes and CDATA content
   - DO NOT translate: attribute values, entity names, placeholder tokens
   - Adapt text length appropriately for UI constraints when possible

4. OUTPUT FORMAT:
   - Provide ONLY the translated XML (no explanations, no markdown code blocks)
   - Ensure valid, well-formed XML structure
   - Use ONLY valid UTF-8 characters that are compatible with XML 1.0
   - Start output immediately with XML content

5. LOCALIZATION BEST PRACTICES:
   - Use region-appropriate date/time formats in descriptions if mentioned
   - Adapt units of measurement when culturally appropriate
   - Consider text expansion/contraction ratios for UI elements
   - Maintain brand voice and tone consistent with app identity

6. CHARACTER ENCODING:
   - Use proper UTF-8 characters for the target language
   - Avoid any characters that could cause encoding issues
   - Ensure all special characters are properly represented

XML Content to Translate:
{masked_chunk}

Professional Translation:"""

    # Blocking call -> run in executor when used from async context
    resp = llm.invoke(prompt)
    if provider == "gemini":
        if hasattr(resp, "content"):
            if isinstance(resp.content, list):
                text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in resp.content)
            else:
                text = str(resp.content)
        else:
            text = str(resp)
    else:
        text = str(resp)
    
    # Clean and normalize the response
    text = strip_code_fences(text)
    text = normalize_unicode(text)
    text = safe_encode_decode(text)
    text = remove_invalid_xml_chars(text)
    
    # Trim to first '<'
    ix = text.find('<')
    result = text[ix:] if ix >= 0 else text
    
    return result
# -------------------------------------------------------------------

# ------------------------ per-chunk pipeline ------------------------
async def translate_one_chunk(raw_chunk: str, target_language: str, llm, provider: str, prefer_numeric: bool, app_name: str) -> str:
    # Normalize input first
    raw_chunk = normalize_unicode(raw_chunk)
    
    # Mask specials/entities
    masked, mask_map = protect_specials(raw_chunk, prefer_numeric=prefer_numeric)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        translated_masked = await loop.run_in_executor(
            executor,
            lambda: llm_translate_chunk(masked, target_language, app_name, llm, provider)
        )

    # Restore masked icons/entities
    translated = restore_placeholders(translated_masked, mask_map)
    
    # Final cleanup
    translated = normalize_unicode(translated)
    translated = safe_encode_decode(translated)
    translated = remove_invalid_xml_chars(translated)
    
    return translated.strip()
# -------------------------------------------------------------------

# -------------------------- per-language run ------------------------
async def translate_language(filtered_xml: str, original_xml: str, lang_code: str, base_dir: str):
    lang_code = lang_code.strip().strip('"')
    print(f"\nStarting translation for {lang_code}...")

    # Early exit: if nothing to translate
    if not re.search(r'<\s*(string|string-array|plurals)\b', filtered_xml, flags=re.IGNORECASE):
        print(f"[{lang_code}] No translatable entries left after filtering.")
        folder_name = f'values-{lang_code}'
        output_file = os.path.join(base_dir, folder_name, 'strings.xml')
        with open(output_file, 'w', encoding='utf-8', newline="", errors='replace') as f:
            f.write(empty_resources_skeleton(original_xml))
        print(f"Completed (empty skeleton) for {lang_code}: {output_file}")
        return

    provider, llm = build_llm()
    app_name = os.getenv("APP_NAME", "Android").strip() or "Android"

    # Chunking
    chunks = chunk_resources(filtered_xml, "cl100k_base", CHUNK_TOKEN_LIMIT)
    print(f"[{lang_code}] {len(chunks)} chunk(s) to translate.")

    # Throttle chunk-level concurrency
    sem = asyncio.Semaphore(CHUNK_CONCURRENCY)

    async def _sem_chunk(c):
        async with sem:
            tok_est = num_tokens_from_string(c, "cl100k_base")
            print(f"[{lang_code}] Chunk tokens (est): {tok_est}")
            return await translate_one_chunk(
                c, lang_code, llm, provider, PREFER_NUMERIC_ENTITIES, app_name
            )

    translated_chunks = await asyncio.gather(*[_sem_chunk(c) for c in chunks])

    # Rebuild final XML
    body = "\n".join(translated_chunks)

    # Keep original XML prolog if present, else add a standard UTF-8 prolog
    prolog_m = re.search(r'^\s*<\?xml[^>]*\?>', original_xml, flags=re.IGNORECASE | re.MULTILINE)
    prolog = prolog_m.group(0) if prolog_m else '<?xml version="1.0" encoding="utf-8"?>'

    # Keep original <resources ...> start tag if present
    m = re.search(r'<\s*resources\b[^>]*>', original_xml, flags=re.IGNORECASE)
    start_tag = m.group(0) if m else '<resources>'

    final_xml = f'{prolog}\n{start_tag}\n{body}\n</resources>\n'

    # Finalize: escape apostrophes in text nodes only
    if ESCAPE_APOSTROPHES:
        final_xml = escape_apostrophes_outside_tags(final_xml)
    
    # Final normalization and cleanup
    final_xml = normalize_unicode(final_xml)
    final_xml = safe_encode_decode(final_xml)
    final_xml = remove_invalid_xml_chars(final_xml)

    # Save with proper encoding
    folder_name = f'values-{lang_code}'
    output_file = os.path.join(base_dir, folder_name, 'strings.xml')
    
    # Write with UTF-8 BOM to ensure proper encoding recognition
    with open(output_file, 'w', encoding='utf-8-sig', newline="", errors='replace') as f:
        # Remove BOM if it exists in the string (we're using utf-8-sig for the file)
        content = final_xml.lstrip('\ufeff')
        f.write(content)

    print(f"Completed translation for {lang_code}: {output_file}")
# -------------------------------------------------------------------

# --------------------------- orchestrator ---------------------------
def read_and_filter_source(source_file: str):
    # Try multiple encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    original = None
    
    for encoding in encodings:
        try:
            with open(source_file, 'r', encoding=encoding, errors='replace') as f:
                original = f.read()
            print(f"Successfully read source file with {encoding} encoding")
            break
        except Exception as e:
            continue
    
    if original is None:
        raise ValueError(f"Could not read source file {source_file} with any known encoding")
    
    # Normalize the source
    original = normalize_unicode(original)
    original = safe_encode_decode(original)
    
    filtered, removed = remove_nontranslatables(original)
    if removed:
        print(f"Excluded {removed} non-translatable item(s).")
    return original, filtered

async def translate_strings_file(source_file: str):
    if not os.path.exists(source_file):
        print(f"Error: Source file not found at '{source_file}'")
        return

    base_dir = create_directory_structure()
    original_xml, filtered_xml = read_and_filter_source(source_file)

    # Process languages with bounded concurrency
    sem = asyncio.Semaphore(LANG_CONCURRENCY)

    async def _sem_lang(lc: str):
        async with sem:
            await translate_language(filtered_xml, original_xml, lc, base_dir)

    await asyncio.gather(*[_sem_lang(lang_code) for lang_code in SUPPORTED_LANGUAGES])

    archive_path = shutil.make_archive('output', 'zip', base_dir)
    print(f"\nTranslation completed! Files are saved in '{archive_path}'")

# -------------------------------------------------------------------

if __name__ == '__main__':
    source_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.getenv("STRINGS_XML_PATH", "strings.xml")
    )
    asyncio.run(translate_strings_file(source_path))