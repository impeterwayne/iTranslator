# translate_strings.py
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import sys
import json
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

# ---------- UTF-8 everywhere (fixes typical “utf8” errors) ----------
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
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

# --- config toggles ---
def _truthy(s: str) -> bool:
    return str(s).strip().lower() not in ("0", "false", "no", "")

# If 1 (default), restore masked specials as numeric entities like &#x1F525;
# If 0, restore them as literal glyphs.
PREFER_NUMERIC_ENTITIES = _truthy(os.getenv("PREFER_NUMERIC_ENTITIES", "0"))

# If 1 (default), escape apostrophes in text nodes as \' (outside tags/comments/CDATA)
ESCAPE_APOSTROPHES = _truthy(os.getenv("ESCAPE_APOSTROPHES", "1"))

# Get supported languages from environment variables and parse JSON array
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

def create_directory_structure():
    base_dir = 'output'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    for lang_code in SUPPORTED_LANGUAGES:
        folder_name = f'values-{lang_code}'
        os.makedirs(os.path.join(base_dir, folder_name), exist_ok=True)
    return base_dir

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    except Exception:
        return max(1, len(string) // 4)

def strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()

# --------------------- prefilter: drop non-translatables ----------------------
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

def empty_resources_skeleton(original_xml: str) -> str:
    m = re.search(r'<\s*resources\b[^>]*>', original_xml, flags=re.IGNORECASE)
    start = m.group(0) if m else '<resources>'
    return f'{start}\n</resources>\n'
# ------------------------------------------------------------------------------

# -------------------- masking: icons / emoji / symbols / entities -------------
HEX_ENTITY_RE = re.compile(r'&\s*#\s*x\s*([0-9A-Fa-f]{2,6})\s*;')  # normalize sloppy hex
DEC_ENTITY_RE = re.compile(r'&\s*#\s*([0-9]{2,7})\s*;')

# Private Use Areas
PUA_RANGES = (
    (0xE000, 0xF8FF),       # BMP PUA
    (0xF0000, 0xFFFFD),     # Plane 15 PUA
    (0x100000, 0x10FFFD),   # Plane 16 PUA
)

# Broad emoji/pictograph coverage (supersets the usual emoji blocks)
EMOJI_PICTO_RANGES = (
    (0x1F000, 0x1FFFF),     # general emoji/pictographs, cards, etc.
    (0x2600,  0x26FF),      # Misc Symbols (☀, ☎, ♻)
    (0x2700,  0x27BF),      # Dingbats (✨, ✓, ✗)
    (0x2B00,  0x2BFF),      # Misc Symbols and Arrows (⬅, ⬆, ⬇, ⏺)
)

# Regional indicators and skin tone modifiers (redundant with 1F000..1FFFF but explicit)
EXTRA_EMOJI_RANGES = (
    (0x1F1E6, 0x1F1FF),     # Regional indicator symbols
    (0x1F3FB, 0x1F3FF),     # Fitzpatrick skin tones
)

# Variation selectors & ZWJ and Tags (emoji composition)
VARIATION_SELECTOR_RANGES = (
    (0xFE00, 0xFE0F),       # VS1..VS16 (FE0E/FE0F commonly used)
    (0xE0100, 0xE01EF),     # Variation Selector Supplement
)
TAG_RANGES = (
    (0xE0000, 0xE007F),     # Tags (plane 14)
)
EMOJI_HELPERS = {0x200D}    # ZWJ

# Common singleton emoji code points outside big blocks
EMOJI_SINGLETONS = {
    0x00A9,   # ©
    0x00AE,   # ®
    0x203C,   # ‼
    0x2049,   # ⁉
    0x2122,   # ™
    0x2139,   # ℹ
    0x3030,   # 〰
    0x303D,   # 〽
    0x3297,   # ㊗
    0x3299,   # ㊙
    0x24C2,   # Ⓜ
    0x231A,   # ⌚
    0x231B,   # ⌛
    0x2328,   # ⌨
    0x23CF,   # ⏏
    0x23E9, 0x23EA, 0x23EB, 0x23EC, 0x23ED, 0x23EE, 0x23EF,
    0x23F0, 0x23F1, 0x23F2, 0x23F3,
    0x23F8, 0x23F9, 0x23FA,
    0x2B50,   # ⭐
    0x2B55,   # ⭕
    0x20E3,   # combining keycap
}

def _in_ranges(cp: int, ranges) -> bool:
    for a, b in ranges:
        if a <= cp <= b:
            return True
    return False

def _is_special_cp(cp: int) -> bool:
    # Treat as "special" if it's likely to be an icon/emoji/pictograph
    # or an emoji formatting/control needed for correct rendering.
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

    # 1) Protect existing hex entities first to avoid double-handling
    def _hex_repl(m):
        cp = int(m.group(1), 16)
        key = f"__HEXU{cp:X}__"
        mapping[key] = f"&#x{cp:X};"   # normalized spacing/casing
        return key
    s = HEX_ENTITY_RE.sub(_hex_repl, text)

    # 2) Protect existing decimal entities
    def _dec_repl(m):
        cp = int(m.group(1))
        key = f"__DECU{cp:X}__"
        mapping[key] = f"&#{cp};"
        return key
    s = DEC_ENTITY_RE.sub(_dec_repl, s)

    # 3) Protect literal specials by scanning
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
    masked = "".join(out)
    return masked, mapping

# Force any leftover emoji / non-BMP / PUA / specials to hex entities
def force_hex_entities_for_specials(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if _is_special_cp(cp) or cp >= 0x10000:
            out.append(f"&#x{cp:X};")
        else:
            out.append(ch)
    return "".join(out)

def restore_placeholders(text: str, mapping: dict) -> str:
    # Normalize any broken entities like "& # x e156 ;"
    text = HEX_ENTITY_RE.sub(lambda m: f"&#x{m.group(1)};", text)
    text = DEC_ENTITY_RE.sub(lambda m: f"&#{m.group(1)};", text)
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text

# -------------------- apostrophe escaping (outside of tags) -------------------
_TAG_OR_CDATA_OR_COMMENT = re.compile(r'(<!\[CDATA\[.*?\]\]>|<!--.*?-->|<[^>]+>)', re.DOTALL)
_APOS_NEEDS_ESCAPE = re.compile(r"(?<!\\)'")  # bare ' not already escaped

def escape_apostrophes_outside_tags(xml_text: str) -> str:
    """
    Escapes apostrophes in text nodes only (not inside tags/comments/CDATA):
      Don't -> Don\'t
    Leaves things like <font color='#A005FF'> untouched.
    """
    parts = _TAG_OR_CDATA_OR_COMMENT.split(xml_text)
    out = []
    for part in parts:
        if not part:
            continue
        if _TAG_OR_CDATA_OR_COMMENT.match(part):
            out.append(part)  # tag/comment/CDATA: leave as-is
        else:
            out.append(_APOS_NEEDS_ESCAPE.sub(r"\\'", part))
    return "".join(out)
# ------------------------------------------------------------------------------

async def translate_file(source_file: str, target_language: str, llm, provider: str) -> str:
    with open(source_file, 'r', encoding='utf-8') as f:
        original = f.read()

    content, removed = remove_nontranslatables(original)
    if removed:
        print(f"[{target_language}] Excluded {removed} non-translatable item(s).")

    if not re.search(r'<\s*(string|string-array|plurals)\b', content, flags=re.IGNORECASE):
        print(f"[{target_language}] No translatable entries left after filtering.")
        return empty_resources_skeleton(original)

    # Mask specials/entities before sending to the model
    masked_content, mask_map = protect_specials(content, prefer_numeric=PREFER_NUMERIC_ENTITIES)

    app_name = os.getenv("APP_NAME", "Android").strip() or "Android"
    token_est = num_tokens_from_string(masked_content, "cl100k_base")
    print(f"[{target_language}] Input tokens (est): {token_est}")

    prompt = f"""You are a professional application translator. Translate this Android strings.xml file to the language with ISO code: {target_language}, in the context of a {app_name} application.

Important rules:
1. Preserve all XML tags, attributes, and structure exactly.
2. Preserve all entities like &appname; and &author;.
3. Only translate the text values enclosed within XML tags (and CDATA if present).
4. Keep all special characters and formatting.
5. Do NOT alter or split any placeholder tokens matching __U[0-9A-F]+__, __HEXU[0-9A-F]+__, or __DECU[0-9A-F]+__ (these are protected glyphs/entities).
6. Preserve printf-style placeholders (e.g., %s, %1$s), escaped quotes, and HTML-like markup.
7. Be accurate; do not invent translations.
8. Output ONLY the translated XML file content (no explanations, no markdown fences).
9. BE CONCISE
Here is the file to translate:

{masked_content}

Translated file:
"""

    try:
        with ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: llm.invoke(prompt)
            )

        if provider == "gemini":
            if hasattr(response, "content"):
                if isinstance(response.content, list):
                    response_text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in response.content
                    )
                else:
                    response_text = str(response.content)
            else:
                response_text = str(response)
        else:
            response_text = str(response)

        translated_content = strip_code_fences(response_text) or response_text

        # Restore masked icons/entities back to original form (or numeric entities)
        translated_content = restore_placeholders(translated_content, mask_map)

        # Finalize: escape apostrophes in text nodes only (leave HTML-like tags alone)
        if ESCAPE_APOSTROPHES:
            translated_content = escape_apostrophes_outside_tags(translated_content)

        # Optional: force stray specials to hex entities
        # translated_content = force_hex_entities_for_specials(translated_content)

        return translated_content

    except Exception as e:
        print(f"[{target_language}] Error in translation: {e}")
        return empty_resources_skeleton(original)

async def translate_language(source_file: str, lang_code: str, base_dir: str):
    lang_code = lang_code.strip().strip('"')
    print(f"\nStarting translation for {lang_code}...")
    provider = os.getenv("TRANSLATION_PROVIDER", "openai").lower()

    if provider == "openai":
        llm = OpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "16000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "360")),
        )
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(
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

    translated_content = await translate_file(source_file, lang_code, llm, provider)

    folder_name = f'values-{lang_code}'
    output_file = os.path.join(base_dir, folder_name, 'strings.xml')
    with open(output_file, 'w', encoding='utf-8', newline="") as f:
        f.write(translated_content)

    print(f"Completed translation for {lang_code}: {output_file}")

async def translate_strings_file(source_file: str):
    base_dir = create_directory_structure()

    # You set this high; keep it if your provider limits allow it.
    concurrency = int(os.getenv("MAX_CONCURRENT_TRANSLATIONS", "100"))
    sem = asyncio.Semaphore(concurrency)

    async def sem_task(lang_code: str):
        async with sem:
            await translate_language(source_file, lang_code, base_dir)

    tasks = [sem_task(lang_code) for lang_code in SUPPORTED_LANGUAGES]
    await asyncio.gather(*tasks)

    archive_path = shutil.make_archive('output', 'zip', base_dir)
    print(f"\nTranslation completed! Files are saved in '{archive_path}'")

if __name__ == '__main__':
    source_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.getenv("STRINGS_XML_PATH", "strings.xml")
    )
    asyncio.run(translate_strings_file(source_path))
