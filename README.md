# 🌍 iTranslator

AI-powered Android `strings.xml` translator that supports **90+ languages** using OpenAI GPT or Google Gemini. Includes both a **CLI tool** and a modern **Web UI**.

---

## ✨ Features

- 🤖 **Dual AI Providers** — OpenAI GPT and Google Gemini, switchable per job
- 🌐 **90+ Languages** — Full ISO 639 language support with human-readable names
- ⚡ **Parallel Translation** — Translate all languages simultaneously (28+ at once)
- 🎯 **Context-Aware** — App name and locale-specific context for accurate translations
- 🔒 **Preserves XML Structure** — Keeps placeholders (`%s`, `%1$d`), HTML tags, entities, and special characters intact
- 📦 **ZIP Export** — Downloads all translations as a ready-to-use ZIP with `values-{lang}/strings.xml` folders
- 🚫 **Auto-Filters** — Skips `translatable="false"` entries automatically
- 🔑 **Flexible API Keys** — Use `.env` file or enter keys directly in the Web UI
- 📋 **Live Progress** — Real-time translation progress with per-language status
- 👁️ **Preview** — Preview translated XML before downloading

## 📁 Project Structure

```
iTranslator/
├── main.py              # CLI translation engine
├── server.py            # FastAPI backend (Web UI)
├── languages.py         # Language code → name mapping
├── requirements.txt     # Python dependencies
├── .env                 # Configuration (API keys, languages, etc.)
├── strings.xml          # Source file for CLI mode
├── frontend/            # Vite + React Web UI
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── UploadZone.jsx
│   │       ├── ConfigPanel.jsx
│   │       ├── LanguageSelector.jsx
│   │       ├── TranslationProgress.jsx
│   │       └── ResultView.jsx
│   ├── index.html
│   └── vite.config.js
└── output/              # CLI output directory
```

---

## 🚀 Getting Started

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/)
- An API key from [OpenAI](https://platform.openai.com/api-keys) or [Google AI Studio](https://aistudio.google.com/apikey)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
# API Keys (at least one required)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Which provider to use: "openai" or "gemini"
TRANSLATION_PROVIDER=gemini

# Your app name (helps the AI translate more accurately)
APP_NAME=AI Photo Editor

# Target languages (JSON array of ISO codes)
SUPPORTED_LANGUAGES='[
    "ar", "bn", "cs", "de", "el", "es", "fa", "fi", "fil", "fr",
    "hi", "hr", "in", "it", "ja", "ko", "ms", "nl", "pl", "pt",
    "ru", "sk", "sr", "sv", "th", "tr", "vi", "zh"
]'
```

### 3. Run

```bash
python start.py
```

This starts both the backend and frontend together. Open **http://localhost:5173** in your browser.  
Press `Ctrl+C` to stop both servers. Frontend dependencies are auto-installed on first run.

<details>
<summary>Start servers separately (alternative)</summary>

```bash
# Terminal 1 — Backend
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
npm install       # first time only
npx vite --port 5173
```

</details>

### Workflow

1. **Upload** your `strings.xml` file (drag & drop or click to browse)
2. **Configure** app name, AI provider, and API keys
3. **Select languages** — use presets (Common / All / None) or pick individually
4. **Start Translation** — watch real-time progress with a live log
5. **Download** the ZIP with all translated files, or preview individual languages

> 💡 API keys entered in the Web UI override `.env` keys for that session only — they are not saved to disk.

---

## ⌨️ CLI Mode

For scripting or CI/CD, use the command-line interface directly:

```bash
# Translate using the default strings.xml in the project folder
python main.py

# Translate a specific file
python main.py path/to/your/strings.xml
```

Output is saved to the `output/` directory and zipped as `output.zip`.

---

## ⚙️ Advanced Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `TRANSLATION_PROVIDER` | `openai` | `openai` or `gemini` |
| `APP_NAME` | `Android` | App name for translation context |
| `SUPPORTED_LANGUAGES` | `[]` | JSON array of target language codes |
| `CHUNK_TOKEN_LIMIT` | `4000` | Max tokens per translation chunk |
| `CHUNK_CONCURRENCY` | `8` | Parallel chunks per language |
| `MAX_CONCURRENT_TRANSLATIONS` | `50` | Max parallel language translations |
| `PREFER_NUMERIC_ENTITIES` | `0` | Use `&#x1F525;` instead of literal emoji |
| `ESCAPE_APOSTROPHES` | `1` | Escape `'` as `\'` in text nodes |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/languages` | List all 90+ supported languages |
| `GET` | `/api/config` | Current server configuration |
| `POST` | `/api/translate` | Upload XML and start translation job |
| `GET` | `/api/jobs/{id}` | Poll job progress |
| `GET` | `/api/jobs/{id}/download` | Download translated ZIP |
| `GET` | `/api/jobs/{id}/preview/{lang}` | Preview a single language result |

---

## 📝 License

MIT