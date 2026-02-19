import { useState } from 'react'

export default function ConfigPanel({
    appName, setAppName,
    provider, setProvider,
    openaiKey, setOpenaiKey,
    geminiKey, setGeminiKey,
    config,
}) {
    const [showOpenai, setShowOpenai] = useState(false)
    const [showGemini, setShowGemini] = useState(false)

    const openaiReady = !!(openaiKey?.trim() || config?.has_openai_key)
    const geminiReady = !!(geminiKey?.trim() || config?.has_gemini_key)

    return (
        <div>
            <div className="form-row">
                <div className="form-group">
                    <label className="form-label" htmlFor="app-name">App Name</label>
                    <input
                        className="form-input"
                        id="app-name"
                        type="text"
                        value={appName}
                        onChange={(e) => setAppName(e.target.value)}
                        placeholder="e.g. AI Photo Editor"
                    />
                </div>

                <div className="form-group">
                    <label className="form-label" htmlFor="provider-select">AI Provider</label>
                    <select
                        className="form-select"
                        id="provider-select"
                        value={provider}
                        onChange={(e) => setProvider(e.target.value)}
                    >
                        <option value="gemini">
                            Google Gemini {geminiReady ? '✓' : '⚠ needs key'}
                        </option>
                        <option value="openai">
                            OpenAI GPT {openaiReady ? '✓' : '⚠ needs key'}
                        </option>
                    </select>
                </div>
            </div>

            {/* API Key inputs */}
            <div className="form-group">
                <label className="form-label" htmlFor="openai-key">
                    OpenAI API Key
                    {config?.has_openai_key && !openaiKey && (
                        <span style={{ color: 'var(--success)', fontWeight: 400, marginLeft: 8 }}>
                            ✓ set in .env
                        </span>
                    )}
                </label>
                <div style={{ position: 'relative' }}>
                    <input
                        className="form-input"
                        id="openai-key"
                        type={showOpenai ? 'text' : 'password'}
                        value={openaiKey}
                        onChange={(e) => setOpenaiKey(e.target.value)}
                        placeholder={config?.has_openai_key ? '••••• (using .env key, type to override)' : 'sk-...'}
                        autoComplete="off"
                        style={{ paddingRight: 48 }}
                    />
                    <button
                        type="button"
                        onClick={() => setShowOpenai((v) => !v)}
                        style={{
                            position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
                            background: 'none', border: 'none', color: 'var(--text-muted)',
                            cursor: 'pointer', fontSize: '0.85rem', padding: '4px 6px',
                        }}
                        tabIndex={-1}
                    >
                        {showOpenai ? '🙈' : '👁️'}
                    </button>
                </div>
            </div>

            <div className="form-group">
                <label className="form-label" htmlFor="gemini-key">
                    Google Gemini API Key
                    {config?.has_gemini_key && !geminiKey && (
                        <span style={{ color: 'var(--success)', fontWeight: 400, marginLeft: 8 }}>
                            ✓ set in .env
                        </span>
                    )}
                </label>
                <div style={{ position: 'relative' }}>
                    <input
                        className="form-input"
                        id="gemini-key"
                        type={showGemini ? 'text' : 'password'}
                        value={geminiKey}
                        onChange={(e) => setGeminiKey(e.target.value)}
                        placeholder={config?.has_gemini_key ? '••••• (using .env key, type to override)' : 'AIza...'}
                        autoComplete="off"
                        style={{ paddingRight: 48 }}
                    />
                    <button
                        type="button"
                        onClick={() => setShowGemini((v) => !v)}
                        style={{
                            position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
                            background: 'none', border: 'none', color: 'var(--text-muted)',
                            cursor: 'pointer', fontSize: '0.85rem', padding: '4px 6px',
                        }}
                        tabIndex={-1}
                    >
                        {showGemini ? '🙈' : '👁️'}
                    </button>
                </div>
            </div>

            {/* Status summary */}
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.7 }}>
                <span style={{ color: openaiReady ? 'var(--success)' : 'var(--error)' }}>●</span>{' '}
                {openaiReady ? 'OpenAI key ready' : 'OpenAI key missing'} &nbsp;·&nbsp;{' '}
                <span style={{ color: geminiReady ? 'var(--success)' : 'var(--error)' }}>●</span>{' '}
                {geminiReady ? 'Gemini key ready' : 'Gemini key missing'}
                {!openaiReady && !geminiReady && (
                    <span style={{ display: 'block', color: 'var(--warning)', marginTop: 4 }}>
                        ⚠ Enter at least one API key to translate
                    </span>
                )}
            </div>
        </div>
    )
}
