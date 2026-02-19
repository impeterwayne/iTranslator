import { useState, useEffect, useRef, useCallback } from 'react'
import UploadZone from './components/UploadZone'
import ConfigPanel from './components/ConfigPanel'
import LanguageSelector from './components/LanguageSelector'
import TranslationProgress from './components/TranslationProgress'
import ResultView from './components/ResultView'

const STEPS = {
  SETUP: 'setup',
  TRANSLATING: 'translating',
  COMPLETED: 'completed',
}

function App() {
  const [step, setStep] = useState(STEPS.SETUP)
  const [file, setFile] = useState(null)
  const [appName, setAppName] = useState('')
  const [provider, setProvider] = useState('gemini')
  const [selectedLangs, setSelectedLangs] = useState([])
  const [allLanguages, setAllLanguages] = useState([])
  const [config, setConfig] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [errors, setErrors] = useState([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [apiOnline, setApiOnline] = useState(false)
  const [openaiKey, setOpenaiKey] = useState('')
  const [geminiKey, setGeminiKey] = useState('')

  // Fetch config & languages on mount
  useEffect(() => {
    const init = async () => {
      try {
        const [configRes, langsRes] = await Promise.all([
          fetch('/api/config'),
          fetch('/api/languages'),
        ])
        if (configRes.ok) {
          const cfg = await configRes.json()
          setConfig(cfg)
          setAppName(cfg.app_name || '')
          setProvider(cfg.provider || 'gemini')
          setSelectedLangs(cfg.supported_languages || [])
          setApiOnline(true)
        }
        if (langsRes.ok) {
          const data = await langsRes.json()
          setAllLanguages(data.languages || [])
        }
      } catch {
        setApiOnline(false)
      }
    }
    init()
  }, [])

  // Poll job status
  useEffect(() => {
    if (!jobId || step !== STEPS.TRANSLATING) return
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`)
        if (res.ok) {
          const data = await res.json()
          setJobStatus(data)
          if (data.status === 'completed') {
            setStep(STEPS.COMPLETED)
            clearInterval(interval)
          } else if (data.status === 'failed') {
            setErrors(data.errors || ['Translation failed'])
            setStep(STEPS.COMPLETED)
            clearInterval(interval)
          }
        }
      } catch { }
    }, 1000)
    return () => clearInterval(interval)
  }, [jobId, step])

  const handleSubmit = async () => {
    if (!file || selectedLangs.length === 0) return
    setIsSubmitting(true)
    setErrors([])

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('app_name', appName || 'Android')
      formData.append('provider', provider)
      formData.append('languages', JSON.stringify(selectedLangs))
      if (openaiKey.trim()) formData.append('openai_api_key', openaiKey.trim())
      if (geminiKey.trim()) formData.append('gemini_api_key', geminiKey.trim())

      const res = await fetch('/api/translate', { method: 'POST', body: formData })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to start translation')
      }

      const data = await res.json()
      setJobId(data.job_id)
      setJobStatus({
        status: 'running',
        completed_languages: 0,
        total_languages: data.total_languages,
        progress: [],
        errors: [],
      })
      setStep(STEPS.TRANSLATING)
    } catch (err) {
      setErrors([err.message])
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleReset = () => {
    setStep(STEPS.SETUP)
    setFile(null)
    setJobId(null)
    setJobStatus(null)
    setErrors([])
  }

  const handleDownload = async () => {
    if (!jobId) return
    window.open(`/api/jobs/${jobId}/download`, '_blank')
  }

  return (
    <div className="app-layout">
      {/* BG Decorations */}
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />

      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <div className="header-logo">iT</div>
            <div>
              <div className="header-title">iTranslator</div>
              <div className="header-subtitle">AI Android strings.xml resources translator</div>
            </div>
          </div>
          <div className="header-status">
            <div className={`status-dot ${apiOnline ? '' : 'offline'}`} />
            <span className="status-text">{apiOnline ? 'API Connected' : 'API Offline'}</span>
          </div>
        </div>
      </header>

      <main className="main-content">


        {/* Errors */}
        {errors.length > 0 && (
          <div className="error-banner">
            <span className="error-banner-icon">⚠️</span>
            <div className="error-banner-text">
              {errors.map((e, i) => <div key={i}>{e}</div>)}
            </div>
          </div>
        )}

        {/* Step: Setup */}
        {step === STEPS.SETUP && (
          <div className="wizard-grid fade-in-up">
            {/* Upload */}
            <div className="card" style={{ animationDelay: '0.05s' }}>
              <div className="card-header">
                <div className="card-icon">📄</div>
                <div>
                  <div className="step-number">Step 1</div>
                  <div className="card-title">Upload strings.xml</div>
                </div>
              </div>
              <UploadZone file={file} onFileChange={setFile} />
            </div>

            {/* Config */}
            <div className="card" style={{ animationDelay: '0.1s' }}>
              <div className="card-header">
                <div className="card-icon">⚙️</div>
                <div>
                  <div className="step-number">Step 2</div>
                  <div className="card-title">Configure Translation</div>
                </div>
              </div>
              <ConfigPanel
                appName={appName}
                setAppName={setAppName}
                provider={provider}
                setProvider={setProvider}
                openaiKey={openaiKey}
                setOpenaiKey={setOpenaiKey}
                geminiKey={geminiKey}
                setGeminiKey={setGeminiKey}
                config={config}
              />
            </div>

            {/* Languages */}
            <div className="card" style={{ animationDelay: '0.15s' }}>
              <div className="card-header">
                <div className="card-icon">🌍</div>
                <div>
                  <div className="step-number">Step 3</div>
                  <div className="card-title">Select Languages</div>
                </div>
              </div>
              <LanguageSelector
                allLanguages={allLanguages}
                selectedLangs={selectedLangs}
                setSelectedLangs={setSelectedLangs}
              />
            </div>

            {/* Submit */}
            <div style={{ textAlign: 'center', marginTop: 8, animationDelay: '0.2s' }} className="fade-in-up">
              <div className="config-pills" style={{ justifyContent: 'center', marginBottom: 16 }}>
                <div className="config-pill">
                  Provider: <span className="config-pill-value">{provider}</span>
                </div>
                <div className="config-pill">
                  Languages: <span className="config-pill-value">{selectedLangs.length}</span>
                </div>
                {file && (
                  <div className="config-pill">
                    File: <span className="config-pill-value">{file.name}</span>
                  </div>
                )}
              </div>
              <button
                className="btn btn-primary btn-lg"
                onClick={handleSubmit}
                disabled={!file || selectedLangs.length === 0 || isSubmitting || !apiOnline}
                id="start-translation-btn"
              >
                {isSubmitting ? (
                  <>
                    <div className="spinner" />
                    Starting...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🚀</span>
                    Start Translation
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Step: Translating */}
        {step === STEPS.TRANSLATING && jobStatus && (
          <TranslationProgress jobStatus={jobStatus} />
        )}

        {/* Step: Completed */}
        {step === STEPS.COMPLETED && (
          <ResultView
            jobStatus={jobStatus}
            jobId={jobId}
            errors={errors}
            onDownload={handleDownload}
            onReset={handleReset}
          />
        )}
      </main>


    </div>
  )
}

export default App
