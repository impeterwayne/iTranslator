import { useState, useEffect } from 'react'

export default function ResultView({ jobStatus, jobId, errors, onDownload, onReset }) {
    const [previewLang, setPreviewLang] = useState(null)
    const [previewContent, setPreviewContent] = useState('')
    const [previewLoading, setPreviewLoading] = useState(false)

    const hasErrors = errors && errors.length > 0
    const completedLangs = jobStatus?.progress
        ?.filter((p) => p.includes('✓'))
        .map((p) => {
            const match = p.match(/\(([^)]+)\)/)
            return match ? match[1] : null
        })
        .filter(Boolean) || []

    const loadPreview = async (code) => {
        if (!jobId) return
        setPreviewLang(code)
        setPreviewLoading(true)
        try {
            const res = await fetch(`/api/jobs/${jobId}/preview/${code}`)
            if (res.ok) {
                const data = await res.json()
                setPreviewContent(data.content)
            } else {
                setPreviewContent('Could not load preview')
            }
        } catch {
            setPreviewContent('Error loading preview')
        } finally {
            setPreviewLoading(false)
        }
    }

    return (
        <div className="result-section">
            <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
                {!hasErrors ? (
                    <>
                        <div className="result-celebration">🎉</div>
                        <div className="result-title">Translation Complete!</div>
                        <div className="result-subtitle">
                            {jobStatus?.completed_languages || 0} languages translated successfully
                        </div>
                    </>
                ) : (
                    <>
                        <div className="result-celebration">⚠️</div>
                        <div className="result-title">Translation Finished with Errors</div>
                        <div className="result-subtitle">
                            Some languages may have failed. Check the logs for details.
                        </div>
                    </>
                )}

                <div className="result-actions">
                    <button className="btn btn-success btn-lg" onClick={onDownload} id="download-btn">
                        <span className="btn-icon">📥</span>
                        Download ZIP
                    </button>
                    <button className="btn btn-secondary btn-lg" onClick={onReset} id="reset-btn">
                        <span className="btn-icon">🔄</span>
                        Translate Another
                    </button>
                </div>
            </div>

            {/* Preview Section */}
            {completedLangs.length > 0 && (
                <div className="card">
                    <div className="card-header">
                        <div className="card-icon">👁️</div>
                        <div>
                            <div className="card-title">Preview Translations</div>
                            <div className="card-desc">Click a language to preview the result</div>
                        </div>
                    </div>

                    <div className="preview-tabs">
                        {completedLangs.map((code) => (
                            <button
                                key={code}
                                className={`preview-tab ${previewLang === code ? 'active' : ''}`}
                                onClick={() => loadPreview(code)}
                                id={`preview-tab-${code}`}
                            >
                                {code}
                            </button>
                        ))}
                    </div>

                    {previewLang && (
                        <div className="preview-content" id="preview-content">
                            {previewLoading ? (
                                <div style={{ textAlign: 'center', padding: '24px' }}>
                                    <div className="spinner spinner-accent spinner-lg" />
                                </div>
                            ) : (
                                previewContent
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
