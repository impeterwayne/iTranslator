import { useEffect, useRef } from 'react'

export default function TranslationProgress({ jobStatus }) {
    const logEndRef = useRef(null)

    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [jobStatus?.progress?.length])

    if (!jobStatus) return null

    const { completed_languages = 0, total_languages = 1, progress = [], current_language, status } = jobStatus
    const pct = total_languages > 0 ? Math.round((completed_languages / total_languages) * 100) : 0
    const isRunning = status === 'running' || status === 'pending'

    return (
        <div className="progress-section">
            <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
                <div className="card-header">
                    <div className="card-icon" style={{ fontSize: '1.4rem' }}>
                        {isRunning ? (
                            <div className="spinner spinner-accent" style={{ width: 24, height: 24 }} />
                        ) : '✅'}
                    </div>
                    <div>
                        <div className="card-title">
                            {isRunning ? 'Translating...' : 'Translation Complete'}
                        </div>
                        <div className="card-desc">
                            {current_language
                                ? `Currently translating: ${current_language}`
                                : isRunning ? 'Preparing...' : 'All languages processed'}
                        </div>
                    </div>
                </div>

                {/* Percentage */}
                <div className="progress-percentage">{pct}%</div>

                {/* Progress bar */}
                <div className="progress-stats">
                    <span className="progress-label">Languages Translated</span>
                    <span className="progress-count">{completed_languages} / {total_languages}</span>
                </div>
                <div className="progress-bar-wrapper">
                    <div
                        className={`progress-bar-fill ${!isRunning ? 'completed' : ''}`}
                        style={{ width: `${pct}%` }}
                    />
                </div>
            </div>

            {/* Live log */}
            <div className="card">
                <div className="card-header">
                    <div className="card-icon">📋</div>
                    <div>
                        <div className="card-title">Live Log</div>
                        <div className="card-desc">{progress.length} entries</div>
                    </div>
                </div>
                <div className="log-console" id="log-console">
                    {progress.map((msg, i) => (
                        <div
                            key={i}
                            className={`log-entry ${msg.includes('✓') || msg.includes('completed') ? 'success' :
                                    msg.includes('✗') || msg.includes('failed') || msg.includes('Error') ? 'error' :
                                        'info'
                                }`}
                        >
                            {msg}
                        </div>
                    ))}
                    <div ref={logEndRef} />
                </div>
            </div>
        </div>
    )
}
