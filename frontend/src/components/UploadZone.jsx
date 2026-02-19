import { useRef, useState, useCallback } from 'react'

export default function UploadZone({ file, onFileChange }) {
    const inputRef = useRef(null)
    const [dragging, setDragging] = useState(false)

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        setDragging(false)
        const dropped = e.dataTransfer.files[0]
        if (dropped && dropped.name.endsWith('.xml')) {
            onFileChange(dropped)
        }
    }, [onFileChange])

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        setDragging(true)
    }, [])

    const handleDragLeave = useCallback(() => {
        setDragging(false)
    }, [])

    const handleClick = () => inputRef.current?.click()

    const handleChange = (e) => {
        const selected = e.target.files[0]
        if (selected) onFileChange(selected)
    }

    const handleClear = (e) => {
        e.stopPropagation()
        onFileChange(null)
        if (inputRef.current) inputRef.current.value = ''
    }

    const formatSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`
        if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
        return `${(bytes / 1048576).toFixed(1)} MB`
    }

    return (
        <div
            className={`upload-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
            onClick={handleClick}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            id="upload-zone"
        >
            <input
                ref={inputRef}
                type="file"
                accept=".xml"
                className="upload-input"
                onChange={handleChange}
                id="file-input"
            />

            {file ? (
                <>
                    <span className="upload-icon">✅</span>
                    <div className="upload-text">File ready</div>
                    <div className="upload-file-info">
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">{formatSize(file.size)}</span>
                        <button className="file-clear-btn" onClick={handleClear} id="clear-file-btn">
                            ✕ Remove
                        </button>
                    </div>
                </>
            ) : (
                <>
                    <span className="upload-icon">📁</span>
                    <div className="upload-text">
                        Drop your <code style={{ color: 'var(--accent-3)', background: 'var(--accent-glow)', padding: '2px 6px', borderRadius: '4px' }}>strings.xml</code> here
                    </div>
                    <div className="upload-subtext">or click to browse · XML files only</div>
                </>
            )}
        </div>
    )
}
