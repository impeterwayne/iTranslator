import { useState, useMemo } from 'react'

export default function LanguageSelector({ allLanguages, selectedLangs, setSelectedLangs }) {
    const [search, setSearch] = useState('')

    const filtered = useMemo(() => {
        if (!search.trim()) return allLanguages
        const q = search.toLowerCase()
        return allLanguages.filter(
            (l) => l.name.toLowerCase().includes(q) || l.code.toLowerCase().includes(q)
        )
    }, [allLanguages, search])

    const toggle = (code) => {
        setSelectedLangs((prev) =>
            prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]
        )
    }

    const selectAll = () => {
        const allCodes = allLanguages.map((l) => l.code)
        setSelectedLangs(allCodes)
    }

    const clearAll = () => setSelectedLangs([])

    const selectCommon = () => {
        const common = [
            'ar', 'bn', 'cs', 'de', 'el', 'es', 'fa', 'fi', 'fil', 'fr',
            'hi', 'hr', 'in', 'it', 'ja', 'ko', 'ms', 'nl', 'pl', 'pt',
            'ru', 'sk', 'sr', 'sv', 'th', 'tr', 'vi', 'zh',
        ]
        setSelectedLangs(common)
    }

    return (
        <div>
            <div className="lang-selector-header">
                <span className="lang-count">{selectedLangs.length} / {allLanguages.length} selected</span>
                <div className="lang-actions">
                    <button className="lang-action-btn" onClick={selectCommon} id="select-common-btn">
                        Common
                    </button>
                    <button className="lang-action-btn" onClick={selectAll} id="select-all-btn">
                        All
                    </button>
                    <button className="lang-action-btn" onClick={clearAll} id="clear-all-btn">
                        None
                    </button>
                </div>
            </div>

            <div className="lang-search-wrapper">
                <span className="lang-search-icon">🔍</span>
                <input
                    className="lang-search"
                    type="text"
                    placeholder="Search languages..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    id="lang-search"
                />
            </div>

            <div className="lang-grid">
                {filtered.map((lang) => {
                    const isSelected = selectedLangs.includes(lang.code)
                    return (
                        <div
                            key={lang.code}
                            className={`lang-chip ${isSelected ? 'selected' : ''}`}
                            onClick={() => toggle(lang.code)}
                            id={`lang-${lang.code}`}
                        >
                            <div className="lang-chip-checkbox">
                                {isSelected && '✓'}
                            </div>
                            <span>{lang.name}</span>
                            <span className="lang-chip-code">{lang.code}</span>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
