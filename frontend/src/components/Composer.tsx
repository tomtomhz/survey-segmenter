import { useEffect, useRef, useState } from 'react'

const HINT = 'Your file and your data stay on this computer. Nothing is uploaded.'
const NEEDS_FILE =
  'Attach a survey file first (the paperclip) so I have results to talk about.'

/**
 * The message box. It also owns the attach button, because "send a question" and "send a file"
 * are the same gesture from the user's side.
 */
export function Composer({
  busy,
  dragging,
  hasSession,
  onSend,
  onFile,
}: {
  busy: boolean
  dragging: boolean
  hasSession: boolean
  onSend: (text: string) => void
  onFile: (file: File) => void
}) {
  const [text, setText] = useState('')
  const [hint, setHint] = useState(HINT)
  const box = useRef<HTMLTextAreaElement>(null)
  const picker = useRef<HTMLInputElement>(null)

  // Grow with the content up to a ceiling, rather than scrolling a two-line box.
  //
  // Collapse to 0 before measuring, not `auto`: this textarea is a flex item, and an `auto`
  // height there resolves against the container rather than the content, so scrollHeight comes
  // back as the 180px cap and the empty composer opens four lines tall.
  useEffect(() => {
    const el = box.current
    if (!el) return
    el.style.height = '0px'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [text])

  // A hint that scolds the user needs to stop scolding them.
  useEffect(() => {
    if (hint === HINT) return
    const timer = setTimeout(() => setHint(HINT), 4000)
    return () => clearTimeout(timer)
  }, [hint])

  function send() {
    if (busy) return
    const trimmed = text.trim()
    if (!trimmed) return
    if (!hasSession) {
      setHint(NEEDS_FILE)
      return
    }
    setText('')
    onSend(trimmed)
  }

  return (
    <div className="composer">
      <div className={`cbox${dragging ? ' drag' : ''}`}>
        <button
          type="button"
          className="iconbtn"
          title="Attach a survey file (.csv or .xlsx)"
          aria-label="Attach a survey file"
          disabled={busy}
          onClick={() => picker.current?.click()}
        >
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.4 11.05l-8.49 8.49a5 5 0 01-7.07-7.07l8.49-8.49a3.33 3.33 0 014.71 4.71l-8.49 8.49a1.67 1.67 0 01-2.36-2.36l7.78-7.78" />
          </svg>
        </button>
        <textarea
          ref={box}
          rows={1}
          value={text}
          placeholder={hasSession ? 'Ask about your segments…' : 'Attach a survey file to begin'}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              send()
            }
          }}
        />
        <button
          type="button"
          className="iconbtn send"
          title="Send"
          aria-label="Send"
          disabled={busy || !text.trim()}
          onClick={send}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19V5M6 11l6-6 6 6" />
          </svg>
        </button>
        <input
          ref={picker}
          type="file"
          accept=".csv,.xlsx,.xls"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0]
            event.target.value = ''
            if (file && !busy) onFile(file)
          }}
        />
      </div>
      <div className="hint">{hint}</div>
    </div>
  )
}
