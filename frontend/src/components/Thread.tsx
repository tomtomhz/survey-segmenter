import { useEffect, useLayoutEffect, useRef } from 'react'
import { Boundary, CardFailure } from './Boundary'
import { ResultCard } from './ResultCard'
import { STARTER_QUESTIONS } from '../lib/labels'
import type { Message } from '../lib/thread'

/** How close to the bottom still counts as "following the conversation", in pixels. */
const AT_BOTTOM = 80

/**
 * The conversation. Follows new messages down only while the reader is already at the bottom.
 */
export function Thread({
  messages,
  busy,
  setBusy,
  regroupError,
  onRegroup,
  onNeedsKey,
  onPolarity,
  onAsk,
  footer,
}: {
  messages: Message[]
  busy: boolean
  setBusy: (busy: boolean) => void
  regroupError: string | null
  onRegroup: (items: string[]) => void
  onNeedsKey: () => void
  /** Answer to "which code means best" for a wide best-worst export. */
  onPolarity: (file: File, code: number) => void
  onAsk: (question: string) => void
  /**
   * Rendered after the messages, inside the scrolling area. Used for the study planner, which
   * belongs in the conversation on the start screen rather than pinned above the composer where
   * it would sit over every later result too.
   */
  footer?: React.ReactNode
}) {
  const thread = useRef<HTMLDivElement>(null)
  // Whether the reader was at the bottom *before* this render. Captured in a layout effect so it
  // reflects the scroll position before the new message changed the height.
  const wasAtBottom = useRef(true)

  useLayoutEffect(() => {
    const el = thread.current
    if (!el) return
    // A statistical report is long, and reading it means scrolling up and staying there. Pinning
    // to the bottom on every new message would yank the reader away mid-sentence whenever Claude
    // answered. Follow along only for someone who is already following along.
    if (wasAtBottom.current) el.scrollTop = el.scrollHeight
  }, [messages])

  useEffect(() => {
    const el = thread.current
    if (!el) return
    const remember = () => {
      wasAtBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < AT_BOTTOM
    }
    remember()
    el.addEventListener('scroll', remember, { passive: true })
    return () => el.removeEventListener('scroll', remember)
  }, [])

  return (
    <div className="thread" ref={thread}>
      {/* Replies arrive without any focus change, so a screen reader would otherwise never be
          told the answer had come back. Polite: it waits for the reader to finish a sentence. */}
      <div className="wrap" role="log" aria-live="polite" aria-relevant="additions">
        {messages.map((message) => {
          if (message.kind === 'you') {
            return (
              <div className="msg you" key={message.id}>
                <div className="bubble">{message.text}</div>
              </div>
            )
          }
          if (message.kind === 'suggestions') {
            return (
              <div className="chips" key={message.id}>
                {STARTER_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    type="button"
                    className="chip"
                    disabled={busy}
                    onClick={() => onAsk(question)}
                  >
                    {question}
                  </button>
                ))}
              </div>
            )
          }
          return (
            // A finished result is a dashboard, not prose: stat tiles, charts and downloads are
            // scanned, and they were being held inside a 780px reading measure that left a third
            // of a laptop screen empty. It takes the full width; the long statistical report
            // inside it keeps the measure, because that part genuinely is prose.
            <div className={`msg ai${message.kind === 'result' ? ' wide' : ''}`} key={message.id}>
              <div className="av">✦</div>
              <div className="bubble">
                {message.kind === 'thinking' && (
                  <div className="think">
                    <span className="dots">
                      <i />
                      <i />
                      <i />
                    </span>
                    {message.label}
                  </div>
                )}
                {message.kind === 'note' && (
                  <div
                    className={`note${message.tone === 'error' ? ' err' : ''}`}
                    onClick={(event) => {
                      // Errors caused by a missing key offer a way to fix them inline.
                      if ((event.target as HTMLElement).closest('.js-settings')) onNeedsKey()
                    }}
                    dangerouslySetInnerHTML={{ __html: message.html }}
                  />
                )}
                {message.kind === 'ai' && (
                  <div dangerouslySetInnerHTML={{ __html: message.html }} />
                )}
                {message.kind === 'polarity' && (
                  <div className="note">
                    <p style={{ margin: '0 0 10px' }}>{message.note.split('\n\n')[0]}</p>
                    <p style={{ margin: '0 0 10px' }}>
                      <b>Which of these numbers means the item they liked most?</b> It is in your
                      file {message.codes.length} times over — I can see the codes, just not what
                      they were meant to mean.
                    </p>
                    <div className="chips">
                      {message.codes.map((c) => (
                        <button
                          key={c.code}
                          type="button"
                          className="chip"
                          onClick={() => onPolarity(message.file, c.code)}
                        >
                          {c.code} means best
                          <span style={{ color: 'var(--muted)' }}>
                            {' '}· appears {c.times.toLocaleString()} times
                          </span>
                        </button>
                      ))}
                    </div>
                    <p className="hint" style={{ margin: '10px 0 0', textAlign: 'left' }}>
                      The most common code is usually the one for &ldquo;shown but not picked&rdquo;,
                      since it covers every item that was neither. Pick wrong and the ranking comes
                      out upside down, so it is worth checking how your survey was set up.
                    </p>
                  </div>
                )}
                {message.kind === 'result' && (
                  /* Per card, not just per app: a result that cannot be drawn should cost the
                     reader that one card, not the conversation around it. */
                  <Boundary
                    fallback={(error, reset) => <CardFailure error={error} onRetry={reset} />}
                  >
                    <ResultCard
                      result={message.result}
                      busy={busy}
                      setBusy={setBusy}
                      regroupError={regroupError}
                      onRegroup={onRegroup}
                      onNeedsKey={onNeedsKey}
                    />
                  </Boundary>
                )}
              </div>
            </div>
          )
        })}
        {footer}
      </div>
    </div>
  )
}
