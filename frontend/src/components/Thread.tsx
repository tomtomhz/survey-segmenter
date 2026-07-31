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
  onAsk,
}: {
  messages: Message[]
  busy: boolean
  setBusy: (busy: boolean) => void
  regroupError: string | null
  onRegroup: (items: string[]) => void
  onNeedsKey: () => void
  onAsk: (question: string) => void
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
      <div className="wrap">
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
            <div className="msg ai" key={message.id}>
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
      </div>
    </div>
  )
}
