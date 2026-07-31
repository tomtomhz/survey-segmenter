import { useEffect, useRef } from 'react'
import { ResultCard } from './ResultCard'
import { STARTER_QUESTIONS } from '../lib/labels'
import type { Message } from '../lib/thread'

/** The conversation. Scrolls itself to the bottom whenever it grows. */
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

  useEffect(() => {
    const el = thread.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

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
                  <ResultCard
                    result={message.result}
                    busy={busy}
                    setBusy={setBusy}
                    regroupError={regroupError}
                    onRegroup={onRegroup}
                    onNeedsKey={onNeedsKey}
                  />
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
