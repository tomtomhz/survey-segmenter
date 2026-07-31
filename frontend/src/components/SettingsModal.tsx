import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { AiStatus, Failure } from '../api/types'

type Status = AiStatus | Failure | null

/**
 * Where someone turns on Claude interpretation with their own Anthropic key.
 *
 * The wording is the feature: it has to be unambiguous that the results go to Anthropic and the
 * individual answers never do, because that is the actual privacy guarantee the tool makes.
 */
export function SettingsModal({
  open,
  onClose,
  onConfigured,
}: {
  open: boolean
  onClose: () => void
  onConfigured: (configured: boolean) => void
}) {
  const [status, setStatus] = useState<Status>(null)
  const [key, setKey] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) {
      setKey('')
      return
    }
    setStatus(null)
    void api.settings().then(setStatus)
  }, [open])

  if (!open) return null

  async function save() {
    const trimmed = key.trim()
    if (!trimmed) return
    setSaving(true)
    const reply = await api.saveKey(trimmed)
    setSaving(false)
    setStatus(reply)
    if (reply.ok === false) return
    setKey('')
    onConfigured(Boolean((reply as AiStatus).configured))
    onClose()
  }

  async function clear() {
    setStatus(await api.clearKey())
    onConfigured(false)
  }

  const message = describe(status)
  // An environment key cannot be changed from here, and a missing SDK cannot be fixed from here.
  const locked = message.tone === 'warn' && message.locksInput

  return (
    <div
      className="scrim on"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="modal">
        <h2>AI interpretation</h2>
        <p>
          Survey Segmenter can send your <b>results</b> — the group sizes, profiles and confidence,
          never anyone’s individual answers — to Claude to interpret them and answer your
          questions, using your own Anthropic account.
        </p>
        <div className={`status ${message.tone}`}>{message.text}</div>
        <label htmlFor="api-key">Anthropic API key</label>
        <input
          id="api-key"
          type="password"
          placeholder="sk-ant-..."
          autoComplete="off"
          spellCheck={false}
          disabled={locked}
          value={key}
          onChange={(event) => setKey(event.target.value)}
        />
        <p>
          Get one at{' '}
          <span
            className="link"
            role="button"
            tabIndex={0}
            onClick={() => window.open('https://console.anthropic.com/', '_blank')}
          >
            console.anthropic.com
          </span>
          . It is stored only on this computer.
        </p>
        <div className="row">
          <button
            type="button"
            className="btn primary"
            disabled={locked || saving || !key.trim()}
            onClick={() => void save()}
          >
            Save key
          </button>
          <button
            type="button"
            className="btn sub"
            disabled={locked || !isConfigured(status)}
            onClick={() => void clear()}
          >
            Remove
          </button>
          <span style={{ flex: 1 }} />
          <button type="button" className="btn sub" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

function isConfigured(status: Status): boolean {
  return status != null && status.ok !== false && Boolean((status as AiStatus).configured)
}

function describe(status: Status): { tone: 'ok' | 'warn'; text: string; locksInput: boolean } {
  if (status == null) return { tone: 'warn', text: 'Checking…', locksInput: true }
  if (status.ok === false) {
    return { tone: 'warn', text: status.error, locksInput: false }
  }
  const ai = status as AiStatus
  if (!ai.sdk_installed) {
    return {
      tone: 'warn',
      locksInput: true,
      text: 'The AI add-on is not installed. Install it with `pip install anthropic` (or rebuild '
        + 'the app), then reopen this page. The statistics work without it.',
    }
  }
  if (ai.env_key) {
    return {
      tone: 'ok',
      locksInput: true,
      text: 'A key is set from your environment (ANTHROPIC_API_KEY) and will be used. '
        + 'Claude interpretation is on.',
    }
  }
  if (ai.configured) {
    return {
      tone: 'ok',
      locksInput: false,
      text: 'A key is saved on this computer. Claude interpretation is on.',
    }
  }
  return {
    tone: 'warn',
    locksInput: false,
    text: 'No API key yet. Paste one below to turn on Claude interpretation.',
  }
}
