import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api/client'
import { isFailure, type Analysis, type ProjectSummary } from './api/types'
import { AppFailure, Boundary } from './components/Boundary'
import { Composer } from './components/Composer'
import { Header } from './components/Header'
import { SettingsModal } from './components/SettingsModal'
import { Sidebar } from './components/Sidebar'
import { Thread } from './components/Thread'
import { useDropTarget } from './hooks/useDropTarget'
import { PlanPanel } from './components/PlanPanel'
import { usePrintExpansion } from './hooks/usePrintExpansion'
import { droppedADirectory, fileProblem } from './lib/upload'
import { messageId, replace, withoutSuggestions, type Message } from './lib/thread'

const GREETING: Message = {
  id: 0,
  kind: 'note',
  html:
    '<div class="eyebrow">Start here</div>'
    + '<h2>Turn a survey into clear customer groups</h2>'
    + '<p>Drop a <code>.csv</code> or <code>.xlsx</code> export anywhere on this page, or use the '
    + 'paperclip. I will find the groups, tell you how much to trust them, and give you the files '
    + 'to act on.</p>'
    + '<p style="color:var(--muted);font-size:.9rem">With an API key in Settings, '
    + '<strong>Claude</strong> also explains what the groups mean for your team and answers your '
    + 'questions about them.</p>',
}

const AI_NUDGE =
  'Add your Anthropic API key in <button type="button" class="link js-settings">Settings'
  + '</button> to have Claude '
  + 'interpret these results and answer questions about them. The statistics above are complete '
  + 'either way.'

export function App() {
  const [messages, setMessages] = useState<Message[]>([GREETING])
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  // `busy` is state, so it does not update until the next render — two files dropped in the same
  // tick both pass an `if (busy)` check and start two analyses whose messages then interleave.
  // A ref changes immediately, so it is the guard; the state is only what the interface renders.
  const running = useRef(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [regroupError, setRegroupError] = useState<string | null>(null)
  // Where focus goes when the settings dialog closes: back to whatever opened it.
  const returnFocusTo = useRef<HTMLElement | null>(null)

  usePrintExpansion()

  const append = useCallback((message: Message) => {
    setMessages((current) => [...current, message])
  }, [])

  /** Mark the app busy or free. One place, so the ref and the rendered state cannot drift. */
  const setWorking = useCallback((working: boolean) => {
    running.current = working
    setBusy(working)
  }, [])

  // Last line of defence, restored from the previous interface: whatever goes wrong — an error
  // thrown outside a promise, a rejection nobody handled — never leave the composer disabled
  // with no way back. A stuck spinner is the one failure a non-technical user cannot work around.
  useEffect(() => {
    const release = () => setWorking(false)
    window.addEventListener('unhandledrejection', release)
    window.addEventListener('error', release)
    return () => {
      window.removeEventListener('unhandledrejection', release)
      window.removeEventListener('error', release)
    }
  }, [setWorking])

  const refreshProjects = useCallback(async () => {
    const reply = await api.projects()
    if (!isFailure(reply)) setProjects(reply.projects)
  }, [])

  useEffect(() => {
    void refreshProjects()
  }, [refreshProjects])

  /**
   * Ask Claude — either the opening interpretation of a result, or a follow-up question.
   *
   * `session` is passed explicitly by the callers that have just created one. Reading it from
   * state instead looks equivalent and is not: this closure captured `sessionId` when it was
   * still null, so `setSessionId` from the analysis that finished a microsecond ago has not
   * reached it yet, and the opening interpretation silently never happened.
   */
  const ask = useCallback(
    async (question: string | null, initial: boolean, session?: string) => {
      const sid = session ?? sessionId
      if (!sid) return
      setMessages((current) => withoutSuggestions(current))
      if (question != null) append({ id: messageId(), kind: 'you', text: question })
      const placeholder = messageId()
      append({
        id: placeholder,
        kind: 'thinking',
        label: initial ? 'Claude is reading your results…' : 'Claude is thinking…',
      })
      setWorking(true)
      const reply = await api.chat(sid, question ?? '', initial)
      setWorking(false)
      if (isFailure(reply)) {
        const offerSettings = reply.kind === 'nokey' || reply.kind === 'nosdk' || reply.kind === 'auth'
        setMessages((current) =>
          replace(current, placeholder, {
            id: placeholder,
            kind: 'note',
            tone: 'error',
            // Escaped: _explain_run_error quotes the raw exception, and an exception can carry
            // text straight out of the uploaded file — a column heading is attacker-controlled
            // the moment someone analyses a spreadsheet a third party sent them.
            html: offerSettings
              ? `${escapeHtml(reply.error)} `
                + '<button type="button" class="link js-settings">Open Settings</button>'
              : escapeHtml(reply.error),
          }),
        )
        return
      }
      setMessages((current) => {
        const next = replace(current, placeholder, {
          id: placeholder, kind: 'ai', html: reply.reply_html,
        })
        return initial ? [...next, { id: messageId(), kind: 'suggestions' }] : next
      })
      void refreshProjects()
    },
    [append, refreshProjects, sessionId, setWorking],
  )

  /** Show a finished analysis, and start the interpretation if a key is configured. */
  const present = useCallback(
    (result: Analysis, placeholder?: number) => {
      setSessionId(result.session_id)
      setRegroupError(null)
      const card: Message = { id: placeholder ?? messageId(), kind: 'result', result }
      setMessages((current) =>
        placeholder != null ? replace(current, placeholder, card) : [...current, card],
      )
      void refreshProjects()
      if (!result.ai_available) {
        append({ id: messageId(), kind: 'note', html: AI_NUDGE })
      }
      return result.ai_available
    },
    [append, refreshProjects],
  )

  const analyse = useCallback(
    async (file: File) => {
      if (running.current) return
      const problem = fileProblem(file)
      append({ id: messageId(), kind: 'you', text: `Analyse my survey: ${file.name || 'file'}` })
      if (problem) {
        append({
          id: messageId(), kind: 'note', tone: 'error',
          html: `<b>I cannot use that file.</b> ${escapeHtml(problem)}`,
        })
        return
      }
      const placeholder = messageId()
      append({
        id: placeholder,
        kind: 'thinking',
        label: 'Crunching the numbers — clustering and validating. This can take up to a minute…',
      })
      setWorking(true)
      const reply = await api.analyze(file)
      setWorking(false)
      if (isFailure(reply)) {
        setMessages((current) =>
          replace(current, placeholder, {
            id: placeholder, kind: 'note', tone: 'error', html: escapeHtml(reply.error),
          }),
        )
        return
      }
      if (present(reply, placeholder)) void ask(null, true, reply.session_id)
    },
    [append, ask, present, setWorking],
  )

  const regroup = useCallback(
    async (items: string[]) => {
      if (!sessionId || busy) return
      setRegroupError(null)
      setWorking(true)
      const reply = await api.regroup(sessionId, items)
      setWorking(false)
      if (isFailure(reply)) {
        setRegroupError(reply.error)
        return
      }
      append({ id: messageId(), kind: 'you', text: `Group people on: ${items.join(', ')}` })
      if (present(reply)) void ask(null, true, reply.session_id)
    },
    [append, ask, busy, present, sessionId, setWorking],
  )

  const openProject = useCallback(
    async (id: string) => {
      if (busy) return
      setWorking(true)
      const reply = await api.openProject(id)
      setWorking(false)
      if (isFailure(reply)) {
        void refreshProjects()
        return
      }
      setSessionId(reply.session_id)
      setRegroupError(null)
      // Replay in the order it happened: the request, then the result, then the conversation
      // about it. Appending the result last would show the report below answers discussing it.
      const replayed: Message[] = []
      const transcript = [...(reply.transcript ?? [])]
      if (transcript[0]?.role === 'you') {
        replayed.push({ id: messageId(), kind: 'you', text: transcript.shift()?.text ?? '' })
      }
      if (reply.report_html) replayed.push({ id: messageId(), kind: 'result', result: reply })
      for (const entry of transcript) {
        if (entry.role === 'you') {
          replayed.push({ id: messageId(), kind: 'you', text: entry.text ?? '' })
        } else if (entry.html) {
          replayed.push({ id: messageId(), kind: 'ai', html: entry.html })
        }
      }
      setMessages(replayed.length > 0 ? replayed : [GREETING])
      void refreshProjects()
    },
    [busy, refreshProjects, setWorking],
  )

  const deleteProject = useCallback(
    async (id: string) => {
      const reply = await api.deleteProject(id)
      if (id === sessionId) {
        setSessionId(null)
        setMessages([GREETING])
      }
      if (!isFailure(reply)) setProjects(reply.projects)
    },
    [sessionId],
  )

  /** Remember what had focus, so closing the dialog can hand it back. */
  const openSettings = useCallback(() => {
    returnFocusTo.current = document.activeElement as HTMLElement | null
    setSettingsOpen(true)
  }, [])

  const startNew = useCallback(() => {
    if (busy) return
    setSessionId(null)
    setMessages([GREETING])
    void refreshProjects()
  }, [busy, refreshProjects])

  const dragging = useDropTarget(
    useCallback(
      (transfer: DataTransfer) => {
        if (busy) return
        if (droppedADirectory(transfer)) {
          append({
            id: messageId(), kind: 'note', tone: 'error',
            html: '<b>That is a folder.</b> Please drop the survey file itself — a .csv or .xlsx.',
          })
          return
        }
        const file = transfer.files?.[0]
        if (file) void analyse(file)
      },
      [analyse, append, busy],
    ),
  )

  return (
    <Boundary fallback={(error) => <AppFailure error={error} />}>
      <Header onNew={startNew} onSettings={() => openSettings()} />
      <div className="body">
        <Sidebar
          projects={projects}
          activeId={sessionId}
          onOpen={(id) => void openProject(id)}
          onDelete={(id) => void deleteProject(id)}
          onNew={startNew}
        />
        <div className="main">
          <Thread
            messages={messages}
            busy={busy}
            setBusy={setWorking}
            regroupError={regroupError}
            onRegroup={(items) => void regroup(items)}
            onNeedsKey={() => openSettings()}
            onAsk={(question) => void ask(question, false)}
            footer={
              // Only before there is a result. Once an analysis exists the question has been
              // answered by the data itself, and leaving a planning panel under every result would
              // be clutter offering to plan a study already fielded.
              sessionId === null ? <PlanPanel busy={busy} setBusy={setWorking} /> : null
            }
          />
          <Composer
            busy={busy}
            dragging={dragging}
            hasSession={sessionId != null}
            onSend={(text) => void ask(text, false)}
            onFile={(file) => void analyse(file)}
          />
        </div>
      </div>
      <SettingsModal
        open={settingsOpen}
        onClose={() => {
          setSettingsOpen(false)
          // Dropping focus to the top of the document after a dialog closes strands anyone
          // navigating by keyboard, who then has to tab back through the whole page.
          returnFocusTo.current?.focus()
        }}
        onConfigured={(configured) => {
          if (configured && sessionId) void ask(null, true)
        }}
      />
    </Boundary>
  )
}

/** Server errors are plain text but land in an HTML slot alongside markup we author. */
function escapeHtml(text: string): string {
  const entities: Record<string, string> = {
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }
  return text.replace(/[&<>"']/g, (character) => entities[character] ?? character)
}
