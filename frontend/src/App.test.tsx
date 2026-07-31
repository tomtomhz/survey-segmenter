import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { api } from './api/client'
import { analysis, file } from './test/fixtures'

afterEach(() => vi.restoreAllMocks())

function noProjects() {
  vi.spyOn(api, 'projects').mockResolvedValue({ ok: true, projects: [] })
  vi.spyOn(api, 'settings').mockResolvedValue({
    sdk_installed: false, configured: false, source: null, env_key: false, model: null,
  })
}

/**
 * Attach a survey through the paperclip, the way a user does.
 *
 * `applyAccept: false` because the input carries accept=".csv,.xlsx,.xls" and user-event
 * enforces it, which would silently drop the very files these tests exist to reject. The browser
 * treats accept as a filter on the picker, not a guarantee — a file can still arrive by drag and
 * drop — so the app's own check is the one that matters and has to be exercised.
 */
function person() {
  return userEvent.setup({ applyAccept: false })
}

async function attach(user: ReturnType<typeof userEvent.setup>, name = 'survey.csv', size = 64) {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  await user.upload(input, file(name, size))
}

/** The message box, told apart from the group-name boxes that appear inside a result. */
function composer(): HTMLTextAreaElement {
  return document.querySelector('.cbox textarea') as HTMLTextAreaElement
}

describe('analysing a survey', () => {
  it('shows the result, the charts and the downloads', async () => {
    noProjects()
    const user = person()
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis())
    render(<App />)

    await attach(user)

    expect(await screen.findByText('Groups found')).toBeInTheDocument()
    expect(screen.getByText('240')).toBeInTheDocument()
    expect(screen.getByText('Trust these groups')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Segment map' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Who is in which group (CSV)' })).toBeInTheDocument()
  })

  it('says a key is needed rather than silently skipping the interpretation', async () => {
    noProjects()
    const user = person()
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: false }))
    const chat = vi.spyOn(api, 'chat')
    render(<App />)

    await attach(user)

    expect(await screen.findByText(/Add your Anthropic API key/)).toBeInTheDocument()
    expect(chat).not.toHaveBeenCalled()
  })

  it('asks Claude to interpret straight away when a key is configured', async () => {
    noProjects()
    const user = person()
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: true }))
    const chat = vi.spyOn(api, 'chat').mockResolvedValue({
      ok: true, reply_html: '<p>Segment 0 is your best first target.</p>',
    })
    render(<App />)

    await attach(user)

    expect(await screen.findByText(/best first target/)).toBeInTheDocument()
    expect(chat).toHaveBeenCalledWith('sess-1', '', true)
    // The starter questions appear only after that first interpretation.
    expect(screen.getByRole('button', { name: /Which segment should we target first/ }))
      .toBeInTheDocument()
  })

  it('rejects an unreadable file without contacting the server', async () => {
    noProjects()
    const user = person()
    const analyze = vi.spyOn(api, 'analyze')
    render(<App />)

    await attach(user, 'notes.pdf')

    expect(await screen.findByText(/I cannot use that file/)).toBeInTheDocument()
    expect(analyze).not.toHaveBeenCalled()
  })

  it('recovers from a failed analysis instead of leaving the app wedged', async () => {
    noProjects()
    const user = person()
    vi.spyOn(api, 'analyze').mockResolvedValue({
      ok: false, error: 'I could not find any questions to group people on.',
    })
    render(<App />)

    await attach(user)

    expect(await screen.findByText(/could not find any questions/)).toBeInTheDocument()
    // The composer is usable again — the failure released the busy state.
    await waitFor(() =>
      expect(screen.getByLabelText('Attach a survey file')).not.toBeDisabled())
  })
})

describe('asking questions about the result', () => {
  it('refuses to send a question before there is anything to discuss', async () => {
    noProjects()
    const user = person()
    const chat = vi.spyOn(api, 'chat')
    render(<App />)

    await user.type(composer(), 'which segment is biggest?{Enter}')

    expect(chat).not.toHaveBeenCalled()
    expect(await screen.findByText(/Attach a survey file first/)).toBeInTheDocument()
  })

  it('sends a follow-up once a survey has been analysed', async () => {
    noProjects()
    const user = person()
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: false }))
    const chat = vi.spyOn(api, 'chat').mockResolvedValue({
      ok: true, reply_html: '<p>Segment 1, at 43%.</p>',
    })
    render(<App />)
    await attach(user)
    await screen.findByText('Groups found')

    await user.type(composer(), 'which segment is biggest?{Enter}')

    expect(chat).toHaveBeenCalledWith('sess-1', 'which segment is biggest?', false)
    expect(await screen.findByText(/Segment 1, at 43%/)).toBeInTheDocument()
  })
})

describe('projects', () => {
  it('replays a reopened project in the order it happened', async () => {
    vi.spyOn(api, 'settings').mockResolvedValue({
      sdk_installed: true, configured: true, source: 'file', env_key: false, model: 'claude-opus-5',
    })
    vi.spyOn(api, 'projects').mockResolvedValue({
      ok: true,
      projects: [{
        id: 'sess-1', title: 'community_survey.csv', k: 3, n_people: 240,
        confidence: 'high', updated: new Date().toISOString(),
      }],
    })
    vi.spyOn(api, 'openProject').mockResolvedValue({
      ...analysis(),
      reopened: true,
      transcript: [
        { role: 'you', text: 'Analyse: community_survey.csv' },
        { role: 'you', text: 'Which segment first?' },
        { role: 'ai', html: '<p>Start with Segment 0.</p>' },
      ],
    })
    const user = person()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: /community_survey.csv/ }))

    // The report must come before the answers that discuss it.
    const thread = document.querySelector('.wrap') as HTMLElement
    const order = [...thread.querySelectorAll('.msg')].map((m) => m.textContent ?? '')
    expect(order[0]).toContain('Analyse: community_survey.csv')
    expect(order[1]).toContain('Groups found')
    expect(order[2]).toContain('Which segment first?')
    expect(order[3]).toContain('Start with Segment 0.')
  })
})
