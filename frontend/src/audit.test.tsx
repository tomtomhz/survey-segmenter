/**
 * Findings from the post-rebuild audit, written as tests first so each one is demonstrated
 * before it is fixed rather than asserted afterwards.
 */
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { SettingsModal } from './components/SettingsModal'
import { api } from './api/client'
import { analysis, file } from './test/fixtures'

afterEach(() => vi.restoreAllMocks())

function person() {
  return userEvent.setup({ applyAccept: false })
}

async function attach(user: ReturnType<typeof userEvent.setup>, name = 'survey.csv') {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  await user.upload(input, file(name))
}

describe('a key that comes from the environment cannot be edited here', () => {
  it('disables the box and the buttons, because typing in them would do nothing', async () => {
    vi.spyOn(api, 'settings').mockResolvedValue({
      sdk_installed: true, configured: true, source: 'env', env_key: true, model: 'claude-opus-5',
    })
    render(<SettingsModal open onClose={() => {}} onConfigured={() => {}} />)

    // ANTHROPIC_API_KEY wins over the saved key, so a key typed here is silently ignored.
    // Offering an enabled box and a Save button is a lie about what the app will do.
    expect(await screen.findByLabelText(/Anthropic API key/i)).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Save key' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Remove' })).toBeDisabled()
  })
})

describe('error text from the server is never treated as markup', () => {
  it('escapes a failed answer, which can quote a column name from the file', async () => {
    vi.spyOn(api, 'projects').mockResolvedValue({ ok: true, projects: [] })
    vi.spyOn(api, 'settings').mockResolvedValue({
      sdk_installed: true, configured: true, source: 'file', env_key: false, model: 'claude-opus-5',
    })
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: true }))
    // _explain_run_error puts the raw exception text in the message, and an exception can quote a
    // column heading straight out of the uploaded file.
    vi.spyOn(api, 'chat').mockResolvedValue({
      ok: false,
      error: 'Could not read column <img src=x onerror="alert(1)">.',
    })
    const user = person()
    render(<App />)

    await attach(user)

    const note = await screen.findByText(/Could not read column/)
    expect(note.querySelector('img')).toBeNull()
    expect(note.textContent).toContain('<img src=x')
  })
})

describe('one broken result cannot take the whole app down', () => {
  it('shows a failure in place, leaving the rest of the conversation usable', async () => {
    vi.spyOn(api, 'projects').mockResolvedValue({ ok: true, projects: [] })
    vi.spyOn(api, 'settings').mockResolvedValue({
      sdk_installed: false, configured: false, source: null, env_key: false, model: null,
    })
    // A payload missing a field the UI reads without a guard. DownloadBar does
    // `result.downloads.length`, so an absent `downloads` is a TypeError during render — and in
    // React a render error unmounts the entire tree, not just the component that threw.
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ downloads: undefined as never }))
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const user = person()
    render(<App />)

    await attach(user)

    // Whatever happens, the page must not be blank: the chrome and the composer survive.
    // Scoped to the header, because the "add a key" note also offers a Settings button now.
    await waitFor(() =>
      expect(within(screen.getByRole('banner')).getByRole('button', { name: 'Settings' }))
        .toBeInTheDocument())
    expect(document.querySelector('.cbox textarea')).toBeInTheDocument()
    // And the failure is explained where the result would have been, rather than silently absent.
    expect(screen.getByText(/could not be displayed/i)).toBeInTheDocument()
  })
})

describe('the thread does not yank the reader around', () => {
  it('leaves the scroll alone when the reader has scrolled up to read the report', async () => {
    vi.spyOn(api, 'projects').mockResolvedValue({ ok: true, projects: [] })
    vi.spyOn(api, 'settings').mockResolvedValue({
      sdk_installed: false, configured: false, source: null, env_key: false, model: null,
    })
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: false }))
    const user = person()
    render(<App />)
    await attach(user)
    await screen.findByText('Groups found')

    const thread = document.querySelector('.thread') as HTMLElement
    // happy-dom reports zero heights, so describe a tall thread the reader has scrolled up in.
    Object.defineProperty(thread, 'scrollHeight', { value: 4000, configurable: true })
    Object.defineProperty(thread, 'clientHeight', { value: 800, configurable: true })
    thread.scrollTop = 500                     // a long way from the bottom: they are reading
    // A browser fires `scroll` whenever the position changes; happy-dom does not synthesise one
    // from a property assignment, and that event is how the component learns they moved.
    thread.dispatchEvent(new Event('scroll'))

    await user.type(document.querySelector('.cbox textarea') as HTMLElement, 'hello{Enter}')

    // Appending a message must not drag them back down to the newest thing.
    expect(thread.scrollTop).toBe(500)
  })
})


describe('the thread does follow along for a reader who is at the bottom', () => {
  it('scrolls to the newest message when they had not scrolled away', async () => {
    vi.spyOn(api, 'projects').mockResolvedValue({ ok: true, projects: [] })
    vi.spyOn(api, 'settings').mockResolvedValue({
      sdk_installed: false, configured: false, source: null, env_key: false, model: null,
    })
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: false }))
    const user = person()
    render(<App />)
    await attach(user)
    await screen.findByText('Groups found')

    const thread = document.querySelector('.thread') as HTMLElement
    Object.defineProperty(thread, 'scrollHeight', { value: 4000, configurable: true })
    Object.defineProperty(thread, 'clientHeight', { value: 800, configurable: true })
    thread.scrollTop = 3200                    // pinned to the bottom: following along
    thread.dispatchEvent(new Event('scroll'))

    await user.type(document.querySelector('.cbox textarea') as HTMLElement, 'hello{Enter}')

    expect(thread.scrollTop).toBe(4000)
  })
})
