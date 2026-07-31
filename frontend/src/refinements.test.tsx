/**
 * The second audit pass: things that were not crashes, but were wrong.
 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { ChartsCard } from './components/ChartsCard'
import { api } from './api/client'
import { analysis, chart, file } from './test/fixtures'

afterEach(() => vi.restoreAllMocks())

const person = () => userEvent.setup({ applyAccept: false })

function quiet() {
  vi.spyOn(api, 'projects').mockResolvedValue({ ok: true, projects: [] })
  vi.spyOn(api, 'settings').mockResolvedValue({
    sdk_installed: false, configured: false, source: null, env_key: false, model: null,
  })
}

describe('Save PDF carries the whole result', () => {
  it('opens every collapsed panel before the browser takes its snapshot', async () => {
    quiet()
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: false }))
    const user = person()
    render(<App />)
    await user.upload(document.querySelector('input[type="file"]') as HTMLInputElement,
                      file('survey.csv'))
    await screen.findByText('Groups found')

    const panels = () => [...document.querySelectorAll('details')]
    const closedBefore = panels().filter((p) => !p.open)
    // The statistical report, the group names and the column picker all start collapsed, and a
    // browser does not paint the contents of a collapsed <details> — so the PDF was losing them.
    expect(closedBefore.length).toBeGreaterThan(0)

    window.dispatchEvent(new Event('beforeprint'))
    expect(panels().every((p) => p.open)).toBe(true)

    // And the reader's own view is put back exactly as they left it.
    window.dispatchEvent(new Event('afterprint'))
    expect(panels().filter((p) => !p.open)).toHaveLength(closedBefore.length)
  })

  it('leaves panels the reader opened themselves open afterwards', async () => {
    quiet()
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: false }))
    const user = person()
    render(<App />)
    await user.upload(document.querySelector('input[type="file"]') as HTMLInputElement,
                      file('survey.csv'))
    await screen.findByText('Groups found')

    const report = [...document.querySelectorAll('details')].find((p) => !p.open)!
    report.open = true                     // the reader expanded this one

    window.dispatchEvent(new Event('beforeprint'))
    window.dispatchEvent(new Event('afterprint'))

    expect(report.open).toBe(true)
  })
})

describe('Enter while an input method is composing', () => {
  it('does not send a half-finished word', async () => {
    quiet()
    vi.spyOn(api, 'analyze').mockResolvedValue(analysis({ ai_available: false }))
    const chat = vi.spyOn(api, 'chat')
    const user = person()
    render(<App />)
    await user.upload(document.querySelector('input[type="file"]') as HTMLInputElement,
                      file('survey.csv'))
    await screen.findByText('Groups found')

    const box = document.querySelector('.cbox textarea') as HTMLTextAreaElement
    await user.type(box, 'segment')
    // Japanese, Chinese and Korean input methods use Enter to accept a candidate word. Sending
    // on that keystroke fires off whatever half of the sentence exists so far.
    box.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'Enter', bubbles: true, cancelable: true, isComposing: true,
    } as KeyboardEventInit))

    expect(chat).not.toHaveBeenCalled()
  })
})

describe('a request that never comes back', () => {
  it('ends as an error rather than a spinner nobody can cancel', async () => {
    vi.useFakeTimers()
    try {
      // A server that accepts the connection and then says nothing at all.
      vi.stubGlobal('fetch', (_url: string, init?: RequestInit) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            const error = new Error('aborted')
            error.name = 'AbortError'
            reject(error)
          })
        }))
      const pending = api.projects()
      await vi.advanceTimersByTimeAsync(15 * 60 * 1000 + 1000)
      const reply = await pending
      expect(reply.ok).toBe(false)
      expect(reply.ok === false && reply.error).toContain('took too long')
    } finally {
      vi.useRealTimers()
      vi.unstubAllGlobals()
    }
  })
})

describe('the chart tabs behave like a tablist', () => {
  it('moves between charts with the arrow keys, and wraps', async () => {
    const user = userEvent.setup()
    render(<ChartsCard charts={[chart('map'), chart('fit'), chart('k')]} />)

    const selected = () =>
      (document.querySelector('.ctab[aria-selected="true"]') as HTMLElement)?.textContent

    await user.click(screen.getByRole('tab', { name: 'Segment map' }))
    await user.keyboard('{ArrowRight}')
    expect(selected()).toBe('Who belongs')
    await user.keyboard('{End}')
    expect(selected()).toBe('How many groups')
    await user.keyboard('{ArrowRight}')            // wraps rather than dead-ending
    expect(selected()).toBe('Segment map')
    await user.keyboard('{ArrowLeft}')
    expect(selected()).toBe('How many groups')
  })

  it('is a single tab stop, with each panel tied to its tab', () => {
    render(<ChartsCard charts={[chart('map'), chart('fit')]} />)
    const tabs = screen.getAllByRole('tab')
    expect(tabs.map((t) => t.getAttribute('tabindex'))).toEqual(['0', '-1'])

    const panel = screen.getAllByRole('tabpanel')[0]
    expect(panel.getAttribute('aria-labelledby')).toBe(tabs[0].id)
    expect(tabs[0].getAttribute('aria-controls')).toBe(panel.id)
  })
})

describe('closing the settings dialog', () => {
  it('gives focus back to whatever opened it', async () => {
    quiet()
    const user = person()
    render(<App />)
    const settings = screen.getByRole('button', { name: 'Settings' })

    await user.click(settings)
    await user.click(await screen.findByRole('button', { name: 'Close' }))

    // Otherwise focus falls to the top of the document and a keyboard user has to tab back in.
    await waitFor(() => expect(document.activeElement).toBe(settings))
  })
})

describe('the conversation is announced', () => {
  it('marks the thread as a live region, since replies arrive without a focus change', () => {
    quiet()
    render(<App />)
    const log = screen.getByRole('log')
    expect(log).toHaveAttribute('aria-live', 'polite')
  })
})
