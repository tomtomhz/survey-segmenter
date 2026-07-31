/**
 * The interface is used by people who are not going to debug it. These cover the ways it could
 * lose their work or trap them, rather than the ways it could show the wrong number.
 */
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { SettingsModal } from './components/SettingsModal'
import { api } from './api/client'
import { analysis, file } from './test/fixtures'

afterEach(() => vi.restoreAllMocks())

const person = () => userEvent.setup({ applyAccept: false })

function withProjects(projects: { id: string; title: string }[] = []) {
  vi.spyOn(api, 'settings').mockResolvedValue({
    sdk_installed: false, configured: false, source: null, env_key: false, model: null,
  })
  vi.spyOn(api, 'projects').mockResolvedValue({
    ok: true,
    projects: projects.map((p) => ({
      ...p, k: 3, n_people: 240, confidence: 'high' as const, updated: new Date().toISOString(),
    })),
  })
}

describe('deleting a project takes two decisions', () => {
  it('does not delete on the first click, because there is no undo', async () => {
    withProjects([{ id: 'p1', title: 'community_survey.csv' }])
    const remove = vi.spyOn(api, 'deleteProject')
    const user = person()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Delete community_survey.csv' }))

    // Deleting drops the analysis *and* the original upload from disk. One stray click on an
    // unlabelled × next to every row is not an acceptable way to lose an afternoon.
    expect(remove).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: /Confirm deleting/ })).toBeInTheDocument()
  })

  it('deletes once confirmed', async () => {
    withProjects([{ id: 'p1', title: 'community_survey.csv' }])
    const remove = vi.spyOn(api, 'deleteProject').mockResolvedValue({ ok: true, projects: [] })
    const user = person()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Delete community_survey.csv' }))
    await user.click(screen.getByRole('button', { name: /Confirm deleting/ }))

    expect(remove).toHaveBeenCalledWith('p1')
  })

  it('lets someone back out', async () => {
    withProjects([{ id: 'p1', title: 'community_survey.csv' }])
    const remove = vi.spyOn(api, 'deleteProject')
    const user = person()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Delete community_survey.csv' }))
    await user.click(screen.getByRole('button', { name: 'Keep this project' }))

    expect(remove).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Delete community_survey.csv' })).toBeInTheDocument()
  })
})

describe('the settings dialog can be left', () => {
  it('closes on Escape, like every other dialog on the machine', async () => {
    vi.spyOn(api, 'settings').mockResolvedValue({
      sdk_installed: true, configured: false, source: null, env_key: false, model: null,
    })
    const onClose = vi.fn()
    const user = person()
    render(<SettingsModal open onClose={onClose} onConfigured={() => {}} />)

    await user.keyboard('{Escape}')

    expect(onClose).toHaveBeenCalled()
  })

  it('announces itself as a dialog', () => {
    vi.spyOn(api, 'settings').mockResolvedValue({
      sdk_installed: true, configured: false, source: null, env_key: false, model: null,
    })
    render(<SettingsModal open onClose={() => {}} onConfigured={() => {}} />)
    expect(screen.getByRole('dialog')).toHaveAccessibleName('AI interpretation')
  })
})

describe('a second file is ignored while an analysis is running', () => {
  it('does not start a second run on top of the first', async () => {
    withProjects()
    let resolve: (value: ReturnType<typeof analysis>) => void = () => {}
    const analyze = vi.spyOn(api, 'analyze').mockImplementation(
      () => new Promise((r) => { resolve = r }),
    )
    const user = person()
    render(<App />)
    const input = document.querySelector('input[type="file"]') as HTMLInputElement

    await user.upload(input, file('one.csv'))
    await waitFor(() => expect(analyze).toHaveBeenCalledTimes(1))
    await user.upload(input, file('two.csv'))

    // Two analyses at once would interleave their messages and race to set the session id.
    expect(analyze).toHaveBeenCalledTimes(1)
    resolve(analysis())
  })

  // NOTE: the guard in App.tsx is a ref rather than the `busy` state, because state does not
  // update until the next render and two drops in the SAME tick would both pass a state check.
  // That same-tick case is deliberately not asserted here: user-event flushes React between
  // actions, and dispatching the events by hand did not reproduce it either — the test passed
  // with the old state-based guard, so it would have been evidence of nothing. The ref is kept
  // because it is free and obviously correct, not because a test demonstrates the race.
})

describe('the app never stays stuck', () => {
  it('releases the busy state if something fails outside the request path', async () => {
    withProjects()
    let resolve: (value: ReturnType<typeof analysis>) => void = () => {}
    vi.spyOn(api, 'analyze').mockImplementation(() => new Promise((r) => { resolve = r }))
    const user = person()
    render(<App />)

    await user.upload(document.querySelector('input[type="file"]') as HTMLInputElement,
                      file('survey.csv'))
    await waitFor(() =>
      expect(screen.getByLabelText('Attach a survey file')).toBeDisabled())

    // Something throws where no handler will see it — the composer must not stay disabled.
    window.dispatchEvent(new Event('unhandledrejection'))

    await waitFor(() =>
      expect(screen.getByLabelText('Attach a survey file')).not.toBeDisabled())
    resolve(analysis())
  })
})
