import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Header } from './Header'

const props = { onNew: () => {}, onSettings: () => {}, onToggleProjects: () => {},
                projectsOpen: false }

describe('the app chrome', () => {
  it('offers a way to reach the projects panel', async () => {
    // Below 820px the projects column used to be display:none with nothing to bring it back, so
    // every saved study — and renaming, pinning, searching, deleting — was simply unreachable on
    // a narrow window. CSS hides this button on wide ones; it must exist for the narrow case.
    const user = userEvent.setup()
    const onToggleProjects = vi.fn()
    render(<Header {...props} onToggleProjects={onToggleProjects} />)

    await user.click(screen.getByRole('button', { name: /Projects/ }))
    expect(onToggleProjects).toHaveBeenCalled()
  })

  it('tells assistive tech whether the panel is open, and what it controls', () => {
    const { rerender } = render(<Header {...props} />)
    const toggle = screen.getByRole('button', { name: /Projects/ })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(toggle).toHaveAttribute('aria-controls', 'projects-panel')

    rerender(<Header {...props} projectsOpen />)
    expect(screen.getByRole('button', { name: /Projects/ })).toHaveAttribute('aria-expanded', 'true')
  })
})
