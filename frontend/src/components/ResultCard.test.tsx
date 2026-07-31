import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ResultCard } from './ResultCard'
import { api } from '../api/client'
import { analysis } from '../test/fixtures'

afterEach(() => vi.restoreAllMocks())

const props = {
  busy: false, setBusy: () => {}, regroupError: null, onRegroup: () => {}, onNeedsKey: () => {},
}

describe('the card keeps one file list', () => {
  it('shows the new group-names file as soon as the groups are named', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'saveNames').mockResolvedValue({
      ok: true,
      names: ['Sceptics', 'Fans', 'Lurkers'],
      downloads: ['segment_assignments.csv', 'group_profiles.csv', 'group_names.csv'],
    })
    render(<ResultCard result={analysis()} {...props} />)

    // Before naming, that file does not exist yet.
    expect(screen.queryByRole('link', { name: 'Your group names (CSV)' })).toBeNull()

    const boxes = screen.getAllByRole('textbox')
    await user.type(boxes[0], 'Sceptics')
    await user.type(boxes[1], 'Fans')
    await user.type(boxes[2], 'Lurkers')
    await user.click(screen.getByRole('button', { name: 'Save these names' }))

    // The download bar picks it up, rather than the name panel printing a link of its own.
    expect(await screen.findByRole('link', { name: 'Your group names (CSV)' }))
      .toHaveAttribute('href', expect.stringContaining('group_names.csv'))
  })
})
