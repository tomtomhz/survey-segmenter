import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { NamePanel } from './NamePanel'
import { api } from '../api/client'
import { analysis } from '../test/fixtures'

afterEach(() => vi.restoreAllMocks())

const props = { busy: false, setBusy: () => {}, onNeedsKey: () => {} }

describe('naming the groups', () => {
  it('offers one box per group, pre-filled with names already saved', () => {
    render(<NamePanel result={analysis({ names: ['Sceptics', 'Fans', 'Lurkers'] })} {...props} />)
    expect(screen.getAllByRole('textbox')).toHaveLength(3)
    expect(screen.getByDisplayValue('Sceptics')).toBeInTheDocument()
  })

  it('will not save a half-named set, because the blanks would reach the CRM', async () => {
    const user = userEvent.setup()
    const save = vi.spyOn(api, 'saveNames')
    render(<NamePanel result={analysis()} {...props} />)

    await user.type(screen.getAllByRole('textbox')[0], 'Sceptics')
    await user.click(screen.getByRole('button', { name: 'Save these names' }))

    expect(save).not.toHaveBeenCalled()
    expect(screen.getByText(/Give every group a name/)).toBeInTheDocument()
  })

  it('saves trimmed names and then points at the files they went into', async () => {
    const user = userEvent.setup()
    const save = vi.spyOn(api, 'saveNames').mockResolvedValue({
      ok: true, names: ['Sceptics', 'Fans', 'Lurkers'],
    })
    render(<NamePanel result={analysis()} {...props} />)

    const boxes = screen.getAllByRole('textbox')
    await user.type(boxes[0], '  Sceptics  ')
    await user.type(boxes[1], 'Fans')
    await user.type(boxes[2], 'Lurkers')
    await user.click(screen.getByRole('button', { name: 'Save these names' }))

    expect(save).toHaveBeenCalledWith('sess-1', ['Sceptics', 'Fans', 'Lurkers'])
    expect(await screen.findByText(/The names are now in the downloads/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Group names/ }))
      .toHaveAttribute('href', expect.stringContaining('group_names.csv'))
  })

  it('fills the boxes with what Claude suggested', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'suggestNames').mockResolvedValue({
      ok: true, names: ['Privacy-First Lurkers', 'Social Sharers', 'Quiet Readers'],
    })
    render(<NamePanel result={analysis()} {...props} />)

    await user.click(screen.getByRole('button', { name: /Suggest names/ }))

    expect(await screen.findByDisplayValue('Privacy-First Lurkers')).toBeInTheDocument()
  })

  it('offers Settings when the failure is a missing key, not just the error', async () => {
    const user = userEvent.setup()
    const onNeedsKey = vi.fn()
    vi.spyOn(api, 'suggestNames').mockResolvedValue({
      ok: false, error: 'No API key is configured.', kind: 'nokey',
    })
    render(<NamePanel result={analysis()} {...props} onNeedsKey={onNeedsKey} />)

    await user.click(screen.getByRole('button', { name: /Suggest names/ }))
    await user.click(await screen.findByText('Open Settings'))

    expect(onNeedsKey).toHaveBeenCalled()
  })

  it('does not offer Settings for a failure a key would not fix', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'suggestNames').mockResolvedValue({
      ok: false, error: 'Please analyse a survey file first.',
    })
    render(<NamePanel result={analysis()} {...props} />)

    await user.click(screen.getByRole('button', { name: /Suggest names/ }))

    expect(await screen.findByText(/analyse a survey file first/)).toBeInTheDocument()
    expect(screen.queryByText('Open Settings')).not.toBeInTheDocument()
  })
})
