import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DesignPanel } from './DesignPanel'
import { api } from '../api/client'
import type { DesignResult } from '../api/types'

afterEach(() => vi.restoreAllMocks())

const props = { busy: false, setBusy: () => {} }

function built(over: Partial<DesignResult['report']> = {}): DesignResult {
  return {
    ok: true,
    items: ['Free delivery', 'Lower prices', 'Longer returns'],
    prose: 'A best-worst questionnaire for 3 items',
    csv: 'respondent_id,task,position,item\nR0001,1,1,Free delivery\n',
    report: {
      n_respondents: 200,
      n_items: 3,
      sets_per_respondent: 10,
      items_per_set: 2,
      times_each_item_shown: [66, 67],
      exposures_per_respondent: 6.7,
      pair_appearances: [660, 680],
      never_paired: 0,
      evenly_divisible: false,
      ...over,
    },
  }
}

describe('building the questionnaire', () => {
  it('will not build from fewer than three items, because there is nothing to compare', async () => {
    const user = userEvent.setup()
    const design = vi.spyOn(api, 'design')
    render(<DesignPanel {...props} />)
    await user.click(screen.getByText(/Running a best-worst study/))

    await user.type(screen.getByLabelText(/Items to compare/), 'Free delivery\nLower prices')
    expect(screen.getByRole('button', { name: /Build the questionnaire/ })).toBeDisabled()
    expect(design).not.toHaveBeenCalled()
  })

  it('sends only the non-blank lines, so a trailing newline is not an item', async () => {
    const user = userEvent.setup()
    const design = vi.spyOn(api, 'design').mockResolvedValue(built())
    render(<DesignPanel {...props} />)
    await user.click(screen.getByText(/Running a best-worst study/))

    await user.type(screen.getByLabelText(/Items to compare/), 'A\nB\n\nC\n')
    await user.click(screen.getByRole('button', { name: /Build the questionnaire/ }))

    expect(design).toHaveBeenCalledWith(['A', 'B', 'C'], 4, 10, 200)
  })

  it('reports what the design achieved rather than claiming it is balanced', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'design').mockResolvedValue(built())
    render(<DesignPanel {...props} />)
    await user.click(screen.getByText(/Running a best-worst study/))

    await user.type(screen.getByLabelText(/Items to compare/), 'A\nB\nC')
    await user.click(screen.getByRole('button', { name: /Build the questionnaire/ }))

    expect(await screen.findByText(/66/)).toBeInTheDocument()
    expect(screen.getByText(/Every pair of items appears together/)).toBeInTheDocument()
    // The uneven-exposure note is arithmetic, not a fault, and must be said rather than hidden.
    expect(screen.getByText(/not a multiple of/)).toBeInTheDocument()
  })

  it('warns loudly when two items never meet, because that ranking cannot be trusted', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'design').mockResolvedValue(built({ never_paired: 4, evenly_divisible: true }))
    render(<DesignPanel {...props} />)
    await user.click(screen.getByText(/Running a best-worst study/))

    await user.type(screen.getByLabelText(/Items to compare/), 'A\nB\nC')
    await user.click(screen.getByRole('button', { name: /Build the questionnaire/ }))

    const warning = await screen.findByText(/4 pairs of items never appear together/)
    expect(warning.closest('p')).toHaveClass('warn')
  })

  it('shows a refusal from the server as an error, not as a broken panel', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'design').mockResolvedValue({
      ok: false,
      error: 'Show between 2 and 6 items on each screen.',
    })
    render(<DesignPanel {...props} />)
    await user.click(screen.getByText(/Running a best-worst study/))

    await user.type(screen.getByLabelText(/Items to compare/), 'A\nB\nC')
    await user.click(screen.getByRole('button', { name: /Build the questionnaire/ }))

    expect(await screen.findByText(/between 2 and 6 items/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Download/ })).not.toBeInTheDocument()
  })
})
