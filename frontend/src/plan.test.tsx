import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PlanPanel } from './components/PlanPanel'
import { api } from './api/client'
import type { PlanResult } from './api/types'

afterEach(() => vi.restoreAllMocks())

const reply = (over: Partial<PlanResult> = {}): PlanResult => ({
  ok: true,
  sizes: [50, 100, 200],
  seeds: 5,
  questions: 6,
  segments: 3,
  regimes: [
    { name: 'obvious', separation: 2 },
    { name: 'moderate', separation: 1 },
    { name: 'subtle', separation: 0.6 },
  ],
  cells: [
    ...[50, 100, 200].map((n) => ({
      regime: 'obvious', separation: 2, n_people: n, runs: 5, right_k: 5,
      hit_rate: 1, mean_ari: 0.95, confidently_wrong: 0,
    })),
    ...[[50, 0], [100, 4], [200, 5]].map(([n, hit]) => ({
      regime: 'moderate', separation: 1, n_people: n, runs: 5, right_k: hit,
      hit_rate: hit / 5, mean_ari: 0.6, confidently_wrong: 0,
    })),
    ...[50, 100, 200].map((n) => ({
      regime: 'subtle', separation: 0.6, n_people: n, runs: 5, right_k: 0,
      hit_rate: 0, mean_ari: 0.2, confidently_wrong: 0,
    })),
  ],
  recommended_n: 100,
  subtle_reachable: false,
  prose: 'unused by the panel',
  ...over,
})

/**
 * The planner is the one thing in this app that runs with no data, and the only thing a user
 * consults before spending money on fieldwork. What it must never do is imply more certainty than
 * it has.
 */
describe('the study planner panel', () => {
  it('shows the sweep and the sample size to field', async () => {
    vi.spyOn(api, 'plan').mockResolvedValue(reply())
    render(<PlanPanel busy={false} setBusy={() => {}} />)
    await userEvent.click(screen.getByText(/how many people you need/))
    await userEvent.click(screen.getByRole('button', { name: /Work it out/ }))

    await waitFor(() => expect(screen.getByText(/Field about/)).toBeTruthy())
    expect(screen.getByText(/Field about/).textContent).toContain('100')
    // The counts, not percentages: "4/5" carries how much evidence is behind it, "80%" hides it.
    const rows = screen.getAllByRole('row').slice(1)
    expect(rows[0].textContent).toContain('5/5')
    expect(rows.map((r) => (r as HTMLTableRowElement).cells[0].textContent)).toEqual(
      ['50', '100', '200'],
    )
  })

  it('warns when more respondents would not help', async () => {
    vi.spyOn(api, 'plan').mockResolvedValue(reply({ subtle_reachable: false }))
    render(<PlanPanel busy={false} setBusy={() => {}} />)
    await userEvent.click(screen.getByText(/how many people you need/))
    await userEvent.click(screen.getByRole('button', { name: /Work it out/ }))
    await waitFor(() => expect(screen.getByText(/will not rescue subtle/)).toBeTruthy())
    // And it must never claim to know the segments are real.
    expect(screen.getByText(/cannot tell you whether your segments exist/)).toBeTruthy()
  })

  it('says plainly when no sample size worked, instead of naming one anyway', async () => {
    vi.spyOn(api, 'plan').mockResolvedValue(reply({ recommended_n: null }))
    render(<PlanPanel busy={false} setBusy={() => {}} />)
    await userEvent.click(screen.getByText(/how many people you need/))
    await userEvent.click(screen.getByRole('button', { name: /Work it out/ }))
    await waitFor(() => expect(screen.getByText(/No sample size tried was reliable/)).toBeTruthy())
    expect(screen.queryByText(/Field about/)).toBeNull()
  })

  it('surfaces a server error rather than an empty table', async () => {
    vi.spyOn(api, 'plan').mockResolvedValue({ ok: false, error: 'Plan for between 2 and 8 segments.' })
    render(<PlanPanel busy={false} setBusy={() => {}} />)
    await userEvent.click(screen.getByText(/how many people you need/))
    await userEvent.click(screen.getByRole('button', { name: /Work it out/ }))
    await waitFor(() => expect(screen.getByText(/between 2 and 8 segments/)).toBeTruthy())
    expect(screen.queryByRole('table')).toBeNull()
  })
})
