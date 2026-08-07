import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DownloadBar } from './components/DownloadBar'
import { api } from './api/client'
import type { ScoreResult } from './api/types'
import { analysis, file } from './test/fixtures'

afterEach(() => vi.restoreAllMocks())

/**
 * A "no answer" code such as 99 in a follow-up file is scored anyway, and it pulls the respondent
 * towards whichever group is extreme on that question. Measured on the Python side: 35 of 60
 * affected people landed in the wrong segment, and agreement with the truth fell from 0.967 to
 * 0.593.
 *
 * The scored CSV has carried a per-person count since that was found, and the command line prints
 * it — but the app said nothing, so the people most likely to be scoring a follow-up were the
 * least likely to hear about it.
 */
describe('scoring a follow-up file', () => {
  const reply = (over: Partial<ScoreResult> = {}): ScoreResult => ({
    ok: true, n: 250, breakdown: { '0': 90, '1': 80, '2': 80 },
    mean_confidence: 0.58, file: 'scored_new_people.csv', ...over,
  })

  async function scoreWith(result: ScoreResult) {
    vi.spyOn(api, 'score').mockResolvedValue(result)
    render(
      <DownloadBar
        result={analysis()}
        downloads={['segment_assignments.csv']}
        busy={false}
        setBusy={() => {}}
      />,
    )
    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    await userEvent.upload(input, file('followup.csv'), { applyAccept: false })
    await waitFor(() => expect(screen.getByText(/250 people scored/i)).toBeInTheDocument())
  }

  it('warns when answers fall outside the original scale, and names the column', async () => {
    await scoreWith(reply({ off_scale: 60 }))
    expect(screen.getByText(/60 of 250 answered something outside your original scale/i))
      .toBeInTheDocument()
    expect(screen.getByText(/answers_off_the_original_scale/)).toBeInTheDocument()
  })

  it('stays quiet when every answer is on the scale, so the warning means something', async () => {
    await scoreWith(reply({ off_scale: 0 }))
    expect(screen.queryByText(/outside your original scale/i)).not.toBeInTheDocument()
  })

  it('stays quiet against an older server that does not send the count', async () => {
    await scoreWith(reply())
    expect(screen.queryByText(/outside your original scale/i)).not.toBeInTheDocument()
  })
})
