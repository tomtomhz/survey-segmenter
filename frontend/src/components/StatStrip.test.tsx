import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { StatStrip } from './StatStrip'
import { analysis } from '../test/fixtures'

describe('the result at a glance', () => {
  it('shows the number of groups, the sample size and the confidence', () => {
    render(<StatStrip result={analysis({ k: 3, n_people: 420, confidence: 'high' })} />)
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText('420')).toBeInTheDocument()
    expect(screen.getByText('High')).toBeInTheDocument()
  })

  it('keeps the caveat folded away, so the verdict is still what is scanned', () => {
    render(<StatStrip result={analysis({ confidence: 'high' })} />)
    const details = screen.getByText(/actually mean\?/).closest('details')
    expect(details).not.toBeNull()
    expect(details).not.toHaveAttribute('open')
  })

  it('says a green light means a floor, not an exact headcount', async () => {
    // The measured caveat is the point of this component's addition: "Trust these groups" on its
    // own invites a reader to treat the number as exact, and sixty planted studies say that is
    // the one thing it does not support — when green is wrong, it has merged two real groups.
    const user = userEvent.setup()
    render(<StatStrip result={analysis({ confidence: 'high' })} />)
    await user.click(screen.getByText(/actually mean\?/))
    expect(screen.getByText(/merged two real groups/)).toBeInTheDocument()
    expect(screen.getByText(/floor, not a headcount/)).toBeInTheDocument()
  })

  it('gives a different, equally concrete account for a red light', async () => {
    const user = userEvent.setup()
    render(<StatStrip result={analysis({ confidence: 'low' })} />)
    await user.click(screen.getByText(/actually mean\?/))
    expect(screen.getByText(/almost never recovered the true grouping/)).toBeInTheDocument()
  })
})
