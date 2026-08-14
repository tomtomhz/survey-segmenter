import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Thread } from './Thread'
import type { Message } from '../lib/thread'

const file = new File(['x'], 'wide_export.csv', { type: 'text/csv' })

const props = {
  busy: false,
  setBusy: () => {},
  regroupError: null,
  onRegroup: () => {},
  onNeedsKey: () => {},
  onAsk: () => {},
  onPolarity: () => {},
}

const question = (over: Partial<Extract<Message, { kind: 'polarity' }>> = {}): Message => ({
  id: 1,
  kind: 'polarity',
  note: 'That looks like a best-worst survey saved with one row per PERSON.',
  file,
  codes: [
    { code: 2, times: 1440 },
    { code: 1, times: 720 },
    { code: 3, times: 720 },
  ],
  ...over,
})

describe('asking which code means best', () => {
  it('offers every code that is actually in the file, with how often it appears', () => {
    render(<Thread {...props} messages={[question()]} />)
    expect(screen.getByRole('button', { name: /2 means best/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /1 means best/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /3 means best/ })).toBeInTheDocument()
    // The counts are the hint that tells the reader which code is "shown but not picked".
    expect(screen.getByText(/1,440 times/)).toBeInTheDocument()
  })

  it('sends the chosen code back with the same file, so nothing is re-uploaded by hand', async () => {
    const user = userEvent.setup()
    const onPolarity = vi.fn()
    render(<Thread {...props} messages={[question()]} onPolarity={onPolarity} />)

    await user.click(screen.getByRole('button', { name: /3 means best/ }))
    expect(onPolarity).toHaveBeenCalledWith(file, 3)
  })

  it('warns that a wrong answer inverts the ranking rather than failing', () => {
    // This is the whole reason the question exists: stating the polarity backwards does not
    // error, it silently turns the ranking upside down. Measured at Spearman -1.000.
    render(<Thread {...props} messages={[question()]} />)
    expect(screen.getByText(/upside down/)).toBeInTheDocument()
  })

  it('still explains what kind of file it is, above the question', () => {
    render(<Thread {...props} messages={[question()]} />)
    expect(screen.getByText(/one row per PERSON/)).toBeInTheDocument()
  })
})
