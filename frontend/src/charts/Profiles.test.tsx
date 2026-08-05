import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { Profiles } from './Profiles'
import { Gorge } from './Gorge'
import { usableProfiles, usableGorge, SUPPORTED_SPEC_VERSION } from './spec'
import type { ProfilesSpec, GorgeSpec } from './spec'

function profiles(over: Partial<ProfilesSpec> = {}): ProfilesSpec {
  return {
    version: SUPPORTED_SPEC_VERSION,
    kind: 'profiles',
    items: ['Price matters most', 'I buy on impulse'],
    segments: [
      { index: 0, label: 'Loyal Fans', colour: '#2a78d6', colour_dark: '#3987e5', marker: 'o' },
      { index: 1, label: 'Price Hunters', colour: '#eb6834', colour_dark: '#d95926', marker: 's' },
    ],
    values: [[3.1, 4.7], [4.7, 1.3]],
    measure: 'average',
    trimmed: 0,
    ...over,
  }
}

function gorge(over: Partial<GorgeSpec> = {}): GorgeSpec {
  return {
    version: SUPPORTED_SPEC_VERSION,
    kind: 'gorge',
    edges: [0, 0.25, 0.5, 0.75, 1],
    counts: [10, 40, 30, 20],
    median: 0.45,
    people: 100,
    ...over,
  }
}

describe('the profile dots', () => {
  it('refuses a group that has no value for every question', () => {
    expect(usableProfiles(profiles({ values: [[1], [2, 3]] }))).toBeNull()
    expect(usableProfiles(profiles())).not.toBeNull()
  })

  it('settles whether two near-touching dots are a real difference', async () => {
    // The static chart raises this question and cannot answer it: dots that nearly touch might be
    // a meaningful gap or noise. The spread across the row turns it into a number.
    const user = userEvent.setup()
    render(<Profiles spec={profiles()} title="What differs" />)
    await user.hover(screen.getByRole('button', { name: /Loyal Fans, Price matters most/ }))
    const reading = screen.getByRole('status')
    expect(reading.textContent).toMatch(/average 3\.10/)
    expect(reading.textContent).toMatch(/span 1\.60 on this question/)
  })

  it('calls out a question that separates nobody', async () => {
    const user = userEvent.setup()
    render(<Profiles spec={profiles({ values: [[3.1, 4.7], [3.15, 1.3]] })} title="What differs" />)
    await user.hover(screen.getByRole('button', { name: /Loyal Fans, Price matters most/ }))
    expect(screen.getByRole('status').textContent).toMatch(/barely any difference at all/)
  })

  it('never anchors the axis at zero on a rating scale', () => {
    // A 1-5 scale drawn from 0 spends a fifth of the axis on a region no respondent can occupy.
    render(<Profiles spec={profiles()} title="What differs" />)
    const marks = [...document.querySelectorAll('.iprofiles-mark')]
    const xs = marks.map((m) => Number(m.getAttribute('transform')!.match(/translate\(([\d.]+)/)![1]))
    // The lowest value (1.3) must not sit at the very left edge, which is what a zero-anchored
    // axis would do to a range that starts at 1.3.
    expect(Math.min(...xs)).toBeGreaterThan(210)
  })
})

describe('the gorge', () => {
  it('refuses counts that do not line up with their edges', () => {
    expect(usableGorge(gorge({ counts: [1, 2] }))).toBeNull()
    expect(usableGorge(gorge())).not.toBeNull()
  })

  it('turns the shape into a number', async () => {
    // "Looks a bit bimodal" is not a finding. How many people sit in a band, and what share of the
    // study that is, can be acted on.
    const user = userEvent.setup()
    render(<Gorge spec={gorge()} title="Do they separate" />)
    await user.hover(screen.getAllByRole('button')[1])
    const reading = screen.getByRole('status')
    expect(reading.textContent).toMatch(/40 respondents/)
    expect(reading.textContent).toMatch(/40\.0%/)
    expect(reading.textContent).toMatch(/between 0\.25 and 0\.50/)
  })

  it('is drawn in ink, because it describes everybody', () => {
    // Wearing a segment's colour implied it was about that segment.
    render(<Gorge spec={gorge()} title="Do they separate" />)
    for (const bar of document.querySelectorAll('.igorge rect')) {
      expect(bar.getAttribute('fill')).toBe('currentColor')
    }
  })

  it('every band is reachable by keyboard', () => {
    render(<Gorge spec={gorge()} title="Do they separate" />)
    expect(document.querySelectorAll('.igorge rect[tabindex="0"]').length).toBe(4)
  })
})
