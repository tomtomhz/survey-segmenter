import { render, screen } from '@testing-library/react'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { StatStrip } from './StatStrip'
import { analysis } from '../test/fixtures'

/**
 * jsdom has no layout, so IntersectionObserver never fires on its own — and neither does it in the
 * headless browser this project uses for visual checks, where a fresh observer on a real element
 * logs nothing and programmatic scrolling dispatches no scroll events either. So the observer is
 * driven by hand here: this is the only place the sticky bar's behaviour is actually verified.
 */
let fire: ((entries: unknown[]) => void) | null = null

class FakeObserver {
  constructor(cb: (entries: unknown[]) => void) {
    fire = cb
  }
  observe() {}
  disconnect() {}
}

vi.stubGlobal('IntersectionObserver', FakeObserver)
afterEach(() => {
  fire = null
})

const scrolledPastTiles = () =>
  act(() => fire?.([{ isIntersecting: false, boundingClientRect: { top: -420 } }]))

const tilesInView = () =>
  act(() => fire?.([{ isIntersecting: true, boundingClientRect: { top: 120 } }]))

describe('keeping the answer in view', () => {
  it('shows nothing extra while the tiles are on screen', () => {
    render(<StatStrip result={analysis({ k: 3, n_people: 240, confidence: 'high' })} />)
    tilesInView()
    expect(document.querySelector('.statbar')).toBeNull()
  })

  it('pins the three facts once the tiles have scrolled by', () => {
    render(<StatStrip result={analysis({ k: 3, n_people: 240, confidence: 'high' })} />)
    scrolledPastTiles()

    const bar = document.querySelector('.statbar')
    expect(bar).not.toBeNull()
    expect(bar!.textContent).toContain('3')
    expect(bar!.textContent).toContain('240')
    expect(bar!.textContent).toContain('High')
  })

  it('does not appear when the tiles are below the fold rather than above it', () => {
    // Scrolled UP past the result, not down through it: the answer is about to come into view, so
    // restating it would be noise. Only `isIntersecting: false` is not enough to know which.
    render(<StatStrip result={analysis()} />)
    act(() => fire?.([{ isIntersecting: false, boundingClientRect: { top: 900 } }]))
    expect(document.querySelector('.statbar')).toBeNull()
  })

  it('hides itself again when the tiles come back', () => {
    render(<StatStrip result={analysis()} />)
    scrolledPastTiles()
    expect(document.querySelector('.statbar')).not.toBeNull()

    tilesInView()
    expect(document.querySelector('.statbar')).toBeNull()
  })

  it('is hidden from screen readers, because it repeats what is already on the page', () => {
    render(<StatStrip result={analysis()} />)
    scrolledPastTiles()
    expect(document.querySelector('.statbar')).toHaveAttribute('aria-hidden', 'true')
    // The real tiles stay announced.
    expect(screen.getByText('Groups found')).toBeInTheDocument()
  })
})
