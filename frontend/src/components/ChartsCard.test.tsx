import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { ChartsCard } from './ChartsCard'
import { chart } from '../test/fixtures'

describe('the charts card', () => {
  it('names every chart in plain English rather than by its id', async () => {
    render(<ChartsCard charts={[chart('map'), chart('profiles'), chart('heatmap')]} />)
    for (const label of ['Segment map', 'What differs', 'Full grid']) {
      expect(screen.getByRole('tab', { name: label })).toBeInTheDocument()
    }
  })

  it('shows one chart at a time and switches on click', async () => {
    const user = userEvent.setup()
    const { container } = render(<ChartsCard charts={[chart('map'), chart('heatmap')]} />)

    const visible = () =>
      [...container.querySelectorAll('.cpane')]
        .filter((pane) => !pane.classList.contains('hide'))
        .map((pane) => pane.querySelector('svg')?.dataset.id)

    expect(visible()).toEqual(['map'])
    await user.click(screen.getByRole('tab', { name: 'Full grid' }))
    expect(visible()).toEqual(['heatmap'])
    expect(screen.getByRole('tab', { name: 'Full grid' })).toHaveAttribute('aria-selected', 'true')
  })

  it('keeps every pane in the document, because print un-hides all of them', () => {
    // The print stylesheet drops .ctabs and shows every .cpane. Unmounting the inactive ones
    // would mean a circulated PDF silently carried one chart out of six.
    const { container } = render(<ChartsCard charts={[chart('map'), chart('fit'), chart('k')]} />)
    expect(container.querySelectorAll('.cpane')).toHaveLength(3)
    expect(container.querySelectorAll('.cpane svg')).toHaveLength(3)
  })

  it('draws nothing at all when there are no charts', () => {
    const { container } = render(<ChartsCard charts={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})
