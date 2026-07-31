import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ColumnPicker } from './ColumnPicker'
import { analysis } from '../test/fixtures'

const noop = () => {}

describe('grouping people on different questions', () => {
  it('starts from what the detector chose, and says why each question was or was not used', () => {
    render(
      <ColumnPicker result={analysis()} busy={false} error={null} onRegroup={noop} />,
    )
    expect(screen.getByRole('checkbox', { name: /q1/ })).toBeChecked()
    // A background trait is excluded but visible — being able to pull it in is the whole point.
    expect(screen.getByRole('checkbox', { name: /age/ })).not.toBeChecked()
    expect(screen.getByText('background trait')).toBeInTheDocument()
  })

  it('sends exactly the ticked questions', async () => {
    const user = userEvent.setup()
    const onRegroup = vi.fn()
    render(<ColumnPicker result={analysis()} busy={false} error={null} onRegroup={onRegroup} />)

    await user.click(screen.getByRole('checkbox', { name: /age/ }))
    await user.click(screen.getByRole('button', { name: /Re-group/ }))

    expect(onRegroup).toHaveBeenCalledWith(['q1', 'q2', 'age'])
  })

  it('refuses to re-group on a single question instead of letting the server fail', async () => {
    const user = userEvent.setup()
    const onRegroup = vi.fn()
    render(<ColumnPicker result={analysis()} busy={false} error={null} onRegroup={onRegroup} />)

    await user.click(screen.getByRole('checkbox', { name: /q1/ }))
    await user.click(screen.getByRole('button', { name: /Re-group/ }))

    expect(onRegroup).not.toHaveBeenCalled()
    expect(screen.getByText('Pick at least two questions.')).toBeInTheDocument()
  })

  it('clears its complaint as soon as the user does something about it', async () => {
    const user = userEvent.setup()
    render(<ColumnPicker result={analysis()} busy={false} error={null} onRegroup={noop} />)

    await user.click(screen.getByRole('checkbox', { name: /q1/ }))
    await user.click(screen.getByRole('button', { name: /Re-group/ }))
    expect(screen.getByText('Pick at least two questions.')).toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: /age/ }))
    expect(screen.queryByText('Pick at least two questions.')).not.toBeInTheDocument()
  })

  it('cannot be fired while an analysis is already running', () => {
    render(<ColumnPicker result={analysis()} busy error={null} onRegroup={noop} />)
    expect(screen.getByRole('button', { name: /Re-group/ })).toBeDisabled()
  })

  it('shows nothing when the server sent no column information', () => {
    const { container } = render(
      <ColumnPicker result={analysis({ columns: {} })} busy={false} error={null} onRegroup={noop} />,
    )
    expect(container).toBeEmptyDOMElement()
  })
})
