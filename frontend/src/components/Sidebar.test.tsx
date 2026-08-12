import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Sidebar, groupByAge } from './Sidebar'
import type { ProjectSummary } from '../api/types'

const iso = (daysAgo: number) => {
  const d = new Date()
  d.setDate(d.getDate() - daysAgo)
  return d.toISOString()
}

const project = (over: Partial<ProjectSummary> = {}): ProjectSummary => ({
  id: 'p1',
  title: 's.csv',
  updated: iso(0),
  k: 3,
  n_people: 200,
  confidence: 'high',
  ...over,
})

const props = {
  activeId: null,
  onOpen: () => {},
  onDelete: () => {},
  onRename: () => {},
  onPin: () => {},
  onNew: () => {},
}

describe('the projects list', () => {
  it('does not offer search until there is enough to search', () => {
    const few = Array.from({ length: 4 }, (_, i) => project({ id: `p${i}`, title: `study ${i}` }))
    const { rerender } = render(<Sidebar {...props} projects={few} />)
    expect(screen.queryByLabelText(/Search projects/)).not.toBeInTheDocument()

    const many = Array.from({ length: 9 }, (_, i) => project({ id: `p${i}`, title: `study ${i}` }))
    rerender(<Sidebar {...props} projects={many} />)
    expect(screen.getByLabelText(/Search projects/)).toBeInTheDocument()
  })

  it('filters by name, and says so when nothing matches', async () => {
    const user = userEvent.setup()
    const many = [
      ...Array.from({ length: 7 }, (_, i) => project({ id: `p${i}`, title: `filler ${i}` })),
      project({ id: 'wanted', title: 'Nordic pricing study' }),
    ]
    render(<Sidebar {...props} projects={many} />)

    await user.type(screen.getByLabelText(/Search projects/), 'nordic')
    expect(screen.getByText('Nordic pricing study')).toBeInTheDocument()
    expect(screen.queryByText('filler 0')).not.toBeInTheDocument()

    await user.clear(screen.getByLabelText(/Search projects/))
    await user.type(screen.getByLabelText(/Search projects/), 'zzz')
    expect(screen.getByText(/Nothing matches/)).toBeInTheDocument()
  })

  it('groups by age, and shows no heading for a bucket with nothing in it', () => {
    render(
      <Sidebar
        {...props}
        projects={[project({ id: 'a', title: 'today one' }),
                   project({ id: 'b', title: 'old one', updated: iso(60) })]}
      />,
    )
    expect(screen.getByText('Today')).toBeInTheDocument()
    expect(screen.getByText('Older')).toBeInTheDocument()
    expect(screen.queryByText('Yesterday')).not.toBeInTheDocument()
  })

  it('renames on Enter, because the filename is not what the study was', async () => {
    const user = userEvent.setup()
    const onRename = vi.fn()
    render(<Sidebar {...props} projects={[project()]} onRename={onRename} />)

    await user.click(screen.getByLabelText('Rename s.csv'))
    const box = screen.getByLabelText('Rename s.csv')
    await user.clear(box)
    await user.type(box, 'Pricing study wave 2{Enter}')

    expect(onRename).toHaveBeenCalledWith('p1', 'Pricing study wave 2')
  })

  it('abandons the edit on Escape rather than saving what was typed', async () => {
    // Escape has to beat the blur handler. Saving the very thing the user just asked to throw
    // away is the kind of bug that only shows up on the day someone leans on the key.
    const user = userEvent.setup()
    const onRename = vi.fn()
    render(<Sidebar {...props} projects={[project()]} onRename={onRename} />)

    await user.click(screen.getByLabelText('Rename s.csv'))
    await user.clear(screen.getByLabelText('Rename s.csv'))
    await user.type(screen.getByLabelText('Rename s.csv'), 'half a thought{Escape}')

    expect(onRename).not.toHaveBeenCalled()
    expect(screen.getByText('s.csv')).toBeInTheDocument()
  })

  it('does not fire a rename when the name comes back unchanged', async () => {
    const user = userEvent.setup()
    const onRename = vi.fn()
    render(<Sidebar {...props} projects={[project()]} onRename={onRename} />)

    await user.click(screen.getByLabelText('Rename s.csv'))
    await user.keyboard('{Enter}')
    expect(onRename).not.toHaveBeenCalled()
  })

  it('clears the search when the new name would hide the row you just renamed', async () => {
    // Found by using it: search "typing", rename that project to something else, and the list
    // empties with "Nothing matches" — the name you searched for is the one you just replaced.
    const user = userEvent.setup()
    const many = [
      ...Array.from({ length: 7 }, (_, i) => project({ id: `p${i}`, title: `filler ${i}` })),
      project({ id: 'target', title: 'typing_demo.csv' }),
    ]
    render(<Sidebar {...props} projects={many} onRename={() => {}} />)

    await user.type(screen.getByLabelText(/Search projects/), 'typing')
    await user.click(screen.getByLabelText('Rename typing_demo.csv'))
    const box = screen.getByLabelText('Rename typing_demo.csv')
    await user.clear(box)
    await user.type(box, 'Q3 brand tracker{Enter}')

    expect(screen.getByLabelText(/Search projects/)).toHaveValue('')
    expect(screen.queryByText(/Nothing matches/)).not.toBeInTheDocument()
  })

  it('leaves the search alone when the new name still matches it', async () => {
    const user = userEvent.setup()
    const many = [
      ...Array.from({ length: 7 }, (_, i) => project({ id: `p${i}`, title: `filler ${i}` })),
      project({ id: 'target', title: 'brand tracker' }),
    ]
    render(<Sidebar {...props} projects={many} />)

    await user.type(screen.getByLabelText(/Search projects/), 'brand')
    await user.click(screen.getByLabelText('Rename brand tracker'))
    const box = screen.getByLabelText('Rename brand tracker')
    await user.clear(box)
    await user.type(box, 'Brand tracker Q3{Enter}')

    expect(screen.getByLabelText(/Search projects/)).toHaveValue('brand')
  })

  it('still makes you confirm a delete', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    render(<Sidebar {...props} projects={[project()]} onDelete={onDelete} />)

    await user.click(screen.getByLabelText('Delete s.csv'))
    expect(onDelete).not.toHaveBeenCalled()
    await user.click(screen.getByLabelText('Confirm deleting s.csv'))
    expect(onDelete).toHaveBeenCalledWith('p1')
  })
})

describe('bucketing by age', () => {
  it('puts a project with no usable timestamp in Older rather than dropping it', () => {
    const rows = groupByAge([project({ id: 'x', title: 'no date', updated: undefined })])
    const [heading, items] = rows[0]
    expect(heading).toBe('Older')
    expect(items).toHaveLength(1)
  })

  it('keeps the order the server sent inside each bucket', () => {
    const rows = groupByAge([
      project({ id: 'a', title: 'newer' }),
      project({ id: 'b', title: 'older' }),
    ])
    expect(rows[0][1].map((p) => p.id)).toEqual(['a', 'b'])
  })
})


describe('pinning and the cap', () => {
  it('lifts pinned projects into their own section above the dates', () => {
    render(
      <Sidebar
        {...props}
        projects={[
          project({ id: 'a', title: 'ordinary', updated: iso(0) }),
          project({ id: 'b', title: 'kept', updated: iso(40), pinned: true }),
        ]}
      />,
    )
    const headings = screen.getAllByText(/Pinned|Today|Older/).map((n) => n.textContent)
    expect(headings[0]).toBe('Pinned')
    expect(headings).toContain('Today')
  })

  it('sends the state it wants rather than toggling blind', async () => {
    const user = userEvent.setup()
    const onPin = vi.fn()
    render(<Sidebar {...props} projects={[project({ title: 'study' })]} onPin={onPin} />)

    await user.click(screen.getByLabelText('Pin study'))
    expect(onPin).toHaveBeenCalledWith('p1', true)
  })

  it('offers to unpin something already pinned', async () => {
    const user = userEvent.setup()
    const onPin = vi.fn()
    render(
      <Sidebar {...props} projects={[project({ title: 'study', pinned: true })]} onPin={onPin} />,
    )
    await user.click(screen.getByLabelText('Unpin study'))
    expect(onPin).toHaveBeenCalledWith('p1', false)
  })

  it('says when the list is not showing everything, because a silent cap looks like data loss', () => {
    const rows = Array.from({ length: 3 }, (_, i) => project({ id: `p${i}`, title: `study ${i}` }))
    render(<Sidebar {...props} projects={rows} total={73} />)
    expect(screen.getByText(/Showing the 3 most recent of 73/)).toBeInTheDocument()
  })

  it('says nothing when the list is the whole of it', () => {
    const rows = Array.from({ length: 3 }, (_, i) => project({ id: `p${i}`, title: `study ${i}` }))
    render(<Sidebar {...props} projects={rows} total={3} />)
    expect(screen.queryByText(/Showing the/)).not.toBeInTheDocument()
  })
})
