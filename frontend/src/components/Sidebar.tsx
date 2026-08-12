import { useMemo, useRef, useState } from 'react'
import type { ProjectSummary } from '../api/types'
import { relativeTime } from '../lib/labels'

/**
 * Analysed surveys, so someone can come back to one rather than re-running it.
 *
 * This started as a flat list of filenames, which is fine for five projects and unusable for
 * sixty. A real workspace fills up with `export (3).csv` and four copies of `s.csv` that cannot be
 * told apart, so three things were added, in the order they were missed:
 *
 * * **A name you chose.** The filename is what the survey arrived as, not what the study was.
 * * **Search**, because past about twenty rows scrolling is not finding.
 * * **Date grouping**, so "the one from this morning" is a place to look rather than a scan.
 */
export function Sidebar({
  projects,
  activeId,
  onOpen,
  onDelete,
  onRename,
  onPin,
  onDeleteMany,
  onNew,
  total,
}: {
  projects: ProjectSummary[]
  activeId: string | null
  onOpen: (id: string) => void
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
  onPin: (id: string, pinned: boolean) => void
  onDeleteMany: (ids: string[]) => void
  onNew: () => void
  /** How many projects exist. Larger than the list when the cap has bitten. */
  total?: number
}) {
  // Deleting a project removes the analysis and the original upload from disk, with no undo.
  // A single unlabelled × next to every row is one slip away from destroying an afternoon's work,
  // so the × arms the row and a second, explicit click carries it out.
  const [armed, setArmed] = useState<string | null>(null)
  const [renaming, setRenaming] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [query, setQuery] = useState('')
  // Selection is a mode rather than always-on checkboxes: a checkbox beside every row makes the
  // list look like a form, and picking a project to open is the common action by a wide margin.
  const [selecting, setSelecting] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [confirmingMany, setConfirmingMany] = useState(false)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return projects
    return projects.filter((p) => (p.title || '').toLowerCase().includes(needle))
  }, [projects, query])

  // Pinned first as their own section, then by when they were last touched, in the order a person
  // thinks about time. The date buckets are computed from the same `updated` string the rows show,
  // so a row can never appear under a heading that disagrees with its own timestamp.
  const groups = useMemo(() => {
    const pinned = matches.filter((p) => p.pinned)
    const rest = matches.filter((p) => !p.pinned)
    return [
      ...(pinned.length ? ([['Pinned', pinned]] as [string, ProjectSummary[]][]) : []),
      ...groupByAge(rest),
    ]
  }, [matches])

  function startRename(project: ProjectSummary) {
    setArmed(null)
    setRenaming(project.id)
    setDraft(project.title || '')
    // Focus after the input exists. Selecting the text means typing replaces the old name, which
    // is what renaming usually is, while leaving it editable for a small correction.
    window.setTimeout(() => {
      inputRef.current?.focus()
      inputRef.current?.select()
    }, 0)
  }

  function commitRename(id: string) {
    const cleaned = draft.trim()
    const previous = projects.find((p) => p.id === id)?.title || ''
    setRenaming(null)
    if (!cleaned || cleaned === previous) return
    // Renaming while a search is active would otherwise make the row vanish, because the name you
    // searched for is the one you just replaced: found by doing it — search "typing", rename to
    // something else, and the list empties with "Nothing matches". Clearing the filter keeps the
    // thing you just acted on in front of you, which is the point of having acted on it.
    if (query.trim() && !cleaned.toLowerCase().includes(query.trim().toLowerCase())) setQuery('')
    onRename(id, cleaned)
  }

  function toggleSelected(id: string) {
    setSelected((previous) => {
      const next = new Set(previous)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
    setConfirmingMany(false)
  }

  function leaveSelectMode() {
    setSelecting(false)
    setSelected(new Set())
    setConfirmingMany(false)
  }

  // Only what is on screen can be selected, so "select all" means the rows the search left —
  // which is what makes clearing a category of old work practical without a bulk action that
  // reaches things the user cannot see.
  const visibleIds = matches.map((p) => p.id)
  const allVisibleSelected = visibleIds.length > 0 && visibleIds.every((id) => selected.has(id))

  return (
    <aside className="side">
      <div className="head">
        <span className="eyebrow">Projects</span>
      </div>
      <button type="button" className="newbtn" onClick={onNew}>
        +&nbsp;&nbsp;New analysis
      </button>

      {/* Only once there is enough to search. A filter box above three rows is furniture. */}
      {projects.length > 6 && (
        <div className="side-search">
          <input
            type="search"
            value={query}
            placeholder="Search projects"
            aria-label="Search projects by name"
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            type="button"
            className="linkbtn"
            onClick={() => (selecting ? leaveSelectMode() : setSelecting(true))}
          >
            {selecting ? 'Cancel' : 'Select'}
          </button>
        </div>
      )}

      {selecting && (
        <div className="side-bulk">
          <button
            type="button"
            className="linkbtn"
            onClick={() =>
              setSelected(allVisibleSelected ? new Set() : new Set(visibleIds))
            }
          >
            {allVisibleSelected ? 'Clear' : `Select all ${visibleIds.length}`}
          </button>
          {selected.size > 0 &&
            (confirmingMany ? (
              <span className="bulkconfirm">
                <button
                  type="button"
                  className="xbtn danger"
                  onClick={() => {
                    onDeleteMany([...selected])
                    leaveSelectMode()
                  }}
                >
                  Delete {selected.size}
                </button>
                <button type="button" className="xbtn" onClick={() => setConfirmingMany(false)}>
                  Keep
                </button>
              </span>
            ) : (
              /* Two clicks, like the single-row delete. This removes the analysis AND the original
                 upload for every selected project, with no undo, so the count is stated in the
                 confirming button rather than left to whatever the user thinks they ticked. */
              <button type="button" className="linkbtn danger"
                      onClick={() => setConfirmingMany(true)}>
                Delete {selected.size} selected
              </button>
            ))}
        </div>
      )}

      <div className="list">
        {projects.length === 0 ? (
          <div className="empty">
            Your analysed surveys will be saved here, so you can come back to them.
          </div>
        ) : matches.length === 0 ? (
          <div className="empty">
            Nothing matches &ldquo;{query.trim()}&rdquo;. Projects are searched by name — rename one
            from the &ldquo;⋯&rdquo; button if it is still called what the file was.
          </div>
        ) : (
          groups.map(([heading, rows]) => (
            <div key={heading}>
              <div className="side-group">{heading}</div>
              {rows.map((project) => {
                const name = project.title || 'Untitled survey'
                if (renaming === project.id) {
                  return (
                    <div className="projrow renaming" key={project.id}>
                      <input
                        ref={inputRef}
                        className="renameinput"
                        value={draft}
                        maxLength={80}
                        aria-label={`Rename ${name}`}
                        onChange={(e) => setDraft(e.target.value)}
                        onBlur={() => commitRename(project.id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') commitRename(project.id)
                          // Escape has to abandon the edit, not save it — and it must not fall
                          // through to the blur handler, which would save the very thing the user
                          // just asked to throw away.
                          if (e.key === 'Escape') {
                            setDraft(project.title || '')
                            setRenaming(null)
                          }
                        }}
                      />
                    </div>
                  )
                }
                return (
                  <div className="projrow" key={project.id}>
                    {selecting && (
                      <input
                        type="checkbox"
                        className="selbox"
                        checked={selected.has(project.id)}
                        aria-label={`Select ${name}`}
                        onChange={() => toggleSelected(project.id)}
                      />
                    )}
                    <button
                      type="button"
                      className={`proj${project.id === activeId ? ' active' : ''}`}
                      // In select mode the row picks rather than opens. Opening a project mid-
                      // selection would swap the whole page out from under a half-made choice.
                      onClick={() => (selecting ? toggleSelected(project.id) : onOpen(project.id))}
                      onDoubleClick={() => !selecting && startRename(project)}
                    >
                      <div className="t">{name}</div>
                      <div className="m">
                        <span className={`dot ${project.confidence ?? 'unknown'}`} />
                        {summarise(project)}
                      </div>
                      <div className="m">{relativeTime(project.updated)}</div>
                    </button>
                    {selecting ? null : armed === project.id ? (
                      <div className="confirm">
                        <button
                          type="button"
                          className="xbtn danger"
                          aria-label={`Confirm deleting ${name}`}
                          onClick={() => {
                            setArmed(null)
                            onDelete(project.id)
                          }}
                        >
                          Delete
                        </button>
                        <button
                          type="button"
                          className="xbtn"
                          aria-label="Keep this project"
                          onClick={() => setArmed(null)}
                        >
                          Keep
                        </button>
                      </div>
                    ) : (
                      <div className="rowacts">
                        <button
                          type="button"
                          className={`xbtn${project.pinned ? ' pinned' : ''}`}
                          aria-label={project.pinned ? `Unpin ${name}` : `Pin ${name}`}
                          title={project.pinned ? 'Unpin' : 'Pin to the top'}
                          aria-pressed={!!project.pinned}
                          onClick={() => onPin(project.id, !project.pinned)}
                        >
                          {project.pinned ? '★' : '☆'}
                        </button>
                        <button
                          type="button"
                          className="xbtn"
                          aria-label={`Rename ${name}`}
                          title="Rename"
                          onClick={() => startRename(project)}
                        >
                          ⋯
                        </button>
                        <button
                          type="button"
                          className="xbtn"
                          aria-label={`Delete ${name}`}
                          title="Delete this project"
                          onClick={() => setArmed(project.id)}
                        >
                          ×
                        </button>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          ))
        )}
        {/* The list is capped. Before this the cap was silent, which looks exactly like projects
            having been deleted — so it says what it is showing and what exists. */}
        {total != null && total > projects.length && (
          <div className="side-more">
            Showing the {projects.length} most recent of {total}. Pin one to keep it here.
          </div>
        )}
      </div>
    </aside>
  )
}

/**
 * Bucket projects by age, keeping the order the server sent (newest first) within each bucket.
 *
 * Empty buckets are dropped rather than rendered as bare headings, so a workspace where everything
 * happened today shows one heading rather than five, four of which say nothing.
 */
export function groupByAge(
  projects: ProjectSummary[],
  now: Date = new Date(),
): [string, ProjectSummary[]][] {
  const startOfToday = new Date(now)
  startOfToday.setHours(0, 0, 0, 0)
  const day = 24 * 60 * 60 * 1000

  const buckets: [string, ProjectSummary[]][] = [
    ['Today', []],
    ['Yesterday', []],
    ['Previous 7 days', []],
    ['Previous 30 days', []],
    ['Older', []],
  ]

  for (const project of projects) {
    const when = project.updated ? new Date(project.updated) : null
    // A project with no timestamp, or an unparseable one, still has to appear somewhere. "Older"
    // is the honest bucket: it makes no claim the data does not support.
    const age = when && !Number.isNaN(when.getTime())
      ? startOfToday.getTime() - new Date(when).setHours(0, 0, 0, 0)
      : Number.POSITIVE_INFINITY
    const index = age <= 0 ? 0 : age <= day ? 1 : age <= 7 * day ? 2 : age <= 30 * day ? 3 : 4
    buckets[index][1].push(project)
  }
  return buckets.filter(([, rows]) => rows.length > 0)
}

function summarise(project: ProjectSummary): string {
  const parts: string[] = []
  if (project.k != null) parts.push(`${project.k} groups`)
  if (project.n_people != null) parts.push(`${project.n_people} people`)
  return parts.join(' · ')
}
