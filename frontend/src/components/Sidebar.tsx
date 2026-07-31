import { useState } from 'react'
import type { ProjectSummary } from '../api/types'
import { relativeTime } from '../lib/labels'

/** Analysed surveys, so someone can come back to one rather than re-running it. */
export function Sidebar({
  projects,
  activeId,
  onOpen,
  onDelete,
  onNew,
}: {
  projects: ProjectSummary[]
  activeId: string | null
  onOpen: (id: string) => void
  onDelete: (id: string) => void
  onNew: () => void
}) {
  // Deleting a project removes the analysis and the original upload from disk, with no undo.
  // A single unlabelled × next to every row is one slip away from destroying an afternoon's work,
  // so the × arms the row and a second, explicit click carries it out.
  const [armed, setArmed] = useState<string | null>(null)

  return (
    <aside className="side">
      <div className="head">
        <span className="eyebrow">Projects</span>
      </div>
      <button type="button" className="newbtn" onClick={onNew}>
        +&nbsp;&nbsp;New analysis
      </button>
      <div className="list">
        {projects.length === 0 ? (
          <div className="empty">
            Your analysed surveys will be saved here, so you can come back to them.
          </div>
        ) : (
          projects.map((project) => {
            const name = project.title || 'Untitled survey'
            return (
              <div className="projrow" key={project.id}>
                <button
                  type="button"
                  className={`proj${project.id === activeId ? ' active' : ''}`}
                  onClick={() => onOpen(project.id)}
                >
                  <div className="t">{name}</div>
                  <div className="m">
                    <span className={`dot ${project.confidence ?? 'unknown'}`} />
                    {summarise(project)}
                  </div>
                  <div className="m">{relativeTime(project.updated)}</div>
                </button>
                {armed === project.id ? (
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
                  <button
                    type="button"
                    className="xbtn"
                    aria-label={`Delete ${name}`}
                    title="Delete this project"
                    onClick={() => setArmed(project.id)}
                  >
                    ×
                  </button>
                )}
              </div>
            )
          })
        )}
      </div>
    </aside>
  )
}

function summarise(project: ProjectSummary): string {
  const parts: string[] = []
  if (project.k != null) parts.push(`${project.k} groups`)
  if (project.n_people != null) parts.push(`${project.n_people} people`)
  return parts.join(' · ')
}
