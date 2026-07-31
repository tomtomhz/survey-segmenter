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
          projects.map((project) => (
            <div className="projrow" key={project.id}>
              <button
                type="button"
                className={`proj${project.id === activeId ? ' active' : ''}`}
                onClick={() => onOpen(project.id)}
              >
                <div className="t">{project.title || 'Untitled survey'}</div>
                <div className="m">
                  <span className={`dot ${project.confidence ?? 'unknown'}`} />
                  {summarise(project)}
                </div>
                <div className="m">{relativeTime(project.updated)}</div>
              </button>
              <button
                type="button"
                className="xbtn"
                title="Delete this project"
                onClick={() => onDelete(project.id)}
              >
                ×
              </button>
            </div>
          ))
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
