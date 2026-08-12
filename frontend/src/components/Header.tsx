/** App chrome. Quit is a real button because the app is a local process, not a website. */
export function Header({
  onNew,
  onSettings,
  onToggleProjects,
  projectsOpen,
}: {
  onNew: () => void
  onSettings: () => void
  onToggleProjects: () => void
  projectsOpen: boolean
}) {
  return (
    <header>
      {/* Shown only on the widths where the sidebar is hidden. Below 820px the projects column
          was simply removed with nothing to bring it back, so every saved study — and renaming,
          pinning, searching and deleting them — was unreachable on a narrow window. */}
      <button
        type="button"
        className="hbtn ghost projtoggle"
        aria-expanded={projectsOpen}
        aria-controls="projects-panel"
        onClick={onToggleProjects}
      >
        ☰ Projects
      </button>
      <div className="brand">
        <svg className="mark" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 2l2.6 6.8L21.5 12l-6.9 3.2L12 22l-2.6-6.8L2.5 12l6.9-3.2z" />
        </svg>
        <span>Survey Segmenter</span>
      </div>
      <span className="sp" />
      <button type="button" className="hbtn ghost" title="Start over" onClick={onNew}>
        + New
      </button>
      <button
        type="button"
        className="hbtn"
        title="Save this conversation as a PDF"
        onClick={() => window.print()}
      >
        Save PDF
      </button>
      <button type="button" className="hbtn" onClick={onSettings}>
        Settings
      </button>
      <button
        type="button"
        className="hbtn"
        onClick={() => {
          window.location.href = '/quit'
        }}
      >
        Quit
      </button>
    </header>
  )
}
