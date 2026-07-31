/** App chrome. Quit is a real button because the app is a local process, not a website. */
export function Header({
  onNew,
  onSettings,
}: {
  onNew: () => void
  onSettings: () => void
}) {
  return (
    <header>
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
