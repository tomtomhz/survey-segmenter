import { Component, type ErrorInfo, type ReactNode } from 'react'

/**
 * Stops one broken thing from taking the page with it.
 *
 * React unmounts the *entire* tree when any component throws during render, so before this
 * existed a single malformed result — a project file written by an older version, a field the
 * server stopped sending — replaced the whole app with a blank white page. No header, no
 * composer, no way back except reloading and hoping.
 *
 * That matters more here than in a typical web app: this is a desktop tool someone is running
 * on their own machine against their own data, and a blank window reads as "the tool is broken",
 * not "this one card could not be drawn".
 *
 * A class component because that is still the only way to catch a render error in React.
 */
interface Props {
  children: ReactNode
  /** What to show instead. Given the error so a card can explain itself in context. */
  fallback: (error: Error, reset: () => void) => ReactNode
}

interface State {
  error: Error | null
}

export class Boundary extends Component<Props, State> {
  override state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  override componentDidCatch(error: Error, info: ErrorInfo) {
    // The app runs locally with devtools one keystroke away, so the console is the right place
    // for the stack. Nothing is reported anywhere else: no telemetry leaves this machine.
    console.error('Survey Segmenter: a component failed to render.', error, info.componentStack)
  }

  private reset = () => this.setState({ error: null })

  override render() {
    if (this.state.error) return this.props.fallback(this.state.error, this.reset)
    return this.props.children
  }
}

/** The message shown when a single result card cannot be drawn. */
export function CardFailure({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div className="note err">
      <b>This result could not be displayed.</b> The analysis itself finished — it is the display
      of it that failed, so the download links in your other results still work, and re-running
      the file will usually produce a card that renders.
      <div className="chips" style={{ margin: '10px 0 0' }}>
        <button type="button" className="chip" onClick={onRetry}>
          Try showing it again
        </button>
      </div>
      <p style={{ fontSize: '.8rem', color: 'var(--muted)', margin: '10px 0 0' }}>
        {error.message}
      </p>
    </div>
  )
}

/** The last resort, when something outside any single card failed. */
export function AppFailure({ error }: { error: Error }) {
  return (
    <div style={{ padding: '48px 24px', maxWidth: '34rem', margin: '0 auto' }}>
      <h2 style={{ marginBottom: 8 }}>Survey Segmenter hit a problem it could not recover from.</h2>
      <p style={{ color: 'var(--muted)' }}>
        Your saved projects are on disk and are not affected — reloading will bring them back.
      </p>
      <div className="chips" style={{ margin: '16px 0 0' }}>
        <button type="button" className="chip" onClick={() => window.location.reload()}>
          Reload
        </button>
      </div>
      <p style={{ fontSize: '.8rem', color: 'var(--muted)', marginTop: 16 }}>{error.message}</p>
    </div>
  )
}
