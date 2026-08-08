import { useState } from 'react'
import { api } from '../api/client'
import { isFailure, type PlanCell, type PlanResult } from '../api/types'

/**
 * How many people do you need? Asked and answered before there is any data.
 *
 * This is the one thing in the app that runs with no file, which is why it lives on the start
 * screen rather than beside a result: the question comes up while you are still deciding whether
 * to field at all, and the answer changes what you commission.
 *
 * The wait is real — every cell is a full simulated study put through the actual analysis — so the
 * button says so before it is pressed rather than leaving someone watching a spinner and wondering
 * whether it has hung.
 */
export function PlanPanel({ busy, setBusy }: { busy: boolean; setBusy: (b: boolean) => void }) {
  const [open, setOpen] = useState(false)
  const [questions, setQuestions] = useState(6)
  const [segments, setSegments] = useState(3)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PlanResult | null>(null)

  async function run() {
    setRunning(true)
    setBusy(true)
    setError(null)
    setResult(null)
    const reply = await api.plan(questions, segments)
    setBusy(false)
    setRunning(false)
    if (isFailure(reply)) setError(reply.error)
    else setResult(reply)
  }

  const cellFor = (regime: string, n: number): PlanCell | undefined =>
    result?.cells.find((c) => c.regime === regime && c.n_people === n)

  return (
    <details className="card" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>
        <span className="chev">▸</span>
        Planning a survey? Work out how many people you need
      </summary>

      <p className="hint" style={{ marginTop: 4 }}>
        No file needed — you have not fielded yet. This builds surveys with a known number of
        segments hidden in them, at the shape you describe, and puts them through the real analysis
        to see which sample sizes actually find the answer.
      </p>

      <div className="chips" style={{ margin: '12px 0', alignItems: 'center', gap: 14 }}>
        <label>
          Questions{' '}
          <input
            type="number"
            min={2}
            max={60}
            value={questions}
            disabled={running}
            onChange={(e) => setQuestions(Number(e.target.value))}
            style={{ width: 70 }}
          />
        </label>
        <label>
          Segments you expect{' '}
          <input
            type="number"
            min={2}
            max={8}
            value={segments}
            disabled={running}
            onChange={(e) => setSegments(Number(e.target.value))}
            style={{ width: 70 }}
          />
        </label>
        <button type="button" className="chip" disabled={busy || running} onClick={run}>
          {running ? 'Simulating…' : 'Work it out (a minute or two)'}
        </button>
      </div>

      {running && (
        <p className="hint">
          Running seventy-five complete segmentations — five sample sizes, three levels of
          distinctness, five simulated studies each. That is why it is slow, and why the answer is
          about what this tool will really do rather than what a formula assumes.
        </p>
      )}

      {error && <p className="note err">{error}</p>}

      {result && (
        <>
          <table className="rank" style={{ marginTop: 6 }}>
            <thead>
              <tr>
                <th scope="col">People</th>
                {result.regimes.map((r) => (
                  <th key={r.name} scope="col">
                    {r.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.sizes.map((n) => (
                <tr key={n}>
                  <td className="num">{n}</td>
                  {result.regimes.map((r) => {
                    const cell = cellFor(r.name, n)
                    if (!cell) return <td key={r.name}>—</td>
                    if (cell.impossible)
                      return (
                        <td key={r.name} className="hint" title="Segments that distinct do not fit
                          on this answer scale, whatever the sample size">
                          n/a
                        </td>
                      )
                    // Found it every time, or not: the raw count rather than a percentage, because
                    // "4 of 5" says how much evidence is behind it and "80%" hides that.
                    return (
                      <td key={r.name} className="num">
                        {cell.right_k}/{cell.runs}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>

          <p className="hint" style={{ marginTop: 10 }}>
            How often the tool found the right number of segments. <b>obvious</b> means groups you
            could spot by eye; <b>moderate</b> is the realistic case and the only one where sample
            size decides; <b>subtle</b> is near the limit of what any method can separate.
          </p>

          <p className={result.recommended_n ? 'note' : 'note warn'} style={{ marginTop: 12 }}>
            {result.recommended_n ? (
              <>
                <b>Field about {result.recommended_n} people.</b> Below that, moderately distinct
                segments start being missed rather than found. Above it you are buying more precise
                profiles, not a better chance of discovering the groups.
              </>
            ) : (
              <>
                <b>No sample size tried was reliable</b>, even for moderately distinct segments.
                The questionnaire probably needs to separate people more sharply — more questions,
                or questions people genuinely disagree about.
              </>
            )}
          </p>

          {!result.subtle_reachable && (
            <p className="note warn" style={{ marginTop: 10 }}>
              <b>More people will not rescue subtle differences.</b> At the subtle level nothing
              tried found the right number reliably. A bigger sample measures a weak signal more
              precisely; it does not make it strong. Spend on sharper questions instead.
            </p>
          )}

          <p className="hint" style={{ marginTop: 10 }}>
            This says how many people are needed to <i>find</i> segments of a given distinctness. It
            cannot tell you whether your segments exist — nothing can, before you field.
          </p>
        </>
      )}
    </details>
  )
}
