import { useState } from 'react'
import { api } from '../api/client'
import { isFailure, type DesignResult } from '../api/types'

/**
 * Build the questionnaire, before anyone has been asked anything.
 *
 * This sits beside the planner on the start screen because both answer questions that come up
 * while you are still deciding what to commission — and because neither needs a file, which is the
 * thing that makes them look out of place next to everything else the app does.
 *
 * Items are pasted rather than uploaded. Someone deciding what to ask has the list in an email or
 * a slide; making them save a .txt first is a step that exists only because the command line
 * needed it.
 *
 * The CSV is built into a Blob here rather than fetched from a download route, because a design
 * has no session behind it — see the endpoint's own note on why inventing one would be worse.
 */
export function DesignPanel({ busy, setBusy }: { busy: boolean; setBusy: (b: boolean) => void }) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [perScreen, setPerScreen] = useState(4)
  const [screens, setScreens] = useState(10)
  const [people, setPeople] = useState(200)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DesignResult | null>(null)

  const items = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)

  async function run() {
    setRunning(true)
    setBusy(true)
    setError(null)
    setResult(null)
    const reply = await api.design(items, perScreen, screens, people)
    setBusy(false)
    setRunning(false)
    if (isFailure(reply)) setError(reply.error)
    else setResult(reply)
  }

  function download() {
    if (!result) return
    // A BOM, because the overwhelmingly likely next step is opening this in Excel, and Excel reads
    // a UTF-8 CSV as the system code page without one — which turns every accented item name into
    // mojibake in the questionnaire itself.
    const blob = new Blob(['﻿' + result.csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'maxdiff_design.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <details className="card" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>
        <span className="chev">▸</span>
        Running a best-worst study? Build the questionnaire
      </summary>

      <p className="hint" style={{ marginTop: 4 }}>
        No file needed. Paste the things you want ranked — one per line — and this writes which
        items each person sees on each screen, ready to import into a survey platform. The design
        decides what the data can possibly say: two items that never appear together cannot be
        compared afterwards, however many people answer.
      </p>

      <textarea
        value={text}
        disabled={running}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        placeholder={'Free next-day delivery\nA longer returns window\nLower prices\n…one per line'}
        style={{ width: '100%', marginTop: 8, fontFamily: 'inherit' }}
        aria-label="Items to compare, one per line"
      />
      <p className="hint">
        {items.length === 0
          ? 'At least three items.'
          : `${items.length} item${items.length === 1 ? '' : 's'}.`}
      </p>

      <div className="chips" style={{ margin: '12px 0', alignItems: 'center', gap: 14 }}>
        <label>
          Items per screen{' '}
          <input
            type="number"
            min={2}
            max={6}
            value={perScreen}
            disabled={running}
            onChange={(e) => setPerScreen(Number(e.target.value))}
            style={{ width: 64 }}
          />
        </label>
        <label>
          Screens each{' '}
          <input
            type="number"
            min={2}
            max={15}
            value={screens}
            disabled={running}
            onChange={(e) => setScreens(Number(e.target.value))}
            style={{ width: 64 }}
          />
        </label>
        <label>
          Versions{' '}
          <input
            type="number"
            min={20}
            max={300}
            value={people}
            disabled={running}
            onChange={(e) => setPeople(Number(e.target.value))}
            style={{ width: 70 }}
          />
        </label>
        <button
          type="button"
          className="chip"
          disabled={busy || running || items.length < 3}
          onClick={run}
        >
          {running ? 'Building…' : 'Build the questionnaire'}
        </button>
      </div>

      {running && (
        <p className="hint">
          Balancing which items appear together, which is the part that decides whether the ranking
          can be trusted. A large list can take twenty seconds or so.
        </p>
      )}

      {error && <p className="note err">{error}</p>}

      {result && (
        <>
          <p className="note" style={{ marginTop: 6 }}>
            <b>
              {result.report.sets_per_respondent} screens of {result.report.items_per_set}, over{' '}
              {result.report.n_items} items.
            </b>{' '}
            Each item is shown {result.report.times_each_item_shown[0]}–
            {result.report.times_each_item_shown[1]} times across the study, and each pair of items
            appears together {result.report.pair_appearances[0]}–{result.report.pair_appearances[1]}{' '}
            times.
          </p>

          {result.report.never_paired > 0 ? (
            <p className="note warn">
              <b>{result.report.never_paired} pairs of items never appear together.</b> Those items
              can only be compared through other items, which is weaker evidence. More screens per
              person, or more versions, would fix it.
            </p>
          ) : (
            <p className="hint">
              Every pair of items appears together somewhere, so every comparison the study needs to
              make is supported directly.
            </p>
          )}

          {!result.report.evenly_divisible && (
            <p className="hint">
              The {result.report.sets_per_respondent} × {result.report.items_per_set} slots each
              person sees is not a multiple of {result.report.n_items} items, so some items are
              shown once more than others. That is arithmetic rather than a flaw, and it evens out
              across people.
            </p>
          )}

          <div className="chips" style={{ marginTop: 12 }}>
            <button type="button" className="chip" onClick={download}>
              Download the design (CSV)
            </button>
          </div>

          <p className="hint" style={{ marginTop: 10 }}>
            One row per item shown, which is the shape most survey platforms import. Field it, then
            bring the answers back here: add a <code>choice</code> column saying best or worst
            against the item each person picked, and analyse that file.
          </p>
        </>
      )}
    </details>
  )
}
