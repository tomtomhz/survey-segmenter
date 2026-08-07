import type { RankedItem } from '../api/types'

/**
 * What a best-worst study was fielded to find out: which items people want, strongest first.
 *
 * This is shown as its own card rather than left inside the statistical report, because that
 * report is a collapsed `<details>` sitting below the charts — so a study whose entire purpose
 * was to rank items was hiding the ranking two clicks down, under a heading about statistics.
 *
 * The column that earns its place is the last one. A ranking prints an order whether or not the
 * data supports one, and a reader acts on the order rather than on the intervals nobody put in
 * the table. Where two neighbours overlap, this says so on the row itself.
 */
export function RankingCard({ ranking }: { ranking?: RankedItem[] | null }) {
  if (!ranking || ranking.length === 0) return null

  const tied = ranking.filter((r) => r.clear_of_next === false)
  const strongest = ranking[0]
  const weakest = ranking[ranking.length - 1]
  // Every position unsettled means the study found no differences at all, not that it found a
  // ranking with caveats. `clear_of_next` is null on the last row, which is absence of a question
  // rather than a negative answer, so it is excluded from both counts.
  const pairs = ranking.filter((r) => r.clear_of_next !== null && r.clear_of_next !== undefined)
  const nothingSettled = pairs.length > 0 && pairs.every((r) => r.clear_of_next === false)
  const topUnsettled = !nothingSettled && ranking[0]?.clear_of_next === false
  // `!= null` deliberately, not `!== null`: the type says `number | null`, but JSON can simply
  // omit the key, and TypeScript cannot see that. A row without `low` at all then passed the
  // `!== null` test and reached `row.low.toFixed(2)` — "Cannot read properties of undefined".
  const hasIntervals = ranking.some((r) => r.low != null)
  // The bars are drawn from the full span of the scale rather than from zero: these are relative
  // preferences centred on zero, so half of them are negative and a zero-based bar would render
  // the bottom half as nothing at all.
  const lo = Math.min(...ranking.map((r) => r.low ?? r.score))
  const hi = Math.max(...ranking.map((r) => r.high ?? r.score))
  const span = hi - lo || 1
  const pct = (v: number) => ((v - lo) / span) * 100

  return (
    <div className="card">
      <h3 style={{ margin: '0 0 2px' }}>What matters most</h3>
      {/* The opening line has to agree with the table under it. Announcing a winner and then
          explaining two lines later that nothing was separated is the card arguing with itself. */}
      <p className="hint" style={{ margin: '0 0 12px' }}>
        {nothingSettled ? (
          <>
            Across everyone, before any grouping — except that <b>this study did not separate these
            items</b>. <b>{strongest.item}</b> scores highest and <b>{weakest.item}</b> lowest, but
            no item is clearly ahead of any other, so do not read this as a ranking.
          </>
        ) : topUnsettled ? (
          <>
            Across everyone, before any grouping. <b>{strongest.item}</b> scores highest and{' '}
            <b>{weakest.item}</b> lowest — though <b>{strongest.item}</b> is not clearly ahead of{' '}
            <b>{ranking[1].item}</b>, so treat the top as a pair rather than a winner.
          </>
        ) : (
          <>
            Across everyone, before any grouping. <b>{strongest.item}</b> comes out strongest and{' '}
            <b>{weakest.item}</b> weakest.
          </>
        )}
      </p>
      <table className="rank">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Item</th>
            <th scope="col" style={{ textAlign: 'right' }}>
              Score
            </th>
            {hasIntervals && <th scope="col">How sure</th>}
          </tr>
        </thead>
        <tbody>
          {ranking.map((row) => (
            <tr key={row.item}>
              <td className="num">{row.rank}</td>
              <td>
                {row.item}
                {/* A real space, not only the badge's left margin: a screen reader reads the cell's
                    text, and without it the row announces as "Supporttied with next". */}
                {row.clear_of_next === false && ' '}
                {row.clear_of_next === false && (
                  <span
                    className="tie"
                    title={
                      row.prob_ahead != null
                        ? `${(row.prob_ahead * 100).toFixed(0)}% chance this really is ahead of the item below — under the 95% mark this study treats as settled`
                        : 'This study did not establish which of these two is ahead'
                    }
                  >
                    {/* The number, not just the word. "58% sure" and "93% sure" are different
                        findings, and a single "tied" label reports them identically. */}
                    {row.prob_ahead != null
                      ? `${(row.prob_ahead * 100).toFixed(0)}% sure of this order`
                      : 'order not settled'}
                  </span>
                )}
              </td>
              <td className="num" style={{ textAlign: 'right' }}>
                {row.score > 0 ? '+' : ''}
                {row.score.toFixed(2)}
              </td>
              {hasIntervals && (
                <td>
                  {row.low != null && row.high != null ? (
                    <span
                      className="range"
                      aria-label={`${row.low.toFixed(2)} to ${row.high.toFixed(2)}`}
                      title={`95% range: ${row.low.toFixed(2)} to ${row.high.toFixed(2)}`}
                    >
                      <span
                        className="range-bar"
                        style={{
                          left: `${pct(row.low)}%`,
                          width: `${Math.max(pct(row.high) - pct(row.low), 1.5)}%`,
                        }}
                      />
                    </span>
                  ) : null}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {hasIntervals && (
        <p className={tied.length > 0 ? 'note warn' : 'note'} style={{ marginTop: 12 }}>
          {tied.length > 0 ? (
            <>
              <b>
                {tied.length} {tied.length === 1 ? 'position is' : 'positions are'} not settled
              </b>{' '}
              at the usual 95% mark. The percentage on those rows is the chance that pair really is
              the right way round — read it rather than the position. More respondents, or more
              questions each, is what would settle them.
            </>
          ) : (
            <>
              <b>Every item beats the one below it with at least 95% certainty,</b> so this order is
              one you can act on.
            </>
          )}
        </p>
      )}
      <p className="hint" style={{ marginTop: 10 }}>
        Scores are relative and centred on zero: above zero is wanted more than the average item,
        below zero less. Only the differences carry meaning — <b>+2.00</b> is not “twice” <b>+1.00</b>,
        it is further ahead of the average.
      </p>
    </div>
  )
}
