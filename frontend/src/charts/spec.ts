/**
 * The contract between the two chart renderers.
 *
 * A chart is computed once, in Python, and drawn twice: matplotlib produces the static SVG that
 * goes into the report, the PDF and the image Claude reads, and the components in this folder
 * draw the same numbers interactively. The obvious way to add interactive charts is to write a
 * second chart engine in TypeScript — and then the two quietly disagree about what the data says
 * the first time somebody edits one of them. Sending a spec is what makes that impossible.
 *
 * These types mirror `charts.py`. They are hand-written rather than generated because there is one
 * producer and one consumer and a build step to keep in sync would cost more than it saves — but
 * if you change one side, change the other in the same commit, and bump `SPEC_VERSION` if the
 * shape changes in a way an older interface could not read.
 */

/** Bumped in `charts.py` when the shape changes. Anything else is drawn from the static SVG. */
export const SUPPORTED_SPEC_VERSION = 1

/** Per-segment identity, sent rather than re-derived here.
 *
 * The palette is the part with the measured colour-vision properties, and the marker shapes are
 * what keep segments apart where colour alone cannot. Both belong in exactly one place; copying
 * them into TypeScript is how the two renderings drift apart.
 */
export interface SegmentKey {
  index: number
  label: string
  /** Light-theme hex. */
  colour: string
  /** Dark-theme step — chosen for the dark surface, not derived from the light one. */
  colour_dark: string
  /** matplotlib marker code: o s ^ D v P X * */
  marker: string
}

/** One dot per distinct answer pattern, with how many people share it.
 *
 * Parallel arrays rather than an array of objects: at a few thousand points this is markedly
 * smaller over the wire and cheaper to iterate, and it is the shape numpy produces anyway.
 */
export interface SegmentMapPoints {
  x: number[]
  y: number[]
  segment: number[]
  people: number[]
}

export interface SegmentMapSpec {
  version: number
  kind: 'segment_map'
  points: SegmentMapPoints
  centroids: { segment: number; x: number; y: number }[]
  segments: SegmentKey[]
  /** Share of all variation each drawn direction carries, and the two together. */
  axes: { x_share: number; y_share: number; kept: number }
  /** Respondents represented — the sum of `points.people`, not the number of dots. */
  people: number
  /** How many share the busiest single answer pattern. */
  busiest_spot: number
  /**
   * How often this flat picture's own nearest-centre rule reproduces the real assignment, which
   * was made using every question. Surfaced so hovering a dot never implies more precision than
   * the projection has.
   */
  faithful: number
}

export type ChartSpec = SegmentMapSpec

/**
 * A spec is only usable if it is the version this build understands. Anything else falls back to
 * the static drawing — a saved project from an older or newer build must render as a correct
 * picture, never as a broken one.
 */
export function usableSpec(spec: unknown): SegmentMapSpec | null {
  if (!spec || typeof spec !== 'object') return null
  const candidate = spec as Partial<SegmentMapSpec>
  if (candidate.version !== SUPPORTED_SPEC_VERSION) return null
  if (candidate.kind !== 'segment_map') return null
  const points = candidate.points
  if (!points || !Array.isArray(points.x)) return null
  // Parallel arrays only mean anything if they are the same length; a truncated one would silently
  // drop respondents off the chart, which is the one thing this chart exists not to do.
  const n = points.x.length
  if (points.y?.length !== n || points.segment?.length !== n || points.people?.length !== n) {
    return null
  }
  if (!Array.isArray(candidate.segments) || !candidate.segments.length) return null
  return candidate as SegmentMapSpec
}
