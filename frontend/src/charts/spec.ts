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

export interface HeatmapSpec {
  version: number
  kind: 'heatmap'
  /** Question labels, one per row of the grid. */
  items: string[]
  segments: SegmentKey[]
  /** values[segment][question] — the group's answer, on the original scale. */
  values: number[][]
  /** The same cell measured against that question's own average; what the colour encodes. */
  deviation: number[][]
  /** The largest deviation in either direction, so both arms of the scale are comparable. */
  limit: number
  /** "average" for ratings, "share" on the categorical path. */
  kind_of_value: string
}

export interface FitSpec {
  version: number
  kind: 'fit'
  /** Shared bin edges: one more than the number of bands in each row. */
  edges: number[]
  segments: SegmentKey[]
  rows: { segment: number; counts: number[]; median: number; people: number }[]
  overall_median: number
  /** Respondents left out, non-zero only on the latent-class path's sampled fallback. */
  sampled: number
}

export interface KChoiceSpec {
  version: number
  kind: 'k_choice'
  ks: number[]
  chosen: number
  /** The conventional "this partition reproduces" threshold, drawn on the chart. */
  cutoff: number
  series: { key: string; label: string; lead: boolean; values: (number | null)[] }[]
}

export interface ProfilesSpec {
  version: number
  kind: 'profiles'
  items: string[]
  segments: SegmentKey[]
  /** values[segment][item] — the same array the dots were placed from. */
  values: number[][]
  /** "average" for ratings, "share" on the categorical path. */
  measure: string
  /** Questions left off because they separated the groups least and would not fit legibly. */
  trimmed: number
}

export interface GorgeSpec {
  version: number
  kind: 'gorge'
  edges: number[]
  counts: number[]
  median: number
  people: number
}

export type ChartSpec =
  | SegmentMapSpec | HeatmapSpec | FitSpec | KChoiceSpec | ProfilesSpec | GorgeSpec

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

/** The same gate for the grid: right version, right kind, and rows that line up with columns. */
export function usableHeatmap(spec: unknown): HeatmapSpec | null {
  if (!spec || typeof spec !== 'object') return null
  const candidate = spec as Partial<HeatmapSpec>
  if (candidate.version !== SUPPORTED_SPEC_VERSION) return null
  if (candidate.kind !== 'heatmap') return null
  const { items, segments, values, deviation } = candidate
  if (!Array.isArray(items) || !Array.isArray(segments) || !segments.length) return null
  if (!Array.isArray(values) || values.length !== segments.length) return null
  if (!Array.isArray(deviation) || deviation.length !== segments.length) return null
  // Every row must cover every question, or a cell would silently read as blank rather than as
  // the number it is.
  if (values.some((row) => row.length !== items.length)) return null
  if (deviation.some((row) => row.length !== items.length)) return null
  return candidate as HeatmapSpec
}

/** Bands must line up with the edges that bound them, and every row must cover every band. */
export function usableFit(spec: unknown): FitSpec | null {
  if (!spec || typeof spec !== 'object') return null
  const c = spec as Partial<FitSpec>
  if (c.version !== SUPPORTED_SPEC_VERSION || c.kind !== 'fit') return null
  if (!Array.isArray(c.edges) || c.edges.length < 2) return null
  if (!Array.isArray(c.rows) || !c.rows.length) return null
  if (!Array.isArray(c.segments) || !c.segments.length) return null
  const bands = c.edges.length - 1
  if (c.rows.some((r) => !Array.isArray(r.counts) || r.counts.length !== bands)) return null
  return c as FitSpec
}

/** Every series must have one value per candidate, or a line would be drawn against the wrong k. */
export function usableKChoice(spec: unknown): KChoiceSpec | null {
  if (!spec || typeof spec !== 'object') return null
  const c = spec as Partial<KChoiceSpec>
  if (c.version !== SUPPORTED_SPEC_VERSION || c.kind !== 'k_choice') return null
  if (!Array.isArray(c.ks) || !c.ks.length) return null
  if (!Array.isArray(c.series) || !c.series.length) return null
  if (c.series.some((s) => !Array.isArray(s.values) || s.values.length !== c.ks!.length)) return null
  return c as KChoiceSpec
}

/** Every group must have a value for every question, or a dot lands on the wrong row. */
export function usableProfiles(spec: unknown): ProfilesSpec | null {
  if (!spec || typeof spec !== 'object') return null
  const c = spec as Partial<ProfilesSpec>
  if (c.version !== SUPPORTED_SPEC_VERSION || c.kind !== 'profiles') return null
  if (!Array.isArray(c.items) || !c.items.length) return null
  if (!Array.isArray(c.segments) || !c.segments.length) return null
  if (!Array.isArray(c.values) || c.values.length !== c.segments.length) return null
  if (c.values.some((row) => row.length !== c.items!.length)) return null
  return c as ProfilesSpec
}

/** Counts must line up with the edges bounding them. */
export function usableGorge(spec: unknown): GorgeSpec | null {
  if (!spec || typeof spec !== 'object') return null
  const c = spec as Partial<GorgeSpec>
  if (c.version !== SUPPORTED_SPEC_VERSION || c.kind !== 'gorge') return null
  if (!Array.isArray(c.edges) || !Array.isArray(c.counts)) return null
  if (c.counts.length !== c.edges.length - 1 || !c.counts.length) return null
  return c as GorgeSpec
}
