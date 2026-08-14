/**
 * The shapes the Python app sends back.
 *
 * These mirror `_analysis_payload` and the `/project`, `/settings`, `/projects`, `/score` and
 * `/name` handlers in segment_kmeans.py. They are hand-written rather than generated because
 * there is one server, it is in this repository, and a code generator would be more machinery
 * than the contract is worth — but they are the contract, and a change on the Python side that
 * is not reflected here should fail the type check on the next build.
 */

/** How much the tool thinks its own answer can be trusted. Drives colour throughout the UI. */
export type Confidence = 'high' | 'moderate' | 'low' | 'unknown'

/** Why a question was or was not used to group people on. */
export type ColumnRole = 'used' | 'background' | 'choice' | 'rating' | 'skipped'

/** Which failure it was, so the UI can offer Settings when a key is the actual problem. */
export type ErrorKind = 'nokey' | 'nosdk' | 'auth' | string

/** One code found inside a wide best-worst export, and how often it appears. */
export interface WideCode {
  code: number
  times: number
}

export interface Failure {
  ok: false
  error: string
  kind?: ErrorKind
  /** A best-worst export saved one row per person: readable, but only once someone says which
   *  code means "best". The server sends the codes actually present so the page can ask rather
   *  than leaving the reader to pivot the file by hand. */
  needs_polarity?: boolean
  codes?: WideCode[]
}

export type ChartId = 'map' | 'fit' | 'k' | 'profiles' | 'heatmap' | 'gorge'

export interface Chart {
  id: ChartId
  title: string
  /** Inline SVG built by the Python chart engine. Trusted: it never contains user input. */
  svg: string
  /** A sentence of interpretation, containing safe inline markup (<strong>, <em>). */
  caption: string
  /**
   * The numbers behind the picture, when the chart has an interactive renderer and is small
   * enough for one to stay responsive. Typed as unknown here and narrowed by `usableSpec`, so a
   * spec from a different build version can never be read as though it were this one.
   */
  spec?: unknown
}

/** A finished analysis — returned by /analyze, /regroup and /project alike. */
export interface Analysis {
  ok: true
  session_id: string
  title: string
  /** The full statistical report. Server-rendered and server-escaped. */
  report_html: string
  ai_available: boolean
  downloads: string[]
  k: number
  n_people: number
  columns: Record<string, ColumnRole>
  charts: Chart[]
  confidence: Confidence
  names?: string[]
  /**
   * Best-worst (MaxDiff) studies only, and absent for every ordinary survey. This is the answer
   * such a study was fielded to produce, so it is sent as data rather than left only as prose
   * inside `report_html` — that panel is collapsed by default and sits below the charts.
   */
  ranking?: RankedItem[] | null
}

/** One row of the overall preference ranking, strongest first. */
export interface RankedItem {
  rank: number
  item: string
  /** Relative preference, centred so the average item is zero. Only differences mean anything. */
  score: number
  low: number | null
  high: number | null
  /**
   * Probability that this item really does beat the one below it, from the joint posterior — so
   * it accounts for the fact that these scores are centred and therefore correlated. `null` on the
   * last row, which has nothing below it to beat.
   *
   * Shown rather than reduced to a verdict: 0.58 and 0.93 are very different findings, and the
   * first version of this card gave both the same three words.
   */
  prob_ahead?: number | null
  /** Whether `prob_ahead` clears 95%. `null` on the last row. */
  clear_of_next: boolean | null
}

/** /project additionally replays the conversation that happened about the analysis. */
export interface TranscriptEntry {
  role: 'you' | 'ai'
  text?: string
  html?: string
}

export interface Project extends Analysis {
  transcript: TranscriptEntry[]
  reopened: true
}

export interface ProjectSummary {
  id: string
  title: string | null
  k: number | null
  n_people: number | null
  confidence: Confidence | null
  updated: string | null
  /** Kept at the top of the list, and never cut off by the cap. Optional so an older server
   *  still parses; absent reads as not pinned. */
  pinned?: boolean
}

export interface ProjectList {
  ok: true
  projects: ProjectSummary[]
  /** How many projects EXIST, which is not always how many are listed — the list is capped.
   *  Optional so an older server still parses; without it no "showing x of y" line is shown. */
  total?: number
}

/**
 * GET /settings answers with a bare status and no `ok` at all; POST /settings adds `ok: true`
 * on success. Typing it as `ok?: true` — never `boolean` — is what lets `status.ok === false`
 * narrow the union to `Failure`, so the modal can tell "the app is unreachable" apart from
 * "the AI add-on is not installed", which need different advice.
 */
export interface AiStatus {
  sdk_installed: boolean
  configured: boolean
  source: string | null
  env_key: boolean
  model: string | null
  ok?: true
}

export interface ScoreResult {
  ok: true
  n: number
  breakdown: Record<string, number>
  mean_confidence: number | string
  /** The value that means "no better than guessing" — 1/k, so it moves with the group count.
   *  Optional so an older server still parses; without it the figure is shown bare. */
  confidence_floor?: number
  /** How many of the scored people answered something outside the scale the study used —
   *  typically a "no answer" code such as 99. Optional so an older server still parses. */
  off_scale?: number
  file: string
}

export interface ChatReply {
  ok: true
  reply_html: string
}

export interface NamesResult {
  ok: true
  names: string[]
  /** The refreshed file list — naming the groups creates group_names.csv and rewrites the
      assignments. The server has always sent this; ignoring it left the download links stale. */
  downloads: string[]
}

/** Every endpoint either succeeds with its own shape or fails with `Failure`. */
export type Result<T> = T | Failure

/**
 * `ok?: true` rather than `ok: true` so this also accepts an `AiStatus`, which GET /settings
 * returns with no `ok` field at all. Anything that is not explicitly `ok: true` is a failure.
 */
export function isFailure<T extends { ok?: true }>(r: T | Failure): r is Failure {
  return r.ok !== true
}

/** One cell of the study planner's sweep: a distinctness regime at one sample size. */
export interface PlanCell {
  regime: string
  separation: number
  n_people: number
  runs: number
  right_k: number
  hit_rate: number
  mean_ari: number
  confidently_wrong: number
  /** True when the design cannot fit the answer scale at all, so no sample size would help. */
  impossible?: boolean
}

export interface PlanResult {
  ok: true
  cells: PlanCell[]
  sizes: number[]
  seeds: number
  questions: number
  segments: number
  regimes: { name: string; separation: number }[]
  /** Smallest sample that reliably found the right number of segments, or null if none did. */
  recommended_n: number | null
  subtle_reachable: boolean
  prose: string
}

/** What a generated questionnaire achieved, reported rather than assumed. */
export interface DesignReport {
  n_respondents: number
  n_items: number
  sets_per_respondent: number
  items_per_set: number
  times_each_item_shown: [number, number]
  exposures_per_respondent: number
  pair_appearances: [number, number]
  never_paired: number
  evenly_divisible: boolean
}

export interface DesignResult {
  ok: true
  items: string[]
  report: DesignReport
  prose: string
  /** The design itself, one row per item shown. Carried in the reply — there is no session yet. */
  csv: string
}
