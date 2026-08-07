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

export interface Failure {
  ok: false
  error: string
  kind?: ErrorKind
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
}

export interface ProjectList {
  ok: true
  projects: ProjectSummary[]
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
