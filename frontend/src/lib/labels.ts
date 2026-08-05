/**
 * Plain-English names for everything the server calls by its internal id.
 *
 * The audience is a marketer with a survey export, not a statistician: "Who is in which group"
 * is the file they want, `segment_assignments.csv` is merely what it is called on disk.
 */
import type { ChartId, ColumnRole, Confidence } from '../api/types'

export const CONFIDENCE_ADVICE: Record<Confidence, string> = {
  high: 'Trust these groups',
  moderate: 'Treat as directional',
  low: 'Do not rely on these',
  unknown: 'Confidence unclear',
}

/**
 * Declared in reading order: are these groups real (map, gorge, fit, k), and only then what is in
 * them (profiles, heatmap). The tab order itself comes from the server's chart list; this map
 * mirrors it so the two cannot drift.
 */
export const CHART_TAB: Record<ChartId, string> = {
  map: 'Segment map',
  gorge: 'Do they separate',
  fit: 'Who belongs',
  k: 'How many groups',
  profiles: 'What differs',
  heatmap: 'Full grid',
}

/**
 * Every file the app can produce needs an entry — the two the user creates through the interface
 * (naming the groups, scoring new people) were once falling through to the raw filename.
 */
export const DOWNLOAD_LABEL: Record<string, string> = {
  'segment_assignments.csv': 'Who is in which group (CSV)',
  'group_profiles.csv': 'What defines each group (CSV)',
  'typing_rule.json': 'Scoring rule (JSON)',
  'group_names.csv': 'Your group names (CSV)',
  'scored_new_people.csv': 'Newly scored people (CSV)',
}

export const COLUMN_ROLE: Record<ColumnRole, string> = {
  used: 'grouped on',
  background: 'background trait',
  choice: 'multiple choice',
  rating: 'rating',
  skipped: 'set aside',
}

export const STARTER_QUESTIONS = [
  'Which segment should we target first?',
  'Draft a landing-page headline for the top segment.',
  'How much should we trust these groups?',
  'What would make the least-clear segment sharper?',
]

/** "3 min ago", falling back to a date once it stops being useful to count. */
export function relativeTime(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return ''
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return ''
  const seconds = (now - then.getTime()) / 1000
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`
  if (seconds < 604800) return `${Math.floor(seconds / 86400)} d ago`
  return then.toLocaleDateString()
}

export function titleCase(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1)
}
