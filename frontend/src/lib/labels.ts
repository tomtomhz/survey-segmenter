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
 * What each light is actually worth, from measurement rather than from the word itself.
 *
 * Sixty simulated studies were run through the real pipeline with the true number of groups known
 * in advance. Green found the right number about seven times in ten and averaged 0.71 agreement
 * with the truth; amber, three in ten; red, almost never. Two things held across every one of the
 * sixty and are the useful ones to tell a reader: the tool never reported MORE groups than
 * existed, and it never showed green on the weakest data.
 *
 * "Trust these groups" on its own invites a reader to treat the number as exact, which is the one
 * thing the measurement does not support — when green is wrong, it has merged two real groups.
 * Saying so costs three lines and is the difference between a claim and an honest claim.
 */
export const CONFIDENCE_DETAIL: Record<Confidence, string> = {
  high:
    'Checked against sixty studies where the right answer was known: a green light found the '
    + 'right number of groups about seven times in ten, and never invented groups that were not '
    + 'there. When it is wrong it has merged two real groups into one — so read the number as a '
    + 'floor, not a headcount.',
  moderate:
    'Checked against sixty studies where the right answer was known: an amber light found the '
    + 'right number about three times in ten. The groups usually point in the right direction, '
    + 'but do not build a budget on the exact number or on who sits in which one.',
  low:
    'Checked against sixty studies where the right answer was known: a red light almost never '
    + 'recovered the true grouping. Either the questions do not separate people, or there is no '
    + 'grouping to find. More people will not fix the first and cannot fix the second.',
  unknown:
    'The checks that produce this light did not complete, so there is no evidence here either '
    + 'way. Read the full report below before using the groups.',
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
  // Best-worst studies only. Named for the question they answer rather than for how they were
  // computed: nobody downloading these is looking for the word "utility".
  'item_utilities.csv': 'What matters most, ranked (CSV)',
  'respondent_utilities.csv': 'Each person’s preferences (CSV)',
  'what_to_launch.csv': 'Which few to launch (CSV)',
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
