import type { Analysis, Chart } from '../api/types'

export function chart(id: Chart['id'], title = 'A chart'): Chart {
  return { id, title, svg: `<svg class="chart" data-id="${id}"></svg>`, caption: 'What it shows.' }
}

/** A finished analysis, close enough to a real one that components behave as they would. */
export function analysis(overrides: Partial<Analysis> = {}): Analysis {
  return {
    ok: true,
    session_id: 'sess-1',
    title: 'community_survey.csv',
    report_html: '<h2>Report</h2><p>Three groups.</p>',
    ai_available: false,
    downloads: ['segment_assignments.csv', 'group_profiles.csv'],
    k: 3,
    n_people: 240,
    columns: { q1: 'used', q2: 'used', age: 'background' },
    charts: [chart('map', 'Segment map'), chart('heatmap', 'Group shapes')],
    confidence: 'high',
    ...overrides,
  }
}

/** A File the browser will report a real size for, so upload checks can be exercised. */
export function file(name: string, bytes = 64): File {
  return new File([new Uint8Array(bytes)], name, { type: 'text/csv' })
}
