import { describe, expect, it } from 'vitest'
import { droppedADirectory, fileProblem } from './upload'
import { file } from '../test/fixtures'

/**
 * Catching these before the upload is the difference between an instant answer and a minute of
 * waiting for the server to say the same thing.
 */
describe('what we refuse to upload, and how we say so', () => {
  it('accepts the formats a survey tool actually exports', () => {
    for (const name of ['survey.csv', 'RESPONSES.XLSX', 'export.tsv', 'old.xls', 'macro.xlsm']) {
      expect(fileProblem(file(name)), name).toBeNull()
    }
  })

  it('names the file it cannot read, rather than stating a rule', () => {
    const problem = fileProblem(file('holiday-photo.png'))
    expect(problem).toContain('holiday-photo.png')
    expect(problem).toContain('.csv and .xlsx')
  })

  it('spots an empty export before it wastes a round trip', () => {
    expect(fileProblem(file('survey.csv', 0))).toContain('empty')
  })

  it('refuses a file too big to be one row per person', () => {
    expect(fileProblem(file('huge.csv', 100 * 1024 * 1024 + 1))).toContain('100 MB')
    // The limit itself is fine — only past it.
    expect(fileProblem(file('big.csv', 100 * 1024 * 1024))).toBeNull()
  })

  it('handles no file at all', () => {
    expect(fileProblem(null)).toContain('choose a file')
  })
})

describe('dropped folders', () => {
  /** A folder arrives as an item with a directory entry and no real file behind it. */
  function transfer(entry: { isDirectory: boolean } | null): DataTransfer {
    return { items: [{ webkitGetAsEntry: () => entry }] } as unknown as DataTransfer
  }

  it('recognises a folder, which would otherwise upload zero bytes', () => {
    expect(droppedADirectory(transfer({ isDirectory: true }))).toBe(true)
  })

  it('lets an ordinary file through', () => {
    expect(droppedADirectory(transfer({ isDirectory: false }))).toBe(false)
  })

  it('does not crash on a browser that cannot tell us', () => {
    expect(droppedADirectory(transfer(null))).toBe(false)
    expect(droppedADirectory({ items: [] } as unknown as DataTransfer)).toBe(false)
    expect(droppedADirectory(null)).toBe(false)
  })
})
