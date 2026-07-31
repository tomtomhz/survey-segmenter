/**
 * Catch the obvious problems before uploading, so the answer is instant instead of a round trip
 * through a minute of clustering. The server re-checks everything — this is courtesy, not
 * security.
 */
const MAX_BYTES = 100 * 1024 * 1024
const READABLE = /\.(csv|tsv|txt|xlsx|xlsm|xls)$/

export function fileProblem(file: File | null | undefined): string | null {
  if (!file) return 'Please choose a file.'
  if (file.size === 0) {
    return 'That file is empty. Export your survey again and try once more.'
  }
  if (file.size > MAX_BYTES) {
    return 'That file is bigger than 100 MB. Please export a smaller one — one row per person is all I need.'
  }
  const name = (file.name || '').toLowerCase()
  if (name && !READABLE.test(name)) {
    return `I can read .csv and .xlsx survey exports. "${file.name}" does not look like one.`
  }
  return null
}

/**
 * A dropped folder arrives as a DataTransferItem with no real file behind it. Uploading it would
 * send zero bytes and look like a mysterious failure, so it is worth naming.
 */
export function droppedADirectory(transfer: DataTransfer | null): boolean {
  const item = transfer?.items?.[0]
  if (!item || typeof item.webkitGetAsEntry !== 'function') return false
  return item.webkitGetAsEntry()?.isDirectory === true
}
