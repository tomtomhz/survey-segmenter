import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './client'
import { isFailure } from './types'

afterEach(() => vi.unstubAllGlobals())

/**
 * The app is a local process the user can quit from a button in its own interface, so "the
 * server went away mid-request" is an ordinary event, not an exceptional one. Nothing here may
 * escape as a thrown error: a rejected promise would leave the busy state stuck and the composer
 * disabled with no way back.
 */
describe('every failure comes back as a value, never a throw', () => {
  it('turns an unreachable app into a message that says what to do', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const reply = await api.projects()
    expect(isFailure(reply)).toBe(true)
    expect(isFailure(reply) && reply.error).toContain('reopen Survey Segmenter')
  })

  it('survives a reply that is not JSON at all', async () => {
    // A crashed handler answers with an HTML traceback page; JSON.parse would throw on it.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      text: () => Promise.resolve('<html>500 Internal Server Error</html>'),
    }))
    const reply = await api.chat('sess-1', 'why?', false)
    expect(isFailure(reply)).toBe(true)
    expect(isFailure(reply) && reply.error).toContain('something unexpected')
  })

  it('survives a body that never arrives', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      text: () => Promise.reject(new Error('aborted')),
    }))
    expect(isFailure(await api.settings())).toBe(true)
  })

  it('passes a real failure from the server through untouched, kind included', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      text: () => Promise.resolve(JSON.stringify({ ok: false, error: 'No API key.', kind: 'nokey' })),
    }))
    const reply = await api.chat('sess-1', 'hello', true)
    expect(isFailure(reply) && reply.kind).toBe('nokey')
  })
})

describe('request construction', () => {
  it('escapes ids and filenames in download links', () => {
    // A session id is generated, but a filename comes from the server and a `&` in one would
    // otherwise silently truncate the query.
    const url = api.downloadUrl('a b&c', 'my report.csv')
    expect(url).toBe('/download?session_id=a%20b%26c&file=my%20report.csv')
  })

  it('sends the file as multipart form data, not JSON', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ text: () => Promise.resolve('{"ok":true}') })
    vi.stubGlobal('fetch', fetchMock)
    await api.analyze(new File(['a,b\n1,2'], 'survey.csv'))
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/analyze')
    expect(init.body).toBeInstanceOf(FormData)
    expect((init.body as FormData).get('file')).toBeInstanceOf(File)
  })
})
