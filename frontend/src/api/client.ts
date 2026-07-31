/**
 * The only place this app talks to the server.
 *
 * Every call goes through `request`, so a dropped connection, a crashed handler or a non-JSON
 * reply can never surface as a raw exception or leave the page stuck mid-action — it comes back
 * as an ordinary `Failure` the caller renders like any other error. That behaviour is load-bearing:
 * the app is a local process the user can quit from a button, so "the server went away" is a
 * normal thing to happen, not an exceptional one.
 */
import type {
  AiStatus, Analysis, ChatReply, Failure, NamesResult, Project, ProjectList, Result, ScoreResult,
} from './types'

const UNREACHABLE =
  'Could not reach the app. It may have been closed — reopen Survey Segmenter and try again.'
const UNPARSEABLE = 'The app sent back something unexpected. Please try again.'

async function request<T extends { ok: true }>(
  url: string,
  init?: RequestInit,
): Promise<Result<T>> {
  let text: string
  try {
    const response = await fetch(url, init)
    text = await response.text()
  } catch {
    return { ok: false, error: UNREACHABLE }
  }
  try {
    return JSON.parse(text) as Result<T>
  } catch {
    return { ok: false, error: UNPARSEABLE }
  }
}

function postJSON<T extends { ok: true }>(url: string, body: unknown): Promise<Result<T>> {
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
}

function upload<T extends { ok: true }>(url: string, file: File): Promise<Result<T>> {
  const form = new FormData()
  form.append('file', file)
  return request<T>(url, { method: 'POST', body: form })
}

export const api = {
  analyze: (file: File) => upload<Analysis>('/analyze', file),

  score: (sessionId: string, file: File) =>
    upload<ScoreResult>(`/score?session_id=${encodeURIComponent(sessionId)}`, file),

  regroup: (sessionId: string, items: string[]) =>
    postJSON<Analysis>('/regroup', { session_id: sessionId, items }),

  chat: (sessionId: string, message: string, initial: boolean) =>
    postJSON<ChatReply>('/chat', { session_id: sessionId, message, initial }),

  suggestNames: (sessionId: string) =>
    postJSON<NamesResult>('/name', { session_id: sessionId, suggest: true }),

  saveNames: (sessionId: string, names: string[]) =>
    postJSON<NamesResult>('/name', { session_id: sessionId, names }),

  projects: () => request<ProjectList>('/projects'),

  openProject: (id: string) => request<Project>(`/project?id=${encodeURIComponent(id)}`),

  deleteProject: (id: string) => postJSON<ProjectList>('/delete_project', { session_id: id }),

  /**
   * The settings endpoint answers with a status object rather than `{ok:true}`, so a transport
   * failure has to be turned into something the modal can render without pretending the SDK is
   * missing — an unreachable app and an uninstalled add-on need different advice.
   */
  async settings(): Promise<AiStatus | Failure> {
    const r = await request<AiStatus & { ok: true }>('/settings')
    return r as AiStatus | Failure
  },

  async saveKey(apiKey: string): Promise<AiStatus | Failure> {
    const r = await postJSON<AiStatus & { ok: true }>('/settings', { api_key: apiKey })
    return r as AiStatus | Failure
  },

  async clearKey(): Promise<AiStatus | Failure> {
    const r = await postJSON<AiStatus & { ok: true }>('/settings', { clear: true })
    return r as AiStatus | Failure
  },

  downloadUrl: (sessionId: string, file: string) =>
    `/download?session_id=${encodeURIComponent(sessionId)}&file=${encodeURIComponent(file)}`,
}

export type Api = typeof api
