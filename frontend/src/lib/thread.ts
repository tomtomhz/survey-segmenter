/**
 * What the conversation is made of.
 *
 * The old interface appended DOM nodes as things happened, which meant the thread's contents
 * existed only as markup — there was no way to ask what was in it, and replaying a saved project
 * had to reconstruct the same nodes by a different path. Here the thread is a list of values and
 * rendering is a function of it.
 */
import type { Analysis, WideCode } from '../api/types'

export type Message =
  /** Something the user said or did. */
  | { id: number; kind: 'you'; text: string }
  /** Prose from Claude, or a server-rendered note. HTML comes from our own server. */
  | { id: number; kind: 'ai'; html: string }
  /** Plain text from the app itself — greetings, refusals, explanations. */
  | { id: number; kind: 'note'; html: string; tone?: 'error' }
  /** Claude is working; replaced in place when the reply lands. */
  | { id: number; kind: 'thinking'; label: string }
  /** A finished analysis. */
  | { id: number; kind: 'result'; result: Analysis }
  /** The starter questions offered after the first interpretation. */
  | { id: number; kind: 'suggestions' }
  /** A wide best-worst export needs one fact before it can be read: which code means "best".
   *  Its own kind rather than a note, because a note is rendered from HTML and this needs real
   *  buttons — and because the file has to travel with the question to be re-sent with the answer. */
  | { id: number; kind: 'polarity'; codes: WideCode[]; file: File; note: string }

let nextId = 0
export function messageId(): number {
  nextId += 1
  return nextId
}

/** Replace one message in place — how a "thinking…" placeholder becomes its answer. */
export function replace(messages: Message[], id: number, replacement: Message): Message[] {
  return messages.map((message) => (message.id === id ? replacement : message))
}

export function withoutSuggestions(messages: Message[]): Message[] {
  return messages.filter((message) => message.kind !== 'suggestions')
}
