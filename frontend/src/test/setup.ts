import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, vi } from 'vitest'

afterEach(cleanup)

/**
 * No test may touch the network.
 *
 * happy-dom serves documents from `http://localhost:3000`, so any component calling `fetch` with a
 * relative path during a test really did try to open a socket there. Those attempts failed, the
 * tests still passed — the API client is built so a failed request never throws — and every run,
 * local and on CI, printed a wall of ECONNREFUSED that reads like an infrastructure problem and
 * is not.
 *
 * Failing loudly instead is the point. A test that reaches the network is either exercising
 * something it did not mean to, or is one slow lookup away from a red build for no reason. A test
 * that needs a response should say which response, by stubbing `fetch` itself.
 */
beforeEach(() => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    throw new Error(
      `A test tried to fetch ${String(input)}. Stub fetch in the test that needs it — the suite `
      + 'must not depend on anything outside this process.',
    )
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})
