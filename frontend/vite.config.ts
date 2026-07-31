import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

/**
 * The build lands in ../webui, which the Python app serves and PyInstaller bundles into the
 * packaged .app. Committing that output is deliberate: someone who clones the repo and runs
 * `python3 run_app.py` gets a working interface without Node installed, and the packaged app
 * has no build step at all. `npm run build` is the only thing that should ever write there.
 *
 * In development, `npm run dev` serves the UI with hot reload and forwards every API call to the
 * Python process on 8000, so both halves can be worked on at once.
 */
const API_ROUTES = [
  '/analyze', '/chat', '/score', '/regroup', '/name',
  '/settings', '/projects', '/project', '/delete_project', '/download', '/quit',
]

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../webui',
    emptyOutDir: true,
    assetsDir: 'assets',
    // No sourcemaps in the shipped build. They were 923 kB — four times the app — inside a
    // directory that is committed to git and bundled into the .app, and the actual source is
    // one folder away in frontend/. `npm run dev` has full sourcemaps where debugging happens.
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_ROUTES.map((route) => [route, { target: 'http://127.0.0.1:8000', changeOrigin: false }]),
    ),
  },
  test: {
    // happy-dom rather than jsdom: jsdom 30 hangs on import in this toolchain — not slow, never
    // resolves — which surfaces as vitest timing out waiting for a worker that never speaks, with
    // no error of its own. happy-dom starts the DOM in ~170ms and runs these tests identically.
    environment: 'happy-dom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    coverage: { provider: 'v8', include: ['src/**/*.{ts,tsx}'], exclude: ['src/test/**'] },
  },
})
