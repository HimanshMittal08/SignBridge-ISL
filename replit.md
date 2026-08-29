# SignBridge

SignBridge is a transparent two-way Indian Sign Language communication workspace for Deaf and hearing people.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/signbridge/src/App.tsx` — Phase 1 routes, local conversation state, concept extraction, demo scenarios, and TTS controls
- `artifacts/signbridge/src/index.css` — shared SignBridge theme, responsive layout, accessibility states, and motion preferences
- `lib/api-spec/openapi.yaml` — shared API contract (currently health-only; SignBridge Phase 1 is local-first)

## Architecture decisions

- Phase 1 uses deterministic local data so the UI can be demonstrated without claiming live ML recognition.
- Live device controls are explicitly labeled as unavailable/simulated until the camera and microphone phases are implemented.
- Raw signs, exact transcripts, interpretations, alternatives, and concept sequences are separate fields in conversation state.
- Vocabulary cards use clearly labeled placeholders until verified INCLUDE/ISL video assets are connected.

## Product

- Landing page focused on transparent communication.
- Two-way conversation workspace with demo scenarios, typed replies, concept extraction, interpretation alternatives, browser TTS, and optional diagnostics.
- Searchable/filterable ISL vocabulary reference with confidence metadata and placeholder demonstration states.

## User preferences

No additional preferences recorded.

## Gotchas

- Do not present deterministic demo output as live model recognition.
- Replace placeholder vocabulary demonstrations only with verified ISL/INCLUDE assets.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
