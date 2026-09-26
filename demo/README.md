# tell-4b-demo — live classification demo page

One-page Next.js app. 50 preloaded tickets (invoice / PO matching,
reconciliation triage, duplicate detection, tolerance handling, adversarial),
each classified in real time by the server (`/api/classify`) under Tell's
label-only contract.

## Run

```bash
cd demo
npm install
npm run build
npx .                 # serves http://localhost:3000
TELL_PORT=4000 npx .  # custom port
```

Dev mode: `npm run dev`.

## How it works

- `GET /api/tickets` — the 50 preloaded tickets.
- `POST /api/classify` — `{ticket}` → `{label, status, latencyMs, prompt}`.
- `lib/tell.ts` — prompt template + label validation identical in contract to
  `src/tell/{prompt,inference}.py`; the decision engine reads only structured
  evidence (STATE text is never parsed for instructions).
- `lib/tickets.ts` — deterministic seeded generator (50 tickets).

The page streams all 50 through the server with a pace control, live accuracy
/ invalid-rate / latency stats, per-class breakdown, filters, and expandable
prompts per ticket.
