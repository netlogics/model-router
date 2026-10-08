The router currently routes each request with **OpenJev**, answered through a
separate shim process (`openjev/helper/shim.py`) that exposes OpenJev's
`/v1/systemone` on `:3000`. We are replacing OpenJev with **laya**, which serves
the identical `/v1/systemone` wire protocol directly over HTTP on `LAYA_PORT`
(default 8000), so the shim can be dropped entirely.

## Why this matters

- `laya` loads ModernBERT and ships the same decision API shape
  (`choice` / `score` / `noul` answers with a keyed `probabilities` object),
  so the parse path in `decider.py` is already correct for it.
- No shim: fewer moving parts, no separate `:3000` process, no per-request shim
  latency.
- The decision model stays self-hosted (Apache-2.0 / ModernBERT) rather than the
  OpenJev vLLM + shim stack.

## Non-negotiables

- The decision model (laya) must never take the service down: any error,
  timeout, or low-confidence answer still **fails open** to the default worker.
- Local privacy routes (`orinth-local`, `vast-local`) must **never** fail over to
  a hosted worker.
- The gate is the winning option's own probability, not the response
  `confidence`.

## Outcome

The whole routing decision runs on laya with the shim gone, environment variables
renamed to `LAYA_*`, the default model is `orinth-local`, and the direct-answer
route is `laya-direct`.
