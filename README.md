# Routing with a small decision model

One OpenAI-compatible endpoint that decides, per request, which worker should
serve it — a strong model, a cheap one, or a local one for anything that must
not leave the building.

```
 client ──► LiteLLM :4000            model: "laya-router"
              │
              ├──► laya :8000   /v1/systemone  ~70–500ms, typed answer
              │
              └──► laya-direct      answered by the decision model, no LLM
                   minicpm5-local     MiniCPM5-2B on the laptop
                   bonsai-local       Ternary-Bonsai-2-9B on the laptop
                   ornith-local       Ornith-1.5-35B on the laptop
                   runpod             Qwen3.8-Flash-Next on the rented GPU
                   deepseek-flash │ deepseek-v4-pro
```

## Why a decision model rather than a prompt

A router has to be cheaper than what it routes to, or it is a tax on every
request. Three ways to do it, roughly in order of cost:

| | Cost | Trade |
|---|---|---|
| Keyword or embedding rules | ~free | Brittle; no notion of difficulty |
| **Decision model** (this) | ~70–500ms, one forward pass | Needs a GPU if self-hosted |
| A small LLM prompted to classify | A generation, plus parsing | Can go off-schema; you own the retries |

The third option is what most hand-rolled routers do, and it is the one with
the failure mode that bites: the classifier answers in prose, the parse fails,
and the fallback swallows it. **laya** avoids that by construction — it reads
the answer off the probability distribution at a single forward pass instead of
generating text, so the reply is one of *your* option names with a calibrated
probability attached. There is no parse step, so there is no parse step to fail.

That property is worth more than the accuracy numbers. A router that is wrong
10% of the time costs you some quality; a router that *silently stops routing*
costs you the whole feature, and looks identical from outside.

## Is LiteLLM the right thing to build on

Yes, with one caveat.

**For:** it is the piece you would otherwise write by hand — one OpenAI-shaped
endpoint over many providers, plus retries, fallbacks, timeouts, cost
attribution, keys and rate limits per caller. Routing is a small decision that
needs all of that scaffolding around it, and none of the scaffolding is
interesting to own. The worker names in the config are the option names the
decider returns, so the decision needs no translation layer.

**Caveat:** LiteLLM shipped **Auto Routing** in v1.94.x, and one of its
classifier backends is JEV. If hosted TypeSafe Jev is acceptable, most of
`decider.py` collapses into a `complexity_router_config` block — see the bottom
of `LLM_config.yaml`. Keep the hook if you need routing on something that is
not complexity, if beta config keys are a problem, or if the decision itself
must stay in your network. Two routes here are outside a complexity ladder by
construction: `laya-direct`, where no LLM is called at all, and the local
routes, where "this must not leave the building" is not a point on that scale.

## Which laya

`convaiinnovations/laya-typed-decisions` is a ModernBERT-large checkpoint
trained for exactly this shape of question — a typed-decisions task over a
fixed answer space, answered from the probability distribution at a single
output position rather than by generating prose. It runs in about 1024 tokens
of context, which is why `decider.py` sends only the tail of the conversation.
It serves the same `/v1/systemone` API directly, so the whole decision is one
HTTP call with no shim in front of it.

One thing to check after the model is loaded: the whole value here is
*calibrated* probabilities, and quantized or fine-tuned weights move the
distribution. Re-fit `MIN_PROB` against your model rather than carrying a
threshold over from an evaluation you did not run.

## Run

```bash
# 1) The decision model + its /v1/systemone server
#    (laya_runner.py is a CPython 3.14 workaround — see below)
LAYA_PORT=8000 LAYA_DEVICE=cpu python laya_runner.py

#    ...or, on a Python where torch.compile actually imports, the plain path:
LAYA_PORT=8000 LAYA_DEVICE=cpu laya-serve

# 2) The local llama.cpp server, one OpenAI-compatible endpoint serving the
#    GGUF checkpoints configured in ~/.config/eca/config.json
llama-server -m MiniCPM5-2B-Q4_K_M.gguf --port 9080
# switch checkpoints as needed; all three share :9080
# llama-server -m Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf --port 9080
# llama-server -m Ornith-1.5-35B-Q4_K_M.gguf --port 9080
# and the RunPod pod at the URL in LLM_config.yaml, keyed by $RUNPOD_API_KEY

# 3) The gateway
DEEPSEEK_API_KEY=... RUNPOD_API_KEY=... LITELLM_MASTER_KEY=... \
  PYTHONPATH=. litellm --config LLM_config.yaml --port 4000

# 4) Call it
curl http://localhost:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"laya-router","messages":[{"role":"user","content":"Hi"}]}'
```

`LAYA_PORT` / `LAYA_DEVICE` configure the decision server; `LAYA_API_KEY`
(authenticated calls) and `LAYA_DECISION_MODEL` (the typed-decisions task) are
optional. `LAYA_DECISION_MODEL` is read at call-time so it can be overridden per
process.

`PYTHONPATH=.` is not optional — it is how `decider.proxy_handler_instance`
resolves.

### Why `laya_runner.py` instead of `laya-serve`?

`laya` loads ModernBERT through `transformers`, and the current `transformers`
decorates ModernBERT's `forward` with `@torch.compile(dynamic=True)`. Torch
refuses to even import that decorator on CPython 3.14+::

    RuntimeError: torch.compile is not supported on Python 3.14+

Compiling is a pure-speed optimisation; the uncompiled graph runs identically,
just more slowly. `laya_runner.py` swaps `torch.compile` for a passthrough
decorator before `laya.serve.main()` runs, so the model imports and serves on
3.14 unchanged. On a Python where `torch.compile` works it simply leaves the
model uncompiled — so the plain `laya-serve` entry point is fine anywhere else.

## Tests

```bash
python -m pytest
```

Offline: no shim, no GPU, no LiteLLM install. They assert the decision against
the published `/v1/systemone` response shape, which is the thing that was wrong
before — see below.

## Tuning

| | Default | |
|---|---|---|
| `LAYA_BASE_URL` | `http://localhost:8000` | Where laya serves `/v1/systemone` |
| `LAYA_DEFAULT_MODEL` | `minicpm5-local` | Where every fallback goes |
| `LAYA_MIN_PROB` | `0.5` | Mass on the winning option before it is used |
| `LAYA_TIMEOUT_S` | `1.0` | Hard cap; laya answers in well under it |
| `LAYA_CONTEXT_CHARS` | `4000` | Tail of the conversation sent as `state` |
| `LAYA_VIRTUAL_MODEL` | `laya-router` | The name that triggers routing |
| `LAYA_HEALTH_TTL_S` | `30` | How long a worker health probe is trusted |
| `LAYA_PROBE_TIMEOUT_S` | `0.3` | Probe cap; must not eat the decision budget |
| `LLAMACPP_BASE_URL` | `:9080/v1` | Local llama.cpp server (all three checkpoints) |
| `RUNPOD_BASE_URL` | the pod proxy | Rented GPU; used by the health probe |

LiteLLM also needs `DEEPSEEK_API_KEY` and `RUNPOD_API_KEY`. `RUNPOD_BASE_URL`
in this table only drives the health probe; the dispatch URL lives in
`LLM_config.yaml`, so keep both in step with the pod.

## Workers that come and go

A rented GPU runs a few hours a day and the laptop may be closed, so which
workers exist is a runtime fact, not a config-file one. Availability is resolved
**before** the question is asked: a worker that does not answer a health probe
is simply not one of the options laya is given.

That ordering matters more than it looks. Offering a switched-off worker is
worse than picking the wrong one — the decision comes back confident and
correct, dispatch fails, and the fallback sends the request somewhere the
decision deliberately avoided. For a privacy route that is not a degraded
answer but the wrong one: the request leaves the building for exactly the reason
it was meant to stay.

Two things deliberately fail open: a decision below `MIN_PROB`, and any error
reaching laya. A router that takes the service down when its decider is
unreachable has made reliability worse, not better — the fallback still answers,
just without the benefit of a choice.

## What was wrong with the first version

Kept here because the failure is instructive, and because it is the reason the
tests exist.

- **The response parsing matched no real schema, so nothing ever routed.** It
  read `answers.route.choices[0].label` and `.probability`. The documented
  shape is `answers.route.choice` (the option name), `.probabilities` (an
  object keyed by option name) and `.confidence`. Every response raised
  `KeyError` inside a `try/except` that returned the default — so every request
  went to the default worker, nothing logged, and the laya call became pure
  added latency. The system looked like it was routing.
- **`pre_call_hooks:` inside a `model_list` entry is not a LiteLLM key.** Hooks
  are registered under `litellm_settings.callbacks`. An unknown key in
  `litellm_params` is passed through rather than rejected, so the hook would
  never have run even after the parsing was fixed.
- **`${OPENAI_API_KEY}` is not interpolated.** LiteLLM reads
  `os.environ/OPENAI_API_KEY`; the shell-style form is sent verbatim as the key.
- **The router's LLM server and the `local-qwen` worker were both on `:8000`.**
- **`criteria` gave every option a `null` description**, asking the model to
  route on the strings `"gpt-4o"` and `"gpt-4o-mini"`. The criteria are how a
  choice question conveys what the options mean.
- **`criteria` for the score question was a dict**, where score takes an
  ordered array of level labels — and the score was computed on every request
  and then never read.
- **A blocking `requests.post` inside an async proxy**, with a 10s timeout, on
  a decision budgeted at half a second. One slow call would stall the event loop
  for every concurrent request.
- **The conversation was truncated from the front**, so a long chat routed on
  its oldest turns rather than on what was just asked.
