# Routing with a small decision model

One OpenAI-compatible endpoint that decides, per request, which worker should
serve it — a strong model, a cheap one, or a local one for anything that must
not leave the building.

```
 client ──► LiteLLM :4000            model: "jev-router"
              │
              ├──► OpenJev :8001  /v1/systemone  ~70–500ms, typed answer
              │                                  (26B-A4B, NVFP4, ~14GB)
              │
              └──► openjev-direct   answered by the decision model, no LLM
                   orinth-local     Ornith-1.5-35B-A3B on the laptop
                   vast-local       the rented GPU, when it is up
                   haiku │ sonnet │ opus
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
and the fallback swallows it. OpenJev avoids that by construction — it reads
the answer off the probability distribution at a single output position instead
of generating text, so the reply is one of *your* option names with a
calibrated probability attached. There is no parse step, so there is no parse
step to fail.

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
construction: `openjev-direct`, where no LLM is called at all, and the local
routes, where "this must not leave the building" is not a point on that scale.

## Which OpenJev

Two different projects share the name, and they differ on the thing that
decides whether you can ship:

| | Weights | Fits 32GB? | Licence |
|---|---|---|---|
| [`openjev/openjev`](https://huggingface.co/openjev/openjev) | 27B, bf16 (~54GB) | no | **CC BY-NC 4.0** — non-commercial |
| [`razorback16/openjev`](https://github.com/razorback16/openjev) | DiffusionGemma 26B-A4B, NVFP4 (~14GB) | **yes** | **Apache-2.0** |

**This starter targets the second.** On an RTX 5090 it is the only one of the
two that fits at all — 32GB of VRAM against 54GB of bf16 weights — NVFP4 is
native on Blackwell rather than emulated, and Apache-2.0 is what lets the
result ship to a customer. Both serve the same `/v1/systemone` API, so the
choice is a URL change.

One thing to check after quantizing: the whole value here is *calibrated*
probabilities, and 4-bit weights move the distribution. Re-fit `MIN_PROB`
against the quantized model rather than carrying a threshold over from an
fp16 evaluation.

## Run

```bash
# 1) The decision model + its /v1/systemone server (razorback16/openjev)
vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
  --served-model-name dgemma --diffusion-config '{"canvas_length": 64}' \
  --max-logprobs 32 --enable-prefix-caching --port 8001

# 2) Your local workers, on their own ports
llama-server -m Ornith-1.5-35B-A3B-Q4_K_M.gguf --port 8003   # the laptop
# and whatever the rented box is serving, at $VAST_BASE_URL

# 3) The gateway
ANTHROPIC_API_KEY=... LITELLM_MASTER_KEY=... \
  PYTHONPATH=. litellm --config LLM_config.yaml --port 4000

# 4) Call it
curl http://localhost:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"jev-router","messages":[{"role":"user","content":"Hi"}]}'
```

`PYTHONPATH=.` is not optional — it is how `decider.proxy_handler_instance`
resolves.

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
| `OPENJEV_SHIM_URL` | `http://localhost:3000` | |
| `OPENJEV_DEFAULT_MODEL` | `haiku` | Where every fallback goes |
| `OPENJEV_MIN_PROB` | `0.5` | Mass on the winning option before it is used |
| `OPENJEV_TIMEOUT_S` | `1.0` | Hard cap; the shim answers in well under it |
| `OPENJEV_CONTEXT_CHARS` | `4000` | Tail of the conversation sent as `state` |
| `OPENJEV_VIRTUAL_MODEL` | `jev-router` | The name that triggers routing |
| `OPENJEV_HEALTH_TTL_S` | `30` | How long a worker health probe is trusted |
| `OPENJEV_PROBE_TIMEOUT_S` | `0.3` | Probe cap; must not eat the decision budget |
| `ORNITH_BASE_URL` | `:8003/v1` | Local laptop worker |
| `VAST_BASE_URL` | unset | Rented GPU; unset means "not a route" |

## Workers that come and go

A rented GPU runs a few hours a day, so which workers exist is a runtime fact,
not a config-file one. Availability is resolved **before** the question is
asked: a worker that does not answer a health probe is simply not one of the
options OpenJev is given.

That ordering matters more than it looks. Offering a switched-off worker is
worse than picking the wrong one — the decision comes back confident and
correct, dispatch fails, and the fallback sends the request somewhere the
decision deliberately avoided. For a privacy route that is not a degraded
answer but the wrong one: the request leaves the building for exactly the
reason it was meant to stay.

Two things deliberately fail open: a decision below `MIN_PROB`, and any error
reaching the shim. A router that takes the service down when its decider is
unreachable has made reliability worse — the fallback still answers, just
without the benefit of a choice.

## What was wrong with the first version

Kept here because the failure is instructive, and because it is the reason the
tests exist.

- **The response parsing matched no real schema, so nothing ever routed.** It
  read `answers.route.choices[0].label` and `.probability`. The documented
  shape is `answers.route.choice` (the option name), `.probabilities` (an
  object keyed by option name) and `.confidence`. Every response raised
  `KeyError` inside a `try/except` that returned the default — so every request
  went to the default worker, nothing logged, and the OpenJev round trip was pure
  added latency. The system looked like it was routing.
- **`pre_call_hooks:` inside a `model_list` entry is not a LiteLLM key.** Hooks
  are registered under `litellm_settings.callbacks`. An unknown key in
  `litellm_params` is passed through rather than rejected, so the hook would
  never have run even after the parsing was fixed.
- **`${OPENAI_API_KEY}` is not interpolated.** LiteLLM reads
  `os.environ/OPENAI_API_KEY`; the shell-style form is sent verbatim as the key.
- **The router's vLLM and the `local-qwen` worker were both on `:8000`.**
- **`criteria` gave every option a `null` description**, asking the model to
  route on the strings `"gpt-4o"` and `"gpt-4o-mini"`. The criteria are how a
  choice question conveys what the options mean.
- **`criteria` for the score question was a dict**, where score takes an
  ordered array of level labels — and the score was computed on every request
  and then never read.
- **A blocking `requests.post` inside an async proxy**, with a 10s timeout, on
  a decision budgeted at half a second. One slow shim would stall the event
  loop for every concurrent request.
- **The conversation was truncated from the front**, so a long chat routed on
  its oldest turns rather than on what was just asked.
