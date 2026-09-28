"""Route each request to a worker model, using OpenJev to make the call.

OpenJev answers typed questions by reading probabilities off a single forward
pass rather than generating text, so "which worker should serve this?" comes
back as one of *your* option names with a calibrated probability attached, in
roughly 70-500ms. That is the property that makes it usable on the hot path:
the answer cannot go off-schema, so there is no parse step to fail.

The decision logic here is deliberately free of any LiteLLM import, so it can
be exercised without the proxy installed -- see test_decider.py. LiteLLM is
imported lazily at the bottom, where the hook is defined.

Serve OpenJev first (see README):

    vllm serve ./openjev --port 8001
    VLLM=http://localhost:8001/v1 TOKENIZER=./openjev READOUT_T=0.85 \
        python openjev/helper/shim.py --port 3000
"""

from __future__ import annotations

import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

SHIM_URL = os.environ.get("OPENJEV_SHIM_URL", "http://localhost:3000")
DEFAULT_MODEL = os.environ.get("OPENJEV_DEFAULT_MODEL", "haiku")

# Minimum probability mass on the winning option before the decision is used.
# This is the winner's own probability, not OpenJev's `confidence` -- see
# choose() for why the two are not interchangeable.
MIN_PROB = float(os.environ.get("OPENJEV_MIN_PROB", "0.5"))

# A router that can add ten seconds to a request is worse than no router. The
# shim answers in well under a second, so anything past this is a fault and
# should be treated as one rather than waited out.
TIMEOUT_S = float(os.environ.get("OPENJEV_TIMEOUT_S", "1.0"))

# A health probe is a local round trip and must not eat the decision budget.
PROBE_TIMEOUT_S = float(os.environ.get("OPENJEV_PROBE_TIMEOUT_S", "0.3"))

# Characters of conversation to route on. Routing needs the shape of the
# request, not the whole transcript, and the shim charges for what it reads.
CONTEXT_CHARS = int(os.environ.get("OPENJEV_CONTEXT_CHARS", "4000"))

# How long a health probe is trusted. Rented GPUs run a few hours a day, so
# which workers exist changes during the day rather than at deploy time.
HEALTH_TTL_S = float(os.environ.get("OPENJEV_HEALTH_TTL_S", "30"))

# The option names ARE the worker model names in LiteLLM's model_list, so the
# answer is used directly and there is no label mapping to get wrong.
#
# The descriptions are not decoration: for a choice question OpenJev reads the
# criteria to understand what each option means. Passing bare model names with
# no description asks it to route on the strings "gpt-4o" and "gpt-4o-mini".
ROUTES = {
    # The cheapest possible outcome: the request *is* a decision, so OpenJev
    # answers it and no LLM is called at all. Classification, yes/no, scoring,
    # ranking, guardrail checks, picking one option from a fixed set. Routing
    # is only one instance of this shape -- the model is a general typed-answer
    # engine, and every request served here costs a forward pass instead of a
    # per-token bill.
    "openjev-direct": (
        "The request is itself a decision with a fixed answer space: classify "
        "this, is this X yes or no, score this, pick one of these options, does "
        "this violate a policy. Anything answerable by choosing a label rather "
        "than by writing prose."
    ),
    "orinth-local": (
        "Simple requests that must not leave the building, or that a small "
        "local model can finish: short factual answers, reformatting, "
        "extraction, boilerplate. Also anything containing personal or customer "
        "data, credentials, or internal source."
    ),
    "vast-local": (
        "Substantial work that must still stay self-hosted: longer code, "
        "multi-file reasoning, or analysis over internal documents that cannot "
        "be sent to a third party."
    ),
    "haiku": (
        "Ordinary requests with no privacy constraint: summarising, short "
        "answers, chat, straightforward code, drafting."
    ),
    "sonnet": (
        "Harder requests: multi-step reasoning, non-trivial code, longer "
        "analysis, ambiguous instructions needing interpretation."
    ),
    "opus": (
        "The hardest requests only: deep multi-step reasoning, large refactors, "
        "long-horizon agentic work, problems where a wrong answer is expensive."
    ),
}


# Hosted routes, plus the decision model itself, which is up whenever this hook
# is running. LiteLLM owns retries and failover for these.
ALWAYS_UP = ("openjev-direct", "haiku", "sonnet", "opus")

# Self-hosted routes, and where to probe them. A rented GPU is up for part of
# the day; a laptop is up when it is open.
PROBES = {
    "orinth-local": os.environ.get("ORNITH_BASE_URL", "http://localhost:8003/v1"),
    "vast-local": os.environ.get("VAST_BASE_URL", ""),
}

_health = {}  # route -> (checked_at, is_up)


async def available_routes():
    """The routes that can actually serve a request right now.

    Offering a worker that is switched off is not a smaller problem than
    picking the wrong one -- it is a worse one. The decision comes back
    confident and correct, dispatch fails, and the fallback sends the request
    somewhere the decision deliberately avoided. For a privacy route that is
    not a degraded answer, it is the wrong answer: the request leaves the
    building for exactly the reason it was meant to stay.

    So availability is resolved before the question is asked rather than after
    it is answered. A worker that is down is simply not one of the options.
    """
    live = {name: ROUTES[name] for name in ALWAYS_UP if name in ROUTES}

    for name, base_url in PROBES.items():
        if name in ROUTES and await _is_up(name, base_url):
            live[name] = ROUTES[name]

    return live


async def _is_up(route, base_url):
    """Whether a self-hosted worker answers, cached for HEALTH_TTL_S.

    Probed rather than configured. A rented box's schedule lives in whoever
    started it, not in this file, and a hand-maintained "is it up on Tuesdays"
    table is wrong the first time someone changes their mind.
    """
    if not base_url:
        return False

    checked_at, was_up = _health.get(route, (0.0, False))
    now = time.monotonic()
    if now - checked_at < HEALTH_TTL_S:
        return was_up

    try:
        response = await _client.get(
            f"{base_url.rstrip('/')}/models", timeout=PROBE_TIMEOUT_S
        )
        is_up = response.status_code < 400
    except httpx.HTTPError:
        is_up = False

    _health[route] = (now, is_up)
    if is_up != was_up:
        log.info("openjev: %s is now %s", route, "up" if is_up else "down")

    return is_up


def context_from(messages, budget=CONTEXT_CHARS):
    """The tail of the conversation, as plain text, within a character budget.

    The tail rather than the head. A long conversation truncated from the front
    routes on its oldest turns, which is precisely the part that no longer
    describes what is being asked.
    """
    chunks = []
    remaining = budget

    for message in reversed(messages or []):
        if remaining <= 0:
            break

        if message.get("role") not in ("user", "assistant"):
            continue

        text = _text_of(message.get("content"))
        if not text:
            continue

        kept = text[-remaining:]
        chunks.append(kept)
        # The newline joining this chunk to the next one is a character too,
        # and unbudgeted separators are how a budget quietly overruns.
        remaining -= len(kept) + 1

    return "\n".join(reversed(chunks)).strip() or "No user input."


def _text_of(content):
    """Message content, which may be a string or a list of typed parts."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )

    return ""


def build_request(messages, routes=None):
    """The /v1/systemone payload for one routing decision.

    One question, not two. An extra question is an extra distribution to read
    on every request, and a "how complex is this?" score that nothing consumes
    is latency with no consumer.

    `routes` is the currently-serving subset -- see available_routes().
    """
    return {
        "state": context_from(messages),
        "questions": {
            "route": {
                "type": "choice",
                "instructions": (
                    "Which worker should serve this request? Prefer the "
                    "cheapest option that can do the job well, and prefer a "
                    "local worker whenever the request carries data that "
                    "should not leave the building."
                ),
                "criteria": routes or ROUTES,
            }
        },
    }


def choose(body, routes=None):
    """The worker named by a /v1/systemone response, or None to fall back.

    The documented shape for a choice answer is:

        {"answers": {"route": {"type": "choice",
                               "choice": "gpt-4o-mini",
                               "confidence": 0.91,
                               "probabilities": {"gpt-4o": 0.04, ...}}}}

    `choice` is the option name itself, so it is looked up rather than parsed.

    The gate is the winning option's own probability, not `confidence`.
    Confidence is the sharpness of the whole distribution (1 - H(p)/ln K), so a
    three-way split of 0.4/0.35/0.25 and a clean 0.9 on one option can report
    similar-looking numbers for different reasons. "How much mass is on the
    answer I am about to act on" is the question actually being asked here.
    """
    answer = (body or {}).get("answers", {}).get("route")
    if not isinstance(answer, dict):
        log.warning("openjev: no route answer in response; falling back")
        return None

    choice = answer.get("choice")
    if choice not in (routes or ROUTES):
        log.warning("openjev: unknown choice %r; falling back", choice)
        return None

    # probabilities is an object keyed by option name, not an array.
    probability = (answer.get("probabilities") or {}).get(choice, 0.0)
    if probability < MIN_PROB:
        log.info(
            "openjev: %s at %.2f is below the %.2f gate; falling back to %s",
            choice, probability, MIN_PROB, DEFAULT_MODEL,
        )
        return None

    return choice


# One client for the process. A new connection per request would add a TLS or
# TCP handshake to a decision that is supposed to be the cheap part.
_client = httpx.AsyncClient(timeout=TIMEOUT_S)


async def decide(messages):
    """The worker to serve this request. Never raises.

    Fails open: any error, timeout or low-confidence answer routes to
    DEFAULT_MODEL. A router that can take the whole service down when its
    decider is unreachable has made reliability worse, not better -- the
    fallback is a working answer, just not the best-chosen one.
    """
    try:
        routes = await available_routes()
        response = await _client.post(
            f"{SHIM_URL}/v1/systemone", json=build_request(messages, routes)
        )
        response.raise_for_status()
        return choose(response.json(), routes) or DEFAULT_MODEL
    except httpx.HTTPError as error:
        log.warning("openjev: %s; falling back to %s", error, DEFAULT_MODEL)
        return DEFAULT_MODEL
    except (ValueError, KeyError, TypeError) as error:
        log.warning("openjev: malformed response (%s); falling back", error)
        return DEFAULT_MODEL


try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover - lets the logic above be tested alone
    CustomLogger = object


class OpenJevRouter(CustomLogger):
    """Rewrites `model` before dispatch, so LiteLLM serves the chosen worker.

    Registered as a callback, which is the supported way to reach into a
    request from a YAML-configured proxy:

        litellm_settings:
          callbacks: ["decider.proxy_handler_instance"]

    There is no `pre_call_hooks` key inside a model_list entry. An unknown key
    there is not rejected -- it is carried into the provider call and ignored,
    so a hook configured that way never runs and nothing says so.

    The alternative is litellm.router.CustomRoutingStrategyBase, whose
    async_get_available_deployment picks a deployment rather than rewriting a
    name. It is the better fit conceptually, but it is installed in Python
    (router.set_custom_routing_strategy) rather than from the config file.
    """

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if data.get("model") != os.environ.get("OPENJEV_VIRTUAL_MODEL", "jev-router"):
            return data

        data["model"] = await decide(data.get("messages", []))
        return data


proxy_handler_instance = OpenJevRouter()
