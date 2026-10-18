"""Route each request to a worker model, using laya to make the call.

laya answers typed questions by reading probabilities off a single forward
pass rather than generating text, so "which worker should serve this?" comes
back as one of *your* option names with a calibrated probability attached, in
roughly 70-500ms. That is the property that makes it usable on the hot path:
the answer cannot go off-schema, so there is no parse step to fail.

The decision logic here is deliberately free of any LiteLLM import, so it can
be exercised without the proxy installed -- see test_decider.py. LiteLLM is
imported lazily at the bottom, where the hook is defined.

Serve laya first (see README). It exposes /v1/systemone directly over HTTP on
LAYA_PORT (default 8000), so there is no shim in front of it:

    LAYA_PORT=8000 LAYA_DEVICE=cpu python laya_runner.py
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx

log = logging.getLogger(__name__)

LAYA_BASE_URL = os.environ.get("LAYA_BASE_URL", "http://localhost:8000")
DEFAULT_MODEL = os.environ.get("LAYA_DEFAULT_MODEL", "minicpm5-local")

# The worker that answers decision-shaped requests itself (the grammar-
# constrained local SLM), selected when laya's is_decision probe fires.
DECISION_WORKER = os.environ.get("LAYA_DECISION_WORKER", "decision")

# Probability mass on `true` before the is_decision probe is believed. laya
# reports a noul answer as P(true), so this is the natural 0.5 gate.
DECISION_PROB = float(os.environ.get("LAYA_DECISION_PROB", "0.5"))

# The local llama.cpp checkpoint the decision worker prompts for structured
# decisions, and how long that generation may take.
DECISION_SLM_MODEL = os.environ.get(
    "LAYA_DECISION_SLM_MODEL", "MiniCPM5-2B-Q4_K_M.gguf"
)
DECISION_SLM_TIMEOUT_S = float(os.environ.get("LAYA_DECISION_SLM_TIMEOUT_S", "30"))

# Opt-in: re-score the SLM's decision with laya so the response carries
# calibrated probabilities over the options. Off by default because it adds a
# second hop to every decision; a per-request `decision_score` param overrides
# this default. The scorer never changes which answer the SLM chose -- it only
# attaches (or withholds) a calibrated confidence to that answer.
DECISION_SCORE = os.environ.get("LAYA_DECISION_SCORE", "").strip().lower() in (
    "1", "true", "yes", "on",
)

# Minimum probability mass on the winning option before the decision is used.
# This is the winner's own probability, not laya's `confidence` -- see
# choose() for why the two are not interchangeable.
MIN_PROB = float(os.environ.get("LAYA_MIN_PROB", "0.5"))

# A router that can add ten seconds to a request is worse than no router. laya
# answers in well under a second, so anything past this is a fault and
# should be treated as one rather than waited out.
TIMEOUT_S = float(os.environ.get("LAYA_TIMEOUT_S", "1.0"))

# A health probe is a local round trip and must not eat the decision budget.
PROBE_TIMEOUT_S = float(os.environ.get("LAYA_PROBE_TIMEOUT_S", "0.3"))

# Characters of conversation to route on. Routing needs the shape of the
# request, not the whole transcript, and laya charges for what it reads.
CONTEXT_CHARS = int(os.environ.get("LAYA_CONTEXT_CHARS", "4000"))

# How long a health probe is trusted. Rented GPUs run a few hours a day, so
# which workers exist changes during the day rather than at deploy time.
HEALTH_TTL_S = float(os.environ.get("LAYA_HEALTH_TTL_S", "30"))

# The option names ARE the worker model names in LiteLLM's model_list, so the
# answer is used directly and there is no label mapping to get wrong.
#
# The descriptions are not decoration: for a choice question laya reads the
# criteria to understand what each option means. Passing bare model names with
# no description asks it to route on the strings "gpt-4o" and "gpt-4o-mini".
ROUTES = {
    # The cheapest possible outcome: the request *is* a decision, so laya
    # answers it and no LLM is called at all. Classification, yes/no, scoring,
    # ranking, guardrail checks, picking one option from a fixed set. Routing
    # is only one instance of this shape -- the model is a general typed-answer
    # engine, and every request served here costs a forward pass instead of a
    # per-token bill.
    "laya-direct": (
        "The request is itself a decision with a fixed answer space: classify "
        "this, is this X yes or no, score this, pick one of these options, does "
        "this violate a policy. Anything answerable by choosing a label rather "
        "than by writing prose."
    ),
    "minicpm5-local": (
        "The smallest local worker: simple requests that must not leave the "
        "building, or that a 2B model can finish -- short factual answers, "
        "reformatting, extraction, boilerplate. Also anything containing "
        "personal or customer data, credentials, or internal source."
    ),
    "bonsai-local": (
        "Local work a step beyond the smallest model: moderate reformatting, "
        "summarising an internal document, simple code, or extraction that "
        "needs a little more capacity, still with data that must not leave the "
        "building."
    ),
    "ornith-local": (
        "Substantial work that must still stay self-hosted: longer code, "
        "multi-file reasoning, or analysis over internal documents that cannot "
        "be sent to a third party."
    ),
    "runpod": (
        "Substantial work that may use the rented GPU: longer code, multi-file "
        "reasoning, or analysis too large for the laptop but not appropriate "
        "for a hosted API."
    ),
    "deepseek-flash": (
        "Ordinary requests with no privacy constraint: summarising, short "
        "answers, chat, straightforward code, drafting."
    ),
    "deepseek-v4-pro": (
        "The hardest requests: deep multi-step reasoning, large refactors, "
        "long-horizon agentic work, problems where a wrong answer is expensive."
    ),
}


# Hosted routes, plus the decision model itself, which is up whenever this hook
# is running. LiteLLM owns retries and failover for these.
ALWAYS_UP = ("laya-direct", "deepseek-flash", "deepseek-v4-pro")

# Self-hosted routes, and where to probe them. A rented GPU is up for part of
# the day; a laptop is up when it is open. All three local checkpoints live on
# the one llama.cpp server, so they share a URL and probe it independently.
LLAMACPP_BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "http://127.0.0.1:9080/v1")
RUNPOD_BASE_URL = os.environ.get(
    "RUNPOD_BASE_URL", "https://xh22rnllkuq2so-8080.proxy.runpod.net/v1"
)

PROBES = {
    "minicpm5-local": LLAMACPP_BASE_URL,
    "bonsai-local": LLAMACPP_BASE_URL,
    "ornith-local": LLAMACPP_BASE_URL,
    "runpod": RUNPOD_BASE_URL,
}

_health = {}  # route -> (checked_at, is_up)


def _probe_headers(route):
    """Auth for a probe, if the worker needs it.

    RunPod's proxy rejects even /models without the key, so the probe carries
    the same bearer token dispatch will use. The local llama.cpp server does
    not ask for one.
    """
    if route != "runpod":
        return {}
    api_key = os.environ.get("RUNPOD_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


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
            f"{base_url.rstrip('/')}/models",
            timeout=PROBE_TIMEOUT_S,
            headers=_probe_headers(route),
        )
        is_up = response.status_code < 400
    except httpx.HTTPError:
        is_up = False

    _health[route] = (now, is_up)
    if is_up != was_up:
        log.info("laya: %s is now %s", route, "up" if is_up else "down")

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

    Two questions, answered in a single forward pass: `route` (which worker)
    and `is_decision` (whether this request is itself a decision the decision
    worker should answer directly). The probe costs no extra round trip because
    laya reads every question in the payload at once.

    `routes` is the currently-serving subset -- see available_routes().

    laya never auto-selects a typed-decisions task, so the decision task the
    caller wants routed is carried in the body too. It is read at call-time
    (like LAYA_API_KEY) so it can be monkeypatched in tests: leave it unset to
    send nothing and let the caller's own default stand, though "typed-decisions"
    is the sensible default.
    """
    payload = {
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
            },
            "is_decision": {
                "type": "noul",
                "instructions": (
                    "Is this request itself a decision with a fixed answer "
                    "space -- classify, yes/no, score, or pick one of a set "
                    "of options -- rather than a request to generate prose?"
                ),
            },
        },
    }

    decision_model = os.environ.get("LAYA_DECISION_MODEL", "typed-decisions")
    if decision_model:
        payload["model"] = decision_model

    return payload


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
        log.warning("laya: no route answer in response; falling back")
        return None

    choice = answer.get("choice")
    if choice not in (routes or ROUTES):
        log.warning("laya: unknown choice %r; falling back", choice)
        return None

    # probabilities is an object keyed by option name, not an array.
    probability = (answer.get("probabilities") or {}).get(choice, 0.0)
    if probability < MIN_PROB:
        log.info(
            "laya: %s at %.2f is below the %.2f gate; falling back to %s",
            choice, probability, MIN_PROB, DEFAULT_MODEL,
        )
        return None

    return choice


def is_decision(body, threshold=DECISION_PROB):
    """Whether laya's is_decision probe says the request is itself a decision.

    A noul answer carries the probability mass on `true` in a field named
    `noul` -- there is no `choice` or `probabilities` for noul. The request is
    treated as a decision when that mass meets the threshold. Anything missing
    or malformed is treated as prose, which is the safe default: it routes to
    an LLM rather than the decision worker.
    """
    answer = (body or {}).get("answers", {}).get("is_decision")
    if not isinstance(answer, dict):
        return False

    prob_true = answer.get("noul")
    if not isinstance(prob_true, (int, float)):
        log.warning("laya: malformed is_decision answer; treating as prose")
        return False

    return prob_true >= threshold


# The shape the decision SLM must emit. llama.cpp enforces this grammar via
# `response_format: json_schema`, so the SLM cannot go off-schema: there is no
# free-text parse step to fail, only a schema to validate.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["choice", "score", "noul"]},
        "criteria": {"type": ["object", "array", "null"]},
        "answer": {"type": ["string", "number", "boolean", "null"]},
    },
    "required": ["type", "answer"],
    "additionalProperties": False,
}

DECISION_TYPES = ("choice", "score", "noul")


def decision_payload(prompt, model=None):
    """The llama.cpp chat request that extracts a structured decision."""
    return {
        "model": model or DECISION_SLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "decision", "schema": DECISION_SCHEMA},
        },
    }


def parse_decision(text):
    """Validate the SLM's JSON decision, or raise ValueError.

    The grammar guarantees valid JSON of the schema above; this adds the
    semantic checks the schema cannot express:

    - choice: `criteria` is an object, `answer` is one of its keys.
    - score:  `criteria` is an array, `answer` is a number within its range.
    - noul:   `answer` is a boolean (normalised from "true"/"false"/0/1).
    """
    try:
        decision = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"decision worker returned invalid JSON: {error}") from error

    if not isinstance(decision, dict):
        raise ValueError("decision worker returned a non-object")

    dtype = decision.get("type")
    if dtype not in DECISION_TYPES:
        raise ValueError(f"decision worker returned unknown type {dtype!r}")

    answer = decision.get("answer")
    criteria = decision.get("criteria")

    if dtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice decision must carry a non-empty criteria object")
        if answer not in criteria:
            raise ValueError(f"choice answer {answer!r} is not one of the criteria")
    elif dtype == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("score decision must carry a non-empty criteria list")
        if not isinstance(answer, (int, float)):
            raise ValueError("score answer must be a number")
        if not 0 <= answer < len(criteria):
            raise ValueError(f"score answer {answer!r} is out of range")
    else:  # noul
        if isinstance(answer, bool):
            pass
        elif answer in (0, 1):
            decision["answer"] = bool(answer)
        elif isinstance(answer, str) and answer.lower() in ("true", "false"):
            decision["answer"] = answer.lower() == "true"
        else:
            raise ValueError(f"noul answer must be a boolean, got {answer!r}")

    return decision


def format_decision(decision):
    """The assistant message content handed back to the client."""
    return json.dumps(decision)


# One client for the process. A new connection per request would add a TLS or
# TCP handshake to a decision that is supposed to be the cheap part.
_client = httpx.AsyncClient(timeout=TIMEOUT_S)


def _auth_headers():
    """Bearer auth headers, if laya is configured to require them.

    laya refuses the request when LAYA_API_KEY is set and the caller does not
    present it, so the key is forwarded unchanged when it is configured and
    omitted otherwise. Nothing here reads the key beyond sending it back on the
    wire.
    """
    api_key = os.environ.get("LAYA_API_KEY")
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


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
            f"{LAYA_BASE_URL}/v1/systemone",
            json=build_request(messages, routes),
            headers=_auth_headers(),
        )
        response.raise_for_status()
        body = response.json()
        if is_decision(body):
            return DECISION_WORKER
        return choose(body, routes) or DEFAULT_MODEL
    except httpx.HTTPError as error:
        log.warning("laya: %s; falling back to %s", error, DEFAULT_MODEL)
        return DEFAULT_MODEL
    except (ValueError, KeyError, TypeError) as error:
        log.warning("laya: malformed response (%s); falling back", error)
        return DEFAULT_MODEL


try:
    import litellm as _litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.utils import Choices, Message, ModelResponse
except ImportError:  # pragma: no cover - lets the logic above be tested alone
    _litellm = None
    CustomLogger = object
    CustomLLM = object
    Choices = Message = ModelResponse = None


class LayaRouter(CustomLogger):
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
        if data.get("model") != os.environ.get("LAYA_VIRTUAL_MODEL", "laya-router"):
            return data

        data["model"] = await decide(data.get("messages", []))
        return data


proxy_handler_instance = LayaRouter()


# --- decision worker -------------------------------------------------------

# One blocking client for the SLM call. The proxy is async, so acompletion()
# runs this in a worker thread; completion() uses it directly.
_slm_client = httpx.Client(timeout=DECISION_SLM_TIMEOUT_S)

# A blocking client for the scorer hop, so completion() can re-score without
# touching the event loop. acompletion() scores through `_client` instead.
_laya_sync_client = httpx.Client(timeout=TIMEOUT_S)


def _extract_decision(messages):
    """Ask the local SLM for a decision; returns the parsed decision dict.

    Raises httpx.HTTPError or ValueError on failure -- the decision worker fails
    loudly rather than returning a wrong answer.
    """
    prompt = context_from(messages)
    response = _slm_client.post(
        f"{LLAMACPP_BASE_URL.rstrip('/')}/chat/completions",
        json=decision_payload(prompt),
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return parse_decision(content)


def decision_score_request(messages, decision):
    """The /v1/systemone payload that re-scores an extracted decision.

    Re-asks the SLM's typed question so laya can attach calibrated
    probabilities to the answer space. The SLM's own `answer` is withheld so
    the calibrated distribution is independent of the SLM's choice.
    """
    question = {
        "type": decision["type"],
        "instructions": "Answer this decision based on the request.",
    }
    # Only choice and score carry an answer space worth re-scoring; a noul's
    # space is the fixed {true, false} pair, and its criteria is not part of
    # the extracted decision.
    if decision["type"] in ("choice", "score"):
        criteria = decision.get("criteria")
        if criteria is not None:
            question["criteria"] = criteria

    payload = {
        "state": context_from(messages),
        "questions": {"decision": question},
    }
    decision_model = os.environ.get("LAYA_DECISION_MODEL", "typed-decisions")
    if decision_model:
        payload["model"] = decision_model
    return payload


def _degraded(decision, note):
    """The decision, flagged as not confidently calibrated."""
    result = dict(decision)
    result["calibrated"] = False
    result["note"] = note
    return result


def _apply_score(decision, laya_answer):
    """Enrich a decision with laya's calibrated probabilities, or degrade.

    The gate is the winning option's own probability -- the SLM's answer, the
    same quantity choose() gates on -- not laya's `confidence`.
    """
    dtype = decision["type"]
    probabilities = laya_answer.get("probabilities") or {}

    if dtype == "choice":
        winner = probabilities.get(decision["answer"], 0.0)
        calibrated = probabilities
    elif dtype == "score":
        winner = probabilities.get(str(decision["answer"]), 0.0)
        calibrated = probabilities
    else:  # noul
        noul = laya_answer.get("noul")
        if not isinstance(noul, (int, float)):
            return _degraded(decision, "laya returned no noul probability")
        winner = noul if decision["answer"] else 1.0 - noul
        calibrated = {"true": noul, "false": round(1.0 - noul, 4)}

    result = dict(decision)
    result["calibrated_probabilities"] = calibrated
    if winner < MIN_PROB:
        result["calibrated"] = False
        result["note"] = (
            f"calibrated probability {winner:.2f} on {decision['answer']!r} "
            f"is below the {MIN_PROB} gate"
        )
        return result
    result["calibrated"] = True
    return result


def _score_with_laya_sync(messages, decision):
    """The sync scorer hop. Degrades explicitly, never raises.

    The scorer is a convenience on top of an already-valid SLM answer, so a
    failed score must not take the decision path down: it returns the SLM's
    answer flagged `calibrated: false` with the reason.
    """
    try:
        response = _laya_sync_client.post(
            f"{LAYA_BASE_URL}/v1/systemone",
            json=decision_score_request(messages, decision),
            headers=_auth_headers(),
        )
        response.raise_for_status()
        answer = response.json().get("answers", {}).get("decision")
        if not isinstance(answer, dict):
            return _degraded(decision, "laya returned no decision score")
        return _apply_score(decision, answer)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        log.warning("decision scorer: %s; returning SLM answer", error)
        return _degraded(decision, f"scorer unavailable: {error}")


async def _score_with_laya(messages, decision):
    """The async scorer hop. Degrades explicitly, never raises."""
    try:
        response = await _client.post(
            f"{LAYA_BASE_URL}/v1/systemone",
            json=decision_score_request(messages, decision),
            headers=_auth_headers(),
        )
        response.raise_for_status()
        answer = response.json().get("answers", {}).get("decision")
        if not isinstance(answer, dict):
            return _degraded(decision, "laya returned no decision score")
        return _apply_score(decision, answer)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        log.warning("decision scorer: %s; returning SLM answer", error)
        return _degraded(decision, f"scorer unavailable: {error}")


def _decide_with_slm_sync(messages, score=False):
    """Answer a decision prompt; returns assistant text.

    With `score`, the extracted decision is re-scored by laya so the response
    carries calibrated probabilities. The SLM call still fails loudly; only
    the optional scorer hop degrades.
    """
    decision = _extract_decision(messages)
    if score:
        decision = _score_with_laya_sync(messages, decision)
    return format_decision(decision)


async def _decide_with_slm(messages, score=False):
    """Async wrapper so acompletion() does not block the event loop."""
    import asyncio

    decision = await asyncio.to_thread(_extract_decision, messages)
    if score:
        decision = await _score_with_laya(messages, decision)
    return format_decision(decision)


def _score_enabled(optional_params):
    """Whether the scorer hop is on: per-request flag, else DECISION_SCORE."""
    flag = (optional_params or {}).get("decision_score")
    if flag is None:
        return DECISION_SCORE
    return bool(flag)


def _model_response(model_response, content):
    """Fill a LiteLLM ModelResponse with one assistant message."""
    if ModelResponse is None:  # pragma: no cover - litellm missing
        raise RuntimeError("litellm is required to serve the decision worker")
    model_response.choices = [
        Choices(message=Message(content=content, role="assistant"), finish_reason="stop")
    ]
    return model_response


class DecisionWorker(CustomLLM):
    """A LiteLLM CustomLLM that answers decision prompts with the local SLM.

    Registered under the provider name "decision", so model_list references it
    as `model: decision/decision`. decision_payload() constrains the SLM to a
    fixed JSON shape, and parse_decision() validates it, so the worker never
    hands a malformed decision back to the caller.

    With `decision_score` in the request (or DECISION_SCORE set), the extracted
    decision is re-scored by laya so the response carries calibrated
    probabilities; below the MIN_PROB gate the SLM's answer is returned with an
    explicit `calibrated: false` note.
    """

    def completion(
        self,
        model,
        messages,
        api_base,
        custom_prompt_dict,
        model_response,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ):
        score = _score_enabled(optional_params)
        content = _decide_with_slm_sync(messages, score=score)
        return _model_response(model_response, content)

    async def acompletion(
        self,
        model,
        messages,
        api_base,
        custom_prompt_dict,
        model_response,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ):
        score = _score_enabled(optional_params)
        content = await _decide_with_slm(messages, score=score)
        return _model_response(model_response, content)


_decision_worker_instance = None
if _litellm is not None:
    _decision_worker_instance = DecisionWorker()
    _litellm.custom_provider_map.append(
        {"provider": "decision", "custom_handler": _decision_worker_instance}
    )
