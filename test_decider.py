"""Tests for the routing decision, offline.

The bug these exist to prevent was silent. The first version of decider.py
read `answers.route.choices[0].label`, which is not the documented shape, so
every response raised KeyError inside a try/except and every request fell back
to the default worker. Nothing failed, nothing logged, and the laya call
became pure added latency -- the system looked like it was routing.

So the responses below are written to the published /v1/systemone schema, and
choose() is tested against them directly. No network, no litellm, no server.

    pytest
"""

import os

import pytest

import decider


def choice_response(choice, probabilities, confidence=0.9):
    """A /v1/systemone response, in the documented shape.

    probabilities is an object keyed by option name -- not an array, and not a
    list of {label, probability} pairs.
    """
    return {
        "answers": {
            "route": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        }
    }


def noul_response(prob_true, confidence=0.9):
    """A /v1/systemone noul answer, in the documented shape.

    noul carries P(true) directly -- there is no `choice` or `probabilities`
    key for a noul question.
    """
    return {
        "answers": {
            "is_decision": {
                "type": "noul",
                "noul": prob_true,
                "confidence": confidence,
            }
        }
    }


class TestChoose:
    def test_takes_the_named_option(self):
        body = choice_response(
            "deepseek-v4-pro",
            {"deepseek-v4-pro": 0.82, "deepseek-flash": 0.15, "ornith-local": 0.03},
        )

        assert decider.choose(body) == "deepseek-v4-pro"

    def test_the_option_name_is_the_worker_name(self):
        # Nothing maps a label to a model. The criteria keys are the model
        # names LiteLLM knows, so the answer is usable as it stands.
        for worker in decider.ROUTES:
            body = choice_response(worker, {worker: 1.0})

            assert decider.choose(body) == worker

    def test_falls_back_when_the_winner_is_below_the_gate(self):
        split = {"deepseek-v4-pro": 0.4, "deepseek-flash": 0.35, "ornith-local": 0.25}

        assert decider.choose(choice_response("deepseek-v4-pro", split)) is None

    # Confidence and the winner's probability are different numbers. A high
    # confidence on a distribution whose winner is weak must not get through
    # the gate: it is the mass on the option being acted upon that matters.
    def test_a_confident_response_with_a_weak_winner_still_falls_back(self):
        body = choice_response(
            "deepseek-v4-pro",
            {"deepseek-v4-pro": 0.4, "deepseek-flash": 0.35, "ornith-local": 0.25},
            confidence=0.99,
        )

        assert decider.choose(body) is None

    def test_falls_back_on_an_option_that_is_not_a_worker(self):
        assert decider.choose(choice_response("gemini", {"gemini": 1.0})) is None

    @pytest.mark.parametrize(
        "body",
        [
            {},
            None,
            {"answers": {}},
            {"answers": {"route": None}},
            # The shape the first version of this file expected. If it ever
            # comes back, it must fall back loudly rather than be mistaken for
            # an answer.
            {"answers": {"route": {"choices": [{"label": "deepseek-v4-pro", "probability": 0.9}]}}},
        ],
    )
    def test_falls_back_on_anything_it_does_not_recognise(self, body):
        assert decider.choose(body) is None


class TestContext:
    def test_keeps_the_end_of_a_long_conversation(self):
        messages = [
            {"role": "user", "content": "x" * 5000},
            {"role": "user", "content": "the actual question"},
        ]

        assert decider.context_from(messages).endswith("the actual question")

    def test_stays_within_its_budget(self):
        messages = [{"role": "user", "content": "x" * 100} for _ in range(50)]

        assert len(decider.context_from(messages, budget=500)) <= 500

    def test_reads_typed_content_parts(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]}]

        assert decider.context_from(messages) == "hello"

    def test_ignores_system_messages(self):
        # A system prompt is the same on every request, so it carries no signal
        # about which worker this one needs.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ]

        assert decider.context_from(messages) == "hi"

    def test_says_so_rather_than_sending_an_empty_state(self):
        assert decider.context_from([]) == "No user input."


class TestRequest:
    def test_asks_two_questions(self):
        # One routing question plus the is_decision probe, both answered in a
        # single forward pass. The probe is consumed, so it is not the unused
        # second question the first version threw away.
        assert list(build_questions()) == ["route", "is_decision"]

    def test_is_decision_is_a_noul_probe(self):
        probe = build_questions()["is_decision"]

        assert probe["type"] == "noul"
        assert probe["instructions"]

    def test_gives_every_option_a_description(self):
        # laya reads the criteria to learn what the options mean. Bare model
        # names with null descriptions ask it to route on the strings.
        criteria = build_questions()["route"]["criteria"]

        assert set(criteria) == set(decider.ROUTES)
        assert all(isinstance(text, str) and text for text in criteria.values())

    def test_sends_the_conversation_as_state(self):
        payload = decider.build_request([{"role": "user", "content": "hello"}])

        assert payload["state"] == "hello"

    def test_includes_the_decision_model_by_default(self, monkeypatch):
        # laya never auto-selects a typed-decisions task, so the model must ride
        # in the body. Leave any environment value out of the way to prove the
        # default is what ships.
        monkeypatch.delenv("LAYA_DECISION_MODEL", raising=False)

        payload = decider.build_request([{"role": "user", "content": "hi"}])

        assert payload["model"] == "typed-decisions"

    def test_omits_the_model_when_unset(self, monkeypatch):
        # An empty decision task means "ask the caller's own default": send no
        # model at all rather than a literal empty string.
        monkeypatch.setitem(os.environ, "LAYA_DECISION_MODEL", "")

        payload = decider.build_request([{"role": "user", "content": "hi"}])

        assert "model" not in payload


def build_questions():
    return decider.build_request([{"role": "user", "content": "hi"}])["questions"]


class TestAvailability:
    """Which workers are offered at all.

    A rented GPU runs a few hours a day, so the option set is a runtime fact.
    """

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        decider._health.clear()
        yield
        decider._health.clear()

    @pytest.mark.asyncio
    async def test_always_offers_the_hosted_routes(self, monkeypatch):
        _stub_probe(monkeypatch, up=False)

        assert set(await decider.available_routes()) == set(decider.ALWAYS_UP)

    @pytest.mark.asyncio
    async def test_offers_a_local_worker_that_answers(self, monkeypatch):
        _stub_probe(monkeypatch, up=True)
        monkeypatch.setitem(decider.PROBES, "runpod", "http://gpu:8000/v1")

        assert "runpod" in await decider.available_routes()

    # The important one. A confident, correct decision to keep a request local
    # is worse than useless if that worker is switched off: dispatch fails and
    # the fallback sends it off-premises anyway.
    @pytest.mark.asyncio
    async def test_withholds_a_local_worker_that_is_down(self, monkeypatch):
        _stub_probe(monkeypatch, up=False)
        monkeypatch.setitem(decider.PROBES, "runpod", "http://gpu:8000/v1")

        assert "runpod" not in await decider.available_routes()

    @pytest.mark.asyncio
    async def test_treats_an_unconfigured_worker_as_absent(self, monkeypatch):
        monkeypatch.setitem(decider.PROBES, "runpod", "")

        assert await decider._is_up("runpod", "") is False

    @pytest.mark.asyncio
    async def test_does_not_probe_again_within_the_cache_window(self, monkeypatch):
        probes = _stub_probe(monkeypatch, up=True)

        await decider._is_up("runpod", "http://gpu:8000/v1")
        await decider._is_up("runpod", "http://gpu:8000/v1")

        assert probes["count"] == 1

    @pytest.mark.asyncio
    async def test_a_withheld_route_cannot_be_chosen(self, monkeypatch):
        # Even if laya names it -- a stale model, a replayed response --
        # an option that was not offered is not an answer.
        _stub_probe(monkeypatch, up=False)
        routes = await decider.available_routes()

        body = choice_response("runpod", {"runpod": 1.0})

        assert decider.choose(body, routes) is None

    @pytest.mark.asyncio
    async def test_runpod_probe_forwards_the_key(self, monkeypatch):
        # RunPod's proxy rejects even /models without the token, so the probe
        # must carry the same bearer key dispatch will use.
        monkeypatch.setitem(os.environ, "RUNPOD_API_KEY", "pod-key")
        captured = {}

        class Response:
            status_code = 200

        async def get(_url, **kwargs):
            captured.update(kwargs)
            return Response()

        monkeypatch.setattr(decider._client, "get", get)

        assert await decider._is_up("runpod", "http://gpu:8000/v1") is True
        assert captured["headers"].get("Authorization") == "Bearer pod-key"


class TestDecide:
    """The network path, with the transport stubbed."""

    @pytest.fixture(autouse=True)
    def no_local_workers(self, monkeypatch):
        _stub_probe(monkeypatch, up=False)
        decider._health.clear()
        yield
        decider._health.clear()

    @pytest.mark.asyncio
    async def test_returns_the_chosen_worker(self, monkeypatch):
        _stub_post(monkeypatch, json=choice_response("deepseek-v4-pro", {"deepseek-v4-pro": 0.9}))

        assert await decider.decide([{"role": "user", "content": "hi"}]) == "deepseek-v4-pro"

    @pytest.mark.asyncio
    async def test_routes_a_decision_to_the_decision_worker(self, monkeypatch):
        _stub_post(monkeypatch, json=noul_response(0.9))

        assert await decider.decide([{"role": "user", "content": "classify this"}]) == decider.DECISION_WORKER

    @pytest.mark.asyncio
    async def test_a_prose_request_below_the_threshold_routes_to_an_llm(self, monkeypatch):
        body = {
            "answers": {
                "is_decision": noul_response(0.4)["answers"]["is_decision"],
                "route": choice_response("deepseek-flash", {"deepseek-flash": 0.9})["answers"]["route"],
            }
        }
        _stub_post(monkeypatch, json=body)

        assert await decider.decide([{"role": "user", "content": "write me an email"}]) == "deepseek-flash"

    @pytest.mark.asyncio
    async def test_fails_open_when_laya_is_unreachable(self, monkeypatch):
        # The point of the fallback: a decider that is down degrades the
        # choice, it does not take the service with it.
        import httpx

        async def refuse(*_args, **_kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(decider._client, "post", refuse)

        assert await decider.decide([]) == decider.DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_fails_open_on_a_timeout(self, monkeypatch):
        import httpx

        async def hang(*_args, **_kwargs):
            raise httpx.ReadTimeout("too slow")

        monkeypatch.setattr(decider._client, "post", hang)

        assert await decider.decide([]) == decider.DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_fails_open_on_an_unrecognised_body(self, monkeypatch):
        _stub_post(monkeypatch, json={"answers": {"route": {"choices": []}}})

        assert await decider.decide([]) == decider.DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_forwards_the_bearer_key_when_configured(self, monkeypatch):
        # laya refuses the request without it, so the key must ride along.
        monkeypatch.setitem(os.environ, "LAYA_API_KEY", "s3cret")

        captured = {}

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return choice_response("deepseek-v4-pro", {"deepseek-v4-pro": 1.0})

        async def post(*_args, **_kwargs):
            captured.update(_kwargs)
            return Response()

        monkeypatch.setattr(decider._client, "post", post)

        assert await decider.decide([{"role": "user", "content": "hi"}]) == "deepseek-v4-pro"
        assert captured["headers"].get("Authorization") == "Bearer s3cret"

    @pytest.mark.asyncio
    async def test_sends_no_auth_header_without_a_key(self, monkeypatch):
        # Without LAYA_API_KEY there is nothing to prove, so no header goes out.
        monkeypatch.delenv("LAYA_API_KEY", raising=False)

        captured = {}

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return choice_response("deepseek-v4-pro", {"deepseek-v4-pro": 1.0})

        async def post(*_args, **_kwargs):
            captured.update(_kwargs)
            return Response()

        monkeypatch.setattr(decider._client, "post", post)

        assert await decider.decide([{"role": "user", "content": "hi"}]) == "deepseek-v4-pro"
        assert "Authorization" not in captured["headers"]


def _stub_probe(monkeypatch, up):
    """Stub the health probe. Returns a dict whose "count" tracks calls."""
    probes = {"count": 0}

    class Response:
        status_code = 200 if up else 503

    async def get(*_args, **_kwargs):
        probes["count"] += 1
        return Response()

    monkeypatch.setattr(decider._client, "get", get)
    return probes


def _stub_post(monkeypatch, json):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return json

    async def post(*_args, **_kwargs):
        return Response()

    monkeypatch.setattr(decider._client, "post", post)
