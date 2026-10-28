# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The end_call guard's closing classifier: did the caller's last words close the call?

One narrow question, asked of the platform's own default LLM (Gemini
Flash-Lite through the pinned Google plugin) on the LAST exchange only: the
assistant line before the caller's turn and the caller's turn. No history, on
purpose: the model that wanted to hang up had drifted on its own accumulated
replies to a caller who kept talking; the judge must not carry that.

Four answers, by what the CALLER's words do: ``closing`` (a goodbye, or
confirming nothing else is needed given the agent's question),
``wrong_person`` (the answerer says the person is not available, not them, or
wrong number, and does not offer to handle the matter), ``machine`` (a
recorded greeting, IVR or hold announcement; not a person asking to hold),
``not_closing`` (everything else, including "don't say goodbye yet", "please
hold while I get her", "I'm not Maria but I handle this", and a garbled
fragment). Anything else is unavailable, and unavailable is today's baseline:
the guard fails open on it, counted.

**Credentials.** The platform's own key from the worker's BASE environment
(``GEMINI_API_KEY``, falling back to ``GOOGLE_API_KEY``), never a call's BYOK
overlay: the component-credential overlay is a dict copy the worker hands its
own stack and never mutates ``os.environ``. ``vertexai=False`` so an ambient
Vertex configuration cannot discard the supplied key.

**Lifecycle.** One instance per job (a warm process serves more than one job,
so the disarm is per job). The self-test is STARTED at job start and awaited
lazily, under its own timeout, before the first real request, never on the
call-answer path: end_call cannot be attempted before the first exchange.
Construction and self-test sit inside the classifier-only failure boundary:
any failure disarms this instance (``available`` is False) and the structural
rules keep working. Each request runs with ``max_retry=0`` under the guard's
remaining deadline and closes its stream; ``aclose()`` at job teardown closes
the shared client.

**Model name.** The Google plugin treats a name containing ``gemini-3`` and
``flash`` as a Gemini 3 Flash and turns thinking on; a one-token answer needs
none of that, so thinking is pinned off.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Mapping

try:  # package import in tests; flat /app import in the pinned worker image
    from .end_call_guard import (
        ASSISTANT_TAIL_CHARS,
        VERDICT_CLOSING,
        VERDICTS,
        Exchange,
        parse_verdict,
    )
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from end_call_guard import (  # type: ignore[no-redef]
        ASSISTANT_TAIL_CHARS,
        VERDICT_CLOSING,
        VERDICTS,
        Exchange,
        parse_verdict,
    )

#: The platform's default LLM; passed as ``str`` exactly as the call's own
#: LLM is (the plugin's ``ChatModels`` Literal predates it).
CLASSIFIER_MODEL = "gemini-3.5-flash-lite"
SELF_TEST_TIMEOUT_SECONDS = 2.0
#: The request deadline handed to the plugin, which forwards it to Gemini as
#: the HTTP deadline; the API refuses any deadline under 10 s ("Manually set
#: deadline 2s is too short. Minimum allowed deadline is 10s.", 400). The
#: guard's latency bound (1.5 s per decision, 2.0 s for the self-test) is
#: therefore enforced LOCALLY with ``asyncio.wait_for``, never as the
#: request deadline. Found by the harness: with the bound forwarded, every
#: call's self-test failed and the classifier disarmed before its first
#: decision, while the offline gate (timeout 15 s) never noticed.
REQUEST_DEADLINE_SECONDS = 10.0
SELF_TEST_EXCHANGE = Exchange(
    caller_text="That is everything I needed, thank you, goodbye.",
    agent_text="Is there anything else I can help you with?",
    agent_interrupted=False,
    language="en",
)
CUT_MARKER = "[the agent's line was cut off here]"

INSTRUCTION = (
    "You judge one exchange from a phone call between an assistant and a caller. "
    "Answer with exactly one word from this list and nothing else: closing, "
    "wrong_person, machine, not_closing.\n"
    "closing: the caller said goodbye, or confirmed they need nothing else, given "
    "what the assistant had just asked.\n"
    "wrong_person: the person who answered says the person the assistant asked for "
    "is not available, is not them, or that this is the wrong number, and does not "
    "offer to handle the matter themselves.\n"
    "machine: the caller's words are a recorded greeting, a voicemail prompt, an "
    "IVR menu or a hold announcement; a live person asking you to hold is NOT a "
    "machine.\n"
    "not_closing: anything else, including a request, a question, a new detail, a "
    "fragment, garbled words, 'don't hang up yet', or a person who says they are "
    "not the one you asked for but will help.\n"
    "If the assistant's line ends with the marker "
    f"{CUT_MARKER} it was cut off there: infer nothing the assistant did not finish "
    "saying, and a question it did not finish asking was not asked. When in doubt, "
    "answer not_closing."
)


def platform_api_key(environ: Mapping[str, str]) -> str:
    """The platform's Gemini key from the BASE environment, or ``""``."""

    return str(environ.get("GEMINI_API_KEY") or environ.get("GOOGLE_API_KEY") or "").strip()


def render_exchange(exchange: Exchange) -> str:
    agent = exchange.agent_text[-ASSISTANT_TAIL_CHARS:]
    if exchange.agent_interrupted:
        agent = f"{agent} {CUT_MARKER}"
    language = f" (language: {exchange.language})" if exchange.language else ""
    return f"Assistant{language}: {agent}\nCaller: {exchange.caller_text}\nAnswer:"


class GoogleClosingClassifier:
    """The adapter over the pinned Google plugin. See the module docstring."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = CLASSIFIER_MODEL,
        emit: Callable[..., None] = lambda *_a, **_k: None,
        llm_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._emit = emit
        self._clock = clock
        self._llm: Any = None
        self._available = False
        self._self_test: asyncio.Task[bool] | None = None
        self.self_test_ms: int | None = None
        self.self_test_error = ""
        if not api_key:
            self._disarm("no_platform_key")
            return
        try:
            self._llm = (llm_factory or (lambda: self._build(api_key, model)))()
            self._available = True
        except Exception as exc:  # noqa: BLE001 - construction is inside the failure boundary
            self._disarm(type(exc).__name__)

    @staticmethod
    def _build(api_key: str, model: str) -> Any:
        from livekit.plugins import google

        return google.LLM(
            model=model,
            api_key=api_key,
            vertexai=False,
            temperature=0.0,
            # A one-token answer needs no thinking. The plugin configures a
            # name containing "gemini-3" and "flash" by LEVEL, not budget, and
            # would otherwise leave the model's default level on; "low" is the
            # lowest the level-based models accept. No ``http_options``: an
            # explicit one would override each request's ``conn_options``
            # timeout.
            thinking_config={"thinking_level": "low"},
        )

    @property
    def available(self) -> bool:
        return self._available

    def _disarm(self, error: str) -> None:
        self._available = False
        self.self_test_error = error
        try:
            self._emit("end_call_guard_classifier_disarmed", error_type=error)
        except Exception:  # noqa: BLE001 - a diagnostic cannot decide anything
            pass

    def start_self_test(self) -> None:
        """Begin the self-test concurrently with session bring-up; never awaited here."""

        if not self._available or self._self_test is not None:
            return
        self._self_test = asyncio.ensure_future(self._run_self_test())

    async def _run_self_test(self) -> bool:
        started = self._clock()
        try:
            raw = await asyncio.wait_for(
                self._request(SELF_TEST_EXCHANGE, timeout=SELF_TEST_TIMEOUT_SECONDS),
                timeout=SELF_TEST_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - inside the failure boundary
            self._disarm(type(exc).__name__)
            return False
        self.self_test_ms = int((self._clock() - started) * 1000)
        if parse_verdict(raw) != VERDICT_CLOSING:
            self._disarm("self_test_wrong_answer")
            return False
        return True

    async def classify(self, exchange: Exchange, *, timeout: float) -> str:
        """One of ``VERDICTS`` or raises; the guard maps everything else to unavailable."""

        if not self._available:
            raise RuntimeError("classifier disarmed")
        if self._self_test is not None and not self._self_test.done():
            # Lazily, before the first real request: whatever the self-test
            # budget leaves of THIS request's timeout.
            await asyncio.wait_for(asyncio.shield(self._self_test), timeout=timeout)
        if not self._available:
            raise RuntimeError("classifier disarmed")
        # The latency bound is local: the request itself carries the API's
        # minimum deadline (see ``REQUEST_DEADLINE_SECONDS``).
        raw = await asyncio.wait_for(self._request(exchange, timeout=timeout), timeout=timeout)
        verdict = parse_verdict(raw)
        if verdict not in VERDICTS:
            raise ValueError("answer off the set")
        return verdict

    async def _request(self, exchange: Exchange, *, timeout: float) -> str:
        from livekit.agents import llm
        from livekit.agents.types import APIConnectOptions

        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="system", content=INSTRUCTION)
        chat_ctx.add_message(role="user", content=render_exchange(exchange))
        stream = self._llm.chat(
            chat_ctx=chat_ctx,
            # Never below the API's minimum: the caller bounds the wait.
            conn_options=APIConnectOptions(max_retry=0, timeout=max(REQUEST_DEADLINE_SECONDS, float(timeout))),
        )
        text = ""
        try:
            async for chunk in stream:
                delta = getattr(chunk, "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    text += str(content)
                    if len(text) > 40:
                        break
        finally:
            # Per-request: the stream and its background task. The shared
            # client is closed only at job teardown (``aclose``).
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001 - closing a finished stream is a no-op
                pass
        return text

    async def aclose(self) -> None:
        if self._self_test is not None and not self._self_test.done():
            self._self_test.cancel()
        llm_client = self._llm
        self._llm = None
        self._available = False
        if llm_client is not None:
            try:
                await llm_client.aclose()
            except Exception:  # noqa: BLE001 - teardown
                pass


def build_platform_classifier(
    environ: Mapping[str, str] | None = None,
    *,
    emit: Callable[..., None] = lambda *_a, **_k: None,
) -> GoogleClosingClassifier:
    """The per-job classifier on the platform's own key; disarmed when there is none."""

    return GoogleClosingClassifier(api_key=platform_api_key(environ if environ is not None else os.environ), emit=emit)


__all__ = [
    "CLASSIFIER_MODEL",
    "CUT_MARKER",
    "GoogleClosingClassifier",
    "INSTRUCTION",
    "REQUEST_DEADLINE_SECONDS",
    "SELF_TEST_EXCHANGE",
    "build_platform_classifier",
    "platform_api_key",
    "render_exchange",
]
