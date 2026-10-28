# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""GPT-Live-1 session for the LiveKit worker.

The control plane cannot be imported here; what it renders into the call's
realtime config (the prompt, the greeting, the call variables) is consumed
as delivered, never rebuilt.
"""

from __future__ import annotations

try:
    from .voice_configuration import refuse_freeform_workflow
except ImportError:
    from voice_configuration import refuse_freeform_workflow

import asyncio
import os
import re
import socket
from typing import Any, Callable

GPT_LIVE_MODEL = "gpt-live-1"
DEFAULT_VOICE = "marin"
DEFAULT_BACKEND = "gpt-5.6-luna"

# Safety net only. The real hangup is the end_call tool. This is not a
# language allowlist: representative closings from several families so EN/ET
# is not special. Primary path remains the tool.
_FAREWELL = re.compile(
    r"("
    r"\b(goodbye|good bye|bye|farewell)\b|"
    r"\b(head aega|nägemist)\b|"
    r"\b(adiós|adios|hasta luego)\b|"
    r"\b(au revoir)\b|"
    r"\b(tschüss|tschuss|auf wiedersehen)\b|"
    r"\b(пока|до свидания)\b|"
    r"さようなら|"
    r"مع السلامة"
    r")",
    re.IGNORECASE,
)


# Voice-to-voice engines share opening, witness, hangup and language-follow
# seams. Gemini does not get a copy of those helpers.
try:
    from .conversation_engines_generated import ELEVEN_REALTIME_MODEL, ENGINE_DECLARATIONS
except ImportError:
    from conversation_engines_generated import ELEVEN_REALTIME_MODEL, ENGINE_DECLARATIONS

S2S_LIVE_FRONTENDS = frozenset(
    name for name, declaration in ENGINE_DECLARATIONS.items()
    if not declaration["has_language_axis"] and not declaration["speaks_text"]
)


def _voice_frontend(config: dict[str, Any] | None) -> str:
    data = config if isinstance(config, dict) else {}
    voice = data.get("voice") if isinstance(data.get("voice"), dict) else {}
    return str(data.get("voice_frontend") or voice.get("voice_frontend") or "").strip().lower()


def voice_frontend(config: dict[str, Any] | None) -> str:
    """The call's engine key (``gpt_live``, ``gemini_live``; empty for cascade)."""

    return _voice_frontend(config)


def is_gpt_live_config(config: dict[str, Any] | None) -> bool:
    return _voice_frontend(config) == "gpt_live"


def is_gemini_live_config(config: dict[str, Any] | None) -> bool:
    return _voice_frontend(config) == "gemini_live"


def is_eleven_native_config(config: dict[str, Any] | None) -> bool:
    return _voice_frontend(config) == "eleven_realtime"


def supports_manual_live_replies(config: dict[str, Any] | None) -> bool:
    return bool(ENGINE_DECLARATIONS.get(_voice_frontend(config), {}).get("supports_manual_replies", True))


def is_s2s_live_config(config: dict[str, Any] | None) -> bool:
    return _voice_frontend(config) in S2S_LIVE_FRONTENDS


#: The spoken-line budget the control plane admits a literal opening under,
#: in code points (``call_variables/template.LINE_BUDGET``); a second copy,
#: pinned equal by the control-plane suite. An opening past it here is a
#: bug in admission, never something to trim: nothing is cut on this path.
OPENING_LINE_BUDGET = 1000


class OpeningOverBudget(ValueError):
    """An opening reached the worker past the budget admission enforces."""


def greet_directive(
    opening: str,
    *,
    opening_mode: str = "literal",
    language_name: str = "",
) -> str:
    """The instruction the session's opening reply is generated from.

    Handed to ``session.generate_reply(instructions=...)`` by the worker's
    opening seam, never appended on its own: the reply owns a speech handle,
    a timeout and the opening protection; a bare commentary owns nothing.

    ``opening_mode`` is the call's contract for the opening. ``literal`` is
    what the control plane sets whenever a per-call ``first_message`` is
    merged over the frozen profile, and it means exactly that: the line is
    spoken as written, in the script and language it is written in. The
    first GPT-Live customer batch (2026-09-15, Sinhala) received its
    openings paraphrased into English because this directive handed every
    opening over "as intent"; that wording now applies only to a generated
    opening. ``language_name`` is the English name of the call's language
    (``"Sinhala"``); empty omits the language clause rather than instructing
    the model to speak "in ".

    This is an instruction to a voice-to-voice model. It raises adherence
    and cannot guarantee it.
    """
    opening = (opening or "").strip()
    if len(opening) > OPENING_LINE_BUDGET:
        raise OpeningOverBudget(f"opening of {len(opening)} code points exceeds the admitted budget of {OPENING_LINE_BUDGET}")
    language_name = (language_name or "").strip()
    speak_in = f" Speak it in {language_name}." if language_name else ""
    if opening and str(opening_mode or "").strip().casefold() == "literal":
        return (
            "Greet now by saying this opening exactly as written, in the language and "
            "script it is written in. Do not translate it, paraphrase it, shorten it or "
            f"add to it.{speak_in} Then stop and wait for the caller. "
            f"The opening: {opening}"
        )
    if opening:
        return (
            "Greet now using this opening as intent, not a script to read word-for-word."
            f"{speak_in} The opening: {opening}"
        )
    # A generated opening: the call's instructions already carry the opening
    # guidance and the call variables, rendered by the control plane, so the
    # directive points the model at them and names no detail of its own.
    parts = ["Greet the person using the opening guidance in your instructions."]
    if language_name:
        parts.append(f"Speak in {language_name}.")
    parts.append("Then wait and listen.")
    return " ".join(parts)


class LiveEngineUnavailable(RuntimeError):
    """This worker image cannot build the voice-to-voice engine a call names.

    Raised before any media exists, so a caller refuses the session rather
    than connecting it to silence. Each entrypoint turns it into its own
    refusal: the phone worker's configuration error, the homepage demo's
    rejected job.
    """


def live_model_needs_http_session(config: dict[str, Any] | None) -> bool:
    """Whether :func:`build_live_model` borrows a worker-owned HTTP transport.

    GPT-Live's model rides an ``aiohttp`` session the WORKER owns and must
    close after the live session (:func:`close_live_session_and_transport`);
    Gemini Live's owns its own. Asked separately so the caller creates, holds
    and closes the transport -- a builder that created it would leak it the
    moment the plugin constructor raised.
    """

    return is_gpt_live_config(config) or is_eleven_native_config(config)


class CallAgentRegistry:
    """The control plane's Eleven agent registry, reached through one call."""

    def __init__(self, call: Any) -> None:
        self.call = call

    async def _ask(self, operation: str, tool_set: str, agent_id: str = "") -> dict[str, Any]:
        return await asyncio.to_thread(self.call.native_agent, operation, tool_set, agent_id)

    async def resolve(self, tool_set: str) -> dict[str, Any]:
        return await self._ask("resolve", tool_set)

    async def register(self, tool_set: str, agent_id: str) -> dict[str, Any]:
        return await self._ask("register", tool_set, agent_id)

    async def forget(self, tool_set: str, agent_id: str) -> dict[str, Any]:
        return await self._ask("forget", tool_set, agent_id)


def build_live_model(
    config: dict[str, Any],
    *,
    environ: Any,
    http_session: Any = None,
    navigation: bool = False,
    emit: Callable[..., None] | None = None,
    native_call_id: str = "",
    native_max_duration_seconds: int = 600,
    native_registry: Any = None,
) -> Any:
    """The duplex model for a voice-to-voice call, from its realtime config.

    One construction for every entrypoint that runs a live engine: the phone
    and workspace browser-test worker (``production_agent``) and the homepage
    demo (``browser_demo_agent``). Moved here unchanged from the
    phone worker so the two cannot drift into two live runtimes.

    Plugins are imported here, at call time, because only the voice image
    ships them; a missing one raises :class:`LiveEngineUnavailable`.
    """

    refuse_freeform_workflow(config)
    voice_id = str((config.get("voice") or {}).get("voice_id") or "")
    if is_eleven_native_config(config):
        if not native_call_id or native_registry is None:
            raise LiveEngineUnavailable("native realtime requires a call identity and agent registry")
        key = environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise LiveEngineUnavailable("Eleven Realtime credential is unavailable")
        voice = config.get("voice") or {}
        model = config.get("model") or {}
        opening = config.get("opening") or {}
        if (navigation or voice.get("model") != ELEVEN_REALTIME_MODEL
                or model.get("model") != ELEVEN_REALTIME_MODEL
                or config.get("opening_mode") != "literal"
                or not str(config.get("first_message") or "").strip()
                or opening.get("first_speaker") != "agent"):
            raise LiveEngineUnavailable("Eleven Realtime requires a literal agent-first opening and its own model")
        try:
            from .eleven_native import build_eleven_native
        except ImportError:
            try:
                from eleven_native import build_eleven_native
            except ModuleNotFoundError as exc:
                # The self-hosted gateway does not ship this engine.
                if exc.name != "eleven_native":
                    raise
                raise LiveEngineUnavailable("Eleven Realtime is not part of this build") from None
        return build_eleven_native(
            api_key=key, voice=voice_id, language=str(config.get("language") or ""),
            first_message=str(config["first_message"]),
            instructions=str(config.get("system_prompt") or ""),
            call_id=native_call_id, max_duration_seconds=native_max_duration_seconds,
            http_session=http_session, registry=native_registry,
        )
    if is_gpt_live_config(config):
        try:
            from livekit.plugins.openai.realtime import GPTLiveModel
            from livekit.agents.types import APIConnectOptions as GptLiveConnectOptions
        except ImportError as exc:
            raise LiveEngineUnavailable("GPT-Live is not pinned on this voice worker") from exc

        backend = str((config.get("model") or {}).get("model") or DEFAULT_BACKEND)
        policy, in_prompt = live_call_ending_policy(config)
        return GPTLiveModel(
            voice=voice_id or DEFAULT_VOICE,
            delegation="responses",
            responses_options={
                "model": backend,
                # The backend decides to call end_call, and it is instructed
                # separately from the voice session, so it gets the same
                # rendered prompt (reference form, call variables included).
                # A current control plane composes the policy INTO that
                # prompt; only a legacy config's sentence is appended, so the
                # backend reads the policy exactly once.
                "instructions": (
                    str(config.get("system_prompt") or "").strip()
                    + ("\n" + policy if policy and not in_prompt else "")
                ),
            },
            http_session=http_session,
            conn_options=GptLiveConnectOptions(max_retry=3, retry_interval=2.0, timeout=30.0),
        )
    if is_gemini_live_config(config):
        try:
            from livekit.plugins.google.realtime import RealtimeModel as GeminiRealtimeModel
            from livekit.plugins.google.realtime.realtime_api import RealtimeSession as GeminiRealtimeSession
        except ImportError as exc:
            raise LiveEngineUnavailable("Gemini Live is not pinned on this voice worker") from exc
        gemini_model = str((config.get("model") or {}).get("model") or "gemini-3.8-live")
        gemini_key = environ.get("GOOGLE_API_KEY") or environ.get("GEMINI_API_KEY")
        if navigation:
            # A call that may wait in a queue must survive the provider refusing a resumption
            # handle (see ``gemini_live_hold``); every other call keeps the stock model.
            try:
                from navigation_entry import build_hold_safe_gemini
            except ImportError:  # package import in tests
                from .navigation_entry import build_hold_safe_gemini
            try:
                return build_hold_safe_gemini(
                    model=gemini_model, voice=voice_id or "Kore", api_key=gemini_key, emit=emit
                )
            except ImportError:
                if emit is not None:
                    try:
                        emit("gemini_live_hold_unavailable")
                    except Exception:  # noqa: BLE001 - a diagnostic never decides which model the call gets
                        pass
        try:
            from realtime_session_clock import session_clock_classes
        except ImportError:  # package import in tests
            from .realtime_session_clock import session_clock_classes
        # Gemini's protocol reports tokens and no session time, and the call is
        # priced on that time: the clocked session reports its connected seconds.
        model_cls, _ = session_clock_classes(GeminiRealtimeModel, GeminiRealtimeSession)
        return model_cls(model=gemini_model, voice=voice_id or "Kore", api_key=gemini_key)
    raise LiveEngineUnavailable("config names no voice-to-voice engine")


def openai_http_session():
    import aiohttp

    family = socket.AF_INET if os.environ.get("SKYSAY_GPT_LIVE_FORCE_IPV4") == "1" else 0
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60, sock_connect=30),
        connector=aiohttp.TCPConnector(family=family) if family else aiohttp.TCPConnector(),
    )


def _attr_or_none(obj: Any, name: str) -> Any:
    """``getattr`` that also absorbs a raising property.

    Some duplex-session candidates raise instead of simply being absent when
    their model is the wrong kind (e.g. not a duplex model). Plain
    ``getattr(obj, name, None)`` only swallows ``AttributeError``, so a
    raising read is treated the same as one that raises ``AttributeError``:
    absent, not fatal.
    """
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _duplex_session(agent: Any, model: Any | None) -> Any:
    for candidate in (
        _attr_or_none(agent, "duplex_session"),
        _attr_or_none(model, "duplex_session"),
        _attr_or_none(agent, "_realtime_session"),
        _attr_or_none(model, "_realtime_session"),
        model,
        agent,
    ):
        if candidate is not None and (
            hasattr(candidate, "append_thinking") or hasattr(candidate, "append_commentary")
        ):
            return candidate
    return None


# The live session's close has to finish before the transport goes, but it is
# teardown: a provider that never answers the close handshake must not hold the
# rest of the settlement behind it. The same budget, and the same reason, as
# ``await_bounded_production_session_close`` -- this worker still owns evidence,
# usage and terminal settlement after it, and neither close may cede the exit
# deadline.
LIVE_SESSION_CLOSE_TIMEOUT_SECONDS = 2.0

# How long a cancellation waits for a transport close it already started.
# Same shape and the same purpose as ``cancellation_grace_seconds`` in
# ``await_bounded_production_session_close``: long enough for an aiohttp
# session close, which does not wait on the network, and short enough that a
# cancelled job is not held open by it.
TRANSPORT_CLOSE_GRACE_SECONDS = 0.2


def _live_session_owning_transport(agent: Any, model: Any | None) -> Any:
    """The duplex SESSION that holds the provider socket, or ``None``.

    ``_duplex_session`` answers a different question -- "what can I append
    context to" -- and falls back to the model and then the agent so that a
    context injection still lands somewhere. Neither of those owns a socket,
    and closing the agent here would end far more than a transport, so the
    fallbacks are exactly what this lookup must reject.
    """

    candidate = _duplex_session(agent, model)
    if candidate is None or candidate is model or candidate is agent:
        return None
    return candidate


def live_duplex_session(agent: Any, model: Any | None) -> Any:
    """The duplex session a call's caller audio flows through, or ``None``.

    The same lookup the teardown uses to find the socket's owner, so a hold on
    what the model hears and the close of what it hears through can never
    disagree about which object that is.
    """

    return _live_session_owning_transport(agent, model)


def notify_native_close_reclaimed(
    agent: Any, model: Any | None, door: Callable[[str], str | None] | None = None
) -> bool:
    """Correct GPT's pending-close context without requesting spoken output.

    ``door`` is the value-mode redactor's door for a string the controller adds to a model's
    context (``navigation_chat.ModelContextGate.thinking``): ``None`` from it adds nothing.
    """
    duplex = live_duplex_session(agent, model)
    append = getattr(duplex, "append_thinking", None)
    if not callable(append):
        return False
    text = (
        "Call status: the previous end_call request was canceled because the caller "
        "triggered caller-side activity. The call is still open. Listen for and answer any new request. "
        "If the caller later asks to finish, invoke end_call again; the previous request "
        "will not close this call. Do not announce this internal status."
    )
    if door is not None:
        text = door(text)
        if text is None:
            return False
    append(text)
    return True


async def close_live_session_and_transport(
    agent: Any,
    model: Any | None,
    transport: Any,
    *,
    timeout: float = LIVE_SESSION_CLOSE_TIMEOUT_SECONDS,
    on_transport_error: Callable[[str], None] | None = None,
) -> str:
    """Close the live session, then ALWAYS the HTTP transport it borrows.

    Both halves live here because the ORDER is the contract and the transport
    close is unconditional. Split across the call site, a job cancellation
    arriving during the session close would leave this worker's own aiohttp
    session open with nothing left to release it -- a leak the previous
    transport-first teardown could not have.

    The worker hands the duplex model an ``aiohttp`` session it owns itself
    (``openai_http_session``), so teardown closes two things that are not
    independent. Closing the transport first tears the websocket out from
    under a session that still believes it is running: the plugin reports
    ``connection closed unexpectedly``, retries, and the retry raises
    ``RuntimeError("Session is closed")`` from the closed ``aiohttp`` session.
    A ``RuntimeError`` is not an ``APIError``, so it bypasses the plugin's
    retry budget entirely and is emitted as an UNRECOVERABLE session error --
    which is how an ordinary unanswered call came to be reported as a failed
    AI session, twice per call, on the voice worker's error tracker.

    Closed in this order the session sets its own ``closing`` flag first, so
    the receive task returns quietly and nothing is reported. Returns a
    bounded status label for the call's evidence; it never raises, because at
    this point the transport is closing either way and the only choice left is
    whether the session is told about it first.
    """

    try:
        return await _close_live_session(agent, model, timeout=timeout)
    finally:
        closing = asyncio.ensure_future(transport.close())
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            # ``shield`` keeps the close RUNNING, but this await returns at
            # once -- so re-raising here would let the job unwind and the loop
            # tear down a close that is still in flight, which is the leak
            # this block exists to prevent. Finish what was started, bounded,
            # and only then let the cancellation through.
            await asyncio.wait({closing}, timeout=TRANSPORT_CLOSE_GRACE_SECONDS)
            if closing.done() and not closing.cancelled():
                # Consumed so a failed close is not reported as an unretrieved
                # task exception; on this path there is no longer anywhere to
                # report it to.
                closing.exception()
            raise
        except Exception as exc:
            if on_transport_error is not None:
                on_transport_error(type(exc).__name__)


async def _close_live_session(
    agent: Any,
    model: Any | None,
    *,
    timeout: float,
) -> str:
    """Tell the live session the call is over. Never raises but for cancellation.

    ``timeout`` and ``failed`` are degraded, not safe: a session that did not
    reach its own closing state still reads the transport close as a provider
    fault and reports it, exactly as it did before this ordering existed. That
    is the status quo rather than a regression, and the returned label is what
    makes the degraded case visible instead of silent.
    """

    session = _live_session_owning_transport(agent, model)
    if session is None:
        return "absent"
    aclose = _attr_or_none(session, "aclose")
    if aclose is None:
        return "unsupported"
    try:
        await asyncio.wait_for(aclose(), timeout)
    except asyncio.CancelledError:
        # An external cancellation of the job is not this helper's to swallow.
        raise
    except asyncio.TimeoutError:
        return "timeout"
    except Exception:
        return "failed"
    return "closed"


async def await_duplex_session(
    agent: Any,
    *,
    model: Any | None = None,
) -> bool:
    """Wait for the live session's own start; report whether one exists.

    The session's context is its instructions, rendered by the control plane
    with the call's variables; nothing is appended here. The opening is
    spoken through the session's own awaited reply seam
    (``generate_opening_reply`` in the worker, handed :func:`greet_directive`
    as its instruction), which owns a speech handle, a reply timeout and the
    opening protection. Returns whether a duplex session was found; the
    worker records that, and never routes the opening on it.
    """

    live = _duplex_session(agent, model)
    if live is None:
        return False
    started = getattr(live, "_session_started_fut", None)
    if started is not None:
        # Shielded so that cancelling this task -- by the timeout below, or
        # from outside -- cannot cancel the session's own startup future.
        await asyncio.wait_for(asyncio.shield(started), 20)
    return True


def should_hangup_from_user_text(text: str) -> bool:
    return bool(_FAREWELL.search(text or ""))


NATIVE_REPLY_RECOVERY_INSTRUCTION = (
    "The caller has spoken and is waiting. Respond to their latest turn now. "
    "If you have decided the conversation is over, ask your tool-calling backend "
    "to execute end_call now; saying goodbye alone does not disconnect the call. "
    "If the caller is continuing or asking a question, answer and keep the call open. "
    "Do not repeat any completed business action."
)


def request_native_reply_recovery(realtime: Any, *, emit: Callable[..., None]) -> bool:
    """Nudge the native stream without reserving a forced speech turn.

    A forced SDK reply waits for audio before releasing its scheduler slot,
    delaying a tool-only end_call behind that wait. The native commentary
    command uses the same provider input while leaving speech and tools on
    their ordinary autonomous path. The watchdog still owns the deadline.
    """
    duplex = getattr(realtime, "duplex_session", None)
    append = getattr(duplex, "append_commentary", None)
    if not callable(append):
        return False
    try:
        append(
            "Immediately follow the instruction below. Do not wait for the caller "
            "to speak first. After that, pause and listen.\n\n"
            + NATIVE_REPLY_RECOVERY_INSTRUCTION
        )
        emit("native_reply_recovery_requested")
    except Exception as exc:
        emit("native_reply_recovery_failed", error_type=type(exc).__name__)
    return True


class NativeFarewellFallback:
    """A revocable delayed fallback for a caller's closing caption.

    The end-call owner still decides whether a close is allowed. This helper
    only owns the delay: a new caller speech run invalidates an old caption,
    and a refused or reclaimed close must not consume the fallback forever.
    """

    def __init__(
        self,
        end_call: Any,
        message: str,
        emit: Callable[..., None],
        *,
        delay: float = 2.5,
        suspended: Callable[[], bool] | None = None,
    ):
        # True while the call is waiting in a phone menu or a hold queue, where
        # a recorded "goodbye" is not the caller closing the conversation.
        self.suspended = suspended or (lambda: False)
        self.end_call = end_call
        self.message = message
        self.emit = emit
        self.delay = delay
        self.task: asyncio.Task | None = None
        self.closed = False

    def note_item(self, *, role: str, text: str) -> None:
        if self.suspended():
            return
        if not self.closed and self.task is None and should_hangup_from_conversation_item(role=role, text=text):
            self.task = asyncio.create_task(self._run())

    def note_caller_speaking(self, speaking: bool) -> None:
        if speaking and self.task is not None:
            # This cancels only the fallback waiter. The worker's independent
            # terminal task is shielded by wait_for_completion and reclaims
            # itself from its own caller-floor evidence.
            self.task.cancel()
            self.task = None

    async def _run(self) -> None:
        try:
            await asyncio.sleep(self.delay)
            if self.suspended():
                # Re-read after the delay: the call may have entered a menu or a
                # queue since the words were heard. Nothing was started.
                return
            self.emit("native_farewell_fallback")
            await self.end_call({"message": self.message})
            await self.end_call.wait_for_completion()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.emit("gpt_live_farewell_hangup_failed")
        finally:
            if self.task is asyncio.current_task():
                self.task = None

    async def close(self) -> None:
        self.closed = True
        if self.task is not None:
            task = self.task
            self.task = None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def should_hangup_from_conversation_item(*, role: str, text: str) -> bool:
    """Hang up after the *caller* says a closing.

    Assistant speech is too full of ordinary phrases to regex-hangup. The
    end_call tool remains the worker-owned close for a spoken goodbye.
    """

    role_name = str(role or "").strip().lower()
    if role_name not in {"user", "human"}:
        return False
    return should_hangup_from_user_text(text)


# The call-ending sentence a voice-to-voice session got before the control
# plane composed one policy for every engine. Kept ONLY as the
# fallback for a config that predates that policy -- a new worker behind an
# older control plane -- and only when the call actually has end_call: it used
# to be appended unconditionally, telling a model without the tool to call it.
GPT_LIVE_END_CALL_INSTRUCTION = (
    "When the caller says goodbye, that's all, or otherwise ends the conversation, "
    "you MUST invoke end_call after a short goodbye. Saying goodbye out loud is not "
    "enough — the call stays up until end_call runs."
)

#: The config key the control plane writes the composed call-ending policy to.
#: Present (possibly ``""``) on every config from a control plane that composes
#: it; absent only on an older one.
CALL_ENDING_POLICY_KEY = "call_ending_policy"


def _has_end_call(config: dict[str, Any] | None) -> bool:
    tools = (config or {}).get("tools")
    return isinstance(tools, list) and any(
        isinstance(tool, dict)
        and tool.get("type") == "builtin"
        and tool.get("name") == "end_call"
        for tool in tools
    )


def live_call_ending_policy(config: dict[str, Any] | None) -> tuple[str, bool]:
    """``(policy, already_in_prompt)`` for a voice-to-voice call.

    The control plane composes the policy into the system prompt AND writes the
    same block to ``call_ending_policy``, so the voice session needs nothing
    added and the GPT-Live tool-calling backend -- instructed separately, and
    the one that decides to call end_call -- gets the identical text. An empty
    value means end_call is off and nothing is added anywhere.

    A config without the key comes from a control plane that predates the
    policy; the legacy sentence is used then, and only if end_call is enabled.
    """

    data = config if isinstance(config, dict) else {}
    if CALL_ENDING_POLICY_KEY in data:
        return str(data.get(CALL_ENDING_POLICY_KEY) or ""), True
    if _has_end_call(data):
        return GPT_LIVE_END_CALL_INSTRUCTION, False
    return "", False


def opening_language_instruction(language_name: str, *, opening_mode: str = "literal") -> str:
    """The session sentence that orders the opening before language following.

    The session used to carry an unconditional "Match the caller's latest
    language" sentence, so a callee's "Hello" before the greeting flipped a
    Sinhala call to English (first customer batch, 2026-09-15). The opening
    is now pinned to the call's language and following starts only after it
    has been spoken.

    Following itself is limited to a PERSON speaking to the agent. "The
    caller's latest language" was satisfied by hold music with English lyrics
    on 2026-09-24: the model answered the song twice, in English, on a Sinhala
    call. A voice-to-voice model cannot be mechanically locked to a language,
    so this is the only lever there is; it raises adherence and cannot
    guarantee it. ``language_name`` empty keeps the ordering and drops the
    language name. ``opening_mode`` keeps this sentence and the greeting
    directive telling the model the same thing: a literal opening is said as
    written, a generated one is opened from the directive it is given.

    Written as policy for the moment the greeting is given, never as a
    present-tense command to open: the session is created well before the
    callee answers, and the greeting directive only arrives then. A sentence
    that ordered "open the call now" would invite an improvised greeting in
    the window before the scripted line exists (review finding, 2026-09-15).
    """
    language_name = (language_name or "").strip()
    literal = str(opening_mode or "").strip().casefold() == "literal"
    in_language = f", in {language_name}" if language_name else ", in the language it is written in"
    if literal:
        line = f"If you are given an opening line, say it exactly as written{in_language}."
    else:
        line = "Open with the greeting you are directed to give" + (
            f", in {language_name}." if language_name else "."
        )
    opening = f"When you greet, speak in {language_name}. {line}" if language_name else line
    stay_in = language_name or "the language of your opening"
    return (
        f"{opening} Only after that opening has been spoken, follow the language of the "
        "person you are talking to: switch when they speak to you in another language in a "
        "full sentence, or ask you to. "
        "Never switch because of music, recordings, announcements, background voices, or "
        f"speech you could not make out; when you are not sure, stay in {stay_in}. "
        "Do not announce the switch."
    )


def gpt_live_session_instructions(
    instructions: str,
    *,
    language_name: str = "",
    opening_mode: str = "literal",
    call_ending_policy: str = "",
) -> str:
    """The full instruction string a voice-to-voice session is created with.

    Stored prompt first, then the opening/language ordering, then -- only for
    a config from an older control plane, see ``live_call_ending_policy`` --
    the legacy end_call sentence. A current config already carries the policy
    inside the stored prompt, so ``call_ending_policy`` is ``""`` and nothing
    is repeated. One builder so the order is a tested fact rather than an
    inline concatenation in the worker.
    """
    text = (
        str(instructions or "")
        + "\n"
        + opening_language_instruction(language_name, opening_mode=opening_mode)
    )
    policy = str(call_ending_policy or "")
    return text + "\n" + policy if policy else text
