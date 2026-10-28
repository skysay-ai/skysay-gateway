# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Azure AI Speech adapter construction: metered identity and SSML safety.

One module for both components rather than the ``<provider>_tts.py`` shape its
neighbours use, because Azure is the first provider whose STT and TTS share a
credential, a locale convention, and -- the reason this file is not optional --
the same defect in how the pinned plugin labels its usage metrics.

**The metric identity is the billing identity.** ``component_billing`` compares
every SDK usage record against the exact ``(provider, model)`` pinned in the
call's billing contract and raises rather than guessing, which strands the
caller's wallet reservation until an operator releases it. The pinned Azure
plugin reports ``provider = "Azure STT"`` with ``model = "unknown"`` for
transcription, and ``provider = "Azure TTS"`` with ``model =
"<region>:<voice-id>"`` for synthesis -- neither of which is
``azure``/``realtime`` or ``azure``/``neural``. Every Azure call would have
settled into that refusal. The subclasses below report the catalog identity
instead, and a test drives a usage record carrying exactly what they report
through the settlement path.


Azure's synthesis endpoint takes an SSML *document*, not a text field, and the
pinned LiveKit plugin builds that document by interpolating the reply straight
into it (``ssml += self.input_text`` in ``ChunkedStream._build_ssml``).  Any
``&``, ``<`` or ``>`` an LLM writes therefore leaves the document malformed,
Azure answers 400, LiveKit retries the same body twice more, and the turn ends
with the caller hearing nothing at all -- the silent-line failure mode this
repository has already paid for twice (eleven_v3 on the realtime websocket,
and Cartesia's retired voice payload).

It is not a hypothetical for THIS launch customer: a collections agent reads
company names and amounts back to the caller, and ``&`` is ordinary in a
company name.  So escape the five XML metacharacters before the plugin builds
its document, and restore the original text immediately afterwards so LiveKit's
own ``characters_count`` metric -- which is what settlement bills this provider
on -- still counts the characters the customer actually asked for rather than
the entity expansion of them.

Billing is not that metric: Microsoft bills the escaped body with Han
characters doubled, so the stream hands ``billable_characters`` of the escaped
text to the call's native unit meter when aiohttp writes the request body, and
the capability's request-billing data decides what each ending is charged.
"""
from __future__ import annotations

import contextvars
import re
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # annotation only; ``from __future__ import annotations`` keeps it a string
    from .native_tts_usage import NativeTTSUnitMeter


def _native_tts_usage() -> ModuleType:
    """The synthesis-only billing module, imported when a TTS path needs it.

    Never at module import: the isolated workflow recognizer
    (``workflow_azure_listener_worker``) executes THIS file by path under
    ``python -I`` to build its STT adapter. Isolation puts no worker directory
    on ``sys.path``, and the recognizer refuses any other worker module it
    finds in ``sys.modules`` afterwards, so an eager import here failed every
    recognition at CONFIG, before Azure was contacted. The STT path
    must keep needing nothing from this directory.
    """

    try:  # package import in the API image/tests
        from . import native_tts_usage as module
    except ImportError:  # pragma: no cover - direct voice-worker entrypoint
        import native_tts_usage as module
    return module


#: The catalog identity a metered Azure call settles against. These are the
#: ``(component, provider, model)`` keys in the capability artifact, and the
#: values the adapters below report to LiveKit's metrics collector so the two
#: cannot drift.
CATALOG_PROVIDER = "Azure"
CATALOG_STT_MODEL = "realtime"
CATALOG_TTS_MODEL = "neural"


class AzureTTSRuntimeCompatibilityError(RuntimeError):
    """The pinned plugin cannot safely build Azure's SSML request."""


# ``&`` first: escaping it after the others would re-escape the ampersands
# those replacements just introduced.
#
# Three characters, not five. Quotes were escaped too until the billable-
# character meter arrived and made over-escaping cost real money: the text goes
# into element CONTENT, where a quote is legal, and Microsoft bills the markup
# in the request body -- so turning every apostrophe in "don't" into
# ``&apos;`` charged the customer five characters for nothing. If a future
# plugin release moves the text into an attribute, escape there rather than
# here, where the expansion is billed on every ordinary English reply.
_SSML_ESCAPES = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
)


#: Azure's region grammar, and the reason it is checked rather than trusted:
#: the region IS the endpoint host, and both pinned plugins defer building it
#: -- TTS until the first synthesis, STT until ``SpeechConfig`` is created at
#: stream time. So a value like ``West Europe`` (a real thing to type: it is
#: how the portal displays ``westeurope``) passes every readiness check this
#: worker runs, the caller is admitted, and the failure lands on the first
#: spoken turn instead of before media.
#:
#: The same grammar is enforced in the control plane's preview and inventory
#: paths, which live in a different image and cannot share this constant. The
#: test drives both over one list for that reason -- and the list has to
#: contain the values they DISAGREE about, or it proves nothing: a first
#: attempt at this spelled the worker's copy ``[a-z0-9]{3,40}`` and passed,
#: because every region anyone would think to type is accepted by both. A
#: leading digit and a 33-character name are not, and those are the shapes a
#: divergence actually ships as.
AZURE_REGION = re.compile(r"^[a-z][a-z0-9]{1,31}$")


def azure_region(region: str) -> str:
    """Return the region having proved its shape, or refuse before media."""

    candidate = str(region or "").strip()
    if AZURE_REGION.fullmatch(candidate) is None:
        raise AzureTTSRuntimeCompatibilityError("Azure speech region is not a region name")
    return candidate


#: XML 1.0 allows exactly three C0 controls. Every other code point below
#: U+0020, plus the two permanently-unassigned noncharacters at the end of the
#: BMP, cannot appear in a well-formed document AT ALL -- not escaped, not as a
#: character reference. Azure answers such a document with 400 and the caller
#: hears silence where the greeting was.
#:
#: Stripped rather than refused, deliberately. A stray vertical tab pasted into
#: a greeting is a typo, not a reason for an agent to stop speaking, and the
#: alternative fails the call at exactly the moment the customer is listening.
#: Stripping happens before the meter, so nobody is billed for characters that
#: were never sent.
_XML_FORBIDDEN = re.compile(
    "[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]"
)


def escape_ssml_text(text: str) -> str:
    """Return ``text`` safe to interpolate into an SSML element body.

    Exactly the three characters element content cannot hold. Quotes are left
    alone: they are legal there, and escaping them is not free -- the meter
    below counts the escaped body because that is what Microsoft bills, so
    ``&apos;`` would charge five characters for one apostrophe on every
    ordinary English reply.
    """
    if not isinstance(text, str):
        raise AzureTTSRuntimeCompatibilityError("Azure TTS requires text input")
    # Before the escapes: a character XML cannot hold is not made holdable by
    # escaping it, and one of them anywhere in the body voids the whole
    # document rather than the character.
    text = _XML_FORBIDDEN.sub("", text)
    for character, entity in _SSML_ESCAPES:
        text = text.replace(character, entity)
    return text


#: Azure's documented short-name shape. Wider than the obvious
#: ``{language}-{REGION}-{Name}`` in three ways, each of which is in the live
#: catalogue rather than hypothetical:
#:
#:   * a script subtag -- ``iu-Cans-CA-SiqiniqNeural``, ``sr-Latn-RS-SophieNeural``;
#:   * hyphens inside the name -- ``zh-CN-shaanxi-XiaoniNeural``;
#:   * a colon before a model suffix -- ``en-US-Ava:DragonHDLatestNeural``.
#:
#: A narrower pattern is not a safe default here. The inventory importer does
#: no shape validation, so a rejected identifier still reaches the customer's
#: voice library and is refused later by the Listen button and by call
#: construction -- a voice you can pick and cannot use. Six GA voices in the
#: account's own listing were in exactly that state before this widened.
#:
#: What it still refuses is what an XML attribute cannot hold: quotes, angle
#: brackets, ampersands and whitespace. A test drives every identifier in a
#: real ``voices/list`` capture through both copies of this pattern.
_AZURE_VOICE_ID = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z]{4})?-[A-Za-z]{2}-[A-Za-z0-9][A-Za-z0-9:-]{0,60}$")


def azure_speaking_locale(voice_id: str, selected_locale: str) -> str:
    """The tag to open the document with for THIS voice and THIS selection.

    The selected language always wins -- that is the invariant this provider
    was built around, and reading a language off a voice name is the defect
    that would have spoken German for a Tamil agent. What this adds is one
    narrower question the artifact cannot answer: WHICH REGION of the selected
    language.

    Azure names about 540 monolingual voices for an exact locale, and the
    inventory advertises each under its base language, so ``en-GB-RyanNeural``
    is offered for ``en`` while the artifact's table says ``en`` means
    ``en-US``. That contradiction is ours, not the customer's: the voice is
    monolingual and will speak British English whatever the root tag says, and
    measurement against the live account confirms a mismatched root changes
    the audio while the duration and the 200 stay the same. So when the voice
    names a region of the very language that was selected, the voice's own
    locale is the truthful tag.

    It is applied ONLY when the languages already agree, which is what keeps
    it from becoming the name-reading defect: a multilingual ``de-DE`` voice
    selected for Tamil still opens in the artifact's ``ta-IN``, because ``de``
    is not ``ta``.
    """

    artifact = str(selected_locale or "").strip()
    if not artifact:
        raise AzureTTSRuntimeCompatibilityError("Azure TTS requires a resolved locale")
    try:
        identity = azure_voice_identity(voice_id)
    except AzureTTSRuntimeCompatibilityError:
        return artifact
    own, separator, _ = identity.rpartition("-")
    if not separator or not own:
        return artifact
    if own.split("-")[0].lower() != artifact.split("-")[0].lower():
        return artifact
    return own


def azure_voice_identity(voice_id: str) -> str:
    """Return the voice identifier, having proved its shape.

    Shape only. This deliberately does NOT derive a language from the
    identifier, and that is the correction: Azure's multilingual voices speak
    dozens of locales, so ``de-DE-SeraphinaMultilingualNeural`` reading Tamil
    is an ordinary selection rather than a mismatch, and reading the tag off
    the name would have sent it as German. The tag comes from the language the
    customer selected, resolved through the artifact's locale table.

    What the shape check IS for: the identifier is interpolated into an SSML
    attribute one call below, and the matched string is returned so that
    "validated" and "used" are the same value -- a later relaxation of the
    pattern cannot widen what reaches the document without also widening what
    was checked.
    """
    match = _AZURE_VOICE_ID.fullmatch(str(voice_id or ""))
    if match is None:
        raise AzureTTSRuntimeCompatibilityError("Azure TTS voice id is not a recognisable identity")
    return match.group(0)


#: Code-point ranges Microsoft bills at two characters each: "Each Chinese
#: character is counted as two characters for billing, including kanji used in
#: Japanese, hanja used in Korean, or hanzi used in other languages."
#: (learn.microsoft.com/azure/ai-services/speech-service/text-to-speech,
#: "Billable characters", read 2026-08-31.)
#:
#: These are the Han blocks only. Japanese kana and Korean Hangul syllables
#: appear inside the same writing systems but are not Han, and Microsoft's
#: sentence names only the Han-derived scripts -- so they are counted singly.
#: If that reading is ever shown to be wrong the correction is one range, and
#: the direction of the error is under-billing Skysay rather than
#: over-billing a customer.
_DOUBLE_BILLED_RANGES: tuple[tuple[int, int], ...] = (
    (0x2E80, 0x2EFF),    # CJK radicals supplement
    (0x3400, 0x4DBF),    # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),    # CJK unified ideographs
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0x20000, 0x2FA1F),  # CJK extensions B-F, I, and compatibility supplement
    (0x30000, 0x323AF),  # CJK extensions G and H
)


def billable_characters(ssml_text: str) -> int:
    """Microsoft's billable-character count for one synthesis request.

    NOT ``len(text)``, and the difference is money. Microsoft bills "the total
    number of characters in each successfully processed request", counting
    letters, punctuation, spaces, tabs, **markup**, and every Unicode code
    point, with each Han character counted twice. Only the ``<speak>`` and
    ``<voice>`` tags are exempt.

    Two consequences the generic counter cannot express:

    * The XML escaping this module applies is BILLABLE. ``&`` leaves as
      ``&amp;`` and Microsoft charges five characters for it, because entity
      markup is markup within the text field. Counting the pre-escape text
      would under-report every reply containing an ampersand.
    * LiveKit's own ``TTSMetrics.characters_count`` is ``len(self._input_text)``
      -- the submitted length, single-counted, unescaped. For an English reply
      the two agree; for Chinese they differ by a factor of two. Settlement
      therefore takes this number and ignores that one.

    Passed the SSML *body text* rather than the whole document, so the exempt
    tags are never counted: the caller owns the document and knows which part
    is the text field.
    """
    if not isinstance(ssml_text, str):
        raise AzureTTSRuntimeCompatibilityError("Azure TTS requires text input")
    total = 0
    for character in ssml_text:
        point = ord(character)
        total += 2 if any(low <= point <= high for low, high in _DOUBLE_BILLED_RANGES) else 1
    return total


#: The billable characters of the SSML body the current attempt just built;
#: read by the metered session when the same task posts that body.
_BILLABLE_BODY: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "skysay_azure_billable_body", default=None
)


def _azure_post_body(arguments: dict[str, Any]) -> tuple[bytes, str | None, int, dict[str, Any]]:
    """The SSML document exactly as the plugin posts it, and its billable units."""

    body = arguments.pop("data", None)
    billable = _BILLABLE_BODY.get()
    if not isinstance(body, str) or billable is None:
        raise AzureTTSRuntimeCompatibilityError("Azure synthesis request has no metered SSML body")
    # The plugin sends its own Content-Type header; the payload must not add one.
    return body.encode("utf-8"), None, billable, arguments


def ssml_escaped_chunked_stream(base: type, metered: bool = False) -> type:
    """Wrap one verified plugin stream class so its SSML body is well formed.

    Subclassing rather than patching the module: Azure's ``TTS.synthesize`` is
    the only construction site for this class and it is a plain method, so the
    override below reaches every synthesis without the process-global patching
    Cartesia's retired wire format needed.
    """

    if not callable(getattr(base, "_build_ssml", None)):
        raise AzureTTSRuntimeCompatibilityError("Azure plugin does not expose an SSML builder")
    usage = _native_tts_usage() if metered else None

    class SsmlEscapedChunkedStream(base):  # type: ignore[misc, valid-type]
        def _build_ssml(self) -> str:
            original = self._input_text
            escaped = escape_ssml_text(original)
            # The ESCAPED text is what the request body carries, so it is what
            # Microsoft bills. Recorded here because this is the only place
            # the exact body text exists; the metered session reads it when
            # this same attempt posts the document.
            _BILLABLE_BODY.set(billable_characters(escaped))
            if escaped == original:
                return super()._build_ssml()
            # Restored in ``finally`` rather than left escaped: LiveKit's base
            # class reads this same attribute for ``characters_count``, and a
            # retry calls this method again, so a permanent replacement would
            # both over-count the entity expansion and double-escape on the
            # second attempt.
            self._input_text = escaped
            try:
                return super()._build_ssml()
            finally:
                self._input_text = original

        async def _run(self, output_emitter: Any) -> None:
            meter = getattr(self._tts, "_skysay_meter", None)
            if not metered or meter is None:
                await super()._run(output_emitter)
                return
            # One attempt, one metered request: its units are handed when the
            # body is written, and how it ends -- including a turn the caller
            # interrupts after Microsoft accepted it -- is charged by data.
            try:
                await usage.run_metered_request(
                    meter, lambda: super(SsmlEscapedChunkedStream, self)._run(output_emitter)
                )
            except usage.TTSMeterFrozen:
                # Usage evidence was taken at teardown: close like a client
                # that hung up rather than inviting LiveKit to retry.
                from livekit.agents import APIStatusError

                raise APIStatusError(
                    "TTS usage evidence was already taken", status_code=499, request_id=None, body=None
                ) from None

    return SsmlEscapedChunkedStream


def _plugin_module(*, component: str = "tts") -> ModuleType:
    try:
        from livekit.plugins import azure
    except ImportError as exc:  # pragma: no cover - exercised in the worker image
        raise AzureTTSRuntimeCompatibilityError("Azure LiveKit plugin is unavailable") from exc
    anchor = azure.STT if component == "stt" else azure.TTS
    module = sys.modules.get(anchor.__module__)
    if not isinstance(module, ModuleType):
        raise AzureTTSRuntimeCompatibilityError("Azure plugin module is unavailable")
    return module


def _drop_sends_after_close(channel: Any) -> None:
    """Make a send into this stream's closed event channel do nothing.

    The pinned plugin's recognizer callbacks run on the Azure SDK's own thread
    and hand each event to the loop as
    ``call_soon_threadsafe(self._event_ch.send_nowait, event)``; their
    ``suppress(RuntimeError)`` covers only a closed loop. The base stream
    closes that channel when its task ends, so an event the SDK delivers after
    a hangup closed the stream raised ``ChanClosed`` inside the loop callback,
    which asyncio reports as "Exception in callback Chan.send_nowait(...)".

    This changes nothing but that case: a closed channel is exactly the
    condition under which the plugin's send raises, so the event was never
    going to be delivered. A send into an open channel, and every other
    failure of one, is the plugin's own. The callbacks look the method up on
    the channel at call time, which is why replacing it on this instance
    reaches them; the stream's own iterator reads the channel itself, not this
    attribute.
    """

    send = channel.send_nowait

    def send_unless_closed(value: Any) -> None:
        if channel.closed:
            return
        send(value)

    channel.send_nowait = send_unless_closed


def build_stt(
    *,
    language: str = "",
    speech_key: str,
    speech_region: str,
    candidate_locales: Sequence[str] | None = None,
) -> Any:
    """Construct the Azure LiveKit STT adapter under its catalog identity.

    The only thing this adds to the plugin is the identity its usage metrics
    carry. Upstream reports the literal string ``"unknown"`` as the model,
    which settlement cannot match against anything, so an Azure call would
    reach the end of its life and fail to report -- holding the reservation.

    ``candidate_locales`` is the multi-language form and is exclusive with
    ``language``: exactly one of the two decides what the recognition socket
    listens for, and accepting both would leave the answer to argument order.
    The plugin turns a list of more than one tag into continuous language
    identification by itself -- it sets
    ``SpeechServiceConnection_LanguageIdMode`` to ``Continuous`` and builds an
    ``AutoDetectSourceLanguageConfig`` -- which is why this passes a list
    rather than reaching for the SDK's own configuration objects. The
    two-tag minimum is enforced here rather than left to that branch: a
    one-entry list takes the plugin's single-language path silently, and the
    call would listen for one language while the customer was told two.
    """

    if not str(speech_key or "").strip() or not str(speech_region or "").strip():
        raise AzureTTSRuntimeCompatibilityError("Azure STT requires a subscription key and region")
    tag = str(language or "").strip()
    if candidate_locales is None:
        if not tag:
            raise AzureTTSRuntimeCompatibilityError("Azure STT requires an explicit locale")
        requested: str | list[str] = tag
    else:
        if tag:
            raise AzureTTSRuntimeCompatibilityError(
                "Azure STT takes either one locale or a candidate set, never both"
            )
        tags = [str(item or "").strip() for item in candidate_locales]
        if len(tags) < 2 or not all(tags):
            raise AzureTTSRuntimeCompatibilityError(
                "Azure STT language identification requires at least two candidate locales"
            )
        if len(set(tags)) != len(tags):
            raise AzureTTSRuntimeCompatibilityError("Azure STT candidate locales must be distinct")
        requested = tags
    module = _plugin_module(component="stt")
    stt_class = getattr(module, "STT", None)
    if not isinstance(stt_class, type):
        raise AzureTTSRuntimeCompatibilityError("Azure plugin does not expose an STT class")

    class MeteredSTT(stt_class):  # type: ignore[misc, valid-type]
        @property
        def model(self) -> str:
            return CATALOG_STT_MODEL

        def stream(self, *args: Any, **kwargs: Any) -> Any:
            # The plugin's own ``stream()``, so the stream stays registered
            # where ``force_finalization`` below reaches it.
            stream = super().stream(*args, **kwargs)
            _drop_sends_after_close(stream._event_ch)
            return stream

        @property
        def provider(self) -> str:
            return CATALOG_PROVIDER

        def force_finalization(self) -> None:
            """Make the plugin emit the finals it is holding, now.

            The pinned plugin has no flush: its ``_FlushSentinel`` only
            reports billing usage, and the one thing that makes the Azure
            SDK finalize buffered audio is closing the push stream. The
            plugin's run loop does exactly that on a reconnect -- close the
            stream ("to flush finals for buffered audio before teardown",
            in its own words), wait for the session to stop, rebuild the
            recognizer -- and ``update_options()`` with nothing to change is
            the public way to request one. It reaches every live stream this
            adapter has opened; a stream that has already ended ignores it.

            Frames pushed while the recognizer is rebuilt are queued on the
            stream's input channel and reach the new recognizer (measured
            against the pinned plugin, not assumed). What the rebuild does
            lose is the recognizer's context at the boundary, which is why
            the caller of this method only fires it after the SDK's own
            timer has reported silence -- see ``transcription_stall.py``.
            """

            self.update_options()

    return MeteredSTT(
        speech_key=speech_key, speech_region=speech_region, language=requested
    )


def build_tts(
    *,
    voice: str,
    language: str,
    sample_rate: int,
    speech_key: str,
    speech_region: str,
    native_unit_meter: NativeTTSUnitMeter | None = None,
) -> Any:
    """Construct the Azure LiveKit TTS adapter with a well-formed SSML body.

    ``language`` is the tag the SELECTED language resolves to, handed in by the
    caller from the artifact's locale table -- never inferred here. It reaches
    Azure as the SSML ``xml:lang``, which is the control that makes a
    multilingual voice speak the language the customer chose; the plugin
    defaults it to ``en-US`` when unset, so leaving it out is not neutral.
    """

    if not str(speech_key or "").strip() or not str(speech_region or "").strip():
        raise AzureTTSRuntimeCompatibilityError("Azure TTS requires a subscription key and region")
    if not str(language or "").strip():
        raise AzureTTSRuntimeCompatibilityError("Azure TTS requires an explicit locale")
    module = _plugin_module()
    base = getattr(module, "ChunkedStream", None)
    tts_class = getattr(module, "TTS", None)
    if not isinstance(base, type) or not isinstance(tts_class, type):
        raise AzureTTSRuntimeCompatibilityError("Azure plugin does not expose compatible classes")
    usage = _native_tts_usage() if native_unit_meter is not None else None
    if native_unit_meter is not None and (
        not isinstance(native_unit_meter, usage.NativeTTSUnitMeter) or native_unit_meter.billing.unit != "character"
    ):
        raise AzureTTSRuntimeCompatibilityError("Azure TTS requires a billable-character native unit meter")
    stream_class = ssml_escaped_chunked_stream(base, metered=native_unit_meter is not None)
    # LiveKit's ``StreamAdapter`` -- which is what wraps a non-streaming TTS
    # inside an ``AgentSession`` -- always passes ``conn_options``, but the
    # public method has a default and a direct caller may omit it. Read that
    # default off the plugin module rather than importing a second copy of it.
    default_conn_options = getattr(module, "DEFAULT_API_CONNECT_OPTIONS", None)
    if default_conn_options is None:
        raise AzureTTSRuntimeCompatibilityError("Azure plugin does not expose connect options")

    class SsmlSafeTTS(tts_class):  # type: ignore[misc, valid-type]
        _skysay_meter = native_unit_meter

        def synthesize(self, text: str, **kwargs: Any) -> Any:
            kwargs.setdefault("conn_options", default_conn_options)
            return stream_class(tts=self, input_text=text, **kwargs)

        def _ensure_session(self) -> Any:
            session = super()._ensure_session()
            if native_unit_meter is None:
                return session
            return usage.MeteredHTTPSession(session, native_unit_meter, _azure_post_body)

        @property
        def model(self) -> str:
            # Upstream reports "<region>:<voice-id>" here, which is useful
            # diagnostics and unusable as a billing identity: the voice is a
            # per-agent choice and the region a deployment one, while
            # settlement pins one model per rate row.
            return CATALOG_TTS_MODEL

        @property
        def provider(self) -> str:
            return CATALOG_PROVIDER

    return SsmlSafeTTS(
        voice=azure_voice_identity(voice),
        language=language,
        sample_rate=sample_rate,
        speech_key=speech_key,
        speech_region=speech_region,
    )
