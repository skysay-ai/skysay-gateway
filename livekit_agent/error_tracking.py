# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Crash reporting for the voice worker, with customer data left behind.

An unhandled exception inside a job process does not stop the container: the
worker keeps its LiveKit registration, the call it was serving goes silent, and
the five-minute host monitor sees a healthy process. Nothing on this host turns
that into a signal, which is why this module exists.

Three properties are load-bearing, and each is pinned by
``tests/test_error_tracking_install.py`` and
``tests/test_error_tracking_scrubbing.py``:

* **Starting the worker never depends on the error backend.** A wrong DSN, a
  dead host or a missing SDK returns ``False`` from
  :func:`install_error_tracking`; it never raises. The backend observes this
  process, it is never a precondition for it.
* **The audio path is never blocked.** ``traces_sample_rate`` is 0 (a span
  envelope per call is pure cost for a process whose budget is milliseconds),
  the SDK transport owns its own background thread with a bounded queue that
  DROPS rather than blocks, and nothing here calls ``flush()``. The only
  bounded wait is the atexit flush, capped by ``shutdown_timeout``.
* **Customer data does not leave this process.** Local variables are off,
  ``send_default_pii`` is off, and every event AND every breadcrumb is walked
  before it reaches the transport. Breadcrumbs matter as much as the event
  body: a LiveKit room name and a SIP participant identity both embed the
  dialed number, and they reach breadcrumbs through ordinary INFO logging long
  before anything fails.

The standard to meet is the one
``infra/voice-host/skysay-sip-call-statistics-capture`` already holds
itself to: the dialed number, the SIP ``Call-ID`` and the participant name do
not reach a file this host writes. They must not reach a request it sends
either.

Where this runs matters. On Linux the agent framework starts job processes
through a forkserver, so a job process inherits nothing from the entrypoint's
``__main__`` block — it is the ``setup_fnc`` that runs in the process where a
call actually executes. Both call :func:`install_error_tracking`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

# --- shared scrubbing contract (start) -------------------------------------
# Everything between these two markers is byte-identical in the voice worker's
# ``error_tracking`` module and the control plane's, and
# ``tests/test_error_tracking_parity.py`` fails if it stops being. The two
# runtimes cannot import one copy: the worker image is a deliberate file
# closure with no control-plane package in it, and the API image asserts at
# build time that it cannot import the worker tree. So the duplication is
# deliberate, and the test rather than an import is what keeps the rules the
# same. (The paths are deliberately not written out: this file ships in the
# self-host export, which may not carry monorepo paths.)

#: What replaces anything these rules will not let out.
REDACTED = "[redacted]"

#: Bounds on the walk. A scrubber runs in the thread that raised, so it is
#: written to be cheap and total rather than exhaustive: past these limits the
#: structure is summarised instead of traversed.
MAX_DEPTH = 12
MAX_ITEMS = 50
MAX_TEXT = 1024

#: How much of one string the rules will read. The rules are linear, so this
#: is not about their cost but about refusing to scan a pathological value at
#: all; no real field in this codebase comes close to it.
MAX_SCAN = 64 * 1024

#: A ceiling on the WHOLE walk, not on one container. ``MAX_ITEMS`` bounds a
#: single mapping and ``MAX_DEPTH`` a single chain, but 50 items at each of 12
#: levels is 50**12 nodes: a nested payload logged as an ``extra`` -- a parsed
#: provider response, a webhook body -- would make the scrubber itself the
#: thing that stalls the thread that raised. Past this count the remainder is
#: summarised rather than traversed, which is the safe direction: a node never
#: visited is a node never sent.
#:
#: The number is set an order of magnitude above a REAL event rather than just
#: above zero, because running out mid-walk does not truncate an event, it
#: garbles one: the next key becomes the string "[redacted]", so
#: ``breadcrumbs`` stops being ``{"values": [...]}`` and the backend rejects
#: the whole report — at exactly the deep-stack moment worth reporting. A real
#: control-plane event with source context on, a 90-frame stack and the
#: breadcrumb ring full measures about 950 nodes, and the SDK's own
#: ``max_stack_frames`` ceiling of 100 bounds the frame half of that. 20,000
#: leaves an order of magnitude of headroom, which the suite asserts rather
#: than assumes, while a hostile payload still cannot pass.
MAX_NODES = 20_000

#: Keys removed outright rather than redacted. ``vars`` is a stack frame's
#: local variables: ``include_local_variables=False`` should already keep it
#: away, and this is the second lock on the door that phone numbers, secrets
#: and transcripts would otherwise walk through. The rest are request and
#: breadcrumb containers with no safe projection.
DROPPED_KEYS = frozenset({"vars", "cookies", "headers", "query_string", "env"})

#: Substrings that make a key's value a credential whatever it looks like.
#: Taken from ``skysay_common.runtime.SECRET_KEY_PARTS`` with two
#: deliberate differences: bare ``session`` is not here (it would blank
#: ``session_id`` and ``session_status``, which carry no secret and are most of
#: what makes an event readable), and ``signature``/``cookie`` are.
#:
#: Matching is SUBSTRING, and the cost is known and accepted: ``max_tokens``
#: and ``token_count`` are LLM counters, not credentials, and they come back
#: redacted. Every rule that would free them — a segment match, a word
#: boundary — also frees ``apitoken`` and ``gwtoken``. Losing a counter from a
#: crash report is cheap; the other direction is not.
SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "credential",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "one_time",
    "otp",
    "secret",
    "session_key",
    "session_token",
    "headless_session",
    "signature",
    "token",
)

#: The segment a producer uses to declare, in the key itself, that a value is
#: personal data. The pinned ``livekit-agents`` SDK writes the convention down
#: in ``livekit/agents/telemetry/trace_types.py``: an attribute carrying
#: conversational content "must include a dot-delimited ``pii`` segment
#: (``lk.pii.<name>``)". When the PRODUCER has already labelled a value, the
#: only correct thing to do with it is refuse it — whatever the value looks
#: like, and whatever vocabulary below does or does not know the name after the
#: label.
#:
#: Matching is SUBSTRING, like :data:`SECRET_KEY_PARTS` and for the same
#: reason: a segment match or a word boundary would free ``piiText``,
#: ``user-pii`` and every other spelling a producer might pick, and this is a
#: label, not a guess. The cost is that an unrelated key containing these three
#: letters comes back redacted; losing such a key from a crash report is cheap
#: and the other direction is not.
PII_KEY_PART = "pii"

#: Keys whose value is Customer Content: what a caller said, what an agent
#: said, what a message carried, what a customer configured an agent to say.
#: No pattern can recognise free text, so the key is the only handle on it.
CONTENT_KEYS = frozenset(
    {
        "agent_text",
        "body",
        "caller_text",
        "content",
        "first_message",
        "greeting",
        "instruction",
        "instructions",
        "message_body",
        "note",
        "notes",
        "prompt",
        "prompts",
        "sms_body",
        "speech",
        "summary",
        "system_prompt",
        "text",
        "transcript",
        "transcripts",
        "turns",
        "utterance",
        "utterances",
    }
)

#: Keys that identify a human or the call they are on. ``room`` and
#: ``participant`` are here because the SIP bridge builds both names out of the
#: dialed number; ``from``/``to`` because that is what every carrier payload in
#: this codebase calls the two ends of a call.
IDENTIFIER_KEYS = frozenset(
    {
        "call_id_header",
        "called",
        "called_did",
        "caller",
        "caller_id",
        "callerid",
        "destination",
        "did",
        "e164",
        "email",
        "from",
        "from_e164",
        "identity",
        "msisdn",
        "number",
        "participant",
        "participant_identity",
        "phone",
        "phone_number",
        "phonenumber",
        "recipient",
        "room",
        "room_name",
        "sip_call_id",
        "sip_from",
        "sip_to",
        "sip_uri",
        "to",
        "to_e164",
        "trunk",
        "trunk_host",
    }
)

# Text rules, applied in the order written. Order matters: a credential is
# taken out before the number rules can nibble digits off it, and an address is
# taken whole before its local part is read as a number.

_AUTHORIZATION = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{4,}")
#: This repository's own API-key and gateway-token prefixes -- current
#: (``sky_``/``skygw_``) and prior (``tai_``/``tapi_``/``tgw_``, kept because
#: verification is by hash and issued keys of every generation keep working)
#: -- plus Stripe's webhook-secret prefix (``whsec_``), which really is
#: configured here. A letters-only secret matches no other rule in this file:
#: it has no digit run for the number branch and no assignment for
#: ``_ASSIGNED_SECRET``, so this prefix is the ONLY thing standing between it
#: and the wire.
#: The floor is 4, not 8, and matches ``_AUTHORIZATION``'s: a MINTED key is
#: long, but a short one quoted back in an error (``whsec_a1b2c3``) clears no
#: other rule here — no digit run, no assignment, no hex-16 shape. The prefixes
#: are specific enough that a false positive is not a real shape.
# ``opsess_`` is the dashboard session bearer (services/orgs.py). It is letters
# and a prefix, the same shape as ``whsec_``: no digit run, no assignment, so
# without this alternative a session token quoted into an event — or placed in
# ``user.id`` — would leave unchanged.
_SKYSAY_KEY = re.compile(r"\b(?:sky|tai|tapi|skygw|tgw|gw|whsec|opsess)_[A-Za-z0-9_-]{4,}")
#: ``token=abc``, ``api_key: "abc"``, ``secret=abc`` — the assignment forms a
#: log line or an exception message puts a credential in.
#: The leading ``[a-z0-9_]*`` is what makes this fire inside a COMPOUND name.
#: ``\btoken\b`` does not match in ``refresh_token=`` — ``_`` is a word
#: character, so there is no boundary in front of ``token`` — and a
#: letters-only value there clears the prefix, number and address rules too.
#: The trailing ``\b`` is kept, which is what still leaves ``max_tokens=5``
#: alone: the alternation must end the word.
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b([a-z0-9_]*(?:api[_-]?key|apikey|token|secret|password|passwd|signature"
    r"|credential|authorization|nonce))\b\s*[=:]\s*[\"\']?[^\s\"\',;&)]+"
)
#: SIP Digest authentication, the half the vocabulary above cannot take.
#: ``nonce`` is in it (and the ``[a-z0-9_]*`` prefix covers ``cnonce`` and
#: ``nextnonce``); ``response`` deliberately is NOT, because a bare
#: ``response=200`` is a status. The digest ``response`` is the credential
#: itself — an MD5 over the trunk password and the nonce printed beside it, so
#: shipping both leaves an offline dictionary attack as the only step. Nothing
#: else in this file takes it: the opaque-identifier branch below deliberately
#: KEEPS a run of 16 or more hex characters, which is exactly this shape.
#: Bounded to that shape, so ``response=ok`` still travels.
_DIGEST_RESPONSE = re.compile(r"(?i)\b(response)\s*[=:]\s*[\"\']?[0-9a-fA-F]{16,}")
#: A URL's query string carries keys, tokens and phone numbers as parameters.
#: The path is kept: it is what makes the breadcrumb worth having.
_URL_QUERY = re.compile(r"(?i)\b(https?://[^\s?#\"\']+)\?[^\s#\"\']*")
#: The same, for the RELATIVE form. A browser fetch breadcrumb records the URL
#: as the caller wrote it, and in this app that is almost always
#: ``/v1/...``; without this rule the absolute-URL rule above never fires and
#: ``?to=%2B372...`` goes out intact.
_PATH_QUERY = re.compile(r"(?<![\w:/])(/[^\s?#\"\'<>]*)\?[^\s#\"\'<>]*")
#: ``https://token@host/path`` — credentials in the authority, which the query
#: rule above keeps because it keeps everything before the ``?``.
_URL_USERINFO = re.compile(r"(?i)(https?://)[^\s/@\"\']+@")
#: A SIP or tel URI is a phone number with a host attached.
_SIP_URI = re.compile(r"(?i)\b(sips?|tel):[^\s<>\"\',;]+")
#: An address. Both halves are BOUNDED, and the bound is an availability
#: control, not tidiness: ``[A-Za-z0-9._%+-]+@`` backtracks quadratically over
#: a long run of local-part characters that never reaches an ``@``, and a
#: 64 KB base64url token — the ceiling :data:`MAX_SCAN` allows — measured
#: **3.7 seconds** on the thread that logged it, which in the voice worker is
#: the audio loop. The bounds are RFC 5321's own: 64 octets of local part, 255
#: of domain. The same input now measures 6.7 ms in this rule and 15 ms through
#: the whole set.
#:
#: What the bound costs, stated rather than implied: an address that BREAKS
#: those limits is no longer replaced whole. Past 64 local-part characters the
#: match starts later and a leading fragment survives; past 255 domain
#: characters the rule matches nowhere and the whole string survives. Neither
#: shape is a deliverable address, and the key rule still covers both — but
#: "every address goes" is true of valid ones only.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,}")
#: Numbers, in one pass so that opaque identifiers can be claimed FIRST and
#: handed back untouched. Without that branch a 32-character LiveKit id or a
#: SHA-256 digest comes back with a hole punched in the middle of it, which
#: costs exactly the debuggability this module exists to buy. The
#: ``(?![0-9]+\b)`` guard keeps a long run of pure digits — a card number, an
#: account number — out of the protected branch, and the digit branch has no
#: upper bound for the same reason: capping it at 15 let a 16-digit run fall
#: through BOTH branches untouched.
#:
#: The bare branch refuses a following digit, and a following dot only when a
#: DIGIT follows that dot: ``1757800000.123`` is one fractional timestamp and
#: must survive whole, while ``37255512345.`` is a phone number at the end of a
#: sentence and must not.
#:
#: The ``+`` branch accepts separators because a leading ``+`` is already proof
#: of intent: ``+372 5551 2345`` is a phone number however it is spaced, and an
#: unnormalised one quoted back in a validation message is exactly the shape
#: that would otherwise escape. The bare-digit branch takes no separators and
#: refuses a neighbouring ``.`` on either side, which is what keeps a dotted
#: version string (``7.35.1.20260906``) and a fractional timestamp intact.
#:
#: The ``grouped`` branch is the plus-less form a customer types into a field
#: and the server quotes back: ``372 5551 2345``, ``555-123-4567``. It fires
#: only on groups of 2 to 4 digits, each optionally in parentheses because
#: ``(555) 123-4567`` is how a North American number is printed, joined by a
#: single space or hyphen, and the
#: whole run must be 9 digits or more, which is what keeps a timestamp out of
#: it: ``2026-09-14`` is eight digits, and ``2026-09-14 02`` is refused because
#: a ``:`` may not follow the run. Dots are not separators here for the same
#: reason they bound the bare branch.
#:
#: A dot MAY follow the run, though — a number at the end of a sentence still
#: has to go, and a following dot cannot turn eight digits into nine.
_NUMBERS = re.compile(
    r"(?P<opaque>\b(?![0-9]+\b)[0-9a-fA-F]{16,}\b)"
    r"|(?P<plus>\+\d[\d ().-]{4,17}\d)"
    r"|(?P<grouped>(?<![\d.:-])\(?\d{2,4}\)?(?:[ -]\(?\d{2,4}\)?){1,4}(?![\d:-]))"
    r"|(?P<digits>(?<![\d.])\d{7,}(?!\d)(?!\.\d))"
)

#: A grouped run only counts as a number if it carries as many digits as one.
_GROUPED_MIN_DIGITS = 9


def _keep_opaque(match: "re.Match[str]") -> str:
    opaque = match.group("opaque")
    if opaque is not None:
        return opaque
    grouped = match.group("grouped")
    if grouped is not None and sum(c.isdigit() for c in grouped) < _GROUPED_MIN_DIGITS:
        return grouped
    return REDACTED


def redact_text(value: Any) -> str:
    """Rewrite one string so it carries no credential, address or number.

    The rules run BEFORE the length cap, deliberately. Cutting first
    manufactures the leak the cut claims to prevent: a value straddling the
    boundary comes back as a fragment no rule recognises any more —
    ``jane.doe@example.c`` has no TLD for the address rule, and half a phone
    number is under the digit floor. Only the WORK is bounded first, at a
    ceiling no real field reaches.
    """
    text = value if isinstance(value, str) else str(value)
    if len(text) > MAX_SCAN:
        text = text[:MAX_SCAN]
    text = _AUTHORIZATION.sub(REDACTED, text)
    text = _SKYSAY_KEY.sub(REDACTED, text)
    text = _ASSIGNED_SECRET.sub(lambda match: match.group(1) + "=" + REDACTED, text)
    text = _DIGEST_RESPONSE.sub(lambda match: match.group(1) + "=" + REDACTED, text)
    text = _URL_USERINFO.sub(lambda match: match.group(1) + REDACTED + "@", text)
    text = _URL_QUERY.sub(lambda match: match.group(1) + "?" + REDACTED, text)
    text = _PATH_QUERY.sub(lambda match: match.group(1) + "?" + REDACTED, text)
    text = _SIP_URI.sub(lambda match: match.group(1) + ":" + REDACTED, text)
    text = _EMAIL.sub(REDACTED, text)
    text = _NUMBERS.sub(_keep_opaque, text)
    if len(text) > MAX_TEXT:
        # Now a size control AND still a privacy control: a transcript that no
        # pattern recognises is bounded here, and everything a pattern DID
        # recognise was already replaced above.
        text = text[:MAX_TEXT] + "...[truncated]"
    return text


#: The same two vocabularies with the separators taken out, so a camelCase
#: spelling reaches them. ``smsBody`` lowercases to ``smsbody``, which is not
#: ``sms_body``; the browser half of this contract sees camelCase constantly,
#: and free text has NO defence but its key. Deliberately not applied to
#: :data:`SECRET_KEY_PARTS`, which matches substrings: compacting there would
#: make ``snapshot_pending`` contain ``otp``.
_COMPACT_CONTENT_KEYS = frozenset(key.replace("_", "") for key in CONTENT_KEYS)
_COMPACT_IDENTIFIER_KEYS = frozenset(key.replace("_", "") for key in IDENTIFIER_KEYS)


def _normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _key_segments(normalized: str) -> tuple[str, ...]:
    """A dotted key, and every name inside it.

    A producer namespaces its keys — ``lk.pii.transcript``, ``http.query`` —
    and a namespaced key is not the name it ends in. An exact-match vocabulary
    reads ``lk.pii.transcript`` and does not see ``transcript``, which is how a
    caller's words left this process with a person's name in them on
    2026-09-23. The WHOLE key is still checked first, so a key that genuinely
    is spelled with a dot keeps its own meaning; the segments are checked as
    well, so a name the vocabulary already refuses whole is refused inside a
    namespace too.
    """
    if "." not in normalized:
        return (normalized,)
    return (normalized, *(part for part in normalized.split(".") if part))


def is_sensitive_key(key: Any) -> bool:
    """True when the key alone settles it, whatever the value looks like."""
    normalized = _normalize_key(key)
    if PII_KEY_PART in normalized:
        return True
    for segment in _key_segments(normalized):
        if segment in CONTENT_KEYS or segment in IDENTIFIER_KEYS:
            return True
        compact = segment.replace("_", "")
        if compact in _COMPACT_CONTENT_KEYS or compact in _COMPACT_IDENTIFIER_KEYS:
            return True
    return any(part in normalized for part in SECRET_KEY_PARTS)


def scrub_structure(value: Any, *, _depth: int = 0, _budget: list[int] | None = None) -> Any:
    """Walk a payload, redacting every string and dropping sensitive keys.

    ``_budget`` is a one-element list carrying what is left of ``MAX_NODES``
    across the whole walk; callers pass nothing and get a fresh one.
    """
    if _budget is None:
        _budget = [MAX_NODES]
    if _depth > MAX_DEPTH or _budget[0] <= 0:
        return REDACTED
    _budget[0] -= 1
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # A bare number reaching here is a count, a status code, a timestamp or
        # a duration — never a dialed number, which is a string everywhere in
        # this codebase because a leading "+" and leading zeros are significant.
        return value
    if isinstance(value, Mapping):
        scrubbed: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            key_str = str(key)
            if index >= MAX_ITEMS:
                scrubbed["..."] = "[" + str(len(value) - MAX_ITEMS) + " more]"
                break
            if _normalize_key(key_str) in DROPPED_KEYS:
                continue
            if is_sensitive_key(key_str):
                scrubbed[key_str] = REDACTED
                continue
            scrubbed[key_str] = scrub_structure(item, _depth=_depth + 1, _budget=_budget)
        return scrubbed
    if isinstance(value, (list, tuple, set, frozenset)):
        # A list or tuple is indexable already; only a set has to be built.
        # Copying a million-element ``extra`` list before capping it at 50
        # would be work this walk exists to avoid.
        items = value if isinstance(value, (list, tuple)) else list(value)
        if len(items) > MAX_ITEMS:
            # Both ends, not the first 50. A stack trace is ordered
            # oldest-first, so keeping the head throws away the frame that
            # actually raised — the one line the report exists to show. The SDK
            # hands over up to ``max_stack_frames`` (100 by default), so this
            # really does fire. Breadcrumbs are ordered the same way and want
            # the same treatment: the oldest are context, the newest are cause.
            head = MAX_ITEMS // 2
            kept = list(items[:head]) + list(items[len(items) - (MAX_ITEMS - head) :])
            elided = "[" + str(len(items) - MAX_ITEMS) + " more]"
        else:
            head = len(items)
            kept = list(items)
            elided = None
        scrubbed_items: list[Any] = [
            scrub_structure(item, _depth=_depth + 1, _budget=_budget) for item in kept
        ]
        if elided is not None:
            scrubbed_items.insert(head, elided)
        return scrubbed_items
    return redact_text(value)


def _account_email(value: str) -> bool:
    """One address, with no credential prefix hiding in it.

    This is the only string that does not take :func:`redact_text`. That walk
    replaces every email, which would make the account slot useless. A prefix
    the key rule already knows (``opsess_``, ``skygw_``, ``whsec_``) is still
    refused, and so is anything that is not exactly one address.
    """
    if _SKYSAY_KEY.search(value) or _AUTHORIZATION.search(value):
        return False
    return _EMAIL.fullmatch(value) is not None


def account_identity(user: Any) -> dict[str, str] | None:
    """The signed-in account to reattach after the scrub walk.

    ``id`` takes :func:`redact_text`: a session token or a phone-shaped value
    the redactor changes is not an account id, and is omitted rather than
    sent. ``email`` is the one exception to that walk, checked by
    :func:`_account_email`. ``username``, ``name``, ``ip_address`` and every
    other key are dropped. ``send_default_pii`` stays off; this is not that.
    """
    if not isinstance(user, dict):
        return None
    kept: dict[str, str] = {}
    raw_id = user.get("id")
    if isinstance(raw_id, str) and raw_id and redact_text(raw_id) == raw_id:
        kept["id"] = raw_id
    raw_email = user.get("email")
    if isinstance(raw_email, str) and _account_email(raw_email):
        kept["email"] = raw_email
    return kept or None


def scrub_event(event: Any, hint: Any = None) -> Any:
    """``before_send``: the last thing that touches an event before transport.

    Returns ``None`` — dropping the event — if anything here raises. An event
    that could not be scrubbed is not sent unscrubbed. The account, when one
    was set on the scope, is taken off before the walk and put back after it:
    the walk would redact the email, and the walk is what still redacts that
    same email everywhere else in the event.
    """
    try:
        if not isinstance(event, dict):
            return None
        user = event.pop("user", None)
        scrubbed = scrub_structure(event)
        if not isinstance(scrubbed, dict):
            return None
        account = account_identity(user)
        if account:
            scrubbed["user"] = account
        return scrubbed
    except Exception:
        return None


def scrub_breadcrumb(crumb: Any, hint: Any = None) -> Any:
    """``before_breadcrumb``: the same walk, one crumb at a time.

    Breadcrumbs are recorded from INFO upward, so they are collected on the
    happy path and carry identifiers and URLs whether or not anything ever
    fails. A clean event body with a dirty breadcrumb trail is still a leak.
    """
    try:
        if not isinstance(crumb, dict):
            return None
        return scrub_structure(crumb)
    except Exception:
        return None

# --- shared scrubbing contract (end) ---------------------------------------


@contextmanager
def error_tracking_job_scope() -> Iterator[None]:
    """One isolation scope for one voice job.

    A warm process handles more than one job. The organization tag set inside
    this scope must not still be there when the next job starts. If the SDK
    cannot be imported, the job still runs, with nothing tagged.
    """
    try:
        import sentry_sdk
    except Exception:
        yield
        return
    with sentry_sdk.isolation_scope():
        yield


def note_error_tracking_organization(organization_id: str) -> None:
    """Tag the current job with the organization it already resolved.

    Not a Sentry user. The value has to survive :func:`redact_text`.
    """
    org = str(organization_id or "").strip()
    if not org or redact_text(org) != org:
        return
    try:
        import sentry_sdk

        if sentry_sdk.get_client().is_active():
            sentry_sdk.get_isolation_scope().set_tag("organization_id", org)
    except Exception:
        return


class BrowserDemoErrorContext:
    """Job-owned diagnostic identity, also usable by its later shutdown task."""

    _PHASES = frozenset({
        "dispatch", "profile", "setup", "session_start", "opening",
        "conversation", "cleanup",
    })

    def __init__(self) -> None:
        self.session_id = ""
        self.phase = "dispatch"
        self._reported: list[BaseException] = []

    def bind_session(self, session_id: str) -> None:
        self.session_id = (
            session_id
            if re.fullmatch(r"lds_[A-Za-z0-9]{1,16}", session_id)
            and redact_text(session_id) == session_id
            else ""
        )
        self._bind_scope(self.phase)

    def set_phase(self, phase: str) -> None:
        self.phase = phase if phase in self._PHASES else "unknown"
        self._bind_scope(self.phase)

    def _bind_scope(self, phase: str, *, reset: bool = False) -> None:
        try:
            import sentry_sdk

            if not sentry_sdk.get_client().is_active():
                return
            scope = sentry_sdk.get_isolation_scope()
            if reset:
                # isolation_scope forks both; a late callback must not inherit
                # the job that happens to invoke it, including its breadcrumbs.
                scope.clear()
                sentry_sdk.get_current_scope().clear()
            scope.set_tag("component", "browser-demo")
            scope.set_tag("phase", phase)
            scope.remove_tag("session_id")
            if self.session_id:
                scope.set_tag("session_id", self.session_id)
        except Exception:
            return

    @contextmanager
    def scope(self, *, cleanup: bool = False) -> Iterator[None]:
        with error_tracking_job_scope():
            self._bind_scope("cleanup" if cleanup else self.phase, reset=True)
            yield

    def capture(self, exc: BaseException, *, cleanup: bool = False) -> None:
        if isinstance(exc, asyncio.CancelledError) or _browser_demo_refusal(exc):
            return
        if any(previous is exc for previous in self._reported):
            return
        self._reported.append(exc)
        tags = {"component": "browser-demo", "phase": "cleanup" if cleanup else self.phase}
        if self.session_id:
            tags["session_id"] = self.session_id
        if capture_exception(exc, **tags):
            # A finalizer runs in another task, so the SDK's task-local dedupe
            # cannot suppress the framework logging that same exception later.
            try:
                exc._skysay_browser_demo_reported = True
            except Exception:
                pass


def _browser_demo_refusal(exc: Any) -> bool:
    """Only the known admission refusal, never a similarly named fault."""
    kind = type(exc)
    return kind.__name__ == "BrowserDemoRejected" and kind.__module__ in {
        "browser_demo", "apps.voice.browser_demo",
    }


# Below this line is the voice worker's own, and deliberately NOT shared: the
# control plane has no LiveKit FFI bridge, so a rule about one does not belong
# in a contract both runtimes must carry byte for byte.

#: The logger the LiveKit Rust FFI bridge writes through. Its Python name is
#: ``livekit`` -- NOT ``livekit.agents``, which is the Python SDK and reports
#: real failures this worker must keep hearing about.
FFI_LOGGER = "livekit"

#: Teardown noise, matched on what the record SAYS and not only on where it
#: came from. A track handle is invalidated when the room closes, and anything
#: still asking it for statistics gets this back. The Rust side emits it at
#: ERROR, so the logging integration turns each one into its own GlitchTip
#: issue and, for ``voice-worker``, an ntfy push -- a push that exists so a
#: human wakes up when a CALL goes silent.
#:
#: Measured before deciding it was noise: all 12 events of the error
#: tracker's issue for it came from six jobs, and three of those jobs served
#: browser tests
#: that completed normally (``call.ended / remote_party_disconnected``), while
#: the job that crashed at entry that evening produced none of them. It
#: accompanies a healthy hang-up; it is not a symptom of anything.
#:
#: All three fragments are required, and that is the whole of why this is
#: narrow: ``GetStats`` alone would drop a genuine statistics failure,
#: ``InvalidRequest`` alone would drop every other malformed FFI request, the
#: two together would still drop a live-call bug that asked for statistics on
#: the wrong KIND of handle, and the logger alone would silence the whole
#: bridge. ``handle is not a`` is the dead-handle condition itself.
#:
#: Quoted from the production record rather than from memory. The full line is
#: ``InvalidRequest("handle is not a livekit_ffi::server::room::FfiTrack")``:
#: the fully-qualified Rust type is in there, so matching on a guess like
#: "handle is not a track" would have matched nothing in production while a
#: test fixture written from the same guess stayed green.
FFI_DEAD_HANDLE_FRAGMENTS = ("GetStats", "InvalidRequest", "handle is not a")


def is_ffi_dead_handle_noise(text: Any, logger: Any) -> bool:
    """Whether this record is the FFI bridge complaining about a closed track.

    One predicate for BOTH paths: the same record arrives once as an event (the
    logging integration's ERROR capture) and again as a breadcrumb on a LATER
    event. Dropping only one of them leaves either the page or the trail.
    """

    if str(logger or "") != FFI_LOGGER:
        return False
    message = str(text or "")
    return all(fragment in message for fragment in FFI_DEAD_HANDLE_FRAGMENTS)


#: The loggers and phrases a duplex plugin and the agent framework
#: use when a provider connection goes away. Matched on what the record SAYS
#: as well as where it came from, exactly like the FFI rule above.
#: The duplex PLUGINS only. ``livekit.agents`` is deliberately absent from the
#: general rule: its records are the framework's, and phrases like "connection
#: failed" under it are not specific to a provider transport, so matching them
#: would retire genuine framework failures raised in the same window.
PROVIDER_TEARDOWN_LOGGERS = ("livekit.plugins.openai", "livekit.plugins.google")
PROVIDER_TEARDOWN_FRAGMENTS = (
    "connection closed unexpectedly",
    "connection failed",
    "session failed",
    "Error in _main_task",
)
#: The one framework record that is unambiguously the session giving up on a
#: provider, matched as a whole phrase rather than by a generic fragment.
AGENT_SESSION_UNRECOVERABLE = (
    "livekit.agents",
    "AgentSession is closing due to unrecoverable error",
)

#: Open ONLY while this worker is performing its own deliberate close of a
#: provider session on a call that ended before the answer gate released the
#: caller's audio. Both conditions matter and neither is enough alone:
#:
#: * not "the call is ending" -- a provider failure during ringing ENDS the
#:   call, and a window opened on every ending swallows that failure's own
#:   records, which is the page this worker exists to send;
#: * not "before answer" alone -- same reason, one step removed.
#:
#: Opened immediately before the close and closed immediately after it, so a
#: record that arrives outside it reports. A straggler landing after the
#: window is reported rather than dropped, which is the safe direction.
#: The window also carries WHOSE close it is. A caller that knows its own
#: call id gets an exact answer; the error-tracking filter, which sees only a
#: log record, necessarily gets the process-wide one. This deployment runs one
#: LiveKit job per OS process (all 46 jobs in the 2026-09-22 window had
#: distinct pids), so the two agree today -- but the caller-scoped answer is
#: the one that stays correct if that ever stops being true, and a filter that
#: cannot ask is the only place left relying on it.
_PROVIDER_CLOSE: dict[str, Any] = {"active": False, "call_id": ""}


def mark_provider_close_in_progress(call_id: Any = "") -> None:
    """This worker is closing a provider session on an unanswered call.

    The CAUSE is deliberately not kept here. The one place that reports a
    cause -- the worker's own informational line -- reads it from the
    arbiter, which owns it; a second copy in this module would have no reader
    and could only drift from the first.
    """

    _PROVIDER_CLOSE["active"] = True
    _PROVIDER_CLOSE["call_id"] = str(call_id or "")[:64]


def clear_provider_close_in_progress() -> None:
    """That close is done. Anything after it is a fault again."""

    _PROVIDER_CLOSE["active"] = False
    _PROVIDER_CLOSE["call_id"] = ""


def provider_close_in_progress(call_id: Any = "") -> bool:
    """Whether this worker is inside its own deliberate provider close.

    With a ``call_id``, whether it is inside the close OF THAT CALL: another
    call's teardown never relabels this one's fault.
    """

    if not _PROVIDER_CLOSE["active"]:
        return False
    wanted = str(call_id or "")
    if not wanted:
        return True
    return _PROVIDER_CLOSE["call_id"] == wanted


def is_provider_teardown_noise(text: Any, logger: Any) -> bool:
    """Whether this record is a provider connection dying in a teardown of a
    call that was never answered.

    Returns False unless :func:`mark_provider_close_in_progress` has opened
    the window, so the same record outside this worker's own close -- on a
    live call, or from the provider failure that ENDED the call -- reports.
    """

    if not provider_close_in_progress():
        return False
    name = str(logger or "")
    message = str(text or "")
    agent_logger, agent_phrase = AGENT_SESSION_UNRECOVERABLE
    if name == agent_logger or name.startswith(agent_logger + "."):
        return agent_phrase in message
    if not any(name == owner or name.startswith(owner + ".") for owner in PROVIDER_TEARDOWN_LOGGERS):
        return False
    return any(fragment in message for fragment in PROVIDER_TEARDOWN_FRAGMENTS)


def _event_message_text(event: dict[str, Any]) -> str:
    """Everything an event says, for matching. Never for transport."""

    parts: list[str] = []
    message = event.get("message")
    if isinstance(message, dict):
        # The SDK writes either shape depending on how the record was captured.
        parts.append(str(message.get("formatted") or message.get("message") or ""))
    else:
        parts.append(str(message or ""))
    logentry = event.get("logentry")
    if isinstance(logentry, dict):
        parts.append(str(logentry.get("formatted") or logentry.get("message") or ""))
    return " ".join(part for part in parts if part)


try:  # the call-values registry is reached through the one navigation seam
    from .navigation_entry import scrub_registered_structure as _scrub_call_values
except ImportError:
    from navigation_entry import scrub_registered_structure as _scrub_call_values  # type: ignore[no-redef]


def _scrub_values_or_drop(value: Any) -> Any:
    """A keypad value a call in this process holds is masked in what is sent; a
    report that cannot be scrubbed of it is dropped, like any other."""
    try:
        return None if value is None else _scrub_call_values(value)
    except Exception:
        return None


def filter_and_scrub_event(event: Any, hint: Any = None) -> Any:
    """``before_send`` for this worker: drop FFI teardown noise, then scrub.

    Wrapping rather than editing :func:`scrub_event`, because that function is
    half of a contract this worker shares byte-for-byte with the control plane.
    The drop happens BEFORE the walk: a record nobody should be paged for is
    not worth scrubbing, and the decision is about the record, not its contents.
    """

    try:
        if not isinstance(event, dict):
            return None
        # The framework logs rejected dispatches after the job scope exits.
        # These refusals are still raised, but are not crashed conversations.
        exc_info = hint.get("exc_info") if isinstance(hint, dict) else None
        if isinstance(exc_info, tuple) and len(exc_info) == 3:
            if _browser_demo_refusal(exc_info[1]):
                return None
            if getattr(exc_info[1], "_skysay_browser_demo_reported", False) is True:
                return None
        if _reported_by_origin(hint):
            return None
        text = _event_message_text(event)
        if is_ffi_dead_handle_noise(text, event.get("logger")):
            return None
        if is_provider_teardown_noise(text, event.get("logger")):
            return None
    except Exception:
        # Never let the filter decide to SEND something; fall through to the
        # scrubber, which drops anything it cannot handle.
        pass
    # Held values first: the generic scrubber truncates long text, and a value
    # cut in half by that is no longer a run the value scrub can recognise.
    return _scrub_values_or_drop(scrub_event(_scrub_values_or_drop(event), hint))


def filter_and_scrub_breadcrumb(crumb: Any, hint: Any = None) -> Any:
    """``before_breadcrumb`` for this worker. Same rule, same reason.

    A breadcrumb names its logger in ``category``; an event names it in
    ``logger``. One record, one rule, two field names.
    """

    try:
        if not isinstance(crumb, dict):
            return None
        if _reported_by_origin(hint):
            return None
        if is_ffi_dead_handle_noise(crumb.get("message"), crumb.get("category")):
            return None
        if is_provider_teardown_noise(crumb.get("message"), crumb.get("category")):
            return None
    except Exception:
        pass
    return _scrub_values_or_drop(scrub_breadcrumb(_scrub_values_or_drop(crumb), hint))


#: Attribute names every ``LogRecord`` already has, read off the running
#: standard library rather than copied into a list here: a hand-kept copy of
#: someone else's field set is the thing that drifts. ``message`` and
#: ``asctime`` are added because a FORMATTER puts them on the record, so they
#: are not on a freshly built one but are on the record a handler sees.
_RECORD_ATTRIBUTES = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None))
) | {"message", "asctime"}

#: The loggers in the pinned worker image whose records carry a ``lk.pii.*``
#: extra. The set is DERIVED at install time — every logger already registered
#: under the ``livekit`` tree — because a hand-kept list of provider names is
#: exactly how a second copy of a registry goes stale. These names are the
#: floor under that scan: a job process installs error tracking in
#: ``setup_fnc``, which can run before a plugin module is imported, and a
#: logger that does not exist yet cannot be found by scanning.
#:
#: The floor is not a guess. Every file in ``livekit-agents`` 1.8.1 and the
#: four pinned plugins that attaches a ``lk.pii.*`` extra was enumerated, and
#: each one resolves to a PACKAGE logger in this tuple — the emit sites take
#: ``from ..log import logger``, not ``getLogger(__name__)``, so there are no
#: per-module children to miss. A filter runs only on the logger that
#: ORIGINATED the record, so that distinction is the whole question.
#:
#: ``dtmf-inputs`` is the exception that proves why the tuple is enumerated
#: rather than assumed: ``livekit/agents/beta/workflows/dtmf_inputs.py`` calls
#: ``logging.getLogger("dtmf-inputs")`` and emits ``lk.pii.dtmf_inputs``, so it
#: sits OUTSIDE the ``livekit`` tree and no prefix scan will ever find it.
PII_EXTRA_LOGGER_ROOT = "livekit"
PII_EXTRA_LOGGERS = (
    "dtmf-inputs",
    "livekit",
    "livekit.agents",
    "livekit.plugins.fishaudio",
    "livekit.plugins.google",
    "livekit.plugins.openai",
    "livekit.plugins.turn_detector",
)


def is_pii_labelled_key(key: Any) -> bool:
    """Whether the PRODUCER declared this key to be personal data.

    Deliberately NARROWER than :func:`is_sensitive_key`, and the difference is
    the whole reason this function exists. The scrubbers decide what may leave
    the process, so they apply the full rule -- the label, every dotted
    segment, and both vocabularies. This one decides what a log RECORD may
    carry, which is also what this worker prints to its own stdout, so it only
    honours the label the producer itself attached.

    Measured, not assumed. ``room`` and ``job_id`` usually reach a record
    through the agent framework's own context filter, which is a HANDLER
    filter; a handler filter runs after a logger filter, so this one never sees
    those. But the framework also logs "received job request" with an explicit
    ``extra={"room": ..., "job_id": ...}``, and the full rule would blank that
    line's room name in the worker's docker logs -- taking an operator's
    correlation handle for a value the scrubbers already refuse at the wire.
    The label is the signal the SDK went to the trouble of providing; a second
    guess about somebody else's field is not this control's job.
    """

    return PII_KEY_PART in _normalize_key(key)


_REPORT_OWNER = "_skysay_error_owner_pid"


def _reported_by_origin(hint: Any) -> bool:
    """LiveKit relays child records through the parent's callHandlers again."""
    record = hint.get("log_record") if isinstance(hint, dict) else None
    owner = getattr(record, _REPORT_OWNER, None)
    return (
        type(owner) is int
        and owner == getattr(record, "process", None)
        and owner != os.getpid()
    )


class ReportingOwnerFilter(logging.Filter):
    """Claim before LiveKit copies/pickles the record into its logging queue.

    Marking in before_send is too late: handlers already queued the copy.
    Unowned relayed records still report in the parent when child reporting
    was unavailable. Distinct errors are never collapsed by message or time.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            import sentry_sdk

            if record.process == os.getpid() and sentry_sdk.get_client().is_active():
                setattr(record, _REPORT_OWNER, os.getpid())
        except Exception:
            pass
        return True


class PiiExtraFilter(logging.Filter):
    """Replace a record's PII-labelled extras before anything reads them.

    Belt and braces, and the braces are :func:`filter_and_scrub_event` and
    :func:`filter_and_scrub_breadcrumb`, which refuse the same keys at every
    depth in both directions. This filter exists because the two live at
    different distances from the leak: the scrubbers run on a payload the SDK
    has already built, while this runs on the record, before
    ``logging.Logger.callHandlers`` — which the Sentry logging integration
    patches — is reached at all. A record whose extras never carried the value
    cannot carry it into an event, a breadcrumb, or a future SDK path nobody
    has audited.

    Only EXTRAS are touched: a record's own attributes are left exactly as the
    standard library set them, so the message, its arguments and the source
    location are unchanged.

    The value is REPLACED, not deleted. A filter must not change which fields a
    record has — a formatter written against ``%(lk.pii.transcript)s`` would
    start raising — and ``[redacted]`` is also the honest thing to print.

    Named consequence: a record is one object, so this also redacts the value
    in what this worker writes to its own stdout. That is the standard this
    module's docstring already sets for the host's files, applied to the host's
    logs; the SDK labelled the value, and this worker is not a better place to
    keep it than the error backend was. It is also why the predicate here is
    :func:`is_pii_labelled_key` and not :func:`is_sensitive_key`: see that
    function for what the difference costs and buys.

    Named LIMIT, because a belt-and-braces control that is described as
    complete is worse than one that is not: a filter runs only on the logger
    that ORIGINATED the record, so a ``livekit.*`` logger that comes into
    existence after :func:`install_pii_extra_filter` has run -- a plugin added
    to the image and imported late -- is not covered by this half.
    :func:`filter_and_scrub_event` and :func:`filter_and_scrub_breadcrumb` are,
    and they are the complete defence; this one is the early, partial second
    lock, and :data:`PII_EXTRA_LOGGERS` is what keeps it covering the pinned
    emitters.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            for name in list(vars(record)):
                if name in _RECORD_ATTRIBUTES or name.startswith("_"):
                    continue
                if is_pii_labelled_key(name):
                    setattr(record, name, REDACTED)
        except Exception:
            # A filter that raises drops the record. Reporting is never worth
            # losing a log line over, and the scrubbers still hold.
            pass
        return True


def pii_extra_logger_names(manager: Any = None) -> tuple[str, ...]:
    """Every logger this process should filter, known names plus live ones."""

    registry = getattr(manager or logging.Logger.manager, "loggerDict", {})
    names = set(PII_EXTRA_LOGGERS)
    prefix = PII_EXTRA_LOGGER_ROOT + "."
    for name in list(registry):
        text = str(name)
        if text == PII_EXTRA_LOGGER_ROOT or text.startswith(prefix):
            names.add(text)
    return tuple(sorted(names))


def install_pii_extra_filter() -> tuple[str, ...]:
    """Attach :class:`PiiExtraFilter` to those loggers. Never raises.

    Idempotent: a warm job process installs error tracking once per job, and a
    filter added per job would walk every record N times for nothing.

    The attachment point is the ORIGINATING logger deliberately. A filter on an
    ancestor does not run — ``Logger.handle`` applies its own filters and then
    calls ``callHandlers``, which visits ancestors' HANDLERS only — so filtering
    at ``livekit`` would leave ``livekit.plugins.openai`` unfiltered.
    """

    attached: list[str] = []
    try:
        # Materialise the known names BEFORE scanning. ``logging.getLogger``
        # inserts a ``PlaceHolder`` for every missing ancestor, so asking for
        # ``livekit.plugins.openai`` puts ``livekit.plugins`` into the registry
        # as a side effect of this very function. Scanning first therefore made
        # the SECOND call find a name the first had created and attach again:
        # idempotent in a process where something else had already imported the
        # plugins, and not idempotent in a fresh one.
        for name in PII_EXTRA_LOGGERS:
            logging.getLogger(name)
        for name in pii_extra_logger_names():
            target = logging.getLogger(name)
            if not any(isinstance(existing, ReportingOwnerFilter) for existing in target.filters):
                target.addFilter(ReportingOwnerFilter())
            if any(isinstance(existing, PiiExtraFilter) for existing in target.filters):
                continue
            target.addFilter(PiiExtraFilter())
            attached.append(name)
    except Exception:
        return tuple(attached)
    return tuple(attached)


def error_tracking_options(
    *,
    dsn: str,
    component: str,
    environment: str,
    release: str | None,
) -> dict[str, Any]:
    """The exact keyword arguments handed to ``sentry_sdk.init``.

    Split out from :func:`install_error_tracking` so the contract test asserts
    the values this worker really sends rather than re-deriving them.
    """
    return {
        "dsn": dsn,
        "environment": environment,
        "release": release,
        # The hostname the SDK would otherwise resolve says nothing this
        # deployment does not already know (there is one voice host) and puts
        # one more internal name on the wire.
        "server_name": component,
        # No spans. A conversation's latency budget is milliseconds and a
        # transaction envelope per call buys nothing a crash report needs.
        "traces_sample_rate": 0.0,
        "profiles_sample_rate": 0.0,
        "sample_rate": 1.0,
        # Stack-frame locals are where a phone number, a transcript chunk and a
        # provider key live in this process. The SDK defaults this to True.
        "include_local_variables": False,
        # Source context is kept: seeing the failing line is most of what
        # makes a report worth reading. It is repository source -- static text
        # written by us -- so it carries no customer data, and it is scrubbed
        # like everything else. The one consequence worth naming: a string
        # LITERAL in our own code appears in a report verbatim, so a literal is
        # not a place to put a credential.
        "include_source_context": True,
        "send_default_pii": False,
        # NOT ``MAX_TEXT``. The SDK applies this cap BEFORE ``before_send``
        # runs, so a 1024 ceiling here hands the scrubber a value that has
        # already been cut -- and a cut address (``jane.doe@examp``) matches no
        # rule and ships. Measured: at 1024 the local part of an address at
        # offset 1006 reached the transport. The scrubber does its own cut, to
        # MAX_TEXT, AFTER the rules have run; this number only has to be large
        # enough that the SDK's cut never lands in front of them.
        "max_value_length": MAX_SCAN,
        "max_breadcrumbs": 20,
        "attach_stacktrace": False,
        # Bounded atexit flush. A job process exits after its call, so this is
        # the only wait the SDK ever imposes on this worker, and an unreachable
        # backend must not turn it into a stall.
        "shutdown_timeout": 1.0,
        # The filtering wrappers, not the bare scrubbers: LiveKit FFI teardown
        # noise must not become an error-tracker issue, or a push.
        "before_send": filter_and_scrub_event,
        "before_breadcrumb": filter_and_scrub_breadcrumb,
        # No OPTIONAL third-party patching inside a realtime worker: this
        # switches off the auto-enabling set (aiohttp, Django, Celery and the
        # rest), which is patching we gain nothing from. It does NOT disable
        # the default integrations, and it is not meant to: `logging` is how a
        # job crash becomes an event at all, and `stdlib`, `threading`,
        # `excepthook`, `dedupe`, `modules`, `argv` and `atexit` come with it.
        "auto_enabling_integrations": False,
        "auto_session_tracking": False,
        "enable_metrics": False,
        "enable_logs": False,
    }


def dsn_from_environment(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return str(source.get("SKYSAY_GLITCHTIP_DSN") or "").strip()


def install_error_tracking(
    *,
    component: str,
    environ: Mapping[str, str] | None = None,
    report: Any = None,
) -> bool:
    """Start error reporting. Returns whether it started; never raises.

    ``report`` is an optional ``emit``-style callable used only to record the
    outcome as one structured line. It is called with a reason code, never with
    a DSN, a host or an exception message.
    """
    source = os.environ if environ is None else environ

    def note(outcome: str, **details: Any) -> None:
        if report is None:
            return
        try:
            report("error_tracking", outcome=outcome, component=component, **details)
        except Exception:
            pass

    dsn = dsn_from_environment(source)
    if not dsn:
        note("disabled", reason="no_dsn")
        return False
    try:
        import sentry_sdk
    except Exception:
        # The SDK is a normal pinned dependency; an ImportError here means a
        # broken image, and a broken image must still answer calls.
        note("unavailable", reason="sdk_import_failed")
        return False
    try:
        sentry_sdk.init(
            **error_tracking_options(
                dsn=dsn,
                component=component,
                environment=str(source.get("SKYSAY_ENV") or source.get("APP_ENV") or "production"),
                release=str(source.get("SKYSAY_VERSION") or "").strip() or None,
            )
        )
    except Exception:
        # A malformed DSN raises here. Everything else the backend can do wrong
        # — refused connection, bad TLS, 500s — happens later on the transport
        # thread, where the SDK already swallows it.
        note("unavailable", reason="init_failed")
        return False
    # After init, never before: this filter only matters because the SDK is now
    # watching every log record, and before init there is nothing to keep a
    # value away from.
    install_pii_extra_filter()
    note("enabled")
    return True


def capture_exception(exc: BaseException | None = None, **tags: str) -> bool:
    """Report one exception. Never raises, never waits on the network."""
    try:
        import sentry_sdk

        if not sentry_sdk.get_client().is_active():
            return False
        with sentry_sdk.new_scope() as scope:
            # Through the KEY rule as well as the text rules. Every call site
            # today passes a static label (``component=``, ``phase=``), but
            # ``capture_exception(exc, transcript=caller_said)`` is the obvious
            # way to write the mistake, and no pattern can recognise a sentence.
            for key, value in scrub_structure(dict(tags)).items():
                scope.set_tag(key, value)
            sentry_sdk.capture_exception(exc)
        return True
    except Exception:
        return False


__all__: Sequence[str] = (
    "BrowserDemoErrorContext",
    "CONTENT_KEYS",
    "DROPPED_KEYS",
    "IDENTIFIER_KEYS",
    "MAX_DEPTH",
    "MAX_ITEMS",
    "MAX_TEXT",
    "PII_KEY_PART",
    "REDACTED",
    "SECRET_KEY_PARTS",
    "FFI_DEAD_HANDLE_FRAGMENTS",
    "FFI_LOGGER",
    "PII_EXTRA_LOGGERS",
    "PII_EXTRA_LOGGER_ROOT",
    "PiiExtraFilter",
    "capture_exception",
    "dsn_from_environment",
    "error_tracking_options",
    "filter_and_scrub_breadcrumb",
    "filter_and_scrub_event",
    "install_error_tracking",
    "install_pii_extra_filter",
    "is_ffi_dead_handle_noise",
    "is_pii_labelled_key",
    "is_sensitive_key",
    "pii_extra_logger_names",
    "redact_text",
    "scrub_breadcrumb",
    "scrub_event",
    "scrub_structure",
)
