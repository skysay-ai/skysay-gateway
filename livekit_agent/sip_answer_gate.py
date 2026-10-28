# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""An outbound greeting must wait for the callee to pick up.

LiveKit SIP puts the outbound participant in the room ~30 ms after the INVITE,
while the leg is still dialing, and nothing downstream of that join depends on
whether a human has answered.  The agent's opening turn therefore started at a
fixed ~3.5-4.4 s after the join and was rendered into a leg that had no RTP
destination at all -- measured across eight production calls, the fraction of
the greeting a human actually heard was exactly the race between that fixed
offset and the carrier's ring->answer time (7.41 s answer -> 0 % audible;
3.99 s -> 95 %).  Real callees answer in 5-20 s, so the greeting reliably
played to nobody.

The answer evidence LiveKit does publish is the SIP participant attribute
``sip.callStatus``; the pinned runtime's own answering-machine detector waits
on exactly this before it will look at any audio, precisely so that "pre-answer
audio (ringback, carrier early media, dialtone)" is excluded.

Every DECISION lives here, in a module that imports no livekit, so the
requirement can be driven from a test suite that has no livekit dependency.
The glue in ``agent/production_agent.py`` supplies the runtime's own
``wait_for_participant_attribute`` and the participant's current attributes.

Two deliberate fail-open dispositions, because a gate that can silence the
agent forever is worse than the defect it fixes:

* ``NOT_REPORTED`` -- the runtime never published ``sip.callStatus`` for this
  participant, so there is no answer signal to wait for.  Behave EXACTLY as
  before this change: speak immediately and report ``sip.answered``.  A
  degraded mode must be a no-op, not a new failure mode.
* ``TIMED_OUT`` -- the attribute exists but never reached ``active`` within
  the ringing budget while the participant stayed connected.  Speak (the
  signal may be stuck), but do NOT report ``sip.answered``: an "answered"
  event must carry the PSTN leg's answer evidence, and here there is none.

``DISCONNECTED`` is the only disposition that suppresses the greeting.  The
callee is gone; there is nobody to greet and no answer to report.

The wait is started at room join, NOT at the opening turn.  The two are
separated by several seconds of provider warm-up, and a callee who answers and
hangs up inside that window has genuinely answered: the lifecycle report is
owed even though the greeting is not.  ``may_speak`` therefore takes the
far end's presence as a second, separate condition.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Mapping

SIP_CALL_STATUS_ATTRIBUTE = "sip.callStatus"
SIP_CALL_STATUS_ACTIVE = "active"

# The dialer's own bound, read the same way: ``agent/outbound_worker.py``
# builds the LiveKit ``ringing_timeout`` from this variable, with this default
# and this clamp.  A self-host stack sources one env file into both roles, so
# there they agree by construction; elsewhere the role definitions belong to
# the host, and a host that lengthens ringing for the dialer alone only makes
# this wait expire early -- which fails OPEN to the pre-gate behaviour rather
# than dropping a call.
RINGING_TIMEOUT_VARIABLE = "SKYSAY_OUTBOUND_RINGING_TIMEOUT_SECONDS"
DEFAULT_RINGING_TIMEOUT_SECONDS = 45
MIN_RINGING_TIMEOUT_SECONDS = 5
MAX_RINGING_TIMEOUT_SECONDS = 120

# Dispositions.
ANSWERED = "answered"
ALREADY_ACTIVE = "already_active"
DISCONNECTED = "disconnected"
TIMED_OUT = "timed_out"
NOT_REPORTED = "not_reported"
# The gate did not apply: this leg is not an outbound SIP call.
NOT_APPLICABLE = "not_applicable"

# The callee is gone. Everything else speaks.
_SILENT_DISPOSITIONS = frozenset({DISCONNECTED})
# Only a `sip.callStatus` of `active` is evidence the PSTN leg was answered.
# `NOT_REPORTED` is included because it is the pre-gate behaviour verbatim:
# without the attribute this worker knows exactly what it knew before, and
# dropping the customer's `call.answered` webhook in that mode would be a
# regression introduced by a fallback path.
_ANSWER_EVIDENCE_DISPOSITIONS = frozenset({ANSWERED, ALREADY_ACTIVE, NOT_REPORTED})


def ringing_timeout_seconds(environ: Mapping[str, str]) -> float:
    """The dialer's ringing budget, clamped exactly as the dialer clamps it."""

    raw = str(environ.get(RINGING_TIMEOUT_VARIABLE) or "").strip()
    try:
        seconds = int(raw)
    except ValueError:
        seconds = DEFAULT_RINGING_TIMEOUT_SECONDS
    if seconds < MIN_RINGING_TIMEOUT_SECONDS or seconds > MAX_RINGING_TIMEOUT_SECONDS:
        seconds = DEFAULT_RINGING_TIMEOUT_SECONDS
    return float(seconds)


def sip_call_status(attributes: Mapping[str, Any]) -> str | None:
    """``sip.callStatus`` as the runtime currently reports it, or ``None``.

    ``None`` here means the ATTRIBUTE is absent for a participant that is
    present -- not that the participant is gone, and not that the call is
    inactive.  Those are three different answers and the gate below treats
    them three different ways; conflating the first two is what would let a
    hung-up leg be greeted.
    """

    value = attributes.get(SIP_CALL_STATUS_ATTRIBUTE)
    if value is None:
        return None
    return str(value)


async def await_callee_answer(
    *,
    participant_attributes: Callable[[], Mapping[str, Any] | None],
    wait_for_active: Callable[[], Awaitable[None]],
    timeout_seconds: float,
) -> str:
    """Block until the callee picks up; return which of the five it was.

    ``participant_attributes`` returns the SIP participant's current
    attributes, or ``None`` when that participant is no longer in the room.

    ``wait_for_active`` is the runtime's ``wait_for_participant_attribute``
    bound to this participant.  It raises ``RuntimeError`` when the
    participant or the room goes away before the attribute arrives -- the
    reference implementation in the pinned SDK's answering-machine detector
    treats that branch the same way.
    """

    attributes = participant_attributes()
    if attributes is None:
        return DISCONNECTED
    current = sip_call_status(attributes)
    if current is None:
        return NOT_REPORTED
    if current == SIP_CALL_STATUS_ACTIVE:
        return ALREADY_ACTIVE
    try:
        await asyncio.wait_for(wait_for_active(), timeout=timeout_seconds)
    except (TimeoutError, asyncio.TimeoutError):
        # `asyncio.wait_for` cancels `wait_for_active()` and AWAITS that
        # cancellation before raising here -- which yields to the event loop.
        # A `sip.callStatus` update delivered inside that yield lands on the
        # participant object even though the cancelled future can never
        # resolve it, so re-reading is not enough: the re-read must be judged
        # by its VALUE, not just its presence, or a callee who answered in
        # that exact window is reported as never having answered at all.
        resumed = participant_attributes()
        if resumed is None:
            return DISCONNECTED
        if sip_call_status(resumed) == SIP_CALL_STATUS_ACTIVE:
            return ANSWERED
        return TIMED_OUT
    except RuntimeError:
        return DISCONNECTED
    return ANSWERED


def may_speak(disposition: str, *, still_present: bool = True) -> bool:
    """May the opening turn be spoken now?

    Two independent conditions, because the answer is observed at room join
    and the greeting happens seconds later, after provider warm-up.  A callee
    who answered and then hung up during that gap really did answer -- the
    lifecycle report is owed -- but there is nobody left to greet.
    ``still_present`` is best-effort by nature: the far end can leave in the
    microsecond after it is read, and speaking into a leg that has just gone
    is harmless.  What it prevents is the several-second case.
    """

    if disposition in _SILENT_DISPOSITIONS:
        return False
    return still_present


def has_answer_evidence(disposition: str) -> bool:
    """May this worker report ``sip.answered`` for the PSTN leg?"""

    return disposition in _ANSWER_EVIDENCE_DISPOSITIONS


#: Where an outbound call's listen window starts, per disposition. The
#: callee-first opening (``opening_turn``) waits a fixed window after the
#: answer for the person's hello; this says what "the answer" is.
#:
#: * ``answered`` -- the watch saw ``sip.callStatus`` turn ``active``: the
#:   observation instant is the answer.
#: * ``observed_late`` -- it was already ``active`` when first read (the answer
#:   came before this worker was watching): the observation instant is the
#:   earliest one there is, and the record says it is late.
#: * ``first_audio`` -- no answer evidence to anchor on (the carrier never
#:   reported, or the report never arrived): the first caller frame received
#:   after input release is the answer.
#:
#: ``DISCONNECTED`` has no anchor: nobody is left to listen to.
CLOCK_ANCHOR_ANSWERED = "answered"
CLOCK_ANCHOR_OBSERVED_LATE = "observed_late"
CLOCK_ANCHOR_FIRST_AUDIO = "first_audio"
_CLOCK_ANCHORS = {
    ANSWERED: CLOCK_ANCHOR_ANSWERED,
    ALREADY_ACTIVE: CLOCK_ANCHOR_OBSERVED_LATE,
    NOT_REPORTED: CLOCK_ANCHOR_FIRST_AUDIO,
    TIMED_OUT: CLOCK_ANCHOR_FIRST_AUDIO,
}


def clock_anchor(disposition: str) -> str | None:
    """How the listen window is anchored for this disposition, or ``None``."""

    return _CLOCK_ANCHORS.get(disposition)


__all__ = [
    "ALREADY_ACTIVE",
    "ANSWERED",
    "CLOCK_ANCHOR_ANSWERED",
    "CLOCK_ANCHOR_FIRST_AUDIO",
    "CLOCK_ANCHOR_OBSERVED_LATE",
    "DEFAULT_RINGING_TIMEOUT_SECONDS",
    "DISCONNECTED",
    "MAX_RINGING_TIMEOUT_SECONDS",
    "MIN_RINGING_TIMEOUT_SECONDS",
    "NOT_APPLICABLE",
    "NOT_REPORTED",
    "RINGING_TIMEOUT_VARIABLE",
    "SIP_CALL_STATUS_ACTIVE",
    "SIP_CALL_STATUS_ATTRIBUTE",
    "TIMED_OUT",
    "await_callee_answer",
    "clock_anchor",
    "has_answer_evidence",
    "may_speak",
    "ringing_timeout_seconds",
    "sip_call_status",
]
