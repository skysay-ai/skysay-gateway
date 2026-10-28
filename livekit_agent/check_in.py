# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The platform's "are you still there?" line, spoken on every engine.

Built for the mutual-silence ending and shaped to be the seam the
scripted-agent check-in extends: one localized line, one way to say
it per engine, and a handle back so the caller of this module owns what
happens after.

**How it is spoken is a fact about the session, not the engine name.** A
session with a TTS speaks a fixed line with ``say``. A voice-to-voice session
has none -- LiveKit refuses ``say(text)`` there -- so the model is asked to say
it, in the conversation's own language, with ``generate_reply`` and no tools:
a presence check is never an opportunity to act. A session whose replies come
from a workflow gets no check-in at all from this module: a caller's "yes"
would be delivered to the pending business step as its answer. The scripted
check-in owns that path, with its own renderings and suspended-step handling.

**The line itself is a small authored table, like the fallback goodbye.** A
language with no entry has no fixed line; the text-capable path then asks the
model for one instead, which is the same thing the voice-to-voice path always
does. Unlike the goodbye table, these lines have not been measured through
each TTS provider; a line that fails to synthesize is a check-in not
delivered, which the rule treats exactly like one that was never spoken.
"""

from __future__ import annotations

from typing import Any

#: The fixed check-in, per language. Deliberately short: one question, which
#: a caller answers with one word. Languages not listed use the generated path.
CHECK_IN_PHRASES = {
    "en": "Are you still there?",
    "et": "Kas te olete veel liinil?",
}

#: The instruction the generated check-in is spoken from.
CHECK_IN_INSTRUCTION = (
    "The caller has been silent for a while. Ask them, in one short question in "
    "the language of this conversation, whether they are still there. Say nothing "
    "else and do not repeat anything you said before."
)

#: The instruction a voice-to-voice farewell is spoken from when the platform
#: ends a silent call (the model has no fixed-line path).
SILENCE_FAREWELL_INSTRUCTION = (
    "The caller has not answered. Say one short, polite goodbye in the language of "
    "this conversation and nothing else."
)


def check_in_phrase(language: Any) -> str:
    """The fixed check-in for ``language``, or ``""`` when none is authored."""

    return CHECK_IN_PHRASES.get(str(language or "").strip().lower(), "")


def speak_check_in(
    session: Any,
    *,
    speaks_text: bool,
    scripted: bool,
    language: Any,
) -> Any:
    """Speak the check-in and return its speech handle, or ``None``.

    ``None`` means no check-in was spoken: a scripted session, which this
    module never speaks for. A failure to schedule it raises; the caller
    treats both as "not delivered".
    """

    if scripted:
        return None
    phrase = check_in_phrase(language) if speaks_text else ""
    if phrase:
        return session.say(phrase, allow_interruptions=True, add_to_chat_ctx=True)
    return session.generate_reply(
        instructions=CHECK_IN_INSTRUCTION,
        tool_choice="none",
        allow_interruptions=True,
    )


__all__ = [
    "CHECK_IN_INSTRUCTION",
    "CHECK_IN_PHRASES",
    "SILENCE_FAREWELL_INSTRUCTION",
    "check_in_phrase",
    "speak_check_in",
]
