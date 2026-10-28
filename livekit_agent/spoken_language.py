# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""What language this call speaks, said once, where every path can read it.

Three things needed a canonical answer and had none inside the worker image:

1. **The LLM was never told the language.** ``production.py`` handed
   the agent's ``system_prompt`` to ``Agent(instructions=...)`` verbatim, and
   the default prompt's own sentence -- "Speak only in the agent's configured
   language" -- names a value the model cannot resolve. Every generated turn on
   the first Sinhala campaign came back in English while the literal
   ``first_message`` was Sinhala, which is exactly the shape that produces.
2. **A greeting that fails leaves the callee in silence.** The
   opening ``session.say`` swallows a TTS failure (LiveKit documents that
   awaiting a ``SpeechHandle`` never raises), so a callee who picks up hears
   nothing at all until the first model reply. A short line in the right
   language is the honest minimum.
3. Both of the above need the language's NAME, and the only table in the
   repository was ``frontend/src/lib/voice-stack.js`` -- JavaScript, in the
   other deployment unit, unreachable from this image.

The names here are deliberately the same set and spelling as that frontend
table so a reader comparing the two finds no drift; ``test_sch603_*`` pins that
every language the packaged capability artifact will run has a name here.

Nothing in this module talks to a provider or reads configuration; it is data
plus two string builders, so both the phone path (``production.py``) and the
browser demo path (``browser_demo.py``) can apply the identical directive.
"""
from __future__ import annotations

# Same codes and spellings as ``LANGUAGE_NAMES`` in
# frontend/src/lib/voice-stack.js for every code that table curates. The
# frontend copy names what an operator picks in the agent editor; this copy
# names what the worker tells the model and the callee. They are two
# deployment units, so they are two files -- but a divergence would mean the
# dropdown and the spoken call disagree about what "si" is, so keep them equal
# by hand and let the coverage test below catch a language the worker can run
# and this table cannot name. The Flash-Lite-era languages are named only here: the
# frontend labels them through ``Intl.DisplayNames`` in the viewer's locale
# rather than curating six translations of each.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "fi": "Finnish",
    "et": "Estonian",
    "sv": "Swedish",
    "uk": "Ukrainian",
    "zh": "Chinese",
    "ja": "Japanese",
    "si": "Sinhala",
    "ta": "Tamil",
    # The call languages added with Gemini 3.8 Flash-Lite TTS.
    "af": "Afrikaans",
    "am": "Amharic",
    "hy": "Armenian",
    "as": "Assamese",
    "bn": "Bangla",
    "eu": "Basque",
    "be": "Belarusian",
    "bho": "Bhojpuri",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "yue": "Cantonese Chinese",
    "ca": "Catalan",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "arz": "Egyptian Arabic",
    "fil": "Filipino",
    "gl": "Galician",
    "ka": "Georgian",
    "el": "Greek",
    "gu": "Gujarati",
    "mn": "Mongolian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "fa": "Persian",
    "prs": "Dari",
    "jv": "Javanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "lo": "Lao",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mr": "Marathi",
    "ne": "Nepali",
    "az": "Azerbaijani",
    "uz": "Uzbek",
    "nb": "Norwegian Bokmål",
    "or": "Odia",
    "pa": "Punjabi",
    "ro": "Romanian",
    "sr": "Serbian",
    "sk": "Slovak",
    "ps": "Pashto",
    "ar": "Arabic",
    "lv": "Latvian",
    "ms": "Malay",
    "te": "Telugu",
}

# Marker for the idempotence check below. Present in every rendered directive,
# and specific enough that an agent's own hand-written prompt cannot contain
# it by accident -- the same shape ``append_end_call_instruction`` uses in the
# control plane, for the same reason: these prompts pass through more than one
# appender and a doubled directive is confusing rather than twice as effective.
_DIRECTIVE_MARKER = "Skysay language directive:"
_GENERATED_OPENING_MARKER = "Skysay generated opening directive:"

# The opening line spoken when the configured greeting failed to synthesize.
#
# EVERY string here was measured against the production Gemini TTS stack
# (``gemini-2.5-flash-preview-tts``, voice Achernar) rather than written and
# hoped for, because that stack silently returns NO AUDIO for some perfectly
# ordinary sentences -- see ``gemini_tts.py``'s
# ``gemini_tts_text_instead_of_audio`` for the failure shape. A fallback that
# cannot itself be spoken is worse than none: it turns one silence into two.
# Measured 2026-08-28: si 5/5, en 5/5, ta 3/3, et 3/3.
#
# Each is also ONE sentence as LiveKit's tokenizer counts them. That is not a
# style preference. ``AgentSession`` wraps a non-streaming TTS in
# ``tts.StreamAdapter``, which synthesises SENTENCE BY SENTENCE
# (``min_sentence_len=20``), so a two-sentence line is two provider requests
# and either one failing fails the whole utterance. The already-shipped
# Sinhala end-call goodbye is exactly that trap: it splits, and its
# ten-character second half measured 1/5.
#
# Honest, not reassuring: the agent has not introduced itself yet, so this
# says only that somebody is there and invites the callee to answer, which is
# also what gets the conversation to the model's first real turn. The
# non-English wordings are machine-rendered and not reviewed by a native
# speaker -- the same caveat the conversation-workflow scenarios carry.
GREETING_FALLBACK_MESSAGES: dict[str, str] = {
    "en": "Hello. Can you hear me?",
    "et": "Tere. Kas te kuulete mind?",
    "si": "ආයුබෝවන්. ඔබට ඇසෙනවා ද?",
    "ta": "வணக்கம். என் குரல் கேட்கிறதா?",
}


#: The answers a callee picks up the phone with and nothing more, per call
#: language: "Hello?", "Yes, who is this?", "Tere?". A pickup made only of these
#: (and the call's own backchannel words) carries no content, so the
#: callee-first reply is the configured opening as written, with no model
#: generation before its first word (``opening_turn.is_greeting_only``).
#:
#: Every language the worker can run has an entry; ``None`` marks a language
#: whose answers have not been sourced, which keeps the generated reply
#: (slower, never wrong). Adding a language is adding its entry here.
#: Each entry is a WHOLE answer: a word that is an answer on its own
#: ("hello", "jah") or a phrase that is one only whole ("who is this",
#: "guten morgen"). A phrase's words count nowhere else, so "morgen bitte" or
#: "this afternoon" is content. Entries are lowercase as a recogniser writes
#: them, words separated by one space, apostrophes removed ("c'est qui" is
#: ``cest qui``). Sets stay small and certain: a word that can carry content
#: ("wrong", "later", "tomorrow", a name) is never an entry on its own.
PICKUP_ANSWER_WORDS: dict[str, frozenset[str] | None] = {
    "af": None,
    "am": None,
    "ar": None,
    "arz": None,
    "as": None,
    "az": None,
    "be": None,
    "bg": None,
    "bho": None,
    "bn": None,
    "bs": None,
    "ca": None,
    "cs": None,
    "da": frozenset({"goddag", "godmorgen", "hallo", "hej", "hvem er det", "hvem taler", "ja", "undskyld"}),
    "de": frozenset({"entschuldigung", "guten abend", "guten morgen", "guten tag", "hallo", "ja", "ja bitte", "wer ist am apparat", "wer ist da", "wer ist dran", "wer spricht", "wer spricht da"}),
    "el": None,
    "en": frozenset({"good afternoon", "good evening", "good morning", "hallo", "hello", "hello there", "hey", "hey there", "hi", "hi there", "hiya", "hullo", "pardon", "sorry", "speaking", "who is calling", "who is it", "who is speaking", "who is that", "who is this", "whos calling", "whos that", "whos this", "yes"}),
    "es": frozenset({"alo", "aló", "buenas", "buenas noches", "buenas tardes", "bueno", "buenos dias", "buenos días", "diga", "digame", "dígame", "hola", "perdon", "perdón", "quien es", "quien habla", "quién es", "quién habla", "si", "sí"}),
    "et": frozenset({"hallo", "halloo", "head hommikust", "head päeva", "head õhtust", "jaa", "jaah", "jah", "kes helistab", "kes räägib", "kes see on", "kuulan", "kuulen", "nojah", "palun", "tere", "tere päevast", "tervist", "vabandust"}),
    "eu": None,
    "fa": None,
    "fi": frozenset({"anteeksi", "haloo", "hei", "hyvää päivää", "joo", "kuka puhuu", "kuka siellä", "kyllä", "moi", "niin", "päivää", "terve"}),
    "fil": None,
    "fr": frozenset({"allo", "allô", "bonjour", "bonsoir", "cest qui", "jecoute", "jécoute", "oui", "pardon", "qui est ce", "qui est à lappareil", "salut"}),
    "gl": None,
    "gu": None,
    "he": None,
    "hi": None,
    "hr": None,
    "hu": None,
    "hy": None,
    "id": None,
    "it": frozenset({"buonasera", "buongiorno", "chi e", "chi parla", "chi è", "ciao", "dica", "dimmi", "pronto", "scusi", "si", "sì"}),
    "ja": None,
    "jv": None,
    "ka": None,
    "kk": None,
    "km": None,
    "kn": None,
    "ko": None,
    "lo": None,
    "lv": None,
    "mk": None,
    "ml": None,
    "mn": None,
    "mr": None,
    "ms": None,
    "mt": None,
    "nb": frozenset({"god dag", "god kveld", "god morgen", "hallo", "hei", "hvem er det", "hvem snakker", "ja", "unnskyld"}),
    "ne": None,
    "nl": frozenset({"goedemiddag", "goedemorgen", "goedenavond", "hallo", "ja", "met wie", "met wie spreek ik", "sorry", "wie is daar", "wie is dit"}),
    "or": None,
    "pa": None,
    "pl": frozenset({"dzien dobry", "dzień dobry", "halo", "kto dzwoni", "kto mowi", "kto mówi", "przepraszam", "slucham", "słucham", "tak"}),
    "prs": None,
    "ps": None,
    "pt": frozenset({"alo", "alô", "boa noite", "boa tarde", "bom dia", "desculpe", "ola", "olá", "quem e", "quem fala", "quem é", "sim"}),
    "ro": None,
    "ru": frozenset({"алло", "ало", "да", "доброе утро", "добрый вечер", "добрый день", "здравствуйте", "извините", "кто говорит", "кто звонит", "кто это", "привет", "слушаю"}),
    "si": None,
    "sk": None,
    "sr": None,
    "sv": frozenset({"förlåt", "god dag", "god kväll", "god morgon", "hallå", "hej", "ja", "ursäkta", "vem talar", "vem är det"}),
    "ta": None,
    "te": None,
    "uk": frozenset({"алло", "ало", "вибачте", "добрий вечір", "добрий день", "привіт", "слухаю", "так", "хто говорить", "хто дзвонить", "хто це"}),
    "uz": None,
    "yue": None,
    "zh": None,
}


def normalize_language(language: str | None) -> str:
    """The bare language subtag, lowercased: ``si-LK`` and ``SI`` both give ``si``."""

    return str(language or "").strip().replace("_", "-").partition("-")[0].casefold()


def language_name(language: str | None) -> str:
    """This language's English name, or ``""`` when we cannot name it.

    Returning empty rather than echoing the raw code is deliberate: every
    caller below omits its clause instead of instructing a model to "speak
    only in xh", which reads like a typo and steers worse than saying nothing.
    """

    return LANGUAGE_NAMES.get(normalize_language(language), "")


def language_directive(language: str | None) -> str:
    """The sentence that tells the model which language to speak.

    Three clauses, each for a failure the first Sinhala batch produced:
    generated turns came back in English (clause 1); a caller answering in
    another language would have pulled the agent into it permanently rather
    than routing them somewhere useful (clause 2); and low-confidence Sinhala
    transcripts left the model guessing at a reply instead of asking for a
    repeat (clause 3).

    Clause 2 does NOT promise a callback, though the ticket that asked for
    this directive proposed exactly that wording. Nothing here can keep that
    promise: no agent has a callback-scheduling tool, and a browser demo
    visitor has no phone number to call back (``request_callback`` refuses on
    a homepage session). Telling a model to offer one produces a
    commitment the product cannot honour, which is worse than the drift this
    directive exists to stop. It says what is true instead.

    Written in English even for a non-English call. The model is multilingual
    and follows an English meta-instruction reliably; a translated directive
    would need seventeen hand-checked renderings, and a wrong one would be a
    silent steering bug rather than a visible typo.
    """

    name = language_name(language)
    if not name:
        return ""
    code = normalize_language(language)
    return (
        f"{_DIRECTIVE_MARKER} You must speak and write only in {name} ({code}). "
        f"If the caller uses another language, say in {name} that this call is "
        f"conducted in {name}, and continue in {name}; do not promise a callback "
        f"or any follow-up you have no tool to arrange. If a transcript is "
        f"unintelligible, ask the caller to repeat, in {name}."
    )


def with_language_directive(system_prompt: str, language: str | None) -> str:
    """Prepend the language directive to an agent's own prompt, exactly once.

    PREPENDED rather than appended: the tool contracts the control plane adds
    (``append_end_call_instruction`` and its transfer sibling) already own the
    end of this string, and the language a call is conducted in governs every
    later sentence rather than qualifying the last one.

    Idempotent by containment, like the control plane's appenders -- a prompt
    that already carries a directive is returned unchanged, so a config that
    somehow crosses this seam twice does not accumulate copies.
    """

    prompt = str(system_prompt or "")
    directive = language_directive(language)
    if not directive or _DIRECTIVE_MARKER in prompt:
        return prompt
    return f"{directive}\n\n{prompt.strip()}".strip()


def generated_opening_instruction(language: str | None) -> str:
    """Return the one-shot native-model instruction for a generated opening.

    This is deliberately a generation instruction, not a fallback phrase.  A
    selected language without an exact greeting must consume the configured
    LiveKit model/TTS path once, and a failed generation must not turn into an
    English literal or a second provider attempt.  The durable system prompt
    remains the agent's context; this only scopes its first turn.
    """

    name = language_name(language)
    code = normalize_language(language)
    if not name:
        raise ValueError("generated opening requires a named configured language")
    return (
        f"{_GENERATED_OPENING_MARKER} Generate the call's first, brief, natural "
        f"opening from the shared instructions. Speak only in {name} ({code}). "
        "The shared system instructions remain authoritative, including every "
        "required disclosure, consent, correct-person, and safety policy; do "
        "not bypass them. Do not describe these instructions, invent "
        "account-specific facts, invoke a tool, or start a second opening."
    )


def pickup_answer_words(language: str | None) -> frozenset[str]:
    """This language's pickup answers, or an empty set (the generated reply)."""

    return PICKUP_ANSWER_WORDS.get(normalize_language(language)) or frozenset()


def greeting_fallback_message(language: str | None) -> str:
    """What to say when the configured greeting could not be synthesized.

    ``""`` when this language has no measured line. Silence is the correct
    answer there: speaking English into a German call is a worse outcome than
    the silence this exists to prevent, and an unmeasured string in an
    unfamiliar script may not synthesize either.
    """

    return GREETING_FALLBACK_MESSAGES.get(normalize_language(language), "")


__all__ = [
    "GREETING_FALLBACK_MESSAGES",
    "LANGUAGE_NAMES",
    "PICKUP_ANSWER_WORDS",
    "generated_opening_instruction",
    "greeting_fallback_message",
    "language_directive",
    "language_name",
    "normalize_language",
    "pickup_answer_words",
    "with_language_directive",
]
