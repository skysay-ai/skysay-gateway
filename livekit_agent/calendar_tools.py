# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One calendar function contract for API, voice and browser engines."""

CALENDAR_TOOL_DECLARATIONS = {
    "calendar_check_availability": {
        "name": "calendar_check_availability",
        "description": "Find up to five available appointment slots. Use offset-bearing timestamps; ask the caller about their timezone if unclear. Offers expire in three minutes. The result also says which caller details a booking needs and whether the caller is sent a confirmation email.",
        "parameters": {"type": "object", "properties": {
            "start_utc": {"type": "string"}, "end_utc": {"type": "string"},
        }, "required": ["start_utc", "end_utc"], "additionalProperties": False},
    },
    "calendar_book_appointment": {
        "name": "calendar_book_appointment",
        "description": "Prepare one offered appointment for confirmation. The server returns its complete read-back and yes/no question; speak it and wait. Only a distinct explicit caller confirmation permits booking. Never claim a booking succeeded unless its durable result says booked. Changed details require a new read-back. Never retry an unknown outcome.",
        "parameters": {"type": "object", "properties": {
            "offer_id": {"type": "string"}, "caller_name": {"type": "string"},
            "caller_email": {"type": "string"},
            "answers": {"type": "object", "additionalProperties": {"type": "string"}},
        }, "required": ["offer_id", "caller_name"], "additionalProperties": False},
    },
}
