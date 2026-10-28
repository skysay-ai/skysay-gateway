# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Generated engine properties. Run scripts/generate_conversation_engines.py."""
from types import MappingProxyType
from typing import Any

_ROWS = {'cascade': {'has_language_axis': True,
             'required_runtime_fields': ('voice_id', 'tts_model', 'llm_model', 'system_prompt'),
             'speaks_text': True,
             'reports_caller_turns': False,
             'billable_components': ('stt', 'llm', 'tts'),
             'workflow_output_role': 'authored',
             'supports_manual_replies': True,
             'supports_navigation': True,
             'supports_deferred_tool_replies': True,
             'self_hostable': True,
             'display': 'Cascade'},
 'gpt_live': {'has_language_axis': False,
              'required_runtime_fields': (),
              'speaks_text': False,
              'reports_caller_turns': False,
              'billable_components': ('llm', 'tts'),
              'workflow_output_role': 'live',
              'supports_manual_replies': True,
              'supports_navigation': False,
              'supports_deferred_tool_replies': True,
              'self_hostable': True,
              'display': 'GPT-Live'},
 'gemini_live': {'has_language_axis': False,
                 'required_runtime_fields': (),
                 'speaks_text': False,
                 'reports_caller_turns': True,
                 'billable_components': ('llm', 'tts'),
                 'workflow_output_role': None,
                 'supports_manual_replies': True,
                 'supports_navigation': True,
                 'supports_deferred_tool_replies': True,
                 'self_hostable': True,
                 'display': 'Gemini Live'},
 'eleven_realtime': {'has_language_axis': False,
                     'required_runtime_fields': (),
                     'speaks_text': False,
                     'reports_caller_turns': True,
                     'billable_components': ('realtime',),
                     'workflow_output_role': None,
                     'supports_manual_replies': False,
                     'supports_navigation': False,
                     'supports_deferred_tool_replies': False,
                     'self_hostable': False,
                     'display': 'Eleven Realtime'}}
ENGINE_DECLARATIONS = MappingProxyType({key: MappingProxyType(value) for key, value in _ROWS.items()})
DEFAULT_CONVERSATION_ENGINE = 'cascade'
WORKFLOW_BILLABLE_COMPONENTS = ('stt', 'tts')
DEFERRED_TOOL_NAMES = frozenset(['cancel_transfer', 'transfer_to_human'])
ELEVEN_REALTIME_MODEL = 'eleven_realtime_v1_max'

IMPLIED_BUILTIN_TOOLS = {'transfer_to_human': ('request_callback', 'cancel_transfer')}

def implied_tool_names(names: Any) -> frozenset[str]:
    """The built-ins ``names`` bring with them, excluding ``names`` themselves."""
    present = {str(name) for name in names or ()}
    implied = {extra for trigger in present for extra in IMPLIED_BUILTIN_TOOLS.get(trigger, ())}
    return frozenset(implied - present)

def deferred_tool_conflicts(supports_deferred_tool_replies: bool, names) -> tuple[str, ...]:
    """Configured and implied tools whose result protocol this engine cannot serve."""
    if supports_deferred_tool_replies:
        return ()
    entries = names.split(",") if isinstance(names, str) else names or ()
    configured = {str(item.get("name") or "").strip() if isinstance(item, dict)
                  else str(item).strip() for item in entries}
    expanded = configured | set(implied_tool_names(configured))
    return tuple(sorted(expanded & DEFERRED_TOOL_NAMES))

def engine_deferred_tool_conflicts(engine_id, names):
    return deferred_tool_conflicts(ENGINE_DECLARATIONS[engine_id]['supports_deferred_tool_replies'], names)
