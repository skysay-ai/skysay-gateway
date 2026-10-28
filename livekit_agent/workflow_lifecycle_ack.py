# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Joined start acknowledgements choose exactly one root settlement phase."""
from __future__ import annotations

import asyncio

try:
    from .control_plane import ControlPlaneRejected, SessionStateError
    from .voice_configuration import ProductionConfigurationError
except ImportError:
    from control_plane import ControlPlaneRejected, SessionStateError
    from voice_configuration import ProductionConfigurationError


class WorkflowLifecycleAcknowledgements:
    def __init__(self, call_id):
        self.call_id = call_id
        self.media_started = self.answer_acknowledged = self.uncertain = False

    async def report(self, operation, *, event_id, canonical_type):
        task = asyncio.create_task(operation)
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
            except Exception:
                break
        try:
            response = task.result()
        except asyncio.CancelledError:
            self.uncertain = True
            raise cancellation or asyncio.CancelledError()
        except Exception as exc:
            if not (isinstance(exc, SessionStateError) or isinstance(exc, ControlPlaneRejected)
                    and exc.status_code in {400, 401, 403, 404, 409, 422}):
                self.uncertain = True
                if cancellation is not None:
                    raise cancellation
                raise ProductionConfigurationError("workflow start acknowledgement is unresolved",
                    reason="workflow_lifecycle_ack_unknown") from exc
            if cancellation is not None:
                raise cancellation
            raise
        event = response.get("event") if type(response) is dict else None
        call = response.get("call") if type(response) is dict else None
        payload = event.get("payload") if type(event) is dict else None
        if (type(response) is not dict or response.get("canonical_type") != canonical_type
                or type(call) is not dict or call.get("id") != self.call_id
                or type(event) is not dict or event.get("type") != canonical_type or event.get("call_id") != self.call_id
                or type(payload) is not dict or payload.get("gateway_event_id") != event_id
                or payload.get("source_component") != ("livekit_room" if canonical_type == "media.started" else "livekit_sip")):
            self.uncertain = True
            if cancellation is not None:
                raise cancellation
            raise ProductionConfigurationError("workflow start acknowledgement differs from the owned event",
                reason="workflow_lifecycle_ack_unknown")
        if canonical_type == "media.started":
            self.media_started = True
        elif canonical_type == "call.answered":
            self.answer_acknowledged = True
        else:
            raise ValueError("workflow start acknowledgement requires a start event")
        if cancellation is not None:
            raise cancellation
        return response
