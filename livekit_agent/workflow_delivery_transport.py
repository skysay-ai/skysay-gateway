# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Call-bound async journal port; source and start grants are never retried.

Cancellation retains ownership of the synchronous HTTP operation until it
joins. The controller can durably fence an unknown start while that happens.
"""
from __future__ import annotations

import asyncio
import copy
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable

try:
    from .workflow_speech import SpeechBinding, SpeechRefused, WorkflowSpeechGate, speech_step_token
    from .workflow_turns import authorized_spoken_text
except ImportError:
    from workflow_speech import SpeechBinding, SpeechRefused, WorkflowSpeechGate, speech_step_token
    from workflow_turns import authorized_spoken_text

_HASH = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[0-9a-f]{32}")
_ATTEMPT = re.compile(r"[A-Za-z0-9_-]{16,96}")
_FINAL = {"completed_local", "interrupted", "uncertain", "aborted_unexposed"}
#: A line the call's live GPT session speaks from an instruction: no sealed
#: PCM exists, so its ``pcm_digest`` is the SHA-256 of the exact instruction
#: sent to the session.
LIVE_APPROVAL_KIND = "live_session"


@dataclass(frozen=True, slots=True)
class DeliveryOwner:
    attempt_id: str
    owner_token: str = field(repr=False)
    authorization_digest: str


class WorkflowDeliveryTransport:
    def __init__(self, *, call_id: str, run_id: str, submission: dict[str, Any],
                 reply: dict[str, Any], step_index: int,
                 send: Callable[[dict[str, Any]], dict[str, Any]], observe_sources=None) -> None:
        state, steps = reply.get("runtime_state"), reply.get("steps")
        if (not isinstance(state, dict) or type(state.get("turn_count")) is not int
                or not isinstance(steps, list) or type(step_index) is not int
                or not 0 <= step_index < len(steps) or not isinstance(steps[step_index], dict)
                or steps[step_index].get("spoken") is not True
                or reply.get("request_id") != submission.get("request_id")
                or reply.get("expected_prior_turn") != submission.get("expected_prior_turn")
                or not isinstance(reply.get("pinned_contract"), dict)
                or not isinstance(call_id, str) or not call_id
                or not isinstance(run_id, str) or not run_id):
            raise SpeechRefused("delivery requires its acknowledged spoken turn")
        self.call_id, self.run_id = call_id, run_id
        self._submission = copy.deepcopy(submission)
        self._contract = copy.deepcopy(reply["pinned_contract"])
        self._turn = state["turn_count"]
        self._step, self._step_index = steps[step_index].get("step_id"), step_index
        self._send, self._observe_sources = send, observe_sources

    async def _request(self, payload: dict[str, Any], *, abandoned_claim: str | None = None) -> dict[str, Any]:
        operation = asyncio.create_task(asyncio.to_thread(self._send, payload))
        cancelled = False
        while True:
            try:
                response = await asyncio.shield(operation)
                break
            except asyncio.CancelledError:
                cancelled = True
                if operation.done():
                    try:
                        response = operation.result()
                        break
                    except BaseException:
                        raise asyncio.CancelledError from None
            except Exception:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise
        if not isinstance(response, dict) or response.get("call_id") != self.call_id:
            if cancelled:
                raise asyncio.CancelledError
            raise SpeechRefused("delivery response belongs to a different call")
        if cancelled:
            # A response already returned by the owned original HTTP request
            # may contain a known fresh lease. Retire only that validated
            # lease; never retry claim or fetch an unknown owner's token.
            if abandoned_claim is not None:
                try:
                    owner = self._claim_owner(response, abandoned_claim)
                    if owner is not None:
                        cleanup = asyncio.create_task(self.finish(owner, "aborted_unexposed"))
                        while not cleanup.done():
                            try:
                                await asyncio.shield(cleanup)
                            except asyncio.CancelledError:
                                continue
                        cleanup.result()
                except Exception:
                    # Unknown cleanup remains conservatively fenced.
                    pass
            raise asyncio.CancelledError
        return response

    async def present(self, boundary: dict[str, int]) -> None:
        """Join the exact whole-reply ACK before input can answer this question."""
        if (not isinstance(boundary, dict) or set(boundary) != {"input_epoch", "boundary_sample", "sample_rate"}
                or any(type(value) is not int for value in boundary.values())
                or not 0 <= boundary["input_epoch"] <= 2147483647
                or not 0 <= boundary["boundary_sample"] < 2**53
                or boundary["sample_rate"] not in (8000, 16000, 24000, 48000)):
            raise SpeechRefused("presentation requires a complete bounded input frontier")
        expected = {"request_id": self._submission["request_id"], "step_id": self._step,
                    "boundary": dict(boundary)}
        response = await self._request({"command": "present", "request_id": expected["request_id"],
                                        "boundary": dict(boundary)})
        fact = response.get("presentation")
        if (not isinstance(fact, dict) or set(fact) != {*expected, "language"}
                or any(fact[key] != value for key, value in expected.items())
                or not isinstance(fact["language"], str) or not 1 <= len(fact["language"]) <= 64):
            raise SpeechRefused("presentation acknowledgement differs from the controlled reply")

    @staticmethod
    def _fact(response: dict[str, Any], owner: DeliveryOwner | None = None, *,
              approval_kinds=(None, "authored_tts", LIVE_APPROVAL_KIND)) -> dict[str, Any]:
        fact = response.get("fact")
        fields = {"attempt_id", "status", "authorization_digest", "pcm_digest", "approval_kind", "exposure_started_at", "finished_at"}
        if (not isinstance(fact, dict) or set(fact) != fields
                or not isinstance(fact.get("attempt_id"), str) or _ATTEMPT.fullmatch(fact["attempt_id"]) is None
                or not isinstance(fact.get("authorization_digest"), str)
                or _HASH.fullmatch(fact["authorization_digest"]) is None
                or not isinstance(fact.get("status"), str)
                or fact["status"] not in {"capturing", "approved", "exposure_started", *_FINAL}
                or (owner is not None and (fact["attempt_id"] != owner.attempt_id
                                          or fact["authorization_digest"] != owner.authorization_digest))):
            raise SpeechRefused("delivery response contains an invalid lifecycle fact")
        status, pcm = fact["status"], fact["pcm_digest"]
        kind = fact["approval_kind"]
        exposed, finished = fact["exposure_started_at"], fact["finished_at"]
        if (kind not in approval_kinds
                or (status not in {"capturing", "aborted_unexposed"} and kind is None)
                or not isinstance(pcm, str) or (pcm != "" and _HASH.fullmatch(pcm) is None)
                or any(t is not None and (not isinstance(t, str) or not t or len(t) > 64)
                       for t in (exposed, finished))
                or ((status in _FINAL) != (finished is not None))
                or ((status in {"exposure_started", "completed_local", "interrupted", "uncertain"}) != (exposed is not None))
                or (status == "capturing" and (pcm != "" or kind is not None))
                or (status not in {"capturing", "aborted_unexposed"} and pcm == "")):
            raise SpeechRefused("delivery lifecycle fields contradict their status")
        return fact

    async def claim(self, binding: SpeechBinding, gate: WorkflowSpeechGate, language: str) -> DeliveryOwner | None:
        line = gate.line
        expected = (self.call_id, self.run_id, self._turn, speech_step_token(self._step), line.digest, gate.media_epoch)
        actual = (binding.call_id, binding.run_id, binding.turn_index, binding.step_id,
                  binding.line_digest, binding.media_epoch)
        if (actual != expected or (line.call_id, line.run_id, line.turn_index, line.step_id) != expected[:4]
                or not 0 <= binding.ordinal < len(line.segments)
                or line.text != authorized_spoken_text(self._contract, self._step, language)):
            raise SpeechRefused("speech binding differs from its acknowledged published line")
        response = await self._request({"command": "claim", "submission": copy.deepcopy(self._submission), "spec": {
            "attempt_id": binding.attempt_id, "step_index": self._step_index, "segment_index": binding.ordinal,
            "segments": list(line.segments), "language": language, "media_epoch": binding.media_epoch}},
            abandoned_claim=binding.attempt_id)
        return self._claim_owner(response, binding.attempt_id)

    async def claim_live(self, *, attempt_id: str, text: str, language: str, media_epoch: int) -> DeliveryOwner | None:
        """One whole published line as one segment: the live session speaks it, unsegmented."""
        if (not isinstance(attempt_id, str) or _ATTEMPT.fullmatch(attempt_id) is None
                or type(media_epoch) is not int or not 0 <= media_epoch <= 2147483647
                or not isinstance(text, str) or not text.strip()
                or text != authorized_spoken_text(self._contract, self._step, language)):
            raise SpeechRefused("live delivery differs from its acknowledged published line")
        response = await self._request({"command": "claim", "submission": copy.deepcopy(self._submission), "spec": {
            "attempt_id": attempt_id, "step_index": self._step_index, "segment_index": 0,
            "segments": [text], "language": language, "media_epoch": media_epoch}}, abandoned_claim=attempt_id)
        return self._claim_owner(response, attempt_id)

    def _claim_owner(self, response: dict[str, Any], attempt_id: str) -> DeliveryOwner | None:
        fact = self._fact(response)
        if fact["attempt_id"] != attempt_id or "lease" not in response:
            raise SpeechRefused("delivery claim does not match its source attempt")
        lease = response["lease"]
        if lease is None:
            return None
        if (not isinstance(lease, dict) or set(lease) != {"attempt_id", "owner_token", "authorization_digest"}
                or lease["attempt_id"] != attempt_id
                or lease["authorization_digest"] != fact["authorization_digest"]
                or not isinstance(lease["owner_token"], str) or _TOKEN.fullmatch(lease["owner_token"]) is None
                or fact["status"] != "capturing" or fact["pcm_digest"] != ""
                or fact["exposure_started_at"] is not None or fact["finished_at"] is not None):
            raise SpeechRefused("delivery response contains no fresh source ownership")
        return DeliveryOwner(**lease)

    async def allow_authored_source(self, owner: DeliveryOwner) -> None:
        response = await self._request({"command": "allow-authored-source", "lease": asdict(owner)})
        if response.get("allowed") is not True:
            raise SpeechRefused("authored TTS source has no engine authorization")

    async def approve_authored(self, owner: DeliveryOwner, pcm_digest: str) -> None:
        await self._approve(owner, pcm_digest, "authored_tts")

    async def approve_live(self, owner: DeliveryOwner, instruction_digest: str) -> None:
        if not isinstance(instruction_digest, str) or _HASH.fullmatch(instruction_digest) is None:
            raise SpeechRefused("live delivery approval requires its instruction digest")
        await self._approve(owner, instruction_digest, LIVE_APPROVAL_KIND)

    async def _approve(self, owner: DeliveryOwner, pcm_digest: str, kind: str) -> None:
        response = await self._request({"command": "approve", "lease": asdict(owner), "pcm_digest": pcm_digest, "approval_kind": kind})
        fact = self._fact(response, owner)
        if (fact["status"] != "approved" or fact["pcm_digest"] != pcm_digest or fact["approval_kind"] != kind
                or fact["exposure_started_at"] is not None or fact["finished_at"] is not None):
            raise SpeechRefused("delivery did not acknowledge the exact source approval")

    async def start(self, owner: DeliveryOwner) -> bool:
        response = await self._request({"command": "start", "lease": asdict(owner)})
        if type(response.get("granted")) is not bool:
            raise SpeechRefused("delivery start omitted its one-time grant")
        return response["granted"]

    async def finish(self, owner: DeliveryOwner, status: str) -> str:
        response = await self._request({"command": "finish", "lease": asdict(owner), "status": status})
        fact = self._fact(response, owner)
        if fact["status"] not in _FINAL or not isinstance(fact["finished_at"], str) or not fact["finished_at"]:
            raise SpeechRefused("delivery omitted its terminal acknowledgement")
        return fact["status"]

    @staticmethod
    def _floor_ack(response, owner, generation):
        if (response.get("attempt_id") != owner.attempt_id
                or response.get("authorization_digest") != owner.authorization_digest
                or type(response.get("generation")) is not int or response["generation"] != generation):
            raise SpeechRefused("business floor acknowledgement changed its owner or generation")
        until = response.get("lease_until")
        try:
            if not isinstance(until, str) or len(until) != 20:
                raise ValueError
            parsed = datetime.fromisoformat(until.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.isoformat(timespec="seconds").replace("+00:00", "Z") != until:
                raise ValueError
        except ValueError:
            raise SpeechRefused("business floor acknowledgement has no bounded lease") from None

    async def park_business(self, owner: DeliveryOwner, *, generation: int) -> int:
        if type(owner) is not DeliveryOwner or type(generation) is not int or not 0 <= generation < 128:
            raise SpeechRefused("business park requires its existing owner and generation")
        response = await self._request({"command": "park-business", "lease": asdict(owner), "generation": generation})
        self._floor_ack(response, owner, generation + 1)
        return response["generation"]

    async def resume_business(self, owner: DeliveryOwner, *, generation: int) -> bool:
        if type(owner) is not DeliveryOwner or type(generation) is not int or not 1 <= generation <= 128:
            raise SpeechRefused("business resume requires its existing paused generation")
        response = await self._request({"command": "resume-business", "lease": asdict(owner), "generation": generation})
        self._floor_ack(response, owner, generation)
        if type(response.get("granted")) is not bool:
            raise SpeechRefused("business resume omitted its one-time grant")
        return response["granted"]

    def source_journal(self):
        try:
            from .workflow_source_transport import WorkflowSourceTransport
        except ImportError:
            from workflow_source_transport import WorkflowSourceTransport
        return WorkflowSourceTransport(call_id=self.call_id, run_id=self.run_id, send=self._send, observe=self._observe_sources)


class WorkflowControllerTransport(WorkflowDeliveryTransport):
    """A call-bound controller has no acknowledged business-turn authority."""
    def __init__(self, *, call_id, run_id, spec, send, observe_sources=None):
        if (type(call_id) is not str or not call_id or type(run_id) is not str or not run_id
                or type(spec) is not dict):
            raise SpeechRefused("controller speech requires its call-bound specification")
        self.call_id, self.run_id = call_id, run_id
        self._spec = copy.deepcopy(spec)
        self._send, self._observe_sources = send, observe_sources

    @staticmethod
    def _fact(response, owner=None):
        return WorkflowDeliveryTransport._fact(response, owner,
            approval_kinds=(None, "authored_tts", "controller_composed"))

    async def _request(self, payload, *, abandoned_claim=None):
        if payload.get("command") in {"claim", "allow-authored-source", "approve", "start", "finish"}:
            payload = {**payload, "command": "controller-" + payload["command"]}
        return await super()._request(payload, abandoned_claim=abandoned_claim)

    async def claim_spec(self, spec):
        if type(spec) is not dict or not isinstance(spec.get("attempt_id"), str):
            raise SpeechRefused("controller claim has no attempt identity")
        response = await self._request({"command": "claim", "spec": copy.deepcopy(spec)},
                                       abandoned_claim=spec["attempt_id"])
        return self._claim_owner(response, spec["attempt_id"])

    async def claim(self, binding, gate, language):
        line = gate.line
        expected = (self.call_id, self.run_id, line.turn_index, line.step_id, line.digest, gate.media_epoch)
        actual = (binding.call_id, binding.run_id, binding.turn_index, binding.step_id,
                  binding.line_digest, binding.media_epoch)
        if (line.call_id != self.call_id or line.run_id != self.run_id or actual != expected
                or binding.ordinal != 0 or len(line.segments) != 1):
            raise SpeechRefused("controller source differs from its call-bound exact line")
        return await self.claim_spec({**self._spec, "attempt_id": binding.attempt_id,
            "segments": list(line.segments), "language": language, "media_epoch": binding.media_epoch})

    async def present(self, _boundary):
        raise SpeechRefused("controller speech cannot present a business question")

    async def approve_live(self, _owner, _instruction_digest):
        raise SpeechRefused("controller speech cannot borrow live session approval")

    async def approve_composed(self, owner, pcm_digest):
        response = await self._request({"command": "approve", "lease": asdict(owner),
                                       "pcm_digest": pcm_digest, "approval_kind": "controller_composed"})
        fact = self._fact(response, owner)
        if (fact["status"] != "approved" or fact["approval_kind"] != "controller_composed"
                or type(fact["pcm_digest"]) is not str or _HASH.fullmatch(fact["pcm_digest"]) is None
                or fact["exposure_started_at"] is not None or fact["finished_at"] is not None):
            raise SpeechRefused("controller composition has no keyed source acknowledgement")
        return fact["pcm_digest"]
