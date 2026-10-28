# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One bounded live workflow submission, retained across an uncertain response.

An identical later utterance is still a new business turn. Only an explicit
retry reuses the pending request; words are never used as an idempotency key.
"""
from __future__ import annotations

import threading
import copy
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from .capabilities.workflow_listener import MAX_CALLER_TURN_CHARACTERS
except ImportError:
    from capabilities.workflow_listener import MAX_CALLER_TURN_CHARACTERS


class WorkflowRequestError(RuntimeError):
    pass


#: How the worker's transfer dial can end (``transfer.TransferController``);
#: the control plane accepts exactly these as a ``transfer_outcome`` input.
TRANSFER_OUTCOMES = frozenset({"answered", "no_answer", "busy", "rejected", "failed", "caller_hangup", "caller_cancelled"})


def _input(value: Any) -> dict[str, Any]:
    """Validate and own the current wire observation before retaining a retry."""
    fields = {"opening": {"kind"}, "resume": {"kind", "origin"}, "interrupt": {"kind", "text", "origin"},
              "answer": {"kind", "text", "origin", "answers_request_id"}, "transfer_outcome": {"kind", "outcome"}, "action_result": {"kind", "action_id"}}
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise WorkflowRequestError("workflow input requires an explicit kind")
    kind = value["kind"]
    if kind not in fields or set(value) != fields[kind]:
        raise WorkflowRequestError("workflow input has invalid fields")
    if kind == "opening":
        return {"kind": "opening"}
    if kind == "transfer_outcome":
        if not isinstance(value["outcome"], str) or value["outcome"] not in TRANSFER_OUTCOMES:
            raise WorkflowRequestError("workflow transfer outcome is not one the dial reports")
        return {"kind": "transfer_outcome", "outcome": value["outcome"]}
    if kind == "action_result":
        action_id = value["action_id"]
        if not isinstance(action_id, str) or re.fullmatch(r"action_[A-Za-z0-9]{1,100}", action_id) is None:
            raise WorkflowRequestError("action result requires a bounded action identity")
        return {"kind": "action_result", "action_id": action_id}
    origin = value["origin"]
    if kind != "resume":
        text = value["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_CALLER_TURN_CHARACTERS:
            raise WorkflowRequestError("workflow input requires bounded nonblank text")
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise WorkflowRequestError("workflow input requires UTF-8 text") from None
    if (not isinstance(origin, dict) or set(origin) != {"input_epoch", "first_sample", "end_sample", "sample_rate"}
            or any(type(v) is not int for v in origin.values())
            or not 0 <= origin["input_epoch"] <= 2147483647
            or not 0 <= origin["first_sample"] < origin["end_sample"] < 2**53
            or origin["sample_rate"] not in (8000, 16000, 24000, 48000)):
        raise WorkflowRequestError("workflow input requires its bounded original sample clock")
    if kind == "answer" and (not isinstance(value["answers_request_id"], str)
            or re.fullmatch(r"[A-Za-z0-9_-]{16,96}", value["answers_request_id"]) is None):
        raise WorkflowRequestError("workflow answer requires its original question")
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class _Submission:
    request_id: str
    expected_prior_turn: int
    input: dict[str, Any] = field(repr=False)

    def payload(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "expected_prior_turn": self.expected_prior_turn,
                "input": copy.deepcopy(self.input)}


class WorkflowTurnRequests:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turn = 0
        self._pending: _Submission | None = None
        self._acknowledged: _Submission | None = None
        self._in_flight = False
        self._terminal = False

    def advance(self, voice_input: dict[str, Any], *,
                send: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        observation = _input(voice_input)
        with self._lock:
            if self._terminal:
                raise WorkflowRequestError("workflow already returned its terminal response")
            if self._pending is not None:
                raise WorkflowRequestError("retry the unresolved workflow request before starting another turn")
            if observation["kind"] == "opening" and self._turn != 0:
                raise WorkflowRequestError("opening input cannot resume a workflow")
            if observation["kind"] in ("resume", "transfer_outcome", "action_result") and self._turn == 0:
                raise WorkflowRequestError(f"{observation['kind']} input requires a preceding reply")
            if observation["kind"] == "answer" and (self._turn == 0 or self._acknowledged is None
                    or observation["answers_request_id"] != self._acknowledged.request_id):
                raise WorkflowRequestError("answer input requires its last acknowledged question")
            self._pending = _Submission(uuid.uuid4().hex, self._turn, observation)
            self._acknowledged = None
            # Reserve ownership before releasing the lock: a concurrent retry
            # cannot send the new submission ahead of its creator.
            self._in_flight = True
            submission = self._pending
        return self._send(submission, send)

    def retry(self, *, send: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            if self._pending is None or self._in_flight:
                raise WorkflowRequestError("no retryable workflow request is available")
            self._in_flight = True
            submission = self._pending
        return self._send(submission, send)

    def acknowledged_submission(self, request_id: str) -> dict[str, Any]:
        """Private bounded authority for presentation of the latest reply only."""
        with self._lock:
            submission = self._acknowledged
            if submission is None or submission.request_id != request_id:
                raise WorkflowRequestError("no matching acknowledged workflow submission")
            return submission.payload()

    def forget_acknowledged(self, request_id: str) -> None:
        with self._lock:
            if self._acknowledged is not None and self._acknowledged.request_id == request_id:
                self._acknowledged = None

    def _send(self, submission: _Submission, send: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        try:
            response = send(submission.payload())
            state = response.get("runtime_state") if isinstance(response, dict) else None
            turn = state.get("turn_count") if isinstance(state, dict) else None
            if (not isinstance(response, dict) or response.get("request_id") != submission.request_id
                    or type(response.get("expected_prior_turn")) is not int
                    or response["expected_prior_turn"] != submission.expected_prior_turn
                    or type(turn) is not int or not submission.expected_prior_turn <= turn <= submission.expected_prior_turn + 1
                    or turn > 200 or response.get("status") not in ("running", "completed", "transferred", "failed_closed")):
                raise WorkflowRequestError("workflow response does not match its submitted request")
            with self._lock:
                self._turn = turn
                self._pending = None
                self._acknowledged = submission
                self._terminal = response["status"] != "running"
                self._in_flight = False
            return response
        finally:
            with self._lock:
                if self._pending is submission:
                    self._in_flight = False
