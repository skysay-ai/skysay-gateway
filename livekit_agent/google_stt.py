# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Google Cloud Speech-to-Text adapter that keeps the endpointing the API accepts.

The pinned LiveKit Google STT plugin (``livekit-plugins-google`` 1.7.1) takes an
``endpointing_sensitivity`` argument and then discards it for every model other
than ``chirp_3``, with a warning, before the request ever leaves the host::

    if is_given(endpointing_sensitivity) and model != "chirp_3":
        logger.warning("endpointing_sensitivity is only supported with the chirp_3 model; ignoring.")
        endpointing_sensitivity = NOT_GIVEN

That restriction is the plugin's, not Google's.  Measured from the production
voice host on 2026-09-05: Speech-to-Text V2 accepted ``SHORT`` and
``SUPERSHORT`` on ``chirp_2`` in 42 of 42 streams, and with ``SHORT`` the FINAL
transcript after the caller stopped arrived 358-557 ms sooner than the default
(en-US 850 ms against 1208 ms, si-LK 1002 ms against 1559 ms; ``europe-west4``,
same audio at wall-clock pace) with every word kept.  ``chirp_3`` at the same
settings was not faster, so this adapter is how the measured gain reaches the
Chirp 2 stacks the catalog already sells, rather than a reason to add a model.
The worker asks for ``SHORT``.  ``SUPERSHORT`` is accepted here because the API
accepts it, but it fragmented short replies into two finals and returned a
Sinhala account number as a word in six of six streams, so nothing in this
tree selects it.

The adapter is deliberately narrow.  Construction goes through the upstream
plugin unchanged -- credentials, project resolution, connection pool, the
``chirp_3`` word-timestamp rule, every default -- and only afterwards is the
sensitivity written onto the plugin's own ``STTOptions``.  The streaming
request builder reads that field (``StreamingRecognitionFeatures(...,
endpointing_sensitivity=...)``) with no second model check, which is what makes
the option reach the wire.  Nothing else about the stream changes: results,
interims, timeouts and the shared connection pool are the upstream plugin's.

The sensitivity only shortens how long Google waits after speech before
committing the FINAL.  Whether a caller's turn has ended is still decided by
the worker's VAD and turn detector, not by this field, so a shorter wait cannot
end a turn earlier on its own; it makes the transcript the turn was already
waiting for arrive sooner.
"""

from __future__ import annotations

from typing import Any

from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.google import stt as upstream

#: The literal names Speech-to-Text V2 accepts, as the plugin's own
#: ``EndpointingSensitivity`` literal type spells them.  ``UNSPECIFIED`` is the
#: proto default and would be an instruction to change nothing; refusing it here
#: keeps "no setting" spelled as ``NOT_GIVEN`` in exactly one way.
ENDPOINTING_SENSITIVITIES = frozenset(
    {
        "ENDPOINTING_SENSITIVITY_STANDARD",
        "ENDPOINTING_SENSITIVITY_SHORT",
        "ENDPOINTING_SENSITIVITY_SUPERSHORT",
    }
)


class STT(upstream.STT):
    """The upstream plugin with ``endpointing_sensitivity`` honoured on every V2 model."""

    def __init__(
        self,
        *,
        endpointing_sensitivity: NotGivenOr[str] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        if is_given(endpointing_sensitivity) and endpointing_sensitivity not in ENDPOINTING_SENSITIVITIES:
            raise ValueError(
                "endpointing_sensitivity must be one of "
                + ", ".join(sorted(ENDPOINTING_SENSITIVITIES))
                + f", not {endpointing_sensitivity!r}"
            )
        # Never hand the option to the upstream constructor: for any model but
        # chirp_3 it would be dropped with a warning, and the warning would be
        # the only trace.  Writing it onto the options object afterwards is the
        # one seam the streaming request builder reads it from.
        super().__init__(**kwargs)
        if is_given(endpointing_sensitivity):
            self._config.endpointing_sensitivity = endpointing_sensitivity

    @property
    def endpointing_sensitivity(self) -> str | None:
        """The sensitivity every stream from this adapter requests, or ``None``."""
        value = self._config.endpointing_sensitivity
        return str(value) if is_given(value) else None
