# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""OpenPhonex media gateway (RTP transport + realtime voice pipeline).

Kept intentionally import-light. Importing this package must NOT pull in
``realtime`` / ``pipeline`` / ``worker``: the ``telephony_ai.media`` compat shim
does ``import openphonex_gateway`` purely to borrow this package's ``__path__``,
and eager submodule imports here would load those modules under the
``openphonex_gateway.*`` name and create a dual module identity versus
``telephony_ai.media.*`` (breaking ``mock.patch`` / ``isinstance``).
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
