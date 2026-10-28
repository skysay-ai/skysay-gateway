# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Portable Skysay evidence and RTP utilities.

The production media runtime is LiveKit and remains separate from this package.
This package stays import-light so a legacy codec adapter can remain narrow
without recreating a media runtime.
"""

__all__ = ["__version__"]
__version__ = "0.2.0"
