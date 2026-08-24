# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Versioned voice capability data shared by the control plane and voice worker.

The package deliberately contains no provider SDK import.  Both runtime
artifacts load the same checked-in JSON document through :mod:`registry` and
expose its content as immutable values.
"""
