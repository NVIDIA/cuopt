# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuopt_sh_client.cuopt_sh import seconds_or_none


@pytest.mark.parametrize("value", ["None", "none", " None "])
def test_http_timeout_accepts_none(value):
    # The flag's own help says "Set to None to never timeout", and
    # CuOptServiceSelfHostClient accepts None for http_general_timeout on
    # purpose. type=int rejected the spelling the help asks for.
    assert seconds_or_none(value) is None


@pytest.mark.parametrize(
    ("value", "expected"), [("30", 30), ("0", 0), ("600", 600)]
)
def test_http_timeout_still_takes_seconds(value, expected):
    assert seconds_or_none(value) == expected


@pytest.mark.parametrize("value", ["abc", "", "4.5s"])
def test_http_timeout_rejects_nonsense(value):
    with pytest.raises(ValueError):
        seconds_or_none(value)
