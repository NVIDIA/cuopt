# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuopt import routing
from cuopt.routing import vehicle_routing_wrapper
from cuopt.routing._deferred import _SKIP_GETTERS


def test_deferred_covers_wrapper_surface():
    """Every public wrapper DataModel method must be handled by the recording
    layer (installed as a recorder/getter or an explicit override). This fails
    loudly if a new wrapper method -- e.g. a mutator not named set_*/add_* --
    is added without being recorded, instead of silently dropping its data.
    """
    dm = routing.DataModel(1, 1)  # triggers _install_methods
    handled = set(dir(type(dm)))
    missing = [
        name
        for name in dir(vehicle_routing_wrapper.DataModel)
        if not name.startswith("_")
        and name not in _SKIP_GETTERS
        and name not in handled
    ]
    assert not missing, (
        f"deferred-build layer does not handle wrapper methods {missing}; "
        "add a recorder/getter or list them in _SKIP_GETTERS"
    )


def test_unknown_method_is_not_recorded():
    """A call to a method that is not part of the DataModel surface raises
    AttributeError rather than being silently recorded. The recording layer
    installs only the declared setters/getters (no ``__getattr__`` catch-all),
    so a typo'd or unknown call fails loudly and never enters the IR.
    """
    dm = routing.DataModel(3, 1)
    with pytest.raises(AttributeError):
        dm.random_func_call(1, 2, 3)
    assert dm._calls == []
