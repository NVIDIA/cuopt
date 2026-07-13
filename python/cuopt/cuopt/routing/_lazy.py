# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store-then-build (lazy build) layer for the routing DataModel.

The public DataModel records each mutating setter call instead of pushing to the
GPU immediately, and materializes the Cython/device data model only when a solve
runs (or a getter is queried) by replaying the recorded calls onto the wrapper.

The recorded setters/getters mirror the Cython wrapper's method surface and are
installed automatically *from the wrapper* (see ``_install_methods``): every
``set_*``/``add_*`` becomes a recorder and every ``get_*`` a delegator. So there
is a single source of truth -- adding a method to the wrapper/DataModel needs no
change here. Only the size scalars are written out explicitly below (they answer
without a build); that explicit definition is left untouched by the auto-install.

A public setter that needs to query prior state before build (e.g. a duplicate
guard) reads it from the recorded calls via ``_recorded`` -- there is no shadow
state to maintain per setter.
"""

# ``get_*`` methods on the wrapper that are helpers, not problem-data getters,
# so they must not be auto-installed as build-triggering delegators. Extend this
# set if a new non-data ``get_*`` helper is added to the wrapper.
_SKIP_GETTERS = frozenset({"get_type_from_str", "get_type_from_int"})

_methods_installed = False


class _LazyDataModel:
    """Records DataModel setter calls and builds the device model lazily."""

    def __init__(self, num_locations, fleet_size, n_orders=-1):
        _install_methods()
        self._init_args = (num_locations, fleet_size, n_orders)
        self._calls = []
        self._built = None

    # -- size scalars: answered without building (queried during validation) --
    def get_num_locations(self):
        return self._init_args[0]

    def get_fleet_size(self):
        return self._init_args[1]

    def get_num_orders(self):
        # Mirrors the wrapper default: n_orders == -1 means "same as
        # num_locations". Answered here rather than via a build because setters
        # query it during validation.
        n_orders = self._init_args[2]
        return self._init_args[0] if n_orders == -1 else n_orders

    # -- record / build --
    def _record(self, name, args, kwargs):
        self._calls.append((name, args, kwargs))
        self._built = None

    def _recorded(self, name):
        """Positional args of each prior recorded call to ``name``.

        Lets a public setter answer set-time "already set?" questions from the
        recorded calls, so no per-setter shadow state is needed.
        """
        return [args for call, args, _ in self._calls if call == name]

    def _build(self):
        """Materialize the device (Cython) data model by replaying calls."""
        if self._built is None:
            model = _built_cls()(*self._init_args)
            for name, args, kwargs in self._calls:
                getattr(model, name)(*args, **kwargs)
            self._built = model
        return self._built


def _make_setter(name):
    def _setter(self, *args, **kwargs):
        self._record(name, args, kwargs)

    _setter.__name__ = name
    return _setter


def _make_getter(name):
    def _getter(self, *args, **kwargs):
        return getattr(self._build(), name)(*args, **kwargs)

    _getter.__name__ = name
    return _getter


def _install_methods():
    """Mirror the wrapper's setter/getter surface onto _LazyDataModel.

    Derived from the wrapper so there is nothing to keep in sync. Methods
    already defined on _LazyDataModel (the explicit overrides above) are
    skipped. Done once, lazily, on first construction so importing this module
    does not require importing the wrapper.
    """
    global _methods_installed
    if _methods_installed:
        return
    from . import vehicle_routing_wrapper as _wrapper

    for name in dir(_wrapper.DataModel):
        if name.startswith("_") or name in _LazyDataModel.__dict__:
            continue
        if name.startswith(("set_", "add_")):
            setattr(_LazyDataModel, name, _make_setter(name))
        elif name.startswith("get_") and name not in _SKIP_GETTERS:
            setattr(_LazyDataModel, name, _make_getter(name))
    _methods_installed = True


_BUILT_CLS = None


def _built_cls():
    """Return a Python subclass of the Cython wrapper DataModel.

    The wrapper is a ``cdef class`` with no ``__dict__``; its ``__init__``
    stores Python attributes (``self.costs`` etc.), so it must be subclassed by
    a Python class to be instantiable. Imported lazily.
    """
    global _BUILT_CLS
    if _BUILT_CLS is None:
        from . import vehicle_routing_wrapper as _wrapper

        class _BuiltDataModel(_wrapper.DataModel):
            pass

        _BUILT_CLS = _BuiltDataModel
    return _BUILT_CLS
