# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deferred-build layer for the routing DataModel: record setter calls, build on demand.

The public DataModel **records** each setter call on the host instead of applying
it to the GPU immediately. The class does not build anything itself -- it stores
the problem and defers construction of the device (Cython) model until it is
actually needed (at solve, or when a getter is queried), at which point it builds
transiently by replaying the recorded calls onto the wrapper. "Deferred" refers
to that deferred build; the mechanism is call recording. Design points:

  * **Host-resident IR.** Host inputs (numpy/pandas) are copied to a numpy array
    the DataModel owns at record time, so the recorded calls form a serializable,
    host-resident intermediate representation (what remote/gRPC serialization
    needs) and the user's original array can be released. **Device inputs
    (cuDF/cupy) are kept on the device** to preserve the zero-copy GPU path;
    they are exported to host only when explicitly serialized.
  * **Transient build.** The device model is never cached: each solve/getter
    builds it fresh and discards it, so the instance does not hold both a host
    IR and a device copy.
  * **Zero-CUDA construction.** The setter/getter surface is declared explicitly
    below rather than introspected from the compiled wrapper, so constructing and
    serializing a problem imports no CUDA/cuDF -- the wrapper is imported lazily
    only when a device build actually happens. ``test_deferred`` fails if this
    declared surface drifts from the wrapper.

Adding a new setter/getter to the DataModel:

  1. Implement it on the C++ ``data_model_view_t`` and the Cython wrapper
     (``vehicle_routing_wrapper.pyx``).
  2. Add the public method in ``vehicle_routing.py`` -- validation and
     docstring -- forwarding to ``super()``.
  3. Add its name to ``_SETTERS`` (a mutator) or ``_GETTERS`` (a query) below.

``test_deferred`` cross-checks these names against the wrapper's surface, so a
missing entry fails CI rather than silently dropping the call's data.
"""

import threading

import numpy as np

# Mutating setters: recorded and replayed onto the device model at build time.
_SETTERS = (
    "add_break_dimension",
    "add_capacity_dimension",
    "add_cost_matrix",
    "add_initial_solutions",
    "add_order_precedence",
    "add_order_vehicle_match",
    "add_transit_time_matrix",
    "add_vehicle_break",
    "add_vehicle_order_match",
    "set_break_locations",
    "set_drop_return_trips",
    "set_min_vehicles",
    "set_objective_function",
    "set_order_locations",
    "set_order_prizes",
    "set_order_service_times",
    "set_order_time_windows",
    "set_pickup_delivery_pairs",
    "set_skip_first_trips",
    "set_vehicle_fixed_costs",
    "set_vehicle_locations",
    "set_vehicle_max_costs",
    "set_vehicle_max_times",
    "set_vehicle_time_windows",
    "set_vehicle_types",
)

# Non-scalar getters. Currently resolved via a transient build (see
# _make_getter); host-mirror resolution from the IR is migrated incrementally.
_GETTERS = (
    "get_break_dimensions",
    "get_break_locations",
    "get_capacity_dimensions",
    "get_cost_matrix",
    "get_drop_return_trips",
    "get_initial_solutions",
    "get_min_vehicles",
    "get_non_uniform_breaks",
    "get_objective_function",
    "get_order_locations",
    "get_order_prizes",
    "get_order_service_times",
    "get_order_time_windows",
    "get_order_vehicle_match",
    "get_pickup_delivery_pairs",
    "get_skip_first_trips",
    "get_transit_time_matrices",
    "get_transit_time_matrix",
    "get_vehicle_fixed_costs",
    "get_vehicle_locations",
    "get_vehicle_max_costs",
    "get_vehicle_max_times",
    "get_vehicle_order_match",
    "get_vehicle_time_windows",
    "get_vehicle_types",
)

# ``get_*`` methods on the wrapper that are helpers, not problem-data getters,
# so they are neither installed nor required by the coverage test.
_SKIP_GETTERS = frozenset({"get_type_from_str", "get_type_from_int"})

_install_lock = threading.Lock()
_methods_installed = False
_BUILT_CLS = None


def _namespace_of(x):
    """Container kind of ``x`` without importing cudf/pandas: one of
    {"numpy", "pandas", "cudf", "cupy"} or None (scalar / str / other).
    """
    if isinstance(x, np.ndarray):
        return "numpy"
    return {"pandas": "pandas", "cudf": "cudf", "cupy": "cupy"}.get(
        type(x).__module__.split(".", 1)[0]
    )


def _normalize(x):
    """Record-time normalization (mixed IR).

    Host numeric arrays are copied to a numpy array the DataModel owns (so the
    user's array can be released and the IR is serializable). Device (cuDF/cupy)
    inputs are kept as-is to preserve the zero-copy GPU path. Scalars, strings,
    and non-numeric arrays pass through unchanged.
    """
    ns = _namespace_of(x)
    if ns in ("cudf", "cupy"):
        return x
    if ns == "numpy":
        return np.array(x, copy=True) if x.dtype.kind in "biuf" else x
    if ns == "pandas":
        arr = x.to_numpy()
        return np.array(arr, copy=True) if arr.dtype.kind in "biuf" else x
    return x


class _DeferredDataModel:
    """Records DataModel setter calls; builds the device model transiently."""

    def __init__(self, num_locations, fleet_size, n_orders=-1):
        _install_methods()
        self._init_args = (num_locations, fleet_size, n_orders)
        self._calls = []

    # -- size scalars: answered without a build (queried during validation) --
    def get_num_locations(self):
        return self._init_args[0]

    def get_fleet_size(self):
        return self._init_args[1]

    def get_num_orders(self):
        # Mirrors the wrapper default: n_orders == -1 means "same as
        # num_locations".
        n_orders = self._init_args[2]
        return self._init_args[0] if n_orders == -1 else n_orders

    # -- record --
    def _record(self, name, args, kwargs):
        args = tuple(_normalize(a) for a in args)
        kwargs = {k: _normalize(v) for k, v in kwargs.items()}
        self._calls.append((name, args, kwargs))

    def _recorded(self, name):
        """Positional args of each prior recorded call to ``name`` (for
        set-time "already set?" checks; reads the IR, no shadow state).
        """
        return [args for call, args, _ in self._calls if call == name]

    # -- transient build (never cached; see module docstring) --
    def _build(self):
        """Build a fresh device (Cython) data model by replaying the calls."""
        model = _built_cls()(*self._init_args)
        for name, args, kwargs in self._calls:
            getattr(model, name)(*args, **kwargs)
        return model


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
    """Install recorders/getters from the declared surface (not the wrapper).

    Done once, lazily, so importing this module needs no CUDA. Thread-safe on
    first concurrent construction.
    """
    global _methods_installed
    if _methods_installed:
        return
    with _install_lock:
        if _methods_installed:
            return
        for name in _SETTERS:
            if name not in _DeferredDataModel.__dict__:
                setattr(_DeferredDataModel, name, _make_setter(name))
        for name in _GETTERS:
            if name not in _DeferredDataModel.__dict__:
                setattr(_DeferredDataModel, name, _make_getter(name))
        _methods_installed = True


def _built_cls():
    """Return a Python subclass of the Cython wrapper DataModel.

    The wrapper is a ``cdef class`` with no ``__dict__``; its ``__init__`` stores
    Python attributes, so it must be subclassed by a Python class to be
    instantiable. Imported lazily (this is the only CUDA/cuDF entry point).
    """
    global _BUILT_CLS
    if _BUILT_CLS is None:
        with _install_lock:
            if _BUILT_CLS is None:
                from . import vehicle_routing_wrapper as _wrapper

                class _BuiltDataModel(_wrapper.DataModel):
                    pass

                _BUILT_CLS = _BuiltDataModel
    return _BUILT_CLS
