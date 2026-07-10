# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store-then-build recording layer for the routing DataModel.

The public DataModel records each mutating setter call instead of pushing to the
GPU immediately, and materializes the Cython/device data model only when a solve
runs (or a getter is queried). This keeps the user's inputs available on the host
until they are actually needed on the device, which:

  * avoids eager device allocation while a problem is being assembled, and
  * exposes the recorded, host-resident inputs (``_calls``) for host-only
    problem construction (remote / gRPC serialization).

The device model is built by replaying the recorded calls onto the existing
Cython wrapper, so all conversion/validation behavior is unchanged.
"""

# Mutating setters -> recorded and replayed onto the device model at build time.
_SETTERS = (
    "add_break_dimension",
    "add_capacity_dimension",
    "add_initial_solutions",
    "add_order_precedence",
    "add_order_vehicle_match",
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

# Non-scalar getters -> answered from the built device model. The three size
# scalars are answered directly (below) because setters query them during
# validation, before any build should happen.
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


class _RecordingDataModel:
    """Records DataModel setter calls and builds the device model lazily."""

    def __init__(self, num_locations, fleet_size, n_orders=-1):
        self._init_args = (num_locations, fleet_size, n_orders)
        self._calls = []
        self._built = None
        # Track added matrix vehicle-types so the public layer's duplicate
        # check (``if vehicle_type in self.costs``) works before build.
        self.costs = {}
        self.transit_times = {}

    def add_cost_matrix(self, costs, vehicle_type=0):
        self.costs[vehicle_type] = None
        self._record("add_cost_matrix", (costs, vehicle_type), {})

    def add_transit_time_matrix(self, times, vehicle_type=0):
        self.transit_times[vehicle_type] = None
        self._record("add_transit_time_matrix", (times, vehicle_type), {})

    # -- size scalars: answered without building (queried during validation) --
    def get_num_locations(self):
        return self._init_args[0]

    def get_fleet_size(self):
        return self._init_args[1]

    def get_num_orders(self):
        n_orders = self._init_args[2]
        return self._init_args[0] if n_orders == -1 else n_orders

    # -- record / build --
    def _record(self, name, args, kwargs):
        self._calls.append((name, args, kwargs))
        self._built = None

    def _build(self):
        """Materialize the device (Cython) data model by replaying calls."""
        if self._built is None:
            model = _built_cls()(*self._init_args)
            for name, args, kwargs in self._calls:
                getattr(model, name)(*args, **kwargs)
            self._built = model
        return self._built


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


for _name in _SETTERS:
    setattr(_RecordingDataModel, _name, _make_setter(_name))
for _name in _GETTERS:
    setattr(_RecordingDataModel, _name, _make_getter(_name))
