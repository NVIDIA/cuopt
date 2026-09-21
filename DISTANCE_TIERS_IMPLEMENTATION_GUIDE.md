# Guía de Implementación: Distance Tiers API

Esta guía describe cómo agregar el parámetro `distance_tiers` al payload de cuOpt para permitir costos escalonados por distancia.

## 1. Formato de Datos

Los `distance_tiers` se pasarán como una lista de tramos por vehículo. Cada tramo tiene:
- `threshold`: Umbral de distancia
- `fixed_cost`: Coste fijo si aplica
- `cost_per_unit`: Coste por unidad de distancia

### Ejemplo de Uso en Python:

```python
import cudf
from cuopt import routing

# Definir tramos de distancia para cada vehículo
# Formato: lista de diccionarios con 'threshold', 'fixed_cost', 'cost_per_unit'
distance_tiers_vehicle_0 = [
    {"threshold": 100.0, "fixed_cost": 50.0, "cost_per_unit": 0.0},   # < 100 km: coste fijo 50
    {"threshold": 200.0, "fixed_cost": 0.0, "cost_per_unit": 0.1},    # 100-200 km: 0.1 por km
    {"threshold": float('inf'), "fixed_cost": 0.0, "cost_per_unit": 0.5}  # > 200 km: 0.5 por km
]

distance_tiers_vehicle_1 = [
    {"threshold": 150.0, "fixed_cost": 75.0, "cost_per_unit": 0.0},
    {"threshold": float('inf'), "fixed_cost": 0.0, "cost_per_unit": 0.3}
]

# Convertir a formato plano para pasar a cuOpt
# Se almacenará como tres arrays paralelos por vehículo
n_vehicles = 2
n_tiers_per_vehicle = [3, 2]  # vehículo 0 tiene 3 tramos, vehículo 1 tiene 2

# Preparar datos en formato DataFrames
import pandas as pd

tiers_data = {
    'vehicle_id': [0, 0, 0, 1, 1],
    'threshold': [100.0, 200.0, float('inf'), 150.0, float('inf')],
    'fixed_cost': [50.0, 0.0, 0.0, 75.0, 0.0],
    'cost_per_unit': [0.0, 0.1, 0.5, 0.0, 0.3]
}

data_model = routing.DataModel(n_locations=10, fleet_size=2)
data_model.set_vehicle_distance_tiers(
    vehicle_ids=cudf.Series([0, 0, 0, 1, 1]),
    thresholds=cudf.Series([100.0, 200.0, float('inf'), 150.0, float('inf')]),
    fixed_costs=cudf.Series([50.0, 0.0, 0.0, 75.0, 0.0]),
    costs_per_unit=cudf.Series([0.0, 0.1, 0.5, 0.0, 0.3])
)
```

## 2. Archivos a Modificar

### 2.1 Python: `python/cuopt/cuopt/routing/vehicle_routing_wrapper.pyx`

Agregar en `__init__` del DataModel:
```python
self.vehicle_distance_tier_offsets = cudf.Series()  # Offsets para cada vehículo
self.distance_tier_thresholds = cudf.Series()
self.distance_tier_fixed_costs = cudf.Series()
self.distance_tier_costs_per_unit = cudf.Series()
```

Agregar método:
```python
def set_vehicle_distance_tiers(self, vehicle_ids, thresholds, fixed_costs, costs_per_unit):
    """
    Set distance tiers for tiered pricing based on route distance.

    Parameters
    ----------
    vehicle_ids : cudf.Series dtype - int32
        Vehicle ID for each tier entry
    thresholds : cudf.Series dtype - float32
        Distance thresholds for each tier
    fixed_costs : cudf.Series dtype - float32
        Fixed cost for each tier (use 0 if not applicable)
    costs_per_unit : cudf.Series dtype - float32
        Cost per unit distance for each tier
    """
    # Sort by vehicle_id to ensure proper grouping
    df = cudf.DataFrame({
        'vehicle_id': vehicle_ids,
        'threshold': thresholds,
        'fixed_cost': fixed_costs,
        'cost_per_unit': costs_per_unit
    }).sort_values('vehicle_id')

    # Store data
    self.distance_tier_thresholds = type_cast(df['threshold'], np.float32, "thresholds")
    self.distance_tier_fixed_costs = type_cast(df['fixed_cost'], np.float32, "fixed_costs")
    self.distance_tier_costs_per_unit = type_cast(df['cost_per_unit'], np.float32, "costs_per_unit")

    # Calculate offsets for each vehicle
    offsets = [0]
    for vid in range(self.get_fleet_size()):
        count = (df['vehicle_id'] == vid).sum()
        offsets.append(offsets[-1] + count)

    self.vehicle_distance_tier_offsets = cudf.Series(offsets, dtype=np.int32)

    # Pass to C++
    cdef uintptr_t c_thresholds = self.distance_tier_thresholds.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_fixed_costs = self.distance_tier_fixed_costs.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_costs_per_unit = self.distance_tier_costs_per_unit.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_offsets = self.vehicle_distance_tier_offsets.__cuda_array_interface__['data'][0]

    self.c_data_model_view.get().set_vehicle_distance_tiers(
        <float*>c_thresholds,
        <float*>c_fixed_costs,
        <float*>c_costs_per_unit,
        <int*>c_offsets,
        <int>len(self.distance_tier_thresholds)
    )
```

### 2.2 Python: `python/cuopt/cuopt/routing/vehicle_routing.pxd`

Agregar declaración:
```python
void set_vehicle_distance_tiers(
    const f_t* thresholds,
    const f_t* fixed_costs,
    const f_t* costs_per_unit,
    const i_t* offsets,
    i_t total_tiers
) except+
```

### 2.3 Python: `python/cuopt/cuopt/routing/vehicle_routing.py`

Agregar método con validación:
```python
@catch_cuopt_exception
def set_vehicle_distance_tiers(self, vehicle_ids, thresholds, fixed_costs, costs_per_unit):
    """
    Set distance-based tiered pricing for vehicles.

    Each vehicle can have multiple distance tiers with different cost structures.
    For each tier, you can specify either a fixed cost or a cost per unit distance.

    Parameters
    ----------
    vehicle_ids : cudf.Series dtype - int32
        Vehicle ID for each tier entry. Tiers for the same vehicle should be
        consecutive and sorted by threshold.
    thresholds : cudf.Series dtype - float32
        Distance thresholds for each tier. Use float('inf') for the last tier.
    fixed_costs : cudf.Series dtype - float32
        Fixed cost for each tier. Use 0.0 if the tier uses cost_per_unit instead.
    costs_per_unit : cudf.Series dtype - float32
        Cost per distance unit for each tier. Use 0.0 if the tier uses fixed_cost instead.

    Examples
    --------
    >>> from cuopt import routing
    >>> import cudf
    >>> import numpy as np
    >>>
    >>> # Define tiers for 2 vehicles
    >>> # Vehicle 0: <100km = 50 fixed, 100-200km = 0.1/km, >200km = 0.5/km
    >>> # Vehicle 1: <150km = 75 fixed, >150km = 0.3/km
    >>>
    >>> vehicle_ids = cudf.Series([0, 0, 0, 1, 1], dtype=np.int32)
    >>> thresholds = cudf.Series([100.0, 200.0, float('inf'), 150.0, float('inf')])
    >>> fixed_costs = cudf.Series([50.0, 0.0, 0.0, 75.0, 0.0])
    >>> costs_per_unit = cudf.Series([0.0, 0.1, 0.5, 0.0, 0.3])
    >>>
    >>> data_model = routing.DataModel(n_locations=10, fleet_size=2)
    >>> data_model.set_vehicle_distance_tiers(
    ...     vehicle_ids, thresholds, fixed_costs, costs_per_unit
    ... )
    """
    # Validations
    if len(vehicle_ids) != len(thresholds) or len(vehicle_ids) != len(fixed_costs) or len(vehicle_ids) != len(costs_per_unit):
        raise ValueError("All input series must have the same length")

    validate_non_negative(thresholds, "thresholds")
    validate_non_negative(fixed_costs, "fixed_costs")
    validate_non_negative(costs_per_unit, "costs_per_unit")

    # Check that vehicle IDs are valid
    max_vehicle_id = vehicle_ids.max()
    if max_vehicle_id >= self.get_fleet_size():
        raise ValueError(f"vehicle_ids contains {max_vehicle_id} but fleet size is {self.get_fleet_size()}")

    super().set_vehicle_distance_tiers(vehicle_ids, thresholds, fixed_costs, costs_per_unit)
```

### 2.4 C++: Agregar en `cpp/include/cuopt/routing/data_model.hpp`

```cpp
void set_vehicle_distance_tiers(
    f_t const* thresholds,
    f_t const* fixed_costs,
    f_t const* costs_per_unit,
    i_t const* offsets,
    i_t total_tiers
);
```

### 2.5 C++: Implementar en `cpp/src/routing/data_model.cu`

```cpp
template <typename i_t, typename f_t>
void data_model_view_t<i_t, f_t>::set_vehicle_distance_tiers(
    f_t const* thresholds,
    f_t const* fixed_costs,
    f_t const* costs_per_unit,
    i_t const* offsets,
    i_t total_tiers)
{
    distance_tier_thresholds_ = thresholds;
    distance_tier_fixed_costs_ = fixed_costs;
    distance_tier_costs_per_unit_ = costs_per_unit;
    distance_tier_offsets_ = offsets;
    total_distance_tiers_ = total_tiers;
}
```

### 2.6 C++: Agregar campos en `cpp/src/routing/fleet_info.hpp`

En la clase `fleet_info_t`, agregar:
```cpp
rmm::device_uvector<f_t> v_distance_tier_thresholds_;
rmm::device_uvector<f_t> v_distance_tier_fixed_costs_;
rmm::device_uvector<f_t> v_distance_tier_costs_per_unit_;
rmm::device_uvector<i_t> v_distance_tier_offsets_;
```

Y en el método `get_vehicle_info`, agregar:
```cpp
// Set distance tiers span for this vehicle
i_t tier_start = v_distance_tier_offsets_[vehicle_id];
i_t tier_end = v_distance_tier_offsets_[vehicle_id + 1];
i_t n_tiers = tier_end - tier_start;

if (n_tiers > 0) {
    // Create distance_tier_t array for this vehicle
    // This requires temporary storage or a view
    info.distance_tiers = raft::span<distance_tier_t<f_t> const>(
        /* pointer to distance_tier_t array */,
        n_tiers
    );
}
```

## 3. Integración en `fleet_info_t`

Necesitarás crear un método que empaquete los tres arrays (thresholds, fixed_costs, costs_per_unit)
en un array de `distance_tier_t` structures durante la población de fleet_info.

## 4. Testing

```python
import cuopt
import cudf
import numpy as np

# Create simple test
n_locations = 5
n_vehicles = 2

data_model = cuopt.routing.DataModel(n_locations, n_vehicles)

# Set cost matrix
cost_matrix = np.array([
    [0, 10, 20, 30, 40],
    [10, 0, 15, 25, 35],
    [20, 15, 0, 20, 30],
    [30, 25, 20, 0, 25],
    [40, 35, 30, 25, 0]
])
data_model.add_cost_matrix(cudf.DataFrame(cost_matrix))

# Set distance tiers
vehicle_ids = cudf.Series([0, 0, 0, 1, 1], dtype=np.int32)
thresholds = cudf.Series([100.0, 200.0, float('inf'), 150.0, float('inf')])
fixed_costs = cudf.Series([50.0, 0.0, 0.0, 75.0, 0.0])
costs_per_unit = cudf.Series([0.0, 0.1, 0.5, 0.0, 0.3])

data_model.set_vehicle_distance_tiers(
    vehicle_ids, thresholds, fixed_costs, costs_per_unit
)

# Solve
solver_settings = cuopt.routing.SolverSettings()
solver = cuopt.routing.Solver(data_model, solver_settings)
solution = solver.solve()

print(solution.get_status())
```

## 5. Notas Importantes

- Los tramos deben estar ordenados por `threshold` en orden ascendente para cada vehículo
- El último tramo debe tener `threshold = float('inf')` o un valor muy grande
- Para cada tramo, se debe usar SOLO fixed_cost O cost_per_unit (el otro debe ser 0)
- La lógica en `calculate_tiered_cost` ya está implementada en C++

## 6. Formato Alternativo Simplificado

Si prefieres una API más simple, podrías crear un helper:

```python
def create_distance_tiers(tiers_by_vehicle):
    """
    Helper to create distance tiers from a more readable format.

    Parameters
    ----------
    tiers_by_vehicle : list of list of dict
        Each element is a list of tier dictionaries for that vehicle

    Example
    -------
    tiers = [
        # Vehicle 0
        [
            {"threshold": 100, "fixed_cost": 50},
            {"threshold": 200, "cost_per_unit": 0.1},
            {"threshold": float('inf'), "cost_per_unit": 0.5}
        ],
        # Vehicle 1
        [
            {"threshold": 150, "fixed_cost": 75},
            {"threshold": float('inf'), "cost_per_unit": 0.3}
        ]
    ]
    """
    vehicle_ids = []
    thresholds = []
    fixed_costs = []
    costs_per_unit = []

    for vehicle_id, tiers in enumerate(tiers_by_vehicle):
        for tier in tiers:
            vehicle_ids.append(vehicle_id)
            thresholds.append(tier["threshold"])
            fixed_costs.append(tier.get("fixed_cost", 0.0))
            costs_per_unit.append(tier.get("cost_per_unit", 0.0))

    return (
        cudf.Series(vehicle_ids, dtype=np.int32),
        cudf.Series(thresholds),
        cudf.Series(fixed_costs),
        cudf.Series(costs_per_unit)
    )
```
