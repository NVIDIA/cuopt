# API Integration Summary: Distance Tiers

## ✅ Cambios Completados en el Servidor API

### 1. **Definición del Payload** (`data_definition.py`)

**Archivo**: `python/cuopt_server/cuopt_server/utils/routing/data_definition.py`

✅ **Agregado campo `vehicle_distance_tiers`** en la clase `FleetData` (líneas 463-512):

```python
vehicle_distance_tiers: Optional[List[List[Dict[str, float]]]] = Field(
    default=None,
    examples=[...],
    description="Distance-based tiered pricing for each vehicle..."
)
```

### 2. **Procesamiento del Payload** (`solver.py`)

**Archivo**: `python/cuopt_server/cuopt_server/utils/routing/solver.py`

✅ **Agregada lógica de procesamiento** (líneas 268-289):
- Convierte el formato de lista de diccionarios a arrays planos
- Llama a `data_model.set_vehicle_distance_tiers()`

### 3. **Ejemplo Actualizado**

✅ **Actualizado `vrp_example_data`** en `data_definition.py` (líneas 1086-1096)

### 4. **Ejemplo de Uso de API**

✅ **Creado** `examples/api_distance_tiers_example.py`
- Muestra cómo llamar al API REST
- Incluye ejemplos con requests Python
- Incluye comando curl

## 📋 Formato del Payload JSON

### Estructura del Request

```json
{
  "cost_matrix_data": { ... },
  "fleet_data": {
    "vehicle_locations": [[0, 0], [0, 0]],
    "vehicle_ids": ["vehicle-0", "vehicle-1"],
    ...
    "vehicle_distance_tiers": [
      [
        {
          "threshold": 100.0,
          "fixed_cost": 50.0,
          "cost_per_unit": 0.0
        },
        {
          "threshold": 200.0,
          "fixed_cost": 0.0,
          "cost_per_unit": 0.1
        },
        {
          "threshold": 1e9,
          "fixed_cost": 0.0,
          "cost_per_unit": 0.5
        }
      ],
      [
        {
          "threshold": 150.0,
          "fixed_cost": 75.0,
          "cost_per_unit": 0.0
        },
        {
          "threshold": 1e9,
          "fixed_cost": 0.0,
          "cost_per_unit": 0.3
        }
      ]
    ]
  },
  "task_data": { ... },
  "solver_config": { ... }
}
```

### Interpretación

Para el ejemplo anterior:

**Vehículo 0:**
- Distancia < 100 km → Coste fijo: 50
- 100 ≤ Distancia < 200 km → Coste: distancia × 0.1
- Distancia ≥ 200 km → Coste: distancia × 0.5

**Vehículo 1:**
- Distancia < 150 km → Coste fijo: 75
- Distancia ≥ 150 km → Coste: distancia × 0.3

## 🚀 Cómo Desplegar y Probar

### 1. **Iniciar el Servidor cuOpt** (Self-Hosted)

```bash
# Desde el directorio raíz del proyecto
cd python/cuopt_server

# Instalar dependencias (si no está instalado)
pip install -e .

# Iniciar servidor
python -m cuopt_server.webserver
```

Por defecto, el servidor se inicia en `http://localhost:5000`

### 2. **Enviar Request con Distance Tiers**

#### Opción A: Usando Python

```python
import requests
import json

payload = {
    "fleet_data": {
        "vehicle_distance_tiers": [
            [
                {"threshold": 100.0, "fixed_cost": 50.0, "cost_per_unit": 0.0},
                {"threshold": 200.0, "fixed_cost": 0.0, "cost_per_unit": 0.1},
                {"threshold": 1e9, "fixed_cost": 0.0, "cost_per_unit": 0.5}
            ]
        ],
        # ... otros campos
    },
    # ... resto del payload
}

response = requests.post(
    "http://localhost:5000/cuopt/request",
    json=payload
)

print(response.json())
```

#### Opción B: Usando cURL

```bash
curl -X POST http://localhost:5000/cuopt/request \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### 3. **Ejecutar Ejemplo**

```bash
# Asegúrate de que el servidor esté corriendo
python examples/api_distance_tiers_example.py
```

## 📡 Endpoints Disponibles

### POST `/cuopt/request`

Endpoint principal para enviar problemas de routing.

**Headers:**
- `Content-Type: application/json`
- `Accept: application/json` (opcional)

**Query Parameters:**
- `cache`: bool - Si True, cachea los datos y devuelve un ID
- `validation_only`: bool - Si True, solo valida sin resolver

**Response:**
- Si es asíncrono: `{"reqId": "uuid"}`
- Si es síncrono: Solución completa

### GET `/cuopt/request/{id}`

Consultar el estado de un request asíncrono.

**Response:**
```json
{
  "status": "Finished",  // o "Running", "Failed"
  "response": {
    "solver_response": {
      "vehicle_data": {...},
      "cost": 123.45
    }
  }
}
```

## 🔍 Validación del Payload

El servidor valida automáticamente:

✅ **Estructura del payload** (Pydantic)
✅ **Tipos de datos** (int32, float32)
✅ **Valores no negativos** (thresholds, costs)
✅ **Coherencia** (longitud de arrays)

### Errores Comunes

**Error 422: Validation Error**
```json
{
  "detail": [
    {
      "loc": ["fleet_data", "vehicle_distance_tiers", 0, 0, "threshold"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

**Solución**: Verificar tipos de datos y formato

## 📚 Documentación API (Swagger)

Una vez que el servidor esté corriendo, la documentación interactiva está disponible en:

- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

Allí podrás ver:
- Esquemas de datos completos
- Ejemplos interactivos
- Probar requests directamente desde el navegador

## 🔄 Flujo Completo de Datos

```
┌─────────────────┐
│  Client/User    │
│  (JSON Payload) │
└────────┬────────┘
         │
         │ POST /cuopt/request
         ▼
┌─────────────────────────────────────────┐
│ webserver.py                            │
│ - Recibe payload                        │
│ - Valida con data_definition.py        │
│ - Crea SolverJob                        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ solver.py                               │
│ - Procesa fleet_data                    │
│ - Convierte vehicle_distance_tiers      │
│ - Llama data_model.set_vehicle_distance_tiers()
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ vehicle_routing.py (cuOpt API)          │
│ - Valida parámetros                     │
│ - Llama vehicle_routing_wrapper.pyx     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ vehicle_routing_wrapper.pyx (Cython)    │
│ - Convierte a formato C++               │
│ - Llama c_data_model_view.get().set_... │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ data_model_view_t (C++)                 │
│ - Almacena punteros a datos             │
│ - Propaga a fleet_info_t                │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ fleet_info_t & VehicleInfo              │
│ - Construye distance_tier_t structs     │
│ - Disponible para cálculo de costos     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ distance_route_t::calculate_tiered_cost │
│ - Aplica lógica de tramos               │
│ - Retorna costo calculado               │
└─────────────────────────────────────────┘
```

## ⚠️ Pendiente (C++)

Para que funcione end-to-end, aún necesitas implementar en C++:

1. ❌ `data_model_view_t::set_vehicle_distance_tiers()`
2. ❌ Propagación en `fleet_info_t`
3. ❌ Construcción de arrays de `distance_tier_t`
4. ❌ Población en `get_vehicle_info()`

Ver `DISTANCE_TIERS_SUMMARY.md` para detalles de implementación C++.

## 🧪 Testing

### Test Unitario del API

```python
def test_distance_tiers_payload():
    payload = {
        "fleet_data": {
            "vehicle_distance_tiers": [
                [
                    {"threshold": 100.0, "fixed_cost": 50.0, "cost_per_unit": 0.0}
                ]
            ],
            # ... otros campos requeridos
        },
        # ... resto del payload
    }

    response = requests.post(API_URL, json=payload)
    assert response.status_code == 200
```

### Validación de Formato

```python
from pydantic import ValidationError
from cuopt_server.utils.routing.data_definition import FleetData

try:
    fleet_data = FleetData(
        vehicle_locations=[[0, 0]],
        vehicle_distance_tiers=[
            [
                {"threshold": 100.0, "fixed_cost": 50.0, "cost_per_unit": 0.0}
            ]
        ]
    )
    print("✓ Validación exitosa")
except ValidationError as e:
    print(f"✗ Error de validación: {e}")
```

## 📞 Soporte

Si encuentras problemas:

1. Verifica que el servidor esté corriendo
2. Revisa los logs del servidor para errores
3. Valida el formato del payload contra el schema
4. Consulta la documentación Swagger en `/docs`
5. Revisa `examples/api_distance_tiers_example.py` para referencia

## 📝 Logs del Servidor

Los logs del servidor mostrarán el procesamiento:

```
INFO: Processing fleet_data.vehicle_distance_tiers
DEBUG: Converting tiers for 2 vehicles
DEBUG: Total tiers: 5
DEBUG: Calling data_model.set_vehicle_distance_tiers()
INFO: Vehicle distance tiers configured successfully
```

Para habilitar logs detallados:

```bash
export LOG_LEVEL=DEBUG
python -m cuopt_server.webserver
```
