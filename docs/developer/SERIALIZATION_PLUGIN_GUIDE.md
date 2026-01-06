# cuOpt Remote Solve Serialization Plugin Guide

This guide explains how to develop custom serialization plugins for cuOpt's remote solve feature. Plugins allow you to replace the default Protocol Buffers serialization with alternative formats like MsgPack, JSON, FlatBuffers, or custom binary protocols.

## Overview

The remote solve feature uses a pluggable serialization interface (`remote_serializer_t`) that handles:
- Serializing optimization problems (LP/MIP) for network transmission
- Deserializing solver settings
- Serializing solutions back to the client
- Message type identification (LP vs MIP)

```
┌─────────────┐                              ┌─────────────┐
│   Client    │                              │   Server    │
│             │   serialize_lp_request()     │             │
│  Problem ───┼──────────────────────────────┼──► Problem  │
│             │                              │             │
│             │   serialize_lp_solution()    │             │
│  Solution ◄─┼──────────────────────────────┼─── Solution │
└─────────────┘                              └─────────────┘
         ▲                                          ▲
         │                                          │
         └────────── Same Serializer ───────────────┘
```

**Important**: Both client and server must use the same serializer for communication to work.

## The Serializer Interface

Your plugin must implement the `remote_serializer_t<i_t, f_t>` interface defined in:
`cpp/include/cuopt/linear_programming/utilities/remote_serialization.hpp`

### Required Methods

```cpp
template <typename i_t, typename f_t>
class remote_serializer_t {
public:
  virtual ~remote_serializer_t() = default;

  // ═══════════════════════════════════════════════════════════════════
  // CLIENT-SIDE: Serialize requests, deserialize solutions
  // ═══════════════════════════════════════════════════════════════════

  // Serialize an LP problem and settings into bytes for transmission
  virtual std::vector<uint8_t> serialize_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& problem,
    const pdlp_solver_settings_t<i_t, f_t>& settings) = 0;

  // Serialize a MIP problem and settings into bytes
  virtual std::vector<uint8_t> serialize_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& problem,
    const mip_solver_settings_t<i_t, f_t>& settings) = 0;

  // Deserialize an LP solution from bytes received from server
  virtual optimization_problem_solution_t<i_t, f_t> deserialize_lp_solution(
    const std::vector<uint8_t>& data) = 0;

  // Deserialize a MIP solution from bytes
  virtual mip_solution_t<i_t, f_t> deserialize_mip_solution(
    const std::vector<uint8_t>& data) = 0;

  // ═══════════════════════════════════════════════════════════════════
  // SERVER-SIDE: Deserialize requests, serialize solutions
  // ═══════════════════════════════════════════════════════════════════

  // Check if the received data is a MIP request (vs LP)
  virtual bool is_mip_request(const std::vector<uint8_t>& data) = 0;

  // Deserialize LP request into problem data and settings
  virtual bool deserialize_lp_request(
    const std::vector<uint8_t>& data,
    mps_parser::mps_data_model_t<i_t, f_t>& problem_data,
    pdlp_solver_settings_t<i_t, f_t>& settings) = 0;

  // Deserialize MIP request into problem data and settings
  virtual bool deserialize_mip_request(
    const std::vector<uint8_t>& data,
    mps_parser::mps_data_model_t<i_t, f_t>& problem_data,
    mip_solver_settings_t<i_t, f_t>& settings) = 0;

  // Serialize LP solution for transmission back to client
  virtual std::vector<uint8_t> serialize_lp_solution(
    const optimization_problem_solution_t<i_t, f_t>& solution) = 0;

  // Serialize MIP solution
  virtual std::vector<uint8_t> serialize_mip_solution(
    const mip_solution_t<i_t, f_t>& solution) = 0;

  // ═══════════════════════════════════════════════════════════════════
  // METADATA
  // ═══════════════════════════════════════════════════════════════════

  // Human-readable format name (e.g., "msgpack", "json", "flatbuffers")
  virtual std::string format_name() const = 0;

  // Protocol version for compatibility checking
  virtual uint32_t protocol_version() const = 0;
};
```

### Factory Function

Your plugin must export a factory function that creates the serializer:

```cpp
extern "C" {
  // For int32_t indices, double floats (most common)
  std::unique_ptr<remote_serializer_t<int32_t, double>>
    create_cuopt_serializer_i32_f64();

  // Additional type combinations if needed
  std::unique_ptr<remote_serializer_t<int32_t, float>>
    create_cuopt_serializer_i32_f32();
}
```

## Step-by-Step Implementation

### Step 1: Create the Plugin Source File

Create `cpp/src/linear_programming/utilities/serializers/my_serializer.cpp`:

```cpp
#include <cuopt/linear_programming/utilities/remote_serialization.hpp>
#include <vector>
#include <cstring>

namespace cuopt::linear_programming {

// Message type identifiers (first byte of each message)
constexpr uint8_t MSG_LP_REQUEST   = 1;
constexpr uint8_t MSG_MIP_REQUEST  = 2;
constexpr uint8_t MSG_LP_SOLUTION  = 3;
constexpr uint8_t MSG_MIP_SOLUTION = 4;

template <typename i_t, typename f_t>
class my_serializer_t : public remote_serializer_t<i_t, f_t> {
public:
  my_serializer_t() = default;
  ~my_serializer_t() override = default;

  std::string format_name() const override { return "my_format"; }
  uint32_t protocol_version() const override { return 1; }

  //========================================================================
  // CLIENT-SIDE METHODS
  //========================================================================

  std::vector<uint8_t> serialize_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const pdlp_solver_settings_t<i_t, f_t>& settings) override
  {
    std::vector<uint8_t> buffer;

    // Start with message type
    buffer.push_back(MSG_LP_REQUEST);

    // Serialize problem dimensions
    i_t n_rows = view.get_constraint_matrix_offsets().size() > 0
                   ? view.get_constraint_matrix_offsets().size() - 1 : 0;
    i_t n_cols = view.get_objective_coefficients().size();
    i_t nnz = view.get_constraint_matrix_values().size();

    // ... serialize all problem data ...
    // See msgpack_serializer.cpp for complete example

    return buffer;
  }

  std::vector<uint8_t> serialize_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const mip_solver_settings_t<i_t, f_t>& settings) override
  {
    std::vector<uint8_t> buffer;
    buffer.push_back(MSG_MIP_REQUEST);
    // ... similar to LP but with MIP settings ...
    return buffer;
  }

  optimization_problem_solution_t<i_t, f_t> deserialize_lp_solution(
    const std::vector<uint8_t>& data) override
  {
    // Parse the solution data
    // Create and return solution object

    // On error, return error solution:
    // return optimization_problem_solution_t<i_t, f_t>(
    //   cuopt::logic_error("Parse error", cuopt::error_type_t::RuntimeError));
  }

  mip_solution_t<i_t, f_t> deserialize_mip_solution(
    const std::vector<uint8_t>& data) override
  {
    // Similar to LP solution
  }

  //========================================================================
  // SERVER-SIDE METHODS
  //========================================================================

  bool is_mip_request(const std::vector<uint8_t>& data) override
  {
    if (data.empty()) return false;
    return data[0] == MSG_MIP_REQUEST;
  }

  bool deserialize_lp_request(
    const std::vector<uint8_t>& data,
    mps_parser::mps_data_model_t<i_t, f_t>& mps_data,
    pdlp_solver_settings_t<i_t, f_t>& settings) override
  {
    try {
      // Parse message type
      if (data.empty() || data[0] != MSG_LP_REQUEST) return false;

      // Parse problem data and populate mps_data:
      // mps_data.set_problem_name("...");
      // mps_data.set_objective_coefficients(coeffs.data(), coeffs.size());
      // mps_data.set_csr_constraint_matrix(...);
      // mps_data.set_variable_bounds(...);
      // mps_data.set_constraint_bounds(...);

      // Parse settings:
      // settings.time_limit = ...;
      // settings.iteration_limit = ...;

      return true;
    } catch (...) {
      return false;
    }
  }

  bool deserialize_mip_request(
    const std::vector<uint8_t>& data,
    mps_parser::mps_data_model_t<i_t, f_t>& mps_data,
    mip_solver_settings_t<i_t, f_t>& settings) override
  {
    // Similar to LP, also set variable types for integers/binaries:
    // mps_data.set_variable_types(var_types);
    return true;
  }

  std::vector<uint8_t> serialize_lp_solution(
    const optimization_problem_solution_t<i_t, f_t>& solution) override
  {
    std::vector<uint8_t> buffer;
    buffer.push_back(MSG_LP_SOLUTION);

    // NOTE: Server calls solution.to_host() before serialization,
    // so solution data is always in CPU memory. Use:
    //   solution.get_primal_solution_host()
    //   solution.get_dual_solution_host()
    //   solution.get_reduced_cost_host()

    // Serialize termination status, objective, solution vectors, etc.

    return buffer;
  }

  std::vector<uint8_t> serialize_mip_solution(
    const mip_solution_t<i_t, f_t>& solution) override
  {
    std::vector<uint8_t> buffer;
    buffer.push_back(MSG_MIP_SOLUTION);

    // Use solution.get_solution_host() for the solution vector

    return buffer;
  }
};

//==========================================================================
// FACTORY FUNCTIONS - Must be exported with C linkage
//==========================================================================

template <typename i_t, typename f_t>
std::unique_ptr<remote_serializer_t<i_t, f_t>> create_serializer_impl()
{
  return std::make_unique<my_serializer_t<i_t, f_t>>();
}

}  // namespace cuopt::linear_programming

// Export factory functions with C linkage for dlopen/dlsym
extern "C" {

std::unique_ptr<cuopt::linear_programming::remote_serializer_t<int32_t, double>>
create_cuopt_serializer_i32_f64()
{
  return cuopt::linear_programming::create_serializer_impl<int32_t, double>();
}

std::unique_ptr<cuopt::linear_programming::remote_serializer_t<int32_t, float>>
create_cuopt_serializer_i32_f32()
{
  return cuopt::linear_programming::create_serializer_impl<int32_t, float>();
}

// Add more type combinations as needed

}
```

### Step 2: Create CMakeLists.txt for the Plugin

Create `cpp/src/linear_programming/utilities/serializers/CMakeLists.txt`:

```cmake
# Build the custom serializer as a shared library plugin
add_library(cuopt_my_serializer SHARED my_serializer.cpp)

target_link_libraries(cuopt_my_serializer
  PRIVATE
    cuopt  # Link against cuOpt for solution types
)

target_include_directories(cuopt_my_serializer
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set RPATH so the plugin can find libcuopt.so
set_target_properties(cuopt_my_serializer PROPERTIES
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS cuopt_my_serializer
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
```

### Step 3: Add to Parent CMakeLists.txt

In `cpp/CMakeLists.txt`, add:

```cmake
add_subdirectory(src/linear_programming/utilities/serializers)
```

### Step 4: Build the Plugin

```bash
# Build everything including the plugin
./build.sh libcuopt cuopt_remote_server

# Or just the plugin (after initial build)
cd cpp/build
ninja cuopt_my_serializer
```

## Using the Plugin

### Environment Variable

Set `CUOPT_SERIALIZER_LIB` to point to your plugin:

```bash
export CUOPT_SERIALIZER_LIB=/path/to/libcuopt_my_serializer.so
```

### Running Server with Custom Serializer

```bash
# Set the serializer library
export CUOPT_SERIALIZER_LIB=$CONDA_PREFIX/lib/libcuopt_my_serializer.so

# Start the server
cuopt_remote_server -p 8765
```

Server output will show:
```
[remote_solve] Loading custom serializer from: /path/to/libcuopt_my_serializer.so
[remote_solve] Using custom serializer: my_format
```

### Running Client with Custom Serializer

```bash
# Same serializer must be used on client side
export CUOPT_SERIALIZER_LIB=$CONDA_PREFIX/lib/libcuopt_my_serializer.so
export CUOPT_REMOTE_HOST=localhost
export CUOPT_REMOTE_PORT=8765

# Run cuopt_cli
cuopt_cli problem.mps

# Or Python
python my_solver_script.py
```

### Complete Example Session

```bash
# Terminal 1: Start server with msgpack serializer
export CUOPT_SERIALIZER_LIB=$CONDA_PREFIX/lib/libcuopt_msgpack_serializer.so
cuopt_remote_server -p 8765

# Terminal 2: Run client with same serializer
export CUOPT_SERIALIZER_LIB=$CONDA_PREFIX/lib/libcuopt_msgpack_serializer.so
export CUOPT_REMOTE_HOST=localhost
export CUOPT_REMOTE_PORT=8765
cuopt_cli /path/to/problem.mps
```

## Data Model Reference

### Problem Data (`data_model_view_t`)

Key getters for serializing problem data:

```cpp
// Problem metadata
view.get_problem_name()           // std::string
view.get_objective_name()         // std::string
view.get_sense()                  // bool (true = maximize)
view.get_objective_scaling_factor() // f_t
view.get_objective_offset()       // f_t

// Constraint matrix (CSR format)
view.get_constraint_matrix_values()   // span<f_t>
view.get_constraint_matrix_indices()  // span<i_t>
view.get_constraint_matrix_offsets()  // span<i_t>

// Objective and bounds
view.get_objective_coefficients()     // span<f_t>
view.get_variable_lower_bounds()      // span<f_t>
view.get_variable_upper_bounds()      // span<f_t>
view.get_constraint_lower_bounds()    // span<f_t>
view.get_constraint_upper_bounds()    // span<f_t>

// For MIP problems
view.get_variable_types()             // span<char> ('C', 'I', 'B')

// Names (optional)
view.get_variable_names()             // vector<string>
view.get_row_names()                  // vector<string>
```

### Problem Data (`mps_data_model_t`) - Server Side

Key setters for deserializing:

```cpp
mps_data.set_problem_name(name);
mps_data.set_objective_name(name);
mps_data.set_maximize(bool);
mps_data.set_objective_scaling_factor(factor);
mps_data.set_objective_offset(offset);

mps_data.set_objective_coefficients(ptr, size);
mps_data.set_csr_constraint_matrix(values, nvals, indices, nidx, offsets, noff);
mps_data.set_variable_bounds(lower, upper, size);
mps_data.set_constraint_bounds(lower, upper, size);

// For MIP
mps_data.set_variable_types(std::vector<char>);
```

### LP Solution (`optimization_problem_solution_t`)

```cpp
// Getters (for serialization)
solution.get_termination_status()     // pdlp_termination_status_t
solution.get_objective_value()        // f_t
solution.get_primal_solution_host()   // vector<f_t>&
solution.get_dual_solution_host()     // vector<f_t>&
solution.get_reduced_cost_host()      // vector<f_t>&
solution.get_solve_time()             // double
solution.get_l2_primal_residual()     // f_t
solution.get_l2_dual_residual()       // f_t
solution.get_gap()                    // f_t
solution.get_nb_iterations()          // i_t

// Setters (for deserialization on client)
solution.set_termination_status(status);
solution.set_objective_value(value);
solution.set_primal_solution_host(vector);
solution.set_dual_solution_host(vector);
solution.set_reduced_cost_host(vector);
solution.set_solve_time(time);
// ... etc
```

### MIP Solution (`mip_solution_t`)

```cpp
// Getters
solution.get_termination_status()     // mip_termination_status_t
solution.get_objective_value()        // f_t
solution.get_solution_host()          // vector<f_t>&
solution.get_total_solve_time()       // double
solution.get_mip_gap()                // f_t

// Setters
solution.set_solution_host(vector);
solution.set_objective_value(value);
solution.set_mip_gap(gap);
// ... etc
```

## Tips and Best Practices

### 1. Message Type Identification

Always include a message type identifier as the first byte(s):

```cpp
constexpr uint8_t MSG_LP_REQUEST   = 1;
constexpr uint8_t MSG_MIP_REQUEST  = 2;
constexpr uint8_t MSG_LP_SOLUTION  = 3;
constexpr uint8_t MSG_MIP_SOLUTION = 4;
```

### 2. Version Compatibility

Include a protocol version in your messages for future compatibility:

```cpp
// In serialize:
buffer.push_back(MSG_LP_REQUEST);
buffer.push_back(PROTOCOL_VERSION);

// In deserialize:
uint8_t version = data[1];
if (version != PROTOCOL_VERSION) {
  // Handle version mismatch
}
```

### 3. Error Handling

Return proper error solutions on parse failures:

```cpp
optimization_problem_solution_t<i_t, f_t> deserialize_lp_solution(...) {
  try {
    // Parse...
  } catch (const std::exception& e) {
    return optimization_problem_solution_t<i_t, f_t>(
      cuopt::logic_error(
        std::string("Deserialize error: ") + e.what(),
        cuopt::error_type_t::RuntimeError));
  }
}
```

### 4. Solution Memory

The server calls `solution.to_host()` before serialization, so:
- Always use `get_*_host()` methods for solution data
- No need to handle GPU memory in your serializer

### 5. Testing

Test your serializer with both LP and MIP problems:

```bash
# LP test
cuopt_cli /path/to/lp_problem.mps

# MIP test (use a problem with integer variables)
cuopt_cli /path/to/mip_problem.mps
```

## Reference Implementation

See the MsgPack serializer for a complete working example:
- `cpp/src/linear_programming/utilities/serializers/msgpack_serializer.cpp`
- `cpp/src/linear_programming/utilities/serializers/CMakeLists.txt`

## Troubleshooting

### "Failed to load serializer library"

- Check the path in `CUOPT_SERIALIZER_LIB` is correct
- Ensure the library was built: `ls $CONDA_PREFIX/lib/libcuopt_*serializer.so`

### "Factory function not found"

- Ensure factory functions are exported with `extern "C"`
- Check function names match: `create_cuopt_serializer_i32_f64`, etc.

### "Read failed" / Malformed messages

- Ensure client and server use the **same** serializer
- Check message framing is consistent
- Verify all required fields are serialized

### Symbol errors at runtime

- Rebuild and reinstall with `./build.sh libcuopt cuopt_remote_server`
- Ensure plugin links against `cuopt`
