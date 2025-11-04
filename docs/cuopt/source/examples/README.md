# cuOpt Examples

This directory contains all executable examples for the cuOpt documentation. Examples are organized by API type and problem category.

## Directory Structure

```
examples/
├── README.md (this file)
│
cuopt-python/
├── routing/
│   └── examples/
│       └── smoke_test_example.py
└── lp-milp/
    └── examples/
        ├── simple_lp_example.py
        ├── simple_milp_example.py
        ├── production_planning_example.py
        ├── expressions_constraints_example.py
        ├── incumbent_solutions_example.py
        └── pdlp_warmstart_example.py

cuopt-server/examples/
├── routing/
│   └── examples/
│       ├── basic_routing_example.py
│       └── initial_solution_example.py
├── lp/
│   └── examples/
│       ├── basic_lp_example.py
│       ├── warmstart_example.py
│       ├── mps_file_example.py
│       └── mps_datamodel_example.py
└── milp/
    └── examples/
        ├── basic_milp_example.py
        ├── incumbent_callback_example.py
        └── abort_job_example.py

cuopt-c/lp-milp/examples/
├── simple_lp_example.c
├── simple_milp_example.c
├── mps_file_example.c
├── milp_mps_example.c
├── sample.mps (LP test data)
├── mip_sample.mps (MILP test data)
├── Makefile
└── README.md

cuopt-cli/examples/
├── routing/
│   └── examples/
│       ├── basic_routing_example.sh
│       └── initial_solution_example.sh
├── lp/
│   └── examples/
│       ├── basic_lp_example.sh
│       ├── warmstart_example.sh
│       ├── batch_mode_example.sh
│       └── sample.mps (LP test data)
└── milp/
    └── examples/
        ├── basic_milp_example.sh
        └── abort_job_example.sh
```

## Organization Principles

### Nested Examples Structure
All examples follow a consistent nested directory structure:
- Module directory (e.g., `cuopt-python`, `cuopt-server`)
- Problem category subdirectory (e.g., `routing`, `lp`, `milp`)
- `examples/` subdirectory containing actual example files

This structure:
- Co-locates examples with their documentation
- Makes it easy for developers to add new examples
- Simplifies documentation references using relative paths
- Enables automatic example discovery for testing

### Data Files
MPS files and other test data are stored in the `examples/` directory alongside the code that uses them:
- `sample.mps` - Basic LP problem (good-1)
- `mip_sample.mps` - MILP problem (EXAMPLE21 from N&W)

### Documentation Integration
Examples are referenced in RST documentation files using `literalinclude` directives:

```rst
.. literalinclude:: examples/simple_lp_example.py
   :language: python
   :linenos:
   :start-after: def main():
   :end-before: if __name__
```

## Running Examples

### Python API Examples (`cuopt-python/`)
```bash
cd docs/cuopt/source/cuopt-python/<category>/examples
python <example_name>.py
```

### Server Examples (`cuopt-server/`)
**Requires cuOpt server running:**
```bash
# Start server
python -m cuopt_server.cuopt_service --ip localhost --port 5000

# Run example
cd docs/cuopt/source/cuopt-server/examples/<category>/examples
python <example_name>.py
```

### C Examples (`cuopt-c/`)
```bash
cd docs/cuopt/source/cuopt-c/lp-milp/examples
make all
./<executable_name> [mps_file]
```

### CLI Examples (`cuopt-cli/`)
**Requires cuOpt server running:**
```bash
# Start server
python -m cuopt_server.cuopt_service --ip localhost --port 5000

# Run example
cd docs/cuopt/source/cuopt-cli/examples/<category>/examples
bash <example_name>.sh
```

## Testing All Examples

Use the comprehensive test script from the repository root:
```bash
./test_all_examples.sh
```

This script:
- Automatically discovers all examples
- Starts/stops the cuOpt server as needed
- Runs Python, C, and CLI examples
- Reports results with detailed logs
- Saves logs to `test-results/` directory

## Adding New Examples

1. **Choose the appropriate module and category directory**
   - For Python API: `cuopt-python/<category>/examples/`
   - For Server client: `cuopt-server/examples/<category>/examples/`
   - For C API: `cuopt-c/lp-milp/examples/`
   - For CLI: `cuopt-cli/examples/<category>/examples/`

2. **Create your example file**
   - Python: `<example_name>.py`
   - C: `<example_name>.c` (and update Makefile)
   - Shell: `<example_name>.sh`

3. **Include comprehensive documentation**
   - Module-level docstring explaining the example
   - Comments for key steps
   - Requirements and expected output

4. **Add any required data files** (e.g., MPS files) to the same `examples/` directory

5. **Update the RST documentation** to reference your example:
   ```rst
   .. literalinclude:: examples/<example_name>.<ext>
      :language: <language>
      :linenos:
   ```

6. **Test your example**:
   - Run it manually from its directory
   - Run `./test_all_examples.sh` to ensure it's discovered and passes

## Example Categories

### Routing
- Vehicle routing problems
- Optimization with constraints
- Initial solution handling

### LP (Linear Programming)
- Continuous variable optimization
- Warmstart functionality
- MPS file parsing
- Batch processing

### MILP (Mixed Integer Linear Programming)
- Integer and binary variable optimization
- Incumbent callbacks
- Job abortion
- Advanced constraints

## Requirements

### Python Examples
- cuopt package installed
- For server examples: cuopt_sh_client, cuOpt server running
- For MPS examples: cuopt_mps_parser

### C Examples
- gcc compiler
- libcuopt.so library
- cuopt_c.h headers

### CLI Examples
- cuopt_sh command-line tool
- cuOpt server running
- jq (for JSON parsing in some examples)

## Troubleshooting

### Server Connection Issues
- Ensure server is running: `python -m cuopt_server.cuopt_service`
- Check server is accessible: `curl http://localhost:5000/docs`
- Verify IP and port in examples match server configuration

### C Compilation Issues
- Set `INCLUDE_PATH` to cuOpt headers directory
- Set `LD_LIBRARY_PATH` to libcuopt.so location
- Check Makefile for correct paths

### MPS File Errors
- Ensure MPS files are valid (no extra whitespace in comment lines)
- Check file paths are correct relative to example directory
- Verify MPS format follows standard specification
