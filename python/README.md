# Python Modules

This directory contains the Python modules for the cuOpt project.

## Package Structure

- Each subdirectory contains the Python modules for a specific cuOpt package. For example, `libcuopt` directory contains the Python wrappers for the cuOpt C++ library. This is the main package for the cuOpt project. And it just loads shared libraries and make it available for other python modules. `cuopt` python package uses `libcuopt` package as dependency and build on top of it.

```bash
python/
├── libcuopt/
├── cuopt/
└── ...
```
- Each of these python modules have `tests` directory which contains the tests for the module. Python tests are written using `pytest`. For example, `python/cuopt/cuopt/tests/` directory contains the tests for the `cuopt` python package.

```bash
python/
├── cuopt/
│   ├── cuopt/
│   │   └── tests/
│   └── ...
└── ...
```

- Each of these pyhon modules have pyproject.toml file which contains the dependencies for the module. For example, `python/cuopt/pyproject.toml` file contains the dependencies for the `cuopt` python package.

```bash
python/
├── cuopt/
│   ├── pyproject.toml
│   └── ...
└── ...
```

- The dependencies are defined in the [dependencies.yaml](../dependencies.yaml) file in root folder. For example, `python/cuopt/pyproject.toml` file contains the dependencies for the `cuopt` python package. So any changes to dependencies should be done in the [dependencies.yaml](../dependencies.yaml) file. Please refer to different sections in the [dependencies.yaml](../dependencies.yaml) file for more details.






