# C++ Modules

This directory contains the C++ modules for the cuOpt project.

Please refer to the [CMakeLists.txt](CMakeLists.txt) file for details on how to add new modules and tests.

Most of the dependencies are defined in the [dependencies.yaml](../dependencies.yaml) file. Please refer to different sections in the [dependencies.yaml](../dependencies.yaml) file for more details. However, some of the dependencies are defined in [thirdparty modules](cmake/thirdparty/) in case where source code is needed to build, for example, `cccl` and `rmm`.


## Include Structure

Add any new modules in the `include` directory under `include/cuopt/<module_name>` directory.

```bash
cpp/
├── include/
│   ├── cuopt/
│   │   └── linear_programming/
│   │       └── ...
│   │   └── routing/
│   │       └── ...
│   └── ...
└── ...
```

## Source Structure

Add any new modules in the `src` directory under `src/cuopt/<module_name>` directory.

```bash
cpp/
├── src/
│   ├── cuopt/
│   │   └── linear_programming/
│   │       └── ...
│   │   └── routing/
│   │       └── ...
└── ...
```

## Test Structure

Add any new modules in the `test` directory under `test/cuopt/<module_name>` directory.

```bash
cpp/
├── test/
│   ├── cuopt/
│   │   └── linear_programming/
│   │       └── ...
│   │   └── routing/
│   │       └── ...
└── ...
```

## MPS parser

The MPS parser is a standalone module that parses MPS files and converts them into a format that can be used by the cuOpt library.

It is located in the `libmps_parser` directory. This also contains the `CMakeLists.txt` file to build the module.
