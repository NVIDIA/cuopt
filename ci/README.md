# CI scripts

This directory contains the scripts for the CI pipeline.

CI builds are triggered by `pr.yaml`, `build.yaml` and `test.yaml` files in the `.github/workflows` directory. And these scripts are used from those workflows to build and test the code.

cuOpt is packaged in following ways:

## PIP package

### Build

The scripts for building the PIP packages are named as `build_wheel_<package_name>.sh`. For example, `build_wheel_cuopt.sh` is used to build the PIP package for cuOpt.

Please refer to existing scripts for more details and how you can add a new script for a new package.

### Test

The scripts for testing the PIP packages are named as `test_wheel_<package_name>.sh`. For example, `test_wheel_cuopt.sh` is used to test the PIP package for cuOpt.

Please refer to existing scripts for more details and how you can add a new script for a new package.

## Conda Package

### Build

For Conda package,

- all cpp libraries are built under one script called `build_cpp.sh`.
- all python bindings are built under one script called `build_python.sh`.

So if there are new cpp libraries or python bindings, you need to add them to the respective scripts.


### Test

Similarly, for Conda package,

- all cpp libraries are tested under one script called `test_cpp.sh`.
- all python bindings are tested under one script called `test_python.sh`.


There are other scripts in this directory which are used to build and test the code and are also used in the workflows as utlities.
