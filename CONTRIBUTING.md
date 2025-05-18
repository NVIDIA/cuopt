# Contributing to cuOpt

Contributions to NVIDIA cuOpt fall into the following categories:

1. To report a bug, request a new feature, or report a problem with documentation, please file an
   [issue](https://github.com/NVIDIA/cuopt/issues/new/choose) describing the problem or new feature
   in detail. The NVIDIA cuOpt team evaluates and triages issues, and schedules them for a release. If you
   believe the issue needs priority attention, please comment on the issue to notify the team.
2. To propose and implement a new feature, please file a new feature request
   [issue](https://github.com/NVIDIA/cuopt/issues/new/choose). Describe the intended feature and
   discuss the design and implementation with the team and community. Once the team agrees that the
   plan looks good, go ahead and implement it, using the [code contributions](#code-contributions)
   guide below.
3. To implement a feature or bug fix for an existing issue, please follow the [code
   contributions](#code-contributions) guide below. If you need more context on a particular issue,
   please ask in a comment or create a question in issues.


## Code contributions

### Your first issue

1. Follow the guide at the bottom of this page for
   [Setting up your build environment](#setting-up-your-build-environment).
2. Find an issue to work on. The best way is to look for the
   [good first issue](https://github.com/NVIDIA/cuopt/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
   or [help wanted](https://github.com/NVIDIA/cuopt/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
   labels.
3. Comment on the issue stating that you are going to work on it.
4. Create a fork of the cuopt repository and check out a branch with a name that
   describes your planned work. For example, `fix-documentation`.
5. Write code to address the issue or implement the feature.
6. Add unit tests. Please refer to `cpp/src/tests` for examples of unit tests on C and C++ using gtest and `python/cuopt/cuopt/tests` for examples of unit tests on Python using pytest.
7. [Create your pull request](https://github.com/NVIDIA/cuopt/compare). To run continuous integration (CI) tests without requesting review, open a draft pull request.
8. Check if CI is running, if not please request one of the NVIDIA cuOpt developers to trigger it. This might happen in case you have non-verified (non-sign-off) commits or don't have enough permissions to trigger CI.
9. Verify that CI passes all [status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks).
   Fix if needed.
10. Github will automatically assign a reviewer to your pull request. Please wait for the reviewer to review your code. If the reviewer has any comments, address them in the pull request.
11. If your PR is not getting reviewed, please ping the reviewers in the PR.
12. Once reviewed and approved, a NVIDIA cuOpt developer will merge your pull request.


Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications! Please use the Github issues for any questions or for discussion, this would help community find all the answers in one place.

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/NVIDIA/cuopt/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where NVIDIA cuOpt developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Setting up your build environment

The following instructions are for developers and contributors to NVIDIA cuOpt development. These
instructions are tested on Ubuntu Linux LTS releases. Use these instructions to build NVIDIA cuOpt from
source and contribute to its development. Other operating systems may be compatible, but are not
currently tested.

Building NVIDIA cuOpt with the provided conda environment is recommended for users who wish to enable all
library features. The following instructions are for building with a conda environment. Dependencies
for a minimal build of NVIDIA cuOpt without using conda are also listed below.

### General requirements

Compilers:

* `gcc` version 11.4+
* `nvcc` version 11.8+
* `cmake` version 3.29.6+

CUDA/GPU Runtime:

* CUDA 11.4+
* Volta architecture or better ([Compute Capability](https://docs.nvidia.com/deploy/cuda-compatibility/) >=7.0)

You can obtain CUDA from
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Build NVIDIA cuOpt from source

- Clone the repository:

```bash
CUOPT_HOME=$(pwd)/cuopt
git clone https://github.com/NVIDIA/cuopt.git $CUOPT_HOME
cd $CUOPT_HOME
```

#### Building with a conda environment

**Note:** Using a conda environment is the easiest way to satisfy the library's dependencies.

- Create the conda development environment:

Please install conda if you don't have it already. You can install it from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

```bash
# create the conda environment (assuming in base `cuopt` directory)
# note: cuOpt currently doesn't support `channel_priority: strict`;
# use `channel_priority: flexible` instead
conda env create --name cuopt_dev --file conda/environments/all_cuda-128_arch-x86_64.yaml
# activate the environment
conda activate cuopt_dev
```

- **Note**: the conda environment files are updated frequently, so the
  development environment may also need to be updated if dependency versions or
  pinnings are changed.

- A `build.sh` script is provided in `$CUOPT_HOME`. Running the script with no additional arguments
  will install the `libmps_parser`, `libcuopt`, `cuopt_mps_parser`, `cuopt`, `cuopt-server`, `cuopt-sh-client` libraries and build the`documentation`. By default, the libraries are
  installed to the `$CONDA_PREFIX` directory. To install into a different location, set the location
  in `$INSTALL_PREFIX`. Finally, note that the script depends on the `nvcc` executable being on your
  path, or defined in `$CUDACXX`.

```bash
cd $CUOPT_HOME

# Choose one of the following commands, depending on whether
# you want to build and install the libcuopt C++ library only,
# or include the libcuopt and/or cuopt Python libraries:

./build.sh  # All the libraries
./build.sh libmps_parser  # libmps_parser only
./build.sh libmps_parser libcuopt  # libmps_parser and libcuopt only
```

- For the complete list of libraries as well as details about the script usage, run the `help` command:

```bash
./build.sh --help
```

#### Building for development

To build all libraries and tests, simply run

```bash
./build.sh
```

- **Note**: if Cython files (`*.pyx` or `*.pxd`) have changed, the Python build must be rerun.

To run the C++ tests, run

```bash
cd $CUOPT_HOME/datasets && get_test_data.sh
cd $CUOPT_HOME/datasets/linear_programming && download_pdlp_test_dataset.sh
cd $CUOPT_HOME/datasets/mip && download_miplib_test_dataset.sh
export RAPIDS_DATASET_ROOT_DIR=$CUOPT_HOME/datasets/
ctest --test-dir ${CUOPT_HOME}/cpp/build  # libcuopt
```

To run python tests, run

- To run `cuopt` tests:
```bash

cd $CUOPT_HOME/datasets && get_test_data.sh
cd $CUOPT_HOME/datasets/linear_programming && download_pdlp_test_dataset.sh
cd $CUOPT_HOME/datasets/mip && download_miplib_test_dataset.sh
export RAPIDS_DATASET_ROOT_DIR=$CUOPT_HOME/datasets/
cd $CUOPT_HOME/python
pytest -v ${CUOPT_HOME}/python/cuopt/cuopt/tests
```
## Debugging cuOpt

### Building in debug mode from source

Follow the instructions to [build from source](#build-cudf-from-source) and add `-g` to the
`./build.sh` command.

For example:

```bash
./build.sh libcuopt -g
```

This builds `libcuopt` in debug mode which enables some `assert` safety checks and includes symbols
in the library for debugging.

All other steps for installing `libcuopt` into your environment are the same.

### Debugging with `cuda-gdb` and `cuda-memcheck`

When you have a debug build of `libcuopt` installed, debugging with the `cuda-gdb` and
`cuda-memcheck` is easy.

If you are debugging a Python script, run the following:

```bash
cuda-gdb -ex r --args python <program_name>.py <program_arguments>
```

```bash
compute-sanitizer --tool memcheck python <program_name>.py <program_arguments>
```

### Device debug symbols

The device debug symbols are not automatically added with the cmake `Debug` build type because it
causes a runtime delay of several minutes when loading the libcuopt.so library.

Therefore, it is recommended to add device debug symbols only to specific files by setting the `-G`
compile option locally in your `cpp/CMakeLists.txt` for that file. Here is an example of adding the
`-G` option to the compile command for `cpp/src/routing/data_model_view.cu` source file:

```cmake
set_source_files_properties(src/routing/data_model_view.cu PROPERTIES COMPILE_OPTIONS "-G")
```

This will add the device debug symbols for this object file in `libcuopt.so`.  You can then use
`cuda-dbg` to debug into the kernels in that source file.

## Code Formatting

### Using pre-commit hooks

cuOpt uses [pre-commit](https://pre-commit.com/) to execute all code linters and formatters. These
tools ensure a consistent code format throughout the project. Using pre-commit ensures that linter
versions and options are aligned for all developers. Additionally, there is a CI check in place to
enforce that committed code follows our standards.

To use `pre-commit`, install via `conda` or `pip`:

```bash
conda install -c conda-forge pre-commit
```

```bash
pip install pre-commit
```

Then run pre-commit hooks before committing code:

```bash
pre-commit run --all-files --show-diff-on-failure
```

By default, pre-commit runs on staged files (only changes and additions that will be committed).
To run pre-commit checks on all files, execute:

```bash
pre-commit run --all-files
```

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit. This can be done by running:

```bash
pre-commit install
```

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with `git commit --no-verify` or with the short version `git commit -n`.

#### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

  

