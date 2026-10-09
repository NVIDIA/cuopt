# Isolated-install smoke tests

Install **one** cuOpt component (`client`, `mathopt` or `routing`) by itself in a clean
environment, build a small gtest against only that package, and run it. This mimics a user who
installs a single component rather than the `libcuopt` metapackage.

## In PR CI

`.github/workflows/pr.yaml` has two jobs that test the PR's **own freshly built** packages:

| Job | Needs | Script |
|-----|-------|--------|
| `isolated-install-wheels` | `wheel-build-libcuopt-{client,mathopt,routing}` | `ci/test_isolated_install_wheel.sh` |
| `isolated-install-conda` | `conda-cpp-build` | `ci/test_isolated_install_conda.sh` |

They depend only on the packaging builds, not on any test job. Each script loops over the three
components, installs just that component's package (plus the client it depends on) and runs the
gtest in `tests/`, then fails if any component failed. They are intentionally **not** in
`pr-builder`'s `needs` yet (see below).

## Locally

```bash
ci/isolated_install/run.sh <client|mathopt|routing> <pip|conda> [image]
```

This starts a clean container (`docker run --gpus all`, except for `client`) and installs the
**published nightly** package (RAPIDS nightly wheel index / `rapidsai-nightly` conda channel).

| File | Purpose |
|------|---------|
| `run.sh` | Host-side driver for local use: picks the image and starts the container. |
| `container_pip.sh` | Installs the wheel into a venv (`WHEEL_DIRS` = prebuilt wheels, else nightly index), asserts that only the expected cuopt wheels are installed, builds and runs the gtest. |
| `container_conda.sh` | Same with conda (`LOCAL_CHANNEL` = freshly built channel, else `rapidsai-nightly`). |
| `tests/` | One sample gtest per component, linking only `cuopt::<component>`. `mathopt` solves a tiny LP/MILP and `routing` a tiny CVRP. `client` only checks consumability: the client package ships a very small header set (the parsers are in the mathopt headers), so it includes `constants.h` and `dlopen()`s `libcuopt_client` with `RTLD_NOW` to check every symbol resolves. |

Environment: `CUDA_MAJOR` (default `13`), `CUDA_IMAGE_TAG` (pip base image tag for `run.sh`),
`NIGHTLY_WHEEL_INDEX` (override the wheel index).

## Known problems this is meant to catch

These were found while writing the tests (local runs against nightlies; only `routing` on pip passed):

1. **mathopt wheel:** `cuopt_mathopt-targets.cmake` hardcodes the build-machine path
   `/usr/lib64/libcudss/13/./libcudss.so.0` in `INTERFACE_LINK_LIBRARIES`, so consumers fail to link.
2. **client wheel:** its CMake config (from the shared export set) requires `CUDAToolkit`/`nvcc`,
   `rmm` and `raft`, none of which the wheel declares as dependencies. (#2094 makes client-only
   installs not require them in the config.)
3. **conda component packages:** none ships `cuopt-config.cmake` (only the `libcuopt` metapackage
   does), so `find_package(cuopt)` fails after installing a single component.
4. **wheels:** components are split across separate wheel directories, so a consumer has to pass
   every wheel root and every config directory in `CMAKE_PREFIX_PATH` for `find_package(cuopt)` to
   resolve sibling components.

Until the packaging fixes land, expect mathopt/pip, client/pip and all conda cases to fail. Add the
two jobs to `pr-builder`'s `needs` once they pass.
