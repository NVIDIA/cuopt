# cuOpt Development Containers

This directory contains [devcontainer configurations](https://containers.dev/implementors/json_reference/) for using VSCode to [develop in a container](https://code.visualstudio.com/docs/devcontainers/containers) via the `Remote Containers` [extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) or [GitHub Codespaces](https://github.com/codespaces).

This container is a turnkey development environment for building and testing the cuOpt C++ and Python libraries.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Host bind mounts](#host-bind-mounts)
* [Launch a Dev Container](#launch-a-dev-container)
* [Using the devcontainer](#using-the-devcontainer)

## Prerequisites

* [VSCode](https://code.visualstudio.com/download)
* [VSCode Remote Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Host bind mounts

By default, the following directories are bind-mounted into the devcontainer:

* `${repo}:/home/coder/cuopt`
* `${repo}/../.aws:/home/coder/.aws`
* `${repo}/../.local:/home/coder/.local`
* `${repo}/../.cache:/home/coder/.cache`
* `${repo}/../.conda:/home/coder/.conda`
* `${repo}/../.config:/home/coder/.config`

This ensures caches, configurations, dependencies, and your commits are persisted on the host across container runs.

## Launch a Dev Container

To launch a devcontainer from VSCode, open the cuOpt repo and select the "Reopen in Container" button in the bottom right:<br/><img src="https://user-images.githubusercontent.com/178183/221771999-97ab29d5-e718-4e5f-b32f-2cdd51bba25c.png"/>

Alternatively, open the VSCode command palette (typically `cmd/ctrl + shift + P`) and run the "Rebuild and Reopen in Container" command.

## Using the devcontainer

On startup, the devcontainer creates or updates the conda/pip environment using `cuopt/dependencies.yaml`.

The container includes convenience functions to clean, configure, and build the cuOpt components:

```shell
$ clean-cuopt-cpp # only cleans the C++ build dir
$ clean-cuopt-python # only cleans the Python build dir
$ clean-cuopt # cleans both C++ and Python build dirs

$ configure-cuopt-cpp # only configures cuOpt C++ lib

$ build-cuopt-cpp # only builds cuOpt C++ lib
$ build-cuopt-python # only builds cuOpt Python lib
$ build-cuopt # builds both C++ and Python libs
$ build-all # builds all libraries in this repo
```

* The C++ build script is a small wrapper around `cmake -S ~/cuopt/cpp -B ~/cuopt/cpp/build` and `cmake --build ~/cuopt/cpp/build`
* The Python build script is a small wrapper around `pip install --editable ~/cuopt/python/cuopt`

Unlike `build.sh`, these convenience scripts do not install the libraries after building them. Instead, they automatically inject the correct arguments to build the C++ libraries from source and use their build dirs as package roots.
