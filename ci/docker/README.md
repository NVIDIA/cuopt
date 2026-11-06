# Container Image build and test suite

## context

Add all the files and data for the buildx context to ``context`` folder like entrypoint script, and others.

## Dockerfiles

| File | Base | Use case |
|------|------|----------|
| `Dockerfile` | Ubuntu | General use |
| `Dockerfile.ubi` | Red Hat UBI 10 (RHEL 10) | FIPS 140-3 compliant environments |

Both images install the same cuOpt packages via pip and expose the same server entrypoint.

## Build an image from local source

`Dockerfile.source` is for testing changes that have not been released as a
wheel. It creates the conda environment described in `CONTRIBUTING.md`, builds
and installs cuOpt from the Docker build context, then copies the installed
environment into a smaller CUDA runtime image. It targets NVIDIA H100 GPUs
(`CMAKE_CUDA_ARCHITECTURES=90`) by default, which avoids compiling portable
fatbins for every cuOpt-supported GPU architecture.

Build from the branch containing the change (for example,
`codex/break-load-windows`) and give the image a Docker Hub-qualified tag:

```bash
git switch codex/break-load-windows
IMAGE=docker.io/<dockerhub-user>/cuopt:break-load-windows \
  PARALLEL_LEVEL=1 \
  ./ci/docker/build_source_image.sh
```

The source-image default is one compile job because each CUDA compile needs
several GiB of RAM. Raise `PARALLEL_LEVEL` only on a builder with enough
available memory.

The resulting image requires an H100-class GPU. To target a different remote
GPU, override `CUDA_ARCHITECTURES` with its CUDA compute capability before
building; this invalidates the C++ build cache once:

```bash
IMAGE=docker.io/<dockerhub-user>/cuopt:break-load-windows \
  CUDA_ARCHITECTURES=90 \
  ./ci/docker/build_source_image.sh
```

The Docker build persists both `cpp/build` and the project-supported `ccache`
store in BuildKit caches. If one source file fails to compile, fix it and rerun
the same command: unchanged compilation units are reused instead of recompiling
the whole project.

If the change is checked out in another worktree, retain this Docker workflow
checkout and point `SOURCE_CONTEXT` at the worktree that contains the change:

```bash
IMAGE=docker.io/<dockerhub-user>/cuopt:break-load-windows \
  SOURCE_CONTEXT=/path/to/cuopt-break-load-windows \
  ./ci/docker/build_source_image.sh
```

The default uses CUDA 13.3. The remote NVIDIA driver must support that CUDA
runtime. To target CUDA 12.9 instead, select its matching conda environment
and both CUDA base images:

```bash
IMAGE=docker.io/<dockerhub-user>/cuopt:break-load-windows-cu129 \
  CUDA_ENV=129 \
  CUDA_DEVEL_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04 \
  CUDA_RUNTIME_IMAGE=nvidia/cuda:12.9.1-runtime-ubuntu24.04 \
  ./ci/docker/build_source_image.sh
```

After `docker login`, publish only the intended tag:

```bash
IMAGE=docker.io/<dockerhub-user>/cuopt:break-load-windows PUSH=1 \
  ./ci/docker/build_source_image.sh
```

On the GPU host, pull and launch the HTTP proxy plus gRPC server:

```bash
docker pull docker.io/<dockerhub-user>/cuopt:break-load-windows
docker run --rm --gpus all -p 5000:5000 -p 5001:5001 \
  docker.io/<dockerhub-user>/cuopt:break-load-windows
```

The image follows the standard entrypoint: it starts proxy mode by default;
append `grpc` for gRPC-only mode or set `CUOPT_SERVER_TYPE=legacy` for the
legacy REST server.

## test

The [test_image.sh](test_image.sh) script is shared between both images. It detects the OS at runtime (`/etc/redhat-release` present → UBI10/RHEL, absent → Ubuntu) and adjusts package manager and library paths accordingly.

To test either image locally (requires a GPU):

```bash
# Ubuntu image
docker run -it --rm --gpus all -u root --volume $PWD:/repo -w /repo --entrypoint "/bin/bash" nvidia/cuopt:[TAG] ./ci/docker/test_image.sh

# UBI10 image
docker run -it --rm --gpus all -u root --volume $PWD:/repo -w /repo --entrypoint "/bin/bash" nvidia/cuopt:[TAG]-ubi10 ./ci/docker/test_image.sh
```

### Startup smoke (proxy + gRPC, gRPC-only, legacy)

`test_image.sh` runs pytest inside the image and does not launch the servers.
To verify the published entrypoint starts the default HTTP proxy plus
`cuopt_grpc_server` (`proxy`), gRPC-only (`grpc`), and the legacy REST
server (`legacy`):

```bash
./ci/docker/smoke_image.sh nvidia/cuopt:[TAG]
./ci/docker/smoke_image.sh nvidia/cuopt:[TAG]-ubi10
```

CI runs this for both variants after the multiarch manifests are published
(see `.github/workflows/test_images.yaml` job `smoke`).
