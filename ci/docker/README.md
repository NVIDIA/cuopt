# Container Image build and test suite

## context

Add all the files and data for the buildx context to ``context`` folder like entrypoint script, and others.

## Dockerfiles

| File | Base | Use case |
|------|------|----------|
| `Dockerfile` | Ubuntu | General use |
| `Dockerfile.ubi` | Red Hat UBI 10 (RHEL 10) | FIPS 140-3 compliant environments |

Both images install the same cuOpt packages via pip and expose the same server entrypoint.

## test

The [test_image.sh](test_image.sh) script is shared between both images. It detects the OS at runtime (`/etc/redhat-release` present → UBI10/RHEL, absent → Ubuntu) and adjusts package manager and library paths accordingly.

To test either image locally (requires a GPU):

```bash
# Ubuntu image
docker run -it --rm --gpus all -u root --volume $PWD:/repo -w /repo --entrypoint "/bin/bash" nvidia/cuopt:[TAG] ./ci/docker/test_image.sh

# UBI10 image
docker run -it --rm --gpus all -u root --volume $PWD:/repo -w /repo --entrypoint "/bin/bash" nvidia/cuopt:[TAG]-ubi10 ./ci/docker/test_image.sh
```

### Startup smoke (REST + gRPC)

`test_image.sh` runs pytest inside the image and does not launch the servers.
To verify the published entrypoint starts both the default REST server and the
gRPC server (`CUOPT_SERVER_TYPE=grpc`):

```bash
./ci/docker/smoke_image.sh nvidia/cuopt:[TAG]
./ci/docker/smoke_image.sh nvidia/cuopt:[TAG]-ubi10
```

CI runs this for both variants after the multiarch manifests are published
(see `.github/workflows/test_images.yaml` job `smoke`).
