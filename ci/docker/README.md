# Container Image build and test suite

## context

Add all the files and data for the buildx context to ``context`` folder like entrypoint script, and others.

## test

To test the container image, run the [test_image.sh](test_image.sh) script as shown below from the latest github repo:

```bash
docker run -it --rm --gpus all -u root --volume $PWD:/repo -w /repo --entrypoint "/bin/bash" nvidia/cuopt:[TAG] ./ci/docker/test_image.sh
```