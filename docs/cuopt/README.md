# Building Documentation

Documentation dependencies are installed while installing the Conda environment, please refer to the [Build and Test](../../CONTRIBUTING.md#building-with-a-conda-environment) for more details. Assuming you have set-up the Conda environment, you can build the documentation along with all the cuOpt libraries by running:

```bash
./build.sh 
```

In subsequent runs where there are no changes to the cuOpt libraries, documentation can be built by running:

1. From the root directory:
```bash
./build.sh docs
```

2. From the `docs/cuopt` directory:
```bash
make clean;make html
```

Outputs to `build/html/index.html`

## View docs web page by opening HTML in browser:

```bash
python -m http.server --directory=build/html/
```
Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

```
http://<host IP-Address>:8000
```
Now you can check if your docs edits formatted correctly, and read well.
