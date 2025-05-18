# Building Documentation

Documentation dependencies are installed while installing conda environment, please refer to the [CONTRIBUTING](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) for more details. Doc generation 
does not get run by default. There are two ways to generate the docs:

Note: It is assumed that all required libraries are already installed locally. If they haven't been installed yet, please first install all libraries by running:
```bash
./build.sh 
```

1. Run 
```bash
make clean;make html
```
from the `docs/cuopt` directory.
2. Run 
```bash
./build.sh docs
```
from the root directory.

Outputs to `build/html/index.html`

## View docs web page by opening HTML in browser:

First navigate to `/build/html/` folder, i.e., `cd build/html` and then run the following command:

```bash
python -m http.server
```
Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

```
https://<host IP-Address>:8000
```
Now you can check if your docs edits formatted correctly, and read well.
