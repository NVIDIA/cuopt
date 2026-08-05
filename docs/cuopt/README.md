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

## Prose Style Checks

Headings under `docs/cuopt/source/` use **title case**, enforced by
[Vale](https://vale.sh/) through the `vale` pre-commit hook. The rule lives in
`ci/vale/styles/cuOpt/Headings.yml` and follows Chicago style, so short
prepositions and conjunctions stay lowercase:

- `Connect and Solve`, `Where to Find Examples`, `Working with Incumbent Solutions`
- not `Connect and solve`, `Where To Find Examples`

Run it directly with:

```bash
vale docs/cuopt/source
```

Product names, acronyms, and API identifiers that must keep their own casing
(`cuOpt`, `gRPC`, `mTLS`, `solver_configs`) are listed under `exceptions` in
that file.

**Keep that list narrow, and only add a term once a real heading needs it.**
Exceptions match whole words, and a single match makes Vale skip the *entire*
heading — so a needless entry silently disables the check for every heading
that mentions it, with no visible symptom. After editing the list, confirm the
rule still catches a violation:

```bash
printf '# T\n\n## Process model\n' > /tmp/vale-check.md
vale --config=.vale.ini /tmp/vale-check.md   # must report an error
```
