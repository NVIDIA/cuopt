# Benchmarks Scripts

This directory contains the scripts for running benchmarks. Currently it supports running benchmarks on LP with and MILP.


# Linear Programming Benchmarking


- Mittelmann LP benchmark

```bash
benchmarks/linear_programming/utils/benchmark_lp_mittelmann.sh
```

- MIPLIB Benchmark

```
mkdir miplib_data
mkdir miplib_result
wget https://miplib.zib.de/downloads/benchmark.zip -O miplib_data/benchmark.zip
unzip miplib_data/benchmark.zip -d miplib_data
find miplib_data -name "*.gz" -exec gunzip {} \;
find miplib_data -name "*.gz" -delete

benchmarks/linear_programming/run_mps_files.sh --path miplib_data/ --write-log-file --log-to-console false --output-dir miplib_result --time-limit 600 --presolve t > miplib_result/output.log 2>&1
```
