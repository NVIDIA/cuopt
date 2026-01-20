# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from pathlib import Path
import os
import argparse


def create_update_benchamrk_db(benchmark_path, output_path, commit_hash):
    bench_path = Path(benchmark_path) / "benchmarks"
    out_path = output_path + "/benchmarks/"

    # List all benchmark_result files
    benchmark_result_list = bench_path.glob("results*.csv")

    for file in benchmark_result_list:
        with open(file, "r") as openfile:
            data = pd.read_csv(openfile, index_col="test")
            data["commit_hash"] = commit_hash
            for index, rows in data.iterrows():
                out_file = index.split(".")[0] + ".csv"
                out_file_path = out_path + "/" + out_file

                if os.path.exists(out_file_path):
                    data = pd.read_csv(out_file_path)
                    data = pd.concat(
                        [data, rows.to_frame().T], ignore_index=True
                    )
                    data.to_csv(out_file_path, index=False)
                else:
                    rows.to_frame().T.to_csv(out_file_path, index=False)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-b", "--benchmark_path", help="Path to new sets of results"
)
parser.add_argument("-o", "--output_path", help="Path to save results")
parser.add_argument("-c", "--commit_hash", help="Git commit hash for the run")

# Read arguments from command line
args = parser.parse_args()

if args.benchmark_path and args.output_path and args.commit_hash:
    create_update_benchamrk_db(
        args.benchmark_path, args.output_path, args.commit_hash
    )
else:
    raise ValueError(
        "Missing mandatory options, please provide all the options"
    )
