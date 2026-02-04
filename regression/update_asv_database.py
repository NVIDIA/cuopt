# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import platform
import psutil
from asvdb import BenchmarkInfo, BenchmarkResult, ASVDb
import json
import pandas as pd


def update_asv_db(
    commitHash=None,
    commitTime=None,
    branch=None,
    repo_url=None,
    results_dir=None,
    machine_name=None,
    gpu_type=None,
    configs=None,
):
    """
    Read the benchmark_result* files in results_dir/benchmarks and
    update an existing asv benchmark database or create one if one
    does not exist in results_dir/benchmarks/asv.  If no
    benchmark_result* files are present, return without updating or
    creating.
    """

    # commitHash = commitHash + str(int(time.time()))
    benchmark_dir_path = Path(results_dir) / "benchmarks" / "results" / "csvs"
    asv_dir_path = Path(results_dir) / "benchmarks" / "results" / "asv"

    # List all benchmark_result files
    benchmark_result_list = benchmark_dir_path.glob("*.csv")

    bResultList = []
    # Skip these columns from benchmarking
    skip_columns = ["date_time", "git_commit"]

    # Create result objects for each benchmark result and store it in a list
    for file_name in benchmark_result_list:
        # skip if it's regression file
        if "regressions.csv" in str(file_name):
            continue
        with open(file_name, "r") as openfile:
            data = pd.read_csv(openfile, index_col=0).iloc[-1]
            test_name = str(file_name).split("/")[-1].split(".")[-2]
            config_file = None
            if test_name.startswith("lp"):
                config_file = configs + "/" + "lp_config.json"
            elif test_name.startswith("mip"):
                config_file = configs + "/" + "mip_config.json"
            else:
                config_file = configs + "/" + test_name + "_config.json"
            metrics = {}
            with open(config_file, "r") as fp:
                metrics = json.load(fp)["metrics"]
            for col_name in data.index:
                if col_name not in skip_columns:
                    bResult = BenchmarkResult(
                        funcName=test_name + "." + col_name,
                        result=data[col_name].item(),
                        unit="percentage"
                        if "bks" in col_name
                        else metrics[col_name]["unit"],
                    )
                    bResultList.append(bResult)

    if len(bResultList) == 0:
        print(
            "Could not find files matching 'csv' in "
            f"{benchmark_dir_path}, not creating/updating ASV database "
            f"in {asv_dir_path}."
        )
        return

    uname = platform.uname()
    # Maybe also write those metadata to metadata.sh ?
    osType = "%s %s" % (uname.system, uname.release)
    # Remove unnecessary osType detail
    osType = ".".join(osType.split("-")[0].split(".", 2)[:2])
    pythonVer = platform.python_version()
    # Remove unnecessary python version detail
    pythonVer = ".".join(pythonVer.split(".", 2)[:2])
    bInfo_dict = {
        "machineName": machine_name,
        # cudaVer : "10.0",
        "osType": osType,
        "pythonVer": pythonVer,
        "commitHash": commitHash,
        "branch": branch,
        # commit time needs to be in milliseconds
        "commitTime": commitTime * 1000,
        "gpuType": gpu_type,
        "cpuType": uname.processor,
        "arch": uname.machine,
        "ram": "%d" % psutil.virtual_memory().total,
    }
    bInfo = BenchmarkInfo(**bInfo_dict)

    # extract the branch name
    branch = bInfo_dict["branch"]

    db = ASVDb(dbDir=str(asv_dir_path), repo=repo_url, branches=[branch])

    for res in bResultList:
        db.addResult(bInfo, res)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--commitHash", type=str, required=True, help="project version"
    )
    ap.add_argument(
        "--commitTime", type=str, required=True, help="project version date"
    )
    ap.add_argument(
        "--repo-url", type=str, required=True, help="project repo url"
    )
    ap.add_argument("--branch", type=str, required=True, help="project branch")
    ap.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="directory to store the results in json files",
    )
    ap.add_argument(
        "--machine-name", type=str, required=True, help="Slurm cluster name"
    )
    ap.add_argument(
        "--gpu-type",
        type=str,
        required=True,
        help="the official product name of the GPU",
    )
    ap.add_argument(
        "--configs",
        type=str,
        required=True,
        help="the config file for all the tests",
    )
    args = ap.parse_args()

    update_asv_db(
        commitHash=args.commitHash,
        commitTime=int(args.commitTime),
        branch=args.branch,
        repo_url=args.repo_url,
        results_dir=args.results_dir,
        machine_name=args.machine_name,
        gpu_type=args.gpu_type,
        configs=args.configs,
    )
