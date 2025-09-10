# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import platform
import psutil
from asvdb import utils, BenchmarkInfo, BenchmarkResult, ASVDb
import json
import pandas as pd
import time


def update_asv_db(commitHash=None,
                  commitTime=None,
                  branch=None,
                  repo_url=None,
                  results_dir=None,
                  machine_name=None,
                  gpu_type=None):
    """
    Read the benchmark_result* files in results_dir/benchmarks and
    update an existing asv benchmark database or create one if one
    does not exist in results_dir/benchmarks/asv.  If no
    benchmark_result* files are present, return without updating or
    creating.
    """

    # commitHash = commitHash + str(int(time.time()))
    benchmark_dir_path = Path(results_dir)/"benchmarks"
    asv_dir_path = benchmark_dir_path/"asv"

    # List all benchmark_result files
    benchmark_result_list = benchmark_dir_path.glob("results*.csv")

    bResultList = []
    # Create result objects for each benchmark result and store it in a list
    for file_name in benchmark_result_list:
        with open(file_name, 'r') as openfile:
            data = pd.read_csv(openfile, index_col="test")
            if "service_endpoint" in str(file_name) or "service_method" in str(file_name):
                name = "Service_Endpoint" if "service_endpoint" in str(file_name) else "Service_Method"
                for index, rows in data.iterrows():
                    bResult = BenchmarkResult(funcName=name+"."+index+"_runtime", result=rows["run_time"], unit="Seconds")
                    bResultList.append(bResult)
            else:
                for index, rows in data.iterrows():
                    bResult = BenchmarkResult(funcName=index+"_solver_runtime", result=rows["solver_run_time"], unit="Seconds")
                    bResultList.append(bResult)
                    bResult = BenchmarkResult(funcName=index+"_etl_runtime", result=rows["etl_time"], unit="Seconds")
                    bResultList.append(bResult)
                    bResult = BenchmarkResult(funcName=index+"_memory", result=rows["memory"], unit="MB")
                    bResultList.append(bResult)
                    bResult = BenchmarkResult(funcName=index+"_travel_cost", result=rows["travel_cost"], unit="Distance")
                    bResultList.append(bResult)

    if len(bResultList) == 0:
        print("Could not find files matching 'benchmark_result*' in "
              f"{benchmark_dir_path}, not creating/updating ASV database "
              f"in {asv_dir_path}.")
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
        'machineName' : machine_name,
        #cudaVer : "10.0",
        'osType' : osType,
        'pythonVer' : pythonVer,
        'commitHash' : commitHash,
        'branch' : branch,
        #commit time needs to be in milliseconds
        'commitTime' : commitTime*1000,
        'gpuType' : gpu_type,
        'cpuType' : uname.processor,
        'arch' : uname.machine,
        'ram' : "%d" % psutil.virtual_memory().total
    }
    bInfo = BenchmarkInfo(**bInfo_dict)

    # extract the branch name
    branch = bInfo_dict['branch']

    db = ASVDb(dbDir=str(asv_dir_path),
               repo=repo_url,
               branches=[branch])

    for res in bResultList:
        db.addResult(bInfo, res)
    

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--commitHash", type=str, required=True,
                    help="project version")
    ap.add_argument("--commitTime", type=str, required=True,
                    help="project version date")
    ap.add_argument("--repo-url", type=str, required=True,
                    help="project repo url")
    ap.add_argument("--branch", type=str, required=True,
                    help="project branch")
    ap.add_argument("--results-dir", type=str, required=True,
                    help="directory to store the results in json files")
    ap.add_argument("--machine-name", type=str, required=True,
                    help="Slurm cluster name")
    ap.add_argument("--gpu-type", type=str, required=True,
                    help="the official product name of the GPU")
    args = ap.parse_args()

    update_asv_db(commitHash=args.commitHash,
                  commitTime=int(args.commitTime),
                  branch=args.branch,
                  repo_url=args.repo_url,
                  results_dir=args.results_dir,
                  machine_name=args.machine_name,
                  gpu_type=args.gpu_type)
