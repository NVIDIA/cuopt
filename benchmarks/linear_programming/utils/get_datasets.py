# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import urllib.request
import urllib.parse
import ssl
import subprocess


# From: http://plato.asu.edu/bench.html
# Folder containg instances:
# - https://miplib2010.zib.de/miplib2010.php
# - https://www.netlib.org/lp/data/
# - http://old.sztaki.hu/~meszaros/public_ftp/lptestset/ (and it's subfolders)
# - https://plato.asu.edu/ftp/lptestset/ (and it's subfolders)
# - https://miplib.zib.de/tag_benchmark.html
# - https://miplib.zib.de/tag_collection.html

LPFeasibleMittelmannSet = [
    "L1_sixm250obs",
    "Linf_520c",
    "a2864",
    "bdry2",
    "cont1",
    "cont11",
    "datt256_lp",
    "dlr1",
    "ex10",
    "fhnw-binschedule1",
    "fome13",
    "graph40-40",
    "irish-electricity",
    "neos",
    "neos3",
    "neos-3025225",
    "neos-5052403-cygnet",
    "neos-5251015",
    "ns1687037",
    "ns1688926",
    "nug08-3rd",
    "pds-100",
    "physiciansched3-3",
    "qap15",
    "rail02",
    "rail4284",
    "rmine15",
    "s82",
    "s100",
    "s250r10",
    "savsched1",
    "scpm1",
    "shs1023",
    "square41",
    "stat96v2",
    "stormG2_1000",
    "stp3d",
    "supportcase10",
    "tpl-tub-ws1617",
    "woodlands09",
    "Dual2_5000",
    "Primal2_1000",
    "thk_48",
    "thk_63",
    "L1_sixm1000obs",
    "L2CTA3D",
    "degme",
    "dlr2",
    "set-cover-model"
]

MittelmannInstances = {
    "emps": "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/emps.c",
    "problems" : {
        "irish-electricity" : [
            "https://plato.asu.edu/ftp/lptestset/irish-electricity.mps.bz2",
            "mps"
        ],
        "physiciansched3-3" : [
            "https://plato.asu.edu/ftp/lptestset/physiciansched3-3.mps.bz2",
            "mps"
        ],
        "16_n14" : [
            "http://plato.asu.edu/ftp/lptestset/network/16_n14.mps.bz2",
            "mps"
        ],
        "Dual2_5000" : [
            "http://plato.asu.edu/ftp/lptestset/Dual2_5000.mps.bz2",
            "mps"
        ],
        "L1_six1000" : [
            "http://plato.asu.edu/ftp/lptestset/L1_sixm1000obs.bz2",
            "netlib"
        ],
        "L1_sixm" : ["", "mps"],
        "L1_sixm1000obs" : [
            "http://plato.asu.edu/ftp/lptestset/L1_sixm1000obs.bz2",
            "netlib"
        ],
        "L1_sixm250" : ["", "netlib"],
        "L1_sixm250obs" : [
            "http://plato.asu.edu/ftp/lptestset/L1_sixm250obs.bz2",
            "netlib"
        ],
        "L2CTA3D" : [
            "http://plato.asu.edu/ftp/lptestset/L2CTA3D.mps.bz2",
            "mps"
        ],
        "Linf_520c" : [
            "http://plato.asu.edu/ftp/lptestset/Linf_520c.bz2",
            "netlib"
        ],
        "Primal2_1000" : [
            "http://plato.asu.edu/ftp/lptestset/Primal2_1000.mps.bz2",
            "mps"
        ],
        "a2864" : [
            "http://plato.asu.edu/ftp/lptestset/a2864.mps.bz2",
            "mps"
        ],
        "bdry2" : [
            "http://plato.asu.edu/ftp/lptestset/bdry2.bz2",
            "netlib"
        ],
        "braun" : ["", "mps"],
        "cont1" : [
            "http://plato.asu.edu/ftp/lptestset/misc/cont1.bz2",
            "netlib"
        ],
        "cont11" : [
            "http://plato.asu.edu/ftp/lptestset/misc/cont11.bz2",
            "netlib"
        ],
        "datt256" : [
            "http://plato.asu.edu/ftp/lptestset/datt256_lp.mps.bz2",
            "mps"
        ],
        "datt256_lp" : [
            "http://plato.asu.edu/ftp/lptestset/datt256_lp.mps.bz2",
            "mps"
        ],
        "degme" : [
            "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/New/degme.gz",
            "netlib"
        ],
        "dlr1" : [
            "http://plato.asu.edu/ftp/lptestset/dlr1.mps.bz2",
            "mps"
        ],
        "dlr2" : [
            "http://plato.asu.edu/ftp/lptestset/dlr2.mps.bz2",
            "mps"
        ],
        "energy1" : ["", "mps"], # Kept secret by Mittlemman
        "energy2" : ["", "mps"],
        "ex10" : [
            "http://plato.asu.edu/ftp/lptestset/ex10.mps.bz2",
            "mps"
        ],
        "fhnw-binschedule1" : [
            "http://plato.asu.edu/ftp/lptestset/fhnw-binschedule1.mps.bz2",
            "mps"
        ],
        "fome13" : [
            "http://plato.asu.edu/ftp/lptestset/fome/fome13.bz2",
            "netlib"
        ],
        "gamora" : ["", "mps"], # Kept secret by Mittlemman
        "goto14_256_1" : ["", "mps"],
        "goto14_256_2" : ["", "mps"],
        "goto14_256_3" : ["", "mps"],
        "goto14_256_4" : ["", "mps"],
        "goto14_256_5" : ["", "mps"],
        "goto16_64_1" : ["", "mps"],
        "goto16_64_2" : ["", "mps"],
        "goto16_64_3" : ["", "mps"],
        "goto16_64_4" : ["", "mps"],
        "goto16_64_5" : ["", "mps"],
        "goto32_512_1" : ["", "mps"],
        "goto32_512_2" : ["", "mps"],
        "goto32_512_3" : ["", "mps"],
        "goto32_512_4" : ["", "mps"],
        "goto32_512_5" : ["", "mps"],
        "graph40-40" : [
            "http://plato.asu.edu/ftp/lptestset/graph40-40.mps.bz2",
            "mps"
        ],
        "graph40-40_lp" : [
            "http://plato.asu.edu/ftp/lptestset/graph40-40.mps.bz2",
            "mps"
        ],
        "groot" : ["", "mps"], # Kept secret by Mittlemman
        "heimdall" : ["", "mps"], # Kept secret by Mittlemman
        "hulk" : ["", "mps"], # Kept secret by Mittlemman
        "i_n13" : [
            "http://plato.asu.edu/ftp/lptestset/network/i_n13.mps.bz2",
            "mps"
        ],
        "irish-e" : ["", "mps"],
        "karted" : [
            "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/New/karted.gz",
            "netlib"
        ],
        "lo10" : [
            "http://plato.asu.edu/ftp/lptestset/network/lo10.mps.bz2",
            "mps"
        ],
        "loki" : ["", "mps"], # Kept secret by Mittlemman
        "long15" : [
            "http://plato.asu.edu/ftp/lptestset/network/long15.mps.bz2",
            "mps"
        ],
        "nebula" : ["", "mps"], # Kept secret by Mittlemman
        "neos" : [
            "http://plato.asu.edu/ftp/lptestset/misc/neos.bz2",
            "netlib"
        ],
        "neos-3025225" : [
            "http://plato.asu.edu/ftp/lptestset/neos-3025225.mps.bz2",
            "mps"
        ],
        "neos-3025225_lp" : [
            "http://plato.asu.edu/ftp/lptestset/neos-3025225.mps.bz2",
            "mps"
        ],
        "neos-5251015" : [
            "http://plato.asu.edu/ftp/lptestset/neos-5251015.mps.bz2",
            "mps"
        ],
        "neos-5251015_lp" : [
            "http://plato.asu.edu/ftp/lptestset/neos-5251015.mps.bz2",
            "mps"
        ],
        "neos3" : [
            "http://plato.asu.edu/ftp/lptestset/misc/neos3.bz2",
            "netlib"
        ],
        "neos-5052403-cygnet" : [
            "http://plato.asu.edu/ftp/lptestset/neos-5052403-cygnet.mps.bz2",
            "mps"
        ],
        "neos5251015_lp" : [
            "http://plato.asu.edu/ftp/lptestset/neos-5251015.mps.bz2",
            "mps"
        ],
        "neos5251915" : [
            "http://plato.asu.edu/ftp/lptestset/neos-5251015.mps.bz2",
            "mps"
        ],
        "netlarge1" : [
            "http://plato.asu.edu/ftp/lptestset/network/netlarge1.mps.bz2",
            "mps"
        ],
        "netlarge2" : [
            "http://plato.asu.edu/ftp/lptestset/network/netlarge2.mps.bz2",
            "mps"
        ],
        "netlarge3" : [
            "http://plato.asu.edu/ftp/lptestset/network/netlarge3.mps.bz2",
            "mps"
        ],
        "netlarge6" : [
            "http://plato.asu.edu/ftp/lptestset/network/netlarge6.mps.bz2",
            "mps"
        ],
        "ns1687037" : [
            "http://plato.asu.edu/ftp/lptestset/misc/ns1687037.bz2",
            "netlib"
        ],
        "ns1688926" : [
            "http://plato.asu.edu/ftp/lptestset/misc/ns1688926.bz2",
            "netlib"
        ],
        "nug08-3rd" : [
            "http://plato.asu.edu/ftp/lptestset/nug/nug08-3rd.bz2",
            "netlib"
        ],
        "pds-100" : [
            "http://plato.asu.edu/ftp/lptestset/pds/pds-100.bz2",
            "netlib"
        ],
        "psched3-3" : ["", "mps"],
        "qap15" : [
            "http://plato.asu.edu/ftp/lptestset/qap15.mps.bz2",
            "mps"
        ],
        "rail02" : [
            "https://miplib2010.zib.de/download/rail02.mps.gz",
            "mps"
        ],
        "rail4284" : [
            "http://plato.asu.edu/ftp/lptestset/rail/rail4284.bz2",
            "netlib"
        ],
        "rmine15" : [
            "http://plato.asu.edu/ftp/lptestset/rmine15.mps.bz2",
            "mps"
        ],
        "s100" : [
            "http://plato.asu.edu/ftp/lptestset/s100.mps.bz2",
            "mps"
        ],
        "s250r10" : [
            "http://plato.asu.edu/ftp/lptestset/s250r10.mps.bz2",
            "mps"
        ],
        "s82" : [
            "http://plato.asu.edu/ftp/lptestset/s82.mps.bz2",
            "mps"
        ],
        "savsched1" : [
            "http://plato.asu.edu/ftp/lptestset/savsched1.mps.bz2",
            "mps"
        ],
        "scpm1" : [
            "http://plato.asu.edu/ftp/lptestset/scpm1.mps.bz2",
            "mps"
        ],
        "set-cover-model" : [
            "http://plato.asu.edu/ftp/lptestset/set-cover-model.mps.bz2",
            "mps"
        ],
        "shs1023" : [
            "https://miplib2010.zib.de/download/shs1023.mps.gz",
            "mps"
        ],
        "square15" : [
            "http://plato.asu.edu/ftp/lptestset/network/square15.mps.bz2",
            "mps"
        ],
        "square41" : [
            "http://plato.asu.edu/ftp/lptestset/square41.mps.bz2",
            "mps"
        ],
        "stat96v2" : [
            "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/misc/stat96v2.gz",
            "netlib"
        ],
        "stormG2_1000" : [
            "http://plato.asu.edu/ftp/lptestset/misc/stormG2_1000.bz2",
            "netlib"
        ],
        "storm_1000" : ["", "mps"],
        "stp3d" : [
            "https://miplib.zib.de/WebData/instances/stp3d.mps.gz",
            "mps"
        ],
        "supportcase10" : [
            "http://plato.asu.edu/ftp/lptestset/supportcase10.mps.bz2",
            "mps"
        ],
        "support19" : [
            "http://plato.asu.edu/ftp/lptestset/supportcase19.mps.bz2",
            "mps"
        ],
        "supportcase19" : [
            "http://plato.asu.edu/ftp/lptestset/supportcase19.mps.bz2",
            "mps"
        ],
        "test03" : ["", "mps"], # Kept secret by Mittlemman
        "test13" : ["", "mps"], # Kept secret by Mittlemman
        "test23" : ["", "mps"], # Kept secret by Mittlemman
        "test33" : ["", "mps"], # Kept secret by Mittlemman
        "test43" : ["", "mps"], # Kept secret by Mittlemman
        "test53" : ["", "mps"], # Kept secret by Mittlemman
        "test63" : ["", "mps"], # Kept secret by Mittlemman
        "test83" : ["", "mps"], # Kept secret by Mittlemman
        "test93" : ["", "mps"], # Kept secret by Mittlemman
        "mars" : ["", "mps"],  # Kept secret by Mittlemman
        "thk_48" : ["https://plato.asu.edu/ftp/lptestset/thk_48.mps.bz2", "mps"],
        "thk_63" : [
            "https://plato.asu.edu/ftp/lptestset/thk_63.mps.bz2",
            "mps"
        ],
        "thor" : ["", "mps"],  # Kept secret by Mittlemman
        "tpl-tub-ws" : ["", "mps"],
        "tpl-tub-ws1617" : [
            "http://plato.asu.edu/ftp/lptestset/tpl-tub-ws1617.mps.bz2",
            "mps"
        ],
        "wide15" : [
            "http://plato.asu.edu/ftp/lptestset/network/wide15.mps.bz2",
            "mps"
        ],
        "woodlands09" : [
            "http://plato.asu.edu/ftp/lptestset/woodlands09.mps.bz2",
            "mps"
        ]
    },
    "benchmarks" : {
        "simplex" : [
            "L1_sixm",
            "L1_sixm250obs",
            "Linf_520c",
            "a2864",
            "bdry2",
            "braun",
            "cont1",
            "cont11",
            "datt256",
            "dlr1",
            "energy1",
            "energy2",
            "ex10",
            "fhnw-binschedule1",
            "fome13",
            "gamora",
            "graph40-40",
            "groot",
            "heimdall",
            "hulk",
            "irish-e",
            "loki",
            "nebula",
            "neos",
            "neos-3025225_lp",
            "neos-5251015_lp",
            "neos3",
            "neos3025225",
            "neos5052403",
            "neos5251015_lp",
            "ns1687037",
            "ns1688926",
            "nug08-3rd",
            "pds-100",
            "psched3-3",
            "qap15",
            "rail02",
            "rail4284",
            "rmine15",
            "s100",
            "s250r10",
            "s82",
            "savsched1",
            "scpm1",
            "shs1023",
            "square41",
            "stat96v2",
            "stormG2_1000",
            "storm_1000",
            "stp3d",
            "support10",
            "test03",
            "test13",
            "test23",
            "test33",
            "test43",
            "test53",
            "thor",
            "tpl-tub-ws",
            "tpl-tub-ws16",
            "woodlands09"
        ],
        "barrier" : [
            "Dual2_5000",
            "L1_six1000",
            "L1_sixm1000obs",
            "L1_sixm250",
            "L1_sixm250obs",
            "L2CTA3D",
            "Linf_520c",
            "Primal2_1000",
            "a2864",
            "bdry2",
            "cont1",
            "cont11",
            "datt256",
            "degme",
            "dlr1",
            "dlr2",
            "ex10",
            "fhnw-binschedule1",
            "fome13",
            "graph40-40",
            "irish-e",
            "karted",
            "neos",
            "neos-3025225_lp",
            "neos-5251015_lp",
            "neos3",
            "neos3025225",
            "neos5052403",
            "neos5251915",
            "ns1687037",
            "ns1688926",
            "nug08-3rd",
            "pds-100",
            "psched3-3",
            "qap15",
            "rail02",
            "rail4284",
            "rmine15",
            "s100",
            "s250r10",
            "s82",
            "savsched1",
            "scpm1",
            "set-cover-model",
            "shs1023",
            "square41",
            "stat96v2",
            "stormG2_1000",
            "storm_1000",
            "stp3d",
            "support10",
            "support19",
            "supportcase19",
            "thk_63",
            "tpl-tub-ws",
            "tpl-tub-ws16",
            "woodlands09"
        ],
        "large" : [
            "16_n14",
            "goto14_256_1",
            "goto14_256_2",
            "goto14_256_3",
            "goto14_256_4",
            "goto14_256_5",
            "goto16_64_1",
            "goto16_64_2",
            "goto16_64_3",
            "goto16_64_4",
            "goto16_64_5",
            "goto32_512_1",
            "goto32_512_2",
            "goto32_512_3",
            "goto32_512_4",
            "goto32_512_5",
            "i_n13",
            "lo10",
            "long15",
            "netlarge1",
            "netlarge2",
            "netlarge3",
            "netlarge6",
            "square15",
            "wide15"
        ],
        # <=100s in bench: http://plato.asu.edu/ftp/lpbar.html
        "L0" : [
            "ex10",
            "datt256",
            "graph40-40",
            "neos5251915",
            "nug08-3rd",
            "qap15",
            "savsched1",
            "scpm1",
            "a2864",
            "support10",
            "rmine15",
            "fome13",
            "L2CTA3D",
            "neos5052403",
            "karted",
            "stp3d",
            "woodlands09",
            "rail4284",
            "L1_sixm250",
            "tpl-tub-ws"
        ],
        #>100 <1000
        "L1" : [
            "s250r10",
            "pds-100",
            "set-cover-model",
            "neos3025225",
            "rail02",
            "square41",
            "degme",
            "Linf_520c",
            "cont1",
            "neos",
            "stat96v2",
            "support19",
            "shs1023",
            "storm_1000"
        ],
        # >1000
        "L2" : [
            "thk_63",
            "Primal2_1000",
            "L1_six1000",
            "Dual2_5000",
            "s100",
            "fhnw-binschedule1",
            "cont11",
            "psched3-3"
        ],
        #t -> >15000
        "L3" : [
            "dlr2",
            "bdry2",
            "dlr1",
            "irish-e",
            "ns1687037",
            "ns1688926",
            "s82"
        ],
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="get_datasets",
        description="Download Linear Programming datasets")
    parser.add_argument(
        "-LPfeasible",
        action="store_true",
        default=False,
        help="Download only the feasible instances from the Mittelmann set.")
    parser.add_argument(
        "-instance-download-path",
        type=str,
        default="datasets/linear_programming",
        help="Path where the instances will be downloaded")
    parser.add_argument(
        "-datasets",
        action="append",
        help="Name of datasets to be downloaded. To download all the datasets "
             "pass 'ALL'.")
    parser.add_argument(
        "-benchmarks",
        action="append",
        help="Name of the benchmark suites (containing the list of datasets) "
             "to be downloaded.")
    parser.add_argument(
        "-list-benchmarks",
        action="store_true",
        default=False,
        help="List the names of all the benchmarks")
    parser.add_argument(
        "-list-datasets",
        action="store_true",
        default=False,
        help="List the names of all the datasets")
    parser.add_argument(
        "-root",
        type=str,
        default="datasets/linear_programming",
        help="Root folder to store the downloaded datasets")
    args = parser.parse_args()
    if args.datasets and len(args.datasets) == 1 and args.datasets[0] == "ALL":
        args.datasets = MittelmannInstances["problems"].keys()
    return args


def download(url, dst):
    if os.path.exists(dst):
        return
    print(f"Downloading {url} into {dst}...")
    # Bypass SSL verification for plato.asu.edu URLs
    if "plato.asu.edu" in url:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        response = urllib.request.urlopen(url, context=context)
    else:
        response = urllib.request.urlopen(url)
    data = response.read()
    with open(dst, "wb") as fp:
        fp.write(data)

def extract(file, dir, type):
    basefile = os.path.basename(file)
    outfile = ""
    unzippedfile = ""
    if basefile.endswith(".bz2"):
        outfile = basefile.replace(".bz2", ".mps")
        unzippedfile = basefile.replace(".bz2", "")
        subprocess.run(f"cd {dir} && bzip2 -d {basefile}", shell=True)
    elif basefile.endswith(".gz"):
        outfile = basefile.replace(".gz", ".mps")
        unzippedfile = basefile.replace(".gz", "")
        subprocess.run(f"cd {dir} && gunzip -c {basefile} > {unzippedfile}",
                       shell=True)
    else:
        raise Exception(f"Unknown file extension found for extraction {file}")
    #    download emps and compile
    if type == "netlib":
        url = MittelmannInstances["emps"]
        file = os.path.join(dir, "emps.c")
        download(url, file)
        subprocess.run(f"cd {dir} && gcc emps.c -o emps",
                        shell=True)
        # determine output file and run emps
        subprocess.run(f"cd {dir} && ./emps {unzippedfile} > {outfile}",
                        shell=True)
        # cleanup emps and emps.c
        subprocess.run(f"rm -rf {dir}/emps*",
                        shell=True)


def download_dataset(name, root):
    if name not in MittelmannInstances["problems"]:
        raise Exception(f"Unknown dataset {name} passed")
    dir = os.path.join(root, name)
    if os.path.exists(dir):
        if os.path.exists(os.path.join(dir, f"{name}.mps")):
            print(f"Dir for dataset {name} exists and contains {name}.mps. Skipping...")
            return
    url, type  = MittelmannInstances["problems"][name]
    if url == "":
        print(f"Dataset {name} doesn't have a URL. Skipping...")
        return
    else:
        os.mkdir(dir)
    file = os.path.join(dir, os.path.basename(url))
    download(url, file)
    extract(file, dir, type)


def main():
    args = parse_args()
    if args.list_datasets:
        for name in MittelmannInstances["problems"].keys():
            print(name)
    if args.list_benchmarks:
        print("All benchmarks:")
        for bench in MittelmannInstances["benchmarks"].keys():
            print(f"  {bench}")
    if args.list_datasets or args.list_benchmarks:
        return
    instance_download_path = args.root
    if args.instance_download_path:
        if not os.path.exists(args.instance_download_path):
            os.makedirs(args.instance_download_path)
        instance_download_path = args.instance_download_path
    if args.LPfeasible:
        for name in LPFeasibleMittelmannSet:
            download_dataset(name, instance_download_path)
    if args.datasets:
        for name in args.datasets:
            download_dataset(name, instance_download_path)
    if args.benchmarks:
        for bench in args.benchmarks:
            for name in MittelmannInstances["benchmarks"][bench]:
                download_dataset(name, instance_download_path)
    return


if __name__ == "__main__":
    main()
