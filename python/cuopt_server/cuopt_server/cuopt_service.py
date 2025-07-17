# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import argparse
import logging
import os
import signal
import sys
from multiprocessing import Event, Process, Queue

import psutil

import cuopt_server.utils.process_handler as process_handler
import cuopt_server.utils.request_filter as request_filter
import cuopt_server.utils.settings as settings
from cuopt_server._version import __version__
from cuopt_server.utils.job_queue import create_abort_list
from cuopt_server.utils.logutil import (
    get_ncaid,
    get_requestid,
    get_solverid,
    message_init,
)

log_fmt = "%(ncaid)s%(requestid)s%(asctime)s.%(msecs)03d %(levelname)s %(message)s%(solverid)s"  # noqa
date_fmt = "%Y-%m-%d %H:%M:%S"


def watcher(app_exit, results_queue, job_queue, abort_queue, abort_list):
    try:
        process_handler.watch_solvers(
            app_exit, job_queue, results_queue, abort_queue, abort_list
        )
        logging.info("app.exit set, watcher exiting")
    except Exception:
        logging.info("exception, watcher exiting")


if __name__ == "__main__":

    import subprocess

    # Do this before setting the signal handler,
    # because sigchld from the subprocess will terminate us :)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        )
        gpu_string = [r for r in result.stdout.split("\n") if r]

        # extract GPU id number from each line (looks like "GPU 0: ..."
        gpu_ids = [s.split(" ")[1][0:-1] for s in gpu_string]

        # if for some reason the parse failed (nvidia-smi output changed)
        gpu_ids = [s for s in gpu_ids if s.isnumeric()]
    except Exception:
        gpu_string = None
        gpu_ids = []

    try:
        cuda_string = None
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        for r in result.stdout.split("\n"):
            if "CUDA Version" in r:
                cuda_string = r
                break
    except Exception:
        pass

    # Flag for this process that says we have already run the
    # exit handler
    terminated = Event()

    # Flag for all processes that the app is shutting down
    app_exit = Event()

    # Flag set by results thread when all jobs have been
    # marked done, to give a chance for anyone actively
    # waiting to get a graceful response
    jobs_marked_done = Event()

    job_queue = Queue()
    abort_queue = Queue()
    results_queue = Queue()

    w = None

    def handle_exit(signame, frame):
        if terminated.is_set():
            return
        if signame:
            logging.info(f"cuopt received signal {signame}")
        if signame == 17:
            return
        terminated.set()
        app_exit.set()
        process_handler.terminate(
            job_queue, results_queue, abort_queue, signame
        )
        logging.info("handle_exit complete")

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGCHLD, handle_exit)

    # Allow defaults to be overridden from the environment.
    # Commandline arguments have the highest precedence.
    # These values all have commandline analogs below.
    ip = os.environ.get("CUOPT_SERVER_IP", "0.0.0.0")
    port = os.environ.get("CUOPT_SERVER_PORT", 5000)
    log_level = os.environ.get("CUOPT_SERVER_LOG_LEVEL", "debug")
    log_file = os.environ.get("CUOPT_SERVER_LOG_FILE", "")
    log_max = os.environ.get("CUOPT_SERVER_LOG_MAX", 10485760)
    log_backup = os.environ.get("CUOPT_SERVER_LOG_BACKUP", 10)
    tier = os.environ.get("CUOPT_SERVER_TIER", "managed_default")
    datadir = os.environ.get("CUOPT_DATA_DIR", "")
    resultdir = os.environ.get("CUOPT_RESULT_DIR", "")
    maxresult = os.environ.get("CUOPT_MAX_RESULT", 250)
    resultmode = os.environ.get("CUOPT_RESULT_MODE", "644")
    ssl_certfile = os.environ.get("CUOPT_SSL_CERTFILE", "")
    ssl_keyfile = os.environ.get("CUOPT_SSL_KEYFILE", "")
    gpu_count = os.environ.get("CUOPT_GPU_COUNT", 1)

    # This setting is experimental, and so is only settable via an
    # env var. This setting allows multiple solver processes per GPU
    # to be allocated, so that problems may be solved in parallel on
    # the same GPU. Additional work may be needed on the best way to
    # set up rmm in this case.
    procs_per_gpu = os.environ.get("CUOPT_PROCS_PER_GPU", 1)

    if not isinstance(procs_per_gpu, int):
        raise ValueError(
            "Process per GPU should be an integer value, "
            f"current value is of type : {type(procs_per_gpu)}"
        )

    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        help=(
            "IP to the server, also can be set by environment "
            "variable CUOPT_SERVER_IP"
        ),
        default=ip,
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help=(
            "Port of the server, also can be set by environment "
            "variable CUOPT_SERVER_PORT"
        ),
        default=port,
    )
    parser.add_argument(
        "-d",
        "--datadir",
        type=str,
        help=(
            "Data directory used to optionally pass cuopt problem data files "
            "to /cuopt/request endpoint via the filesystem instead of over "
            "http, also can be set by environment variable CUOPT_DATA_DIR"
        ),
        default=datadir,
    )
    parser.add_argument(
        "-r",
        "--resultdir",
        type=str,
        help=(
            "Output directory used to optionally pass cuopt result files from "
            "/cuopt/request endpoint via the filesystem instead of over http, "
            "also can be set by environment variable CUOPT_RESULT_DIR"
        ),
        default=resultdir,
    )
    parser.add_argument(
        "-mr",
        "--maxresult",
        type=int,
        help=(
            "Maximum size (kilobytes) of a result returned over http from "
            "/cuopt/request endpoint when RESULTDIR is set. Set to 0 to have "
            "all results written to RESULTDIR (default 250), also can be "
            "set by environment variable CUOPT_MAX_RESULT"
        ),
        default=maxresult,
    )
    parser.add_argument(
        "-mo",
        "--mode",
        type=str,
        help=(
            "Mode (octal) of result files from /cuopt/request endpoint "
            "(default is 644), also can be set by environment variable "
            "CUOPT_RESULT_MODE"
        ),
        default=resultmode,
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=list(levels.keys()),
        help=(
            "Log level, also can be set by environment variable "
            "CUOPT_SERVER_LOG_LEVEL"
        ),
        default=log_level,
    )
    parser.add_argument(
        "-f",
        "--log-file",
        type=str,
        help=(
            "Log filename (by default logs to stdout), also can be set by "
            "environment variable CUOPT_SERVER_LOG_FILE"
        ),
        default=log_file,
    )
    parser.add_argument(
        "-lm",
        "--log-max",
        type=int,
        help=(
            "Max bytes for a log file, also can be set by environment "
            "variable CUOPT_SERVER_LOG_MAX"
        ),
        default=log_max,
    )
    parser.add_argument(
        "-b",
        "--log-backup",
        type=int,
        help=(
            "Number of backup log files, also can be set by environment "
            "variable CUOPT_SERVER_LOG_BACKUP"
        ),
        default=log_backup,
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        help=(
            "SSL certificate file with complete path, also can be set by "
            "environment variable CUOPT_SSL_CERTFILE"
        ),
        default=ssl_certfile,
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        help=(
            "SSL key file with complete path, also can be set by environment "
            "variable CUOPT_SSL_KEYFILE"
        ),
        default=ssl_keyfile,
    )
    parser.add_argument(
        "-g",
        "--gpu-count",
        type=int,
        help=(
            "Number of available GPUs to use for solver processes "
            "(the available GPUs can be affected by arguments to 'docker' "
            "when using a container and by the CUDA_VISIBLE_DEVICES "
            "environment variable). If more GPUs are requested than are "
            "available, cuopt will use all available. Default is 1, zero "
            "will be ignored, also can be set by environment variable "
            "CUOPT_GPU_COUNT"
        ),
        default=gpu_count,
    )
    parser.add_argument(
        "-t",
        "--tier",
        type=str,
        choices=request_filter.tier_list(),
        help=(
            "Feature tier for /cuopt/cuopt endpoint. This names a feature "
            "tier defined in request_filter.py which restricts allowed "
            "features or values for datasets in a request. Not currently "
            "implemented for the /cuopt/request endpoint. "
            "See request_filter.py for details. "
            "Also can be set by environment variable CUOPT_TIER"
        ),
        default=tier,
    )

    args = parser.parse_args()
    if args.log_file:
        if args.log_backup < 1:
            parser.error(
                "Minimum number of backup log files is 1, also can be set by "
                "environment variable CUOPT_SERVER_LOG_BACKUP"
            )
        handlers = [
            logging.handlers.RotatingFileHandler(
                args.log_file, args.log_max, args.log_backup
            )
        ]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        handlers=handlers,
        level=levels[args.log_level],
        format=log_fmt,
        datefmt=date_fmt,
    )
    log_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = log_factory(*args, **kwargs)
        record.ncaid = get_ncaid()
        record.requestid = get_requestid()
        record.solverid = get_solverid()
        if record.ncaid:
            record.ncaid = f"NCA_ID={record.ncaid} "
        if record.requestid:
            record.requestid = f"NVCF_REQID={record.requestid} "
        if record.solverid:
            record.solverid = f" (GPU {record.solverid})"
        return record

    logging.setLogRecordFactory(record_factory)

    message_init()
    logging.info(f"cuopt server version {__version__}")

    request_filter.set_tier(args.tier)
    if args.datadir:
        settings.set_data_dir(args.datadir)
    if args.resultdir:
        settings.set_result_dir(args.resultdir, args.maxresult, args.mode)

    if not terminated.is_set():
        if cuda_string:
            logging.info(cuda_string)
        if gpu_string is None:
            logging.info("failed to read GPU list with nvidia-smi")
        else:
            logging.info(f"{', '.join(gpu_string)}")
        if not gpu_ids:
            logging.debug("failed to determine gpu ids")
        else:
            # validate visible setting if we can
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible:
                filtered = list(set(visible.split(",")).intersection(gpu_ids))
                if not filtered:
                    logging.error(
                        "CUDA_VISIBLE_DEVICES does not allow "
                        f"any available GPUs {gpu_ids}"
                    )
                    sys.exit(1)
                gpu_ids = filtered
                logging.debug(
                    f"gpu_ids after CUDA_VISIBLE_DEVICES filter {gpu_ids}"
                )

        gpu_count = args.gpu_count
        if not isinstance(gpu_count, int):
            gpu_count = 1
        if not gpu_ids:
            # We failed to read the list, skip the id
            gpu_ids = [None]
        else:
            if gpu_count < 1:
                gpu_count = 1
                logging.warning("GPU count cannot be less than 1")
            if gpu_count < len(gpu_ids):
                gpu_ids = gpu_ids[0:gpu_count]

        abort_list = create_abort_list()

        for id in gpu_ids:
            for p in range(int(procs_per_gpu)):
                process_handler.create_process(
                    app_exit, job_queue, results_queue, abort_list, id
                )

        # For whatever reason, if we put this before the solver process
        # then the cuda health checks fail. Check into this.
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as skt:
            if skt.connect_ex(("localhost", args.port)) == 0:
                logging.info(f"Port {args.port} already in use")
                handle_exit("", "")

        from cuopt_server.webserver import run_server

        w = Process(
            target=run_server,
            args=(
                app_exit,
                results_queue,
                job_queue,
                abort_queue,
                abort_list,
                jobs_marked_done,
                args.ip,
                args.port,
                args.log_level,
                args.log_file,
                args.log_max,
                args.log_backup,
                args.ssl_certfile,
                args.ssl_keyfile,
            ),
        )

        w.start()
        watcher(app_exit, results_queue, job_queue, abort_queue, abort_list)
        jobs_marked_done.wait(timeout=1)
        for _, s in process_handler.get_solver_processes().items():
            s.process.join()

    else:
        logging.info("calling handle_exit")
        handle_exit("", "")

    w.join()

    # Check for leftover threads. Sometimes queue feeder threads do not exit
    import threading

    thread_names = {
        t.name for t in threading.enumerate() if t.name != "MainThread"
    }
    if len(thread_names) > 0:
        logging.info("not all threads exited in main process, killing")
        psutil.Process(os.getpid()).kill()
