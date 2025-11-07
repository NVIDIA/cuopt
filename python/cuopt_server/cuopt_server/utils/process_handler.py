# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import queue
import time
from multiprocessing import Event, Process

import psutil

from cuopt_server.utils.job_queue import ExitJob, Shutdown, abort_by_pid


class SolverProcess:
    def __init__(self, process, gpu_id, complete):
        self.process = process
        self.gpuid = gpu_id
        self.complete = complete


s_procs = {}


def kill_pid(pid):
    try:
        p = psutil.Process(pid)
        p.kill()
    except Exception:
        pass


def terminate(job_queue, results_queue, abort_queue, signame):
    global s_procs

    logging.info("terminate called")
    # Solver and receiver may be asleep
    for i in range(len(s_procs)):
        job_queue.put(ExitJob())
    results_queue.put(Shutdown(signame))
    job_queue.close()
    results_queue.close()
    abort_queue.close()

    # Force parent process to exit
    solvers_done = False
    n = time.time()
    while time.time() - n < 0.2:
        incomplete = False
        for _, s in s_procs.items():
            if not s.complete.is_set():
                incomplete = True
                break
        else:
            solvers_done = True
            break

        if incomplete:
            time.sleep(0.01)

    if not solvers_done:
        logging.info(
            "all jobs completed but solvers may still be running, forcing quit"
        )
        for pid, _ in s_procs.items():
            logging.info(f"sending kill to {pid}")
            kill_pid(pid)
        logging.info("solver terminated")


def create_process(app_exit, job_queue, results_queue, abort_list, gpu_id):
    global s_procs

    complete = Event()

    from cuopt_server.utils import solver

    s = Process(
        target=solver.process_async_solve,
        args=(
            app_exit,
            complete,
            job_queue,
            results_queue,
            abort_list,
            gpu_id,
        ),
    )
    s.start()
    logging.info(f"Starting new process with pid {s.pid}")
    s_procs[s.pid] = SolverProcess(s, gpu_id, complete)


def watch_solvers(app_exit, job_queue, results_queue, abort_queue, abort_list):
    global s_procs

    # Exhaust the abort queue
    while True:
        try:
            abort_id = abort_queue.get(timeout=0.5)
            if app_exit.is_set():
                # don't bother, everything is going to shutdown
                return
            kill_pid(abort_id)
            gpuid = s_procs[abort_id].gpuid
            del s_procs[abort_id]
            create_process(
                app_exit, job_queue, results_queue, abort_list, gpuid
            )
        except queue.Empty:
            pass

        if app_exit.is_set():
            # everything is exiting, main will check s_procs
            return

        to_remove = []
        for pid, s in s_procs.items():
            if s.complete.is_set() or not s.process.is_alive():
                # Send a completion to any job on the abort
                # list that has this pid listed ...
                abort_by_pid(pid, abort_list, results_queue)

                # No kill needed, just restart
                to_remove.append(pid)

        # Solvers shouldn't normally be exiting unless we are shutting down, so
        # go ahead and restart, maybe we got a CUDA error or something
        for pid in to_remove:
            create_process(
                app_exit,
                job_queue,
                results_queue,
                abort_list,
                s_procs[pid].gpuid,
            )
            del s_procs[pid]


def get_solver_processes():
    global s_procs
    return s_procs
