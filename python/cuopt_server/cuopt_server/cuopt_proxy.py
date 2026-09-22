# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC-backed HTTP proxy for LP/MILP/VRP clients.

This process speaks the self-hosted HTTP API and forwards solves to
``cuopt_grpc_server``. It does not run a local job queue or worker pool.
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence


import cuopt_server.utils.settings as settings
from cuopt_server._version import __version__
from cuopt_server.utils.logutil import (
    get_ncaid,
    get_requestid,
    get_solverid,
    message_init,
)

log_fmt = "%(ncaid)s%(requestid)s%(asctime)s.%(msecs)03d %(levelname)s %(message)s%(solverid)s"  # noqa
date_fmt = "%Y-%m-%d %H:%M:%S"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse proxy options from ``argv`` with environment-backed defaults.

    ``argparse`` reports invalid options and numeric environment values by
    raising ``SystemExit``.
    """
    ip = os.environ.get("CUOPT_SERVER_IP", "0.0.0.0")
    port = os.environ.get("CUOPT_SERVER_PORT", 8000)
    grpc_host = os.environ.get("CUOPT_GRPC_HOST", "127.0.0.1")
    grpc_port = os.environ.get("CUOPT_GRPC_PORT", 5001)
    log_level = os.environ.get("CUOPT_SERVER_LOG_LEVEL", "info")
    log_file = os.environ.get("CUOPT_SERVER_LOG_FILE", "")
    datadir = os.environ.get("CUOPT_DATA_DIR", "")
    resultdir = os.environ.get("CUOPT_RESULT_DIR", "")
    maxresult = os.environ.get("CUOPT_MAX_RESULT", 250)
    max_request_size = os.environ.get(
        "CUOPT_MAX_REQUEST_SIZE", 1024 * 1024 * 1024
    )
    resultmode = os.environ.get("CUOPT_RESULT_MODE", "644")
    ssl_certfile = os.environ.get("CUOPT_SSL_CERTFILE", "")
    ssl_keyfile = os.environ.get("CUOPT_SSL_KEYFILE", "")

    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    parser = argparse.ArgumentParser(
        prog="cuopt_proxy",
        description=(
            "HTTP proxy for cuOpt LP/MILP/VRP clients. "
            "Forwards jobs to cuopt_grpc_server."
        ),
    )
    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        help="Bind address (CUOPT_SERVER_IP)",
        default=ip,
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="HTTP listen port (CUOPT_SERVER_PORT)",
        default=port,
    )
    parser.add_argument(
        "--grpc-host",
        type=str,
        help="cuopt_grpc_server host (CUOPT_GRPC_HOST)",
        default=grpc_host,
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        help="cuopt_grpc_server port (CUOPT_GRPC_PORT)",
        default=grpc_port,
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=list(levels.keys()),
        help="Log level (CUOPT_SERVER_LOG_LEVEL)",
        default=log_level,
    )
    parser.add_argument(
        "-f",
        "--log-file",
        type=str,
        help="Log filename (CUOPT_SERVER_LOG_FILE)",
        default=log_file,
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        help="CUOPT_DATA_DIR for CUOPT-DATA-FILE uploads",
        default=datadir,
    )
    parser.add_argument(
        "-r",
        "--result-dir",
        type=str,
        help="CUOPT_RESULT_DIR for large result files",
        default=resultdir,
    )
    parser.add_argument(
        "-m",
        "--max-result",
        type=int,
        help="Result size threshold in KB (CUOPT_MAX_RESULT)",
        default=maxresult,
    )
    parser.add_argument(
        "--max-request-size",
        type=int,
        help="Maximum HTTP request body in bytes (CUOPT_MAX_REQUEST_SIZE)",
        default=max_request_size,
    )
    parser.add_argument(
        "-mo",
        "--mode",
        type=str,
        help="Octal mode for result files (CUOPT_RESULT_MODE)",
        default=resultmode,
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        help="SSL certificate file (CUOPT_SSL_CERTFILE)",
        default=ssl_certfile,
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        help="SSL key file (CUOPT_SSL_KEYFILE)",
        default=ssl_keyfile,
    )
    args = parser.parse_args(argv)
    args._log_level_int = levels[args.log_level]
    return args


def _configure_logging(args: argparse.Namespace) -> None:
    message_init()
    handlers = []
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=args._log_level_int,
        format=log_fmt,
        datefmt=date_fmt,
        handlers=handlers,
        force=True,
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


def main(argv: Sequence[str] | None = None) -> None:
    """Configure and run the HTTP proxy until the server exits."""
    args = parse_args(argv)
    _configure_logging(args)
    logging.info(f"cuOpt HTTP proxy {__version__}")

    if args.data_dir:
        settings.set_data_dir(args.data_dir)
    settings.set_result_dir(args.result_dir, args.max_result, args.mode)

    from cuopt.grpc.linear_programming import Client
    from cuopt.grpc.routing import RoutingClient

    from cuopt_server.proxy_webserver import (
        run_server,
        set_grpc_client,
        set_grpc_routing_client,
        set_max_request_size,
    )

    logging.info(
        f"Connecting to cuopt_grpc_server at {args.grpc_host}:{args.grpc_port}"
    )
    set_grpc_client(Client(args.grpc_host, args.grpc_port))
    set_grpc_routing_client(RoutingClient(args.grpc_host, args.grpc_port))
    set_max_request_size(args.max_request_size)
    run_server(
        args.ip,
        args.port,
        args.log_level,
        args.ssl_certfile,
        args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
