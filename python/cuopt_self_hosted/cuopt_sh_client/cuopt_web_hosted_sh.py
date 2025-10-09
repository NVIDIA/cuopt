#! /usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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
import json
import logging
import os
import warnings

from cuopt_sh_client import (
    CuOptServiceSelfHostClient,
    CuOptServiceWebHostedClient,
    create_client,
    get_version,
    is_uuid,
    mime_type,
    set_log_level,
)

port_default = "5000"
ip_default = "0.0.0.0"

result_types = {
    "json": mime_type.JSON,
    "msgpack": mime_type.MSGPACK,
    "zlib": mime_type.ZLIB,
    "*": mime_type.WILDCARD,
}


def create_cuopt_client(args):
    """Create the appropriate CuOpt client based on arguments."""
    # Check if web-hosted parameters are provided
    endpoint = getattr(args, 'endpoint', None)
    api_key = getattr(args, 'api_key', None)
    
    # Validate parameter combinations
    if endpoint and (args.ip != ip_default or args.port != port_default):
        warnings.warn(
            "Both endpoint and ip/port parameters provided. "
            "Endpoint takes precedence. Consider using only endpoint parameter.",
            UserWarning
        )
    
    if api_key and not endpoint:
        raise ValueError(
            "API key provided but no endpoint specified. "
            "Web-hosted authentication requires an endpoint URL. "
            "Use -e/--endpoint to specify the service endpoint."
        )
    
    # Create client using factory function
    try:
        return create_client(
            endpoint=endpoint,
            api_key=api_key,
            ip=args.ip,
            port=args.port,
            use_https=args.ssl,
            self_signed_cert=args.self_signed_cert,
            polling_timeout=getattr(args, 'poll_timeout', 120),
            only_validate=getattr(args, 'only_validation', False),
            timeout_exception=False,
            result_type=result_types[args.result_type],
            http_general_timeout=getattr(args, 'http_timeout', 30),
        )
    except ValueError as e:
        # Re-raise with more helpful CLI context
        raise ValueError(f"Client creation failed: {e}")


def status(args):
    cuopt_service_client = create_cuopt_client(args)

    try:
        solve_result = cuopt_service_client.status(args.data[0])

        if solve_result:
            print(solve_result)

    except Exception as e:
        if args.log_level == "debug":
            import traceback
            traceback.print_exc()
        print(str(e))


def delete_request(args):
    cuopt_service_client = create_cuopt_client(args)

    try:
        # Interpretation of flags is done on the server, just pass them
        solve_result = cuopt_service_client.delete(
            args.data[0],
            queued=args.queued,
            running=args.running,
            cached=args.cache,
        )

        if solve_result:
            print(solve_result)

    except Exception as e:
        if args.log_level == "debug":
            import traceback
            traceback.print_exc()
        print(str(e))


def upload_solution(args):
    cuopt_service_client = create_cuopt_client(args)

    try:
        # Interpretation of flags is done on the server, just pass them
        reqId = cuopt_service_client.upload_solution(args.data[0])

        if reqId:
            print(reqId)

    except Exception as e:
        if args.log_level == "debug":
            import traceback
            traceback.print_exc()
        print(str(e))


def delete_solution(args):
    cuopt_service_client = create_cuopt_client(args)

    try:
        solve_result = cuopt_service_client.delete_solution(args.data[0])

        if solve_result:
            print(solve_result)

    except Exception as e:
        if args.log_level == "debug":
            import traceback
            traceback.print_exc()
        print(str(e))


def solve(args):
    problem_data = args.data

    # Set the problem data
    solver_settings = None
    repoll_req = False

    def read_input_data(i_file):
        repoll_req = False
        if args.filepath:
            data = i_file
        elif not os.path.isfile(i_file):
            if is_uuid(i_file):
                data = i_file
                repoll_req = True
            else:
                # Might be raw json data for repoll or a problem...
                try:
                    data = json.loads(i_file)
                    repoll_req = "reqId" in data
                except Exception:
                    data = i_file
        else:
            # Allow a repoll requestid to be in a file
            with open(i_file, "r") as f:
                try:
                    val = f.read(32)
                except Exception:
                    val = ""
                if len(val) >= 8 and val[0:8] == '{"reqId"':
                    f.seek(0)
                    data = json.load(f)
                    repoll_req = True  # noqa
                else:
                    data = i_file
        return data, repoll_req

    if len(problem_data) == 1:
        cuopt_problem_data, repoll_req = read_input_data(problem_data[0])
    elif args.type == "LP":
        cuopt_problem_data = []
        for i_file in problem_data:
            i_data, repoll_req = read_input_data(i_file)
            if repoll_req:
                raise Exception(
                    "Cannot have a repoll request in LP batch data"
                )
            cuopt_problem_data.append(i_data)
    else:
        raise Exception("Cannot have multiple problem inputs for VRP")

    if args.type == "LP" and args.solver_settings:
        if not os.path.isfile(args.solver_settings):
            # Might be raw json data...
            try:
                solver_settings = json.loads(args.solver_settings)
            except Exception:
                print("solver settings is neither a filename nor valid JSON")
                return
        else:
            with open(args.solver_settings, "r") as f:
                solver_settings = json.load(f)

    # Initialize the CuOptServiceClient
    cuopt_service_client = create_cuopt_client(args)

    try:
        if repoll_req:
            solve_result = cuopt_service_client.repoll(
                cuopt_problem_data,
                response_type="dict",
                delete_solution=not args.keep,
            )
        elif args.type == "VRP":
            solve_result = cuopt_service_client.get_optimized_routes(
                cuopt_problem_data,
                args.filepath,
                args.cache,
                args.output,
                delete_solution=not args.keep,
                initial_ids=args.init_ids,
            )
        elif args.type == "LP":
            if args.init_ids:
                raise Exception("Initial ids are not supported for LP")

            def log_callback(name):
                def print_log(log):
                    ln = "\n".join(log)
                    if name:
                        with open(name, "a+") as f:
                            f.write(ln)
                    elif ln:
                        print(ln, end="")

                return print_log

            def inc_callback(name):
                def print_inc(sol, cost):
                    if name:
                        with open(name, "a+") as f:
                            f.write(f"{sol} {cost}\n")
                    else:
                        print(sol, cost)

                return print_inc

            logging_callback = None
            incumbent_callback = None
            if args.solver_logs is not None:
                # empty string means log to screen
                if args.solver_logs:
                    with open(args.solver_logs, "w") as f:
                        pass  # truncuate
                logging_callback = log_callback(args.solver_logs)

            if args.incumbent_logs is not None:
                # empty string means log to screen
                if args.incumbent_logs:
                    with open(args.incumbent_logs, "w") as f:
                        pass  # truncuate
                incumbent_callback = inc_callback(args.incumbent_logs)

            solve_result = cuopt_service_client.get_LP_solve(
                cuopt_problem_data,
                solver_settings,
                response_type="dict",
                filepath=args.filepath,
                cache=args.cache,
                output=args.output,
                delete_solution=not args.keep,
                incumbent_callback=incumbent_callback,
                logging_callback=logging_callback,
                warmstart_id=args.warmstart_id,
            )
        else:
            raise Exception("Invalid type of problem.")
        if solve_result:
            if (
                isinstance(solve_result, dict)
                and len(solve_result) == 1
                and "reqId" in solve_result
            ):
                # Build repoll command with appropriate parameters
                repoll = "cuopt_web_sh "
                
                # Add endpoint or ip/port parameters
                endpoint = getattr(args, 'endpoint', None)
                if endpoint:
                    repoll += f"-e '{endpoint}' "
                else:
                    if args.ip != ip_default:
                        repoll += f"-i {args.ip} "
                    if args.port != port_default:
                        repoll += f"-p {args.port} "
                
                # Add authentication parameters if present
                if getattr(args, 'api_key', None):
                    repoll += f"--api-key '{args.api_key}' "
                
                status = repoll + "-st "
                repoll += solve_result["reqId"]
                status += solve_result["reqId"]
                print(
                    "Request timed out.\n"
                    "Check for status with the following command:\n" + status
                )
                print(
                    "\nPoll for a result with the following command:\n"
                    + repoll
                )
            else:
                print(solve_result)

    except Exception as e:
        if args.log_level == "debug":
            import traceback
            traceback.print_exc()
        print(str(e))


def main():
    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    result_types = {
        "json": mime_type.JSON,
        "msgpack": mime_type.MSGPACK,
        "zlib": mime_type.ZLIB,
        "*": mime_type.WILDCARD,
    }

    parser = argparse.ArgumentParser(
        description="Solve a cuOpt problem using a web-hosted or self-hosted service client.",
        epilog="""
Examples:
  # Self-hosted (legacy mode)
  cuopt_web_sh -i 192.168.1.100 -p 5000 problem.json
  
  # Web-hosted with endpoint URL
  cuopt_web_sh -e https://api.nvidia.com/cuopt/v1 --api-key YOUR_KEY problem.json
  
  # Web-hosted with API key (sent as Bearer token)
  cuopt_web_sh -e https://inference.nvidia.com/cuopt --api-key YOUR_API_KEY problem.json
  
Environment Variables:
  CUOPT_API_KEY      - API key for authentication (sent as Bearer token)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data argument
    parser.add_argument(
        "data",
        type=str,
        nargs="*",
        default="",
        help="Filename, or JSON string containing a request id. "
        "Data may be a cuopt problem or a request id "
        "as displayed in the output from a previous request which timed out. "
        "A cuopt problem must be in a file, but a request id may be "
        "passed in a file or as a JSON string. "
        " "
        "For VRP:"
        "A single problem file is expected or file_name."
        " "
        "For LP: "
        "A single problem file in mps/json format or file_name."
        "Batch mode is supported in case of mps files only for LP and"
        "not for MILP, where a list of mps"
        "files can be shared to be solved in parallel.",
    )
    
    # Web-hosted client parameters
    web_group = parser.add_argument_group('Web-hosted service options')
    web_group.add_argument(
        "-e",
        "--endpoint",
        type=str,
        help="Full endpoint URL for the cuOpt service. Examples: "
        "'https://api.nvidia.com/cuopt/v1', "
        "'https://inference.nvidia.com/cuopt', "
        "'http://my-service.com:8080/api'. "
        "If provided, this takes precedence over -i/-p parameters.",
    )
    web_group.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication. Will be sent as Bearer token. Can also be set via CUOPT_API_KEY environment variable.",
    )
    
    # Legacy self-hosted parameters
    legacy_group = parser.add_argument_group('Self-hosted service options (legacy)')
    legacy_group.add_argument(
        "-i",
        "--ip",
        type=str,
        default=ip_default,
        help=f"Host address for the cuOpt server (default {ip_default}). "
        "Ignored if --endpoint is provided.",
    )
    legacy_group.add_argument(
        "-p",
        "--port",
        type=str,
        default=port_default,
        help="Port for the cuOpt server. Set to empty string ('') to omit "
        f"the port number from the url (default {port_default}). "
        "Ignored if --endpoint is provided.",
    )
    legacy_group.add_argument(
        "-s",
        "--ssl",
        action="store_true",
        help="Use https scheme (default is http). Ignored if --endpoint is provided.",
        default=False,
    )
    legacy_group.add_argument(
        "-c",
        "--self-signed-cert",
        type=str,
        help="Path to self signed certificates only, "
        "skip for standard certificates",
        default="",
    )
    
    # All other existing parameters...
    parser.add_argument(
        "-id",
        "--init-ids",
        type=str,
        nargs="*",
        default=None,
        help="reqId of a solution to use as an initial solution for a "
        "VRP problem. There may be more than one, separated by spaces. "
        "The list of ids will be terminated when the next option flag "
        "is seen or there are no more arguments.",
    )
    parser.add_argument(
        "-wid",
        "--warmstart-id",
        type=str,
        default=None,
        help="reqId of a solution to use as a warmstart data for a "
        "single LP problem. This allows to restart PDLP with a "
        "previous solution context. Not enabled for Batch LP problem",
    )
    parser.add_argument(
        "-ca",
        "--cache",
        action="store_true",
        help="Indicates that the DATA needs to be cached. This does not "
        "solve the problem but stores the problem data and returns the reqId. "
        "The reqId can be used later to solve the problem. This flag also "
        "may be used alongside the delete argument to delete a cached request."
        "(see the server documentation for more detail).",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--filepath",
        action="store_true",
        help="Indicates that the DATA argument is the relative path "
        "of a cuopt data file under the server's data directory. "
        "The data directory is specified when the server is started "
        "(see the server documentation for more detail).",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--delete",
        action="store_true",
        help="Deletes cached requests or aborts requests on the cuOpt server. "
        "The DATA argument may be the specific reqId of a request or "
        "or it may be the wildcard '*' which will match any request. "
        "If a specific reqId is given and the -r, -q, and -ca flags are "
        "all unspecified, the reqId will always be deleted if it exists. "
        "If any of the -r, -q, or -ca flags are specified, a specific reqId "
        "will only be deleted if it matches the specified flags. "
        "If the wildard reqId '*' is given, the -r, -q and/or -ca flags "
        "must always be set explicitly. To flush the request queue, give '*' "
        "as the reqId "
        "and specify the -q flag. To delete all currently running requests, "
        "give '*' as the reqId and specify the -r flag. To clear the request "
        "cache, give '*' as the reqId and specify the -ca flag.",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--running",
        action="store_true",
        help="Aborts a request only if it is running. Should be used with "
        "-d argument",
        default=None,
    )
    parser.add_argument(
        "-q",
        "--queued",
        action="store_true",
        help="Aborts a request only if it is queued. Should be used with "
        "-d argument",
        default=None,
    )
    parser.add_argument(
        "-ds",
        "--delete_solution",
        action="store_true",
        help="Deletes solutions on the cuOpt server. "
        "The DATA argument is the specific reqId of a solution.",
    )
    parser.add_argument(
        "-k",
        "--keep",
        action="store_true",
        help="Do not delete a solution from the server when it is retrieved. "
        "Default is to delete the solution when it is retrieved.",
    )
    parser.add_argument(
        "-st",
        "--status",
        action="store_true",
        help="Report the status of a request "
        "(completed, aborted, running, queued)",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="The type of problem to solve. "
        "Supported options are VRP and LP (defaults to VRP)",
        default="VRP",
    )
    parser.add_argument(
        "-ss",
        "--solver-settings",
        type=str,
        default="",
        help="Filename or JSON string containing "
        "solver settings for LP problem type",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional name of the result file. If the server "
        "has been configured to write results to files and "
        "the size of the result is greater than the configured "
        "limit, the server will write the result to a file with "
        "this name under the server's result directory (see the "
        "server documentation for more detail). A default name will "
        "be used if this is not specified.",
        default="",
    )
    parser.add_argument(
        "-pt",
        "--poll-timeout",
        type=int,
        help="Number of seconds to poll for a result before timing out "
        "and returning a request id to re-query (defaults to 120)",
        default=120,
    )
    parser.add_argument(
        "-rt",
        "--result-type",
        type=str,
        choices=list(result_types.keys()),
        help="Mime type of result in response"
        "If not provided it is set to msgpack",
        default="msgpack",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=list(levels.keys()),
        help="Log level",
        default="info",
    )
    parser.add_argument(
        "-ov",
        "--only-validation",
        action="store_true",
        help="If set, only validates input",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Print client version and exit.",
    )
    parser.add_argument(
        "-sl",
        "--solver-logs",
        nargs="?",
        const="",
        default=None,
        help="If set detailed MIP solver logs will be returned. If a filename "
        "argument is given logs will be written to that file. If no argument "
        "is given logs will be written to stdout.",
    ),
    parser.add_argument(
        "-il",
        "--incumbent-logs",
        nargs="?",
        const="",
        default=None,
        help="If set MIP incumbent solutions will be returned. If a filename "
        "argument is given incumbents will be written to that file. "
        "If no argument is given incumbents will be written to stdout.",
    ),
    parser.add_argument(
        "-us",
        "--upload-solution",
        action="store_true",
        help="Upload a solution to be cached on the server. The reqId "
        "returned may be used as an initial solution for VRP.",
    )
    parser.add_argument(
        "-ht",
        "--http-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for http requests. May need to be increased "
        "for large datasets or slow networks. Default is 30s. "
        "Set to None to never timeout.",
    )

    args = parser.parse_args()
    set_log_level(levels[args.log_level])
    
    if args.version:
        print(get_version())
    elif not args.data:
        print("expected data argument")
        parser.print_help()
    elif args.status:
        status(args)
    elif args.delete:
        delete_request(args)
    elif args.delete_solution:
        delete_solution(args)
    elif args.upload_solution:
        upload_solution(args)
    else:
        solve(args)


if __name__ == "__main__":
    main()
