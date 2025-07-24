# cuOpt Self-hosted Service Python Client
The cuOpt Self-hosted Service Python Client is a Python interface to enable access to NVIDIA cuOpt running on user-managed hardware.

Install `cuopt-sh-client` via pip
```bash
$ pip install .
```

## Run client using the CLI
After installation, the command-line utility can be accessed using
```bash
$ cuopt_sh cuopt_problem_data.json
```

If a problem times out, a request id will be returned. The id may be
passed back to the CLI to re-poll as shown. This may be done any number
of times until a result is returned.
```bash
$ cuopt_sh cuopt_problem_data.json
Request timed out.
Re-check the status with the following command:
cuopt_sh '{"reqId": "78238a58-d052-40b4-8dae-852be8f7906e"}'

$ cuopt_sh '{"reqId": "78238a58-d052-40b4-8dae-852be8f7906e"}'
{'response': ...}
```

### Abbreviated List of Arguments:
Check the help with 'cuopt_sh -h' for more detailed information.

      data:   cuOpt problem data file or a request id to repoll. If the -f option is used, this indicates the path of a file accessible to the server.
        -id:  space separated list of reqIds to use as initial solutions for VRP problems. The list is terminated by the next option flag or the end of line.
	-wid: reqId of a solution to use as a warmstart for a single LP problem. Not enabled for batch LP problems.
	-ca:  caches a problem on the server so that it may be run multiple times by reqId. Problem is not solved, only cached.
        -f:   Indicates that the DATA argument is the relative path of a cuOpt data file under the server's data directory.
	-d:   Deletes a cached problem or aborts a running or queued solution.
	-r:   Running. This is a flag for use with the -d option.
	-q:   Queued. This is a flag for use with the -d option.
	-ds:  Delete solution. This is used to delete solutions that were previously cached with the -k option to a solve.
	-k:   Keep the solution on the server after it is retrieved. The solution may be deleted later with the -d option.
	-st:  Status of a request (completed, aborted, running, queued)
	-t:   The type of the problem to solve (VRP or LP). Defaults to LP.
	-ss:  Filename or JSON string containing solver settings for LP problem type.
	-pt:  Number of seconds to poll a request for a result before timeing out.
	-rt:  Mime-type of the result data. Defaults to msgpack. Normally you do not have to worry about this option.
        -o:   Override the default name of the result file if the server has been configured with a results directory.
        -i:   Host address of the cuOpt server (default 0.0.0.0)
        -p:   Port of the cuOpt server (default 5000)
        -s:   Use ssl (default False)
        -c:   Path to self signed certificates only, skip for standard certificates (default "")
        -l:   Log level (critical,error,warning,info,debug)
	-ov:  If set, only validates input
        -v:   Print client version and exit
	-sl:  Detailed solver logs for the MIP solver will be logged to the screen or a file
	-il:  Incumbent solutions from the MIP solver will be logged to the screen or a file
	-us:  Upload a solution. Caches a cuOpt solution on the server for reference by reqId as initial solution for VRP

### Best practice: Using the local data file options (-f and -o)

Data describing a cuOpt problem can be quite large, and performance in most cases will be much better if the server can read files from its local filesystem instead of receiving them over HTTP.
To use this feature do the following:

* Configure the server to use a local data directory [(see below)](#configuring-the-cuopt-server-for-the-local-file-feature)
* Write the cuOpt data file to the server's local data directory. The file may be at the root of the data directory, or it may be placed at a path somewhere under the data directory.
* Specify the filename using the -f option to the cuOpt CLI. Give the path of the data file relative to the root of the server's local data directory.

Example:

```bash
# Copy the file to the server's data directory with scp (or some other mechanism)
$ scp cuopt_problem_data.json my_server:/path/of/cuOpt/data/directory/

# Indicate to the CLI that the data is in the server's local data directory with the name 'cuopt_problem_data.json'. Note that a relative path is used.
$ cuopt_sh -f cuopt_problem_data.json
```

The cuOpt results are usually much smaller than the input data and can easily be returned over HTTP. However, it may also be convenient to have results
written to output files as well. To use this feature do the following:

* Configure the server to use a local results directory and set the result size threshold for writing the results to disk [(see below)](#configuring-the-cuopt-server-for-the-local-file-feature)
* Use the -o option to specify a specific name for the result file, or let the server use a default name.
* Read the result from the file indicated in cuOpt's response.

Example:

```bash
$ cuopt_sh -f cuopt_problem_data.json
{'result_file': 'cuopt_problem_data.json.result'}

$ scp my_server:/path/of/cuOpt/results/directory/cuopt_problem_data.json.result .
```

### Data compression

The cuOpt server can accept data that has been compressed with the Python *msgpack*, *pickle*, or *zlib* libraries.
For the highest efficiency, it is recommended that JSON data be generated with Python and written to a file using msgpack.
However, if the data is not written with msgpack, it can still be compressed later using msgpack, pickle, zlib, or left uncompressed.

When sending data passed as a dictionary, or a dictionary generated from an MPS file, the client will compress the data
for efficiency. The default behavior is to try msgpack first, then try pickle if msgpack serialization fails (due to array
size limits for example).

This behavior can be overridden with the following environment variables (normally this is not necessary):

* CUOPT_PREFER_PICKLE=True: try pickle first and use msgpack if pickle fails (instead of vice versa)
* CUOPT_USE_ZLIB=True: always use zlib to compress immediate data and do not use msgpack or pickle

## Run client directly from Python
### Initialize the CuOptServiceSelfHostClient with optional ip, port, and use_ssl args

```bash
	from cuopt_sh_client import CuOptServiceSelfHostClient
        cuopt_service_client = CuOptServiceSelfHostClient(ip, port)
```

### Get optimized routes

To submit a cuOpt data file over HTTP

```bash
        # Send cuOpt data over HTTP
	# Data may be the path to a file OR a dictionary containing
	# a cuOpt problem
        optimized_routes = cuopt_service_client.get_optimized_routes(
            path_to_cuopt_problem_data
        )
```

To specify a data file that is located on the server in the server's data directory

```bash
        # Tell the cuOpt server that the data is in its local data directory
        optimized_routes = cuopt_service_client.get_optimized_routes(
            relative_path_under_servers_data_directory, filepath = True
        )
```

The problem data file should contain a JSON object with the following details

        cost_waypoint_graph_data
        travel_time_waypoint_graph_data
        cost_matrix_data
        travel_time_matrix_data
        fleet_data
        task_data
        solver_config

An example data file 'cuopt_problem_data.json' is included with this client.

For more details see https://docs.nvidia.com/cuopt/user-guide/serv-api.html

## Configuring the cuOpt Server for the Local File Feature

By default, the local file feature is not enabled in the cuOpt server. To configure the feature, set the environment variables described below in the server's container environment.

### Environment variables

To enable reading of cuOpt data files from the local filesystem, set the following

* CUOPT_DATA_DIR: the absolute path of a directory in the cuOpt server's container environment. Typically this path is the mount point for a volume that exists outside of the container.

To enable writing of cuOpt data files to the local filesystem, set

* CUOPT_RESULT_DIR: the absolute path of a directory in the cuOpt server's container environment. Typically this path is the mount point for a volume that exists outside of the container.
* CUOPT_MAX_RESULT: the maximum size in kilobytes of a result that cuOpt will return over HTTP. To have all results written to disk, set this value to 0. The default is 250.
* CUOPT_RESULT_MODE: the Linux file mode (as with the *chmod* command) to apply to result files created by cuOpt. The default is 644.

### Docker example

In this example, we run the image *cuoptimage* and mount the local directories *./data* and *./results* on the container at */cuopt_data* and */cuopt_resulst* respectively. We set the environment variables to tell cuOpt where the data and results directories are, and to write all results to files instead of HTTP (CUOPT_MAX_RESULT=0).

```bash
$ docker run --rm --gpus=all --network=host -v `pwd`/data:/cuopt_data  -v `pwd`/results:/cuopt_results -e CUOPT_DATA_DIR=/cuopt_data -e CUOPT_RESULT_DIR=/cuopt_results -e CUOPT_MAX_RESULT=0 -it cuoptimage
```

### Directory permissions

The data and results directories mounted on the cuOpt container need to be readable and writable by the container user, and also have the execute permission set.
If they are not, the container will print an error message and exit. Be careful to set permissions correctly on those directories before running the cuOpt server.

When running cuOpt with docker and using the default container user:

* the cuOpt data directory should be readable and executable by group 0 (ie the *root* group)
* the cuOpt results directory should be writable and executable by group 0

When running cuOpt with docker and using the *--user* flag to set only the UID:

* the cuOpt data directory should be readable and executable by group 0 or the specified UID
* the cuOpt results directory should be writable and executable by group 0 or the specified UID

When running cuOpt with docker and using the *--user* flag to set both the UID and GID:

* the cuOpt data directory should be readable and executable by the specified UID or the specified GID
* the cuOpt results directory should be writable and executable by the specified UID or the specified GID
