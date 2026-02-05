#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This file is source'd from script-env.sh to add functions to the
# calling environment, hence no #!/bin/bash as the first line. This
# also assumes the variables used in this file have been defined
# elsewhere.

NUMARGS=$#
ARGS=$*
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function logger {
  echo -e ">>>> $*"
}

# Calling "setTee outfile" will cause all stdout and stderr of the
# current script to be output to "tee", which outputs to stdout and
# "outfile" simultaneously. This is useful by allowing a script to
# "tee" itself at any point without being called with tee.
_origFileDescriptorsSaved=0
function setTee {
    if [[ $_origFileDescriptorsSaved == 0 ]]; then
        # Save off the original file descr 1 and 2 as 3 and 4
        exec 3>&1 4>&2
        _origFileDescriptorsSaved=1
    fi
    teeFile=$1
    # Create a named pipe.
    pipeName=$(mktemp -u)
    mkfifo $pipeName
    # Close the currnet 1 and 2 and restore to original (3, 4) in the
    # event this function is called repeatedly.
    exec 1>&- 2>&-
    exec 1>&3 2>&4
    # Start a tee process reading from the named pipe. Redirect stdout
    # and stderr to the named pipe which goes to the tee process. The
    # named pipe "file" can be removed and the tee process stays alive
    # until the fd is closed.
    tee -a < $pipeName $teeFile &
    exec > $pipeName 2>&1
    rm $pipeName
}

# Call this to stop script output from going to "tee" after a prior
# call to setTee.
function unsetTee {
    if [[ $_origFileDescriptorsSaved == 1 ]]; then
        # Close the current fd 1 and 2 which should stop the tee
        # process, then restore 1 and 2 to original (saved as 3, 4).
        exec 1>&- 2>&-
        exec 1>&3 2>&4
    fi
}

# Creates a unique results dir based on date, then links the common
# results dir name to it.
function setupResultsDir {
    mkdir -p ${RESULTS_ARCHIVE_DIR}/${DATE}
    # Store the target of $RESULTS_DIR before $RESULTS_DIR get linked to
    # a different dir
    previous_results=$(readlink -f $RESULTS_DIR)

    rm -rf $RESULTS_DIR
    ln -s ${RESULTS_ARCHIVE_DIR}/${DATE} $RESULTS_DIR
    mkdir -p $TESTING_RESULTS_DIR
    mkdir -p $BENCHMARK_RESULTS_DIR/results/

    old_asv_dir=$previous_results/benchmarks/results/asv
    if [ -d $old_asv_dir ]; then
        cp -r $old_asv_dir $BENCHMARK_RESULTS_DIR/results
    fi
}


# echos the name of the directory that $1 is linked to. Useful for
# getting the actual path of the results dir since that is often
# sym-linked to a unique (based on timestamp) results dir name.
function getNonLinkedFileName {
    linkname=$1
    targetname=$(readlink -f $linkname)
    if [[ "$targetname" != "" ]]; then
        echo $targetname
    else
        echo $linkname
    fi
}

function waitForSlurmJobsToComplete {
    ids=$*
    jobs=$(python -c "print(\",\".join(\"$ids\".split()))") # make a comma-separated list
    jobsInQueue=$(squeue --noheader --jobs=$jobs)
    while [[ $jobsInQueue != "" ]]; do
        sleep 2
        jobsInQueue=$(squeue --noheader --jobs=$jobs)
    done
}

# Clones repo from URL specified by $1 as name $2 in to directory
# $3. For example:
# "cloneRepo https://github.com/rapidsai/cugraph.git /my/repos cg"
# results in cugraph being cloned to /my/repos/cg.
# NOTE: This removes any existing cloned repos that match the
# destination.
function cloneRepo {
    repo_url=$1
    repo_name=$2
    dest_dir=$3
    mkdir -p $dest_dir
    pushd $dest_dir > /dev/null || exit
    logger "Clone $repo_url in $dest_dir..."
    if [ -d $repo_name ]; then
        rm -rf $repo_name
        if [ -d $repo_name ]; then
            echo "ERROR: ${dest_dir}/$repo_name was not completely removed."
            error 1
        fi
    fi
    git clone $repo_url
    popd > /dev/null || exit
}

# Only define this function if it has not already been defined in the
# current environment, which allows the project to override it from
# its functions.sh file that was previously source'd.
if [[ $(type -t activateCondaEnv) == "" ]]; then
    function activateCondaEnv {
        logger "Activating conda env ${CONDA_ENV}..."
        eval "$(conda shell.bash hook)"
        conda activate $CONDA_ENV
    }
fi
