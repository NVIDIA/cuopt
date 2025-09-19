#!/bin/bash

set -euo pipefail
#straints file for 'pip' and puts its location in an exported variable PIP_EXPORT,
# so those cons
# sets up a contraints will affect all future 'pip install' calls
source rapids-init-pip

./ci/thirdparty-testing/run_jump_tests.sh
