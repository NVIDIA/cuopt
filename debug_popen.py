#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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


import subprocess
import shutil
import sys
import os
import site
import cuopt_server.cuopt_service

python_path = shutil.which('python')
server_script = cuopt_server.cuopt_service.__file__

print(f'python_path: {python_path}')
print(f'server_script: {server_script}')
print(f'sys.executable: {sys.executable}')

# Get the user's site-packages directory
user_site_packages = site.getusersitepackages()
print(f'user_site_packages: {user_site_packages}')

# Set up environment with PYTHONPATH to include user site-packages
env = os.environ.copy()
env.update({
    'CUOPT_SERVER_IP': '0.0.0.0',
    'CUOPT_SERVER_PORT': '5555',
    'CUOPT_SERVER_LOG_LEVEL': 'debug',
    'PYTHONPATH': user_site_packages,
})

print(f'Environment: {env}')

try:
    proc = subprocess.Popen([python_path, server_script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'Process started with PID: {proc.pid}')
    
    # Wait a bit and check if process is still running
    import time
    time.sleep(2)
    
    if proc.poll() is None:
        print('Process is still running')
        proc.terminate()
        proc.wait()
    else:
        stdout, stderr = proc.communicate()
        print(f'Process exited with code: {proc.returncode}')
        print(f'STDOUT: {stdout.decode()}')
        print(f'STDERR: {stderr.decode()}')
        
except Exception as e:
    print(f'Exception: {e}') 