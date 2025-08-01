/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*=====================================================================================

    Filename:     Solver.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include <valgrind/callgrind.h>

#include "Solver.h"

// #include "ittnotify.h"

Solver::Solver()
{
  modelConUtil = new ModelConUtil();
  modelVarUtil = new ModelVarUtil();
  readerMPS    = new ReaderMPS(modelConUtil, modelVarUtil);
  localMIP     = new LocalMIP(modelConUtil, modelVarUtil);
}

Solver::~Solver() {}

void Solver::Run()
{
  // ParseObj();
  // readerMPS->Read(fileName);
  CALLGRIND_START_INSTRUMENTATION;

  // __itt_domain* domain = __itt_domain_create("cuOpt");
  // __itt_string_handle* task = __itt_string_handle_create("CPU FJ");

  clkStart = chrono::high_resolution_clock::now();
  // __itt_task_begin(domain, __itt_null, __itt_null, task);
  int Result = localMIP->LocalSearch(optimalObj, clkStart);
  // __itt_task_end(domain);
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
}

void Solver::ParseObj()
{
  // fileName   = (char*)OPT(instance).c_str();
  // optimalObj = __global_paras.identify_opt(fileName);
}