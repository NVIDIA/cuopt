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

    Filename:     Solver.h

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#pragma once
#include "LocalMIP.h"
#include "ModelCon.h"
#include "ModelVar.h"
#include "ReaderMPS.h"
#include "header.h"

class Solver {
 private:
  char* fileName;
  Value optimalObj;
  void ParseObj();

 public:
  ReaderMPS* readerMPS;
  ModelConUtil* modelConUtil;
  ModelVarUtil* modelVarUtil;
  LocalMIP* localMIP;
  chrono::_V2::system_clock::time_point clkStart = chrono::high_resolution_clock::now();
  Solver();
  ~Solver();
  void Run();
};
