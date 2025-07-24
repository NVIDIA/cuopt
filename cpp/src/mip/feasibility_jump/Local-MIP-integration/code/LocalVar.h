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

    Filename:     LocalVar.h

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#pragma once
#include "utils/paras.h"

class LocalVar {
 public:
  Value nowValue;
  Value bestValue;
  size_t allowIncStep;
  size_t allowDecStep;
  size_t lastIncStep;
  size_t lastDecStep;

  LocalVar();
  ~LocalVar();
};

class LocalVarUtil {
 public:
  vector<LocalVar> varSet;
  vector<Value> lowerDeltaInLiftMove;
  vector<Value> upperDeltaInLifiMove;
  vector<Value> tempDeltas;
  vector<size_t> tempVarIdxs;
  vector<bool> scoreTable;
  vector<size_t> binaryIdx;
  unordered_set<size_t> affectedVar;

  LocalVarUtil();
  ~LocalVarUtil();
  void Allocate(size_t _varNum, size_t _varNumInObj);
  LocalVar& GetVar(size_t _idx);
};