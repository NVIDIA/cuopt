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

    Filename:     LocalVar.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "LocalVar.h"

LocalVar::LocalVar() : allowIncStep(0), allowDecStep(0), lastIncStep(0), lastDecStep(0) {}

LocalVar::~LocalVar() {}

LocalVarUtil::LocalVarUtil() {}

void LocalVarUtil::Allocate(size_t _varNum, size_t _varNumInObj)
{
  lowerDeltaInLiftMove.clear();
  upperDeltaInLifiMove.clear();
  scoreTable.clear();
  affectedVar.clear();
  varSet.clear();
  tempDeltas.clear();
  tempVarIdxs.clear();

  tempDeltas.reserve(_varNum);
  tempVarIdxs.reserve(_varNum);
  affectedVar.reserve(_varNum);
  varSet.resize(_varNum);
  scoreTable.resize(_varNum, false);
  lowerDeltaInLiftMove.resize(_varNumInObj);
  upperDeltaInLifiMove.resize(_varNumInObj);
}

LocalVarUtil::~LocalVarUtil()
{
  lowerDeltaInLiftMove.clear();
  upperDeltaInLifiMove.clear();
  scoreTable.clear();
  affectedVar.clear();
  varSet.clear();
  tempDeltas.clear();
  tempVarIdxs.clear();
}