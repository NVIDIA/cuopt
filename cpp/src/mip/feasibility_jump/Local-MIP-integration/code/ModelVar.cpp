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

    Filename:     ModelVar.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "ModelVar.h"

ModelVar::ModelVar(const string& _name, size_t _idx, bool _integrality)
  : name(_name),
    idx(_idx),
    upperBound(DefaultRealUpperBound),
    lowerBound(DefaultLowerBound),
    termNum(-1),
    type(VarType::Real)
{
  if (_integrality) {
    type       = VarType::Binary;
    upperBound = DefaultIntegerUpperBound;
    lowerBound = DefaultLowerBound;
  }
}

ModelVar::~ModelVar()
{
  conIdxSet.clear();
  posInCon.clear();
}

void ModelVar::SetType(VarType _varType) { type = _varType; }

void ModelVar::SetLowerBound(Value _lowerBound)
{
  if (type == VarType::Real)
    lowerBound = _lowerBound;
  else
    lowerBound = ceil(_lowerBound);
}

void ModelVar::SetUpperBound(Value _upperBound)
{
  if (type == VarType::Real)
    upperBound = _upperBound;
  else
    upperBound = floor(_upperBound);
}

bool ModelVar::IsFixed() { return fabs(lowerBound - upperBound) < FeasibilityTol; }

bool ModelVar::IsBinary()
{
  return type == VarType::Binary || type == VarType::Integer &&
                                      fabs(lowerBound - 0.0) < FeasibilityTol &&
                                      fabs(upperBound - 1.0) < FeasibilityTol;
}

ModelVarUtil::ModelVarUtil()
  : integerNum(0), binaryNum(0), fixedNum(0), realNum(0), isBin(true), varNum(-1), objBias(0)
{
}
ModelVarUtil::~ModelVarUtil()
{
  varIdx2ObjIdx.clear();
  name2idx.clear();
  varSet.clear();
}

size_t ModelVarUtil::MakeVar(const string& _name, const bool _integrality)
{
  auto iter = name2idx.find(_name);
  if (iter != name2idx.end()) return iter->second;
  size_t varIdx = varSet.size();
  varSet.emplace_back(_name, varIdx, _integrality);
  name2idx[_name] = varIdx;
  return varIdx;
}

const ModelVar& ModelVarUtil::GetVar(const size_t _idx) const
{
  assert(_idx < varSet.size());
  return varSet[_idx];
}
ModelVar& ModelVarUtil::GetVar(const string& _name) { return varSet[name2idx[_name]]; }

size_t ModelVarUtil::GetVarIdx(const string& _name) { return name2idx[_name]; }