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

    Filename:     TightOperator.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "LocalMIP.h"

long LocalMIP::TightScore(const ModelVar& _modelVar, Value _delta)
{
  long score = 0;
  size_t conIdx;
  size_t posInCon;
  Value newLHS;
  Value newOBJ;
  bool isPreSat;
  bool isNowSat;
  bool isPreStable;
  bool isNowStable;
  bool isPreBetter;
  bool isNowBetter;
  subscore = 0;
  for (size_t termIdx = 0; termIdx < _modelVar.termNum; ++termIdx) {
    conIdx         = _modelVar.conIdxSet[termIdx];
    posInCon       = _modelVar.posInCon[termIdx];
    auto& localCon = localConUtil.conSet[conIdx];
    auto& modelCon = modelConUtil->conSet[conIdx];
    if (conIdx == 0) {
      if (isFoundFeasible) {
        newOBJ = localCon.LHS + modelCon.coeffSet[posInCon] * _delta;
        if (newOBJ < localCon.LHS)
          score += localCon.weight;
        else
          score -= localCon.weight;
        isPreBetter = localCon.LHS < localCon.RHS;
        isNowBetter = newOBJ < localCon.RHS;
        if (!isPreBetter && isNowBetter)
          subscore += localCon.weight;
        else if (isPreBetter && !isNowBetter)
          subscore -= localCon.weight;
      }
    } else {
      newLHS   = localCon.LHS + modelCon.coeffSet[posInCon] * _delta;
      isPreSat = localCon.SAT();
      isNowSat = newLHS < localCon.RHS + FeasibilityTol;
      if (!isPreSat && isNowSat)
        score += localCon.weight;
      else if (isPreSat && !isNowSat)
        score -= localCon.weight;
      else if (!isPreSat && !isNowSat)
        if (localCon.LHS > newLHS)
          score += localCon.weight >> 1;
        else
          score -= localCon.weight >> 1;
      isPreStable = localCon.LHS < localCon.RHS - FeasibilityTol;
      isNowStable = newLHS < localCon.RHS - FeasibilityTol;
      if (!isPreStable && isNowStable)
        subscore += localCon.weight;
      else if (isPreStable && !isNowStable)
        subscore -= localCon.weight;
    }
  }
  return score;
}

// return delta_x
// a * delta_x + gap <= 0
bool LocalMIP::TightDelta(LocalCon& _localCon,
                          const ModelCon& _modelCon,
                          size_t _termIdx,
                          Value& _res)
{
  Value gap      = _localCon.LHS - _localCon.RHS;
  auto varIdx    = _modelCon.varIdxSet[_termIdx];
  auto& localVar = localVarUtil.GetVar(varIdx);
  auto& modelVar = modelVarUtil->GetVar(varIdx);
  Value delta    = -(gap / _modelCon.coeffSet[_termIdx]);
  if (_modelCon.coeffSet[_termIdx] > 0) {
    if (modelVar.type == VarType::Real)
      _res = delta;
    else
      _res = floor(delta);
  } else {
    if (modelVar.type == VarType::Real)
      _res = delta;
    else
      _res = ceil(delta);
  }

  if (modelVar.InBound(localVar.nowValue + _res))
    return true;
  else
    return false;
}

void LocalMIP::UpdateWeight()
{
  for (size_t conIdx : localConUtil.unsatConIdxs) {
    auto& localCon = localConUtil.conSet[conIdx];
    ++localCon.weight;
  }
  auto& localObj = localConUtil.conSet[0];
  if (isFoundFeasible && localConUtil.unsatConIdxs.empty()) ++localObj.weight;
}

void LocalMIP::SmoothWeight()
{
  for (auto& localCon : localConUtil.conSet)
    if (localCon.SAT() && localCon.weight > 0) --localCon.weight;
}