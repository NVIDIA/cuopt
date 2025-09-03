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

    Filename:     LiftMove.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "LocalMIP.h"

bool LocalMIP::LiftMoveWithoutBreak()
{
  auto& localObj            = localConUtil.conSet[0];
  auto& modelObj            = modelConUtil->conSet[0];
  vector<Value>& lowerDelta = localVarUtil.lowerDeltaInLiftMove;
  vector<Value>& upperDelta = localVarUtil.upperDeltaInLifiMove;
  if (!isKeepFeas) {
    for (size_t termIdx = 0; termIdx < modelObj.termNum; ++termIdx) {
      size_t varIdx       = modelObj.varIdxSet[termIdx];
      auto& localVar      = localVarUtil.GetVar(varIdx);
      auto& modelVar      = modelVarUtil->GetVar(varIdx);
      lowerDelta[termIdx] = modelVar.lowerBound - localVar.nowValue;
      upperDelta[termIdx] = modelVar.upperBound - localVar.nowValue;
      for (size_t j = 0; j < modelVar.termNum; ++j) {
        size_t conIdx   = modelVar.conIdxSet[j];
        auto& localCon  = localConUtil.conSet[conIdx];
        auto& modelCon  = modelConUtil->conSet[conIdx];
        size_t posInCon = modelVar.posInCon[j];
        Value coeff     = modelCon.coeffSet[posInCon];
        if (conIdx == 0) continue;
        Value delta;
        Value gap = localCon.LHS - localCon.RHS;
        if (fabs(gap) < FeasibilityTol) {
          if (coeff > 0)
            upperDelta[termIdx] = 0;
          else
            lowerDelta[termIdx] = 0;
        } else if (!TightDelta(localCon, modelCon, posInCon, delta))
          continue;
        else {
          if (coeff > 0) {
            if (delta < upperDelta[termIdx]) upperDelta[termIdx] = delta;
          } else if (coeff < 0) {
            if (delta > lowerDelta[termIdx]) lowerDelta[termIdx] = delta;
          }
        }
        if (lowerDelta[termIdx] >= upperDelta[termIdx]) break;
      }
    }
  }
  Value bestObjDelta = 0;
  size_t bestVarIdx  = -1;
  Value bestVarDelta = 0;
  Value varDelta;
  Value objDelta;
  Value objDelta_l;
  Value objDelta_u;
  vector<size_t> betterIdx;
  vector<Value> betterDelta;
  size_t bestLastMoveStep = std::numeric_limits<size_t>::max();
  for (size_t termIdx = 0; termIdx < modelObj.termNum; ++termIdx) {
    size_t varIdx = modelObj.varIdxSet[termIdx];
    Value coeff   = modelObj.coeffSet[termIdx];
    Value l_d     = lowerDelta[termIdx];
    Value u_d     = upperDelta[termIdx];
    if (l_d == u_d) continue;
    auto& localVar = localVarUtil.GetVar(varIdx);
    auto& modelVar = modelVarUtil->GetVar(varIdx);
    if (!modelVar.InBound(localVar.nowValue + l_d)) lowerDelta[termIdx] = l_d = 0;
    if (!modelVar.InBound(localVar.nowValue + u_d)) upperDelta[termIdx] = u_d = 0;
    // assert(modelVar.InBound(localVar.nowValue + l_d) &&
    //        modelVar.InBound(localVar.nowValue + u_d));
    if (coeff > 0) {
      objDelta = coeff * l_d;
      varDelta = l_d;
    } else {
      objDelta = coeff * u_d;
      varDelta = u_d;
    }
    size_t lastMoveStep = varDelta < 0 ? localVar.lastDecStep : localVar.lastIncStep;
    if (objDelta < bestObjDelta ||
        objDelta < bestObjDelta + OptimalTol && lastMoveStep < bestLastMoveStep) {
      bestObjDelta     = objDelta;
      bestVarIdx       = varIdx;
      bestVarDelta     = varDelta;
      bestLastMoveStep = lastMoveStep;
    }
    // if (objDelta < 0)
    // {
    //   betterIdx.push_back(varIdx);
    //   betterDelta.push_back(varDelta);
    // }
  }

  if (bestVarIdx != -1 && bestVarDelta != 0) {
    ++liftStep;
    if (DEBUG) printf("move_LIFT: %-10ld; objdelta %g\n", (int)bestVarIdx, bestObjDelta);
    ApplyMove(bestVarIdx, bestVarDelta);
    isKeepFeas                         = true;
    unordered_set<size_t>& affectedVar = localVarUtil.affectedVar;
    affectedVar.clear();
    auto& bestLocalVar = localVarUtil.GetVar(bestVarIdx);
    auto& bestModelVar = modelVarUtil->GetVar(bestVarIdx);
    for (auto conIdx : bestModelVar.conIdxSet) {
      if (conIdx == 0) continue;
      auto& localCon = localConUtil.GetCon(conIdx);
      auto& modelCon = modelConUtil->GetCon(conIdx);
      for (auto varIdx : modelCon.varIdxSet)
        affectedVar.insert(varIdx);
    }
    for (auto varIdx : affectedVar) {
      size_t idxInObj = modelVarUtil->varIdx2ObjIdx[varIdx];
      if (idxInObj == -1) continue;
      auto& localVar       = localVarUtil.GetVar(varIdx);
      auto& modelVar       = modelVarUtil->GetVar(varIdx);
      lowerDelta[idxInObj] = modelVar.lowerBound - localVar.nowValue;
      upperDelta[idxInObj] = modelVar.upperBound - localVar.nowValue;
      for (size_t termIdx = 0; termIdx < modelVar.termNum; ++termIdx) {
        size_t conIdx   = modelVar.conIdxSet[termIdx];
        auto& localCon  = localConUtil.conSet[conIdx];
        auto& modelCon  = modelConUtil->conSet[conIdx];
        size_t posInCon = modelVar.posInCon[termIdx];
        Value coeff     = modelCon.coeffSet[posInCon];
        if (conIdx == 0) continue;
        Value delta;
        Value gap = localCon.LHS - localCon.RHS;
        if (fabs(gap) < FeasibilityTol) {
          if (coeff > 0)
            upperDelta[idxInObj] = 0;
          else
            lowerDelta[idxInObj] = 0;
        } else if (!TightDelta(localCon, modelCon, posInCon, delta))
          continue;
        else {
          if (coeff > 0) {
            if (delta < upperDelta[idxInObj]) upperDelta[idxInObj] = delta;
          } else if (coeff < 0) {
            if (delta > lowerDelta[idxInObj]) lowerDelta[idxInObj] = delta;
          }
        }
        if (lowerDelta[idxInObj] >= upperDelta[idxInObj]) break;
      }
    }
    return true;
  } else
    isKeepFeas = false;
  return false;
}
