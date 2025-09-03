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

    Filename:     UnsatTightMove.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "LocalMIP.h"

bool LocalMIP::UnsatTightMove()
{
  vector<bool>& scoreTable = localVarUtil.scoreTable;
  vector<size_t> scoreIdxs;
  long bestScore                  = 0;
  long bestSubscore               = -std::numeric_limits<long>::max();
  size_t bestVarIdx               = -1;
  Value bestDelta                 = 0;
  vector<size_t>& neighborVarIdxs = localVarUtil.tempVarIdxs;
  vector<Value>& neighborDeltas   = localVarUtil.tempDeltas;
  neighborVarIdxs.clear();
  neighborDeltas.clear();
  if (localConUtil.unsatConIdxs.size() > 0) {
    size_t neighborSize             = localConUtil.unsatConIdxs.size();
    vector<size_t>* neighborConIdxs = &localConUtil.unsatConIdxs;
    // no samplin
    if (sampleUnsat < neighborSize) {
      neighborSize    = sampleUnsat;
      neighborConIdxs = &localConUtil.tempUnsatConIdxs;
      neighborConIdxs->clear();
      neighborConIdxs->assign(localConUtil.unsatConIdxs.begin(), localConUtil.unsatConIdxs.end());
      for (size_t sampleIdx = 0; sampleIdx < sampleUnsat; ++sampleIdx) {
        size_t randomIdx                           = mt() % (neighborConIdxs->size() - sampleIdx);
        size_t temp                                = neighborConIdxs->at(sampleIdx);
        neighborConIdxs->at(sampleIdx)             = neighborConIdxs->at(randomIdx + sampleIdx);
        neighborConIdxs->at(randomIdx + sampleIdx) = temp;
      }
    }
    // printf("neighborSize: %d\n", (int)neighborSize);
    const int average_row_size = 4;
    neighborVarIdxs.reserve(neighborSize * average_row_size);
    neighborDeltas.reserve(neighborSize * average_row_size);
    for (size_t neighborIdx = 0; neighborIdx < neighborSize; ++neighborIdx) {
      auto& localCon = localConUtil.conSet[(*neighborConIdxs)[neighborIdx]];
      auto& modelCon = modelConUtil->conSet[(*neighborConIdxs)[neighborIdx]];
      for (size_t termIdx = 0; termIdx < modelCon.termNum; ++termIdx) {
        size_t varIdx  = modelCon.varIdxSet[termIdx];
        auto& localVar = localVarUtil.GetVar(varIdx);
        auto& modelVar = modelVarUtil->GetVar(varIdx);
        Value delta;
        if (!TightDelta(localCon, modelCon, termIdx, delta))
          if (modelCon.coeffSet[termIdx] > 0)
            delta = modelVar.lowerBound - localVar.nowValue;
          else
            delta = modelVar.upperBound - localVar.nowValue;
        if (delta < 0 && curStep < localVar.allowDecStep ||
            delta > 0 && curStep < localVar.allowIncStep)
          continue;
        if (fabs(delta) < FeasibilityTol) continue;
        neighborVarIdxs.push_back(varIdx);
        neighborDeltas.push_back(delta);
      }
    }
  }
  auto& localObj = localConUtil.conSet[0];
  auto& modelObj = modelConUtil->conSet[0];
  // BM moves
  if (isFoundFeasible && localObj.UNSAT())
    for (size_t termIdx = 0; termIdx < modelObj.termNum; ++termIdx) {
      size_t varIdx  = modelObj.varIdxSet[termIdx];
      auto& localVar = localVarUtil.GetVar(varIdx);
      auto& modelVar = modelVarUtil->GetVar(varIdx);
      Value delta;
      if (!TightDelta(localObj, modelObj, termIdx, delta))
        if (modelObj.coeffSet[termIdx] > 0)
          delta = modelVar.lowerBound - localVar.nowValue;
        else
          delta = modelVar.upperBound - localVar.nowValue;
      if (delta < 0 && curStep < localVar.allowDecStep ||
          delta > 0 && curStep < localVar.allowIncStep)
        continue;
      if (fabs(delta) < FeasibilityTol) continue;
      neighborVarIdxs.push_back(varIdx);
      neighborDeltas.push_back(delta);
    }
  size_t scoreSize = neighborVarIdxs.size();
  if (!isFoundFeasible && scoreSize > bmsUnsatInfeas ||
      isFoundFeasible && scoreSize > bmsUnsatFeas) {
    if (!isFoundFeasible)
      scoreSize = bmsUnsatInfeas;
    else
      scoreSize = bmsUnsatFeas;
    for (size_t bmsIdx = 0; bmsIdx < scoreSize; ++bmsIdx) {
      size_t randomIdx           = (mt() % (neighborVarIdxs.size() - bmsIdx)) + bmsIdx;
      size_t varIdx              = neighborVarIdxs[randomIdx];
      Value delta                = neighborDeltas[randomIdx];
      neighborVarIdxs[randomIdx] = neighborVarIdxs[bmsIdx];
      neighborDeltas[randomIdx]  = neighborDeltas[bmsIdx];
      neighborVarIdxs[bmsIdx]    = varIdx;
      neighborDeltas[bmsIdx]     = delta;
    }
  }
  int relvar_count = scoreSize;
  if (last_var_idx != -1) { relvar_count = relvar_offsets[last_var_idx]; }
  // printf("scoreSize: %d vs %d\n", (int)scoreSize, relvar_count);
  for (size_t idx = 0; idx < scoreSize; ++idx) {
    size_t varIdx  = neighborVarIdxs[idx];
    Value delta    = neighborDeltas[idx];
    auto& localVar = localVarUtil.GetVar(varIdx);
    auto& modelVar = modelVarUtil->GetVar(varIdx);
    if (modelVar.type == VarType::Binary) {
      if (scoreTable[varIdx])
        continue;
      else {
        scoreTable[varIdx] = true;
        scoreIdxs.push_back(varIdx);
      }
    }
    auto [score, subscore] = TightScore(modelVar, delta);
    if (bestScore < score || bestScore == score && bestSubscore < subscore) {
      bestScore    = score;
      bestVarIdx   = varIdx;
      bestDelta    = delta;
      bestSubscore = subscore;
    }
  }
  if (bestScore > 0) {
    if (DEBUG)
      printf("move_UNSAT: %-10ld; score %d; subscore %d\n",
             (int)bestVarIdx,
             bestScore,
             (int)bestSubscore);
    ++tightStepUnsat;
    ApplyMove(bestVarIdx, bestDelta);
    for (auto idx : scoreIdxs)
      scoreTable[idx] = false;
    return true;
  } else {
    bool resFurtherMove = false;
    // if (isFoundFeasible) resFurtherMove = SatTightMove(scoreTable, scoreIdxs);
    // if (!resFurtherMove) resFurtherMove = FlipMove(scoreTable, scoreIdxs);
    for (auto idx : scoreIdxs)
      scoreTable[idx] = false;
    return resFurtherMove;
  }
}
