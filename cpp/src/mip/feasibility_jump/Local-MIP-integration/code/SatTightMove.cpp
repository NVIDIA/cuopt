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

    Filename:     SatTightMove.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "LocalMIP.h"

bool LocalMIP::SatTightMove(vector<bool>& score_table, vector<size_t>& score_idx)
{
  if (modelConUtil->conNum <= 1) return false;
  long bestScore                  = 0;
  long bestSubscore               = -std::numeric_limits<long>::max();
  size_t bestVarIdx               = -1;
  Value bestDelta                 = 0;
  vector<size_t>& neighborVarIdxs = localVarUtil.tempVarIdxs;
  vector<Value>& neighborDeltas   = localVarUtil.tempDeltas;
  neighborVarIdxs.clear();
  neighborDeltas.clear();
  auto& neighborConIdxs = localConUtil.tempSatConIdxs;
  neighborConIdxs.clear();
  localConUtil.sampleSet.clear();
  for (size_t time = 0; time < sampleSat; time++) {
    size_t conIdx = mt() % (modelConUtil->conNum - 1) + 1;
    if (localConUtil.sampleSet.find(conIdx) == localConUtil.sampleSet.end() &&
        localConUtil.conSet[conIdx].SAT() && !modelConUtil->conSet[conIdx].inferSAT) {
      localConUtil.sampleSet.insert(conIdx);
      neighborConIdxs.push_back(conIdx);
    }
  }
  size_t neighborSize = neighborConIdxs.size();
  for (size_t neighborIdx = 0; neighborIdx < neighborSize; ++neighborIdx) {
    auto& localCon = localConUtil.conSet[neighborConIdxs[neighborIdx]];
    auto& modelCon = modelConUtil->conSet[neighborConIdxs[neighborIdx]];
    for (size_t termIdx = 0; termIdx < modelCon.termNum; ++termIdx) {
      size_t varIdx  = modelCon.varIdxSet[termIdx];
      auto& localVar = localVarUtil.GetVar(varIdx);
      auto& modelVar = modelVarUtil->GetVar(varIdx);
      Value delta;
      if (!TightDelta(localCon, modelCon, termIdx, delta))
        if (modelCon.coeffSet[termIdx] > 0)
          delta = modelVar.upperBound - localVar.nowValue;
        else
          delta = modelVar.lowerBound - localVar.nowValue;
      if (delta < 0 && curStep < localVar.allowDecStep ||
          delta > 0 && curStep < localVar.allowIncStep)
        continue;
      if (fabs(delta) < FeasibilityTol) continue;
      neighborVarIdxs.push_back(varIdx);
      neighborDeltas.push_back(delta);
    }
  }
  size_t scoreSize = neighborVarIdxs.size();
  if (neighborVarIdxs.size() > bmsSat) {
    scoreSize = bmsSat;
    for (size_t bmsIdx = 0; bmsIdx < bmsSat; ++bmsIdx) {
      size_t randomIdx           = (mt() % (neighborVarIdxs.size() - bmsIdx)) + bmsIdx;
      size_t varIdx              = neighborVarIdxs[randomIdx];
      Value delta                = neighborDeltas[randomIdx];
      neighborVarIdxs[randomIdx] = neighborVarIdxs[bmsIdx];
      neighborDeltas[randomIdx]  = neighborDeltas[bmsIdx];
      neighborVarIdxs[bmsIdx]    = varIdx;
      neighborDeltas[bmsIdx]     = delta;
    }
  }
  for (size_t idx = 0; idx < scoreSize; ++idx) {
    size_t varIdx  = neighborVarIdxs[idx];
    Value delta    = neighborDeltas[idx];
    auto& localVar = localVarUtil.GetVar(varIdx);
    auto& modelVar = modelVarUtil->GetVar(varIdx);
    if (modelVar.type == VarType::Binary) {
      if (score_table[varIdx])
        continue;
      else {
        score_table[varIdx] = true;
        score_idx.push_back(varIdx);
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
    if (DEBUG) printf("SAT: %-12ld; ", bestScore);
    ++tightStepSat;
    ApplyMove(bestVarIdx, bestDelta);
    return true;
  }
  return false;
}