// /*
//  * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
//  * reserved. SPDX-License-Identifier: Apache-2.0
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

// /*=====================================================================================

//     Filename:     UnsatTightMove.cpp

//     Description:
//         Version:  1.0

//     Author:       Peng Lin, penglincs@outlook.com

//     Organization: Shaowei Cai Group,
//                   State Key Laboratory of Computer Science,
//                   Institute of Software, Chinese Academy of Sciences,
//                   Beijing, China

// =====================================================================================*/

// #include "LocalMIP.h"

// bool LocalMIP::UnsatTightMove2()
// {
//   // Pre-allocated member variables (move these to class definition)
//   static thread_local vector<size_t> scoreIdxs;
//   static thread_local vector<size_t> tempNeighborVarIdxs;
//   static thread_local vector<Value> tempNeighborDeltas;

//   // Reuse existing vectors instead of clearing
//   vector<size_t>& neighborVarIdxs = localVarUtil.tempVarIdxs;
//   vector<Value>& neighborDeltas = localVarUtil.tempDeltas;

//   // Use resize(0) instead of clear() for better performance
//   neighborVarIdxs.resize(0);
//   neighborDeltas.resize(0);
//   scoreIdxs.resize(0);

//   // Cache frequently accessed data
//   const auto& localVars = localVarUtil.varSet;
//   const auto& modelVars = modelVarUtil->varSet;
//   const auto& localCons = localConUtil.conSet;
//   const auto& modelCons = modelConUtil->conSet;

//   // Process unsatisfied constraints
//   if (!localConUtil.unsatConIdxs.empty()) {
//     const size_t neighborSize = std::min(localConUtil.unsatConIdxs.size(), sampleUnsat);

//     // Use reference to avoid copying
//     const vector<size_t>& unsatConIdxs = localConUtil.unsatConIdxs;

//     // Pre-allocate sampling vector if needed
//     if (neighborSize < unsatConIdxs.size()) {
//       tempNeighborVarIdxs.resize(neighborSize);
//       // Use std::sample for efficient sampling (C++17)
//       std::sample(unsatConIdxs.begin(), unsatConIdxs.end(),
//                   tempNeighborVarIdxs.begin(), neighborSize, mt);
//     }

//     const vector<size_t>& neighborConIdxs =
//       (neighborSize < unsatConIdxs.size()) ? tempNeighborVarIdxs : unsatConIdxs;

//     // Process constraints with optimized inner loop
//     for (size_t neighborIdx = 0; neighborIdx < neighborSize; ++neighborIdx) {
//       const size_t conIdx = neighborConIdxs[neighborIdx];
//       const auto& localCon = localCons[conIdx];
//       const auto& modelCon = modelCons[conIdx];

//       // Optimized inner loop
//       for (size_t termIdx = 0; termIdx < modelCon.termNum; ++termIdx) {
//         const size_t varIdx = modelCon.varIdxSet[termIdx];
//         const auto& localVar = localVars[varIdx];
//         const auto& modelVar = modelVars[varIdx];

//         // Combined delta computation with early exit
//         Value delta;
//         if (!TightDelta(localCon, modelCon, termIdx, delta)) {
//           delta = (modelCon.coeffSet[termIdx] > 0)
//                   ? modelVar.lowerBound - localVar.nowValue
//                   : modelVar.upperBound - localVar.nowValue;
//         }

//         // Combined condition check with early exit
//         if ((delta < 0 && curStep < localVar.allowDecStep) ||
//             (delta > 0 && curStep < localVar.allowIncStep) ||
//             (fabs(delta) < FeasibilityTol)) {
//           continue;
//         }

//         neighborVarIdxs.push_back(varIdx);
//         neighborDeltas.push_back(delta);
//       }
//     }
//   }

//   // Process objective function if needed
//   const auto& localObj = localCons[0];
//   const auto& modelObj = modelCons[0];

//   if (isFoundFeasible && localObj.UNSAT()) {
//     for (size_t termIdx = 0; termIdx < modelObj.termNum; ++termIdx) {
//       const size_t varIdx = modelObj.varIdxSet[termIdx];
//       const auto& localVar = localVars[varIdx];
//       const auto& modelVar = modelVars[varIdx];

//       Value delta;
//       if (!TightDelta(localObj, modelObj, termIdx, delta)) {
//         delta = (modelObj.coeffSet[termIdx] > 0)
//                 ? modelVar.lowerBound - localVar.nowValue
//                 : modelVar.upperBound - localVar.nowValue;
//       }

//       if ((delta < 0 && curStep < localVar.allowDecStep) ||
//           (delta > 0 && curStep < localVar.allowIncStep) ||
//           (fabs(delta) < FeasibilityTol)) {
//         continue;
//       }

//       neighborVarIdxs.push_back(varIdx);
//       neighborDeltas.push_back(delta);
//     }
//   }

//   // Optimized scoring and selection
//   const size_t scoreSize = std::min(neighborVarIdxs.size(),
//                                    isFoundFeasible ? bmsUnsatFeas : bmsUnsatInfeas);

//   // Use partial sort instead of full shuffle for better performance
//   if (neighborVarIdxs.size() > scoreSize) {
//     // Use std::partial_sort or reservoir sampling
//     for (size_t bmsIdx = 0; bmsIdx < scoreSize; ++bmsIdx) {
//       const size_t randomIdx = (mt() % (neighborVarIdxs.size() - bmsIdx)) + bmsIdx;
//       std::swap(neighborVarIdxs[bmsIdx], neighborVarIdxs[randomIdx]);
//       std::swap(neighborDeltas[bmsIdx], neighborDeltas[randomIdx]);
//     }
//   }

//   // Optimized scoring loop
//   long bestScore = 0;
//   long bestSubscore = -std::numeric_limits<long>::max();
//   size_t bestVarIdx = -1;
//   Value bestDelta = 0;

//   for (size_t idx = 0; idx < scoreSize; ++idx) {
//     const size_t varIdx = neighborVarIdxs[idx];
//     const Value delta = neighborDeltas[idx];
//     const auto& localVar = localVars[varIdx];
//     const auto& modelVar = modelVars[varIdx];

//     // Optimized binary variable handling
//     if (modelVar.type == VarType::Binary) {
//       if (scoreTable[varIdx]) continue;
//       scoreTable[varIdx] = true;
//       scoreIdxs.push_back(varIdx);
//     }

//     const long score = TightScore(modelVar, delta);
//     if (bestScore < score || (bestScore == score && bestSubscore < subscore)) {
//       bestScore = score;
//       bestVarIdx = varIdx;
//       bestDelta = delta;
//       bestSubscore = subscore;
//     }
//   }

//   // Apply move or fallback
//   if (bestScore > 0) {
//     if (DEBUG) printf("UNSAT: %-10ld; ", bestScore);
//     ++tightStepUnsat;
//     ApplyMove(bestVarIdx, bestDelta);

//     // Reset score table efficiently
//     for (const auto idx : scoreIdxs) {
//       scoreTable[idx] = false;
//     }
//     return true;
//   } else {
//     bool resFurtherMove = false;
//     if (isFoundFeasible) resFurtherMove = SatTightMove(scoreTable, scoreIdxs);
//     if (!resFurtherMove) resFurtherMove = FlipMove(scoreTable, scoreIdxs);

//     // Reset score table efficiently
//     for (const auto idx : scoreIdxs) {
//       scoreTable[idx] = false;
//     }
//     return resFurtherMove;
//   }
// }