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

    Filename:     LocalMIP.h

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#pragma once
#include "LocalCon.h"
#include "LocalVar.h"
#include "ModelCon.h"
#include "ModelVar.h"
#include "header.h"

#include <atomic>
#include <functional>

class LocalMIP {
 public:
  const ModelConUtil* modelConUtil;
  const ModelVarUtil* modelVarUtil;
  LocalVarUtil localVarUtil;
  LocalConUtil localConUtil;
  size_t curStep;
  std::mt19937 mt;
  size_t smoothProbability;
  size_t tabuBase;
  size_t tabuVariation;
  bool isFoundFeasible;
  size_t liftStep;
  size_t breakStep;
  size_t tightStepUnsat;
  size_t tightStepSat;
  size_t flipStep;
  size_t randomStep;
  size_t weightUpperBound;
  size_t objWeightUpperBound;
  size_t lastImproveStep;
  size_t restartTimes;
  bool isBin;
  bool isKeepFeas;
  size_t sampleUnsat;
  size_t bmsUnsatInfeas;
  size_t bmsUnsatFeas;
  size_t sampleSat;
  size_t bmsSat;
  size_t bmsFlip;
  size_t bmsRandom;
  size_t restartStep;
  int last_var_idx{-1};
  Value bestOBJ;
  bool DEBUG;
  // long subscore;
  std::atomic<bool> halted;
  size_t minima{0};
  size_t max_iters{std::numeric_limits<size_t>::max()};
  std::vector<int> relvar_offsets;
  bool VerifySolution();
  void InitState();
  void UpdateBestSolution();
  void Restart();
  bool UnsatTightMove();
  bool UnsatTightMove2();
  bool FlipMove(vector<bool>& _scoreTable, vector<size_t>& _scoreIdx);
  void RandomTightMove();
  void LiftMove();
  bool LiftMoveWithoutBreak();
  bool SatTightMove(vector<bool>& _scoreTable, vector<size_t>& _scoreIdx);
  void UpdateWeight();
  void SmoothWeight();
  void ApplyMove(size_t _varIdx, Value _delta);
  void InitSolution();
  bool Timeout(chrono::_V2::system_clock::time_point& _clkStart);
  void LogObj(chrono::_V2::system_clock::time_point& _clkStart);

  std::pair<long, long> TightScore(const ModelVar& _modelVar, Value _delta)
  {
    long score = 0;
    Value newLHS;
    Value newOBJ;
    bool isPreSat;
    bool isNowSat;
    bool isPreStable;
    bool isNowStable;
    bool isPreBetter;
    bool isNowBetter;
    long subscore = 0;
    // hopefully the compiler performs autovectorization as well, but hopes not high
    // (not sure if SSE/AVX2 has gather/scatter regardless)
    // i miss my blockreduces. I have no hope of GCC figuring out the reduction.
#pragma omp simd reduction(+ : score, subscore)
    for (size_t termIdx = 0; termIdx < _modelVar.termNum; ++termIdx) {
      // conIdx         = _modelVar.conIdxSet[termIdx];
      Value coeff = _modelVar.coeffs[termIdx];
      // auto& localCon = localConUtil.conSet[conIdx];
      auto& localCon = *_modelVar.conRefSet[termIdx];
      if (localCon.isObj) {
        if (isFoundFeasible) {
          newOBJ = localCon.LHS + coeff * _delta;
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
        newLHS   = localCon.LHS + coeff * _delta;
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
    return {score, subscore};
  }

  // return delta_x
  // a * delta_x + gap <= 0
  bool TightDelta(LocalCon& _localCon, const ModelCon& _modelCon, size_t _termIdx, Value& _res)
  {
    Value gap      = _localCon.LHS - _localCon.RHS;
    auto varIdx    = _modelCon.varIdxSet[_termIdx];
    auto& localVar = localVarUtil.GetVar(varIdx);
    auto& modelVar = modelVarUtil->GetVar(varIdx);
    Value delta    = -(gap * _modelCon.invCoeffs[_termIdx]);
    if (_modelCon.invCoeffs[_termIdx] > 0) {
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

 public:
  LocalMIP(const ModelConUtil* _modelConUtil, const ModelVarUtil* _modelVarUtil);
  ~LocalMIP();
  int LocalSearch(Value _optimalObj, chrono::_V2::system_clock::time_point _clkStart);
  void PrintResult(chrono::_V2::system_clock::time_point _clkStart);
  void PrintSol();
  void Allocate();
  void RandomizeParams();
  Value GetObjValue();

  std::string prefix = "";
  std::function<void()> optimum_callback;
  std::function<void()> diversity_callback;
  bool found_better = false;
};