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
#include "utils/paras.h"

class LocalMIP {
 private:
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
  Value bestOBJ;
  bool DEBUG;
  long subscore;

  bool VerifySolution();
  void InitState();
  void UpdateBestSolution();
  void Restart();
  bool UnsatTightMove();
  bool FlipMove(vector<bool>& _scoreTable, vector<size_t>& _scoreIdx);
  void RandomTightMove();
  void LiftMove();
  bool LiftMoveWithoutBreak();
  bool SatTightMove(vector<bool>& _scoreTable, vector<size_t>& _scoreIdx);
  void UpdateWeight();
  void SmoothWeight();
  void ApplyMove(size_t _varIdx, Value _delta);
  long TightScore(const ModelVar& _var, Value _delta);
  bool TightDelta(LocalCon& _con, const ModelCon& _modelCon, size_t _i, Value& _res);
  void InitSolution();
  bool Timeout(chrono::_V2::system_clock::time_point& _clkStart);
  void LogObj(chrono::_V2::system_clock::time_point& _clkStart);

 public:
  LocalMIP(const ModelConUtil* _modelConUtil, const ModelVarUtil* _modelVarUtil);
  ~LocalMIP();
  int LocalSearch(Value _optimalObj, chrono::_V2::system_clock::time_point _clkStart);
  void PrintResult();
  void PrintSol();
  void Allocate();
  Value GetObjValue();
};