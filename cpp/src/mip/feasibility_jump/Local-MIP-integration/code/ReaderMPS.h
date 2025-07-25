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

    Filename:     ReaderMPS.h

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#pragma once
#include "ModelCon.h"
#include "ModelVar.h"
#include "header.h"

class ReaderMPS {
 public:
  ModelConUtil* modelConUtil;
  ModelVarUtil* modelVarUtil;
  istringstream iss;
  string readLine;
  bool integralityMarker;
  bool TightenBound();
  void TightenBoundVar(ModelCon& _modelCon);
  bool TightBoundGlobally();
  bool SetVarType();
  void SetVarIdx2ObjIdx();
  vector<size_t> fixedIdxs;
  size_t deleteConNum;
  size_t deleteVarNum;
  size_t inferVarNum;
  inline void IssSetup();
  void PushCoeffVarIdx(const size_t _conIdx, Value _coeff, const string& _varName);

 public:
  ReaderMPS(ModelConUtil* _modelConUtil, ModelVarUtil* _modelVarUtil);
  ~ReaderMPS();
  void Read(const char* _fileName);
};