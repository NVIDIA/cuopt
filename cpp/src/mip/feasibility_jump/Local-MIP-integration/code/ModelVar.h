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

    Filename:     ModelVar.h

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#pragma once
#include "header.h"

class ModelVar {
 public:
  string name;
  size_t idx;
  Value upperBound;
  Value lowerBound;
  vector<size_t> conIdxSet;
  vector<size_t> posInCon;
  size_t termNum;
  VarType type;

  ModelVar(const string& _name, size_t _idx, bool _integrality);
  ~ModelVar();
  bool InBound(Value _value) const
  {
    return lowerBound - FeasibilityTol < _value && _value < upperBound + FeasibilityTol;
  }
  void SetType(VarType _varType);
  void SetUpperBound(Value _upperBound);
  void SetLowerBound(Value _lowerBound);
  bool IsFixed();
  bool IsBinary();
};

class ModelVarUtil {
 public:
  unordered_map<string, size_t> name2idx;
  vector<ModelVar> varSet;
  vector<size_t> varIdx2ObjIdx;
  bool isBin;
  size_t varNum;
  size_t integerNum;
  size_t binaryNum;
  size_t fixedNum;
  size_t realNum;
  Value objBias;

  ModelVarUtil();
  ~ModelVarUtil();
  size_t MakeVar(const string& _name, const bool _integrality);
  const ModelVar& GetVar(const size_t _idx) const;
  ModelVar& GetVar(const size_t _idx) { return varSet[_idx]; }
  ModelVar& GetVar(const string& _name);
  size_t GetVarIdx(const string& _name);
};