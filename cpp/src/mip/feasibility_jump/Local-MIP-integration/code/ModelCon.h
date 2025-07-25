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

    Filename:     ModelCon.h

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

class ModelCon {
 public:
  string name;
  size_t idx;
  bool isEqual;
  bool isLarge;
  vector<Value> coeffSet;
  vector<size_t> varIdxSet;
  vector<size_t> posInVar;
  Value RHS;
  bool inferSAT;
  size_t termNum;

  ModelCon(const string& _name, const size_t _idx);
  ~ModelCon();
};

class ModelConUtil {
 public:
  unordered_map<string, size_t> name2idx;
  vector<ModelCon> conSet;
  string objName;
  size_t conNum;
  int MIN = 1;

  ModelConUtil();
  ~ModelConUtil();
  size_t MakeCon(const string& _name);
  size_t GetConIdx(const string& _name);
  const ModelCon& GetCon(const size_t _idx) const;
  ModelCon& GetCon(const size_t _idx);
  ModelCon& GetCon(const string& _name);
};