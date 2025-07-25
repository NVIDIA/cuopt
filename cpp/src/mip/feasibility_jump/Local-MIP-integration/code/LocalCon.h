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

    Filename:     LocalCon.h

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

class LocalCon {
 public:
  size_t weight;
  size_t posInUnsatConIdxs;
  Value RHS;
  Value LHS;

  LocalCon();
  ~LocalCon();
  bool SAT();
  bool UNSAT();
};

class LocalConUtil {
 public:
  vector<LocalCon> conSet;
  vector<size_t> unsatConIdxs;
  vector<size_t> tempUnsatConIdxs;
  vector<size_t> tempSatConIdxs;
  unordered_set<size_t> sampleSet;

  LocalConUtil();
  ~LocalConUtil();
  void Allocate(const size_t _conNum);
  LocalCon& GetCon(const size_t _idx);
  void insertUnsat(const size_t _conIdx);
  void RemoveUnsat(const size_t _conIdx);
};
