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

    Filename:     LocalCon.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "LocalCon.h"

LocalCon::LocalCon() : weight(1), RHS(0), LHS(0) {}

LocalCon::~LocalCon() {}

bool LocalCon::SAT() { return LHS < RHS + FeasibilityTol; }

bool LocalCon::UNSAT() { return LHS >= RHS + FeasibilityTol; }

LocalConUtil::LocalConUtil() {}

void LocalConUtil::Allocate(const size_t _conNum)
{
  tempSatConIdxs.clear();
  tempUnsatConIdxs.clear();
  conSet.clear();
  unsatConIdxs.clear();

  unsatConIdxs.reserve(_conNum);
  tempSatConIdxs.reserve(_conNum);
  tempUnsatConIdxs.reserve(_conNum);
  conSet.resize(_conNum);
}

LocalConUtil::~LocalConUtil()
{
  tempSatConIdxs.clear();
  tempUnsatConIdxs.clear();
  conSet.clear();
  unsatConIdxs.clear();
}

LocalCon& LocalConUtil::GetCon(const size_t _idx) { return conSet[_idx]; }

void LocalConUtil::insertUnsat(const size_t _conIdx)
{
  conSet[_conIdx].posInUnsatConIdxs = unsatConIdxs.size();
  unsatConIdxs.push_back(_conIdx);
}

void LocalConUtil::RemoveUnsat(const size_t _conIdx)
{
  assert(unsatConIdxs.size() > 0);
  if (unsatConIdxs.size() == 1) {
    unsatConIdxs.pop_back();
    return;
  }
  size_t pos        = conSet[_conIdx].posInUnsatConIdxs;
  unsatConIdxs[pos] = *unsatConIdxs.rbegin();
  unsatConIdxs.pop_back();
  conSet[unsatConIdxs[pos]].posInUnsatConIdxs = pos;
}