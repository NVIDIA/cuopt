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

    Filename:     header.h

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#pragma once

#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
using Value                          = double;
const Value Infinity                 = 1e20;
const Value NegativeInfinity         = -Infinity;
const Value DefaultIntegerUpperBound = 1.0;
const Value DefaultRealUpperBound    = Infinity;
const Value DefaultLowerBound        = 0.0;
const Value InfiniteUpperBound       = Infinity;
const Value InfiniteLowerBound       = NegativeInfinity;
const Value FeasibilityTol           = 1e-6;
const Value OptimalTol               = 1e-4;
enum class VarType { Binary, Integer, Real, Fixed };
std::chrono::_V2::system_clock::time_point TimeNow();
double ElapsedTime(const std::chrono::_V2::system_clock::time_point& a,
                   const std::chrono::_V2::system_clock::time_point& b);
bool IsBlank(const string& a);
void PrintfError(const string& a);