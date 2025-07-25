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

    Filename:     utils.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "header.h"

std::chrono::_V2::system_clock::time_point TimeNow()
{
  return chrono::high_resolution_clock::now();
}

double ElapsedTime(const std::chrono::_V2::system_clock::time_point& a,
                   const std::chrono::_V2::system_clock::time_point& b)
{
  return chrono::duration_cast<chrono::milliseconds>(a - b).count() / 1000.0;
}

bool IsBlank(const string& a)
{
  for (auto x : a)
    if (x != ' ' && x != '\n' && x != '\r') return false;
  return true;
}

void PrintfError(const string& a)
{
  printf("c error line: %s\n", a.c_str());
  exit(-1);
}
