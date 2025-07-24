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

#ifndef _paras_hpp_INCLUDED
#define _paras_hpp_INCLUDED

#include <cstring>
#include <string>
#include <unordered_map>
#include "header.h"

//        name          , type  , short-name, must-need, default, low, high, comments
#define PARAS                                                                   \
  PARA(cutoff, double, '\0', false, 7200, 0, 1e8, "Cutoff time")                \
  PARA(PrintSol, int, '\0', false, 1, 0, 1, "Print best found solution or not") \
  PARA(DEBUG, int, '\0', false, 0, 0, 1, "")

//            name,   short-name, must-need, default, comments
#define STR_PARAS STR_PARA(instance, 'i', true, "", ".mps format instance")

struct paras {
#define PARA(N, T, S, M, D, L, H, C) T N = D;
  PARAS
#undef PARA

#define STR_PARA(N, S, M, D, C) std::string N = D;
  STR_PARAS
#undef STR_PARA

  void parse_args(int argc, char* argv[]);
  void print_change();
  Value identify_opt(const char* file);
};

#define INIT_ARGS __global_paras.parse_args(argc, argv);

extern paras __global_paras;

#define OPT(N) (__global_paras.N)

#endif