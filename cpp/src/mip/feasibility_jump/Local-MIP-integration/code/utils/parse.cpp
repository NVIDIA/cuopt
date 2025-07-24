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

    Filename:     parse.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include <cstring>
#include <fstream>
#include "header.h"
#include "paras.h"

Value paras::identify_opt(const char* file)
{
  char name[strlen(file) + 1], p = -1, l = strlen(file);
  for (int i = l - 1; i >= 0; i--)
    if (file[i] == '/') {
      p = i;
      break;
    }
  strncpy(name, file + p + 1, l - p - 1);
  name[l - p - 1] = '\0';
  printf("c File name (with path): %s\n", file);
  printf("c File name: %s\n", name);
  return NegativeInfinity;
}
