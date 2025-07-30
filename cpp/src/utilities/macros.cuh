/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

// TODO have levels of debug and test assertions
// the impact can be
// 1) light
// 2) medium
// 3) heavy
#ifdef ASSERT_MODE
#include <cassert>
#define cuopt_assert(val, msg) assert(val&& msg)
#define cuopt_func_call(func)  func;
#else
#define cuopt_assert(val, msg)
#define cuopt_func_call(func) ;
#endif

#ifdef BENCHMARK
#define benchmark_call(func) func;
#else
#define benchmark_call(func) ;
#endif
