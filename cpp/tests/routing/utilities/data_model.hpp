/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <map>
#include <set>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

// from solomon c101
template <typename i_t, typename f_t>
struct test_data_model_t {
  const std::vector<f_t> x_h{40, 45, 45, 42, 42, 42, 40, 40, 38, 38, 35, 35, 25, 22, 22, 20, 20,
                             18, 15, 15, 30, 30, 28, 28, 25, 25, 25, 23, 23, 20, 20, 10, 10, 8,
                             8,  5,  5,  2,  0,  0,  35, 35, 33, 33, 32, 30, 30, 30, 28, 28, 26,
                             25, 25, 44, 42, 42, 40, 40, 38, 38, 35, 50, 50, 50, 48, 48, 47, 47,
                             45, 45, 95, 95, 53, 92, 53, 45, 90, 88, 88, 87, 85, 85, 75, 72, 70,
                             68, 66, 65, 65, 63, 60, 60, 67, 65, 65, 62, 60, 60, 58, 55, 55};
  const std::vector<f_t> y_h{50, 68, 70, 66, 68, 65, 69, 66, 68, 70, 66, 69, 85, 75, 85, 80, 85,
                             75, 75, 80, 50, 52, 52, 55, 50, 52, 55, 52, 55, 50, 55, 35, 40, 40,
                             45, 35, 45, 40, 40, 45, 30, 32, 32, 35, 30, 30, 32, 35, 30, 35, 32,
                             30, 35, 5,  10, 15, 5,  15, 5,  15, 5,  30, 35, 40, 30, 40, 35, 40,
                             30, 35, 30, 35, 30, 30, 35, 65, 35, 30, 35, 30, 25, 35, 55, 55, 58,
                             60, 55, 55, 60, 58, 55, 60, 85, 85, 82, 80, 80, 85, 75, 80, 85};
  const std::vector<i_t> earliest_time_h{
    0,   912, 825, 65,  727, 15,  621, 170, 255, 534, 357, 448, 652, 30,   567, 384,  475,
    99,  179, 278, 10,  914, 812, 732, 65,  169, 622, 261, 546, 358, 449,  200, 31,   87,
    751, 283, 665, 383, 479, 567, 264, 166, 68,  16,  359, 541, 448, 1054, 632, 1001, 815,
    725, 912, 286, 186, 95,  385, 35,  471, 651, 562, 531, 262, 171, 632,  76,  826,  12,
    734, 916, 387, 293, 450, 478, 353, 997, 203, 574, 109, 668, 769, 47,   369, 265,  458,
    555, 173, 85,  645, 737, 20,  836, 368, 475, 285, 196, 95,  561, 30,   743, 647};

  const std::vector<i_t> latest_time_h{
    1236, 967, 870, 146, 782, 67,  702, 225,  324, 605, 410, 505, 721, 92,   620, 429,  528,
    148,  254, 345, 73,  965, 883, 777, 144,  224, 701, 316, 593, 405, 504,  237, 100,  158,
    816,  344, 716, 434, 522, 624, 321, 235,  149, 80,  412, 600, 509, 1127, 693, 1066, 880,
    786,  969, 347, 257, 158, 436, 87,  534,  740, 629, 610, 317, 218, 693,  129, 875,  77,
    777,  969, 456, 360, 505, 551, 412, 1068, 260, 643, 170, 731, 820, 124,  420, 338,  523,
    612,  238, 144, 708, 802, 84,  889, 441,  518, 336, 239, 156, 622, 84,   820, 726};
  const std::vector<i_t> service_time_h{
    0,  90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
    90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
    90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
    90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,
    90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90};
  const std::vector<i_t> demand_h{
    0,  10, 30, 10, 10, 10, 20, 20, 20, 10, 10, 10, 20, 30, 10, 40, 40, 20, 20, 10, 10,
    20, 20, 10, 10, 40, 10, 10, 20, 10, 10, 20, 30, 40, 20, 10, 10, 20, 30, 20, 10, 10,
    20, 10, 10, 10, 30, 10, 10, 10, 10, 10, 10, 20, 40, 10, 30, 40, 30, 10, 20, 10, 20,
    50, 10, 10, 10, 10, 10, 10, 30, 20, 10, 10, 50, 20, 10, 10, 20, 10, 10, 30, 20, 10,
    20, 30, 10, 20, 30, 10, 10, 10, 20, 40, 10, 30, 10, 30, 20, 10, 20};

  const std::vector<i_t> pickup_delivery_demand_h{
    0,   10,  30,  10,  10,  10,  20,  20,  20,  10,  10,  10,  20,  30,  10,  40,  40,
    20,  20,  10,  10,  20,  20,  10,  10,  40,  10,  10,  20,  10,  10,  20,  30,  40,
    20,  10,  10,  20,  30,  20,  10,  10,  20,  10,  10,  10,  30,  10,  10,  10,  10,
    -10, -30, -10, -10, -10, -20, -20, -20, -10, -10, -10, -20, -30, -10, -40, -40, -20,
    -20, -10, -10, -20, -20, -10, -10, -40, -10, -10, -20, -10, -10, -20, -30, -40, -20,
    -10, -10, -20, -30, -20, -10, -10, -20, -10, -10, -10, -30, -10, -10, -10, -10};

  const std::vector<i_t> pickup_earliest_time_h{
    0,    512,  325, 65,   327,  15,  221, 170, 255, 534, 357, 448, 252,  30,  267,  384, 375,
    99,   179,  278, 10,   414,  312, 432, 65,  169, 422, 261, 446, 358,  349, 200,  31,  87,
    451,  283,  665, 383,  379,  467, 264, 166, 68,  16,  359, 341, 448,  654, 632,  501, 815,
    1112, 1025, 265, 927,  215,  821, 370, 455, 734, 557, 648, 852, 230,  767, 584,  675, 299,
    379,  478,  210, 1114, 1012, 832, 265, 369, 822, 461, 746, 558, 649,  400, 231,  287, 951,
    483,  865,  583, 679,  767,  464, 366, 268, 216, 559, 741, 648, 1254, 832, 1201, 1015};

  const std::vector<i_t> pickup_latest_time_h{
    1500, 967,  870, 146,  782,  67,   702, 225, 324,  605, 410, 505,  721,  92,   620,  429,  528,
    148,  254,  345, 73,   965,  883,  777, 144, 224,  701, 316, 593,  405,  504,  237,  100,  158,
    816,  344,  716, 434,  522,  624,  321, 235, 149,  80,  412, 600,  509,  1127, 693,  1066, 880,
    1267, 1170, 446, 1082, 367,  1002, 525, 824, 905,  710, 805, 1021, 392,  920,  729,  828,  448,
    554,  645,  373, 1265, 1183, 1077, 444, 524, 1001, 616, 893, 705,  804,  537,  400,  458,  1116,
    644,  1016, 734, 822,  924,  621,  535, 449, 380,  712, 900, 809,  1427, 993,  1366, 1180};

  const std::vector<i_t> pickup_indices_h{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                          27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                          40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
  const std::vector<i_t> delivery_indices_h{51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                                            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                                            77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                                            90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100};

  const std::vector<i_t> capacity_h{
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
  };
  const std::vector<i_t> mixed_capacity_h{
    500, 0,   220, 200, 180, 50,  300,  1000, 20,  10,  100, 300, 150, 350, 400,  750,  220,
    200, 75,  0,   70,  105, 250, 25,   400,  500, 0,   220, 200, 180, 50,  300,  1000, 20,
    10,  100, 300, 150, 350, 400, 750,  220,  200, 75,  0,   70,  105, 250, 25,   400,  500,
    0,   220, 200, 180, 50,  300, 1000, 20,   10,  100, 300, 150, 350, 400, 750,  220,  200,
    75,  0,   70,  105, 250, 25,  400,  500,  0,   220, 200, 180, 50,  300, 1000, 20,   10,
    100, 300, 150, 350, 400, 750, 220,  200,  75,  0,   70,  105, 250, 25,  400};
  const std::vector<i_t> vehicle_earliest_h{
    0,   200, 100, 50,  0,   10,  0,   0,   600, 400, 300, 0,   750, 0,   400, 0,   220,
    0,   30,  0,   70,  105, 250, 300, 0,   0,   200, 100, 50,  0,   10,  0,   0,   600,
    400, 300, 0,   750, 0,   400, 0,   220, 0,   30,  0,   70,  105, 250, 300, 0,   0,
    200, 100, 50,  0,   10,  0,   0,   600, 400, 300, 0,   750, 0,   400, 0,   220, 0,
    30,  0,   70,  105, 250, 300, 0,   0,   200, 100, 50,  0,   10,  0,   0,   600, 400,
    300, 0,   750, 0,   400, 0,   220, 0,   30,  0,   70,  105, 250, 300, 0};
  const std::vector<i_t> vehicle_latest_h{
    1000, 500,  800,  750,  0,    50,   500,  1000, 1350, 900,  1100, 1200, 800,  500,  1200,
    1700, 1250, 200,  700,  600,  700,  205,  650,  600,  1500, 1000, 500,  800,  750,  0,
    50,   500,  1000, 1350, 900,  1100, 1200, 800,  500,  1200, 1700, 1250, 200,  700,  600,
    700,  205,  650,  600,  1500, 1000, 500,  800,  750,  0,    50,   500,  1000, 1350, 900,
    1100, 1200, 800,  500,  1200, 1700, 1250, 200,  700,  600,  700,  205,  650,  600,  1500,
    1000, 500,  800,  750,  0,    50,   500,  1000, 1350, 900,  1100, 1200, 800,  500,  1200,
    1700, 1250, 200,  700,  600,  700,  205,  650,  600,  1500};
  const std::vector<f_t> vehicle_fixed_costs_h{
    4.59142, 5.65011, 1.22437, 9.46071, 9.51269, 8.17058, 4.73513, 8.38231, 3.06139, 9.18629,
    2.06503, 1.677,   4.68299, 9.64061, 2.88371, 2.74557, 9.0185,  4.94802, 4.21281, 5.90832,
    8.46928, 2.88943, 7.91581, 4.86094, 2.90506, 6.94499, 2.48825, 4.82494, 9.93404, 7.26782,
    3.22467, 7.32528, 7.74466, 9.37263, 1.44492, 1.67506, 7.50033, 9.53029, 4.28192, 2.99341,
    8.00542, 6.7517,  2.86895, 7.34049, 9.64777, 1.53503, 4.02187, 1.55432, 7.32734, 6.0777,
    1.09139, 8.69611, 5.6681,  5.51512, 2.0296,  3.47568, 5.80489, 8.72408, 8.61813, 2.66082,
    6.72413, 7.11933, 4.9671,  5.50932, 8.28712, 6.36603, 5.96277, 4.75224, 3.6646,  7.14487,
    5.25539, 4.80709, 7.32158, 7.35015, 6.34909, 2.55793, 6.62276, 6.56842, 7.4651,  8.90117,
    9.51583, 5.36749, 1.05215, 9.36009, 7.58109, 3.75448, 9.74626, 7.72675, 3.32371, 4.31448,
    6.61499, 1.87346, 8.3551,  6.94131, 4.18942, 6.11851, 9.05324, 3.18034, 6.75229, 8.09458};
  const std::vector<uint8_t> vehicle_types_h{
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  const std::vector<i_t> break_earliest_h{
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200};
  const std::vector<i_t> break_latest_h{
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500};
  const std::vector<i_t> break_duration_h{
    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60};

  const std::vector<i_t> drop_return_h{1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
                                       0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
                                       1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
                                       1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1,
                                       1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0};

  const std::vector<i_t> skip_first_h{0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                                      0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                                      1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
                                      0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
                                      0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0};

  const std::map<i_t, std::set<i_t>> precedence_h{
    std::make_pair(
      94, std::set<i_t>{83, 3, 99, 50, 42}),  // expanded by solver into
                                              // {83,3,99,50,42,49,51,1,2,3,4,5,11,13,15,17,82}
    std::make_pair(93, std::set<i_t>{82, 3, 99, 50, 42}),
    std::make_pair(42, std::set<i_t>{82, 3, 99}),
    std::make_pair(9, std::set<i_t>{14, 5, 98}),
    std::make_pair(11, std::set<i_t>{10, 15, 20}),
    std::make_pair(50, std::set<i_t>{49, 51, 1, 2, 3, 4, 5, 11, 13, 15, 17})};

  // it is possible to generate a route per node
  const i_t n_vehicles               = 100;
  const i_t depot                    = 0;
  const i_t n_locations              = 101;
  const i_t random_demand_dist_range = 30;
};

template <typename i_t, typename f_t>
struct simple_two_routes {
  const std::vector<f_t> x_h{};
  const std::vector<f_t> y_h{};
  const std::vector<i_t> earliest_time_h{};
  const std::vector<i_t> latest_time_h{};
  const std::vector<i_t> service_time_h{0, 0, 0, 0, 0};
  const std::vector<i_t> demand_h{};
  const std::vector<i_t> pickup_delivery_demand_h{0, 10, -10, 10, -10};

  const std::vector<i_t> pickup_earliest_time_h{0, 0, 30, 50, 60};

  const std::vector<i_t> pickup_latest_time_h{100, 20, 40, 60, 70};

  const std::vector<i_t> pickup_indices_h{1, 3};
  const std::vector<i_t> delivery_indices_h{2, 4};

  const std::vector<i_t> capacity_h{10, 10};
  const std::vector<uint8_t> vehicle_types_h{0, 0};
  const std::vector<i_t> mixed_capacity_h{};
  const std::vector<i_t> vehicle_earliest_h{};
  const std::vector<i_t> vehicle_latest_h{};

  const std::vector<i_t> break_earliest_h{};
  const std::vector<i_t> break_latest_h{};
  const std::vector<i_t> break_duration_h{};
  const std::vector<i_t> drop_return_h{};
  const std::vector<i_t> skip_first_h{};

  const std::map<i_t, std::set<i_t>> precedence_h{};

  // clang-format off
  const std::vector<f_t> cost_matrix_h{0,    10,   10,   20,   10,
                                       10,   0,    25,   20,   1000,
                                       10,   25,   0,    10,   1000,
                                       20,   20,   10,   0,    10,
                                       10,   1000, 1000, 10,   0};
  // clang-format on
  const i_t n_vehicles  = 2;
  const i_t depot       = 0;
  const i_t n_locations = 5;

  std::vector<i_t> request{1, 2};
  std::vector<i_t> p_scores{0, 2, 0, 3, 0};

  std::vector<i_t> expected_found_sol{2, 0, 0, 0};
  std::vector<i_t> expected_route{0, 1, 2, 3, 4, 0};
};

template <typename i_t, typename f_t>
struct simple_three_routes {
  const std::vector<f_t> x_h{};
  const std::vector<f_t> y_h{};
  const std::vector<i_t> earliest_time_h{};
  const std::vector<i_t> latest_time_h{};
  const std::vector<i_t> service_time_h{0, 0, 0, 0, 0, 0, 0};
  const std::vector<i_t> demand_h{};
  const std::vector<i_t> pickup_delivery_demand_h{0, 10, -10, 10, -10, 10, -10};

  const std::vector<i_t> pickup_earliest_time_h{0, 0, 10, 0, 30, 20, 50};

  const std::vector<i_t> pickup_latest_time_h{100, 30, 70, 10, 50, 30, 60};

  const std::vector<i_t> pickup_indices_h{1, 3, 5};
  const std::vector<i_t> delivery_indices_h{2, 4, 6};

  const std::vector<i_t> capacity_h{30, 30, 30};
  const std::vector<uint8_t> vehicle_types_h{0, 0, 0};
  const std::vector<i_t> mixed_capacity_h{};
  const std::vector<i_t> vehicle_earliest_h{};
  const std::vector<i_t> vehicle_latest_h{};

  const std::vector<i_t> break_earliest_h{};
  const std::vector<i_t> break_latest_h{};
  const std::vector<i_t> break_duration_h{};
  const std::vector<i_t> drop_return_h{};
  const std::vector<i_t> skip_first_h{};

  const std::map<i_t, std::set<i_t>> precedence_h{};

  // clang-format off
  const std::vector<f_t> cost_matrix_h{0,    10,   10,   10,   10,   10,   10,
                                       10,   0,    10,   10,   10,   10,   1000,
                                       10,   10,   0,    1000, 10, 10,   10,
                                       10,   10,   1000, 0,    10,   10,   1000,
                                       10,   10,   10,   10,   0,    10,   10,
                                       10,   10,   10,   10,   10,   0,    10,
                                       10,   1000, 10,   1000, 10,   10,   0};
  // clang-format on

  const i_t n_vehicles  = 3;
  const i_t depot       = 0;
  const i_t n_locations = 7;
  std::vector<i_t> request{};
  std::vector<i_t> p_scores{};
  std::vector<i_t> expected_found_sol{};
  std::vector<i_t> expected_route{0, 3, 1, 5, 4, 2, 6, 0};
};

template <typename i_t, typename f_t>
struct scross_three_routes {
  const std::vector<f_t> x_h{};
  const std::vector<f_t> y_h{};
  const std::vector<i_t> earliest_time_h{};
  const std::vector<i_t> latest_time_h{};
  const std::vector<i_t> service_time_h{0, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<i_t> demand_h{};
  const std::vector<i_t> pickup_delivery_demand_h{0, 10, -10, 10, -10, 10, -10, 10, -10};

  const std::vector<i_t> pickup_earliest_time_h{0, 0, 0, 0, 0, 0, 0, 0, 0};

  const std::vector<i_t> pickup_latest_time_h{200, 200, 200, 200, 200, 200, 200, 200, 200};

  const std::vector<i_t> pickup_indices_h{1, 3, 5, 7};
  const std::vector<i_t> delivery_indices_h{2, 4, 6, 8};

  const std::vector<i_t> capacity_h{30, 30, 30};
  const std::vector<uint8_t> vehicle_types_h{0, 0, 0};
  const std::vector<i_t> mixed_capacity_h{};
  const std::vector<i_t> vehicle_earliest_h{};
  const std::vector<i_t> vehicle_latest_h{};

  const std::vector<i_t> break_earliest_h{};
  const std::vector<i_t> break_latest_h{};
  const std::vector<i_t> break_duration_h{};
  const std::vector<i_t> drop_return_h{};
  const std::vector<i_t> skip_first_h{};

  const std::map<i_t, std::set<i_t>> precedence_h{};

  // clang-format off                         d     1     2     3     4     5     6     7     8
  const std::vector<f_t> cost_matrix_h{
    /*d*/ 0,   1,    1,    1,    1,    1,    0.5,  1,    1,
    /*1*/ 1,   0,    1,    1000, 1000, 1000, 1000, 1000, 1000,
    /*2*/ 1,   1,    0,    1000, 1000, 1000, 1000, 100,  1000,
    /*3*/ 1,   1000, 1000, 0,    1,    1000, 1000, 1000, 1000,
    /*4*/ 1,   1000, 1000, 1,    0,    1000, 1000, 1000, 1000,
    /*5*/ 1,   1000, 1000, 1000, 1000, 0,    1,    1000, 1000,
    /*6*/ 0.5, 1000, 1000, 1000, 1000, 1,    0,    1,    1000,
    /*7*/ 1,   1000, 100,  1000, 1000, 1000, 1,    0,    1,
    /*8*/ 1,   1000, 1000, 1000, 1000, 1000, 1000, 1,    0};
  // clang-format on

  const i_t n_vehicles  = 3;
  const i_t depot       = 0;
  const i_t n_locations = 9;
  std::vector<i_t> request{};
  std::vector<i_t> p_scores{};

  std::vector<i_t> expected_found_sol{};
  // clang-format off
  std::vector<i_t> expected_route{0, 5, 6, 7, 8, 0,
                                  0, 3, 4, 0,
                                  0, 1, 2, 0};
  // clang-format on
};

static test_data_model_t<int, float> input_;
static test_data_model_t<int, float> input_double_;
static simple_two_routes<int, float> simple_two_routes_;
static simple_three_routes<int, float> simple_three_routes_;
static scross_three_routes<int, float> scross_three_routes_;

}  // namespace test
}  // namespace routing
}  // namespace cuopt
