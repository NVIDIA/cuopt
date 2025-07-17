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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

class solution_reader_t {
 public:
  std::unordered_map<std::string, double> data_map;

  bool read_from_sol(const std::string& filepath)
  {
    std::ifstream file(filepath);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file: " << filepath << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string var_name;
      std::string value_str;
      ss >> var_name >> value_str;
      if (var_name == "=obj=") continue;

      try {
        double value       = std::stod(value_str);
        data_map[var_name] = value;
      } catch (const std::exception& e) {
        std::cerr << "Error converting value for " << var_name << std::endl;
        continue;
      }
    }

    return true;
  }

  double getValue(const std::string& key, double default_value = 0.0) const
  {
    auto it = data_map.find(key);
    return (it != data_map.end()) ? it->second : default_value;
  }

  void printAll() const
  {
    for (const auto& [key, value] : data_map) {
      std::cout << key << ": " << value << std::endl;
    }
  }
};
