/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

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
