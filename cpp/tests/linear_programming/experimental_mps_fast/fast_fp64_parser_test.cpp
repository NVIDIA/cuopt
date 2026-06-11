// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fast_fp64_parser.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

uint64_t bits(double value) { return std::bit_cast<uint64_t>(value); }

double reference_strtod(std::string_view token)
{
  std::string normalized(token);
  for (char& c : normalized) {
    if (c == 'd' || c == 'D') { c = 'e'; }
  }
  char* end    = nullptr;
  errno        = 0;
  double value = std::strtod(normalized.c_str(), &end);
  ASSERT_EQ(end, normalized.c_str() + normalized.size());
  return value;
}

double parse_token(std::string_view token)
{
  const char* p = token.data();
  return mps_fast::fp64::parse_fp64_advance(p, token.data() + token.size());
}

double parse_padded_token(std::string_view token)
{
  std::string padded(token);
  padded.append(40, ' ');
  const char* p = padded.data();
  double value  = mps_fast::fp64::parse_fp64_advance(p, padded.data() + padded.size());
  ASSERT_EQ(p, padded.data() + token.size());
  return value;
}

void check_bitwise_strtod(std::string_view token)
{
  const double ref        = reference_strtod(token);
  const uint64_t ref_bits = bits(ref);
  EXPECT_EQ(ref_bits, bits(parse_token(token))) << "token parse mismatch for '" << token << "'";
  EXPECT_EQ(ref_bits, bits(parse_padded_token(token)))
    << "padded parse mismatch for '" << token << "'";
}

std::string random_token(std::mt19937_64& rng)
{
  std::uniform_int_distribution<int> sign_dist(0, 4);
  std::uniform_int_distribution<int> digit_dist(0, 9);
  std::uniform_int_distribution<int> shape_dist(0, 5);
  std::uniform_int_distribution<int> len_dist(1, 19);
  std::uniform_int_distribution<int> exp_dist(-30, 30);

  std::string token;
  int sign = sign_dist(rng);
  if (sign == 0) {
    token.push_back('-');
  } else if (sign == 1) {
    token.push_back('+');
  }

  int shape = shape_dist(rng);
  if (shape == 0) {
    token.append("0.");
    int frac_len = std::uniform_int_distribution<int>(1, 19)(rng);
    for (int i = 0; i < frac_len; ++i) {
      token.push_back(static_cast<char>('0' + digit_dist(rng)));
    }
  } else {
    int int_len = len_dist(rng);
    token.push_back(static_cast<char>('1' + std::uniform_int_distribution<int>(0, 8)(rng)));
    for (int i = 1; i < int_len; ++i) {
      token.push_back(static_cast<char>('0' + digit_dist(rng)));
    }
    if (shape >= 2) {
      token.push_back('.');
      int remaining = 24 - static_cast<int>(token.size());
      int max_frac  = std::max(0, std::min(19, remaining));
      int frac_len  = max_frac == 0 ? 0 : std::uniform_int_distribution<int>(0, max_frac)(rng);
      for (int i = 0; i < frac_len; ++i) {
        token.push_back(static_cast<char>('0' + digit_dist(rng)));
      }
    }
  }

  if (shape == 5) {
    int exp            = exp_dist(rng);
    std::string suffix = "e" + std::to_string(exp);
    if (token.size() + suffix.size() <= 25) { token += suffix; }
  }

  if (token.size() > 25) { token.resize(25); }
  return token;
}

}  // namespace

TEST(FastFp64ParserTest, CommonTableMatchesStrtodBitwise)
{
  std::setlocale(LC_NUMERIC, "C");
  const std::vector<std::string_view> cases = {
    "0",
    "-0",
    "1",
    "-1",
    "+1",
    "2",
    "42",
    "123456789",
    "57.",
    "-57.",
    "0.1",
    "0.01",
    "0.12345678901234",
    "0.1234567890123456",
    "0.3333333333333333",
    "0.6508282938248958",
    "3.14159",
    "3130000",
    "8594600.16",
    "2344.55",
    "0.000000000000001",
    "9999999999999999",
    "1844674407370955161",
    "1e0",
    "1e-9",
    "1E12",
    "-2.5e3",
    "3.125D-2",
  };

  for (std::string_view token : cases) {
    check_bitwise_strtod(token);
  }
}

TEST(FastFp64ParserTest, CursorAdvancesToTokenEnd)
{
  std::setlocale(LC_NUMERIC, "C");
  std::string text = "123.45  ABC";
  const char* p    = text.data();
  double value     = mps_fast::fp64::parse_fp64_advance(p, text.data() + text.size());

  EXPECT_EQ(bits(reference_strtod("123.45")), bits(value));
  EXPECT_EQ(text.data() + 6, p);
  EXPECT_EQ(std::string_view("  ABC"), std::string_view(p, 5));
}

TEST(FastFp64ParserTest, FixedSeedRandomDifferential)
{
  std::setlocale(LC_NUMERIC, "C");
  std::mt19937_64 rng(0x4d50535f46415354ULL);
  for (int i = 0; i < 100000; ++i) {
    std::string token = random_token(rng);
    ASSERT_LE(token.size(), 25U);
    check_bitwise_strtod(token);
  }
}
