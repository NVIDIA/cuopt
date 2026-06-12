// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utilities/error.hpp>

#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace cuopt::linear_programming::io::detail {

using cuopt::linear_programming::io::error_type_t;
using cuopt::linear_programming::io::mps_parser_expects;
using cuopt::linear_programming::io::mps_parser_fail;

namespace fp64 {

#define FASTP64_MIN_EXP_10    (-307)
#define FASTP64_MAX_EXP_10    288
#define FASTP64_POWER_COUNT   (FASTP64_MAX_EXP_10 - FASTP64_MIN_EXP_10 + 1)
#define FASTP64_MANTISSA_MASK ((uint64_t{1} << 52) - 1)
#define FASTP64_EXPONENT_MASK 0x7FF
#define FASTP64_HALF_MASK     0x1FF

// Fast FP64 parser optimized for the <=19digits case, based on the Eisel-Lemire algorithm
// see Daniel Lemire, Number Parsing at a Gigabyte per Second, Software: Practice and Experience 51
// (8), 2021.
// verified on a large corpus of FP64 values: https://github.com/lemire/simple_fastfloat_benchmark

struct power_10_lut_entry_t {
  uint64_t high;
  uint64_t low;
  int biased_e2;
};

// util class to perform 256bit precision arithmetic in constexpr to build the eisel-lemire lookup
// table
struct cuopt_uint256_t {
  std::array<uint64_t, 4> limb{};

  constexpr uint32_t mul_u32(uint32_t m)
  {
    unsigned __int128 carry = 0;
    for (uint64_t& v : limb) {
      unsigned __int128 x = (unsigned __int128)v * m + carry;
      v                   = (uint64_t)x;
      carry               = x >> 64;
    }
    return (uint32_t)carry;
  }

  constexpr cuopt_uint256_t shl_small(int bits) const
  {
    cuopt_uint256_t out;
    if (bits == 0) return *this;
    for (int i = 3; i >= 0; --i) {
      uint64_t v = limb[i] << bits;
      if (i > 0) v |= limb[i - 1] >> (64 - bits);
      out.limb[i] = v;
    }
    return out;
  }
};

struct cuopt_normalized_uint256_t {
  cuopt_uint256_t sig;
  int exp2 = 0;

  static constexpr cuopt_normalized_uint256_t one()
  {
    cuopt_normalized_uint256_t x;
    x.sig.limb[3] = uint64_t{1} << 63;
    x.exp2        = -255;
    return x;
  }

  constexpr void mul10()
  {
    uint32_t carry = sig.mul_u32(10);
    int shift      = 32 - std::countl_zero(carry);
    // The normalized 256-bit value always overflows into carry after *10; keep
    // the guard explicit because the cross-limb path shifts by 64 - shift.
    if (shift == 0) { return; }
    cuopt_uint256_t out;
    for (int i = 0; i < 4; ++i) {
      uint64_t lower = sig.limb[i] >> shift;
      uint64_t upper = 0;
      if (i + 1 < 4) {
        upper = sig.limb[i + 1] << (64 - shift);
      } else {
        upper = (uint64_t)carry << (64 - shift);
      }
      out.limb[i] = lower | upper;
    }
    sig = out;
    exp2 += shift;
  }

  constexpr void div10()
  {
    constexpr uint64_t div10_shift_4_threshold = 0xA000000000000000ULL;
    int shift                                  = sig.limb[3] < div10_shift_4_threshold ? 4 : 3;
    uint64_t extra                             = sig.limb[3] >> (64 - shift);
    cuopt_uint256_t shifted                    = sig.shl_small(shift);

    cuopt_uint256_t quotient;
    unsigned __int128 rem = extra;
    for (int i = 3; i >= 0; --i) {
      unsigned __int128 cur = (rem << 64) | shifted.limb[i];
      quotient.limb[i]      = (uint64_t)(cur / 10);
      rem                   = cur % 10;
    }
    sig = quotient;
    exp2 -= shift;
  }
};

constexpr power_10_lut_entry_t make_power(const cuopt_normalized_uint256_t& p)
{
  int e2 = p.exp2 + 192;
  return {p.sig.limb[3], p.sig.limb[2], 1150 + e2};
}

// build time LUT for the lemire trick
constexpr std::array<power_10_lut_entry_t, FASTP64_POWER_COUNT> make_power_table()
{
  std::array<power_10_lut_entry_t, FASTP64_POWER_COUNT> table{};
  cuopt_normalized_uint256_t p = cuopt_normalized_uint256_t::one();
  table[-FASTP64_MIN_EXP_10]   = make_power(p);

  for (int e = 1; e <= FASTP64_MAX_EXP_10; ++e) {
    p.mul10();
    table[e - FASTP64_MIN_EXP_10] = make_power(p);
  }

  p = cuopt_normalized_uint256_t::one();
  for (int e = -1; e >= FASTP64_MIN_EXP_10; --e) {
    p.div10();
    table[e - FASTP64_MIN_EXP_10] = make_power(p);
  }
  return table;
}

inline constexpr auto fast_fp64_parse_lut = make_power_table();

inline constexpr std::array<double, 23> small_powers = {
  1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
  1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

inline constexpr std::array<uint64_t, 16> small_integer_powers = {1ULL,
                                                                  10ULL,
                                                                  100ULL,
                                                                  1000ULL,
                                                                  10000ULL,
                                                                  100000ULL,
                                                                  1000000ULL,
                                                                  10000000ULL,
                                                                  100000000ULL,
                                                                  1000000000ULL,
                                                                  10000000000ULL,
                                                                  100000000000ULL,
                                                                  1000000000000ULL,
                                                                  10000000000000ULL,
                                                                  100000000000000ULL,
                                                                  1000000000000000ULL};

struct parsed_decimal_t {
  bool negative      = false;
  bool fast_eligible = false;
  uint64_t mantissa  = 0;
  int exp10          = 0;
};

static inline bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

// SWAR 8char run of digits -> integer representation
// better and more portable than AVX2 stuff since AVX2 doesn't like swizzling across 16B lanes
// saw no real difference w/ 16B SSE
static inline bool parse_8_digits(const char* p, uint32_t& out)
{
  // comply with strict aliasing rules
  std::array<char, sizeof(uint64_t)> bytes{};
  std::memcpy(bytes.data(), p, bytes.size());
  uint64_t raw       = std::bit_cast<uint64_t>(bytes);
  uint64_t high      = raw & 0xF0F0F0F0F0F0F0F0ULL;
  uint64_t low_check = (raw + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL;
  if (high != 0x3030303030303030ULL || low_check != 0x3030303030303030ULL) { return false; }

  uint64_t v     = raw - 0x3030303030303030ULL;
  uint64_t pairs = (v * 10 + (v >> 8)) & 0x00FF00FF00FF00FFULL;
  uint64_t quads = (pairs * 100 + (pairs >> 16)) & 0x0000FFFF0000FFFFULL;
  out            = (uint32_t)((quads * 10000 + (quads >> 32)) & 0xFFFFFFFFULL);
  return true;
}

static inline void parse_u64_digits_advance(const char*& p, const char* end, uint64_t& out)
{
  while (p < end && is_digit(*p)) {
    if (end - p >= 8) {
      uint32_t chunk = 0;
      if (parse_8_digits(p, chunk)) {
        out = out * 100000000ULL + (uint64_t)chunk;
        p += 8;
        continue;
      }
    }
    out = out * 10 + (uint64_t)(*p - '0');
    ++p;
  }
}

static inline void scan_digit_run(const char*& p,
                                  const char* end,
                                  bool after_dot,
                                  parsed_decimal_t& out,
                                  bool& saw_digit,
                                  int& frac_digits,
                                  int& sig_digits,
                                  bool& too_many_digits)
{
  while (p < end) {
    uint32_t chunk = 0;
    if (end - p >= 8 && parse_8_digits(p, chunk)) {
      saw_digit = true;
      if (after_dot) frac_digits += 8;

      if (!too_many_digits) {
        if (sig_digits == 0 && chunk == 0) {
          p += 8;
          continue;
        }

        if (sig_digits + 8 <= 19) {
          out.mantissa = out.mantissa * 100000000ULL + chunk;
          sig_digits += 8;
        } else {
          too_many_digits = true;
        }
      }

      p += 8;
      continue;
    }

    if (!is_digit(*p)) return;
    saw_digit = true;
    int digit = *p - '0';
    if (after_dot) ++frac_digits;
    if (!too_many_digits && (digit != 0 || sig_digits != 0)) {
      if (sig_digits < 19) {
        out.mantissa = (out.mantissa * 10) + (uint64_t)digit;
        ++sig_digits;
      } else {
        too_many_digits = true;
      }
    }
    ++p;
  }
}

static inline bool parse_decimal_advance(const char*& p, const char* end, parsed_decimal_t& out)
{
  if (p < end && (*p == '-' || *p == '+')) {
    out.negative = *p == '-';
    ++p;
  }

  bool saw_digit       = false;
  int frac_digits      = 0;
  int sig_digits       = 0;
  bool too_many_digits = false;

  scan_digit_run(p, end, false, out, saw_digit, frac_digits, sig_digits, too_many_digits);
  if (p < end && *p == '.') {
    ++p;
    scan_digit_run(p, end, true, out, saw_digit, frac_digits, sig_digits, too_many_digits);
  }

  if (!saw_digit) return false;

  int explicit_exp = 0;
  if (p < end && (*p == 'e' || *p == 'E' || *p == 'd' || *p == 'D')) {
    const char* exp_start = p;
    ++p;
    bool exp_negative = false;
    if (p < end && (*p == '-' || *p == '+')) {
      exp_negative = *p == '-';
      ++p;
    }
    if (p == end || !is_digit(*p)) {
      p = exp_start;
    } else {
      int exp_value = 0;
      while (p < end && is_digit(*p)) {
        if (exp_value < 1000000) exp_value = exp_value * 10 + (*p - '0');
        ++p;
      }
      explicit_exp = exp_negative ? -exp_value : exp_value;
    }
  }

  out.exp10         = explicit_exp - frac_digits;
  out.fast_eligible = !too_many_digits;
  return true;
}

// fallback to stdlib for edge case or ambiguous roundings (very rare)
static inline double fallback_strtod(std::string_view s)
{
  char stack_buf[32];
  // The MPS specs mandate that numeric tokens are not longer than 25 characters
  if (s.size() >= sizeof(stack_buf)) {
    mps_parser_fail(error_type_t::ValidationError, "MPS numeric token exceeds supported length");
  }
  std::memcpy(stack_buf, s.data(), s.size());
  stack_buf[s.size()] = '\0';
  for (size_t i = 0; i < s.size(); ++i) {
    if (stack_buf[i] == 'd' || stack_buf[i] == 'D') stack_buf[i] = 'e';
  }

  char* parse_end = nullptr;
  errno           = 0;
  double value    = std::strtod(stack_buf, &parse_end);
  if (parse_end != stack_buf + s.size() || errno == ERANGE) {
    mps_parser_fail(error_type_t::ValidationError, "Invalid or out-of-range MPS numeric token");
  }
  return value;
}

// see Daniel Lemire, Number Parsing at a Gigabyte per Second, Software: Practice and Experience 51
// (8), 2021.
static inline bool eisel_lemire(uint64_t man, int exp10, uint64_t& bits)
{
  if (exp10 < FASTP64_MIN_EXP_10 || exp10 > FASTP64_MAX_EXP_10) { return false; }

  const power_10_lut_entry_t p = fast_fp64_parse_lut[exp10 - FASTP64_MIN_EXP_10];
  int lz                       = std::countl_zero(man);
  uint64_t norm                = man << lz;
  int adj_e2                   = p.biased_e2 - lz;

  unsigned __int128 product = (unsigned __int128)norm * p.high;
  uint64_t hi               = (uint64_t)(product >> 64);
  uint64_t lo               = (uint64_t)product;

  // If the high product lands near the 9-bit halfway window, include the low
  // 64x64 product to disambiguate rounding before deciding whether to fallback.
  if ((hi & FASTP64_HALF_MASK) == FASTP64_HALF_MASK && lo + norm < norm) {
    unsigned __int128 low_product = (unsigned __int128)norm * p.low;
    uint64_t low_hi               = (uint64_t)(low_product >> 64);
    uint64_t low_lo               = (uint64_t)low_product;
    uint64_t old_lo               = lo;
    lo += low_hi;
    hi += lo < old_lo ? 1 : 0;
    if ((hi & FASTP64_HALF_MASK) == FASTP64_HALF_MASK &&
        lo == std::numeric_limits<uint64_t>::max() && low_lo + norm < low_lo) {
      return false;
    }
  }

  uint64_t hi_msb = hi >> 63;
  // Extract 54 bits: 53 significand bits plus one rounding bit. The product
  // may be shifted by one depending on whether hi already has its top bit set.
  uint64_t x54 = hi >> (9 + hi_msb);
  adj_e2 -= (int)(1 - hi_msb);

  // Exact halfway with round-to-even ambiguity; let strtod handle the rare tie.
  if (lo == 0 && (hi & FASTP64_HALF_MASK) == 0 && (x54 & 3) == 1) { return false; }

  // Round 54 -> 53 bits, carry into the exponent if rounding overflows.
  uint64_t x53      = (x54 + (x54 & 1)) >> 1;
  uint64_t overflow = x53 >> 53;
  uint64_t ret_man  = (x53 >> overflow) & FASTP64_MANTISSA_MASK;
  int ret_exp       = adj_e2 + (int)overflow;
  if (ret_exp <= 0 || ret_exp >= FASTP64_EXPONENT_MASK) { return false; }

  bits = ((uint64_t)ret_exp << 52) | ret_man;
  return true;
}

static inline double assemble_fp64(const parsed_decimal_t& dec)
{
  uint64_t bits = dec.negative ? (uint64_t{1} << 63) : 0;
  if (dec.mantissa == 0) { return std::bit_cast<double>(bits); }

  if (dec.fast_eligible) {
    double small    = 0.0;
    bool used_small = false;
    if (dec.exp10 >= 0 && dec.exp10 < (int)small_integer_powers.size()) {
      uint64_t limit = (uint64_t{1} << 53) / small_integer_powers[dec.exp10];
      if (dec.mantissa <= limit) {
        small      = (double)dec.mantissa * small_powers[dec.exp10];
        used_small = true;
      }
    } else if (dec.exp10 < 0 && dec.exp10 >= -22 && dec.mantissa < (uint64_t{1} << 53)) {
      small      = (double)dec.mantissa / small_powers[-dec.exp10];
      used_small = true;
    }
    if (used_small) { return dec.negative ? -small : small; }

    uint64_t mag_bits = 0;
    if (eisel_lemire(dec.mantissa, dec.exp10, mag_bits)) {
      return std::bit_cast<double>(bits | mag_bits);
    }
  }

  return std::numeric_limits<double>::quiet_NaN();
}

static inline double parse_fp64_advance(const char*& p, const char* end)
{
  const char* start = p;
  parsed_decimal_t dec;
  if (!parse_decimal_advance(p, end, dec)) {
    return fallback_strtod(std::string_view(start, (size_t)(p - start)));
  }

  double v = assemble_fp64(dec);
  if (v == v) return v;
  return fallback_strtod(std::string_view(start, (size_t)(p - start)));
}

}  // namespace fp64
}  // namespace cuopt::linear_programming::io::detail
