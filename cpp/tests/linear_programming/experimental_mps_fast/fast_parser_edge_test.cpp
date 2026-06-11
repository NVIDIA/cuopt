// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fast_parser.hpp"
#include "mps_section_scanner.hpp"

#include <cuopt/linear_programming/io/parser.hpp>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

struct skip_test : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void expect_true(bool condition, const std::string& message)
{
  if (!condition) { fail(message); }
}

template <typename A, typename B>
void expect_eq(const A& got, const B& expected, std::string_view context)
{
  if (!(got == expected)) {
    std::ostringstream out;
    out << context << ": got=" << got << " expected=" << expected;
    fail(out.str());
  }
}

template <typename VecA, typename VecB>
void expect_vector_eq(const VecA& got, const VecB& expected, std::string_view context)
{
  if (got.size() != expected.size()) {
    std::ostringstream out;
    out << context << ": size got=" << got.size() << " expected=" << expected.size();
    fail(out.str());
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (!(got[i] == expected[i])) {
      std::ostringstream out;
      out << context << ": first mismatch at " << i;
      fail(out.str());
    }
  }
}

void expect_near_inf(double value, int sign, std::string_view context)
{
  expect_true(std::isinf(value), std::string(context) + ": expected infinity");
  expect_true(std::signbit(value) == (sign < 0), std::string(context) + ": wrong infinity sign");
}

struct TempMpsFile {
  explicit TempMpsFile(std::string contents)
  {
    char path_template[128];
    std::snprintf(path_template,
                  sizeof(path_template),
                  "/tmp/mps_fast_parser_edge_%ld_XXXXXX.mps",
                  static_cast<long>(getpid()));
    int fd = mkstemps(path_template, 4);
    if (fd < 0) { fail(std::string("mkstemps failed: ") + std::strerror(errno)); }
    path       = path_template;
    FILE* file = fdopen(fd, "wb");
    if (file == nullptr) {
      close(fd);
      fail(std::string("fdopen failed: ") + std::strerror(errno));
    }
    if (!contents.empty() &&
        std::fwrite(contents.data(), 1, contents.size(), file) != contents.size()) {
      std::fclose(file);
      fail(std::string("failed to write temporary MPS file: ") + std::strerror(errno));
    }
    if (std::fclose(file) != 0) {
      fail(std::string("failed to close temporary MPS file: ") + std::strerror(errno));
    }
  }

  TempMpsFile(const TempMpsFile&)            = delete;
  TempMpsFile& operator=(const TempMpsFile&) = delete;

  ~TempMpsFile()
  {
    if (!path.empty()) { std::remove(path.c_str()); }
  }

  std::string path;
};

struct TempOwnedPath {
  explicit TempOwnedPath(std::string p) : path(std::move(p)) {}
  TempOwnedPath(const TempOwnedPath&)            = delete;
  TempOwnedPath& operator=(const TempOwnedPath&) = delete;

  ~TempOwnedPath()
  {
    if (!path.empty()) { std::remove(path.c_str()); }
  }

  std::string path;
};

template <typename Fn>
void expect_throws(Fn&& fn, std::string_view context)
{
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  fail(std::string(context) + ": expected exception");
}

void expect_fast_parse_error(std::string_view fixture_name, std::string contents)
{
  TempMpsFile file(std::move(contents));
  expect_throws(
    [&] {
      (void)mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
    },
    fixture_name);
}

std::string_view range_text(const mps_fast::mps_phase_range_t& range)
{
  if (!range.present) { return {}; }
  return std::string_view(range.begin, static_cast<size_t>(range.end - range.begin));
}

void scanner_finds_section_split_across_blocks()
{
  const std::string mps =
    "NAME EDGE\n"
    "ROWS\n"
    " N OBJ\n"
    " L rowA\n"
    "COLUMNS\n"
    " x1 OBJ 1\n"
    " x1 rowA 2\n"
    "RHS\n"
    " rhs rowA 3\n"
    "ENDATA\n";

  const size_t columns_pos = mps.find("COLUMNS");
  expect_true(columns_pos != std::string::npos, "failed to place COLUMNS split");
  const size_t split = columns_pos + 3;

  mps_fast::mps_phase_registry_t registry;
  mps_fast::mps_section_block_scanner_t scanner(mps.data(), 2, registry);

  scanner.observe_block(1, mps.data() + split, mps.data() + mps.size());
  scanner.publish_ready(0);
  scanner.observe_block(0, mps.data(), mps.data() + split);
  scanner.publish_ready(mps.size());

  expect_true(registry.ready(mps_fast::mps_phase_kind::header), "header not ready");
  expect_true(registry.ready(mps_fast::mps_phase_kind::rows), "rows not ready");
  expect_true(registry.ready(mps_fast::mps_phase_kind::columns), "columns not ready");
  expect_true(registry.ready(mps_fast::mps_phase_kind::rhs), "rhs not ready");
  expect_true(registry.ready(mps_fast::mps_phase_kind::quadratic), "quadratic sentinel not ready");

  expect_true(range_text(registry.range(mps_fast::mps_phase_kind::columns)).starts_with("COLUMNS"),
              "columns range begins at wrong boundary");
  expect_true(range_text(registry.range(mps_fast::mps_phase_kind::rhs)).starts_with("RHS"),
              "rhs range begins at wrong boundary");
}

void scanner_rejects_unknown_column_one_records_after_rows()
{
  const std::string mps =
    "NAME BAD\n"
    "ROWS\n"
    " N OBJ\n"
    "FOO\n"
    "COLUMNS\n"
    " x OBJ 1\n"
    "ENDATA\n";

  expect_throws(
    [&] {
      mps_fast::mps_phase_registry_t registry;
      mps_fast::mps_section_block_scanner_t scanner(mps.data(), 1, registry);
      scanner.observe_block(0, mps.data(), mps.data() + mps.size());
      scanner.publish_ready(mps.size());
    },
    "unknown column-1 record after ROWS");
}

uint64_t bits(double value) { return std::bit_cast<uint64_t>(value); }

void expect_double_bitwise_eq(double got, double expected, std::string_view context)
{
  if (bits(got) != bits(expected)) {
    std::ostringstream out;
    out << context << ": got=0x" << std::hex << bits(got) << " expected=0x" << bits(expected);
    fail(out.str());
  }
}

template <typename VecA, typename VecB>
void expect_double_vector_bitwise_eq(const VecA& got,
                                     const VecB& expected,
                                     std::string_view context)
{
  if (got.size() != expected.size()) {
    std::ostringstream out;
    out << context << ": size got=" << got.size() << " expected=" << expected.size();
    fail(out.str());
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (bits(got[i]) != bits(expected[i])) {
      std::ostringstream out;
      out << context << ": first bitwise mismatch at " << i << " got=0x" << std::hex << bits(got[i])
          << " expected=0x" << bits(expected[i]);
      fail(out.str());
    }
  }
}

void expect_models_match_reference_bitwise(
  const mps_fast::parser_model_t<int, double>& fast,
  const cuopt::linear_programming::io::mps_data_model_t<int, double>& reference,
  std::string_view context)
{
  expect_eq(fast.n_vars_, reference.n_vars_, std::string(context) + " n_vars");
  expect_eq(fast.n_constraints_, reference.n_constraints_, std::string(context) + " n_constraints");
  expect_eq(fast.nnz_, reference.nnz_, std::string(context) + " nnz");
  expect_eq(fast.maximize_, reference.maximize_, std::string(context) + " maximize");
  expect_eq(fast.problem_name_, reference.problem_name_, std::string(context) + " problem_name");
  expect_eq(
    fast.objective_name_, reference.objective_name_, std::string(context) + " objective_name");

  expect_double_bitwise_eq(fast.objective_scaling_factor_,
                           reference.objective_scaling_factor_,
                           std::string(context) + " objective_scaling_factor");
  expect_double_bitwise_eq(fast.objective_offset_,
                           reference.objective_offset_,
                           std::string(context) + " objective_offset");

  expect_double_vector_bitwise_eq(fast.A_, reference.A_, std::string(context) + " A");
  expect_vector_eq(fast.A_indices_, reference.A_indices_, std::string(context) + " A_indices");
  expect_vector_eq(fast.A_offsets_, reference.A_offsets_, std::string(context) + " A_offsets");
  expect_double_vector_bitwise_eq(fast.b_, reference.b_, std::string(context) + " b");
  expect_double_vector_bitwise_eq(fast.c_, reference.c_, std::string(context) + " c");
  expect_double_vector_bitwise_eq(fast.variable_lower_bounds_,
                                  reference.variable_lower_bounds_,
                                  std::string(context) + " variable_lower_bounds");
  expect_double_vector_bitwise_eq(fast.variable_upper_bounds_,
                                  reference.variable_upper_bounds_,
                                  std::string(context) + " variable_upper_bounds");
  expect_double_vector_bitwise_eq(fast.constraint_lower_bounds_,
                                  reference.constraint_lower_bounds_,
                                  std::string(context) + " constraint_lower_bounds");
  expect_double_vector_bitwise_eq(fast.constraint_upper_bounds_,
                                  reference.constraint_upper_bounds_,
                                  std::string(context) + " constraint_upper_bounds");
  expect_vector_eq(fast.var_types_, reference.var_types_, std::string(context) + " var_types");
  expect_vector_eq(fast.row_types_, reference.row_types_, std::string(context) + " row_types");
  expect_vector_eq(fast.var_names_, reference.var_names_, std::string(context) + " var_names");
  expect_vector_eq(fast.row_names_, reference.row_names_, std::string(context) + " row_names");
}

void verify_fixture_bitwise(std::string_view fixture_name, std::string contents)
{
  TempMpsFile file(std::move(contents));
  auto fast = mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
  auto reference = cuopt::linear_programming::io::read_mps<int, double>(file.path, false);
  expect_models_match_reference_bitwise(fast, reference, fixture_name);
}

std::string row_name(size_t i)
{
  std::ostringstream out;
  out << 'R' << std::setw(6) << std::setfill('0') << i;
  return out.str();
}

size_t find_var(const mps_fast::parser_model_t<int, double>& model, std::string_view name)
{
  for (size_t i = 0; i < model.var_names_.size(); ++i) {
    if (model.var_names_[i] == name) { return i; }
  }
  fail("variable not found: " + std::string(name));
}

void expect_model_shapes(const mps_fast::parser_model_t<int, double>& model,
                         int rows,
                         int vars,
                         int nnz,
                         std::string_view context)
{
  expect_eq(model.n_constraints_, rows, std::string(context) + " rows");
  expect_eq(model.n_vars_, vars, std::string(context) + " vars");
  expect_eq(model.nnz_, nnz, std::string(context) + " nnz");
  expect_eq(
    model.A_offsets_.size(), static_cast<size_t>(rows + 1), std::string(context) + " offsets");
  expect_eq(model.A_.size(), static_cast<size_t>(nnz), std::string(context) + " values");
  expect_eq(model.A_indices_.size(), static_cast<size_t>(nnz), std::string(context) + " indices");
}

std::string section_split_fixture()
{
  return "NAME SPLITS\n"
         "ROWS\n"
         " N OBJ\n"
         " L R1\n"
         "COLUMNS\n"
         " X1 OBJ 1 R1 2\n"
         "RHS\n"
         " RHS1 R1 3\n"
         "BOUNDS\n"
         " UP BND X1 4\n"
         "ENDATA\n";
}

void scanner_finds_headers_split_at_every_byte()
{
  const std::string mps                       = section_split_fixture();
  const std::vector<std::string_view> headers = {"ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"};

  for (std::string_view header : headers) {
    const size_t pos = mps.find(header);
    expect_true(pos != std::string::npos, "missing header in split fixture");
    for (size_t offset = 1; offset < header.size(); ++offset) {
      const size_t split = pos + offset;
      mps_fast::mps_phase_registry_t registry;
      mps_fast::mps_section_block_scanner_t scanner(mps.data(), 2, registry);

      scanner.observe_block(1, mps.data() + split, mps.data() + mps.size());
      scanner.observe_block(0, mps.data(), mps.data() + split);
      scanner.publish_ready(mps.size());

      expect_true(registry.ready(mps_fast::mps_phase_kind::rows), "rows not ready after split");
      expect_true(registry.ready(mps_fast::mps_phase_kind::columns),
                  "columns not ready after split");
      expect_true(registry.ready(mps_fast::mps_phase_kind::rhs), "rhs not ready after split");
      expect_true(registry.ready(mps_fast::mps_phase_kind::bounds), "bounds not ready after split");
      expect_true(registry.ready(mps_fast::mps_phase_kind::quadratic),
                  "quadratic sentinel not ready after split");
    }
  }
}

void bounds_defaults_and_types_match_reference()
{
  verify_fixture_bitwise("bounds_defaults_and_types",
                         "NAME BOUNDS_EDGE\n"
                         "ROWS\n"
                         " N OBJ\n"
                         " L rowA\n"
                         "COLUMNS\n"
                         " XFREE rowA 1\n"
                         " XUP0 rowA 1\n"
                         " XNEG rowA 1\n"
                         " XBV rowA 1\n"
                         " XFX rowA 1\n"
                         " XLI rowA 1\n"
                         "RHS\n"
                         " RHS1 rowA 10\n"
                         "BOUNDS\n"
                         " FR BND XFREE\n"
                         " UP BND XUP0 0\n"
                         " UP BND XNEG -1\n"
                         " BV BND XBV\n"
                         " FX BND XFX 7\n"
                         " LI BND XLI 2\n"
                         " UI BND XLI 9\n"
                         "ENDATA\n");
}

void duplicate_bounds_last_statement_wins()
{
  const std::string contents =
    "NAME BOUNDS_DUP\n"
    "ROWS\n"
    " N OBJ\n"
    " L rowA\n"
    "COLUMNS\n"
    " X1 rowA 1\n"
    "RHS\n"
    " RHS1 rowA 10\n"
    "BOUNDS\n"
    " LO BND X1 0\n"
    " UP BND X1 5\n"
    " UP BND X1 3\n"
    " LO BND X1 2\n"
    "ENDATA\n";

  verify_fixture_bitwise("duplicate_bounds_last_statement_wins", contents);
  TempMpsFile file(contents);
  auto model =
    mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
  expect_eq(model.n_vars_, 1, "n_vars");
  expect_eq(model.variable_lower_bounds_.at(0), 2.0, "duplicate lower bound");
  expect_eq(model.variable_upper_bounds_.at(0), 3.0, "duplicate upper bound");
}

void nondense_row_and_column_names_use_hash_path()
{
  verify_fixture_bitwise("nondense_row_and_column_names",
                         "NAME HASH_NAMES\n"
                         "ROWS\n"
                         " N obj.row\n"
                         " G demand-east\n"
                         " L capacity-west\n"
                         " E balance.17\n"
                         "COLUMNS\n"
                         " alpha obj.row 4.5 demand-east 1\n"
                         " beta_two capacity-west -2 balance.17 3\n"
                         " z-last demand-east 7 balance.17 -1\n"
                         "RHS\n"
                         " rhs demand-east 2 capacity-west 9\n"
                         " rhs balance.17 0\n"
                         "BOUNDS\n"
                         " LO b alpha -5\n"
                         " UP b beta_two 6\n"
                         " FR b z-last\n"
                         "ENDATA\n");
}

void missing_optional_bounds_fast_path()
{
  TempMpsFile file(
    "NAME OPTIONALS\n"
    "ROWS\n"
    " N OBJ\n"
    " L rowA\n"
    "COLUMNS\n"
    " X1 OBJ 1 rowA 2\n"
    "RHS\n"
    " RHS1 rowA 0\n"
    "ENDATA\n");

  auto model =
    mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
  expect_eq(model.n_vars_, 1, "missing optional n_vars");
  expect_eq(model.n_constraints_, 1, "missing optional n_constraints");
  expect_eq(model.variable_lower_bounds_.at(0), 0.0, "missing BOUNDS lower default");
  expect_near_inf(model.variable_upper_bounds_.at(0), 1, "missing BOUNDS upper default");
}

void bounds_only_variables_are_appended_deterministically()
{
  TempMpsFile file(
    "NAME BOUNDS_ONLY\n"
    "ROWS\n"
    " N OBJ\n"
    " L R1\n"
    "COLUMNS\n"
    " XMAIN OBJ 1 R1 2\n"
    "RHS\n"
    " RHS1 R1 0\n"
    "BOUNDS\n"
    " UP B AUX_Z 9\n"
    " LO B AUX_Z -3\n"
    " BV B AUX_A\n"
    " SC B AUX_S 5\n"
    "ENDATA\n");

  auto model =
    mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
  expect_model_shapes(model, 1, 4, 1, "bounds-only");
  expect_eq(model.var_names_.at(0), std::string("XMAIN"), "main var name");
  expect_eq(model.var_names_.at(1), std::string("AUX_A"), "bounds-only sorted name 1");
  expect_eq(model.var_names_.at(2), std::string("AUX_S"), "bounds-only sorted name 2");
  expect_eq(model.var_names_.at(3), std::string("AUX_Z"), "bounds-only sorted name 3");

  size_t aux_a = find_var(model, "AUX_A");
  size_t aux_s = find_var(model, "AUX_S");
  size_t aux_z = find_var(model, "AUX_Z");
  expect_eq(model.var_types_.at(aux_a), 'I', "bounds-only BV type");
  expect_eq(model.variable_lower_bounds_.at(aux_a), 0.0, "bounds-only BV lb");
  expect_eq(model.variable_upper_bounds_.at(aux_a), 1.0, "bounds-only BV ub");
  expect_eq(model.var_types_.at(aux_s), 'S', "bounds-only SC type");
  expect_eq(model.variable_upper_bounds_.at(aux_s), 5.0, "bounds-only SC ub");
  expect_eq(model.variable_lower_bounds_.at(aux_z), -3.0, "bounds-only duplicate lb");
  expect_eq(model.variable_upper_bounds_.at(aux_z), 9.0, "bounds-only duplicate ub");
}

void integer_markers_assign_types_and_default_bounds()
{
  TempMpsFile file(
    "NAME MARKERS\n"
    "ROWS\n"
    " N OBJ\n"
    " L R1\n"
    "COLUMNS\n"
    " MARK000 'MARKER' 'INTORG'\n"
    " XINT OBJ 1 R1 1\n"
    " MARK001 'MARKER' 'INTEND'\n"
    " XCONT OBJ 2 R1 2\n"
    " MARK002 'MARKER' 'INTORG'\n"
    " XBIN OBJ 3 R1 3\n"
    " MARK003 'MARKER' 'INTEND'\n"
    "RHS\n"
    " RHS1 R1 10\n"
    "ENDATA\n");

  auto model =
    mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
  expect_model_shapes(model, 1, 3, 3, "integer markers");
  size_t xint  = find_var(model, "XINT");
  size_t xcont = find_var(model, "XCONT");
  size_t xbin  = find_var(model, "XBIN");
  expect_eq(model.var_types_.at(xint), 'I', "XINT type");
  expect_eq(model.var_types_.at(xcont), 'C', "XCONT type");
  expect_eq(model.var_types_.at(xbin), 'I', "XBIN type");
  expect_eq(model.variable_lower_bounds_.at(xint), 0.0, "XINT default lb");
  expect_eq(model.variable_upper_bounds_.at(xint), 1.0, "XINT default ub");
  expect_eq(model.variable_lower_bounds_.at(xbin), 0.0, "XBIN default lb");
  expect_eq(model.variable_upper_bounds_.at(xbin), 1.0, "XBIN default ub");
}

void numeric_parsing_integration_matches_reference_bitwise()
{
  verify_fixture_bitwise("numeric_parsing_integration",
                         "NAME NUMBERS\n"
                         "ROWS\n"
                         " N OBJ\n"
                         " L R1\n"
                         " G R2\n"
                         " E R3\n"
                         "COLUMNS\n"
                         " X0 OBJ 0.12345678901234 R1 1e-9\n"
                         " X1 OBJ -2.5E3 R2 0.12345678901234567890123\n"
                         " X2 R3 9999999999999999\n"
                         "RHS\n"
                         " RHS1 R1 3.14159 R2 -0.000000000000001\n"
                         " RHS1 R3 42\n"
                         "RANGES\n"
                         " RNG R1 0.25 R2 1E2\n"
                         "BOUNDS\n"
                         " LO B X0 -123456789\n"
                         " UP B X0 123456789\n"
                         " FX B X1 0.3333333333333333\n"
                         " FR B X2\n"
                         "ENDATA\n");
}

std::string to_crlf(std::string text)
{
  std::string converted;
  converted.reserve(text.size() + text.size() / 8);
  for (char c : text) {
    if (c == '\n') {
      converted += "\r\n";
    } else {
      converted.push_back(c);
    }
  }
  return converted;
}

void crlf_line_endings_match_reference_bitwise()
{
  verify_fixture_bitwise("crlf_line_endings",
                         to_crlf("NAME CRLF_EDGE\n"
                                 "OBJSENSE\n"
                                 " MAX\n"
                                 "ROWS\n"
                                 " N OBJ\n"
                                 " L R1\n"
                                 "COLUMNS\n"
                                 " X1 OBJ 1 R1 2\n"
                                 "RHS\n"
                                 " RHS1 R1 3\n"
                                 "BOUNDS\n"
                                 " UP B X1 4\n"
                                 "ENDATA\n"));
}

void comment_placement_supported_cases_match_reference_bitwise()
{
  verify_fixture_bitwise("comment_placement_supported_cases",
                         "* leading star comment\n"
                         "$ leading dollar comment\n"
                         "NAME COMMENTS\n"
                         "$ comment between NAME and ROWS\n"
                         "ROWS\n"
                         "* comment after ROWS header\n"
                         " N OBJ $ row objective comment\n"
                         "$ comment between ROW records\n"
                         " L R1 $ row constraint comment\n"
                         "COLUMNS\n"
                         "* comment after COLUMNS header\n"
                         " X1 OBJ 1 R1 2 $ inline column comment\n"
                         "$ comment before next column\n"
                         " X2 OBJ -1 R1 3\n"
                         "RHS\n"
                         "$ comment after RHS header\n"
                         " RHS1 R1 5 $ inline rhs comment\n"
                         "BOUNDS\n"
                         "* comment after BOUNDS header\n"
                         " LO B X1 0 $ inline bound comment\n"
                         "$ comment before ENDATA\n"
                         "ENDATA\n");
}

void objective_metadata_selects_named_objective()
{
  TempMpsFile file(
    "NAME OBJMETA\n"
    "OBJSENSE\n"
    " MAX\n"
    "OBJNAME\n"
    " COST\n"
    "ROWS\n"
    " N ALT\n"
    " N COST\n"
    " L R1\n"
    "COLUMNS\n"
    " X1 ALT 100 COST 5\n"
    " X1 R1 1\n"
    " X2 COST -2 R1 3\n"
    "RHS\n"
    " RHS1 COST 7 R1 11\n"
    "ENDATA\n");

  auto model =
    mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
  expect_true(model.maximize_, "OBJSENSE MAX not applied");
  expect_eq(model.problem_name_, std::string("OBJMETA"), "problem name");
  expect_eq(model.objective_name_, std::string("COST"), "objective name");
  expect_eq(model.objective_offset_, -7.0, "objective RHS offset");
  size_t x1 = find_var(model, "X1");
  size_t x2 = find_var(model, "X2");
  expect_eq(model.c_.at(x1), 5.0, "named objective coefficient X1");
  expect_eq(model.c_.at(x2), -2.0, "named objective coefficient X2");
}

void malformed_inputs_report_errors()
{
  expect_fast_parse_error("bad objsense",
                          "NAME BADOBJ\n"
                          "OBJSENSE\n"
                          " SIDEWAYS\n"
                          "ROWS\n"
                          " N OBJ\n"
                          " L R1\n"
                          "COLUMNS\n"
                          " X1 OBJ 1 R1 2\n"
                          "RHS\n"
                          " RHS1 R1 0\n"
                          "ENDATA\n");

  expect_fast_parse_error("unknown row in columns",
                          "NAME BADCOLROW\n"
                          "ROWS\n"
                          " N OBJ\n"
                          " L R1\n"
                          "COLUMNS\n"
                          " X1 MISSING 1\n"
                          "RHS\n"
                          " RHS1 R1 0\n"
                          "ENDATA\n");

  expect_fast_parse_error("unknown row in rhs",
                          "NAME BADRHSROW\n"
                          "ROWS\n"
                          " N OBJ\n"
                          " L R1\n"
                          "COLUMNS\n"
                          " X1 OBJ 1 R1 2\n"
                          "RHS\n"
                          " RHS1 MISSING 1\n"
                          "ENDATA\n");

  expect_fast_parse_error("unknown bound type",
                          "NAME BADBOUND\n"
                          "ROWS\n"
                          " N OBJ\n"
                          " L R1\n"
                          "COLUMNS\n"
                          " X1 OBJ 1 R1 2\n"
                          "RHS\n"
                          " RHS1 R1 0\n"
                          "BOUNDS\n"
                          " XX B X1 1\n"
                          "ENDATA\n");

  expect_fast_parse_error("semi-continuous bound without value",
                          "NAME BADSC\n"
                          "ROWS\n"
                          " N OBJ\n"
                          " L R1\n"
                          "COLUMNS\n"
                          " X1 OBJ 1 R1 2\n"
                          "RHS\n"
                          " RHS1 R1 0\n"
                          "BOUNDS\n"
                          " SC B X1\n"
                          "ENDATA\n");
}

void large_columns_repeated_column_chunk_boundary()
{
  constexpr size_t row_count = 180000;
  std::string mps;
  mps.reserve(8 * 1024 * 1024);
  mps += "NAME BIGCOLS\nROWS\n N OBJ\n";
  for (size_t i = 1; i <= row_count; ++i) {
    mps += " L ";
    mps += row_name(i);
    mps += '\n';
  }
  mps += "COLUMNS\n";
  for (size_t i = 1; i <= row_count; ++i) {
    mps += " XBIG ";
    mps += row_name(i);
    mps += " 1\n";
  }
  mps += " XTAIL ";
  mps += row_name(1);
  mps += " 2\nRHS\n RHS1 ";
  mps += row_name(1);
  mps += " 0\nENDATA\n";

  TempMpsFile file(std::move(mps));
  auto model =
    mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
  expect_model_shapes(
    model, static_cast<int>(row_count), 2, static_cast<int>(row_count + 1), "large columns");
  expect_eq(model.var_names_.at(0), std::string("XBIG"), "large repeated column name");
  expect_eq(model.var_names_.at(1), std::string("XTAIL"), "large tail column name");
}

void large_bounds_repeated_var_stays_ordered()
{
  constexpr size_t repeat_count = 700000;
  std::string mps;
  mps.reserve(12 * 1024 * 1024);
  mps +=
    "NAME BIGBOUNDS\nROWS\n N OBJ\n L R1\nCOLUMNS\n alpha OBJ 1 R1 1\nRHS\n RHS1 R1 0\nBOUNDS\n";
  for (size_t i = 0; i < repeat_count; ++i) {
    mps += " UP B alpha ";
    mps += std::to_string(i % 1000);
    mps += '\n';
  }
  mps += "ENDATA\n";

  TempMpsFile file(std::move(mps));
  auto model =
    mps_fast::parse_mps_fast_file<int, double>(file.path, mps_fast::FileReadMethod::Read);
  expect_model_shapes(model, 1, 1, 1, "large bounds");
  expect_eq(model.variable_upper_bounds_.at(0),
            static_cast<double>((repeat_count - 1) % 1000),
            "large repeated bounds last value");
}

void lz4_and_raw_paths_match_on_multiblock_input()
{
  constexpr size_t row_count = 70000;
  std::string mps;
  mps.reserve(4 * 1024 * 1024);
  mps += "NAME LZ4PARITY\nROWS\n N OBJ\n";
  for (size_t i = 1; i <= row_count; ++i) {
    mps += " L ";
    mps += row_name(i);
    mps += '\n';
  }
  mps += "COLUMNS\n";
  for (size_t i = 1; i <= row_count; ++i) {
    mps += " X";
    mps += std::to_string(i);
    mps += ' ';
    mps += row_name(i);
    mps += " 0.125\n";
  }
  mps += "RHS\n RHS1 ";
  mps += row_name(1);
  mps += " 1\nENDATA\n";

  TempMpsFile raw_file(std::move(mps));
  TempOwnedPath lz4_file(raw_file.path + ".lz4");
  const std::string cmd = "lz4 -f -q " + raw_file.path + " " + lz4_file.path;
  if (std::system(cmd.c_str()) != 0) { throw skip_test("lz4 CLI unavailable"); }

  auto raw =
    mps_fast::parse_mps_fast_file<int, double>(raw_file.path, mps_fast::FileReadMethod::Read);
  auto lz4 =
    mps_fast::parse_mps_fast_file<int, double>(lz4_file.path, mps_fast::FileReadMethod::Read);

  expect_model_shapes(lz4, raw.n_constraints_, raw.n_vars_, raw.nnz_, "lz4 parity");
  expect_eq(lz4.var_names_.size(), raw.var_names_.size(), "lz4 var name count");
  expect_eq(lz4.row_names_.size(), raw.row_names_.size(), "lz4 row name count");
  expect_vector_eq(lz4.A_, raw.A_, "lz4 A values");
  expect_vector_eq(lz4.A_indices_, raw.A_indices_, "lz4 A indices");
  expect_vector_eq(lz4.A_offsets_, raw.A_offsets_, "lz4 A offsets");
  expect_vector_eq(lz4.c_, raw.c_, "lz4 objective");
  expect_vector_eq(lz4.b_, raw.b_, "lz4 rhs");
  expect_vector_eq(lz4.var_types_, raw.var_types_, "lz4 var types");
  expect_vector_eq(lz4.variable_lower_bounds_, raw.variable_lower_bounds_, "lz4 lower bounds");
  expect_vector_eq(lz4.variable_upper_bounds_, raw.variable_upper_bounds_, "lz4 upper bounds");
}

void gzip_bzip2_and_raw_paths_match()
{
  std::string mps;
  mps += "NAME COMPRESSED\nROWS\n N OBJ\n L R1\n G R2\nCOLUMNS\n";
  mps += " X1 OBJ 1 R1 2.5\n X2 R1 -3.25 R2 4\n";
  mps += "RHS\n RHS1 R1 7 R2 8\nBOUNDS\n BV BND X1\n UP BND X2 10\nENDATA\n";

  TempMpsFile raw_file(std::move(mps));
  TempOwnedPath gzip_file(raw_file.path + ".gz");
  TempOwnedPath bzip2_file(raw_file.path + ".bz2");

  const std::string gzip_cmd  = "gzip -c " + raw_file.path + " > " + gzip_file.path;
  const std::string bzip2_cmd = "bzip2 -c " + raw_file.path + " > " + bzip2_file.path;
  if (std::system(gzip_cmd.c_str()) != 0) { throw skip_test("gzip CLI unavailable"); }
  if (std::system(bzip2_cmd.c_str()) != 0) { throw skip_test("bzip2 CLI unavailable"); }

  auto raw =
    mps_fast::parse_mps_fast_file<int, double>(raw_file.path, mps_fast::FileReadMethod::Read);
  auto gzip =
    mps_fast::parse_mps_fast_file<int, double>(gzip_file.path, mps_fast::FileReadMethod::Read);
  auto bzip2 =
    mps_fast::parse_mps_fast_file<int, double>(bzip2_file.path, mps_fast::FileReadMethod::Read);

  expect_model_shapes(gzip, raw.n_constraints_, raw.n_vars_, raw.nnz_, "gzip parity");
  expect_model_shapes(bzip2, raw.n_constraints_, raw.n_vars_, raw.nnz_, "bzip2 parity");
  expect_vector_eq(gzip.A_, raw.A_, "gzip A values");
  expect_vector_eq(bzip2.A_, raw.A_, "bzip2 A values");
  expect_vector_eq(gzip.A_indices_, raw.A_indices_, "gzip A indices");
  expect_vector_eq(bzip2.A_indices_, raw.A_indices_, "bzip2 A indices");
  expect_vector_eq(gzip.A_offsets_, raw.A_offsets_, "gzip A offsets");
  expect_vector_eq(bzip2.A_offsets_, raw.A_offsets_, "bzip2 A offsets");
  expect_vector_eq(gzip.c_, raw.c_, "gzip objective");
  expect_vector_eq(bzip2.c_, raw.c_, "bzip2 objective");
  expect_vector_eq(gzip.b_, raw.b_, "gzip rhs");
  expect_vector_eq(bzip2.b_, raw.b_, "bzip2 rhs");
  expect_vector_eq(gzip.variable_lower_bounds_, raw.variable_lower_bounds_, "gzip lower bounds");
  expect_vector_eq(bzip2.variable_lower_bounds_, raw.variable_lower_bounds_, "bzip2 lower bounds");
  expect_vector_eq(gzip.variable_upper_bounds_, raw.variable_upper_bounds_, "gzip upper bounds");
  expect_vector_eq(bzip2.variable_upper_bounds_, raw.variable_upper_bounds_, "bzip2 upper bounds");
  expect_vector_eq(gzip.var_types_, raw.var_types_, "gzip var types");
  expect_vector_eq(bzip2.var_types_, raw.var_types_, "bzip2 var types");
}

}  // namespace

int main()
{
  struct TestCase {
    const char* name;
    void (*fn)();
  };

  const TestCase tests[] = {
    {"ScannerFindsSectionSplitAcrossBlocks", scanner_finds_section_split_across_blocks},
    {"ScannerFindsHeadersSplitAtEveryByte", scanner_finds_headers_split_at_every_byte},
    {"ScannerRejectsUnknownColumnOneRecordsAfterRows",
     scanner_rejects_unknown_column_one_records_after_rows},
    {"BoundsDefaultsAndTypesMatchReference", bounds_defaults_and_types_match_reference},
    {"DuplicateBoundsLastStatementWins", duplicate_bounds_last_statement_wins},
    {"NondenseRowAndColumnNamesUseHashPath", nondense_row_and_column_names_use_hash_path},
    {"MissingOptionalBoundsFastPath", missing_optional_bounds_fast_path},
    {"BoundsOnlyVariablesAreAppendedDeterministically",
     bounds_only_variables_are_appended_deterministically},
    {"IntegerMarkersAssignTypesAndDefaultBounds", integer_markers_assign_types_and_default_bounds},
    {"NumericParsingIntegrationMatchesReferenceBitwise",
     numeric_parsing_integration_matches_reference_bitwise},
    {"CrlfLineEndingsMatchReferenceBitwise", crlf_line_endings_match_reference_bitwise},
    {"CommentPlacementSupportedCasesMatchReferenceBitwise",
     comment_placement_supported_cases_match_reference_bitwise},
    {"ObjectiveMetadataSelectsNamedObjective", objective_metadata_selects_named_objective},
    {"MalformedInputsReportErrors", malformed_inputs_report_errors},
    {"LargeColumnsRepeatedColumnChunkBoundary", large_columns_repeated_column_chunk_boundary},
    {"LargeBoundsRepeatedVarStaysOrdered", large_bounds_repeated_var_stays_ordered},
    {"Lz4AndRawPathsMatchOnMultiblockInput", lz4_and_raw_paths_match_on_multiblock_input},
    {"GzipBzip2AndRawPathsMatch", gzip_bzip2_and_raw_paths_match},
  };

  int failed = 0;
  for (const TestCase& test : tests) {
    std::cout << "[ RUN      ] " << test.name << '\n';
    try {
      test.fn();
      std::cout << "[       OK ] " << test.name << '\n';
    } catch (const skip_test& e) {
      std::cout << "[  SKIPPED ] " << test.name << ": " << e.what() << '\n';
    } catch (const std::exception& e) {
      ++failed;
      std::cerr << "[  FAILED  ] " << test.name << ": " << e.what() << '\n';
    }
  }

  if (failed != 0) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << "[  PASSED  ] " << std::size(tests) << " test(s)\n";
  return 0;
}
