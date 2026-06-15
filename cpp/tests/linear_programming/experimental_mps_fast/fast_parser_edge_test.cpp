// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fast_parser.hpp"
#include "mps_section_scanner.hpp"

#include <cuopt/linear_programming/io/parser.hpp>
#include <mps_parser_internal.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <unistd.h>

namespace cuopt::linear_programming::io::detail {

namespace {

struct TempMpsFile {
  explicit TempMpsFile(std::string contents)
  {
    char path_template[128];
    std::snprintf(path_template,
                  sizeof(path_template),
                  "/tmp/mps_fast_parser_edge_%ld_XXXXXX.mps",
                  static_cast<long>(getpid()));
    int fd = mkstemps(path_template, 4);
    if (fd < 0) {
      throw std::runtime_error(std::string("mkstemps failed: ") + std::strerror(errno));
    }
    path       = path_template;
    FILE* file = fdopen(fd, "wb");
    if (file == nullptr) {
      close(fd);
      throw std::runtime_error(std::string("fdopen failed: ") + std::strerror(errno));
    }
    if (!contents.empty() &&
        std::fwrite(contents.data(), 1, contents.size(), file) != contents.size()) {
      std::fclose(file);
      throw std::runtime_error(std::string("failed to write temporary MPS file: ") +
                               std::strerror(errno));
    }
    if (std::fclose(file) != 0) {
      throw std::runtime_error(std::string("failed to close temporary MPS file: ") +
                               std::strerror(errno));
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

std::string_view range_text(const mps_phase_range_t& range)
{
  if (!range.present) { return {}; }
  return std::string_view(range.begin, static_cast<size_t>(range.end - range.begin));
}

uint64_t bits(double value) { return std::bit_cast<uint64_t>(value); }

template <typename T>
void expect_vectors_bitwise_equal(const std::vector<T>& reference,
                                  const std::vector<T>& fast,
                                  std::string_view field,
                                  std::string_view context)
{
  static_assert(std::is_trivially_copyable_v<T>);
  SCOPED_TRACE(std::string(context) + " " + std::string(field));
  ASSERT_EQ(reference.size(), fast.size()) << "size";
  if (reference.empty()) { return; }
  EXPECT_EQ(0, std::memcmp(reference.data(), fast.data(), reference.size() * sizeof(T)));
}

void check_models_match_reference_bitwise(const parser_model_t<int, double>& fast,
                                          const mps_data_model_t<int, double>& reference,
                                          std::string_view context)
{
  EXPECT_EQ(reference.n_vars_, fast.n_vars_) << std::string(context) + " n_vars";
  EXPECT_EQ(reference.n_constraints_, fast.n_constraints_)
    << std::string(context) + " n_constraints";
  EXPECT_EQ(reference.get_nnz(), fast.get_nnz()) << std::string(context) + " nnz";
  EXPECT_EQ(reference.maximize_, fast.maximize_) << std::string(context) + " maximize";
  EXPECT_EQ(reference.problem_name_, fast.problem_name_) << std::string(context) + " problem_name";
  EXPECT_EQ(reference.objective_name_, fast.objective_name_)
    << std::string(context) + " objective_name";

  EXPECT_EQ(bits(reference.objective_scaling_factor_), bits(fast.objective_scaling_factor_))
    << std::string(context) + " objective_scaling_factor";
  EXPECT_EQ(bits(reference.objective_offset_), bits(fast.objective_offset_))
    << std::string(context) + " objective_offset";

  expect_vectors_bitwise_equal(reference.A_, fast.A_, "A", context);
  EXPECT_EQ(reference.A_indices_, fast.A_indices_) << std::string(context) + " A_indices";
  EXPECT_EQ(reference.A_offsets_, fast.A_offsets_) << std::string(context) + " A_offsets";
  expect_vectors_bitwise_equal(reference.b_, fast.b_, "b", context);
  expect_vectors_bitwise_equal(reference.c_, fast.c_, "c", context);
  expect_vectors_bitwise_equal(reference.variable_lower_bounds_,
                               fast.variable_lower_bounds_,
                               "variable_lower_bounds",
                               context);
  expect_vectors_bitwise_equal(reference.variable_upper_bounds_,
                               fast.variable_upper_bounds_,
                               "variable_upper_bounds",
                               context);
  expect_vectors_bitwise_equal(reference.constraint_lower_bounds_,
                               fast.constraint_lower_bounds_,
                               "constraint_lower_bounds",
                               context);
  expect_vectors_bitwise_equal(reference.constraint_upper_bounds_,
                               fast.constraint_upper_bounds_,
                               "constraint_upper_bounds",
                               context);
  EXPECT_EQ(reference.var_types_, fast.var_types_) << std::string(context) + " var_types";
  EXPECT_EQ(reference.row_types_, fast.row_types_) << std::string(context) + " row_types";
  EXPECT_EQ(reference.var_names_, fast.var_names_) << std::string(context) + " var_names";
  EXPECT_EQ(reference.row_names_, fast.row_names_) << std::string(context) + " row_names";

  ASSERT_EQ(reference.quadratic_constraints_.size(), fast.quadratic_constraints_.size())
    << std::string(context) + " quadratic_constraints size";
  for (size_t q = 0; q < reference.quadratic_constraints_.size(); ++q) {
    const auto& ref_qc  = reference.quadratic_constraints_[q];
    const auto& fast_qc = fast.quadratic_constraints_[q];
    SCOPED_TRACE(std::string(context) + " quadratic_constraint " + std::to_string(q));
    EXPECT_EQ(ref_qc.constraint_row_index, fast_qc.constraint_row_index);
    EXPECT_EQ(ref_qc.constraint_row_name, fast_qc.constraint_row_name);
    EXPECT_EQ(ref_qc.constraint_row_type, fast_qc.constraint_row_type);
    EXPECT_EQ(bits(ref_qc.rhs_value), bits(fast_qc.rhs_value));
    expect_vectors_bitwise_equal(
      ref_qc.linear_values, fast_qc.linear_values, "linear_values", context);
    EXPECT_EQ(ref_qc.linear_indices, fast_qc.linear_indices);
    expect_vectors_bitwise_equal(ref_qc.vals, fast_qc.vals, "qc_vals", context);
    EXPECT_EQ(ref_qc.rows, fast_qc.rows);
    EXPECT_EQ(ref_qc.cols, fast_qc.cols);
  }
}

mps_data_model_t<int, double> parse_reference_model(const std::string& path)
{
  mps_data_model_t<int, double> reference;
  mps_parser_t<int, double> parser(reference, path, false);
  return reference;
}

void verify_fixture_bitwise(std::string_view fixture_name, std::string contents)
{
  TempMpsFile file(std::move(contents));
  auto fast      = parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read);
  auto reference = parse_reference_model(file.path);
  check_models_match_reference_bitwise(fast, reference, fixture_name);
}

std::string row_name(size_t i)
{
  std::ostringstream out;
  out << 'R' << std::setw(6) << std::setfill('0') << i;
  return out.str();
}

int find_var_index(const parser_model_t<int, double>& model, std::string_view name)
{
  for (size_t i = 0; i < model.var_names_.size(); ++i) {
    if (model.var_names_[i] == name) { return static_cast<int>(i); }
  }
  return -1;
}

void check_model_shapes(
  const parser_model_t<int, double>& model, int rows, int vars, int nnz, std::string_view context)
{
  EXPECT_EQ(rows, model.n_constraints_) << std::string(context) + " rows";
  EXPECT_EQ(vars, model.n_vars_) << std::string(context) + " vars";
  EXPECT_EQ(nnz, model.nnz_) << std::string(context) + " nnz";
  EXPECT_EQ(static_cast<size_t>(rows + 1), model.A_offsets_.size())
    << std::string(context) + " offsets";
  EXPECT_EQ(static_cast<size_t>(nnz), model.A_.size()) << std::string(context) + " values";
  EXPECT_EQ(static_cast<size_t>(nnz), model.A_indices_.size()) << std::string(context) + " indices";
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

}  // namespace

TEST(FastMpsParserEdgeTest, ScannerFindsSectionSplitAcrossBlocks)
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
  EXPECT_TRUE(columns_pos != std::string::npos) << "failed to place COLUMNS split";
  const size_t split = columns_pos + 3;

  mps_phase_registry_t registry;
  mps_section_block_scanner_t scanner(mps.data(), 2, registry);

  scanner.observe_block(1, mps.data() + split, mps.data() + mps.size());
  scanner.publish_ready(0);
  scanner.observe_block(0, mps.data(), mps.data() + split);
  scanner.publish_ready(mps.size());

  EXPECT_TRUE(registry.ready(mps_phase_kind::header)) << "header not ready";
  EXPECT_TRUE(registry.ready(mps_phase_kind::rows)) << "rows not ready";
  EXPECT_TRUE(registry.ready(mps_phase_kind::columns)) << "columns not ready";
  EXPECT_TRUE(registry.ready(mps_phase_kind::rhs)) << "rhs not ready";
  EXPECT_TRUE(registry.ready(mps_phase_kind::quadratic)) << "quadratic sentinel not ready";

  EXPECT_TRUE(range_text(registry.range(mps_phase_kind::columns)).starts_with("COLUMNS"))
    << "columns range begins at wrong boundary";
  EXPECT_TRUE(range_text(registry.range(mps_phase_kind::rhs)).starts_with("RHS"))
    << "rhs range begins at wrong boundary";
}

TEST(FastMpsParserEdgeTest, ScannerFindsHeadersSplitAtEveryByte)
{
  const std::string mps                       = section_split_fixture();
  const std::vector<std::string_view> headers = {"ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"};

  for (std::string_view header : headers) {
    const size_t pos = mps.find(header);
    EXPECT_TRUE(pos != std::string::npos) << "missing header in split fixture";
    for (size_t offset = 1; offset < header.size(); ++offset) {
      const size_t split = pos + offset;
      mps_phase_registry_t registry;
      mps_section_block_scanner_t scanner(mps.data(), 2, registry);

      scanner.observe_block(1, mps.data() + split, mps.data() + mps.size());
      scanner.observe_block(0, mps.data(), mps.data() + split);
      scanner.publish_ready(mps.size());

      EXPECT_TRUE(registry.ready(mps_phase_kind::rows)) << "rows not ready after split";
      EXPECT_TRUE(registry.ready(mps_phase_kind::columns)) << "columns not ready after split";
      EXPECT_TRUE(registry.ready(mps_phase_kind::rhs)) << "rhs not ready after split";
      EXPECT_TRUE(registry.ready(mps_phase_kind::bounds)) << "bounds not ready after split";
      EXPECT_TRUE(registry.ready(mps_phase_kind::quadratic))
        << "quadratic sentinel not ready after split";
    }
  }
}

TEST(FastMpsParserEdgeTest, ScannerRejectsUnknownColumnOneRecordsAfterRows)
{
  const std::string mps =
    "NAME BAD\n"
    "ROWS\n"
    " N OBJ\n"
    "FOO\n"
    "COLUMNS\n"
    " x OBJ 1\n"
    "ENDATA\n";

  EXPECT_THROW(
    {
      mps_phase_registry_t registry;
      mps_section_block_scanner_t scanner(mps.data(), 1, registry);
      scanner.observe_block(0, mps.data(), mps.data() + mps.size());
      scanner.publish_ready(mps.size());
    },
    std::logic_error);
}

TEST(FastMpsParserEdgeTest, ParserRejectsUnknownSectionRecords)
{
  TempMpsFile file(
    "NAME BAD_UNKNOWN_SECTION\n"
    "ROWS\n"
    " N OBJ\n"
    " L R1\n"
    "COLUMNS\n"
    " X1 OBJ 1 R1 2\n"
    "RHS\n"
    " RHS1 R1 3\n"
    "BOUNDS\n"
    " FR BND1 X1\n"
    "QSECTION      R1\n"
    " X1 X1 1\n"
    "ENDATA\n");

  EXPECT_THROW(((void)parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read)),
               std::exception);
}

TEST(FastMpsParserEdgeTest, BoundsDefaultsAndTypesMatchReference)
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

TEST(FastMpsParserEdgeTest, DuplicateBoundsLastStatementWins)
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
  auto model = parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read);
  EXPECT_EQ(1, model.n_vars_) << "n_vars";
  EXPECT_EQ(2.0, model.variable_lower_bounds_.at(0)) << "duplicate lower bound";
  EXPECT_EQ(3.0, model.variable_upper_bounds_.at(0)) << "duplicate upper bound";
}

TEST(FastMpsParserEdgeTest, NondenseRowAndColumnNamesUseHashPath)
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

TEST(FastMpsParserEdgeTest, MissingOptionalBoundsFastPath)
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

  auto model = parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read);
  EXPECT_EQ(1, model.n_vars_) << "missing optional n_vars";
  EXPECT_EQ(1, model.n_constraints_) << "missing optional n_constraints";
  EXPECT_EQ(0.0, model.variable_lower_bounds_.at(0)) << "missing BOUNDS lower default";
  EXPECT_EQ(std::numeric_limits<double>::infinity(), model.variable_upper_bounds_.at(0));
}

TEST(FastMpsParserEdgeTest, BoundsOnlyVariablesAreAppendedDeterministically)
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

  auto model = parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read);
  check_model_shapes(model, 1, 4, 1, "bounds-only");
  EXPECT_EQ(std::string("XMAIN"), model.var_names_.at(0)) << "main var name";
  EXPECT_EQ(std::string("AUX_A"), model.var_names_.at(1)) << "bounds-only sorted name 1";
  EXPECT_EQ(std::string("AUX_S"), model.var_names_.at(2)) << "bounds-only sorted name 2";
  EXPECT_EQ(std::string("AUX_Z"), model.var_names_.at(3)) << "bounds-only sorted name 3";

  const int aux_a = find_var_index(model, "AUX_A");
  const int aux_s = find_var_index(model, "AUX_S");
  const int aux_z = find_var_index(model, "AUX_Z");
  ASSERT_GE(aux_a, 0);
  ASSERT_GE(aux_s, 0);
  ASSERT_GE(aux_z, 0);
  EXPECT_EQ('I', model.var_types_.at(aux_a)) << "bounds-only BV type";
  EXPECT_EQ(0.0, model.variable_lower_bounds_.at(aux_a)) << "bounds-only BV lb";
  EXPECT_EQ(1.0, model.variable_upper_bounds_.at(aux_a)) << "bounds-only BV ub";
  EXPECT_EQ('S', model.var_types_.at(aux_s)) << "bounds-only SC type";
  EXPECT_EQ(5.0, model.variable_upper_bounds_.at(aux_s)) << "bounds-only SC ub";
  EXPECT_EQ(-3.0, model.variable_lower_bounds_.at(aux_z)) << "bounds-only duplicate lb";
  EXPECT_EQ(9.0, model.variable_upper_bounds_.at(aux_z)) << "bounds-only duplicate ub";
}

TEST(FastMpsParserEdgeTest, IntegerMarkersAssignTypesAndDefaultBounds)
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

  auto model = parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read);
  check_model_shapes(model, 1, 3, 3, "integer markers");
  const int xint  = find_var_index(model, "XINT");
  const int xcont = find_var_index(model, "XCONT");
  const int xbin  = find_var_index(model, "XBIN");
  ASSERT_GE(xint, 0);
  ASSERT_GE(xcont, 0);
  ASSERT_GE(xbin, 0);
  EXPECT_EQ('I', model.var_types_.at(xint)) << "XINT type";
  EXPECT_EQ('C', model.var_types_.at(xcont)) << "XCONT type";
  EXPECT_EQ('I', model.var_types_.at(xbin)) << "XBIN type";
  EXPECT_EQ(0.0, model.variable_lower_bounds_.at(xint)) << "XINT default lb";
  EXPECT_EQ(1.0, model.variable_upper_bounds_.at(xint)) << "XINT default ub";
  EXPECT_EQ(0.0, model.variable_lower_bounds_.at(xbin)) << "XBIN default lb";
  EXPECT_EQ(1.0, model.variable_upper_bounds_.at(xbin)) << "XBIN default ub";
}

TEST(FastMpsParserEdgeTest, NumericParsingIntegrationMatchesReferenceBitwise)
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

TEST(FastMpsParserEdgeTest, CrlfLineEndingsMatchReferenceBitwise)
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

TEST(FastMpsParserEdgeTest, CommentPlacementSupportedCasesMatchReferenceBitwise)
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

TEST(FastMpsParserEdgeTest, ObjectiveMetadataSelectsNamedObjective)
{
  verify_fixture_bitwise("objective_metadata",
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
}

TEST(FastMpsParserEdgeTest, MalformedInputsReportErrors)
{
  {
    TempMpsFile file(
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
    EXPECT_THROW(((void)parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read)),
                 std::logic_error);
  }

  {
    TempMpsFile file(
      "NAME BADCOLROW\n"
      "ROWS\n"
      " N OBJ\n"
      " L R1\n"
      "COLUMNS\n"
      " X1 MISSING 1\n"
      "RHS\n"
      " RHS1 R1 0\n"
      "ENDATA\n");
    EXPECT_THROW(((void)parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read)),
                 std::logic_error);
  }

  {
    TempMpsFile file(
      "NAME BADRHSROW\n"
      "ROWS\n"
      " N OBJ\n"
      " L R1\n"
      "COLUMNS\n"
      " X1 OBJ 1 R1 2\n"
      "RHS\n"
      " RHS1 MISSING 1\n"
      "ENDATA\n");
    EXPECT_THROW(((void)parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read)),
                 std::logic_error);
  }

  {
    TempMpsFile file(
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
    EXPECT_THROW(((void)parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read)),
                 std::logic_error);
  }

  {
    TempMpsFile file(
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
    EXPECT_THROW(((void)parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read)),
                 std::logic_error);
  }
}

TEST(FastMpsParserEdgeTest, LargeColumnsRepeatedColumnChunkBoundary)
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
  auto model = parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read);
  check_model_shapes(
    model, static_cast<int>(row_count), 2, static_cast<int>(row_count + 1), "large columns");
  EXPECT_EQ(std::string("XBIG"), model.var_names_.at(0)) << "large repeated column name";
  EXPECT_EQ(std::string("XTAIL"), model.var_names_.at(1)) << "large tail column name";
}

TEST(FastMpsParserEdgeTest, LargeBoundsRepeatedVarStaysOrdered)
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
  auto model = parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read);
  check_model_shapes(model, 1, 1, 1, "large bounds");
  EXPECT_EQ(static_cast<double>((repeat_count - 1) % 1000), model.variable_upper_bounds_.at(0))
    << "large repeated bounds last value";
}

TEST(FastMpsParserEdgeTest, Lz4AndRawPathsMatchOnMultiblockInput)
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
  if (std::system(cmd.c_str()) != 0) { GTEST_SKIP() << "lz4 CLI unavailable"; }

  auto raw = parse_mps_fast_file<int, double>(raw_file.path, FileReadMethod::Read);
  auto lz4 = parse_mps_fast_file<int, double>(lz4_file.path, FileReadMethod::Read);

  check_model_shapes(lz4, raw.n_constraints_, raw.n_vars_, raw.nnz_, "lz4 parity");
  EXPECT_EQ(raw.var_names_.size(), lz4.var_names_.size()) << "lz4 var name count";
  EXPECT_EQ(raw.row_names_.size(), lz4.row_names_.size()) << "lz4 row name count";
  EXPECT_EQ(raw.A_, lz4.A_) << "lz4 A values";
  EXPECT_EQ(raw.A_indices_, lz4.A_indices_) << "lz4 A indices";
  EXPECT_EQ(raw.A_offsets_, lz4.A_offsets_) << "lz4 A offsets";
  EXPECT_EQ(raw.c_, lz4.c_) << "lz4 objective";
  EXPECT_EQ(raw.b_, lz4.b_) << "lz4 rhs";
  EXPECT_EQ(raw.var_types_, lz4.var_types_) << "lz4 var types";
  EXPECT_EQ(raw.variable_lower_bounds_, lz4.variable_lower_bounds_) << "lz4 lower bounds";
  EXPECT_EQ(raw.variable_upper_bounds_, lz4.variable_upper_bounds_) << "lz4 upper bounds";
}

TEST(FastMpsParserEdgeTest, GzipBzip2AndRawPathsMatch)
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
  if (std::system(gzip_cmd.c_str()) != 0) { GTEST_SKIP() << "gzip CLI unavailable"; }
  if (std::system(bzip2_cmd.c_str()) != 0) { GTEST_SKIP() << "bzip2 CLI unavailable"; }

  auto raw   = parse_mps_fast_file<int, double>(raw_file.path, FileReadMethod::Read);
  auto gzip  = parse_mps_fast_file<int, double>(gzip_file.path, FileReadMethod::Read);
  auto bzip2 = parse_mps_fast_file<int, double>(bzip2_file.path, FileReadMethod::Read);

  check_model_shapes(gzip, raw.n_constraints_, raw.n_vars_, raw.nnz_, "gzip parity");
  check_model_shapes(bzip2, raw.n_constraints_, raw.n_vars_, raw.nnz_, "bzip2 parity");
  EXPECT_EQ(raw.A_, gzip.A_) << "gzip A values";
  EXPECT_EQ(raw.A_, bzip2.A_) << "bzip2 A values";
  EXPECT_EQ(raw.A_indices_, gzip.A_indices_) << "gzip A indices";
  EXPECT_EQ(raw.A_indices_, bzip2.A_indices_) << "bzip2 A indices";
  EXPECT_EQ(raw.A_offsets_, gzip.A_offsets_) << "gzip A offsets";
  EXPECT_EQ(raw.A_offsets_, bzip2.A_offsets_) << "bzip2 A offsets";
  EXPECT_EQ(raw.c_, gzip.c_) << "gzip objective";
  EXPECT_EQ(raw.c_, bzip2.c_) << "bzip2 objective";
  EXPECT_EQ(raw.b_, gzip.b_) << "gzip rhs";
  EXPECT_EQ(raw.b_, bzip2.b_) << "bzip2 rhs";
  EXPECT_EQ(raw.variable_lower_bounds_, gzip.variable_lower_bounds_) << "gzip lower bounds";
  EXPECT_EQ(raw.variable_lower_bounds_, bzip2.variable_lower_bounds_) << "bzip2 lower bounds";
  EXPECT_EQ(raw.variable_upper_bounds_, gzip.variable_upper_bounds_) << "gzip upper bounds";
  EXPECT_EQ(raw.variable_upper_bounds_, bzip2.variable_upper_bounds_) << "bzip2 upper bounds";
  EXPECT_EQ(raw.var_types_, gzip.var_types_) << "gzip var types";
  EXPECT_EQ(raw.var_types_, bzip2.var_types_) << "bzip2 var types";
}

TEST(FastMpsParserEdgeTest, QcMatrixRowsMatchReferenceBitwise)
{
  verify_fixture_bitwise("qcmatrix rows",
                         "NAME QCMATRIX_TEST\n"
                         "ROWS\n"
                         " N OBJ\n"
                         " L LIN\n"
                         " L QC1\n"
                         " G QC2\n"
                         "COLUMNS\n"
                         " X1 OBJ 1 LIN 2\n"
                         " X1 QC1 3 QC2 4\n"
                         " X2 OBJ 2 LIN 5\n"
                         " X2 QC1 6 QC2 7\n"
                         "RHS\n"
                         " RHS1 LIN 10 QC1 11\n"
                         " RHS1 QC2 12\n"
                         "QCMATRIX   QC1\n"
                         " X1 X1 1.25\n"
                         " X1 X2 -2.5\n"
                         "QCMATRIX   QC2\n"
                         " X2 X2 3.75\n"
                         "ENDATA\n");
}

TEST(FastMpsParserEdgeTest, QcMatrixMalformedCasesMatchReference)
{
  const std::vector<std::string> cases = {
    "NAME DUP_QC\n"
    "ROWS\n"
    " N OBJ\n"
    " L QC1\n"
    "COLUMNS\n"
    " X1 OBJ 1 QC1 2\n"
    "RHS\n"
    " RHS1 QC1 3\n"
    "QCMATRIX QC1\n"
    " X1 X1 1\n"
    "QCMATRIX QC1\n"
    " X1 X1 2\n"
    "ENDATA\n",
    "NAME BAD_QC_ROW\n"
    "ROWS\n"
    " N OBJ\n"
    " L QC1\n"
    "COLUMNS\n"
    " X1 OBJ 1 QC1 2\n"
    "RHS\n"
    " RHS1 QC1 3\n"
    "QCMATRIX UNKNOWN\n"
    " X1 X1 1\n"
    "ENDATA\n",
    "NAME BAD_QC_VAR\n"
    "ROWS\n"
    " N OBJ\n"
    " L QC1\n"
    "COLUMNS\n"
    " X1 OBJ 1 QC1 2\n"
    "RHS\n"
    " RHS1 QC1 3\n"
    "QCMATRIX QC1\n"
    " X1 XBAD 1\n"
    "ENDATA\n"};

  for (const auto& mps : cases) {
    TempMpsFile file(mps);
    EXPECT_THROW(((void)parse_reference_model(file.path)), std::exception);
    EXPECT_THROW(((void)parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read)),
                 std::exception);
  }
}

TEST(FastMpsParserEdgeTest, QuadraticParserRejectsUnknownColumnOneRecords)
{
  const std::vector<std::string> records = {"QSECTION      QC1",
                                            "CSECTION      QC1        0              QUAD"};

  for (const auto& record : records) {
    TempMpsFile file(
      "NAME BAD_QUAD_RECORD\n"
      "ROWS\n"
      " N OBJ\n"
      " L QC1\n"
      "COLUMNS\n"
      " X1 OBJ 1 QC1 2\n"
      " X2 OBJ 3 QC1 4\n"
      "RHS\n"
      " RHS1 QC1 5\n"
      "QMATRIX\n"
      " X1 X1 1\n" +
      record +
      "\n"
      " X2 X2 2\n"
      "ENDATA\n");
    EXPECT_THROW(((void)parse_mps_fast_file<int, double>(file.path, FileReadMethod::Read)),
                 std::exception)
      << record;
  }
}

}  // namespace cuopt::linear_programming::io::detail
