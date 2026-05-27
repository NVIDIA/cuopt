/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/io/parser.hpp>

#include <file_to_string.hpp>
#include <lp_parser.hpp>
#include <utilities/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::io {

namespace {

// ===========================================================================
// Small character / string helpers
// ===========================================================================

// Per the LP-format convention, variable names may use letters and a specific
// set of punctuation characters. Characters used by the grammar (+, -, *, ^,
// :, =, <, >, [, ], \, whitespace) are excluded. Digits and '.' are valid
// mid-name but not as the starting character.
bool is_name_start_char(char c)
{
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
  switch (c) {
    case '!':
    case '"':
    case '#':
    case '$':
    case '%':
    case '&':
    case '(':
    case ')':
    case ',':
    case ';':
    case '?':
    case '@':
    case '_':
    case '`':
    case '\'':
    case '{':
    case '}':
    case '|':
    case '~': return true;
    default: return false;
  }
}

bool is_name_char(char c)
{
  if (is_name_start_char(c)) return true;
  return (c >= '0' && c <= '9') || c == '.' || c == '/';
}

char to_lower(char c)
{
  if (c >= 'A' && c <= 'Z') return static_cast<char>(c - 'A' + 'a');
  return c;
}

std::string lowercase(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out.push_back(to_lower(c));
  return out;
}

// ===========================================================================
// LP section-keyword classifiers (case-insensitive; callers pass lowercased)
// ===========================================================================

bool is_objective_min_keyword(std::string_view lower)
{
  return lower == "minimize" || lower == "minimum" || lower == "min";
}

bool is_objective_max_keyword(std::string_view lower)
{
  return lower == "maximize" || lower == "maximum" || lower == "max";
}

bool is_bounds_keyword(std::string_view lower) { return lower == "bounds" || lower == "bound"; }

bool is_generals_keyword(std::string_view lower)
{
  return lower == "generals" || lower == "general" || lower == "gen" || lower == "integer" ||
         lower == "integers";
}

bool is_binaries_keyword(std::string_view lower)
{
  return lower == "binaries" || lower == "binary" || lower == "bin";
}

bool is_end_keyword(std::string_view lower) { return lower == "end"; }

bool is_free_keyword(std::string_view lower) { return lower == "free"; }

bool is_infinity_text(std::string_view lower) { return lower == "inf" || lower == "infinity"; }

// Builds the symmetric Q in CSR from LP-format raw upper-triangular triples.
// Each input triple (i, j, c) with i <= j represents `c * x_i * x_j` in the
// LP source. The output Q satisfies x^T Q x = sum of those terms.
//   Diagonal (i == j): Q[i,i] = c (one entry).
//   Off-diagonal (i != j): Q[i,j] = Q[j,i] = c/2 (two entries; symmetric split).
template <typename i_t, typename f_t>
void build_symmetric_q_csr(const std::vector<std::tuple<i_t, i_t, f_t>>& raw_triples,
                           i_t n_vars,
                           std::vector<f_t>& out_values,
                           std::vector<i_t>& out_indices,
                           std::vector<i_t>& out_offsets)
{
  std::vector<std::vector<std::pair<i_t, f_t>>> row_data(n_vars);
  for (const auto& [i, j, c] : raw_triples) {
    if (i == j) {
      row_data[i].emplace_back(i, c);
    } else {
      row_data[i].emplace_back(j, c / f_t(2));
      row_data[j].emplace_back(i, c / f_t(2));
    }
  }
  for (auto& row : row_data) {
    std::sort(row.begin(), row.end());
  }
  out_offsets.clear();
  out_indices.clear();
  out_values.clear();
  out_offsets.reserve(static_cast<size_t>(n_vars) + 1);
  out_offsets.push_back(0);
  for (i_t r = 0; r < n_vars; ++r) {
    for (const auto& [col, val] : row_data[r]) {
      out_values.push_back(val);
      out_indices.push_back(col);
    }
    out_offsets.push_back(static_cast<i_t>(out_values.size()));
  }
}

// ===========================================================================
// Token stream
// ===========================================================================

// Kinds of tokens produced by the LP tokenizer. The grammar is small enough
// that a hand-written scanner is easier to follow than a regex engine.
enum class LpTokenKind {
  Number,     // 12, -3.5, 1e-6
  Name,       // variable names and section keywords (also the literal "inf")
  Plus,       // +
  Minus,      // -
  Star,       // *
  Caret,      // ^
  Slash,      // /
  LessEq,     // <= (and < treated as <=)
  GreaterEq,  // >= (and > treated as >=)
  Equal,      // =
  LBracket,   // [
  RBracket,   // ]
  Colon,      // :
  Eof,
};

struct LpToken {
  LpTokenKind kind;
  // Owned copy of the token text so the token stream is independent of the
  // backing file buffer.
  std::string text;
  int line;
  // True when this is the first non-whitespace/non-comment token on its line.
  // Used to detect section headers without emitting newline tokens.
  bool is_line_start;
};

// ===========================================================================
// Parsing engine — holds all transient parsing state and writes directly
// into the lp_parser_t's public fields. Strictly internal to this TU.
// ===========================================================================

template <typename i_t, typename f_t>
class LpParseEngine {
 public:
  LpParseEngine(lp_parser_t<i_t, f_t>& out, const std::string& file);
  // Parses `text` directly (used by read_lp_from_string()).
  LpParseEngine(lp_parser_t<i_t, f_t>& out, std::string_view text);

 private:
  lp_parser_t<i_t, f_t>& out_;
  std::vector<LpToken> tokens_;
  size_t tok_pos_{0};

  std::unordered_map<std::string, i_t> var_names_map_{};
  std::unordered_map<std::string, i_t> row_names_map_{};
  std::unordered_set<i_t> bounds_defined_for_var_id_{};
  // Variables for which a lower bound was set explicitly in the Bounds
  // section (via 'x >= lb', 'x = v', 'x free', or 'lb <= x ...'). Used to
  // reject 'x <= -1' forms with no paired lower bound: the default lower of
  // 0 would collide with the negative upper and silently make the variable
  // infeasible.
  std::unordered_set<i_t> lower_explicitly_set_{};
  // Counter used to generate row names for unlabeled constraints (R0, R1, ...).
  i_t anon_row_counter_{0};

  // File → token stream.
  void read_and_tokenize(const std::string& file);
  void tokenize(const std::string& text);

  // Token stream helpers.
  const LpToken& peek(size_t lookahead = 0) const;
  const LpToken& advance();
  bool at_eof() const;
  bool match(LpTokenKind kind);
  void expect(LpTokenKind kind, const char* context);
  static bool name_equals_ci(const LpToken& tok, std::string_view lower);
  bool is_infinity_keyword(const LpToken& tok) const;
  f_t number_from_text(const std::string& text) const;

  // Variable bookkeeping.
  i_t get_or_add_var(std::string_view name);

  // Top-level dispatch.
  void parse_all();

  // Section parsers.
  void parse_objective_section();
  void parse_constraints_section();
  void parse_bounds_section();
  void parse_integer_list_section(bool is_binary);
  void parse_semi_continuous_section();

  // Expression parsers.
  struct LinearTerm {
    i_t var_id;
    f_t coeff;
  };
  void parse_linear_expression(std::vector<LinearTerm>& out_terms, f_t& out_constant);

  // Where a quadratic '[ ... ]' bracket appears. The two roles differ in
  // post-processing:
  //   Objective:  must be followed by '/ 2'; the inner-coefficient convention
  //               is QUADOBJ-style 0.5 x^T Q x, so off-diagonals are halved
  //               and linear-inside-bracket terms also get /2.
  //   Constraint: must NOT be followed by '/ 2'; coefficients are taken at
  //               face value (x^T Q x); bracket must contain at least one
  //               quadratic term.
  enum class BracketRole { Objective, Constraint };
  void parse_quadratic_bracket(int outer_sign,
                               BracketRole role,
                               std::vector<std::tuple<i_t, i_t, f_t>>& out_quad_entries);

  // Atomic readers.
  f_t parse_signed_number();

  // Section header classification.
  enum class SectionKind {
    None,
    Objective,
    Constraints,
    Bounds,
    Generals,
    Binaries,
    SemiContinuous,
    End,
  };
  SectionKind try_consume_section_header();
  void reject_unsupported_section();
  bool at_section_boundary() const;
};

// ---- Constructor ----------------------------------------------------------

template <typename i_t, typename f_t>
LpParseEngine<i_t, f_t>::LpParseEngine(lp_parser_t<i_t, f_t>& out, const std::string& file)
  : out_(out)
{
  read_and_tokenize(file);
  parse_all();
}

template <typename i_t, typename f_t>
LpParseEngine<i_t, f_t>::LpParseEngine(lp_parser_t<i_t, f_t>& out, std::string_view text)
  : out_(out)
{
  // Skip read_and_tokenize: the caller already supplied the LP text.
  // Make a contiguous null-terminated string for tokenize().
  std::string buffered(text);
  tokenize(buffered);
  parse_all();
}

// ---- File I/O + tokenizer -------------------------------------------------

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::read_and_tokenize(const std::string& file)
{
  // Delegates to the shared helper so .lp.gz / .lp.bz2 are handled the same
  // way as .mps.gz / .mps.bz2 (dlopen-loaded libz / libbz2). The returned
  // buffer is null-terminated; strip it before constructing the string view
  // since `tokenize` walks the entire string range.
  auto buf = detail::file_to_string(file);
  std::string text(buf.data(), buf.size() > 0 ? buf.size() - 1 : 0);
  tokenize(text);
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::tokenize(const std::string& text)
{
  size_t i       = 0;
  int line       = 1;
  bool at_start  = true;  // next non-whitespace token starts a new line
  const size_t n = text.size();

  auto push = [&](LpTokenKind kind, std::string s) {
    tokens_.push_back(LpToken{kind, std::move(s), line, at_start});
    at_start = false;
  };

  while (i < n) {
    char c = text[i];

    if (c == '\n') {
      ++line;
      at_start = true;
      ++i;
      continue;
    }
    if (c == '\r') {
      ++i;
      continue;
    }
    if (c == '\\') {  // LP comment: '\' through end of line
      while (i < n && text[i] != '\n')
        ++i;
      continue;
    }
    if (c == ' ' || c == '\t') {
      ++i;
      continue;
    }

    // Single-character punctuation.
    switch (c) {
      case '+':
        push(LpTokenKind::Plus, "+");
        ++i;
        continue;
      case '-':
        push(LpTokenKind::Minus, "-");
        ++i;
        continue;
      case '*':
        push(LpTokenKind::Star, "*");
        ++i;
        continue;
      case '^':
        push(LpTokenKind::Caret, "^");
        ++i;
        continue;
      case '/':
        push(LpTokenKind::Slash, "/");
        ++i;
        continue;
      case '[':
        push(LpTokenKind::LBracket, "[");
        ++i;
        continue;
      case ']':
        push(LpTokenKind::RBracket, "]");
        ++i;
        continue;
      case ':':
        push(LpTokenKind::Colon, ":");
        ++i;
        continue;
      case '=':
        // Accept the swapped spellings: '=<' ≡ '<=' and '=>' ≡ '>='.
        if (i + 1 < n && text[i + 1] == '<') {
          push(LpTokenKind::LessEq, "=<");
          i += 2;
        } else if (i + 1 < n && text[i + 1] == '>') {
          push(LpTokenKind::GreaterEq, "=>");
          i += 2;
        } else {
          push(LpTokenKind::Equal, "=");
          ++i;
        }
        continue;
      default: break;
    }

    // Relation operators. Our LP dialect treats bare '<' as '<=' and bare
    // '>' as '>='; we do the same for robustness.
    if (c == '<') {
      if (i + 1 < n && text[i + 1] == '=') {
        push(LpTokenKind::LessEq, "<=");
        i += 2;
      } else {
        push(LpTokenKind::LessEq, "<");
        ++i;
      }
      continue;
    }
    if (c == '>') {
      if (i + 1 < n && text[i + 1] == '=') {
        push(LpTokenKind::GreaterEq, ">=");
        i += 2;
      } else {
        push(LpTokenKind::GreaterEq, ">");
        ++i;
      }
      continue;
    }

    // Numbers: [0-9]+ ('.' [0-9]*)? ([eE] [+-]? [0-9]+)?  |  '.' [0-9]+ ...
    if ((c >= '0' && c <= '9') ||
        (c == '.' && i + 1 < n && text[i + 1] >= '0' && text[i + 1] <= '9')) {
      size_t start = i;
      while (i < n && text[i] >= '0' && text[i] <= '9')
        ++i;
      if (i < n && text[i] == '.') {
        ++i;
        while (i < n && text[i] >= '0' && text[i] <= '9')
          ++i;
      }
      if (i < n && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        if (i < n && (text[i] == '+' || text[i] == '-')) ++i;
        mps_parser_expects(i < n && text[i] >= '0' && text[i] <= '9',
                           error_type_t::ValidationError,
                           "Malformed number (missing exponent digits) at line %d",
                           line);
        while (i < n && text[i] >= '0' && text[i] <= '9')
          ++i;
      }
      push(LpTokenKind::Number, text.substr(start, i - start));
      continue;
    }

    // Names: [A-Za-z_] [A-Za-z0-9_.]*
    if (is_name_start_char(c)) {
      size_t start = i;
      while (i < n && is_name_char(text[i]))
        ++i;
      push(LpTokenKind::Name, text.substr(start, i - start));
      continue;
    }

    mps_parser_expects(false,
                       error_type_t::ValidationError,
                       "Unexpected character '%c' (0x%02x) at line %d in LP file",
                       c,
                       static_cast<unsigned>(static_cast<unsigned char>(c)),
                       line);
  }

  tokens_.push_back(LpToken{LpTokenKind::Eof, "", line, true});
}

// ---- Token stream helpers --------------------------------------------------

template <typename i_t, typename f_t>
const LpToken& LpParseEngine<i_t, f_t>::peek(size_t lookahead) const
{
  size_t idx = tok_pos_ + lookahead;
  if (idx >= tokens_.size()) return tokens_.back();  // guaranteed Eof token
  return tokens_[idx];
}

template <typename i_t, typename f_t>
const LpToken& LpParseEngine<i_t, f_t>::advance()
{
  const LpToken& t = tokens_[tok_pos_];
  if (tok_pos_ + 1 < tokens_.size()) ++tok_pos_;
  return t;
}

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::at_eof() const
{
  return peek().kind == LpTokenKind::Eof;
}

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::match(LpTokenKind kind)
{
  if (peek().kind == kind) {
    advance();
    return true;
  }
  return false;
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::expect(LpTokenKind kind, const char* context)
{
  mps_parser_expects(peek().kind == kind,
                     error_type_t::ValidationError,
                     "LP parse error at line %d: expected %s, got '%s'",
                     peek().line,
                     context,
                     peek().text.c_str());
  advance();
}

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::name_equals_ci(const LpToken& tok, std::string_view lower)
{
  if (tok.kind != LpTokenKind::Name) return false;
  if (tok.text.size() != lower.size()) return false;
  for (size_t i = 0; i < tok.text.size(); ++i) {
    if (to_lower(tok.text[i]) != lower[i]) return false;
  }
  return true;
}

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::is_infinity_keyword(const LpToken& tok) const
{
  return tok.kind == LpTokenKind::Name && is_infinity_text(lowercase(tok.text));
}

template <typename i_t, typename f_t>
f_t LpParseEngine<i_t, f_t>::number_from_text(const std::string& text) const
{
  try {
    if constexpr (std::is_same_v<f_t, float>) {
      return std::stof(text);
    } else {
      return std::stod(text);
    }
  } catch (...) {
    mps_parser_expects(false,
                       error_type_t::ValidationError,
                       "LP parse error: could not parse number '%s'",
                       text.c_str());
  }
  return f_t(0);  // unreachable; mps_parser_expects throws
}

// ---- Variable bookkeeping --------------------------------------------------

template <typename i_t, typename f_t>
i_t LpParseEngine<i_t, f_t>::get_or_add_var(std::string_view name)
{
  std::string key(name);
  auto it = var_names_map_.find(key);
  if (it != var_names_map_.end()) return it->second;
  i_t id = static_cast<i_t>(out_.var_names.size());
  out_.var_names.push_back(key);
  var_names_map_.emplace(std::move(key), id);
  out_.var_types.push_back('C');
  out_.c_values.push_back(f_t(0));
  out_.variable_lower_bounds.push_back(f_t(0));
  out_.variable_upper_bounds.push_back(std::numeric_limits<f_t>::infinity());
  return id;
}

// ---- Section header detection ---------------------------------------------

template <typename i_t, typename f_t>
bool LpParseEngine<i_t, f_t>::at_section_boundary() const
{
  if (at_eof()) return true;
  const LpToken& t = peek();
  if (!t.is_line_start || t.kind != LpTokenKind::Name) return false;
  std::string lower = lowercase(t.text);

  if (is_objective_min_keyword(lower) || is_objective_max_keyword(lower)) return true;
  if (is_bounds_keyword(lower)) return true;
  if (is_generals_keyword(lower)) return true;
  if (is_binaries_keyword(lower)) return true;
  if (is_end_keyword(lower)) return true;

  // Multi-word section headers: "Subject To" / "Such That" are supported;
  // "Lazy Constraints", "User Cuts", and "General Constraints" are
  // recognized as boundaries so the prior section ends cleanly, but
  // reject_unsupported_section() throws once dispatch reaches them.
  const LpToken& t2 = peek(1);
  if (lower == "subject" && name_equals_ci(t2, "to")) return true;
  if (lower == "such" && name_equals_ci(t2, "that")) return true;
  if (lower == "st" || lower == "st." || lower == "s.t.") return true;
  if (lower == "lazy" && name_equals_ci(t2, "constraints")) return true;
  if (lower == "user" && name_equals_ci(t2, "cuts")) return true;
  if (lower == "general" && name_equals_ci(t2, "constraints")) return true;

  // Semi-Continuous section header (supported); three spellings:
  //   - 3-token "Semi - Continuous"
  //   - bare "Semi"
  //   - bare "Semis"
  // Plus other section headers that we recognize as boundaries (some
  // supported, some unsupported — dispatch decides).
  if (lower == "semi" || lower == "semis") return true;
  if (lower == "sos") return true;
  if (lower == "pwlobj") return true;
  if (lower == "scenarios" || lower == "scenario") return true;

  return false;
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::reject_unsupported_section()
{
  std::string name  = peek().text;
  std::string lower = lowercase(name);
  // Compose a useful display name for multi-word headers.
  if (lower == "user" && name_equals_ci(peek(1), "cuts")) {
    name = "User Cuts";
  } else if (lower == "lazy" && name_equals_ci(peek(1), "constraints")) {
    name = "Lazy Constraints";
  } else if (lower == "general" && name_equals_ci(peek(1), "constraints")) {
    name = "General Constraints";
  }
  mps_parser_expects(false,
                     error_type_t::ValidationError,
                     "LP section '%s' is not supported (scope is LP/MIP/QP only)",
                     name.c_str());
}

template <typename i_t, typename f_t>
typename LpParseEngine<i_t, f_t>::SectionKind LpParseEngine<i_t, f_t>::try_consume_section_header()
{
  if (at_eof()) return SectionKind::None;
  const LpToken& t = peek();
  mps_parser_expects(t.is_line_start && t.kind == LpTokenKind::Name,
                     error_type_t::ValidationError,
                     "LP parse error at line %d: expected section header, got '%s'",
                     t.line,
                     t.text.c_str());
  std::string lower = lowercase(t.text);

  if (is_objective_min_keyword(lower)) {
    out_.maximize = false;
    advance();
    return SectionKind::Objective;
  }
  if (is_objective_max_keyword(lower)) {
    out_.maximize = true;
    advance();
    return SectionKind::Objective;
  }
  if (lower == "subject" && name_equals_ci(peek(1), "to")) {
    advance();
    advance();
    return SectionKind::Constraints;
  }
  if (lower == "such" && name_equals_ci(peek(1), "that")) {
    advance();
    advance();
    return SectionKind::Constraints;
  }
  if (lower == "st" || lower == "st." || lower == "s.t.") {
    advance();
    return SectionKind::Constraints;
  }
  if (is_bounds_keyword(lower)) {
    advance();
    return SectionKind::Bounds;
  }
  if (is_generals_keyword(lower)) {
    // "General" alone means Generals; "General Constraints" is unsupported.
    if (lower == "general" && name_equals_ci(peek(1), "constraints")) {
      reject_unsupported_section();
    }
    advance();
    return SectionKind::Generals;
  }
  if (is_binaries_keyword(lower)) {
    advance();
    return SectionKind::Binaries;
  }
  // Semi-Continuous section header. Documented spellings:
  //   - 3-token "Semi - Continuous"
  //   - bare "Semi" / "Semis"
  // Check the 3-token form first so it consumes all three tokens.
  if (lower == "semi" && peek(1).kind == LpTokenKind::Minus &&
      name_equals_ci(peek(2), "continuous")) {
    advance();
    advance();
    advance();
    return SectionKind::SemiContinuous;
  }
  if (lower == "semi" || lower == "semis") {
    advance();
    return SectionKind::SemiContinuous;
  }
  if (is_end_keyword(lower)) {
    advance();
    return SectionKind::End;
  }

  // Known unsupported sections → throw with a clear message.
  reject_unsupported_section();
  return SectionKind::None;  // unreachable
}

// ---- Expression parsing ---------------------------------------------------

template <typename i_t, typename f_t>
f_t LpParseEngine<i_t, f_t>::parse_signed_number()
{
  int sign = 1;
  if (match(LpTokenKind::Minus)) {
    sign = -1;
  } else {
    match(LpTokenKind::Plus);  // optional leading '+'
  }
  if (is_infinity_keyword(peek())) {
    advance();
    return sign > 0 ? std::numeric_limits<f_t>::infinity() : -std::numeric_limits<f_t>::infinity();
  }
  mps_parser_expects(peek().kind == LpTokenKind::Number,
                     error_type_t::ValidationError,
                     "LP parse error at line %d: expected a number, got '%s'",
                     peek().line,
                     peek().text.c_str());
  f_t val = number_from_text(peek().text);
  advance();
  return sign * val;
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_linear_expression(std::vector<LinearTerm>& out_terms,
                                                      f_t& out_constant)
{
  out_constant = f_t(0);
  int sign     = 1;
  bool first   = true;

  while (true) {
    // A quadratic bracket ends the linear expression. If the bracket is
    // preceded by a sign, leave the sign unconsumed so the caller can
    // attribute it to the bracket.
    if (peek().kind == LpTokenKind::LBracket) break;
    if ((peek().kind == LpTokenKind::Plus || peek().kind == LpTokenKind::Minus) &&
        peek(1).kind == LpTokenKind::LBracket) {
      break;
    }

    if (peek().kind == LpTokenKind::Plus) {
      advance();
      sign = 1;
    } else if (peek().kind == LpTokenKind::Minus) {
      advance();
      sign = -1;
    } else if (!first) {
      // No sign between terms → expression ends here. (Relation tokens,
      // ']', section headers, EOF all terminate.)
      break;
    }

    // A term is: (number ('*')?)? varname  |  number (constant)  |  varname.
    // 'inf' is a bounds-only keyword and never appears here.
    f_t coeff      = f_t(1);
    bool had_coeff = false;
    bool had_star  = false;
    if (peek().kind == LpTokenKind::Number) {
      coeff     = number_from_text(peek().text);
      had_coeff = true;
      advance();
      had_star = match(LpTokenKind::Star);
    }

    // '<number> *' must be followed by a variable name; a stray '*' before a
    // relation, section header, or EOL would otherwise be silently dropped
    // (and the number would be misinterpreted as a constant).
    if (had_star) {
      mps_parser_expects(
        peek().kind == LpTokenKind::Name && !at_section_boundary() && !is_infinity_keyword(peek()),
        error_type_t::ValidationError,
        "LP parse error at line %d: expected variable name after '*', got '%s'",
        peek().line,
        peek().text.c_str());
    }

    // '<number> [' is ambiguous: did the user mean "<number> times the quadratic
    // bracket" or "constant <number> followed by a separate bracket"? Neither
    // interpretation is supported. The LP convention places the coefficient
    // inside the brackets, so reject and tell the user how to rewrite.
    if (had_coeff && peek().kind == LpTokenKind::LBracket) {
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: a numeric coefficient may not "
                         "directly precede a quadratic bracket '['; place the coefficient "
                         "inside the brackets",
                         peek().line);
    }

    if (peek().kind == LpTokenKind::Name && !at_section_boundary() &&
        !is_free_keyword(lowercase(peek().text)) && !is_infinity_keyword(peek())) {
      std::string var_name = peek().text;
      advance();
      i_t id = get_or_add_var(var_name);
      out_terms.push_back({id, sign * coeff});
    } else if (had_coeff) {
      // It was a pure number → contributes to the constant.
      out_constant += sign * coeff;
    } else {
      // Nothing consumed this iteration → not a term, stop.
      if (!first) {
        // We consumed a sign without a term: malformed.
        mps_parser_expects(false,
                           error_type_t::ValidationError,
                           "LP parse error at line %d: expected a term after '+' or '-'",
                           peek().line);
      }
      break;
    }

    first = false;
    sign  = 1;
  }
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_quadratic_bracket(
  int outer_sign, BracketRole role, std::vector<std::tuple<i_t, i_t, f_t>>& out_quad_entries)
{
  expect(LpTokenKind::LBracket, "'[' at start of quadratic section");

  // Accumulate raw LP-format entries first (diagonal vs off-diagonal), then
  // apply the role-specific convention and outer sign after we see the
  // closing bracket.
  std::vector<std::tuple<i_t, i_t, f_t>> raw_quad;

  int sign   = 1;
  bool first = true;
  while (peek().kind != LpTokenKind::RBracket) {
    mps_parser_expects(!at_eof(),
                       error_type_t::ValidationError,
                       "LP parse error: unterminated quadratic '[' section");

    if (peek().kind == LpTokenKind::Plus) {
      advance();
      sign = 1;
    } else if (peek().kind == LpTokenKind::Minus) {
      advance();
      sign = -1;
    } else if (!first) {
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: expected '+' or '-' between "
                         "quadratic terms, got '%s'",
                         peek().line,
                         peek().text.c_str());
    }

    f_t coeff = f_t(1);
    if (peek().kind == LpTokenKind::Number) {
      coeff = number_from_text(peek().text);
      advance();
      match(LpTokenKind::Star);  // optional
    }

    mps_parser_expects(peek().kind == LpTokenKind::Name,
                       error_type_t::ValidationError,
                       "LP parse error at line %d: expected variable name in quadratic term",
                       peek().line);
    std::string var1 = peek().text;
    advance();
    i_t i1 = get_or_add_var(var1);

    if (match(LpTokenKind::Caret)) {
      // Must be "^ 2".
      mps_parser_expects(peek().kind == LpTokenKind::Number && peek().text == "2",
                         error_type_t::ValidationError,
                         "LP parse error at line %d: only 'x ^ 2' is supported in quadratic "
                         "terms (got '%s')",
                         peek().line,
                         peek().text.c_str());
      advance();
      raw_quad.emplace_back(i1, i1, sign * coeff);
    } else if (match(LpTokenKind::Star)) {
      mps_parser_expects(peek().kind == LpTokenKind::Name,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: expected variable name after '*' in "
                         "quadratic cross term",
                         peek().line);
      std::string var2 = peek().text;
      advance();
      i_t i2 = get_or_add_var(var2);
      // Store in upper-triangular form (i <= j) to match QUADOBJ convention.
      i_t a = std::min(i1, i2);
      i_t b = std::max(i1, i2);
      raw_quad.emplace_back(a, b, sign * coeff);
    } else {
      // Pure linear terms are not allowed inside a quadratic bracket — the
      // LP-format convention reserves '[ ... ]' for squared and product
      // terms only. Place linear terms outside the bracket.
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: bare linear term '%s' is not "
                         "allowed inside a quadratic bracket '[ ... ]'; move it outside",
                         peek().line,
                         var1.c_str());
    }

    first = false;
    sign  = 1;
  }
  expect(LpTokenKind::RBracket, "closing ']' of quadratic section");

  const f_t sign_scale = static_cast<f_t>(outer_sign);

  if (role == BracketRole::Objective) {
    // Require the "/ 2" suffix after a quadratic objective expression.
    // Without it there is no ambiguity-free way to tell whether the user
    // meant /2 and forgot vs. intended bare coefficients, so we enforce the
    // stricter form.
    mps_parser_expects(peek().kind == LpTokenKind::Slash && peek(1).kind == LpTokenKind::Number &&
                         peek(1).text == "2",
                       error_type_t::ValidationError,
                       "LP parse error at line %d: quadratic expression '[ ... ]' in the "
                       "objective must be followed by '/ 2'",
                       peek().line);
    advance();  // '/'
    advance();  // '2'

    // Apply the LP "/ 2" convention uniformly: a bracket coefficient c on
    // either x_i^2 or x_i*x_j contributes c/2 to the corresponding objective
    // term. The resulting upper-triangular quadobj entries are passed
    // directly to cuOpt's set_quadratic_objective_matrix, which internally
    // computes H = Q + Q^T; the solver then minimizes (1/2) x^T H x, which
    // recovers the user's intended objective.
    for (auto& [a, b, v] : raw_quad) {
      v /= f_t(2);
      out_quad_entries.emplace_back(a, b, sign_scale * v);
    }
  } else {
    // Constraint: '/ 2' is forbidden — the LP convention is that constraint
    // quadratic brackets carry bare face-value coefficients of x^T Q x.
    mps_parser_expects(!(peek().kind == LpTokenKind::Slash && peek(1).kind == LpTokenKind::Number &&
                         peek(1).text == "2"),
                       error_type_t::ValidationError,
                       "LP parse error at line %d: quadratic expression '[ ... ]' in a "
                       "constraint must NOT be followed by '/ 2' (the '/ 2' suffix is "
                       "reserved for the objective)",
                       peek().line);
    // A bracket containing only linear terms is meaningless in a constraint
    // — the user can write the same constraint without the brackets.
    mps_parser_expects(!raw_quad.empty(),
                       error_type_t::ValidationError,
                       "LP parse error at line %d: quadratic bracket '[ ... ]' in a "
                       "constraint must contain at least one quadratic term",
                       peek().line);

    // Coefficients are at face value — the post-pass that flushes the
    // quadratic_constraint_block_t to the data model handles the symmetric
    // expansion and the /2 split for off-diagonals.
    for (auto& [a, b, v] : raw_quad) {
      out_quad_entries.emplace_back(a, b, sign_scale * v);
    }
  }
}

// ---- Section bodies -------------------------------------------------------

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_objective_section()
{
  // Optional "name:" label.
  if (peek().kind == LpTokenKind::Name && peek(1).kind == LpTokenKind::Colon &&
      !at_section_boundary()) {
    out_.objective_name = peek().text;
    advance();
    advance();
  }

  std::vector<LinearTerm> linear;
  f_t constant = 0;
  parse_linear_expression(linear, constant);

  // Optional quadratic bracket, possibly preceded by a sign. In this LP
  // dialect the bracket sits inside the objective expression and can
  // appear before or after linear terms; we support one bracket followed
  // by (possibly) more linear terms.
  int quad_sign = 1;
  if (peek().kind == LpTokenKind::Plus && peek(1).kind == LpTokenKind::LBracket) {
    advance();
  } else if (peek().kind == LpTokenKind::Minus && peek(1).kind == LpTokenKind::LBracket) {
    advance();
    quad_sign = -1;
  }
  if (peek().kind == LpTokenKind::LBracket) {
    parse_quadratic_bracket(quad_sign, BracketRole::Objective, out_.quadobj_entries);

    // More linear terms may follow the bracket.
    std::vector<LinearTerm> more;
    f_t more_constant = 0;
    parse_linear_expression(more, more_constant);
    for (const auto& lt : more)
      linear.push_back(lt);
    constant += more_constant;
  }

  // Apply linear terms to the objective vector. Coefficients accumulate in
  // case the same variable appears twice.
  for (const auto& lt : linear) {
    if (static_cast<size_t>(lt.var_id) >= out_.c_values.size()) {
      out_.c_values.resize(lt.var_id + 1, f_t(0));
    }
    out_.c_values[lt.var_id] += lt.coeff;
  }
  // A constant term in the objective becomes the objective offset.
  out_.objective_offset_value += constant;
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_constraints_section()
{
  while (!at_section_boundary()) {
    // Optional "name:" label — present iff the first two tokens are Name + ':'.
    std::string row_name;
    if (peek().kind == LpTokenKind::Name && peek(1).kind == LpTokenKind::Colon) {
      row_name = peek().text;
      advance();
      advance();
    } else {
      row_name = "R" + std::to_string(anon_row_counter_++);
    }

    std::vector<LinearTerm> linear;
    f_t lhs_constant = 0;
    parse_linear_expression(linear, lhs_constant);

    // Optional '+ [ ... ]' or '- [ ... ]' quadratic block in the LHS.
    // Mirrors the objective handling; if present, this row becomes a
    // quadratic constraint and is stored on quadratic_constraint_blocks
    // instead of the linear arrays.
    std::vector<std::tuple<i_t, i_t, f_t>> qc_triples;
    bool is_quadratic_row = false;
    int quad_sign         = 1;
    if (peek().kind == LpTokenKind::Plus && peek(1).kind == LpTokenKind::LBracket) {
      advance();
    } else if (peek().kind == LpTokenKind::Minus && peek(1).kind == LpTokenKind::LBracket) {
      advance();
      quad_sign = -1;
    }
    if (peek().kind == LpTokenKind::LBracket) {
      is_quadratic_row = true;
      parse_quadratic_bracket(quad_sign, BracketRole::Constraint, qc_triples);

      // More linear terms may follow the bracket. parse_linear_expression
      // does not produce a constant unless the user wrote one in the LHS;
      // a constant gets moved to RHS just like the pre-bracket constant.
      std::vector<LinearTerm> more;
      f_t more_constant = 0;
      parse_linear_expression(more, more_constant);
      for (const auto& lt : more)
        linear.push_back(lt);
      lhs_constant += more_constant;
    }

    RowType row_type{};
    if (peek().kind == LpTokenKind::LessEq) {
      row_type = LesserThanOrEqual;
      advance();
    } else if (peek().kind == LpTokenKind::GreaterEq) {
      row_type = GreaterThanOrEqual;
      advance();
    } else if (peek().kind == LpTokenKind::Equal) {
      row_type = Equality;
      advance();
    } else {
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: expected a relation operator "
                         "(<=, >=, =) in constraint, got '%s'",
                         peek().line,
                         peek().text.c_str());
    }

    // Quadratic constraints currently only support '≤' in the data model
    // (see mps_data_model_t::quadratic_constraint_t docs).
    if (is_quadratic_row) {
      mps_parser_expects(row_type == LesserThanOrEqual,
                         error_type_t::ValidationError,
                         "LP parse error at line %d: quadratic constraint '%s' must use "
                         "'<=' (only convex '≤' quadratic constraints are supported)",
                         peek().line,
                         row_name.c_str());
    }

    f_t rhs_value = parse_signed_number();
    // Any constant that appeared on the LHS is moved to the RHS with a sign flip.
    rhs_value -= lhs_constant;

    // Register the row (track name uniqueness regardless of linear/quadratic).
    mps_parser_expects(row_names_map_.find(row_name) == row_names_map_.end(),
                       error_type_t::ValidationError,
                       "Duplicate constraint name '%s'",
                       row_name.c_str());

    // Collect the linear part. Coefficients accumulate for repeated variables;
    // sort by var_id for deterministic CSR output.
    std::unordered_map<i_t, f_t> row_coeffs;
    for (const auto& lt : linear)
      row_coeffs[lt.var_id] += lt.coeff;
    std::vector<std::pair<i_t, f_t>> ordered(row_coeffs.begin(), row_coeffs.end());
    std::sort(ordered.begin(), ordered.end());
    std::vector<i_t> indices;
    std::vector<f_t> values;
    indices.reserve(ordered.size());
    values.reserve(ordered.size());
    for (const auto& [vid, val] : ordered) {
      if (val == f_t(0)) continue;
      indices.push_back(vid);
      values.push_back(val);
    }

    if (is_quadratic_row) {
      // Stash for the post-pass; quadratic rows are *not* added to the
      // linear arrays (row_names/row_types/A_indices/A_values/b_values).
      // We still record the name in row_names_map_ for uniqueness checks
      // — but using a sentinel id below the linear count would be wrong,
      // so use a separate sentinel and a placeholder name reservation.
      typename lp_parser_t<i_t, f_t>::quadratic_constraint_block_t block;
      block.row_name       = row_name;
      block.row_type       = row_type;
      block.linear_indices = std::move(indices);
      block.linear_values  = std::move(values);
      block.rhs_value      = rhs_value;
      block.quad_triples   = std::move(qc_triples);
      out_.quadratic_constraint_blocks.push_back(std::move(block));
      // Use std::numeric_limits<i_t>::max() as a sentinel; the map is only
      // used for uniqueness, never for index lookup.
      row_names_map_.emplace(row_name, std::numeric_limits<i_t>::max());
    } else {
      i_t row_id = static_cast<i_t>(out_.row_names.size());
      out_.row_names.push_back(row_name);
      row_names_map_.emplace(row_name, row_id);
      out_.row_types.push_back(row_type);
      out_.b_values.push_back(rhs_value);
      out_.A_indices.push_back(std::move(indices));
      out_.A_values.push_back(std::move(values));
    }
  }
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_bounds_section()
{
  while (!at_section_boundary()) {
    // Either starts with a variable name or with a signed number. 'inf' /
    // 'infinity' tokens are Names but only valid in the lb-first form.
    if (peek().kind == LpTokenKind::Name && !is_infinity_keyword(peek())) {
      std::string var_name = peek().text;
      advance();
      i_t vid = get_or_add_var(var_name);
      bounds_defined_for_var_id_.insert(vid);

      // Suffix after the name.
      if (peek().kind == LpTokenKind::Name && is_free_keyword(lowercase(peek().text))) {
        advance();
        out_.variable_lower_bounds[vid] = -std::numeric_limits<f_t>::infinity();
        out_.variable_upper_bounds[vid] = std::numeric_limits<f_t>::infinity();
        lower_explicitly_set_.insert(vid);
      } else if (match(LpTokenKind::LessEq)) {
        // x <= ub
        out_.variable_upper_bounds[vid] = parse_signed_number();
      } else if (match(LpTokenKind::GreaterEq)) {
        // x >= lb
        out_.variable_lower_bounds[vid] = parse_signed_number();
        lower_explicitly_set_.insert(vid);
      } else if (match(LpTokenKind::Equal)) {
        // x = value (fixed)
        f_t v                           = parse_signed_number();
        out_.variable_lower_bounds[vid] = v;
        out_.variable_upper_bounds[vid] = v;
        lower_explicitly_set_.insert(vid);
      } else {
        mps_parser_expects(false,
                           error_type_t::ValidationError,
                           "LP parse error at line %d: expected 'free', '<=', '>=' or '=' "
                           "after variable name in Bounds section, got '%s'",
                           peek().line,
                           peek().text.c_str());
      }
    } else {
      // lb <= x [<= ub]
      f_t lb = parse_signed_number();
      expect(LpTokenKind::LessEq, "'<=' in 'lb <= var' bound");
      mps_parser_expects(peek().kind == LpTokenKind::Name && !is_infinity_keyword(peek()),
                         error_type_t::ValidationError,
                         "LP parse error at line %d: expected variable name after 'lb <='",
                         peek().line);
      std::string var_name = peek().text;
      advance();
      i_t vid = get_or_add_var(var_name);
      bounds_defined_for_var_id_.insert(vid);
      out_.variable_lower_bounds[vid] = lb;
      lower_explicitly_set_.insert(vid);
      if (match(LpTokenKind::LessEq)) { out_.variable_upper_bounds[vid] = parse_signed_number(); }
    }
  }

  // A negative upper bound requires an explicitly stated lower bound,
  // otherwise the default lower of 0 would collide with the upper and make
  // the variable silently infeasible. Flag this at parse time.
  for (i_t vid : bounds_defined_for_var_id_) {
    if (out_.variable_upper_bounds[vid] < f_t(0) && !lower_explicitly_set_.count(vid)) {
      mps_parser_expects(false,
                         error_type_t::ValidationError,
                         "LP parse error: variable '%s' has a negative upper bound (%g) "
                         "without an explicit lower bound. Write '-inf <= %s <= %g' or give "
                         "an explicit lower bound alongside the upper bound.",
                         out_.var_names[vid].c_str(),
                         static_cast<double>(out_.variable_upper_bounds[vid]),
                         out_.var_names[vid].c_str(),
                         static_cast<double>(out_.variable_upper_bounds[vid]));
    }
  }
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_integer_list_section(bool is_binary)
{
  while (!at_section_boundary()) {
    mps_parser_expects(peek().kind == LpTokenKind::Name,
                       error_type_t::ValidationError,
                       "LP parse error at line %d: expected variable name in %s section, got '%s'",
                       peek().line,
                       is_binary ? "Binaries" : "Generals",
                       peek().text.c_str());
    std::string var_name = peek().text;
    advance();
    i_t vid = get_or_add_var(var_name);
    // Reject if this variable was previously declared semi-continuous; the
    // combination is ambiguous (integer vs. continuous-or-zero).
    mps_parser_expects(out_.var_types[vid] != 'S',
                       error_type_t::ValidationError,
                       "Variable '%s' appears in both Semi-Continuous and %s sections",
                       var_name.c_str(),
                       is_binary ? "Binaries" : "Generals");
    out_.var_types[vid] = 'I';
    if (is_binary) {
      out_.variable_lower_bounds[vid] = f_t(0);
      out_.variable_upper_bounds[vid] = f_t(1);
      bounds_defined_for_var_id_.insert(vid);
    }
  }
}

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_semi_continuous_section()
{
  while (!at_section_boundary()) {
    mps_parser_expects(
      peek().kind == LpTokenKind::Name,
      error_type_t::ValidationError,
      "LP parse error at line %d: expected variable name in Semi-Continuous section, got '%s'",
      peek().line,
      peek().text.c_str());
    std::string var_name = peek().text;
    advance();
    i_t vid = get_or_add_var(var_name);
    // Reject if the variable was previously declared integer/binary; the
    // combination is ambiguous (integer vs. continuous-or-zero).
    mps_parser_expects(out_.var_types[vid] != 'I',
                       error_type_t::ValidationError,
                       "Variable '%s' appears in both Generals/Binaries and Semi-Continuous "
                       "sections",
                       var_name.c_str());
    out_.var_types[vid] = 'S';
  }
}

// ---- Top-level dispatch ----------------------------------------------------

template <typename i_t, typename f_t>
void LpParseEngine<i_t, f_t>::parse_all()
{
  bool saw_objective = false;
  bool saw_end       = false;

  while (!at_eof()) {
    SectionKind kind = try_consume_section_header();
    switch (kind) {
      case SectionKind::Objective:
        mps_parser_expects(!saw_objective,
                           error_type_t::ValidationError,
                           "LP parse error: multiple objective sections");
        parse_objective_section();
        saw_objective = true;
        break;
      case SectionKind::Constraints: parse_constraints_section(); break;
      case SectionKind::Bounds: parse_bounds_section(); break;
      case SectionKind::Generals: parse_integer_list_section(false); break;
      case SectionKind::Binaries: parse_integer_list_section(true); break;
      case SectionKind::SemiContinuous: parse_semi_continuous_section(); break;
      case SectionKind::End:
        saw_end = true;
        break;  // Break out of the switch; the check below ends parsing.
      case SectionKind::None: break;
    }
    if (saw_end) break;  // Anything after 'End' is ignored.
  }
  if (!saw_end) { printf("LP parser: 'End' section is missing\n"); }
  mps_parser_expects(saw_objective,
                     error_type_t::ValidationError,
                     "LP parse error: no objective (Minimize/Maximize) section found");
}

}  // namespace

// ===========================================================================
// lp_parser_t — thin public wrapper. All parsing state/types live in the
// anonymous namespace above.
// ===========================================================================

namespace {

// Consumes the LP parser's intermediate parsed data and populates `problem`.
//
// CSR flatten, row-type → constraint-bound conversion, quadratic objective
// matrix construction, metadata setters. MPS uses its own fill_problem
// (mps_parser.cpp) because its quadratic rows are interleaved with linear
// rows in the per-row vectors and need a compaction pass; the LP parser
// partitions quadratic rows into quadratic_constraint_blocks at parse time,
// so the per-row vectors here contain only linear rows and no compaction is
// needed. Quadratic LP constraints are emitted by flush_quadratic_constraints
// below.
template <typename i_t, typename f_t>
void finalize_problem(mps_data_model_t<i_t, f_t>& problem, lp_parser_t<i_t, f_t>& parser)
{
  const i_t n_vars = static_cast<i_t>(parser.var_names.size());
  const i_t n_rows = static_cast<i_t>(parser.row_names.size());

  // Pad per-variable vectors that may have grown after their initial size
  // (e.g., a variable first appeared after c_values was already initialized).
  if (static_cast<i_t>(parser.c_values.size()) < n_vars) parser.c_values.resize(n_vars, f_t(0));
  if (static_cast<i_t>(parser.variable_lower_bounds.size()) < n_vars) {
    parser.variable_lower_bounds.resize(n_vars, f_t(0));
  }
  if (static_cast<i_t>(parser.variable_upper_bounds.size()) < n_vars) {
    parser.variable_upper_bounds.resize(n_vars, std::numeric_limits<f_t>::infinity());
  }
  if (static_cast<i_t>(parser.var_types.size()) < n_vars) parser.var_types.resize(n_vars, 'C');

  // Flatten the ragged A_indices / A_values into a single CSR.
  std::vector<i_t> offsets;
  std::vector<i_t> indices;
  std::vector<f_t> values;
  offsets.reserve(n_rows + 1);
  offsets.push_back(0);
  for (i_t i = 0; i < n_rows; ++i) {
    for (i_t idx : parser.A_indices[i])
      indices.push_back(idx);
    for (f_t v : parser.A_values[i])
      values.push_back(v);
    offsets.push_back(static_cast<i_t>(values.size()));
  }
  problem.set_csr_constraint_matrix(values, indices, offsets);

  mps_parser_expects(indices.size() == values.size(),
                     error_type_t::ValidationError,
                     "Constraint matrix nonzero vector (%zu) and column-index vector (%zu) "
                     "must have the same size.",
                     indices.size(),
                     values.size());
  mps_parser_expects(!offsets.empty() && offsets.back() == static_cast<i_t>(values.size()),
                     error_type_t::ValidationError,
                     "CSR offset tail (%d) must equal the nonzero count (%zu).",
                     offsets.empty() ? 0 : offsets.back(),
                     values.size());

  problem.set_constraint_bounds(parser.b_values);
  problem.set_objective_coefficients(parser.c_values);
  problem.set_objective_scaling_factor(f_t(1));
  problem.set_objective_offset(parser.objective_offset_value);

  problem.set_variable_lower_bounds(parser.variable_lower_bounds);
  problem.set_variable_upper_bounds(parser.variable_upper_bounds);

  mps_parser_expects(
    (problem.get_variable_lower_bounds().size() == problem.get_variable_upper_bounds().size()) &&
      (problem.get_variable_upper_bounds().size() == problem.get_objective_coefficients().size()),
    error_type_t::ValidationError,
    "Per-variable vectors are inconsistently sized. objective=%zu, lb=%zu, ub=%zu.",
    problem.get_objective_coefficients().size(),
    problem.get_variable_lower_bounds().size(),
    problem.get_variable_upper_bounds().size());

  // Semi-continuous variables must have a finite upper bound; otherwise the
  // "x = 0 or lb <= x <= ub" semantics collapse to a regular continuous
  // variable. Matches the MPS parser's rule.
  for (i_t i = 0; i < n_vars; ++i) {
    if (parser.var_types[i] == 'S') {
      mps_parser_expects(!std::isinf(parser.variable_upper_bounds[i]),
                         error_type_t::ValidationError,
                         "Semi-continuous variable '%s' must have a finite upper bound",
                         parser.var_names[i].c_str());
    }
  }

  // Row types + RHS → explicit constraint lower/upper bounds.
  const f_t inf = std::numeric_limits<f_t>::infinity();
  std::vector<f_t> clb;
  std::vector<f_t> cub;
  clb.reserve(n_rows);
  cub.reserve(n_rows);
  for (i_t i = 0; i < n_rows; ++i) {
    switch (parser.row_types[i]) {
      case Equality:
        clb.push_back(parser.b_values[i]);
        cub.push_back(parser.b_values[i]);
        break;
      case GreaterThanOrEqual:
        clb.push_back(parser.b_values[i]);
        cub.push_back(inf);
        break;
      case LesserThanOrEqual:
        clb.push_back(-inf);
        cub.push_back(parser.b_values[i]);
        break;
      default:
        mps_parser_expects(false,
                           error_type_t::ValidationError,
                           "Unsupported row type for row '%s'",
                           parser.row_names[i].c_str());
    }
    mps_parser_expects(!std::isnan(clb.back()) && !std::isnan(cub.back()),
                       error_type_t::ValidationError,
                       "Constraint bound for row '%s' is NaN",
                       parser.row_names[i].c_str());
  }
  problem.set_constraint_lower_bounds(clb);
  problem.set_constraint_upper_bounds(cub);

  mps_parser_expects(
    (problem.get_constraint_lower_bounds().size() ==
     problem.get_constraint_upper_bounds().size()) &&
      (problem.get_constraint_upper_bounds().size() == problem.get_constraint_bounds().size()),
    error_type_t::ValidationError,
    "Per-constraint vectors are inconsistently sized. rhs=%zu, lb=%zu, ub=%zu.",
    problem.get_constraint_bounds().size(),
    problem.get_constraint_lower_bounds().size(),
    problem.get_constraint_upper_bounds().size());

  problem.set_problem_name(parser.problem_name);
  problem.set_objective_name(parser.objective_name);
  problem.set_variable_names(parser.var_names);
  problem.set_variable_types(parser.var_types);
  problem.set_row_names(parser.row_names);
  std::vector<char> row_types_chars(parser.row_types.size());
  for (size_t i = 0; i < parser.row_types.size(); ++i) {
    row_types_chars[i] = static_cast<char>(parser.row_types[i]);
  }
  problem.set_row_types(row_types_chars);
  problem.set_maximize(parser.maximize);

  // Quadratic objective: emit the upper-triangular quadobj entries as CSR.
  // cuOpt's GPU-side set_quadratic_objective_matrix applies H = Q + Q^T
  // internally, so no mirror step is needed here — the entries are already
  // /2-scaled inside parse_quadratic_bracket so the solver's (1/2) x^T H x
  // recovers the user's intended objective.
  if (!parser.quadobj_entries.empty()) {
    std::vector<std::vector<std::pair<i_t, f_t>>> row_data(n_vars);
    for (const auto& [row, col, val] : parser.quadobj_entries) {
      row_data[row].emplace_back(col, val);
    }
    for (auto& row : row_data) {
      std::sort(row.begin(), row.end());
    }
    std::vector<f_t> q_values;
    std::vector<i_t> q_indices;
    std::vector<i_t> q_offsets;
    q_offsets.reserve(static_cast<size_t>(n_vars) + 1);
    q_offsets.push_back(0);
    for (i_t row = 0; row < n_vars; ++row) {
      for (const auto& [col, val] : row_data[row]) {
        q_values.push_back(val);
        q_indices.push_back(col);
      }
      q_offsets.push_back(static_cast<i_t>(q_values.size()));
    }
    problem.set_quadratic_objective_matrix(q_values, q_indices, q_offsets);
  }
}

// Emits one quadratic_constraint_block_t to `problem` via
// append_quadratic_constraint(). Row indices are assigned
// linear_row_count..linear_row_count + nqc - 1, mirroring MPS's QCMATRIX
// handling in mps_parser_t::fill_problem.
template <typename i_t, typename f_t>
void flush_quadratic_constraints(mps_data_model_t<i_t, f_t>& problem,
                                 const lp_parser_t<i_t, f_t>& parser)
{
  const i_t n_vars           = static_cast<i_t>(parser.var_names.size());
  const i_t linear_row_count = static_cast<i_t>(parser.row_names.size());
  i_t k                      = 0;
  for (const auto& block : parser.quadratic_constraint_blocks) {
    std::vector<f_t> q_values;
    std::vector<i_t> q_indices;
    std::vector<i_t> q_offsets;
    build_symmetric_q_csr(block.quad_triples, n_vars, q_values, q_indices, q_offsets);
    problem.append_quadratic_constraint(linear_row_count + k,
                                        block.row_name,
                                        static_cast<char>(block.row_type),
                                        block.linear_values,
                                        block.linear_indices,
                                        block.rhs_value,
                                        q_values,
                                        q_indices,
                                        q_offsets);
    ++k;
  }
}

}  // end namespace

template <typename i_t, typename f_t>
lp_parser_t<i_t, f_t>::lp_parser_t(mps_data_model_t<i_t, f_t>& problem, const std::string& file)
{
  LpParseEngine<i_t, f_t> engine(*this, file);
  finalize_problem(problem, *this);
  flush_quadratic_constraints(problem, *this);
}

template <typename i_t, typename f_t>
lp_parser_t<i_t, f_t>::lp_parser_t(mps_data_model_t<i_t, f_t>& problem, std::string_view input)
{
  LpParseEngine<i_t, f_t> engine(*this, input);
  finalize_problem(problem, *this);
  flush_quadratic_constraints(problem, *this);
}

template class lp_parser_t<int, float>;
template class lp_parser_t<int, double>;

// ===========================================================================
// Public read_lp() / read_lp_from_string()
// ===========================================================================

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_lp(const std::string& lp_file_path)
{
  mps_data_model_t<i_t, f_t> problem;
  lp_parser_t<i_t, f_t> parser(problem, lp_file_path);
  return problem;
}

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_lp_from_string(std::string_view lp_contents)
{
  mps_data_model_t<i_t, f_t> problem;
  lp_parser_t<i_t, f_t> parser(problem, lp_contents);
  return problem;
}

template mps_data_model_t<int, float> read_lp<int, float>(const std::string&);
template mps_data_model_t<int, double> read_lp<int, double>(const std::string&);
template mps_data_model_t<int, float> read_lp_from_string<int, float>(std::string_view);
template mps_data_model_t<int, double> read_lp_from_string<int, double>(std::string_view);

}  // namespace cuopt::linear_programming::io
