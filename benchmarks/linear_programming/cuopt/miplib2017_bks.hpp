/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// MIPLIB2017 best-known solution (BKS) lookup for the MIP
// benchmark runner. Self-contained: no env vars, no external CSV.
//
// Coverage: every instance in the MIPLIB2017 *benchmark* set (240
// instances). Of those, 232 have a known BKS and live in
// kBenchmarkBKS; 7 are infeasible and live in kBenchmarkInfeasible
// so the printer can label them clearly instead of returning "no bks".
//
// Lookup uses the basename without directory and stripped of
// .mps / .mps.gz / .lp / .lp.gz / .gz suffixes, lower-cased. So
//   "miplib2017/MAS74.mps.gz" / "mas74.mps" / "mas74"
// all hit the same entry.
//
// Returns std::optional<double>: nullopt means "instance is in our
// benchmark set but infeasible" *or* "we don't have a BKS entry for it".
// is_known_infeasible() distinguishes the two.

#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cuopt_bench {

// Strip directory prefix and any .mps/.lp suffix (with optional .gz),
// then lower-case. Designed to match how MPS instance files are named
// across MIPLIB downloads (case- and extension-insensitive).
inline std::string normalize_instance_name(const std::string& raw)
{
  std::string s    = raw;
  const auto slash = s.find_last_of("/\\");
  if (slash != std::string::npos) { s = s.substr(slash + 1); }
  auto endswith = [&](const std::string& suf) {
    if (s.size() < suf.size()) { return false; }
    for (size_t i = 0; i < suf.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(s[s.size() - suf.size() + i])) !=
          std::tolower(static_cast<unsigned char>(suf[i]))) {
        return false;
      }
    }
    return true;
  };
  for (const char* suf : {".mps.gz", ".lp.gz", ".mps", ".lp", ".gz"}) {
    if (endswith(suf)) {
      s.resize(s.size() - std::char_traits<char>::length(suf));
      break;
    }
  }
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

// MIPLIB2017 benchmark-set best-known solutions (n=232). Source:
// https://miplib.zib.de "The Benchmark Set". Values are stored in the
// double precision they were published at; unit tests should compare
// with a tolerance of ~|bks|*1e-9 rather than exact equality.
inline const std::unordered_map<std::string, double>& kBenchmarkBKS()
{
  static const std::unordered_map<std::string, double> kBKS = {
    {"30n20b8", 302},
    {"50v-10", 3311.1799841000002},
    {"academictimetablesmall", 0},
    {"air05", 26374},
    {"app1-1", -3},
    {"app1-2", -41},
    {"assign1-5-8", 211.99999999999801},
    {"atlanta-ip", 90.009878614000002},
    {"b1c1s1", 24544.25},
    {"bab2", -357544.31150000001},
    {"bab6", -284248.23070000007},
    {"beasleyc3", 753.9999999999128},
    {"binkar10_1", 6742.1998835000004},
    {"blp-ar98", 6205.2147103999996},
    {"blp-ic98", 4491.4475839500001},
    {"bnatt400", 1},
    {"bppc4-08", 53},
    {"brazil3", 24},
    {"buildingenergy", 33283.853236000003},
    {"cbs-cta", 0},
    {"chromaticindex1024-7", 4},
    {"chromaticindex512-7", 4},
    {"cmflsp50-24-8-8", 55789389.886},
    {"cms750_4", 252},
    {"co-100", 2639942.0600000001},
    {"cod105", -12},
    {"comp07-2idx", 6},
    {"comp21-2idx", 74},
    {"cost266-uue", 25148940.55999998},
    {"cryptanalysiskb128n5obj16", 0},
    {"csched007", 350.99999999999551},
    {"csched008", 173},
    {"cvs16r128-89", -97},
    {"dano3_3", 576.34463302999995},
    {"dano3_5", 576.9249159565619},
    {"decomp2", -160},
    {"drayage-100-23", 103333.87407000001},
    {"drayage-25-23", 101282.647018},
    {"dws008-01", 37412.604587945083},
    {"eil33-2", 934.007915999999},
    {"eila101-2", 880.92010799999991},
    {"enlight_hard", 37},
    {"ex10", 100},
    {"ex9", 81},
    {"exp-1-500-5-5", 65887},
    {"fast0507", 174},
    {"fastxgemm-n2r6s0t2", 230},
    {"fhnw-binpack4-48", 0},
    {"fiball", 138},
    {"gen-ip002", -4783.7333920000001},
    {"gen-ip054", 6840.9656417899996},
    {"germanrr", 47095869.648999996},
    {"gfd-schedulen180f7d50m30k18", 1},
    {"glass-sc", 23},
    {"glass4", 1200012599.972384},
    {"gmu-35-40", -2406733.3687999998},
    {"gmu-35-50", -2607958.3300000001},
    {"graph20-20-1rand", -9},
    {"graphdraw-domain", 19685.999975500381},
    {"h80x6320d", 6382.0990482459993},
    {"highschool1-aigio", 0},
    {"hypothyroid-k1", -2851},
    {"ic97_potential", 3941.9999309022501},
    {"icir97_tension", 6375},
    {"irish-electricity", 3723497.5913959998},
    {"irp", 12159.492835396981},
    {"istanbul-no-cutoff", 204.08170701},
    {"k1mushroom", -3288},
    {"lectsched-5-obj", 24},
    {"leo1", 404227536.16000003},
    {"leo2", 404077441.12},
    {"lotsize", 1480195},
    {"mad", 0.026800000000000001},
    {"map10", -495},
    {"map16715-04", -111},
    {"markshare2", 1},
    {"markshare_4_0", 1},
    {"mas74", 11801.185719999999},
    {"mas76", 40005.053989999993},
    {"mc11", 11688.99999999966},
    {"mcsched", 211913},
    {"mik-250-20-75-4", -52301},
    {"milo-v12-6-r2-40-1", 326481.14282799},
    {"momentum1", 109143.4935},
    {"mushroom-best", 0.055333761199999998},
    {"mzzv11", -21718},
    {"mzzv42z", -20540},
    {"n2seq36q", 52200},
    {"n3div36", 130800},
    {"n5-3", 8104.9999999939992},
    {"neos-1122047", 161},
    {"neos-1171448", -309},
    {"neos-1171737", -195},
    {"neos-1354092", 46},
    {"neos-1445765", -17783},
    {"neos-1456979", 176},
    {"neos-1582420", 90.999999999999957},
    {"neos-2657525-crna", 1.810748},
    {"neos-2746589-doon", 2008.1999999999989},
    {"neos-2978193-inde", -2.3880616899999998},
    {"neos-2987310-joes", -607702988.29999995},
    {"neos-3004026-krka", 0},
    {"neos-3024952-loue", 26756},
    {"neos-3046615-murg", 1600},
    {"neos-3083819-nubu", 6307996},
    {"neos-3216931-puriri", 71320},
    {"neos-3381206-awhea", 453},
    {"neos-3402294-bobin", 0.067249999999999491},
    {"neos-3555904-turama", -34.700000000000003},
    {"neos-3627168-kasai", 988585.61999999976},
    {"neos-3656078-kumeu", -13172.200000000001},
    {"neos-3754480-nidda", 12941.73838561778},
    {"neos-4300652-rahue", 2.1415999999999999},
    {"neos-4338804-snowy", 1471},
    {"neos-4387871-tavua", 33.384729927000002},
    {"neos-4413714-turia", 45.370167019999798},
    {"neos-4532248-waihi", 61.599999999999987},
    {"neos-4647030-tutaki", 27265.705999999958},
    {"neos-4722843-widden", 25009.662227000001},
    {"neos-4738912-atrato", 283627956.59500003},
    {"neos-4763324-toguru", 1613.0388458499999},
    {"neos-4954672-berkel", 2612710},
    {"neos-5049753-cuanza", 561.99999716889999},
    {"neos-5052403-cygnet", 182},
    {"neos-5093327-huahum", 6259.9999971258949},
    {"neos-5104907-jarama", 935},
    {"neos-5107597-kakapo", 3644.9999999995198},
    {"neos-5114902-kasavu", 655},
    {"neos-5188808-nattai", 0.110283622999984},
    {"neos-5195221-niemur", 0.0038354325999999999},
    {"neos-631710", 203},
    {"neos-662469", 184379.99999999991},
    {"neos-787933", 30},
    {"neos-827175", 112.00152},
    {"neos-848589", 2351.40309999697},
    {"neos-860300", 3200.9999999999982},
    {"neos-873061", 113.6562385063},
    {"neos-911970", 54.759999999999998},
    {"neos-933966", 318},
    {"neos-950242", 4},
    {"neos-957323", -237.75668150000001},
    {"neos-960392", -238},
    {"neos17", 0.1500025774},
    {"neos5", 15},
    {"neos8", -3719},
    {"net12", 214},
    {"netdiversion", 242},
    {"nexp-150-20-8-5", 231},
    {"ns1116954", 0},
    {"ns1208400", 2},
    {"ns1644855", -1524.3333333333301},
    {"ns1760995", -549.21438505000003},
    {"ns1830653", 20622},
    {"ns1952667", 0},
    {"nu25-pr12", 53904.999999999993},
    {"nursesched-medium-hint03", 115},
    {"nursesched-sprint02", 57.999999999999993},
    {"nw04", 16862},
    {"opm2-z10-s4", -33269},
    {"p200x1188c", 15078},
    {"peg-solitaire-a3", 1},
    {"pg", -8674.3426071199992},
    {"pg5_34", -14339.353450000001},
    {"physiciansched3-3", 2623271.3266670001},
    {"physiciansched6-2", 49324},
    {"piperout-08", 125054.9999999999},
    {"piperout-27", 8123.9999999999727},
    {"pk1", 11},
    {"proteindesign121hz512p9", 1473},
    {"proteindesign122trx11p8", 1747},
    {"qap10", 339.99999999838712},
    {"radiationm18-12-05", 17566},
    {"radiationm40-10-02", 155328},
    {"rail01", -70.569964299999995},
    {"rail02", -200.44990770000001},
    {"rail507", 174},
    {"ran14x18-disj-8", 3712},
    {"rd-rplusc-21", 165395.275295},
    {"reblock115", -36800603.233199999},
    {"rmatr100-p10", 423},
    {"rmatr200-p5", 4521},
    {"roci-4-11", -6020203},
    {"rocii-5-11", -6.6755047315380001},
    {"rococob10-011000", 19449},
    {"rocococ10-001000", 11460},
    {"roi2alpha3n4", -63.208495030000002},
    {"roi5alpha10n8", -52.322274350999997},
    {"roll3000", 12889.999991999999},
    {"s100", -0.16972352705829999},
    {"s250r10", -0.17178048342319999},
    {"satellites2-40", -19},
    {"satellites2-60-fs", -19.000000000099998},
    {"savsched1", 3217.6999999999998},
    {"sct2", -230.9891623},
    {"seymour", 423},
    {"seymour1", 410.76370138999999},
    {"sing326", 7753674.8537600003},
    {"sing44", 8128831.1771999998},
    {"snp-02-004-104", 586803238.65672886},
    {"sorrell3", -16},
    {"sp150x300d", 69},
    {"sp97ar", 660705645.75899994},
    {"sp98ar", 529740623.19999999},
    {"splice1k1", -394},
    {"square41", 15},
    {"square47", 15.9999999997877},
    {"supportcase10", 7},
    {"supportcase12", -7559.5330538170001},
    {"supportcase18", 48},
    {"supportcase19", 12677205.999920519},
    {"supportcase22", 110},  // best-known marked "*" in MIPLIB2017 (not proven optimal)
    {"supportcase26", 1745.1238129999999},
    {"supportcase33", -345},
    {"supportcase40", 24256.3122898},
    {"supportcase42", 7.7586307222700004},
    {"supportcase6", 51906.477370000001},
    {"supportcase7", -1132.2231770000001},
    {"swath1", 379.07129574999999},
    {"swath3", 397.76134365000001},
    {"tbfp-network", 24.163194440000002},
    {"thor50dday", 40417},
    {"timtab1", 764771.99999977998},
    {"tr12-30", 130595.9999999999},
    {"traininstance2", 71820},
    {"traininstance6", 28290},
    {"trento1", 5189487},
    {"triptim1", 22.868099999999899},
    {"uccase12", 11507.4050616},
    {"uccase9", 10993.131409},
    {"uct-subprob", 314},
    {"unitcal_7", 19635558.243999999},
    {"var-smallemery-m6j6", -149.37501},
    {"wachplan", -8},
  };
  return kBKS;
}

// MIPLIB2017 benchmark-set instances flagged as infeasible (n=7).
// Solver should return Infeasible status; we use this set to label
// the printer line with status_extra=KnownInfeasible so a downstream
// "did the run agree with MIPLIB?" check can be a single grep.
inline const std::unordered_set<std::string>& kBenchmarkInfeasible()
{
  static const std::unordered_set<std::string> kInfeas = {
    "bnatt500",
    "cryptanalysiskb128n5obj14",
    "fhnw-binpack4-4",
    "neos-2075418-temuka",
    "neos-3402454-bohle",
    "neos-3988577-wolgan",
    "neos859080",
  };
  return kInfeas;
}

inline std::optional<double> lookup_miplib_bks(const std::string& filename)
{
  const auto& m = kBenchmarkBKS();
  const auto it = m.find(normalize_instance_name(filename));
  if (it == m.end()) { return std::nullopt; }
  return it->second;
}

inline bool is_known_infeasible(const std::string& filename)
{
  return kBenchmarkInfeasible().count(normalize_instance_name(filename)) != 0;
}

// Single grep-friendly per-instance line. Emits to stdout via printf
// so the output survives unconditionally regardless of the project's
// settings_.log routing (NFS-backed log files, gated debug levels)
// and is trivially cross-compared between cuts-config branches.
//
// "Gap closed" is reported relative to the *root LP after cuts*, not
// relative to the final dual bound at the end of solve. The standard
// MIP cutting-plane definition is:
//   gap_closed_pct = 100 * (root_lp_with_cuts - root_lp_no_cuts)
//                          / (opt - root_lp_no_cuts)
// On a minimization-form problem all three differences are >= 0 and
// gap_closed_pct lies in [0, 100]. The ratio is sign-symmetric so the
// formula also holds verbatim for maximization (numerator and
// denominator flip sign together). NaN is emitted when either root
// bound was not published (e.g. B&B never entered the cut loop).
//
// Other field semantics (signed for minimization):
//   abs_root_dual_gap     = opt - root_lp_with_cuts
//   rel_root_dual_gap_pct = 100 * abs_root_dual_gap / max(|opt|, 1)
//   abs_primal_gap        = primal - opt
//   rel_primal_gap_pct    = 100 * abs_primal_gap / max(|opt|, 1)
//
// The line still also reports `final_dual` (solver's bound at the end
// of solve) so the new metric and the previous one can be compared
// without re-running.
//
// "TBD" is emitted when the BKS is unknown so downstream parsers
// can join lines on (instance, field) without dropping rows. "NaN" is
// emitted for root_lp_* when the value is unavailable.
template <typename Solution>
inline void print_miplib_gap_stat(
  const std::string& filename,
  const Solution& solution,
  double solve_time_seconds,
  const std::string& termination_status,
  double root_lp_no_cuts,
  double root_lp_with_cuts,
  double cut_gen_time_sec = std::numeric_limits<double>::quiet_NaN())
{
  const std::string norm   = normalize_instance_name(filename);
  const bool infeasible    = is_known_infeasible(filename);
  const auto bks_opt       = lookup_miplib_bks(filename);
  const double primal      = solution.get_objective_value();
  const double final_dual  = solution.get_solution_bound();
  const double mip_gap     = solution.get_mip_gap();
  const bool primal_finite = std::isfinite(primal);
  const bool root0_finite  = std::isfinite(root_lp_no_cuts);
  const bool root1_finite  = std::isfinite(root_lp_with_cuts);
  constexpr double NaN     = std::numeric_limits<double>::quiet_NaN();

  std::string line = std::format("MIPLIBGapStat instance={}", norm);

  if (infeasible) {
    line += " opt=Infeasible";
  } else if (bks_opt.has_value()) {
    line += std::format(" opt={:.10g}", *bks_opt);
  } else {
    line += " opt=TBD";
  }

  line += std::format(
    " primal={:.10g} final_dual={:.10g} root_lp_no_cuts={:.10g} root_lp_with_cuts={:.10g}",
    primal,
    final_dual,
    root_lp_no_cuts,
    root_lp_with_cuts);

  if (!infeasible && bks_opt.has_value()) {
    const double bks   = *bks_opt;
    const double denom = std::max(std::abs(bks), 1.0);

    const double abs_root_dgap     = root1_finite ? (bks - root_lp_with_cuts) : NaN;
    const double rel_root_dgap_pct = root1_finite ? 100.0 * abs_root_dgap / denom : NaN;

    // Classical gap-closed-by-cuts. Skip when either root bound is
    // missing, when the LP relaxation already proves optimality
    // (denominator = opt - root_lp_no_cuts ~= 0), or when the bound
    // moved the wrong way (numerical noise in either direction).
    double gap_closed_pct = NaN;
    if (root0_finite && root1_finite) {
      const double total_gap = bks - root_lp_no_cuts;
      if (std::abs(total_gap) > 1e-12 * denom) {
        gap_closed_pct = 100.0 * (root_lp_with_cuts - root_lp_no_cuts) / total_gap;
      } else {
        // LP relaxation already (numerically) optimal -> 100% closed
        // by definition. Avoid /0 noise.
        gap_closed_pct = 100.0;
      }
    }

    const double abs_pgap     = primal_finite ? (primal - bks) : NaN;
    const double rel_pgap_pct = primal_finite ? 100.0 * abs_pgap / denom : NaN;

    line += std::format(
      " abs_root_dual_gap={:.10g} rel_root_dual_gap_pct={:.6g} gap_closed_pct={:.6g}"
      " abs_primal_gap={:.10g} rel_primal_gap_pct={:.6g}",
      abs_root_dgap,
      rel_root_dgap_pct,
      gap_closed_pct,
      abs_pgap,
      rel_pgap_pct);
  } else {
    const char* na = infeasible ? "NA" : "TBD";
    line += std::format(
      " abs_root_dual_gap={0} rel_root_dual_gap_pct={0} gap_closed_pct={0}"
      " abs_primal_gap={0} rel_primal_gap_pct={0}",
      na);
  }

  line += std::format(" mip_gap_reported={:.6g} time_s={:.3f} cut_gen_time_s={:.3f} status={}",
                      mip_gap,
                      solve_time_seconds,
                      cut_gen_time_sec,
                      termination_status);

  std::printf("%s\n", line.c_str());
  std::fflush(stdout);
}

}  // namespace cuopt_bench
