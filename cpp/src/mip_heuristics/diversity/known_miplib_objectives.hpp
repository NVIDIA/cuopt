/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace cuopt::linear_programming::detail {

enum class objective_reference_status_t : int8_t { OPTIMAL = 0, BEST_KNOWN = 1 };

struct objective_reference_t {
  double objective_value;
  objective_reference_status_t status;
};

inline const char* objective_reference_status_name(objective_reference_status_t status)
{
  return status == objective_reference_status_t::OPTIMAL ? "optimal" : "best_known";
}

inline std::string normalize_problem_name(std::string problem_name)
{
  auto trim_ascii_whitespace = [](std::string& s) {
    auto is_ws = [](unsigned char c) {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    size_t begin = 0;
    while (begin < s.size() && is_ws(static_cast<unsigned char>(s[begin]))) {
      ++begin;
    }
    size_t end = s.size();
    while (end > begin && is_ws(static_cast<unsigned char>(s[end - 1]))) {
      --end;
    }
    s = s.substr(begin, end - begin);
  };

  trim_ascii_whitespace(problem_name);
  if (!problem_name.empty()) {
    const char first             = problem_name.front();
    const char last              = problem_name.back();
    const bool wrapped_in_quotes = (first == '"' && last == '"') || (first == '\'' && last == '\'');
    if (wrapped_in_quotes && problem_name.size() >= 2) {
      problem_name = problem_name.substr(1, problem_name.size() - 2);
      trim_ascii_whitespace(problem_name);
    }
  }

  const auto slash_pos = problem_name.find_last_of("/\\");
  if (slash_pos != std::string::npos) { problem_name = problem_name.substr(slash_pos + 1); }
  trim_ascii_whitespace(problem_name);
  std::transform(
    problem_name.begin(), problem_name.end(), problem_name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
  const std::array<std::string, 5> suffixes = {".mps.gz", ".mps.bz2", ".mps", ".gz", ".bz2"};
  auto ends_with                            = [](const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  bool removed = true;
  while (removed) {
    removed = false;
    for (const auto& suffix : suffixes) {
      if (ends_with(problem_name, suffix)) {
        problem_name.resize(problem_name.size() - suffix.size());
        removed = true;
        break;
      }
    }
  }
  return problem_name;
}

inline std::optional<objective_reference_t> lookup_known_objective_reference(
  std::string problem_name)
{
  // These benchmark names are intentionally absent because miplib2017-v36.solu does not provide
  // a finite reference objective for them:
  // bnatt500, cryptanalysiskb128n5obj14, fhnw-binpack4-4, neos-2075418-temuka,
  // neos-3402454-bohle, neos-3988577-wolgan, neos859080.
  static const std::unordered_map<std::string, objective_reference_t> k_objective_map = {
    {"30n20b8", {302, objective_reference_status_t::OPTIMAL}},
    {"50v-10", {3311.1799841000002, objective_reference_status_t::OPTIMAL}},
    {"academictimetablesmall", {0, objective_reference_status_t::OPTIMAL}},
    {"air05", {26374, objective_reference_status_t::OPTIMAL}},
    {"app1-1", {-3, objective_reference_status_t::OPTIMAL}},
    {"app1-2", {-41, objective_reference_status_t::OPTIMAL}},
    {"assign1-5-8", {211.99999999999801, objective_reference_status_t::OPTIMAL}},
    {"atlanta-ip", {90.009878614000002, objective_reference_status_t::OPTIMAL}},
    {"b1c1s1", {24544.25, objective_reference_status_t::OPTIMAL}},
    {"bab2", {-357544.31150000001, objective_reference_status_t::OPTIMAL}},
    {"bab6", {-284248.23070000007, objective_reference_status_t::OPTIMAL}},
    {"beasleyc3", {753.9999999999128, objective_reference_status_t::OPTIMAL}},
    {"binkar10_1", {6741.3800239397196, objective_reference_status_t::OPTIMAL}},
    {"blp-ar98", {6205.2147103999996, objective_reference_status_t::OPTIMAL}},
    {"blp-ic98", {4491.4475839500001, objective_reference_status_t::OPTIMAL}},
    {"bnatt400", {1, objective_reference_status_t::OPTIMAL}},
    {"bppc4-08", {53, objective_reference_status_t::OPTIMAL}},
    {"brazil3", {24, objective_reference_status_t::OPTIMAL}},
    {"buildingenergy", {33283.853236000003, objective_reference_status_t::OPTIMAL}},
    {"cbs-cta", {0, objective_reference_status_t::OPTIMAL}},
    {"chromaticindex1024-7", {4, objective_reference_status_t::OPTIMAL}},
    {"chromaticindex512-7", {4, objective_reference_status_t::OPTIMAL}},
    {"cmflsp50-24-8-8", {55789389.886, objective_reference_status_t::OPTIMAL}},
    {"cms750_4", {252, objective_reference_status_t::OPTIMAL}},
    {"co-100", {2639942.0600000001, objective_reference_status_t::OPTIMAL}},
    {"cod105", {-12, objective_reference_status_t::OPTIMAL}},
    {"comp07-2idx", {6, objective_reference_status_t::OPTIMAL}},
    {"comp21-2idx", {74, objective_reference_status_t::OPTIMAL}},
    {"cost266-uue", {25148940.55999998, objective_reference_status_t::OPTIMAL}},
    {"cryptanalysiskb128n5obj16", {0, objective_reference_status_t::OPTIMAL}},
    {"csched007", {350.99999999999551, objective_reference_status_t::OPTIMAL}},
    {"csched008", {173, objective_reference_status_t::OPTIMAL}},
    {"cvs16r128-89", {-97, objective_reference_status_t::OPTIMAL}},
    {"dano3_3", {576.34463302999995, objective_reference_status_t::OPTIMAL}},
    {"dano3_5", {576.9249159565619, objective_reference_status_t::OPTIMAL}},
    {"decomp2", {-160, objective_reference_status_t::OPTIMAL}},
    {"drayage-100-23", {103333.87407000001, objective_reference_status_t::OPTIMAL}},
    {"drayage-25-23", {101282.647018, objective_reference_status_t::OPTIMAL}},
    {"dws008-01", {37412.604587945083, objective_reference_status_t::OPTIMAL}},
    {"eil33-2", {934.007915999999, objective_reference_status_t::OPTIMAL}},
    {"eila101-2", {880.92010799999991, objective_reference_status_t::OPTIMAL}},
    {"enlight_hard", {37, objective_reference_status_t::OPTIMAL}},
    {"ex10", {100, objective_reference_status_t::OPTIMAL}},
    {"ex9", {81, objective_reference_status_t::OPTIMAL}},
    {"exp-1-500-5-5", {65887, objective_reference_status_t::OPTIMAL}},
    {"fast0507", {174, objective_reference_status_t::OPTIMAL}},
    {"fastxgemm-n2r6s0t2", {230, objective_reference_status_t::OPTIMAL}},
    {"fhnw-binpack4-48", {0, objective_reference_status_t::OPTIMAL}},
    {"fiball", {138, objective_reference_status_t::OPTIMAL}},
    {"gen-ip002", {-4783.7333920000001, objective_reference_status_t::OPTIMAL}},
    {"gen-ip054", {6840.9656417899996, objective_reference_status_t::OPTIMAL}},
    {"germanrr", {47095869.648999996, objective_reference_status_t::OPTIMAL}},
    {"gfd-schedulen180f7d50m30k18", {1, objective_reference_status_t::OPTIMAL}},
    {"glass-sc", {23, objective_reference_status_t::OPTIMAL}},
    {"glass4", {1200012599.972384, objective_reference_status_t::OPTIMAL}},
    {"gmu-35-40", {-2406733.3687999998, objective_reference_status_t::OPTIMAL}},
    {"gmu-35-50", {-2607958.3300000001, objective_reference_status_t::OPTIMAL}},
    {"graph20-20-1rand", {-9, objective_reference_status_t::OPTIMAL}},
    {"graphdraw-domain", {19685.999975500381, objective_reference_status_t::OPTIMAL}},
    {"h80x6320d", {6382.0990482459993, objective_reference_status_t::OPTIMAL}},
    {"highschool1-aigio", {0, objective_reference_status_t::OPTIMAL}},
    {"hypothyroid-k1", {-2851, objective_reference_status_t::OPTIMAL}},
    {"ic97_potential", {3941.9999309022501, objective_reference_status_t::OPTIMAL}},
    {"icir97_tension", {6375, objective_reference_status_t::OPTIMAL}},
    {"irish-electricity", {3723497.5913959998, objective_reference_status_t::OPTIMAL}},
    {"irp", {12159.492835396981, objective_reference_status_t::OPTIMAL}},
    {"istanbul-no-cutoff", {204.08170701, objective_reference_status_t::OPTIMAL}},
    {"k1mushroom", {-3288, objective_reference_status_t::OPTIMAL}},
    {"lectsched-5-obj", {24, objective_reference_status_t::OPTIMAL}},
    {"leo1", {404227536.16000003, objective_reference_status_t::OPTIMAL}},
    {"leo2", {404077441.12, objective_reference_status_t::OPTIMAL}},
    {"lotsize", {1480195, objective_reference_status_t::OPTIMAL}},
    {"mad", {0.026800000000000001, objective_reference_status_t::OPTIMAL}},
    {"map10", {-495, objective_reference_status_t::OPTIMAL}},
    {"map16715-04", {-111, objective_reference_status_t::OPTIMAL}},
    {"markshare2", {1, objective_reference_status_t::OPTIMAL}},
    {"markshare_4_0", {1, objective_reference_status_t::OPTIMAL}},
    {"mas74", {11801.185719999999, objective_reference_status_t::OPTIMAL}},
    {"mas76", {40005.053989999993, objective_reference_status_t::OPTIMAL}},
    {"mc11", {11688.99999999966, objective_reference_status_t::OPTIMAL}},
    {"mcsched", {211913, objective_reference_status_t::OPTIMAL}},
    {"mik-250-20-75-4", {-52301, objective_reference_status_t::OPTIMAL}},
    {"milo-v12-6-r2-40-1", {326481.14282799, objective_reference_status_t::OPTIMAL}},
    {"momentum1", {109143.4935, objective_reference_status_t::OPTIMAL}},
    {"mushroom-best", {0.055333761199999998, objective_reference_status_t::OPTIMAL}},
    {"mzzv11", {-21718, objective_reference_status_t::OPTIMAL}},
    {"mzzv42z", {-20540, objective_reference_status_t::OPTIMAL}},
    {"n2seq36q", {52200, objective_reference_status_t::OPTIMAL}},
    {"n3div36", {130800, objective_reference_status_t::OPTIMAL}},
    {"n5-3", {8104.9999999939992, objective_reference_status_t::OPTIMAL}},
    {"neos-1122047", {161, objective_reference_status_t::OPTIMAL}},
    {"neos-1171448", {-309, objective_reference_status_t::OPTIMAL}},
    {"neos-1171737", {-195, objective_reference_status_t::OPTIMAL}},
    {"neos-1354092", {46, objective_reference_status_t::OPTIMAL}},
    {"neos-1445765", {-17783, objective_reference_status_t::OPTIMAL}},
    {"neos-1456979", {176, objective_reference_status_t::OPTIMAL}},
    {"neos-1582420", {90.999999999999957, objective_reference_status_t::OPTIMAL}},
    {"neos-2657525-crna", {1.810748, objective_reference_status_t::OPTIMAL}},
    {"neos-2746589-doon", {2008.1999999999989, objective_reference_status_t::OPTIMAL}},
    {"neos-2978193-inde", {-2.3880616899999998, objective_reference_status_t::OPTIMAL}},
    {"neos-2987310-joes", {-607702988.29999995, objective_reference_status_t::OPTIMAL}},
    {"neos-3004026-krka", {0, objective_reference_status_t::OPTIMAL}},
    {"neos-3024952-loue", {26756, objective_reference_status_t::OPTIMAL}},
    {"neos-3046615-murg", {1600, objective_reference_status_t::OPTIMAL}},
    {"neos-3083819-nubu", {6307996, objective_reference_status_t::OPTIMAL}},
    {"neos-3216931-puriri", {71320, objective_reference_status_t::OPTIMAL}},
    {"neos-3381206-awhea", {453, objective_reference_status_t::OPTIMAL}},
    {"neos-3402294-bobin", {0.067249999999999491, objective_reference_status_t::OPTIMAL}},
    {"neos-3555904-turama", {-34.700000000000003, objective_reference_status_t::OPTIMAL}},
    {"neos-3627168-kasai", {988585.61999999976, objective_reference_status_t::OPTIMAL}},
    {"neos-3656078-kumeu", {-13172.200000000001, objective_reference_status_t::OPTIMAL}},
    {"neos-3754480-nidda", {12939.7540104743, objective_reference_status_t::OPTIMAL}},
    {"neos-4300652-rahue", {2.1415999999999999, objective_reference_status_t::OPTIMAL}},
    {"neos-4338804-snowy", {1471, objective_reference_status_t::OPTIMAL}},
    {"neos-4387871-tavua", {33.384729927000002, objective_reference_status_t::OPTIMAL}},
    {"neos-4413714-turia", {45.370167019999798, objective_reference_status_t::OPTIMAL}},
    {"neos-4532248-waihi", {61.599999999999987, objective_reference_status_t::OPTIMAL}},
    {"neos-4647030-tutaki", {27265.705999999958, objective_reference_status_t::OPTIMAL}},
    {"neos-4722843-widden", {25009.662227000001, objective_reference_status_t::OPTIMAL}},
    {"neos-4738912-atrato", {283627956.59500003, objective_reference_status_t::OPTIMAL}},
    {"neos-4763324-toguru", {1613.0388458499999, objective_reference_status_t::OPTIMAL}},
    {"neos-4954672-berkel", {2612710, objective_reference_status_t::OPTIMAL}},
    {"neos-5049753-cuanza", {561.99999716889999, objective_reference_status_t::OPTIMAL}},
    {"neos-5052403-cygnet", {182, objective_reference_status_t::OPTIMAL}},
    {"neos-5093327-huahum", {6259.9999971258949, objective_reference_status_t::OPTIMAL}},
    {"neos-5104907-jarama", {935, objective_reference_status_t::OPTIMAL}},
    {"neos-5107597-kakapo", {3644.9999999995198, objective_reference_status_t::OPTIMAL}},
    {"neos-5114902-kasavu", {655, objective_reference_status_t::OPTIMAL}},
    {"neos-5188808-nattai", {0.110283622999984, objective_reference_status_t::OPTIMAL}},
    {"neos-5195221-niemur", {0.0038354325999999999, objective_reference_status_t::OPTIMAL}},
    {"neos-631710", {203, objective_reference_status_t::OPTIMAL}},
    {"neos-662469", {184379.99999999991, objective_reference_status_t::OPTIMAL}},
    {"neos-787933", {30, objective_reference_status_t::OPTIMAL}},
    {"neos-827175", {112.00152, objective_reference_status_t::OPTIMAL}},
    {"neos-848589", {2351.40309999697, objective_reference_status_t::OPTIMAL}},
    {"neos-860300", {3200.9999999999982, objective_reference_status_t::OPTIMAL}},
    {"neos-873061", {113.6562385063, objective_reference_status_t::OPTIMAL}},
    {"neos-911970", {54.759999999999998, objective_reference_status_t::OPTIMAL}},
    {"neos-933966", {318, objective_reference_status_t::OPTIMAL}},
    {"neos-950242", {4, objective_reference_status_t::OPTIMAL}},
    {"neos-957323", {-237.75668150000001, objective_reference_status_t::OPTIMAL}},
    {"neos-960392", {-238, objective_reference_status_t::OPTIMAL}},
    {"neos17", {0.1500025774, objective_reference_status_t::OPTIMAL}},
    {"neos5", {15, objective_reference_status_t::OPTIMAL}},
    {"neos8", {-3719, objective_reference_status_t::OPTIMAL}},
    {"net12", {214, objective_reference_status_t::OPTIMAL}},
    {"netdiversion", {242, objective_reference_status_t::OPTIMAL}},
    {"nexp-150-20-8-5", {231, objective_reference_status_t::OPTIMAL}},
    {"ns1116954", {0, objective_reference_status_t::OPTIMAL}},
    {"ns1208400", {2, objective_reference_status_t::OPTIMAL}},
    {"ns1644855", {-1524.3333333333301, objective_reference_status_t::OPTIMAL}},
    {"ns1760995", {-549.21438505000003, objective_reference_status_t::OPTIMAL}},
    {"ns1830653", {20622, objective_reference_status_t::OPTIMAL}},
    {"ns1952667", {0, objective_reference_status_t::OPTIMAL}},
    {"nu25-pr12", {53904.999999999993, objective_reference_status_t::OPTIMAL}},
    {"nursesched-medium-hint03", {115, objective_reference_status_t::OPTIMAL}},
    {"nursesched-sprint02", {57.999999999999993, objective_reference_status_t::OPTIMAL}},
    {"nw04", {16862, objective_reference_status_t::OPTIMAL}},
    {"opm2-z10-s4", {-33269, objective_reference_status_t::OPTIMAL}},
    {"p200x1188c", {15078, objective_reference_status_t::OPTIMAL}},
    {"peg-solitaire-a3", {1, objective_reference_status_t::OPTIMAL}},
    {"pg", {-8674.3426071199992, objective_reference_status_t::OPTIMAL}},
    {"pg5_34", {-14339.353450000001, objective_reference_status_t::OPTIMAL}},
    {"physiciansched3-3", {2623271.3266670001, objective_reference_status_t::OPTIMAL}},
    {"physiciansched6-2", {49324, objective_reference_status_t::OPTIMAL}},
    {"piperout-08", {125054.9999999999, objective_reference_status_t::OPTIMAL}},
    {"piperout-27", {8123.9999999999727, objective_reference_status_t::OPTIMAL}},
    {"pk1", {11, objective_reference_status_t::OPTIMAL}},
    {"proteindesign121hz512p9", {1473, objective_reference_status_t::OPTIMAL}},
    {"proteindesign122trx11p8", {1747, objective_reference_status_t::OPTIMAL}},
    {"qap10", {339.99999999838712, objective_reference_status_t::OPTIMAL}},
    {"radiationm18-12-05", {17566, objective_reference_status_t::OPTIMAL}},
    {"radiationm40-10-02", {155328, objective_reference_status_t::OPTIMAL}},
    {"rail01", {-70.569964299999995, objective_reference_status_t::OPTIMAL}},
    {"rail02", {-200.44990770000001, objective_reference_status_t::OPTIMAL}},
    {"rail507", {174, objective_reference_status_t::OPTIMAL}},
    {"ran14x18-disj-8", {3712, objective_reference_status_t::OPTIMAL}},
    {"rd-rplusc-21", {165395.275295, objective_reference_status_t::OPTIMAL}},
    {"reblock115", {-36800603.233199999, objective_reference_status_t::OPTIMAL}},
    {"rmatr100-p10", {423, objective_reference_status_t::OPTIMAL}},
    {"rmatr200-p5", {4521, objective_reference_status_t::OPTIMAL}},
    {"roci-4-11", {-6020203, objective_reference_status_t::OPTIMAL}},
    {"rocii-5-11", {-6.6755047315380001, objective_reference_status_t::OPTIMAL}},
    {"rococob10-011000", {19449, objective_reference_status_t::OPTIMAL}},
    {"rocococ10-001000", {11460, objective_reference_status_t::OPTIMAL}},
    {"roi2alpha3n4", {-63.208495030000002, objective_reference_status_t::OPTIMAL}},
    {"roi5alpha10n8", {-52.322274350999997, objective_reference_status_t::OPTIMAL}},
    {"roll3000", {12889.999991999999, objective_reference_status_t::OPTIMAL}},
    {"s100", {-0.16972352705829999, objective_reference_status_t::OPTIMAL}},
    {"s250r10", {-0.17178048342319999, objective_reference_status_t::OPTIMAL}},
    {"satellites2-40", {-19, objective_reference_status_t::OPTIMAL}},
    {"satellites2-60-fs", {-19.000000000099998, objective_reference_status_t::OPTIMAL}},
    {"savsched1", {3217.6999999999998, objective_reference_status_t::OPTIMAL}},
    {"sct2", {-230.9891623, objective_reference_status_t::OPTIMAL}},
    {"seymour", {423, objective_reference_status_t::OPTIMAL}},
    {"seymour1", {410.76370138999999, objective_reference_status_t::OPTIMAL}},
    {"sing326", {7753674.8537600003, objective_reference_status_t::OPTIMAL}},
    {"sing44", {8128831.1771999998, objective_reference_status_t::OPTIMAL}},
    {"snp-02-004-104", {586803238.65672886, objective_reference_status_t::OPTIMAL}},
    {"sorrell3", {-16, objective_reference_status_t::OPTIMAL}},
    {"sp150x300d", {69, objective_reference_status_t::OPTIMAL}},
    {"sp97ar", {660705645.75899994, objective_reference_status_t::OPTIMAL}},
    {"sp98ar", {529740623.19999999, objective_reference_status_t::OPTIMAL}},
    {"splice1k1", {-394, objective_reference_status_t::OPTIMAL}},
    {"square41", {15, objective_reference_status_t::OPTIMAL}},
    {"square47", {15.9999999997877, objective_reference_status_t::OPTIMAL}},
    {"supportcase10", {7, objective_reference_status_t::OPTIMAL}},
    {"supportcase12", {-7559.5330538170001, objective_reference_status_t::OPTIMAL}},
    {"supportcase18", {48, objective_reference_status_t::OPTIMAL}},
    {"supportcase19", {12677205.999920519, objective_reference_status_t::OPTIMAL}},
    {"supportcase22", {110, objective_reference_status_t::BEST_KNOWN}},
    {"supportcase26", {1745.1238129999999, objective_reference_status_t::OPTIMAL}},
    {"supportcase33", {-345, objective_reference_status_t::OPTIMAL}},
    {"supportcase40", {24256.3122898, objective_reference_status_t::OPTIMAL}},
    {"supportcase42", {7.7586307222700004, objective_reference_status_t::OPTIMAL}},
    {"supportcase6", {51906.477370000001, objective_reference_status_t::OPTIMAL}},
    {"supportcase7", {-1132.2231770000001, objective_reference_status_t::OPTIMAL}},
    {"swath1", {379.07129574999999, objective_reference_status_t::OPTIMAL}},
    {"swath3", {397.76134365000001, objective_reference_status_t::OPTIMAL}},
    {"tbfp-network", {24.163194440000002, objective_reference_status_t::OPTIMAL}},
    {"thor50dday", {40417, objective_reference_status_t::OPTIMAL}},
    {"timtab1", {764771.99999977998, objective_reference_status_t::OPTIMAL}},
    {"tr12-30", {130595.9999999999, objective_reference_status_t::OPTIMAL}},
    {"traininstance2", {71820, objective_reference_status_t::OPTIMAL}},
    {"traininstance6", {28290, objective_reference_status_t::OPTIMAL}},
    {"trento1", {5189487, objective_reference_status_t::OPTIMAL}},
    {"triptim1", {22.868099999999899, objective_reference_status_t::OPTIMAL}},
    {"uccase12", {11507.4050616, objective_reference_status_t::OPTIMAL}},
    {"uccase9", {10993.131409, objective_reference_status_t::OPTIMAL}},
    {"uct-subprob", {314, objective_reference_status_t::OPTIMAL}},
    {"unitcal_7", {19635558.243999999, objective_reference_status_t::OPTIMAL}},
    {"var-smallemery-m6j6", {-149.37501, objective_reference_status_t::OPTIMAL}},
    {"wachplan", {-8, objective_reference_status_t::OPTIMAL}},
  };
  const auto normalized = normalize_problem_name(std::move(problem_name));
  const auto iter       = k_objective_map.find(normalized);
  if (iter == k_objective_map.end()) { return std::nullopt; }
  return iter->second;
}

}  // namespace cuopt::linear_programming::detail
