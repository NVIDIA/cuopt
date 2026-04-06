/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/mip/heuristics_hyper_params.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace cuopt::linear_programming::test {

using settings_t = solver_settings_t<int, double>;

class HeuristicsHyperParamsTest : public ::testing::Test {
 protected:
  std::string tmp_path;

  void SetUp() override
  {
    tmp_path = std::filesystem::temp_directory_path() / "cuopt_heuristic_params_test.config";
  }

  void TearDown() override { std::remove(tmp_path.c_str()); }
};

TEST_F(HeuristicsHyperParamsTest, DumpedFileIsAllCommentedOut)
{
  settings_t settings;
  settings.dump_parameters_to_file(tmp_path, true);

  // Loading the commented-out dump should leave struct defaults unchanged
  settings_t reloaded;
  reloaded.get_mip_settings().heuristic_params.population_size = 9999;
  reloaded.load_parameters_from_file(tmp_path);
  EXPECT_EQ(reloaded.get_mip_settings().heuristic_params.population_size, 9999);
}

TEST_F(HeuristicsHyperParamsTest, DumpedFileIsParseable)
{
  settings_t settings;
  settings.dump_parameters_to_file(tmp_path, true);

  settings_t reloaded;
  EXPECT_NO_THROW(reloaded.load_parameters_from_file(tmp_path));
}

TEST_F(HeuristicsHyperParamsTest, CustomValuesRoundTrip)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_population_size = 64\n";
    f << "mip_hyper_heuristic_num_cpufj_threads = 4\n";
    f << "mip_hyper_heuristic_presolve_time_ratio = 0.2\n";
    f << "mip_hyper_heuristic_presolve_max_time = 120\n";
    f << "mip_hyper_heuristic_root_lp_time_ratio = 0.05\n";
    f << "mip_hyper_heuristic_root_lp_max_time = 30\n";
    f << "mip_hyper_heuristic_rins_time_limit = 5\n";
    f << "mip_hyper_heuristic_rins_max_time_limit = 40\n";
    f << "mip_hyper_heuristic_rins_fix_rate = 0.7\n";
    f << "mip_hyper_heuristic_stagnation_trigger = 5\n";
    f << "mip_hyper_heuristic_max_iterations_without_improvement = 12\n";
    f << "mip_hyper_heuristic_initial_infeasibility_weight = 500\n";
    f << "mip_hyper_heuristic_n_of_minimums_for_exit = 10000\n";
    f << "mip_hyper_heuristic_enabled_recombiners = 5\n";
    f << "mip_hyper_heuristic_cycle_detection_length = 50\n";
    f << "mip_hyper_heuristic_relaxed_lp_time_limit = 2\n";
    f << "mip_hyper_heuristic_related_vars_time_limit = 60\n";
  }

  settings_t settings;
  settings.load_parameters_from_file(tmp_path);
  const auto& hp = settings.get_mip_settings().heuristic_params;

  EXPECT_EQ(hp.population_size, 64);
  EXPECT_EQ(hp.num_cpufj_threads, 4);
  EXPECT_DOUBLE_EQ(hp.presolve_time_ratio, 0.2);
  EXPECT_DOUBLE_EQ(hp.presolve_max_time, 120.0);
  EXPECT_DOUBLE_EQ(hp.root_lp_time_ratio, 0.05);
  EXPECT_DOUBLE_EQ(hp.root_lp_max_time, 30.0);
  EXPECT_DOUBLE_EQ(hp.rins_time_limit, 5.0);
  EXPECT_DOUBLE_EQ(hp.rins_max_time_limit, 40.0);
  EXPECT_DOUBLE_EQ(hp.rins_fix_rate, 0.7);
  EXPECT_EQ(hp.stagnation_trigger, 5);
  EXPECT_EQ(hp.max_iterations_without_improvement, 12);
  EXPECT_DOUBLE_EQ(hp.initial_infeasibility_weight, 500.0);
  EXPECT_EQ(hp.n_of_minimums_for_exit, 10000);
  EXPECT_EQ(hp.enabled_recombiners, 5);
  EXPECT_EQ(hp.cycle_detection_length, 50);
  EXPECT_DOUBLE_EQ(hp.relaxed_lp_time_limit, 2.0);
  EXPECT_DOUBLE_EQ(hp.related_vars_time_limit, 60.0);
}

TEST_F(HeuristicsHyperParamsTest, PartialConfigKeepsDefaults)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_population_size = 128\n";
    f << "mip_hyper_heuristic_rins_fix_rate = 0.3\n";
  }

  settings_t settings;
  settings.load_parameters_from_file(tmp_path);
  const auto& hp = settings.get_mip_settings().heuristic_params;

  EXPECT_EQ(hp.population_size, 128);
  EXPECT_DOUBLE_EQ(hp.rins_fix_rate, 0.3);

  mip_heuristics_hyper_params_t defaults;
  EXPECT_EQ(hp.num_cpufj_threads, defaults.num_cpufj_threads);
  EXPECT_DOUBLE_EQ(hp.presolve_time_ratio, defaults.presolve_time_ratio);
  EXPECT_EQ(hp.n_of_minimums_for_exit, defaults.n_of_minimums_for_exit);
  EXPECT_EQ(hp.enabled_recombiners, defaults.enabled_recombiners);
}

TEST_F(HeuristicsHyperParamsTest, CommentsAndBlankLinesIgnored)
{
  {
    std::ofstream f(tmp_path);
    f << "# This is a comment\n";
    f << "\n";
    f << "# Another comment\n";
    f << "mip_hyper_heuristic_population_size = 42\n";
    f << "\n";
  }

  settings_t settings;
  settings.load_parameters_from_file(tmp_path);
  EXPECT_EQ(settings.get_mip_settings().heuristic_params.population_size, 42);
}

TEST_F(HeuristicsHyperParamsTest, UnknownKeyThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "bogus_key = 42\n";
  }
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file(tmp_path), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, BadNumericValueThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_population_size = not_a_number\n";
  }
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file(tmp_path), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, TrailingJunkSpaceSeparatedThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_population_size = 64 foo\n";
  }
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file(tmp_path), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, TrailingJunkNoSpaceThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_population_size = 64foo\n";
  }
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file(tmp_path), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, TrailingJunkFloatThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_rins_fix_rate = 0.5abc\n";
  }
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file(tmp_path), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, RangeViolationCycleDetectionThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_cycle_detection_length = 0\n";
  }
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file(tmp_path), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, RangeViolationFixRateThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_rins_fix_rate = 2.0\n";
  }
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file(tmp_path), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, NonexistentFileThrows)
{
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file("/tmp/does_not_exist_cuopt_test.config"),
               cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, DirectoryPathThrows)
{
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file("/tmp"), cuopt::logic_error);
}

TEST_F(HeuristicsHyperParamsTest, IndentedCommentAndWhitespaceLinesIgnored)
{
  {
    std::ofstream f(tmp_path);
    f << "   # indented comment\n";
    f << "  \t  \n";
    f << "mip_hyper_heuristic_population_size = 99\n";
  }
  settings_t settings;
  settings.load_parameters_from_file(tmp_path);
  EXPECT_EQ(settings.get_mip_settings().heuristic_params.population_size, 99);
}

TEST_F(HeuristicsHyperParamsTest, MixedSolverAndHyperParamsFromFile)
{
  {
    std::ofstream f(tmp_path);
    f << "mip_hyper_heuristic_population_size = 100\n";
    f << "time_limit = 42\n";
  }
  settings_t settings;
  settings.load_parameters_from_file(tmp_path);
  EXPECT_EQ(settings.get_mip_settings().heuristic_params.population_size, 100);
  EXPECT_DOUBLE_EQ(settings.get_mip_settings().time_limit, 42.0);
}

TEST_F(HeuristicsHyperParamsTest, QuotedStringValue)
{
  {
    std::ofstream f(tmp_path);
    f << "log_file = \"/path/with spaces/log.txt\"\n";
  }
  settings_t settings;
  settings.load_parameters_from_file(tmp_path);
  EXPECT_EQ(settings.template get_parameter<std::string>(CUOPT_LOG_FILE),
            "/path/with spaces/log.txt");
}

TEST_F(HeuristicsHyperParamsTest, QuotedStringWithEscapedQuote)
{
  {
    std::ofstream f(tmp_path);
    f << R"(log_file = "/path/with \"quotes\"/log.txt")" << "\n";
  }
  settings_t settings;
  settings.load_parameters_from_file(tmp_path);
  EXPECT_EQ(settings.template get_parameter<std::string>(CUOPT_LOG_FILE),
            "/path/with \"quotes\"/log.txt");
}

TEST_F(HeuristicsHyperParamsTest, UnterminatedQuoteThrows)
{
  {
    std::ofstream f(tmp_path);
    f << "log_file = \"/path/no/close\n";
  }
  settings_t settings;
  EXPECT_THROW(settings.load_parameters_from_file(tmp_path), cuopt::logic_error);
}

}  // namespace cuopt::linear_programming::test
