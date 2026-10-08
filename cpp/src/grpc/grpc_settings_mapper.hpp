/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>

#include <cstddef>
#include <cstdint>

namespace cuopt::mathematical_optimization {

// Forward declarations
template <typename i_t, typename f_t>
struct pdlp_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_t;

template <typename i_t, typename f_t>
struct solver_settings_t;

/**
 * @brief Map pdlp_solver_settings_t to protobuf PDLPSolverSettings message.
 *
 * Populates a protobuf message using the generated protobuf C++ API.
 * Does not perform serialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_pdlp_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::PDLPSolverSettings* pb_settings);

/**
 * @brief Client export of PDLP settings.
 *
 * Writes warm start only. set_parameter() values are left for the caller to
 * put in the parameters map. A request built this way needs a server that
 * applies that map. map_pdlp_settings_to_proto still writes the deprecated
 * typed fields so an older client message can be reproduced in tests.
 */
template <typename i_t, typename f_t>
void map_pdlp_client_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
                                       cuopt::remote::PDLPSolverSettings* pb_settings);

/**
 * @brief Map protobuf PDLPSolverSettings message onto solver_settings_t.
 *
 * Reads from a protobuf message using the generated protobuf C++ API.
 * Does not perform deserialization — that is handled by the protobuf library.
 *
 * Deprecated set_parameter() fields are applied with set_parameter() only when
 * parameters is empty. An out-of-range value throws std::invalid_argument. A
 * non-empty map leaves those fields at the C++ defaults; the caller applies
 * the map. Warm start is always read. When @p n_variables and @p n_constraints
 * are non-negative, a warm start is checked against those dimensions before it
 * is stored. A negative count skips that check. Non-finite values and
 * iteration counters below the -1 sentinel are always rejected. Throws
 * std::invalid_argument and leaves the warm start unset.
 */
template <typename i_t, typename f_t>
void map_proto_to_pdlp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
                                solver_settings_t<i_t, f_t>& settings,
                                i_t n_variables   = -1,
                                i_t n_constraints = -1);

/**
 * @brief Bytes of PDLP warm start that ride in the settings message.
 *
 * Returns 0 when the warm start is empty. Otherwise an upper bound on the
 * protobuf size of PDLPWarmStartData. Chunked upload leaves this payload in
 * the header, so the caller counts it against both the unary size and the
 * message cap.
 */
template <typename i_t, typename f_t>
size_t estimate_pdlp_warm_start_proto_size(const pdlp_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Map mip_solver_settings_t to protobuf MIPSolverSettings message.
 *
 * Populates a protobuf message using the generated protobuf C++ API.
 * Does not perform serialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_mip_settings_to_proto(const mip_solver_settings_t<i_t, f_t>& settings,
                               cuopt::remote::MIPSolverSettings* pb_settings);

/**
 * @brief Client export of MIP settings.
 *
 * Writes presolve_absolute_tolerance, which is not a set_parameter() value.
 * Every set_parameter() value is left for the caller to put in the parameters
 * map. A request built this way needs a server that applies that map.
 */
template <typename i_t, typename f_t>
void map_mip_client_settings_to_proto(const mip_solver_settings_t<i_t, f_t>& settings,
                                      cuopt::remote::MIPSolverSettings* pb_settings);

/**
 * @brief Map protobuf MIPSolverSettings message onto solver_settings_t.
 *
 * Deprecated set_parameter() fields are applied with set_parameter() only when
 * parameters is empty. An out-of-range value throws std::invalid_argument. A
 * non-empty map leaves those fields at the C++ defaults; the caller applies
 * the map. presolve_absolute_tolerance is always read.
 */
template <typename i_t, typename f_t>
void map_proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
                               solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Write every registered solver parameter into the map.
 *
 * Keys are CUOPT_* names. A name registered on both LP and MIP must hold the
 * same value; otherwise this throws std::invalid_argument and nothing is
 * sent. set_parameter() writes every registration, so those values agree.
 * Names registered on only one side are included. Values are the text
 * set_parameter_from_string() parses. Floats use max_digits10. Call this
 * after the generated typed-field export.
 */
template <typename i_t, typename f_t>
void append_solver_parameters(const solver_settings_t<i_t, f_t>& settings,
                              google::protobuf::Map<std::string, std::string>* out);

/**
 * @brief Apply PDLPSolverSettings.parameters / MIPSolverSettings.parameters.
 *
 * Each entry is passed to set_parameter_from_string(). Only keys present in
 * the map are set. A name absent from the map stays at the C++ default when
 * the deprecated typed fields were skipped, which is the non-empty-map path.
 * An empty map changes nothing. An unknown name or an out-of-range value
 * throws std::invalid_argument. A map with more entries than twice the number
 * of registered parameter slots throws std::invalid_argument as well. That
 * limit follows the parameter tables.
 */
template <typename i_t, typename f_t>
void apply_parameter_overrides(solver_settings_t<i_t, f_t>& settings,
                               const google::protobuf::Map<std::string, std::string>& parameters);

}  // namespace cuopt::mathematical_optimization
