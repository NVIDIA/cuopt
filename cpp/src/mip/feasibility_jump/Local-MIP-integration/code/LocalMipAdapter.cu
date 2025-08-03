/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "LocalMipAdapter.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <mip/mip_constants.hpp>
#include <mip/utils.cuh>

#include <linear_programming/pdlp.cuh>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/linalg/binary_op.cuh>

#include <thrust/tabulate.h>

namespace cuopt::linear_programming::detail {

void LocalMipRead(Solver& solver,
                  const problem_t<int32_t, double>& problem,
                  solution_t<int32_t, double>& solution)
{
  auto& modelConUtil = solver.modelConUtil;
  auto& modelVarUtil = solver.modelVarUtil;
  auto& readerMPS    = solver.readerMPS;

  auto handle_ptr = problem.handle_ptr;
  // Get host copies of device data
  auto h_reverse_coefficients =
    cuopt::host_copy(problem.reverse_coefficients, handle_ptr->get_stream());
  auto h_reverse_constraints =
    cuopt::host_copy(problem.reverse_constraints, handle_ptr->get_stream());
  auto h_reverse_offsets = cuopt::host_copy(problem.reverse_offsets, handle_ptr->get_stream());
  auto h_coefficients    = cuopt::host_copy(problem.coefficients, handle_ptr->get_stream());
  auto h_offsets         = cuopt::host_copy(problem.offsets, handle_ptr->get_stream());
  auto h_variables       = cuopt::host_copy(problem.variables, handle_ptr->get_stream());
  auto h_obj_coeffs = cuopt::host_copy(problem.objective_coefficients, handle_ptr->get_stream());
  auto h_var_lb     = cuopt::host_copy(problem.variable_lower_bounds, handle_ptr->get_stream());
  auto h_var_ub     = cuopt::host_copy(problem.variable_upper_bounds, handle_ptr->get_stream());
  auto h_cstr_lb    = cuopt::host_copy(problem.constraint_lower_bounds, handle_ptr->get_stream());
  auto h_cstr_ub    = cuopt::host_copy(problem.constraint_upper_bounds, handle_ptr->get_stream());
  auto h_var_types  = cuopt::host_copy(problem.variable_types, handle_ptr->get_stream());
  auto h_lhs        = cuopt::host_copy(solution.constraint_value, handle_ptr->get_stream());

  modelConUtil->conSet.emplace_back("", 0);  // obj
  for (int32_t conIdx = 0; conIdx < problem.n_constraints; ++conIdx) {
    std::string con_name = "R" + std::to_string(conIdx);
    auto lb              = h_cstr_lb[conIdx];
    auto ub              = h_cstr_ub[conIdx];

    cuopt_assert(isfinite(lb) || isfinite(ub), "constraint bounds are not finite");

    // create two constraints for both bounds
    int32_t lmip_conIdx;
    if (isfinite(lb) && isfinite(ub)) {
      lmip_conIdx                                 = modelConUtil->MakeCon(con_name);
      modelConUtil->conSet[lmip_conIdx].isEqual   = true;
      std::string inverseConName                  = con_name + "!";
      int32_t inverseConIdx                       = modelConUtil->MakeCon(inverseConName);
      modelConUtil->conSet[inverseConIdx].isEqual = true;
    }
    // larger than
    else if (isfinite(lb)) {
      lmip_conIdx                               = modelConUtil->MakeCon(con_name);
      modelConUtil->conSet[lmip_conIdx].isLarge = true;
    }
    // lower than
    else if (isfinite(ub)) {
      lmip_conIdx = modelConUtil->MakeCon(con_name);
    } else {
      cuopt_assert(false, "constraint bounds are not finite");
      lmip_conIdx = -1;
    }

    // RHS
    modelConUtil->conSet[lmip_conIdx].RHS = modelConUtil->conSet[lmip_conIdx].isLarge ? lb : ub;
    if (modelConUtil->conSet[lmip_conIdx].isEqual) modelConUtil->conSet[lmip_conIdx + 1].RHS = -lb;

    cuopt_assert(isfinite(modelConUtil->conSet[lmip_conIdx].RHS), "constraint RHS is not finite");

    // add coefficients
    auto [offset_begin, offset_end] = std::make_pair(h_offsets[conIdx], h_offsets[conIdx + 1]);
    for (int32_t j = offset_begin; j < offset_end; ++j) {
      int32_t var_idx      = h_variables[j];
      double coefficient   = h_coefficients[j];
      std::string var_name = "V" + std::to_string(var_idx);

      readerMPS->integralityMarker = h_var_types[var_idx] == var_t::INTEGER;
      readerMPS->PushCoeffVarIdx(lmip_conIdx, coefficient, var_name);
      if (modelConUtil->conSet[lmip_conIdx].isEqual)
        readerMPS->PushCoeffVarIdx(lmip_conIdx + 1, -coefficient, var_name);
    }
  }

  for (int var_idx = 0; var_idx < problem.n_variables; ++var_idx) {
    if (h_obj_coeffs[var_idx] != 0) {
      std::string var_name         = "V" + std::to_string(var_idx);
      readerMPS->integralityMarker = h_var_types[var_idx] == var_t::INTEGER;
      readerMPS->PushCoeffVarIdx(0, h_obj_coeffs[var_idx], var_name);
    }
  }

  // BOUNDS section
  for (int32_t varIdx = 0; varIdx < problem.n_variables; ++varIdx) {
    std::string var_name = "V" + std::to_string(varIdx);
    auto& var            = modelVarUtil->GetVar(var_name);
    var.SetLowerBound(h_var_lb[varIdx]);
    var.SetUpperBound(h_var_ub[varIdx]);
    cuopt_assert(h_var_lb[varIdx] <= h_var_ub[varIdx] + FeasibilityTol,
                 "variable bounds are not valid");
  }

  modelConUtil->conSet[0].RHS = -problem.get_user_obj_from_solver_obj(0);

  for (int conIdx = 1; conIdx < (int)modelConUtil->conSet.size(); ++conIdx) {
    auto& con = modelConUtil->conSet[conIdx];
    if (con.isLarge) {
      for (Value& inverseCoefficient : con.coeffSet)
        inverseCoefficient = -inverseCoefficient;
      con.RHS = -con.RHS;
    }
  }

  modelVarUtil->objBias = -modelConUtil->conSet[0].RHS;
  modelConUtil->conNum  = modelConUtil->conSet.size();
  modelVarUtil->varNum  = modelVarUtil->varSet.size();

  readerMPS->SetVarType();
  readerMPS->SetVarIdx2ObjIdx();

  solver.localMIP->Allocate();

  // initialize solution
  auto& localMIP    = solver.localMIP;
  auto h_assignment = cuopt::host_copy(solution.assignment, handle_ptr->get_stream());
  for (int var_idx = 0; var_idx < problem.n_variables; ++var_idx) {
    std::string var_name                                = "V" + std::to_string(var_idx);
    size_t lmip_varIdx                                  = modelVarUtil->GetVarIdx(var_name);
    localMIP->localVarUtil.GetVar(lmip_varIdx).nowValue = h_assignment[var_idx];

    auto& var = modelVarUtil->GetVar(var_name);
    var.coeffs.resize(var.termNum);

    for (size_t termIdx = 0; termIdx < var.termNum; ++termIdx) {
      size_t conIdx       = var.conIdxSet[termIdx];
      size_t posInCon     = var.posInCon[termIdx];
      var.coeffs[termIdx] = modelConUtil->conSet[conIdx].coeffSet[posInCon];
      if (modelConUtil->conSet[conIdx].invCoeffs.size() <
          modelConUtil->conSet[conIdx].coeffSet.size())
        modelConUtil->conSet[conIdx].invCoeffs.resize(modelConUtil->conSet[conIdx].coeffSet.size());
      modelConUtil->conSet[conIdx].invCoeffs[posInCon] = 1.0 / var.coeffs[termIdx];
    }
  }
  if (problem.n_variables != (int)modelVarUtil->varNum) {
    CUOPT_LOG_ERROR(
      "number of variables mismatch: %d != %d", problem.n_variables, modelVarUtil->varNum);
    cuopt_assert(false, "number of variables mismatch");
  }

  // lets check constraints.
  for (int32_t conIdx = 0; conIdx < problem.n_constraints; ++conIdx) {
    std::string con_name = "R" + std::to_string(conIdx);
    auto lb              = h_cstr_lb[conIdx];
    auto ub              = h_cstr_ub[conIdx];

    auto& con = modelConUtil->GetCon(con_name);
    if (isfinite(lb) && isfinite(ub)) {
      cuopt_assert(con.isEqual, "constraint is not equal");
      cuopt_assert(con.RHS == ub, "constraint RHS mismatch");
    } else if (isfinite(lb)) {
      cuopt_assert(con.isLarge, "constraint is not larger than");
      cuopt_assert(con.RHS == -lb, "constraint RHS mismatch");
    } else if (isfinite(ub)) {
      cuopt_assert(con.RHS == ub, "constraint RHS mismatch");
    }

    if (lb > h_lhs[conIdx] + FeasibilityTol || ub < h_lhs[conIdx] - FeasibilityTol) {
      double tolerance = get_cstr_tolerance<int, double>(
        lb, ub, problem.tolerances.absolute_tolerance, problem.tolerances.relative_tolerance);

      // printf("GPU side constraint %s, lhs vs rhs: %g vs [%g,%g], tol %g\n", con_name.c_str(),
      // h_lhs[conIdx], lb, ub, tolerance);
    }
  }
}

void CopyWeights(Solver& solver, fj_t<int32_t, double>& fj)
{
  auto& modelConUtil = solver.modelConUtil;
  auto& problem      = *fj.pb_ptr;
  auto handle_ptr    = problem.handle_ptr;
  auto& localConUtil = solver.localMIP->localConUtil;

  auto h_cstr_weights     = cuopt::host_copy(fj.cstr_weights);
  auto h_objective_weight = fj.objective_weight.value(rmm::cuda_stream_default);
  cuopt_assert(h_cstr_weights.size() == (size_t)problem.n_constraints, "size mismatch");
  if (h_cstr_weights.size() != (size_t)problem.n_constraints) {
    printf("h_cstr_weights.size() %d, problem.n_constraints %d\n",
           (int)h_cstr_weights.size(),
           problem.n_constraints);
    // exit(1);
    //  weird. as a workaround, just set the weights to 1
    h_cstr_weights.resize(problem.n_constraints);
    std::fill(h_cstr_weights.begin(), h_cstr_weights.end(), 1.0);
  }
  for (int conIdx = 0; conIdx < problem.n_constraints; ++conIdx) {
    std::string con_name = "R" + std::to_string(conIdx);
    auto& con            = localConUtil.GetCon(modelConUtil->GetConIdx(con_name));
    // printf("conIdx %d, n_constraints %d, weight count %d\n", conIdx, problem.n_constraints,
    // (int)h_cstr_weights.size());
    con.weight       = round(h_cstr_weights[conIdx]);
    size_t con_idx_2 = modelConUtil->GetConIdx(con_name);
    if (con_idx_2) {
      auto& con_2  = localConUtil.GetCon(con_idx_2);
      con_2.weight = con.weight;
    }
  }
  localConUtil.GetCon(0).weight = round(h_objective_weight);
}

double GetSolution(Solver& solver, std::vector<double>& solution, bool get_current)
{
  auto& localMIP     = solver.localMIP;
  auto& modelVarUtil = solver.modelVarUtil;
  solution.resize(modelVarUtil->varNum);

  // solver.localMIP->Allocate();

  for (size_t var_idx = 0; var_idx < modelVarUtil->varNum; ++var_idx) {
    std::string var_name = "V" + std::to_string(var_idx);
    size_t lmip_varIdx   = modelVarUtil->GetVarIdx(var_name);
    solution[var_idx]    = get_current ? localMIP->localVarUtil.GetVar(lmip_varIdx).nowValue
                                       : localMIP->localVarUtil.GetVar(lmip_varIdx).bestValue;
  }
  auto& localObj = localMIP->localConUtil.conSet[0];
  return get_current ? localObj.LHS : localMIP->bestOBJ;
}

void GetSolution(Solver& solver, solution_t<int32_t, double>& solution)
{
  auto& localMIP     = solver.localMIP;
  auto& modelVarUtil = solver.modelVarUtil;
  auto& problem      = *solution.problem_ptr;
  auto handle_ptr    = problem.handle_ptr;
  auto h_assignment  = cuopt::host_copy(solution.assignment, handle_ptr->get_stream());

  for (int var_idx = 0; var_idx < problem.n_variables; ++var_idx) {
    std::string var_name  = "V" + std::to_string(var_idx);
    size_t lmip_varIdx    = modelVarUtil->GetVarIdx(var_name);
    h_assignment[var_idx] = localMIP->localVarUtil.GetVar(lmip_varIdx).bestValue;
  }

  solution.copy_new_assignment(h_assignment);
}

void CopySolution(Solver& solver, solution_t<int32_t, double>& solution)
{
  auto& localMIP     = solver.localMIP;
  auto& modelVarUtil = solver.modelVarUtil;
  auto& problem      = *solution.problem_ptr;
  auto handle_ptr    = problem.handle_ptr;

  localMIP->Allocate();

  // initialize solution
  auto h_assignment = cuopt::host_copy(solution.assignment, handle_ptr->get_stream());
  for (int var_idx = 0; var_idx < problem.n_variables; ++var_idx) {
    std::string var_name                                = "V" + std::to_string(var_idx);
    size_t lmip_varIdx                                  = modelVarUtil->GetVarIdx(var_name);
    localMIP->localVarUtil.GetVar(lmip_varIdx).nowValue = h_assignment[var_idx];
  }
  if (problem.n_variables != (int)modelVarUtil->varNum) {
    CUOPT_LOG_ERROR(
      "number of variables mismatch: %d != %d", problem.n_variables, modelVarUtil->varNum);
    cuopt_assert(false, "number of variables mismatch");
  }
}
}  // namespace cuopt::linear_programming::detail