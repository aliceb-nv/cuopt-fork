/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
// Pseudocost branching scores of a batch of solved seats, on the device (CUOPT_BNB_GPU_DEVICE_SCORES).
//
// Per seat, from its primal x: the number of fractional integer columns, whether any of them is
// unreliable (fewer than `reliable_threshold` down or up observations), the column the host's
// pseudocost selection picks when none is (largest score, first column on ties), and the
// pseudocost objective estimate increment. The formulas are the host's (pseudo_costs.cpp:
// is_fractional, calculate_pseudocost_score, obj_estimate); the pseudocosts are a snapshot taken
// when the generation was staged.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace cuopt::linear_programming::dual_simplex::gpu_scores {

struct pseudocosts_t {
  const double* sum_down;
  const double* sum_up;
  const int* num_down;
  const int* num_up;
  double avg_down, avg_up;  // for columns without observations
  int reliable_threshold;
  double integer_tol;
};

struct seat_score_t {
  int num_frac;
  int best_var;     // -1 without fractional columns
  int unreliable;   // 1 when the host's strong-branching path applies
  int pad;
  double estimate;  // added to the node's lower bound by the host
};

// `x`: count seats of n values; `is_integer`: n flags. One warp per seat.
void score_seats(const double* x,
                 const unsigned char* is_integer,
                 int count,
                 int n,
                 const pseudocosts_t& pc,
                 seat_score_t* out,
                 cudaStream_t stream);

}  // namespace cuopt::linear_programming::dual_simplex::gpu_scores
