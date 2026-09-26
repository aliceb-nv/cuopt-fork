/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "gpu_branch_scores.cuh"

#include <climits>

namespace cuopt::linear_programming::dual_simplex::gpu_scores {

namespace {

__global__ void score_kernel(const double* x_all,
                             const unsigned char* is_integer,
                             int count,
                             int n,
                             pseudocosts_t pc,
                             seat_score_t* out)
{
  const int seat = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (seat >= count) { return; }
  const double* x = x_all + static_cast<size_t>(seat) * n;
  constexpr double eps = 1e-6;

  int num_frac = 0, unreliable = 0;
  double best_score = -1.0, estimate = 0.0;
  int best_var      = INT_MAX;
  // Each lane visits its columns in increasing order, so a strict comparison keeps the first
  // column of equal scores, as the host's loop over the (sorted) fractional list does.
  for (int j = lane; j < n; j += 32) {
    if (is_integer[j] == 0) { continue; }
    const double v = x[j];
    if (!(fabs(round(v) - v) > pc.integer_tol)) { continue; }
    ++num_frac;
    const int nd = pc.num_down[j], nu = pc.num_up[j];
    if (nd < pc.reliable_threshold || nu < pc.reliable_threshold) { unreliable = 1; }
    const double f_down  = v - floor(v);
    const double f_up    = ceil(v) - v;
    const double pc_down = nd > 0 ? pc.sum_down[j] / nd : pc.avg_down;
    const double pc_up   = nu > 0 ? pc.sum_up[j] / nu : pc.avg_up;
    const double score   = fmax(f_down * pc_down, eps) * fmax(f_up * pc_up, eps);
    estimate += fmin(pc_down * f_down, pc_up * f_up);
    if (score > best_score) {
      best_score = score;
      best_var   = j;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    num_frac += __shfl_down_sync(0xffffffffu, num_frac, offset);
    unreliable |= __shfl_down_sync(0xffffffffu, unreliable, offset);
    estimate += __shfl_down_sync(0xffffffffu, estimate, offset);
    const double other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
    const int other_var      = __shfl_down_sync(0xffffffffu, best_var, offset);
    if (other_score > best_score || (other_score == best_score && other_var < best_var)) {
      best_score = other_score;
      best_var   = other_var;
    }
  }
  if (lane == 0) {
    out[seat] = seat_score_t{num_frac, num_frac > 0 ? best_var : -1, unreliable, 0, estimate};
  }
}

}  // namespace

void score_seats(const double* x,
                 const unsigned char* is_integer,
                 int count,
                 int n,
                 const pseudocosts_t& pc,
                 seat_score_t* out,
                 cudaStream_t stream)
{
  if (count <= 0) { return; }
  constexpr int threads = 128;
  const int blocks      = (count * 32 + threads - 1) / threads;
  score_kernel<<<blocks, threads, 0, stream>>>(x, is_integer, count, n, pc, out);
}

}  // namespace cuopt::linear_programming::dual_simplex::gpu_scores
