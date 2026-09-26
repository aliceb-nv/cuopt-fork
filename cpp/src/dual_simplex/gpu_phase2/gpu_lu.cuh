/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

// Device port of the host basis factorization (dual_simplex factorize_basis + right_looking_lu):
// the same singleton elimination, the same Markowitz search with threshold partial pivoting on the
// nucleus, the same L/U layout and permutations, then the basic-list reordering and the two
// transposes refactor_basis produces. One thread factorizes one basis. Basis repair is not ported:
// a rank-deficient basis returns kRankDeficient and is left to the host.
//
// Self-contained (plain CUDA, no cuOpt headers) so the offline benchmark can compile it directly.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace cuopt::linear_programming::dual_simplex::gpu_lu {

enum status_t : int {
  kOk             = 0,
  kRankDeficient  = 1,  // needs the host's basis repair
  kNucleusTooBig  = 2,  // nucleus larger than the workspace capacity
  kFactorOverflow = 3,  // L or U larger than its output capacity
  kBasisTooBig    = 4,  // basis nonzeros larger than the workspace capacity
};

// The constraint matrix, CSC, m rows (the basis dimension).
struct matrix_t {
  int m{0};
  const int* col_start{nullptr};
  const int* row_index{nullptr};
  const double* values{nullptr};
};

// Output of one factorization: P B Q = L U with B = A(:, basic_list), the basic list reordered by
// Q in place, L and U in CSC (L diagonal first in each column, U diagonal last), their transposes.
struct factors_t {
  int* L_col_start;  // m + 1
  int* L_i;
  double* L_x;
  int* U_col_start;  // m + 1
  int* U_i;
  double* U_x;
  int* Lt_col_start;  // m + 1
  int* Lt_i;
  double* Lt_x;
  int* Ut_col_start;  // m + 1
  int* Ut_i;
  double* Ut_x;
  int L_cap;  // entries available in L_i / L_x (and Lt_i / Lt_x)
  int U_cap;  // entries available in U_i / U_x (and Ut_i / Ut_x)
  int* row_perm;      // p, m
  int* inv_row_perm;  // pinv, m
};

// Pointers into one basis's workspace.
struct ws_ptrs_t {
  double *c_x, *max_in_col, *pivot_row_val, *pivot_col_val, *urow_x;
  int *Cdeg, *Rdeg, *col_perm, *row_perm, *row_perm_inv, *queue, *Rp, *Rj;
  int *c_i, *r_j, *c_len, *r_len, *col_deg, *row_deg;
  int *col_head, *col_next, *col_prev, *row_head, *row_next, *row_prev;
  int *pinv_S, *q_S, *qinv_S, *pivot_col_index, *urow_start, *urow_j, *u33_start;
  unsigned char* pivot_col_mark;
};

// Carve one basis's workspace starting at `base`; returns its size in bytes. With base == nullptr
// only the size is computed. `c` is the nucleus capacity, `b_cap` the basis nonzero capacity.
__host__ __device__ inline size_t carve_workspace(
  unsigned char* base, int m, int c, long long b_cap, long long urow_cap, ws_ptrs_t& w)
{
  size_t off    = 0;
  auto take_d   = [&](double*& ptr, size_t count) {
    off = (off + 15) & ~size_t{15};
    ptr = base ? reinterpret_cast<double*>(base + off) : nullptr;
    off += count * sizeof(double);
  };
  auto take_i = [&](int*& ptr, size_t count) {
    off = (off + 15) & ~size_t{15};
    ptr = base ? reinterpret_cast<int*>(base + off) : nullptr;
    off += count * sizeof(int);
  };
  const size_t cc = size_t(c) * size_t(c);
  take_d(w.c_x, cc);
  take_d(w.max_in_col, c);
  take_d(w.pivot_row_val, c);
  take_d(w.pivot_col_val, c);
  take_d(w.urow_x, size_t(urow_cap));
  take_i(w.Cdeg, m);
  take_i(w.Rdeg, m);
  take_i(w.col_perm, m);
  take_i(w.row_perm, m);
  take_i(w.row_perm_inv, m);
  take_i(w.queue, m);
  take_i(w.Rp, size_t(m) + 1);
  take_i(w.Rj, size_t(b_cap));
  take_i(w.c_i, cc);
  take_i(w.r_j, cc);
  take_i(w.c_len, c);
  take_i(w.r_len, c);
  take_i(w.col_deg, c);
  take_i(w.row_deg, c);
  take_i(w.col_head, size_t(c) + 1);
  take_i(w.col_next, c);
  take_i(w.col_prev, c);
  take_i(w.row_head, size_t(c) + 1);
  take_i(w.row_next, c);
  take_i(w.row_prev, c);
  take_i(w.pinv_S, c);
  take_i(w.q_S, c);
  take_i(w.qinv_S, c);
  take_i(w.pivot_col_index, c);
  take_i(w.urow_start, size_t(c) + 1);
  take_i(w.urow_j, size_t(urow_cap));
  take_i(w.u33_start, size_t(c) + 1);
  off                = (off + 15) & ~size_t{15};
  w.pivot_col_mark   = base ? base + off : nullptr;
  off += c;
  return (off + 15) & ~size_t{15};
}

// Per-basis workspace layout; every basis of a batch uses the same stride.
struct layout_t {
  int m{0}, nucleus_cap{0};
  long long b_cap{0}, urow_cap{0};
  size_t bytes{0};
  static layout_t make(int m, int nucleus_cap, long long b_cap)
  {
    layout_t l;
    l.m           = m;
    l.nucleus_cap = nucleus_cap;
    l.b_cap       = b_cap;
    l.urow_cap    = (long long)nucleus_cap * (nucleus_cap + 1) / 2 + nucleus_cap;
    ws_ptrs_t w{};
    l.bytes = carve_workspace(nullptr, m, nucleus_cap, b_cap, l.urow_cap, w);
    return l;
  }
};

// Factorize B = A(:, basic_list) into `out`, using `ws` (layout.bytes bytes). `threshold_tol` is
// the relative pivot threshold (the host's settings.threshold_partial_pivoting_tol). On kOk,
// basic_list holds the basic columns in pivot order and *L_nz / *U_nz the factor sizes.
__host__ __device__ int factorize(const matrix_t& A,
                         int* basic_list,
                         double threshold_tol,
                         unsigned char* ws,
                         const layout_t& layout,
                         const factors_t& out,
                         int* L_nz,
                         int* U_nz);

// Structural singleton order of B = A(:, basic_list): col_order[k] / row_order[k] are the basis
// position / row placed k-th (column singletons, row singletons, then the nucleus in natural
// order). Uses the same workspace layout as factorize().
__host__ __device__ int singleton_order(const matrix_t& A,
                                        const int* basic_list,
                                        unsigned char* ws,
                                        const layout_t& layout,
                                        int* col_order,
                                        int* row_order,
                                        int* col_singletons,
                                        int* row_singletons);

// Host launcher: factorize `count` bases, one thread each. One thread per basis is a naive layout
// that serialises parallelisable work; its runtime is not representative of a device sparse LU. Basis b uses basic_lists + b * m,
// workspace + b * layout.bytes, factors[b]; its status goes to status[b], sizes to L_nz[b], U_nz[b].
void launch_batched_factorize(const matrix_t& A,
                              int* basic_lists,
                              int count,
                              double threshold_tol,
                              unsigned char* workspace,
                              const layout_t& layout,
                              const factors_t* factors,
                              int* status,
                              int* L_nz,
                              int* U_nz,
                              cudaStream_t stream);

// ---- Dense batched path (cublasDgetrfBatched) ----
// One staged seat of a dense batch: its basic list (read), where its factors go, and its result
// info[3] = {status, L nz, U nz}. P B = L U with Q = identity, so the basic list keeps its order.
struct dense_seat_t {
  int* basic_list;  // read; run_batch_lu rewrites it in the factor's column order
  int* L_col_start;
  int* L_i;
  double* L_x;
  int* U_col_start;
  int* U_i;
  double* U_x;
  int* Lt_col_start;
  int* Lt_i;
  double* Lt_x;
  int* Ut_col_start;
  int* Ut_i;
  double* Ut_x;
  int L_cap;
  int U_cap;
  int* row_perm;
  int* inv_row_perm;
  int* info;
};

// Factorize `count` bases densely: form B into `dense` (count * m * m doubles, pointer table
// `ptrs`), cublasDgetrfBatched (pivots `piv`, count * m; infos `cublas_info`), then write each
// seat's sparse L / U, transposes and row permutation. A zero pivot or |U(k,k)| < pivot_tol gives
// kRankDeficient, a factor larger than its slot kFactorOverflow. `cublas_handle` is a
// cublasHandle_t; the call leaves it bound to `stream`.
void run_dense_batch(void* cublas_handle,
                     const matrix_t& A,
                     const dense_seat_t* seats,
                     int count,
                     double* dense,
                     double** ptrs,
                     int* piv,
                     int* cublas_info,
                     double pivot_tol,
                     cudaStream_t stream);

// cuBLAS handle for run_dense_batch (kept out of callers' headers).
void* create_blas_handle();
void destroy_blas_handle(void* handle);

// Batch LU of fresh seats without cuBLAS: structural singleton peeling, then the nucleus densely in
// shared memory (m <= kBatchDenseMaxM) or sparsely in rounds of independent Markowitz pivots with
// threshold partial pivoting (m > kBatchDenseMaxM). The columns are reordered (singletons first):
// each seat's basic_list is rewritten in the factor's column order. Same outputs and status codes as
// run_dense_batch (a structurally or numerically singular basis, or a nucleus list overflowing the
// sparse workspace, gives kRankDeficient / kFactorOverflow). `workspace`: batch_lu_workspace_bytes.
constexpr int kBatchDenseMaxM = 128;
size_t batch_lu_workspace_bytes(int m, int count);
void run_batch_lu(const matrix_t& A,
                  const dense_seat_t* seats,
                  int count,
                  void* workspace,
                  double pivot_tol,
                  cudaStream_t stream);

// Fresh seats staged from their packed basis status: per listed seat, the 2-bit statuses (the
// tree's packing: 0 basic, 1 lower, 2 upper, 3 free) become the seat's vstatus, the basic and
// nonbasic lists are built in increasing column order (as get_basis_from_vstatus) and the root edge
// norms are copied. One warp per seat. Runs before the batch LU, which reads the basic lists.
struct expand_seats_t {
  const int* seats;
  int count;
  const std::uint8_t* packed;  // seat s at packed + s * packed_stride
  int packed_stride;
  int m, n;
  std::int8_t* vstatus;  // seat slabs: n, m, n - m, n entries per seat
  int* basic_list;
  int* nonbasic_list;
  double* dy_se;
  const double* root_dy_se;
};
void expand_fresh_seats(const expand_seats_t& e, cudaStream_t stream);

// Upper bound on the nonzeros of any basis of A: the sum of the m longest columns.
long long basis_nnz_bound(int m, int n, const int* host_col_start);

}  // namespace cuopt::linear_programming::dual_simplex::gpu_lu
