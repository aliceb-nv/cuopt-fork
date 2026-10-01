/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Fissioned pivot loop (CUOPT_BNB_GPU_FISSION=1), included by gpu_phase2.cu inside
// gpu_phase2_detail. The monolith's loop (gpu_simplified_dual_phase2_device) split into phase
// kernels over all seats, state in slot_state, BTRAN/FTRAN through explicit inverses of the base
// factors (explicit_inverse.hpp) plus the product-form update replay:
//   A price (warp/seat)  B BTRAN (block/seat)  C choose: dy norm, SE check, dz, ratio, dual update
//   D FTRAN (block/seat)  E update: dx, primal, cutoff, SE norms, swap, refactor check, MPF, infeas
// Seats advance by st.f_state; each kernel skips seats in other states. The host launches K rounds
// of A..E, then counts seats still running.

enum fission_state_t : int { F_PRICE = 0, F_BTRAN = 1, F_CHOOSE = 2, F_FTRAN = 3, F_UPDATE = 4, F_DONE = 9 };

// Per-step cycles and call counts (thread 0 of each seat's warp/block), see gpu_fission_step_report.
enum fission_step_t : int {
  S_PRICE,
  S_B_ZERO, S_B_UROW, S_B_TLOOP, S_B_COMPACT, S_B_LSCATTER, S_B_OUT,
  S_C_NORM, S_C_SECHECK, S_C_DZ, S_C_RATIO, S_C_DUAL, S_C_PREP,
  S_F_ZERO, S_F_LSCATTER, S_F_TLOOP, S_F_COMPACT, S_F_USCATTER, S_F_OUT,
  S_U_PIVCHECK, S_U_DX, S_U_PRIMAL, S_U_OBJ, S_U_CUTOFF, S_U_SENORM, S_U_SWAP, S_U_REFCHECK,
  S_U_MPF, S_U_INFEAS, S_U_TAIL, S_U_JOIN,
  // fission_update_block's primal chain (warps 1..7), concurrent with S_U_MPF: not in the total.
  S_UP_DX, S_UP_PRIMAL, S_UP_OBJ, S_UP_CUTOFF, S_UP_SENORM, S_UP_SWAP,
  kFissionSteps
};
__device__ unsigned long long g_fstep[kFissionSteps][2];
#define FSTEP_BEGIN long long fst_ = clock64();
#define FSTEP(k) FSTEP_AT(k, 0)
#define FSTEP_AT(k, t)                                            \
  if (threadIdx.x == (t)) {                                       \
    const long long c_ = clock64();                               \
    atomicAdd(&g_fstep[k][0], static_cast<unsigned long long>(c_ - fst_)); \
    atomicAdd(&g_fstep[k][1], 1ull);                              \
    fst_ = c_;                                                    \
  }

// CUOPT_BNB_GPU_FISSION_UTRACE: one record per update block, in a ring (diagnostics).
struct fission_utrace_t {
  unsigned long long t0, t1;     // globaltimer at block entry / exit (ns)
  unsigned long long args;       // clone identity
  int seat, sm, live, nzx;       // live: seat was in F_UPDATE
  int unz, enz, c_mpf, c_primal;  // utilde / UTsol nonzeros; cycles of warp 0's and warps 1..7's chains
  int sub[10];                    // warp_update step cycles [0..8], merged nonzeros [9]
};
__device__ fission_utrace_t* g_utrace;
__device__ unsigned long long g_utrace_n;
__device__ fission_utrace_t* g_ctrace;  // CUOPT_BNB_GPU_FISSION_CTRACE: same, per choose block
__device__ unsigned long long g_ctrace_n;
__device__ fission_utrace_t* g_strace;  // CUOPT_BNB_GPU_FISSION_STRACE: same, per setup block
__device__ unsigned long long g_strace_n;
constexpr unsigned long long kUtraceCap = 1ull << 21;
__device__ __forceinline__ unsigned long long fission_gtime()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

constexpr int kFissionThreads = 1024;
constexpr int kShortSeg       = 8;  // gather/mark segments up to this length: one thread  // B/D block size (32 registers, 2 seats per SM)

__device__ __forceinline__ void fission_views(const kernel_args& a, gpu_lp& lp, gpu_settings& settings,
                                              gpu_solution& sol)
{
  const i_t m = a.m, n = a.n;
  lp.num_rows       = m;
  lp.num_cols       = n;
  lp.A              = make_view(m, n, a.A_nz, a.A_col_start, a.A_i, a.A_x);
  lp.Arow_row_start = a.Arow_row_start;
  lp.Arow_j         = a.Arow_j;
  lp.Arow_x         = a.Arow_x;
  lp.rhs            = gpu_span<f_t>(a.rhs, m);
  lp.lower          = gpu_span<f_t>(a.lower, n);
  lp.upper          = gpu_span<f_t>(a.upper, n);
  lp.objective      = gpu_span<f_t>(a.objective, n);
  lp.obj_scale      = a.obj_scale;
  lp.obj_constant   = a.obj_constant;
  settings.primal_tol          = a.primal_tol;
  settings.dual_tol            = a.dual_tol;
  settings.pivot_tol           = a.pivot_tol;
  settings.tight_tol           = a.tight_tol;
  settings.fixed_tol           = a.fixed_tol;
  settings.zero_tol            = a.zero_tol;
  settings.time_limit          = a.time_limit;
  settings.cut_off             = a.cut_off;
  settings.steepest_edge_ratio = a.steepest_edge_ratio;
  settings.refactor_frequency  = a.refactor_frequency;
  settings.yield_count         = nullptr;
  settings.yield_at            = 0;
  sol.x = gpu_span<f_t>(a.x, n);
  sol.y = gpu_span<f_t>(a.y, m);
  sol.z = gpu_span<f_t>(a.z, n);
}

// Terminal exit of a seat (warp): the monolith's finish()/suspend() plus the kernel-body stores.
// full_obj (lane 0's value, optional): c'x already reduced by the caller's block (fission_objective).
__device__ void fission_exit(kernel_args& a, slot_state<gpu_basis_update_t>& st, const gpu_lp& lp,
                             gpu_solution& sol, dual::status_t status, i_t iters, bool suspend,
                             i_t reason, const f_t* full_obj = nullptr)
{
  lane0_serial([&] {
    if (suspend) {
      sol.objective      = st.obj;
      sol.user_objective = compute_user_objective(lp, st.obj);
    } else {
      sol.objective      = full_obj != nullptr ? *full_obj : compute_objective(lp, sol.x.data());
      sol.user_objective = compute_user_objective(lp, sol.objective);
    }
    sol.iterations        = iters;
    a.out_status[0]       = static_cast<i_t>(status);
    a.out_status[1]       = reason;
    *a.out_objective      = sol.objective;
    *a.out_user_objective = sol.user_objective;
    *a.out_iterations     = sol.iterations;
    st.f_state            = F_DONE;
  });
}

// warp_strengthen_bounds over a whole block (same passes and arithmetic, per-column owners).
__device__ void block_strengthen_bounds(kernel_args& a, bool from_deltas)
{
  const i_t m = a.m, n = a.n;
  const i_t tid = static_cast<i_t>(threadIdx.x), nt = static_cast<i_t>(blockDim.x);
  const f_t inf  = cuda::std::numeric_limits<f_t>::infinity();
  const f_t ptol = a.primal_tol;
  f_t* dmin          = reinterpret_cast<f_t*>(a.bs_scratch);
  f_t* dmax          = dmin + m;
  unsigned char* rc  = reinterpret_cast<unsigned char*>(dmax + m);
  unsigned char* rcn = rc + m;
  unsigned char* vc  = rcn + m;
  f_t* lower         = a.lower;
  f_t* upper         = a.upper;
  for (i_t i = tid; i < m; i += nt) { rc[i] = from_deltas ? 0 : 1, rcn[i] = 0; }
  for (i_t j = tid; j < n; j += nt) { vc[j] = 0; }
  __syncthreads();
  if (from_deltas) {
    for (i_t q = 0; q < a.bound_delta_count; ++q) {
      const i_t j = a.bound_deltas[q].column;
      for (i_t p = a.A_col_start[j] + tid; p < a.A_col_start[j + 1]; p += nt) { rc[a.A_i[p]] = 1; }
    }
    __syncthreads();
  }
  for (int iter = 0; iter < 10; ++iter) {
    int infeasible = 0;
    for (i_t i = tid; i < m; i += nt) {
      if (!rc[i]) { continue; }
      f_t min_a = 0.0, max_a = 0.0;
      for (i_t p = a.Arow_row_start[i]; p < a.Arow_row_start[i + 1]; ++p) {
        const i_t j    = a.Arow_j[p];
        const f_t a_ij = a.Arow_x[p];
        vc[j]          = 1;
        if (a_ij > 0) {
          min_a += a_ij * lower[j];
          max_a += a_ij * upper[j];
        } else if (a_ij < 0) {
          min_a += a_ij * upper[j];
          max_a += a_ij * lower[j];
        }
        if (upper[j] == inf && a_ij > 0) { max_a = inf; }
        if (lower[j] == -inf && a_ij < 0) { max_a = inf; }
        if (lower[j] == -inf && a_ij > 0) { min_a = -inf; }
        if (upper[j] == inf && a_ij < 0) { min_a = -inf; }
      }
      const f_t b = a.rhs[i];
      if (min_a > b + ptol || max_a < b - ptol) { infeasible = 1; }
      dmin[i] = b - min_a;
      dmax[i] = b - max_a;
    }
    if (__syncthreads_or(infeasible)) { return; }
    int changed = 0;
    for (i_t k = tid; k < n; k += nt) {
      if (!vc[k]) { continue; }
      const f_t old_lb = lower[k];
      const f_t old_ub = upper[k];
      f_t new_lb       = old_lb;
      f_t new_ub       = old_ub;
      for (i_t p = a.A_col_start[k]; p < a.A_col_start[k + 1]; ++p) {
        const i_t i = a.A_i[p];
        if (!rc[i]) { continue; }
        const f_t a_ik = a.A_x[p];
        const f_t dmn  = dmin[i] + ((a_ik < 0) ? a_ik * old_ub : a_ik * old_lb);
        const f_t dmx  = dmax[i] + ((a_ik > 0) ? a_ik * old_ub : a_ik * old_lb);
        const f_t lb_c = (a_ik < 0.) ? dmn / a_ik : dmx / a_ik;
        const f_t ub_c = (a_ik < 0.) ? dmx / a_ik : dmn / a_ik;
        new_lb         = (new_lb < lb_c) ? lb_c : new_lb;
        new_ub         = (ub_c < new_ub) ? ub_c : new_ub;
      }
      if (a.var_int[k]) {
        new_lb = ceil(new_lb - a.integer_tol);
        new_ub = floor(new_ub + a.integer_tol);
      }
      const bool updated = fabs(new_lb - old_lb) > 1e3 * ptol || fabs(new_ub - old_ub) > 1e3 * ptol;
      new_lb = (new_lb < old_lb) ? old_lb : new_lb;
      new_ub = (old_ub < new_ub) ? old_ub : new_ub;
      if (new_lb > new_ub + ptol) {
        infeasible = 1;
        continue;
      }
      if (new_lb != old_lb || new_ub != old_ub) {
        for (i_t p = a.A_col_start[k]; p < a.A_col_start[k + 1]; ++p) { rcn[a.A_i[p]] = 1; }
      }
      lower[k] = (new_lb < new_ub) ? new_lb : new_ub;
      upper[k] = (new_lb < new_ub) ? new_ub : new_lb;
      changed |= updated ? 1 : 0;
    }
    if (__syncthreads_or(infeasible)) { return; }
    if (!__syncthreads_or(changed)) { return; }
    for (i_t i = tid; i < m; i += nt) { rc[i] = rcn[i], rcn[i] = 0; }
    for (i_t j = tid; j < n; j += nt) { vc[j] = 0; }
    __syncthreads();
  }
}

// Fission setup, part 1 (block): the kernel body's bounds reset, child deltas and strengthening.
constexpr int kPresetupThreads = 512;
__global__ void __launch_bounds__(kPresetupThreads) fission_presetup(kernel_args* args, i_t n_problems)
{
  const i_t idx = static_cast<i_t>(blockIdx.x);
  if (idx >= n_problems) { return; }
  kernel_args& a = args[idx];
  if (a.done != 0) { return; }
  if (a.device_lu != 0 && a.lu_info[0] != 0) { return; }
  const bool child_continue = a.command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE);
  if (a.bounds_from_root != 0) {
    for (i_t j = threadIdx.x; j < a.n; j += blockDim.x) { a.lower[j] = a.root_lower[j], a.upper[j] = a.root_upper[j]; }
    __syncthreads();
    for (i_t q = threadIdx.x; q < a.bound_delta_count; q += blockDim.x) {
      const auto& d = a.bound_deltas[q];
      a.lower[d.column] = d.lower;
      a.upper[d.column] = d.upper;
    }
    __syncthreads();
  }
  if (child_continue) {
    if (threadIdx.x == 0) {
      for (i_t q = 0; q < a.bound_delta_count; ++q) {
        const auto& d = a.bound_deltas[q];
        if ((d.mask & 1u) != 0u) { a.lower[d.column] = d.lower; }
        if ((d.mask & 2u) != 0u) { a.upper[d.column] = d.upper; }
      }
    }
    __syncthreads();
  }
  if (a.strengthen != 0 && (child_continue || a.refactor_resume == 0)) {
    block_strengthen_bounds(a, child_continue);
  }
}

__global__ void __launch_bounds__(32) fission_setup_kernel(kernel_args* args, i_t n_problems)
{
  const i_t idx = static_cast<i_t>(blockIdx.x);
  if (idx >= n_problems) { return; }
  // setup trace sub[]: st.pre legs bounds, strengthen, transpose, slot_init, levels, prologue
  // [0..5]; fission infeasibility rebuild [6]; command [7]; device_lu [8]; done before setup [9].
  const bool tr = g_strace != nullptr;
  fission_utrace_t r{};
  if (tr && threadIdx.x == 0) {
    r.t0   = fission_gtime();
    r.args = reinterpret_cast<unsigned long long>(args);
    r.seat = idx;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r.sm));
    r.sub[7] = args[idx].command;
    r.sub[8] = args[idx].device_lu;
    r.sub[9] = args[idx].done;
  }
  const auto emit = [&] {
    if (tr && threadIdx.x == 0) {
      r.t1 = fission_gtime();
      g_strace[atomicAdd(&g_strace_n, 1ull) & (kUtraceCap - 1)] = r;
    }
  };
  gpu_phase2_kernel_body<true, true, true>(args, idx);
  kernel_args& a = args[idx];
  if (a.done != 0) {
    emit();
    return;
  }
  const long long c6 = clock64();
  if (tr && threadIdx.x == 0) {
    r.live = 1;
    for (int q = 0; q < 6; ++q) { r.sub[q] = static_cast<int>(a.slot->pre[q]); }
  }
  // Incremental infeasibility list (fission_update): squared_infeas[j] != 0 exactly when j is
  // listed. Zero the whole array, then rebuild once.
  auto& st = *a.slot;
  gpu_lp lp;
  gpu_settings settings;
  gpu_solution sol;
  fission_views(a, lp, settings, sol);
  f_t* sq = st.squared_infeas.data();
  for (i_t j = threadIdx.x; j < a.n; j += 32) { sq[j] = 0.0; }
  __syncwarp(0xffffffffu);
  compute_initial_primal_infeasibilities(lp, settings, st.ft, a.basic_list, sol.x.data(), st.squared_infeas,
                                         st.infeas_idx, st.primal_inf);
  __syncwarp(0xffffffffu);
  if (tr && threadIdx.x == 0) { r.sub[6] = static_cast<int>(clock64() - c6); }
  emit();
}

// Incremental infeasibility update after a pivot: the basic variables at the positions of
// scaled_dxB (basis after the swap, so the entering variable is covered) and the leaving variable.
// Invariant: sq[j] != 0 exactly when j is listed. Newly infeasible variables are appended; entries
// that became feasible are zeroed, then dropped by an in-place ordered warp compaction.
__device__ __forceinline__ void fission_infeas_update(const gpu_lp& lp, const gpu_settings& settings,
                                                      const i_t* basic_list, const f_t* x,
                                                      const gpu_sparse_vector& sdx, i_t leaving,
                                                      gpu_span<f_t>& sqs, gpu_span<i_t>& list)
{
  const int lane = static_cast<int>(threadIdx.x);
  f_t* sq        = sqs.data();
  i_t* idx       = list.data();
  if (lane == 0) { sq[leaving] = 0.0; }
  __syncwarp(0xffffffffu);
  i_t count     = static_cast<i_t>(list.size());
  const i_t nzx = static_cast<i_t>(sdx.i.size());
  for (i_t base = 0; base < nzx; base += 32) {
    const i_t k = base + lane;
    bool add    = false;
    i_t j       = -1;
    if (k < nzx) {
      j                = basic_list[sdx.i[k]];
      const f_t infeas = fmax(lp.lower[j] - x[j], x[j] - lp.upper[j]);
      if (infeas > settings.primal_tol) {
        add   = sq[j] == 0.0;
        sq[j] = infeas * infeas;
      } else {
        sq[j] = 0.0;
      }
    }
    const unsigned b = __ballot_sync(0xffffffffu, add);
    if (add) { idx[count + __popc(b & ((1u << lane) - 1u))] = j; }
    count += __popc(b);
  }
  __syncwarp(0xffffffffu);
  i_t out = 0;
  for (i_t base = 0; base < count; base += 32) {
    const i_t k    = base + lane;
    const i_t j    = k < count ? idx[k] : -1;
    const bool keep = j >= 0 && sq[j] != 0.0;
    const unsigned b = __ballot_sync(0xffffffffu, keep);
    __syncwarp(0xffffffffu);  // all reads of this stride before any write into it
    if (keep) { idx[out + __popc(b & ((1u << lane) - 1u))] = j; }
    out += __popc(b);
  }
  __syncwarp(0xffffffffu);
  if (lane == 0) { list.set_size(out); }
  __syncwarp(0xffffffffu);
}

// c'x over the whole block, fixed order; valid on thread 0.
template <int T>
__device__ f_t fission_objective(const gpu_lp& lp, const f_t* x,
                                 typename cub::BlockReduce<f_t, T>::TempStorage& tmp)
{
  f_t part = 0.0;
  for (i_t j = threadIdx.x; j < lp.num_cols; j += T) { part += lp.objective[j] * x[j]; }
  return cub::BlockReduce<f_t, T>(tmp).Sum(part);
}

// A. Steepest-edge pricing.
__global__ void __launch_bounds__(32) fission_price(kernel_args* args, i_t n_problems)
{
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_PRICE) { return; }
  FSTEP_BEGIN
  gpu_lp lp;
  gpu_settings settings;
  gpu_solution sol;
  fission_views(a, lp, settings, sol);
  i_t direction = 0, basic_leaving = -1;
  f_t max_val;
  const i_t leaving = steepest_edge_pricing_with_infeasibilities(
    lp, settings, sol.x.data(), a.dy_se, st.basic_mark.data(), st.squared_infeas, st.infeas_idx,
    direction, basic_leaving, max_val);
  FSTEP(S_PRICE)
  if (leaving == -1) {
    __shared__ typename cub::BlockReduce<f_t, 32>::TempStorage otmp;
    const f_t obj = fission_objective<32>(lp, sol.x.data(), otmp);
    fission_exit(a, st, lp, sol, dual::status_t::OPTIMAL, st.iter, false, 0, &obj);
    return;
  }
  lane0_serial([&] {
    st.leaving       = leaving;
    st.direction     = direction;
    st.basic_leaving = basic_leaving;
    st.f_state       = F_BTRAN;
  });
}

// Deterministic block helpers. Lists are index-ordered (block scans), every float sum has a fixed
// order (one owner thread per output entry, summing a row/column segment in storage order), and
// pattern marks are idempotent writes. Results are bitwise reproducible run to run.

// out[0..count) = ascending k in [0, m) with pred(k); count returned on every thread. Each thread
// owns a contiguous chunk of up to kChunk indices per pass (blocked arrangement), so one block scan
// of per-thread counts orders a pass of kChunk * kFissionThreads indices.
using fission_scan_t = cub::BlockScan<int, kFissionThreads>;
struct fission_scan_smem_t {
  typename fission_scan_t::TempStorage tmp;
  int base;
};
template <class Pred>
__device__ __forceinline__ int fission_ordered_compact(fission_scan_smem_t& sm, int m, Pred pred, i_t* out)
{
  constexpr int kChunk = 8;
  int base_l           = 0;
  for (int t0 = 0; t0 < m; t0 += kChunk * kFissionThreads) {
    const int k0  = t0 + threadIdx.x * kChunk;
    unsigned bits = 0;
#pragma unroll
    for (int c = 0; c < kChunk; ++c) {
      const int k = k0 + c;
      if (k < m && pred(k)) { bits |= 1u << c; }
    }
    int pos = 0, agg = 0;
    fission_scan_t(sm.tmp).ExclusiveSum(static_cast<int>(__popc(bits)), pos, agg);
    pos += base_l;
#pragma unroll
    for (int c = 0; c < kChunk; ++c) {
      if (bits & (1u << c)) { out[pos++] = k0 + c; }
    }
    base_l += agg;
    __syncthreads();
  }
  return base_l;
}

// mark[ri[p]] = 1 over the segments idx[0..n) of (cp, ri): one warp per segment, lanes stride it
// (idempotent writes, so any distribution is deterministic).
__device__ __forceinline__ void fission_mark(const i_t* cp, const i_t* ri, const i_t* idx, int n,
                                             unsigned char* mark)
{
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int t = warp; t < n; t += kFissionThreads / 32) {
    const i_t c = idx[t];
    for (i_t p = cp[c] + lane; p < cp[c + 1]; p += 32) { mark[ri[p]] = 1; }
  }
  __syncthreads();
}

// out[list[q]] = sum over segment list[q] of (cp, ri, cx) of cx[p] * in[ri[p]]. Segments of up to
// kShortSeg entries: one thread, storage order. Longer: one warp, lane-strided partials reduced by
// a fixed butterfly. Both orders depend only on the data, so results are reproducible.
__device__ __forceinline__ void fission_gather(const i_t* cp, const i_t* ri, const f_t* cx,
                                               const i_t* list, int n, const f_t* in, f_t* out)
{
  for (int q = threadIdx.x; q < n; q += kFissionThreads) {
    const i_t i = list[q];
    const i_t b = cp[i], e = cp[i + 1];
    if (e - b > kShortSeg) { continue; }
    f_t s = 0.0;
    for (i_t p = b; p < e; ++p) { s += cx[p] * in[ri[p]]; }
    out[i] = s;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int q = warp; q < n; q += kFissionThreads / 32) {
    const i_t i = list[q];
    const i_t b = cp[i], e = cp[i + 1];
    if (e - b <= kShortSeg) { continue; }
    f_t s = 0.0;
    for (i_t p = b + lane; p < e; p += 32) { s += cx[p] * in[ri[p]]; }
    s = warp_reduce_add(s);
    if (lane == 0) { out[i] = s; }
  }
  __syncthreads();
}

// Pattern sets as shared bitsets (bit k = index k). Setting a bit is an integer atomicOr, so the
// final set is deterministic; the lane that first sets a bit appends it to an unordered list through
// one warp-aggregated atomic. Unordered lists feed only one-owner gathers or further marking.
__device__ __forceinline__ void fission_set(unsigned* bits, int k, bool active, int* cnt, i_t* list)
{
  bool fresh = false;
  if (active) {
    const unsigned bit = 1u << (k & 31);
    fresh              = (atomicOr(&bits[k >> 5], bit) & bit) == 0u;
  }
  if (list == nullptr) { return; }
  const unsigned am = __activemask();
  const unsigned b  = __ballot_sync(am, fresh);
  if (b == 0u) { return; }
  const int lane   = threadIdx.x & 31;
  const int leader = __ffs(b) - 1;
  int base         = 0;
  if (lane == leader) { base = atomicAdd(cnt, __popc(b)); }
  base = __shfl_sync(am, base, leader);
  if (fresh) { list[base + __popc(b & ((1u << lane) - 1u))] = k; }
}

// Warp-aggregated append of k to an unordered list (lanes may be at different loop trips).
__device__ __forceinline__ void fission_append(bool add, int k, int* cnt, i_t* list)
{
  const unsigned am = __activemask();
  const unsigned b  = __ballot_sync(am, add);
  if (b == 0u) { return; }
  const int lane   = threadIdx.x & 31;
  const int leader = __ffs(b) - 1;
  int base         = 0;
  if (lane == leader) { base = atomicAdd(cnt, __popc(b)); }
  base = __shfl_sync(am, base, leader);
  if (add) { list[base + __popc(b & ((1u << lane) - 1u))] = k; }
}

// Unordered list of the set bits k with vals[k] != 0 (one thread per word). Returns the count.
__device__ __forceinline__ int fission_bits_list(const unsigned* bits, const f_t* vals, int m, int* cnt,
                                                 i_t* list)
{
  if (threadIdx.x == 0) { *cnt = 0; }
  __syncthreads();
  const int W = (m + 31) >> 5;
  for (int w = threadIdx.x; w < W; w += kFissionThreads) {
    for (unsigned t = bits[w]; t != 0u; t &= t - 1u) {
      const int k = (w << 5) + __ffs(t) - 1;
      fission_append(vals[k] != 0.0, k, cnt, list);
    }
  }
  __syncthreads();
  const int n = *cnt;
  __syncthreads();
  return n;
}

// Set bits ri[p] over the segments of idx[0..n): segments of up to kShortSeg entries one thread
// each, longer ones one warp each. New bits are appended to out (unordered).
__device__ __forceinline__ void fission_mark_list(const i_t* cp, const i_t* ri, const i_t* idx, int n,
                                                  unsigned* bits, int* cnt, i_t* out)
{
  for (int t = threadIdx.x; t < n; t += kFissionThreads) {
    const i_t c = idx[t];
    const i_t b = cp[c], e = cp[c + 1];
    if (e - b > kShortSeg) { continue; }
    for (i_t p = b; p < e; ++p) { fission_set(bits, ri[p], true, cnt, out); }
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int t = warp; t < n; t += kFissionThreads / 32) {
    const i_t c = idx[t];
    const i_t b = cp[c], e = cp[c + 1];
    if (e - b <= kShortSeg) { continue; }
    for (i_t p0 = b; p0 < e; p0 += 32) {
      const i_t p = p0 + lane;
      fission_set(bits, p < e ? ri[p] : 0, p < e, cnt, out);
    }
  }
  __syncthreads();
}

// Ascending (k, vals[k]) over the set bits with vals[k] != 0: one block scan per 1024 words.
__device__ __forceinline__ int fission_bits_out(fission_scan_smem_t& sm, const unsigned* bits, int m,
                                                const f_t* vals, i_t* oi, f_t* ox)
{
  const int W = (m + 31) >> 5;
  int base    = 0;
  for (int w0 = 0; w0 < W; w0 += kFissionThreads) {
    const int w   = w0 + threadIdx.x;
    unsigned mask = w < W ? bits[w] : 0u;
    for (unsigned t = mask; t != 0u; t &= t - 1u) {
      const int b = __ffs(t) - 1;
      if (vals[(w << 5) + b] == 0.0) { mask &= ~(1u << b); }
    }
    int pos = 0, agg = 0;
    fission_scan_t(sm.tmp).ExclusiveSum(__popc(mask), pos, agg);
    pos += base;
    for (unsigned t = mask; t != 0u; t &= t - 1u) {
      const int k = (w << 5) + __ffs(t) - 1;
      oi[pos]     = k;
      ox[pos++]   = vals[k];
    }
    base += agg;
    __syncthreads();
  }
  return base;
}

// Product-form replay on a dense shared vector by warp 0 (the monolith's warp_tloop_body order):
// ascending: dot v col (2k+1), axpy u col (2k); descending: dot u col (2k), axpy v col (2k+1).
__device__ __forceinline__ void fission_tloop(const gpu_basis_update_t& ft, f_t* x, bool asc, f_t tol,
                                              unsigned* bits = nullptr)
{
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const i_t num  = ft.num_updates_;
    const i_t* cs  = ft.S_.col_start.data();
    const i_t* ri  = ft.S_.i.data();
    const f_t* vx  = ft.S_.x.data();
    const f_t* mu  = ft.mu_values_.data();
    for (i_t s = 0; s < num; ++s) {
      const i_t k    = asc ? s : num - 1 - s;
      const i_t dcol = asc ? 2 * k + 1 : 2 * k;
      const i_t acol = asc ? 2 * k : 2 * k + 1;
      f_t partial    = 0.0;
      for (i_t p = cs[dcol] + lane; p < cs[dcol + 1]; p += 32) { partial += vx[p] * x[ri[p]]; }
      const f_t theta = warp_reduce_add(partial) / mu[k];
      if (fabs(theta) > tol) {
        for (i_t p = cs[acol] + lane; p < cs[acol + 1]; p += 32) {
          x[ri[p]] -= theta * vx[p];
          if (bits != nullptr) { atomicOr(&bits[ri[p] >> 5], 1u << (ri[p] & 31)); }
        }
      }
      __syncwarp(0xffffffffu);
    }
  }
  __syncthreads();
}

// fission_tloop with the update stack (column pointers, indices, values, mu) first copied by the
// whole block into `stage` (shared, cap_bytes), so warp 0's serial loop reads only shared memory.
// Falls back to the global-memory loop when the stack does not fit. Returns the bytes used; the
// caller zeroes them (stage aliases an all-zero accumulator).
__device__ __forceinline__ size_t fission_tloop_staged(const gpu_basis_update_t& ft, f_t* x, bool asc,
                                                       f_t tol, f_t* stage, size_t cap_bytes)
{
  const i_t num = ft.num_updates_;
  if (num == 0) { return 0; }
  const i_t* cs   = ft.S_.col_start.data();
  const i_t base  = cs[0];
  const i_t E     = cs[2 * num] - base;
  const size_t need = static_cast<size_t>(E) * (sizeof(f_t) + sizeof(i_t)) + num * sizeof(f_t) +
                      (2 * static_cast<size_t>(num) + 1) * sizeof(i_t);
  if (need > cap_bytes) {
    fission_tloop(ft, x, asc, tol);
    return 0;
  }
  f_t* svx = stage;
  f_t* smu = svx + E;
  i_t* sri = reinterpret_cast<i_t*>(smu + num);
  i_t* scs = sri + E;
  const i_t* gri = ft.S_.i.data() + base;
  const f_t* gvx = ft.S_.x.data() + base;
  for (i_t e = threadIdx.x; e < E; e += blockDim.x) { svx[e] = gvx[e], sri[e] = gri[e]; }
  for (i_t k = threadIdx.x; k <= 2 * num; k += blockDim.x) { scs[k] = cs[k] - base; }
  for (i_t k = threadIdx.x; k < num; k += blockDim.x) { smu[k] = ft.mu_values_[k]; }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    for (i_t s2 = 0; s2 < num; ++s2) {
      const i_t k    = asc ? s2 : num - 1 - s2;
      const i_t dcol = asc ? 2 * k + 1 : 2 * k;
      const i_t acol = asc ? 2 * k : 2 * k + 1;
      f_t partial    = 0.0;
      for (i_t p = scs[dcol] + lane; p < scs[dcol + 1]; p += 32) { partial += svx[p] * x[sri[p]]; }
      const f_t theta = warp_reduce_add(partial) / smu[k];
      if (fabs(theta) > tol) {
        for (i_t p = scs[acol] + lane; p < scs[acol + 1]; p += 32) { x[sri[p]] -= theta * svx[p]; }
      }
      __syncwarp(0xffffffffu);
    }
  }
  __syncthreads();
  return need;
}

// Double-buffered chunks of the update stack: warp 0 replays a chunk from one half of `stage`
// while warps 1.. load the next chunk into the other half. Chunks are maximal runs of updates in
// replay order whose columns, mu and pointers fit a half. Returns the bytes to zero afterwards.
__device__ __forceinline__ size_t fission_chunk_bytes(const i_t* cs, i_t ka, i_t kb)
{
  return static_cast<size_t>(cs[2 * kb] - cs[2 * ka]) * (sizeof(f_t) + sizeof(i_t)) +
         static_cast<size_t>(kb - ka) * sizeof(f_t) + (2 * static_cast<size_t>(kb - ka) + 1) * sizeof(i_t);
}

__device__ __forceinline__ size_t fission_tloop_chunked(const gpu_basis_update_t& ft, f_t* x, bool asc,
                                                        f_t tol, f_t* stage, size_t cap_bytes)
{
  const i_t num = ft.num_updates_;
  if (num == 0) { return 0; }
  const i_t* cs  = ft.S_.col_start.data();
  const i_t* gri = ft.S_.i.data();
  const f_t* gvx = ft.S_.x.data();
  const f_t* gmu = ft.mu_values_.data();
  const size_t half = (cap_bytes / 2) & ~size_t(15);
  f_t* buf[2]       = {stage, reinterpret_cast<f_t*>(reinterpret_cast<char*>(stage) + half)};
  __shared__ int rng[2][2];  // replay positions [s0, s1) held by each buffer
  // Next chunk starting at replay position s0, planned by threads [t0, t0 + nt) in parallel: each
  // tests one candidate end (sizes grow with the end), the largest that fits wins. bar() syncs them.
  auto plan = [&](int b, i_t s0, int t0, int nt, auto bar) {
    if (t0 == 0) { rng[b][0] = s0, rng[b][1] = s0; }
    bar();
    for (i_t s1 = s0 + 1 + t0; s1 <= num && s1 <= s0 + nt; s1 += nt) {
      const i_t ka = asc ? s0 : num - s1;
      const i_t kb = asc ? s1 : num - s0;
      if (fission_chunk_bytes(cs, ka, kb) <= half) { atomicMax(&rng[b][1], s1); }
    }
    bar();
    if (t0 == 0 && rng[b][1] == s0 && s0 < num) { __trap(); }  // one update larger than half the stage
  };
  auto bar_all     = [] { __syncthreads(); };
  auto bar_loaders = [] { asm volatile("bar.sync 1, %0;" ::"r"(kFissionThreads - 32)); };
  auto load = [&](int b, int t0, int nt) {
    const i_t s0 = rng[b][0], s1 = rng[b][1];
    if (s0 >= s1) { return; }
    const i_t ka = asc ? s0 : num - s1, kb = asc ? s1 : num - s0;
    const i_t E0 = cs[2 * ka], E = cs[2 * kb] - E0, n = kb - ka;
    f_t* svx = buf[b];
    f_t* smu = svx + E;
    i_t* sri = reinterpret_cast<i_t*>(smu + n);
    i_t* scs = sri + E;
    for (i_t e = t0; e < E; e += nt) { svx[e] = gvx[E0 + e], sri[e] = gri[E0 + e]; }
    for (i_t k = t0; k <= 2 * n; k += nt) { scs[k] = cs[2 * ka + k] - E0; }
    for (i_t k = t0; k < n; k += nt) { smu[k] = 1.0 / gmu[ka + k]; }  // reciprocal of mu
  };
  auto replay = [&](int b) {
    const int lane = threadIdx.x;
    const i_t s0 = rng[b][0], s1 = rng[b][1];
    const i_t ka = asc ? s0 : num - s1, kb = asc ? s1 : num - s0;
    const i_t E = cs[2 * kb] - cs[2 * ka], n = kb - ka;
    const f_t* svx = buf[b];
    const f_t* smu = svx + E;
    const i_t* sri = reinterpret_cast<const i_t*>(smu + n);
    const i_t* scs = sri + E;
    for (i_t q = 0; q < n; ++q) {
      const i_t kl   = asc ? q : n - 1 - q;  // local update index
      const i_t dcol = asc ? 2 * kl + 1 : 2 * kl;
      const i_t acol = asc ? 2 * kl : 2 * kl + 1;
      const i_t d0 = scs[dcol], d1 = scs[dcol + 1];
      f_t dot;
      if (d1 - d0 <= 2) {
        // Every lane forms the short dot itself (broadcast reads); the barrier keeps these reads of
        // x ahead of the axpy writes below.
        dot = 0.0;
        if (d1 - d0 >= 1) { dot = svx[d0] * x[sri[d0]]; }
        if (d1 - d0 == 2) { dot += svx[d0 + 1] * x[sri[d0 + 1]]; }
        __syncwarp(0xffffffffu);
      } else {
        f_t partial = 0.0;
        for (i_t p = d0 + lane; p < d1; p += 32) { partial += svx[p] * x[sri[p]]; }
        dot = warp_reduce_add(partial);
      }
      const f_t theta = dot * smu[kl];
      if (fabs(theta) > tol) {
        for (i_t p = scs[acol] + lane; p < scs[acol + 1]; p += 32) { x[sri[p]] -= theta * svx[p]; }
      }
      __syncwarp(0xffffffffu);
    }
  };
  plan(0, 0, threadIdx.x, kFissionThreads, bar_all);
  load(0, threadIdx.x, blockDim.x);
  __syncthreads();
  int cur = 0;
  for (;;) {
    if (threadIdx.x < 32) {
      replay(cur);
    } else {
      plan(1 - cur, rng[cur][1], threadIdx.x - 32, kFissionThreads - 32, bar_loaders);
      load(1 - cur, threadIdx.x - 32, kFissionThreads - 32);
    }
    __syncthreads();
    if (rng[1 - cur][0] >= rng[1 - cur][1]) { break; }
    cur = 1 - cur;
  }
  return 2 * half;
}

__device__ __forceinline__ void fission_unstage(f_t* stage, size_t bytes)
{
  const size_t words = (bytes + sizeof(f_t) - 1) / sizeof(f_t);
  for (size_t k = threadIdx.x; k < words; k += blockDim.x) { stage[k] = 0.0; }
  __syncthreads();
}

struct fission_smem_t {
  int cnt;
};

// B. BTRAN through the inverses: UTsol = row r of Uinv; w = -direction * UTsol; transposed replay;
// delta_y[c] = sum_i w_i (Linv P)(i, c), gathered per output c over column c of Linv P.
// Shared: w, acc (m doubles), list (m ints), wbits, ybits (m bits each).
__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_btran(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_BTRAN) { return; }
  __shared__ fission_scan_smem_t scs;
  __shared__ int cnt;
  const int m   = a.m;
  const int W   = (m + 31) >> 5;
  const auto& I = inv[blockIdx.x];
  f_t* w          = fsh;
  f_t* acc        = fsh + m;
  i_t* list       = reinterpret_cast<i_t*>(fsh + 2 * m);
  unsigned* wbits = reinterpret_cast<unsigned*>(list + m);
  unsigned* ybits = wbits + W;
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { w[k] = 0.0, acc[k] = 0.0; }
  for (int k = threadIdx.x; k < W; k += blockDim.x) { wbits[k] = 0u, ybits[k] = 0u; }
  if (threadIdx.x == 0) { cnt = 0; }
  __syncthreads();
  FSTEP(S_B_ZERO)
  const i_t r   = st.basic_leaving;
  const f_t sgn = static_cast<f_t>(-st.direction);
  const i_t p0 = I.Ur_p[r], p1 = I.Ur_p[r + 1];
  auto& UT = st.UTsol_sparse;
  for (i_t p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
    const i_t j  = I.Ur_i[p];
    UT.i[p - p0] = j;
    UT.x[p - p0] = I.Ur_x[p];
    w[j]         = sgn * I.Ur_x[p];
    atomicOr(&wbits[j >> 5], 1u << (j & 31));
  }
  if (threadIdx.x == 0) {
    UT.n = m;
    UT.i.set_size(p1 - p0);
    UT.x.set_size(p1 - p0);
  }
  __syncthreads();
  FSTEP(S_B_UROW)
  fission_tloop(st.ft, w, false, 1e-13, wbits);
  FSTEP(S_B_TLOOP)
  {
    // w's nonzeros listed in acc's storage (all zero here, room for 2m ints), then dy's rows marked
    // into list; the borrowed ints are zeroed again so acc reads as 0.0.
    i_t* wl      = reinterpret_cast<i_t*>(acc);
    const int nw = fission_bits_list(wbits, w, m, &cnt, wl);
    if (threadIdx.x == 0) { cnt = 0; }
    __syncthreads();
    fission_mark_list(I.Lr_p, I.Lr_i, wl, nw, ybits, &cnt, list);
    for (int t = threadIdx.x; t < nw; t += blockDim.x) { wl[t] = 0; }
    __syncthreads();
  }
  FSTEP(S_B_COMPACT)
  fission_gather(I.Lc_p, I.Lc_i, I.Lc_x, list, cnt, w, acc);
  FSTEP(S_B_LSCATTER)
  auto& dy     = st.delta_y_sparse;
  const int nz = fission_bits_out(scs, ybits, m, acc, dy.i.data(), dy.x.data());
  if (threadIdx.x == 0) {
    dy.n = m;
    dy.i.set_size(nz);
    dy.x.set_size(nz);
    st.f_state = F_CHOOSE;
  }
  FSTEP(S_B_OUT)
}

// ---- Non-deterministic BTRAN / FTRAN (float atomics; default, CUOPT_BNB_GPU_FISSION_DETERMINISTIC=0).
// Same contracts as fission_btran / fission_ftran above: shared w/z, acc (m doubles each) and lists
// of kFissionCapA entries. List order and atomic summation order vary run to run.
constexpr int kFissionCapA = 1536;

// Compaction of a dense shared vector into (index, value) lists by an atomic counter; clears it.
__device__ __forceinline__ int fission_compact_atomic(f_t* dense, int m, i_t* li, f_t* lx, int cap, int* cnt)
{
  if (threadIdx.x == 0) { *cnt = 0; }
  __syncthreads();
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    const f_t v = dense[k];
    if (v != 0.0) {
      const int q = atomicAdd(cnt, 1);
      if (q < cap) { li[q] = k, lx[q] = v; }
    }
    dense[k] = 0.0;
  }
  __syncthreads();
  const int c = *cnt;
  if (c > cap) { __trap(); }
  return c;
}

// acc[ri[p]] += cx[p] * val[t] over the segments idx[t], entries balanced across the block.
__device__ __forceinline__ void fission_scatter_atomic(fission_scan_smem_t& sm, const i_t* cp, const i_t* ri,
                                                       const f_t* cx, const i_t* idx, const f_t* val, int n,
                                                       int* pre, f_t* acc)
{
  __shared__ int total;
  constexpr int kItems = (kFissionCapA + kFissionThreads - 1) / kFissionThreads;
  int len[kItems];
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    const int t = threadIdx.x * kItems + k;
    len[k]      = t < n ? cp[idx[t] + 1] - cp[idx[t]] : 0;
  }
  int agg = 0;
  cub::BlockScan<int, kFissionThreads>(sm.tmp).ExclusiveSum(len, len, agg);
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    const int t = threadIdx.x * kItems + k;
    if (t < n) { pre[t] = len[k]; }
  }
  if (threadIdx.x == 0) { total = agg; }
  __syncthreads();
  const int E = total;
  for (int e = threadIdx.x; e < E; e += kFissionThreads) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (pre[mid] <= e) { lo = mid; } else { hi = mid - 1; }
    }
    const int p = cp[idx[lo]] + (e - pre[lo]);
    atomicAdd(&acc[ri[p]], cx[p] * val[lo]);
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_btran_atomic(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_BTRAN) { return; }
  __shared__ fission_scan_smem_t scs;
  __shared__ int cnt;
  const int m   = a.m;
  const auto& I = inv[blockIdx.x];
  f_t* w   = fsh;
  f_t* acc = fsh + m;
  f_t* lx  = fsh + 2 * m;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { w[k] = 0.0, acc[k] = 0.0; }
  __syncthreads();
  FSTEP(S_B_ZERO)
  const i_t r   = st.basic_leaving;
  const f_t sgn = static_cast<f_t>(-st.direction);
  const i_t p0 = I.Ur_p[r], p1 = I.Ur_p[r + 1];
  auto& UT = st.UTsol_sparse;
  for (i_t p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
    const i_t j  = I.Ur_i[p];
    UT.i[p - p0] = j;
    UT.x[p - p0] = I.Ur_x[p];
    w[j]         = sgn * I.Ur_x[p];
  }
  if (threadIdx.x == 0) {
    UT.n = m;
    UT.i.set_size(p1 - p0);
    UT.x.set_size(p1 - p0);
  }
  __syncthreads();
  FSTEP(S_B_UROW)
  fission_unstage(acc, fission_tloop_chunked(st.ft, w, false, 1e-13, acc, static_cast<size_t>(m) * sizeof(f_t)));
  FSTEP(S_B_TLOOP)
  const int nw = fission_compact_atomic(w, m, li, lx, kFissionCapA, &cnt);
  FSTEP(S_B_COMPACT)
  fission_scatter_atomic(scs, I.Lr_p, I.Lr_i, I.Lr_x, li, lx, nw, pre, acc);
  FSTEP(S_B_LSCATTER)
  auto& dy = st.delta_y_sparse;
  if (threadIdx.x == 0) { cnt = 0; }
  __syncthreads();
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    const f_t v = acc[k];
    if (v != 0.0) {
      const int q = atomicAdd(&cnt, 1);
      dy.i[q]     = k;
      dy.x[q]     = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    dy.n = m;
    dy.i.set_size(cnt);
    dy.x.set_size(cnt);
    st.f_state = F_CHOOSE;
  }
  FSTEP(S_B_OUT)
}

__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_ftran_atomic(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_FTRAN) { return; }
  __shared__ fission_scan_smem_t scs;
  __shared__ int cnt;
  const int m   = a.m;
  const auto& I = inv[blockIdx.x];
  f_t* z   = fsh;
  f_t* acc = fsh + m;
  f_t* lx  = fsh + 2 * m;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { z[k] = 0.0, acc[k] = 0.0; }
  __syncthreads();
  FSTEP(S_F_ZERO)
  for (int r = 0; r < 2; ++r) {
    const gpu_sparse_vector& b = r == 0 ? st.rhs_sparse : st.delta_y_sparse;
    const int nb               = static_cast<int>(b.i.size());
    if (nb > kFissionCapA) { __trap(); }
    fission_scatter_atomic(scs, I.Lc_p, I.Lc_i, I.Lc_x, b.i.data(), b.x.data(), nb, pre, z);
    FSTEP(S_F_LSCATTER)
    fission_unstage(acc, fission_tloop_chunked(st.ft, z, true, 1e-16, acc, static_cast<size_t>(m) * sizeof(f_t)));
    FSTEP(S_F_TLOOP)
    const int nz = fission_compact_atomic(z, m, li, lx, kFissionCapA, &cnt);
    if (r == 0) {
      auto& ut = st.utilde_sparse;
      for (int k = threadIdx.x; k < nz; k += blockDim.x) { ut.i[k] = li[k], ut.x[k] = lx[k]; }
      if (threadIdx.x == 0) {
        ut.n = m;
        ut.i.set_size(nz);
        ut.x.set_size(nz);
      }
    }
    FSTEP(S_F_COMPACT)
    fission_scatter_atomic(scs, I.Uc_p, I.Uc_i, I.Uc_x, li, lx, nz, pre, acc);
    FSTEP(S_F_USCATTER)
    if (r == 0) {
      auto& dx = st.scaled_dxB_sparse;
      if (threadIdx.x == 0) {
        const f_t pv     = acc[st.basic_leaving];
        st.f_pivot       = pv;
        st.f_pivot_found = pv != 0.0 ? 1 : 0;
        cnt              = 0;
      }
      __syncthreads();
      for (int k = threadIdx.x; k < m; k += blockDim.x) {
        const f_t v = acc[k];
        if (v != 0.0) {
          const int q = atomicAdd(&cnt, 1);
          dx.i[q]     = k;
          dx.x[q]     = -v;  // stored negated, see fission_update
        }
        acc[k] = 0.0;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        dx.n = m;
        dx.i.set_size(cnt);
        dx.x.set_size(cnt);
      }
    } else {
      auto& v = st.ft.dense_scratch2_;
      for (int k = threadIdx.x; k < m; k += blockDim.x) { v[k] = acc[k]; }
      if (threadIdx.x == 0) {
        v.set_size(m);
        st.v_sparse.clear();
      }
    }
    __syncthreads();
    FSTEP(S_F_OUT)
  }
  if (threadIdx.x == 0) { st.f_state = F_UPDATE; }
}

inline bool gpu_fission_deterministic()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FISSION_DETERMINISTIC");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}

// C (block). dy norm (BlockReduce), steepest-edge safeguard, column-wise dz over the nonbasic
// positions with ordered touched/eligible lists (block scans), the monolith's sequential near-tie
// ratio rule over the ordered candidates (thread 0), and the dual update.
// Shared: dense dy (m doubles), candidate ratios (n-m doubles) and positions (n-m ints).


using fission_reduce_t = cub::BlockReduce<f_t, kFissionThreads>;
// choose trace sub[]: cycles of norm, dense dy, dz, ratio, dual, prep [0..5]; reset [6]; candidates
// [7]; touched columns [8]; dy nonzeros [9].
__device__ __forceinline__ void fission_choose_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r);
__global__ void __launch_bounds__(kFissionThreads, 2) fission_choose_block(kernel_args* args, i_t n_problems)
{
  __shared__ fission_utrace_t r;
  const bool tr = g_ctrace != nullptr;
  if (tr && threadIdx.x == 0) {
    r      = fission_utrace_t{};
    r.t0   = fission_gtime();
    r.args = reinterpret_cast<unsigned long long>(args);
    r.seat = blockIdx.x;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r.sm));
  }
  __syncthreads();
  fission_choose_block_body(args, n_problems, r);
  __syncthreads();
  if (tr && threadIdx.x == 0) {
    r.t1 = fission_gtime();
    g_ctrace[atomicAdd(&g_ctrace_n, 1ull) & (kUtraceCap - 1)] = r;
  }
}

__device__ __forceinline__ void fission_choose_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r)
{
  extern __shared__ f_t fsh[];
  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_CHOOSE) { return; }
  long long ct = clock64();
  const auto cmark = [&](int k) {
    if (threadIdx.x == 0) {
      const long long c = clock64();
      r.sub[k] += static_cast<int>(c - ct);
      ct = c;
    }
  };
  if (threadIdx.x == 0) {
    r.live   = 1;
    r.sub[9] = static_cast<int>(st.delta_y_sparse.i.size());
  }
  __shared__ fission_scan_smem_t scs;
  __shared__ typename fission_reduce_t::TempStorage rtmp;
  __shared__ f_t s_norm;
  __shared__ int s_reset;
  const int tid = threadIdx.x;
  const i_t m = a.m, n = a.n, nn = n - m;
  f_t* dyd  = fsh;
  f_t* crat = fsh + m;
  i_t* cpos = reinterpret_cast<i_t*>(crat + nn);
  FSTEP_BEGIN
  auto& dys      = st.delta_y_sparse;
  const i_t nzdy = static_cast<i_t>(dys.i.size());
  f_t part       = 0.0;
  for (i_t k = tid; k < nzdy; k += kFissionThreads) { part += dys.x[k] * dys.x[k]; }
  const f_t tot = fission_reduce_t(rtmp).Sum(part);
  const i_t leaving = st.leaving, direction = st.direction;
  if (tid == 0) {
    s_norm  = tot;
    s_reset = a.dy_se[leaving] < a.steepest_edge_ratio * tot ? 1 : 0;
    if (s_reset) {
      a.dy_se[leaving] = tot;
      st.f_state       = F_PRICE;
    }
  }
  __syncthreads();
  FSTEP(S_C_NORM)
  cmark(0);
  if (s_reset) {
    if (tid == 0) { r.sub[6] = 1; }
    FSTEP(S_C_SECHECK)
    return;
  }
  for (i_t k = tid; k < m; k += kFissionThreads) { dyd[k] = 0.0; }
  __syncthreads();
  for (i_t k = tid; k < nzdy; k += kFissionThreads) { dyd[dys.i[k]] = dys.x[k]; }
  __syncthreads();
  cmark(1);
  const i_t* Acs = a.A_col_start;
  const i_t* Ai  = a.A_i;
  const f_t* Ax  = a.A_x;
  f_t* dz        = st.delta_z.data();
  i_t* dzi       = st.delta_z_indices.data();
  i_t* mark      = st.delta_z_mark.data();
  const f_t ptol = a.pivot_tol, dtol = a.dual_tol / 10;
  const f_t* z   = a.z;
  int nidx = 0, ncand = 0;
  for (i_t t0 = 0; t0 < nn; t0 += kFissionThreads) {
    const i_t k  = t0 + tid;
    i_t j        = -1;
    f_t dzj      = 0.0;
    f_t ratio    = 0.0;
    int touched  = 0, eligible = 0;
    if (k < nn) {
      j = a.nonbasic_list[k];
      f_t dot = 0.0;
      for (i_t p = Acs[j]; p < Acs[j + 1]; ++p) { dot += Ax[p] * dyd[Ai[p]]; }
      dzj   = -dot;
      dz[j] = dzj;
      touched = dzj != 0.0 ? 1 : 0;
      const variable_status_t sj = a.vstatus[j];
      if (sj == variable_status_t::NONBASIC_LOWER && dzj < -ptol) {
        ratio = fmax((-dtol - z[j]) / dzj, f_t{0.0}), eligible = 1;
      } else if (sj == variable_status_t::NONBASIC_UPPER && dzj > ptol) {
        ratio = fmax((dtol - z[j]) / dzj, f_t{0.0}), eligible = 1;
      }
    }
    int pos = 0, agg = 0;
    fission_scan_t(scs.tmp).ExclusiveSum(touched, pos, agg);
    if (touched) {
      dzi[nidx + pos] = j;
      mark[j]         = 1;
    }
    nidx += agg;
    __syncthreads();
    fission_scan_t(scs.tmp).ExclusiveSum(eligible, pos, agg);
    if (eligible) {
      cpos[ncand + pos] = k;
      crat[ncand + pos] = ratio;
    }
    ncand += agg;
    __syncthreads();
  }
  if (tid == 0) {
    dz[leaving] = static_cast<f_t>(direction);
    st.delta_z_indices.set_size(nidx);
    r.sub[7] = ncand;
    r.sub[8] = nidx;
  }
  FSTEP(S_C_DZ)
  cmark(2);
  if (tid == 0) {
    // The monolith's ratio rule (warp_rt_body), in nonbasic position order.
    const f_t ztol = a.zero_tol;
    i_t entering = -1, nonbasic_entering = -1;
    f_t step = dev_inf(), min_val = dev_inf();
    for (int c = 0; c < ncand; ++c) {
      const i_t k  = cpos[c];
      const i_t j  = a.nonbasic_list[k];
      const f_t rc = crat[c];
      if (rc < min_val) {
        min_val = rc, entering = j, step = rc, nonbasic_entering = k;
      } else if (rc < min_val + ztol && fabs(z[j]) > fabs(z[entering])) {
        min_val = rc, entering = j, step = rc, nonbasic_entering = k;
      }
    }
    st.entering          = entering;
    st.nonbasic_entering = nonbasic_entering;
    st.step_length       = step;
  }
  __syncthreads();
  FSTEP(S_C_RATIO)
  cmark(3);
  const i_t entering = st.entering;
  if (entering == -1) {
    gpu_lp lp;
    gpu_settings settings;
    gpu_solution sol;
    fission_views(a, lp, settings, sol);
    const f_t obj = fission_objective<kFissionThreads>(lp, sol.x.data(), rtmp);
    if (tid < 32) { fission_exit(a, st, lp, sol, dual::status_t::DUAL_UNBOUNDED, st.iter, false, 0, &obj); }
    return;
  }
  const f_t step = st.step_length;
  for (i_t k = tid; k < nzdy; k += kFissionThreads) { a.y[dys.i[k]] += step * dys.x[k]; }
  for (i_t k = tid; k < nidx; k += kFissionThreads) { a.z[dzi[k]] += step * dz[dzi[k]]; }
  __syncthreads();
  FSTEP(S_C_DUAL)
  cmark(4);
  if (tid == 0) {
    a.z[leaving] += step * dz[leaving];
    gpu_csc_view A = make_view(m, n, a.A_nz, a.A_col_start, a.A_i, a.A_x);
    st.utilde_sparse.clear();
    st.scaled_dxB_sparse.clear();
    st.v_sparse.clear();
    st.rhs_sparse.from_csc_column(A, entering);
    st.dy_norm_sq = s_norm;
    st.f_state    = F_FTRAN;
  }
  FSTEP(S_C_PREP)
  cmark(5);
}

// C. dy norm, steepest-edge safeguard, reduced costs, ratio test, dual update.
__global__ void __launch_bounds__(32) fission_choose(kernel_args* args, i_t n_problems)
{
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_CHOOSE) { return; }
  gpu_lp lp;
  gpu_settings settings;
  gpu_solution sol;
  FSTEP_BEGIN
  fission_views(a, lp, settings, sol);
  const i_t m = a.m, n = a.n;
  auto& ft  = st.ft;
  auto& dys = st.delta_y_sparse;
  const int lane = static_cast<int>(threadIdx.x);
  const f_t* dyx = dys.x.data();
  const i_t nz   = static_cast<i_t>(dys.i.size());
  f_t part       = 0.0;
  for (i_t k = lane; k < nz; k += 32) { part += dyx[k] * dyx[k]; }
  const f_t dy_norm_sq = warp_reduce_add(part);
  FSTEP(S_C_NORM)
  const i_t leaving = st.leaving, direction = st.direction;
  if (a.dy_se[leaving] < settings.steepest_edge_ratio * dy_norm_sq) {
    ft.path(P_SE_RESET);
    lane0_serial([&] {
      a.dy_se[leaving] = dy_norm_sq;
      st.f_state       = F_PRICE;
    });
    FSTEP(S_C_SECHECK)
    return;
  }
  i_t delta_y_nz0 = 0;
  for (i_t base = 0; base < nz; base += 32) {
    const i_t k    = base + lane;
    const bool hit = (k < nz) && (fabs(dyx[k]) > 1e-12);
    delta_y_nz0 += __popc(__ballot_sync(0xffffffffu, hit));
  }
  const bool used_transpose_dz = delta_y_nz0 / static_cast<f_t>(m) * 100.0 <= c_dz_sparse_pct;
  ft.path(used_transpose_dz ? P_DZ_ROWWISE : P_DZ_COLDOT);
  if (used_transpose_dz) {
    compute_delta_z(ft, st.Arow, dys, leaving, direction, st.basic_mark.data(),
                    st.delta_z_mark.data(), st.delta_z_indices, st.delta_z.data());
  } else {
    ft.warp_to_dense(dys, st.delta_y);
    compute_reduced_cost_update(lp, a.nonbasic_list, st.delta_y.data(), leaving, direction,
                                st.delta_z_mark.data(), st.delta_z_indices, st.delta_z.data());
  }
  FSTEP(S_C_DZ)
  warp_rt_body(settings.dual_tol / 10, st.delta_z.data(), &st.entering, &st.nonbasic_entering,
               settings.pivot_tol, &st.step_length, a.vstatus, sol.z.data(), settings.zero_tol,
               a.nonbasic_list, n - m);
  __syncwarp(0xffffffffu);
  FSTEP(S_C_RATIO)
  const i_t entering = st.entering;
  if (entering == -1) {
    fission_exit(a, st, lp, sol, dual::status_t::DUAL_UNBOUNDED, st.iter, false, 0);
    return;
  }
  update_dual_variables(ft, dys, st.delta_z_indices, st.delta_z.data(), st.step_length, leaving,
                        sol.y.data(), sol.z.data());
  FSTEP(S_C_DUAL)
  lane0_serial([&] {
    st.utilde_sparse.clear();
    st.scaled_dxB_sparse.clear();
    st.v_sparse.clear();
    st.rhs_sparse.from_csc_column(lp.A, entering);
    st.dy_norm_sq = dy_norm_sq;
    st.f_state    = F_FTRAN;
  });
  FSTEP(S_C_PREP)
}

// D. FTRAN of (a_q, delta_y) through the inverses, per right-hand side b:
//   z = Linv P b, gathered per row of Linv P over the rows marked by b's columns; forward replay
//   (marking its fill); utilde = z (a_q); x = Uinv z, gathered per row of Uinv over the rows marked
//   by z's columns; scaled_dxB = x (a_q, sparse ascending), v = x (delta_y, dense).
// Shared: z, acc (m doubles), list (m ints), zbits, xbits (m bits each).
__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_ftran(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_FTRAN) { return; }
  __shared__ fission_scan_smem_t scs;
  __shared__ int cnt;
  const int m   = a.m;
  const int W   = (m + 31) >> 5;
  const auto& I = inv[blockIdx.x];
  f_t* z          = fsh;
  f_t* acc        = fsh + m;
  i_t* list       = reinterpret_cast<i_t*>(fsh + 2 * m);
  unsigned* zbits = reinterpret_cast<unsigned*>(list + m);
  unsigned* xbits = zbits + W;
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { z[k] = 0.0, acc[k] = 0.0; }
  FSTEP(S_F_ZERO)
  for (int r = 0; r < 2; ++r) {
    for (int k = threadIdx.x; k < W; k += blockDim.x) { zbits[k] = 0u, xbits[k] = 0u; }
    if (threadIdx.x == 0) { cnt = 0; }
    __syncthreads();
    const gpu_sparse_vector& b = r == 0 ? st.rhs_sparse : st.delta_y_sparse;
    const int nb               = static_cast<int>(b.i.size());
    for (int t = threadIdx.x; t < nb; t += blockDim.x) { acc[b.i[t]] = b.x[t]; }
    fission_mark_list(I.Lc_p, I.Lc_i, b.i.data(), nb, zbits, &cnt, list);
    fission_gather(I.Lr_p, I.Lr_i, I.Lr_x, list, cnt, acc, z);
    for (int t = threadIdx.x; t < nb; t += blockDim.x) { acc[b.i[t]] = 0.0; }
    if (threadIdx.x == 0) { cnt = 0; }
    __syncthreads();
    FSTEP(S_F_LSCATTER)
    fission_tloop(st.ft, z, true, 1e-16, zbits);
    FSTEP(S_F_TLOOP)
    if (r == 0) {
      auto& ut     = st.utilde_sparse;
      const int nu = fission_bits_out(scs, zbits, m, z, ut.i.data(), ut.x.data());
      if (threadIdx.x == 0) {
        ut.n = m;
        ut.i.set_size(nu);
        ut.x.set_size(nu);
      }
    }
    {
      i_t* zl      = reinterpret_cast<i_t*>(acc);  // acc is all zero here (room for 2m ints)
      const int nz = fission_bits_list(zbits, z, m, &cnt, zl);
      if (threadIdx.x == 0) { cnt = 0; }
      __syncthreads();
      fission_mark_list(I.Uc_p, I.Uc_i, zl, nz, xbits, &cnt, list);
      for (int t = threadIdx.x; t < nz; t += blockDim.x) { zl[t] = 0; }
      __syncthreads();
    }
    FSTEP(S_F_COMPACT)
    fission_gather(I.Ur_p, I.Ur_i, I.Ur_x, list, cnt, z, acc);
    FSTEP(S_F_USCATTER)
    if (r == 0) {
      auto& dx     = st.scaled_dxB_sparse;
      const int nx = fission_bits_out(scs, xbits, m, acc, dx.i.data(), dx.x.data());
      // Stored negated (the monolith's compute_delta_x negates in place); the pivot element is
      // recorded so the update phase needs no search.
      for (int q = threadIdx.x; q < nx; q += blockDim.x) { dx.x[q] = -dx.x[q]; }
      if (threadIdx.x == 0) {
        dx.n = m;
        dx.i.set_size(nx);
        dx.x.set_size(nx);
        const f_t pv     = acc[st.basic_leaving];
        st.f_pivot       = pv;
        st.f_pivot_found = pv != 0.0 ? 1 : 0;
      }
      __syncthreads();
      for (int k = threadIdx.x; k < m; k += blockDim.x) { z[k] = 0.0, acc[k] = 0.0; }
    } else {
      auto& v = st.ft.dense_scratch2_;
      for (int k = threadIdx.x; k < m; k += blockDim.x) { v[k] = acc[k]; }
      if (threadIdx.x == 0) {
        v.set_size(m);
        st.v_sparse.clear();
      }
    }
    __syncthreads();
    FSTEP(S_F_OUT)
  }
  if (threadIdx.x == 0) { st.f_state = F_UPDATE; }
}

// E. Primal step, objective, cutoff, SE norms, basis swap, refactor check, MPF update, infeasibilities.
__global__ void __launch_bounds__(32) fission_update(kernel_args* args, i_t n_problems)
{
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_UPDATE) { return; }
  gpu_lp lp;
  gpu_settings settings;
  gpu_solution sol;
  FSTEP_BEGIN
  fission_views(a, lp, settings, sol);
  auto& ft = st.ft;
  const i_t entering = st.entering, leaving = st.leaving, basic_leaving = st.basic_leaving;
  const i_t direction = st.direction, nonbasic_entering = st.nonbasic_entering;
  f_t* x = sol.x.data();
  // compute_delta_x with the pivot element recorded by FTRAN (scaled_dxB is stored negated). The
  // monolith's dense recovery for a missing pivot is not available here; its second fallback is.
  FSTEP(S_U_PIVCHECK)
  {
    auto& sdx = st.scaled_dxB_sparse;
    const f_t delta_x_leaving = direction == 1 ? lp.lower[leaving] - x[leaving] : lp.upper[leaving] - x[leaving];
    f_t scale = -st.f_pivot;
    if (st.f_pivot_found == 0) {
      ft.path(P_NAN_RECOVERY);
      if (st.delta_z[entering] == 0.0) {
        fission_exit(a, st, lp, sol, dual::status_t::NUMERICAL, st.iter, false, 0);
        return;
      }
      scale = -st.delta_z[entering];
      lane0_serial([&] {
        sdx.i.push_back(basic_leaving);
        sdx.x.push_back(scale);
      });
    }
    const f_t primal_step_length = delta_x_leaving / scale;
    const i_t nzx                = static_cast<i_t>(sdx.i.size());
    const int lane               = static_cast<int>(threadIdx.x);
    f_t* delta_x                 = st.delta_x.data();
    for (i_t k = lane; k < nzx; k += 32) { delta_x[a.basic_list[sdx.i[k]]] = primal_step_length * sdx.x[k]; }
    __syncwarp(0xffffffffu);
    if (lane == 0) {
      delta_x[leaving]  = delta_x_leaving;
      delta_x[entering] = primal_step_length;
    }
    __syncwarp(0xffffffffu);
  }
  FSTEP(S_U_DX)
  update_primal_variables(st.scaled_dxB_sparse, a.basic_list, st.delta_x.data(), entering, x);
  FSTEP(S_U_PRIMAL)
  update_objective(a.basic_list, st.scaled_dxB_sparse, lp.objective, st.delta_x.data(), entering,
                   st.obj);
  FSTEP(S_U_OBJ)
  if (isfinite(settings.cut_off) && st.obj >= settings.cut_off) { ft.path(P_CUTOFF_SCAN); }
  if (isfinite(settings.cut_off) && st.obj >= settings.cut_off &&
      dual_feasible_within_tol(lp, a.vstatus, sol.z.data(), settings.dual_tol)) {
    fission_exit(a, st, lp, sol, dual::status_t::CUTOFF, st.iter + 1, false, 0);
    return;
  }
  FSTEP(S_U_CUTOFF)
  update_steepest_edge_norms(settings, a.basic_list, ft, direction, st.delta_y_sparse, st.dy_norm_sq,
                             st.scaled_dxB_sparse, basic_leaving, entering, st.v.data(), st.v_sparse,
                             a.dy_se);
  FSTEP(S_U_SENORM)
  lane0_serial([&] {
    a.vstatus[entering] = variable_status_t::BASIC;
    a.vstatus[leaving]  = (lp.lower[leaving] == lp.upper[leaving])
                            ? variable_status_t::NONBASIC_FIXED
                            : static_cast<variable_status_t>(-direction);
    a.basic_list[basic_leaving]        = entering;
    a.nonbasic_list[nonbasic_entering] = leaving;
    st.nonbasic_pos[leaving]           = nonbasic_entering;
    st.basic_mark[leaving]             = -1;
    st.basic_mark[entering]            = basic_leaving;
  });
  FSTEP(S_U_SWAP)
  if (ft.num_updates() > settings.refactor_frequency) {
    ft.path(P_REFACTOR);
    fission_exit(a, st, lp, sol, dual::status_t::REFACTOR_NEEDED, st.iter + 1, true, 0);
    return;
  }
  FSTEP(S_U_REFCHECK)
  i_t fu = ft.warp_update(st.utilde_sparse, st.UTsol_sparse, basic_leaving);
  fu     = lane0_bcast(fu);
  FSTEP(S_U_MPF)
  if (fu != 0) {
    fission_exit(a, st, lp, sol, dual::status_t::REFACTOR_NEEDED, st.iter + 1, true, 1);
    return;
  }
  fission_infeas_update(lp, settings, a.basic_list, x, st.scaled_dxB_sparse, leaving, st.squared_infeas,
                        st.infeas_idx);
  FSTEP(S_U_INFEAS)
  clear_delta_z(entering, leaving, st.delta_z_mark.data(), st.delta_z_indices, st.delta_z.data());
  lane0_serial([&] {
    st.iter++;
    st.f_state = F_PRICE;
  });
  FSTEP(S_U_TAIL)
}

// Update phase, one block per seat. Warp 0 runs the factor update (needs only utilde, UTsol and
// the leaving position) while warps 1..7 run dx -> x -> objective -> cutoff -> steepest edge ->
// swap. Then the whole block runs the infeasibility update and the clean-up. Exits keep the warp
// kernel's priority: numerical, cutoff, refactor check, factor-update failure.
constexpr int kUpdThreads = 256;
constexpr int kUpdPrimal  = kUpdThreads - 32;
using fission_upd_scan_t   = cub::BlockScan<int, kUpdThreads>;
__device__ __forceinline__ void fission_primal_sync() { asm volatile("bar.sync 1, %0;" ::"n"(kUpdPrimal) : "memory"); }

__device__ __forceinline__ void fission_update_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r);
__global__ void __launch_bounds__(kUpdThreads) fission_update_block(kernel_args* args, i_t n_problems)
{
  __shared__ fission_utrace_t r;
  const bool tr = g_utrace != nullptr;
  if (tr && threadIdx.x == 0) {
    r      = fission_utrace_t{};
    r.t0   = fission_gtime();
    r.args = reinterpret_cast<unsigned long long>(args);
    r.seat = blockIdx.x;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r.sm));
  }
  __syncthreads();
  fission_update_block_body(args, n_problems, r);
  __syncthreads();
  if (tr && threadIdx.x == 0) {
    r.t1 = fission_gtime();
    g_utrace[atomicAdd(&g_utrace_n, 1ull) & (kUtraceCap - 1)] = r;
  }
}

__device__ __forceinline__ void fission_update_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r)
{
  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_UPDATE) { return; }
  if (threadIdx.x == 0) {
    r.live = 1;
    r.nzx  = static_cast<int>(st.scaled_dxB_sparse.i.size());
    r.unz  = static_cast<int>(st.utilde_sparse.i.size());
    r.enz  = static_cast<int>(st.UTsol_sparse.i.size());
  }
  const long long c_beg = clock64();
  __shared__ typename fission_upd_scan_t::TempStorage scan_tmp;
  __shared__ f_t s_part[kUpdThreads / 32];
  __shared__ int s_cut, s_fu;
  gpu_lp lp;
  gpu_settings settings;
  gpu_solution sol;
  FSTEP_BEGIN
  fission_views(a, lp, settings, sol);
  auto& ft  = st.ft;
  auto& sdx = st.scaled_dxB_sparse;
  const int tid = static_cast<int>(threadIdx.x);
  const i_t entering = st.entering, leaving = st.leaving, basic_leaving = st.basic_leaving;
  const i_t direction = st.direction, nonbasic_entering = st.nonbasic_entering;
  f_t* x = sol.x.data();
  f_t scale = -st.f_pivot;
  __shared__ typename cub::BlockReduce<f_t, kUpdThreads>::TempStorage otmp;
  if (st.f_pivot_found == 0) {
    if (st.delta_z[entering] == 0.0) {
      const f_t obj = fission_objective<kUpdThreads>(lp, x, otmp);
      if (tid < 32) {
        ft.path(P_NAN_RECOVERY);
        fission_exit(a, st, lp, sol, dual::status_t::NUMERICAL, st.iter, false, 0, &obj);
      }
      return;
    }
    scale = -st.delta_z[entering];
    if (tid == 0) {
      ft.path(P_NAN_RECOVERY);
      sdx.i.push_back(basic_leaving);
      sdx.x.push_back(scale);
    }
  }
  if (tid == 0) { s_cut = 0, s_fu = 0; }
  __syncthreads();
  FSTEP(S_U_PIVCHECK)
  if (tid == 32) { fst_ = clock64(); }
  const bool refac = ft.num_updates() > settings.refactor_frequency;  // read before warp 0 updates
  __syncthreads();
  if (tid < 32) {
    if (!refac) {
      const i_t fu = ft.warp_update_unsorted(st.utilde_sparse, st.UTsol_sparse, basic_leaving,
                                             g_utrace != nullptr ? r.sub : nullptr);
      if (tid == 0) { s_fu = fu; }
    }
    if (tid == 0) { r.c_mpf = static_cast<int>(clock64() - c_beg); }
    FSTEP(S_U_MPF)
  } else {
    const int t   = tid - 32;
    const i_t nzx = static_cast<i_t>(sdx.i.size());
    const f_t delta_x_leaving =
      direction == 1 ? lp.lower[leaving] - x[leaving] : lp.upper[leaving] - x[leaving];
    const f_t primal_step_length = delta_x_leaving / scale;
    f_t* delta_x                 = st.delta_x.data();
    for (i_t k = t; k < nzx; k += kUpdPrimal) { delta_x[a.basic_list[sdx.i[k]]] = primal_step_length * sdx.x[k]; }
    fission_primal_sync();
    if (t == 0) {
      delta_x[leaving]  = delta_x_leaving;
      delta_x[entering] = primal_step_length;
    }
    fission_primal_sync();
    FSTEP_AT(S_UP_DX, 32)
    f_t part = 0.0;
    for (i_t k = t; k < nzx; k += kUpdPrimal) {
      const i_t j = a.basic_list[sdx.i[k]];
      x[j] += delta_x[j];
      part += delta_x[j] * lp.objective[j];
    }
    if (t == 0) { x[entering] += delta_x[entering]; }
    FSTEP_AT(S_UP_PRIMAL, 32)
    part = warp_reduce_add(part);
    if ((t & 31) == 0) { s_part[t / 32] = part; }
    fission_primal_sync();
    if (t == 0) {
      f_t total = 0.0;
      for (int w = 0; w < kUpdPrimal / 32; ++w) { total += s_part[w]; }
      st.obj += total;
      st.obj += delta_x[entering] * lp.objective[entering];
    }
    fission_primal_sync();
    FSTEP_AT(S_UP_OBJ, 32)
    if (isfinite(settings.cut_off) && st.obj >= settings.cut_off) {
      int bad = 0;
      for (i_t j = t; j < a.n; j += kUpdPrimal) {
        const variable_status_t sj = a.vstatus[j];
        const f_t zj               = a.z[j];
        if (sj != variable_status_t::BASIC && !isfinite(zj)) { bad = 1; }
        else if (sj == variable_status_t::NONBASIC_LOWER && zj < -settings.dual_tol) { bad = 1; }
        else if (sj == variable_status_t::NONBASIC_UPPER && zj > settings.dual_tol) { bad = 1; }
        else if (sj == variable_status_t::NONBASIC_FREE && fabs(zj) > settings.dual_tol) { bad = 1; }
      }
      if (t == 0) { s_cut = 1; }
      fission_primal_sync();
      if (bad) { s_cut = 0; }
      fission_primal_sync();
    }
    FSTEP_AT(S_UP_CUTOFF, 32)
    if (s_cut == 0) {
      // update_steepest_edge_norms; the leaving coefficient of scaled_dxB is `scale`.
      f_t* v = st.v.data();
      const bool v_dense = ft.dense_scratch2_.size() == static_cast<size_t>(st.delta_y_sparse.n);
      if (v_dense) {
        v = ft.dense_scratch2_.data();
      } else {
        for (i_t k = t; k < static_cast<i_t>(st.v_sparse.i.size()); k += kUpdPrimal) { v[st.v_sparse.i[k]] += st.v_sparse.x[k]; }
        fission_primal_sync();
      }
      const f_t wr = -scale;
      if (wr != 0) {
        const f_t dyn    = st.dy_norm_sq;
        const f_t omegar = dyn / (wr * wr);
        f_t* dyse        = a.dy_se;
        for (i_t h = t; h < nzx; h += kUpdPrimal) {
          const i_t k = sdx.i[h];
          const i_t j = a.basic_list[k];
          if (k == basic_leaving) {
            const f_t w    = sdx.x[h];
            dyse[entering] = (1.0 / (w * w)) * dyn;
          } else {
            const f_t wk      = -sdx.x[h];
            const f_t vk      = direction == -1 ? -v[k] : v[k];
            const f_t new_val = dyse[j] + wk * (2.0 * vk / wr + wk * omegar);
            dyse[j]           = fmax(new_val, 1e-4);
          }
        }
      }
      if (!v_dense) {
        fission_primal_sync();
        for (i_t k = t; k < static_cast<i_t>(st.v_sparse.i.size()); k += kUpdPrimal) { v[st.v_sparse.i[k]] = 0.0; }
      }
      fission_primal_sync();
      FSTEP_AT(S_UP_SENORM, 32)
      if (t == 0) {
        a.vstatus[entering] = variable_status_t::BASIC;
        a.vstatus[leaving]  = (lp.lower[leaving] == lp.upper[leaving])
                                ? variable_status_t::NONBASIC_FIXED
                                : static_cast<variable_status_t>(-direction);
        a.basic_list[basic_leaving]        = entering;
        a.nonbasic_list[nonbasic_entering] = leaving;
        st.nonbasic_pos[leaving]           = nonbasic_entering;
        st.basic_mark[leaving]             = -1;
        st.basic_mark[entering]            = basic_leaving;
      }
      FSTEP_AT(S_UP_SWAP, 32)
    }
    if (tid == 32) { r.c_primal = static_cast<int>(clock64() - c_beg); }
  }
  __syncthreads();
  FSTEP(S_U_JOIN)
  if (s_cut != 0 || refac || s_fu != 0) {
    const f_t obj = s_cut != 0 ? fission_objective<kUpdThreads>(lp, x, otmp) : f_t{0.0};
    if (tid < 32) {
      if (s_cut != 0) {
        fission_exit(a, st, lp, sol, dual::status_t::CUTOFF, st.iter + 1, false, 0, &obj);
      } else {
        if (refac) { ft.path(P_REFACTOR); }
        fission_exit(a, st, lp, sol, dual::status_t::REFACTOR_NEEDED, st.iter + 1, true, refac ? 0 : 1);
      }
    }
    return;
  }
  FSTEP(S_U_REFCHECK)
  {
    // fission_infeas_update over the block, same list order.
    f_t* sq   = st.squared_infeas.data();
    i_t* idx  = st.infeas_idx.data();
    if (tid == 0) { sq[leaving] = 0.0; }
    __syncthreads();
    i_t count     = static_cast<i_t>(st.infeas_idx.size());
    const i_t nzx = static_cast<i_t>(sdx.i.size());
    for (i_t base = 0; base < nzx; base += kUpdThreads) {
      const i_t k = base + tid;
      int add     = 0;
      i_t j       = -1;
      if (k < nzx) {
        j                = a.basic_list[sdx.i[k]];
        const f_t infeas = fmax(lp.lower[j] - x[j], x[j] - lp.upper[j]);
        if (infeas > settings.primal_tol) {
          add   = sq[j] == 0.0 ? 1 : 0;
          sq[j] = infeas * infeas;
        } else {
          sq[j] = 0.0;
        }
      }
      int pos = 0, agg = 0;
      fission_upd_scan_t(scan_tmp).ExclusiveSum(add, pos, agg);
      if (add) { idx[count + pos] = j; }
      count += agg;
      __syncthreads();
    }
    i_t out = 0;
    for (i_t base = 0; base < count; base += kUpdThreads) {
      const i_t k   = base + tid;
      const i_t j   = k < count ? idx[k] : -1;
      const int keep = (j >= 0 && sq[j] != 0.0) ? 1 : 0;
      int pos = 0, agg = 0;
      fission_upd_scan_t(scan_tmp).ExclusiveSum(keep, pos, agg);
      __syncthreads();  // all reads of this stride before any write into it
      if (keep) { idx[out + pos] = j; }
      out += agg;
      __syncthreads();
    }
    if (tid == 0) { st.infeas_idx.set_size(out); }
  }
  FSTEP(S_U_INFEAS)
  {
    const i_t nz = st.delta_z_indices.size();
    for (i_t k = tid; k < nz; k += kUpdThreads) {
      const i_t j          = st.delta_z_indices[k];
      st.delta_z[j]        = 0.0;
      st.delta_z_mark[j]   = 0;
    }
    __syncthreads();
    if (tid == 0) {
      st.delta_z[entering] = 0.0;
      st.delta_z[leaving]  = 0.0;
      st.delta_z_indices.clear();
      st.iter++;
      st.f_state = F_PRICE;
    }
  }
  FSTEP(S_U_TAIL)
}

// Straggler cut: every live seat at an iteration boundary (F_PRICE) stops as REFACTOR_NEEDED
// (reason 0); the host resumes it next generation. Seats mid-iteration reach F_PRICE within a round.
__global__ void __launch_bounds__(32) fission_yield(kernel_args* args, i_t n_problems)
{
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_PRICE) { return; }
  gpu_lp lp;
  gpu_settings settings;
  gpu_solution sol;
  fission_views(a, lp, settings, sol);
  fission_exit(a, st, lp, sol, dual::status_t::REFACTOR_NEEDED, st.iter, true, 0);
}

__global__ void fission_count(const kernel_args* args, i_t n_problems, int* live)
{
  const i_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_problems) { return; }
  const kernel_args& a = args[s];
  if (a.done == 0 && a.slot != nullptr && a.slot->f_state != F_DONE) { atomicAdd(live, 1); }
}

inline bool gpu_fission()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FISSION");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}

inline double gpu_fission_yield()
{
  static const double f = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FISSION_YIELD");
    return e != nullptr ? std::atof(e) : 0.1;
  }();
  return f;
}

inline int gpu_fission_rounds()
{
  static const int k = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FISSION_ROUNDS");
    return e != nullptr ? std::max(1, std::atoi(e)) : 8;
  }();
  return k;
}
