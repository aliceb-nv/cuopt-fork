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
  S_F1_LSCATTER, S_F1_TLOOP, S_F1_USCATTER,  // FTRAN dy solve (r = 1); S_F_* above: rhs solve
  S_U_PIVCHECK, S_U_DX, S_U_PRIMAL, S_U_OBJ, S_U_CUTOFF, S_U_SENORM, S_U_SWAP, S_U_REFCHECK,
  S_U_MPF, S_U_INFEAS, S_U_TAIL, S_U_JOIN,
  // fission_update_block's primal chain (warps 1..7), concurrent with S_U_MPF: not in the total.
  S_UP_DX, S_UP_PRIMAL, S_UP_OBJ, S_UP_CUTOFF, S_UP_SENORM, S_UP_SWAP,
  S_TL_DOT, S_TL_AXPY, S_TL_UPD,  // EXP: replay internals (warp 0), not in the total
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
// CUOPT_EXP_SCATSTAT: per scatter class (FTRAN L r0/r1, FTRAN U r0/r1, BTRAN L): calls, sum n, sum E,
// sum of the longest segment, and log2 histograms of n and E.
constexpr int kScatClasses = 5;
__device__ unsigned long long g_fscat[kScatClasses][4];
__device__ unsigned long long g_fscat_h[kScatClasses][2][20];
__device__ unsigned long long g_ftl[2][3];  // replay (ascending, descending): calls, sum K, sum nonzero d
__device__ fission_utrace_t* g_utrace;
__device__ unsigned long long g_utrace_n;
__device__ fission_utrace_t* g_ctrace;  // CUOPT_BNB_GPU_FISSION_CTRACE: same, per choose block
__device__ unsigned long long g_ctrace_n;
__device__ fission_utrace_t* g_strace;  // CUOPT_BNB_GPU_FISSION_STRACE: same, per setup block
__device__ unsigned long long g_strace_n;
__device__ fission_utrace_t* g_ftrace;  // CUOPT_BNB_GPU_FISSION_FTRACE: same, per FTRAN block
__device__ unsigned long long g_ftrace_n;

// Per-pool counters of the list-capacity fallbacks (gpu_fission_inv::fb), read by the host after
// each relaunch: calls of each solve, dense fallbacks per site, the longest list that fell back.
enum fission_fb_t : int {
  FB_BTRAN_CALLS = 0,  // BTRAN solves
  FB_FTRAN_CALLS,      // FTRAN solves (two right-hand sides each)
  FB_BTRAN_DENSE,      // BTRAN L step: w over capacity, dense row gather over Lc
  FB_FTRAN_U0_DENSE,   // FTRAN U step of a_q: z over capacity, dense row gather over Ur
  FB_FTRAN_U1_DENSE,   // FTRAN U step of delta_y (CUOPT_EXP_FTRAN_DY=scatter only)
  FB_FTRAN_IN_CHUNKED, // FTRAN L step: input list over capacity, scattered in chunks
  FB_MAX_LIST,         // longest list that took a fallback
  // Orchestration statistics (same block, kernel_args::fstats):
  FB_CHOOSE_CALLS,     // choose passes that read a candidate row
  FB_SE_RESETS,        // of those, steepest-edge weight below the ratio test: reset, priced again
  FB_GROUPS,           // groups of rounds run (graph WHILE iterations / host-loop iterations)
  FB_SEAT_FRESH,       // seats set up per kind: fresh node
  FB_SEAT_CHILD,       //   child continuation (host plunge or device dive)
  FB_SEAT_RESUME,      //   host-refactor resume
  FB_CHILD_K,          // sum over children of their starting MPF update count
  FB_YIELDED,          // seats suspended by CUOPT_EXP_FISSION_YIELD
  FB_SEAT_INPLACE,     // seats resumed in place (no setup)
  // choose passes [0..6] and resets [7..13] by seat kind (fresh, child, resume) and by the seat's
  // iteration when choosing (<10, <50, <200, >=200)
  FB_CHOOSE_BY,
  FB_RESET_BY = FB_CHOOSE_BY + 7,
  FB_COUNT    = FB_RESET_BY + 7
};
__device__ __forceinline__ void fission_fb_note(unsigned long long* fb, int site, int len)
{
  if (fb != nullptr && threadIdx.x == 0) {
    atomicAdd(fb + site, 1ull);
    if (site >= FB_BTRAN_DENSE && site <= FB_FTRAN_IN_CHUNKED) {
      atomicMax(fb + FB_MAX_LIST, static_cast<unsigned long long>(len));
    }
  }
}

constexpr unsigned long long kUtraceCap = 1ull << 21;
__device__ __forceinline__ unsigned long long fission_gtime()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// CHOSEN: 512 so that 2 blocks per SM (the shared-memory limit) get 64 registers per thread; at 1024
// the 32-register cap spilled (A100: more local-load sectors than global-load sectors in FTRAN).
constexpr int kFissionThreads = 512;
constexpr int kShortDot = 2;   // CHOSEN: replay dots up to this length without shuffles
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
  if (args[idx].in_place != 0 && args[idx].done == 0) {
    // Yielded in place: continue from the saved step; nothing to set up.
    if (threadIdx.x == 0) {
      kernel_args& a = args[idx];
      a.slot->f_state = a.slot->f_yield_state;
      a.slot->f_lean  = 0;
      a.in_place      = 0;
      if (a.fstats != nullptr) { atomicAdd(a.fstats + FB_SEAT_INPLACE, 1ull); }
    }
    emit();
    return;
  }
  const bool child_in = args[idx].command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE);
  const bool live_in  = args[idx].done == 0;
  gpu_phase2_kernel_body<true, true, true>(args, idx);
  kernel_args& a = args[idx];
  if (live_in && threadIdx.x == 0 && a.fstats != nullptr) {
    const int kind = child_in ? FB_SEAT_CHILD : a.refactor_resume != 0 ? FB_SEAT_RESUME : FB_SEAT_FRESH;
    atomicAdd(a.fstats + kind, 1ull);
    if (child_in && a.slot != nullptr) {
      atomicAdd(a.fstats + FB_CHILD_K, static_cast<unsigned long long>(a.slot->ft.num_updates_));
    }
  }
  if (a.done != 0) {
    emit();
    return;
  }
  const long long c6 = clock64();
  if (tr && threadIdx.x == 0) {
    r.live = 1;
    for (int q = 0; q < 6; ++q) { r.sub[q] = static_cast<int>(a.slot->pre[q]); }
  }
  if (threadIdx.x == 0) { a.slot->f_mrows = a.slot->ft.num_updates_ == 0 ? 0 : -1; }
  __syncwarp(0xffffffffu);
  if (a.slot->f_lean != 0) {  // fission_xb finishes this seat
    emit();
    return;
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

// A seat that starts with K inherited MPF updates (a child continuation) has no replay matrix R, so
// every BTRAN / FTRAN replays them through fission_tloop_chunked. Build R from the update store, row
// by row as the FTRAN tail would have (M[k][j] = col(2k+1).col(2j), w_j = M[k][j] / mu_j,
// R[k][i] = w_i - sum_{i<j<k} w_j R[j][i], stored at i * kc + k), then use the closed-form replay.
// Shared: one dense m-vector.
constexpr int kBuildRThreads = 256;
__global__ void __launch_bounds__(kBuildRThreads) fission_build_r(kernel_args* args, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0 || a.slot == nullptr || a.fm == nullptr) { return; }
  auto& st    = *a.slot;
  const i_t K = st.ft.num_updates_;
  if (st.f_mrows != -1 || K <= 0 || K >= a.fm_kcap) { return; }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = kBuildRThreads / 32;
  const i_t* cs = st.ft.S_.col_start.data();
  const i_t* ri = st.ft.S_.i.data();
  const f_t* vx = st.ft.S_.x.data();
  const f_t* mu = st.ft.mu_values_.data();
  f_t* Rm         = a.fm;
  const size_t kc = static_cast<size_t>(a.fm_kcap);
  f_t* acc        = fsh;
  __shared__ f_t w[kBuildRThreads];  // K < fm_kcap <= kBuildRThreads (checked by the host)
  for (i_t i = threadIdx.x; i < a.m; i += kBuildRThreads) { acc[i] = 0.0; }
  __syncthreads();
  for (i_t k = 1; k < K; ++k) {
    for (i_t p = cs[2 * k + 1] + threadIdx.x; p < cs[2 * k + 2]; p += kBuildRThreads) { acc[ri[p]] = vx[p]; }
    __syncthreads();
    for (i_t j = warp; j < k; j += nw) {
      f_t part = 0.0;
      for (i_t p = cs[2 * j] + lane; p < cs[2 * j + 1]; p += 32) { part += vx[p] * acc[ri[p]]; }
      part = warp_reduce_add(part);
      if (lane == 0) { w[j] = part / mu[j]; }
    }
    __syncthreads();
    for (i_t p = cs[2 * k + 1] + threadIdx.x; p < cs[2 * k + 2]; p += kBuildRThreads) { acc[ri[p]] = 0.0; }
    for (i_t i = warp; i < k; i += nw) {
      f_t part = 0.0;
      for (i_t j = i + 1 + lane; j < k; j += 32) { part += w[j] * Rm[i * kc + j]; }
      part = warp_reduce_add(part);
      if (lane == 0) { Rm[i * kc + k] = w[i] - part; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { st.f_mrows = K; }
}

// Fresh fission seat: x_B = U0^{-1} L0^{-1} (b - A_N x_N) through the explicit inverses (row-wise,
// one owner per output, |x_j| < 10 tight_tol skipped as in compute_primal_variables), then the
// objective c'x and the infeasibility list. Shared: two m-vectors.
constexpr int kXbThreads = 512;
__global__ void __launch_bounds__(kXbThreads) fission_xb(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems);

// Duals of a fresh host-factorized seat from the inverses, before setup places the nonbasics:
// y' = (c_B' U0^-1) (L0^-1 P), z_N = c_N - A_N' y, z_B = 0. Shared: two m-vectors.
__global__ void __launch_bounds__(kXbThreads) fission_duals(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0 || a.refactor_resume != 0 || a.device_lu != 0 ||
      a.command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE)) {
    return;
  }
  const i_t m = a.m, n = a.n;
  const int tid = threadIdx.x;
  f_t* cb = fsh;
  f_t* w  = fsh + m;
  for (i_t k = tid; k < m; k += kXbThreads) { cb[k] = a.objective[a.basic_list[k]]; }
  __syncthreads();
  const auto& I = inv[blockIdx.x];
  for (i_t c = tid; c < m; c += kXbThreads) {
    f_t v = 0.0;
    for (i_t p = I.Uc_p[c]; p < I.Uc_p[c + 1]; ++p) { v += I.Uc_x[p] * cb[I.Uc_i[p]]; }
    w[c] = v;
  }
  __syncthreads();
  f_t* y = cb;
  for (i_t j = tid; j < m; j += kXbThreads) {
    f_t v = 0.0;
    for (i_t p = I.Lc_p[j]; p < I.Lc_p[j + 1]; ++p) { v += I.Lc_x[p] * w[I.Lc_i[p]]; }
    y[j]   = v;
    a.y[j] = v;
  }
  __syncthreads();
  for (i_t j = tid; j < n; j += kXbThreads) {
    if (a.vstatus[j] == variable_status_t::BASIC) {
      a.z[j] = 0.0;
      continue;
    }
    f_t dot = 0.0;
    for (i_t p = a.A_col_start[j]; p < a.A_col_start[j + 1]; ++p) { dot += a.A_x[p] * y[a.A_i[p]]; }
    a.z[j] = a.objective[j] - dot;
  }
}

__global__ void __launch_bounds__(kXbThreads) fission_xb(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_lean == 0) { return; }
  __shared__ typename cub::BlockReduce<f_t, kXbThreads>::TempStorage otmp;
  const i_t m = a.m, n = a.n;
  const int tid = threadIdx.x;
  f_t* r = fsh;
  f_t* z = fsh + m;
  f_t* x = a.x;
  const f_t tt = a.tight_tol * 10;
  for (i_t i = tid; i < m; i += kXbThreads) {
    f_t v = a.rhs[i];
    for (i_t p = a.Arow_row_start[i]; p < a.Arow_row_start[i + 1]; ++p) {
      const i_t j = a.Arow_j[p];
      if (a.vstatus[j] == variable_status_t::BASIC) { continue; }
      const f_t xj = x[j];
      if (fabs(xj) < tt) { continue; }
      v -= a.Arow_x[p] * xj;
    }
    r[i] = v;
  }
  __syncthreads();
  const auto& I = inv[blockIdx.x];
  for (i_t i = tid; i < m; i += kXbThreads) {
    f_t v = 0.0;
    for (i_t p = I.Lr_p[i]; p < I.Lr_p[i + 1]; ++p) { v += I.Lr_x[p] * r[I.Lr_i[p]]; }
    z[i] = v;
  }
  __syncthreads();
  for (i_t k = tid; k < m; k += kXbThreads) {
    f_t v = 0.0;
    for (i_t p = I.Ur_p[k]; p < I.Ur_p[k + 1]; ++p) { v += I.Ur_x[p] * z[I.Ur_i[p]]; }
    x[a.basic_list[k]] = v;
  }
  __syncthreads();
  f_t part = 0.0;
  for (i_t j = tid; j < n; j += kXbThreads) { part += a.objective[j] * x[j]; }
  const f_t obj = cub::BlockReduce<f_t, kXbThreads>(otmp).Sum(part);
  if (tid == 0) { st.obj = obj; }
  if (tid < 32) {
    gpu_lp lp;
    gpu_settings settings;
    gpu_solution sol;
    fission_views(a, lp, settings, sol);
    f_t* sq = st.squared_infeas.data();
    for (i_t j = tid; j < n; j += 32) { sq[j] = 0.0; }
    __syncwarp(0xffffffffu);
    compute_initial_primal_infeasibilities(lp, settings, st.ft, a.basic_list, x, st.squared_infeas,
                                           st.infeas_idx, st.primal_inf);
  }
}

// DEBUG (CUOPT_EXP_XBCHECK=1): seats fission_duals will price whose basis is invalid (basic_list
// entry out of range or not BASIC in vstatus). out[0] seats checked, out[1] bad, out[2] first bad
// seat, out[3] its command | refactor_resume << 4 | lean-eligible << 8, out[4] its bad entries.
__global__ void fission_basis_check(const kernel_args* args, i_t n_problems, unsigned long long* out)
{
  if (blockIdx.x >= n_problems) { return; }
  const kernel_args& a = args[blockIdx.x];
  if (a.done != 0 || a.refactor_resume != 0 || a.device_lu != 0 ||
      a.command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE)) {
    return;
  }
  __shared__ int bad;
  if (threadIdx.x == 0) { bad = 0; }
  __syncthreads();
  for (i_t k = threadIdx.x; k < a.m; k += blockDim.x) {
    const i_t j = a.basic_list[k];
    if (j < 0 || j >= a.n || a.vstatus[j] != variable_status_t::BASIC) { atomicAdd(&bad, 1); }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(out, 1ull);
    if (bad != 0 && atomicAdd(out + 1, 1ull) == 0) {
      out[2] = blockIdx.x, out[3] = a.command | (a.refactor_resume << 4), out[4] = bad;
    }
  }
}

// DEBUG (CUOPT_EXP_NANCHECK=1): seats (not done before this step) with a NaN in x, y, z or dy_se.
// out[0] count, out[1] first seat, out[2] which array (1 x, 2 y, 3 z, 4 dy_se), out[3] its lean flag.
__global__ void fission_nan_check(const kernel_args* args, i_t n_problems, unsigned long long* out)
{
  if (blockIdx.x >= n_problems) { return; }
  const kernel_args& a = args[blockIdx.x];
  if (a.slot == nullptr) { return; }
  __shared__ int which;
  if (threadIdx.x == 0) { which = 0; }
  __syncthreads();
  for (i_t j = threadIdx.x; j < a.n; j += blockDim.x) {
    if (isnan(a.x[j])) { atomicMax(&which, 1); }
    if (isnan(a.z[j])) { atomicMax(&which, 3); }
    if (isnan(a.dy_se[j])) { atomicMax(&which, 4); }
  }
  for (i_t i = threadIdx.x; i < a.m; i += blockDim.x) {
    if (isnan(a.y[i])) { atomicMax(&which, 2); }
  }
  __syncthreads();
  if (threadIdx.x == 0 && which != 0) {
    if (atomicAdd(out, 1ull) == 0) {
      out[1] = blockIdx.x, out[2] = which, out[3] = a.slot->f_lean;
      out[4] = a.command | (a.refactor_resume << 4) | (a.transpose_on_device << 8) | (a.device_lu << 12);
      out[5] = reinterpret_cast<unsigned long long>(a.L0t_x), out[6] = reinterpret_cast<unsigned long long>(a.L0_x);
      out[7] = a.L0t_nz;
    }
  }
}

// DEBUG (CUOPT_EXP_XBCHECK=1): max_i |b_i - (A x)_i| of each lean seat after fission_xb.
// out[0] seats checked, out[1] seats over 1e-6, out[2] bits of the max residual, out[3] its seat.
__global__ void __launch_bounds__(kXbThreads) fission_xb_check(const kernel_args* args, i_t n_problems,
                                                              unsigned long long* out)
{
  if (blockIdx.x >= n_problems) { return; }
  const kernel_args& a = args[blockIdx.x];
  if (a.done != 0 || a.slot->f_lean == 0) { return; }
  __shared__ typename cub::BlockReduce<f_t, kXbThreads>::TempStorage tmp;
  f_t worst = 0.0;
  for (i_t i = threadIdx.x; i < a.m; i += kXbThreads) {
    f_t v = a.rhs[i];
    for (i_t p = a.Arow_row_start[i]; p < a.Arow_row_start[i + 1]; ++p) { v -= a.Arow_x[p] * a.x[a.Arow_j[p]]; }
    worst = fmax(worst, fabs(v));
  }
  const f_t w = cub::BlockReduce<f_t, kXbThreads>(tmp).Reduce(worst, [](f_t p, f_t q) { return fmax(p, q); });
  if (threadIdx.x == 0) {
    atomicAdd(out, 1ull);
    if (w > 1e-6) { atomicAdd(out + 1, 1ull); }
    const unsigned long long bits = __double_as_longlong(w);
    if (atomicMax(out + 2, bits) < bits) { out[3] = blockIdx.x; }
  }
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
__device__ __forceinline__ void fission_price_body(kernel_args* args, i_t n_problems)
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

__global__ void __launch_bounds__(32) fission_price(kernel_args* args, i_t n_problems)
{
  fission_price_body(args, n_problems);
}

// Deterministic block helpers. Lists are index-ordered (block scans), every float sum has a fixed
// order (one owner thread per output entry, summing a row/column segment in storage order), and
// pattern marks are idempotent writes. Results are bitwise reproducible run to run.

// out[0..count) = ascending k in [0, m) with pred(k); count returned on every thread. Each thread
// owns a contiguous chunk of up to kChunk indices per pass (blocked arrangement), so one block scan
// of per-thread counts orders a pass of kChunk * kFissionThreads indices.
using fission_scan_t = cub::BlockScan<int, kFissionThreads, cub::BLOCK_SCAN_WARP_SCANS>;  // small temp storage
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
// With a queue (lq, qcap entries, count *qn), the thread pass queues the long segments' list
// positions so that the warps visit only those (overflow: the warps scan the whole list as without one).
__device__ __forceinline__ void fission_gather(const i_t* cp, const i_t* ri, const f_t* cx,
                                               const i_t* list, int n, const f_t* in, f_t* out,
                                               int* lq = nullptr, int qcap = 0, int* qn = nullptr,
                                               int shortseg = kShortSeg)
{
  if (lq != nullptr) {
    if (threadIdx.x == 0) { *qn = 0; }
    __syncthreads();
  }
  for (int q = threadIdx.x; q < n; q += kFissionThreads) {
    const i_t i = list[q];
    const i_t b = cp[i], e = cp[i + 1];
    if (e - b > shortseg) {
      if (lq != nullptr) {
        const int k = atomicAdd(qn, 1);
        if (k < qcap) { lq[k] = q; }
      }
      continue;
    }
    f_t s0 = 0.0, s1 = 0.0;
    i_t p = b;
    for (; p + 1 < e; p += 2) {
      s0 += cx[p] * in[ri[p]];
      s1 += cx[p + 1] * in[ri[p + 1]];
    }
    if (p < e) { s0 += cx[p] * in[ri[p]]; }
    out[i] = s0 + s1;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lq != nullptr) {
    __syncthreads();
    const int nq = *qn;
    if (nq <= qcap) {
      for (int k = warp; k < nq; k += kFissionThreads / 32) {
        const i_t i = list[lq[k]];
        const i_t b = cp[i], e = cp[i + 1];
        f_t s = 0.0;
        for (i_t p = b + lane; p < e; p += 32) { s += cx[p] * in[ri[p]]; }
        s = warp_reduce_add(s);
        if (lane == 0) { out[i] = s; }
      }
      __syncthreads();
      return;
    }
  }
  for (int q = warp; q < n; q += kFissionThreads / 32) {
    const i_t i = list[q];
    const i_t b = cp[i], e = cp[i + 1];
    if (e - b <= shortseg) { continue; }
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
  __shared__ int big[2];     // 1: buffer holds one update replayed from global memory
  // Next chunk starting at replay position s0, planned by threads [t0, t0 + nt) in parallel: each
  // tests one candidate end (sizes grow with the end), the largest that fits wins. bar() syncs them.
  auto plan = [&](int b, i_t s0, int t0, int nt, auto bar) {
    if (t0 == 0) { rng[b][0] = s0, rng[b][1] = s0, big[b] = 0; }
    bar();
    for (i_t s1 = s0 + 1 + t0; s1 <= num && s1 <= s0 + nt; s1 += nt) {
      const i_t ka = asc ? s0 : num - s1;
      const i_t kb = asc ? s1 : num - s0;
      if (fission_chunk_bytes(cs, ka, kb) <= half) { atomicMax(&rng[b][1], s1); }
    }
    bar();
    if (t0 == 0 && rng[b][1] == s0 && s0 < num) { rng[b][1] = s0 + 1, big[b] = 1; }
    bar();
  };
  auto bar_all     = [] { __syncthreads(); };
  auto bar_loaders = [] { asm volatile("bar.sync 1, %0;" ::"r"(kFissionThreads - 32)); };
  auto load = [&](int b, int t0, int nt) {
    const i_t s0 = rng[b][0], s1 = rng[b][1];
    if (s0 >= s1 || big[b]) { return; }
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
    if (big[b]) {
      const i_t k    = asc ? s0 : num - 1 - s0;
      const i_t dcol = asc ? 2 * k + 1 : 2 * k;
      const i_t acol = asc ? 2 * k : 2 * k + 1;
      f_t partial    = 0.0;
      for (i_t p = cs[dcol] + lane; p < cs[dcol + 1]; p += 32) { partial += gvx[p] * x[gri[p]]; }
      const f_t theta = warp_reduce_add(partial) / gmu[k];
      if (fabs(theta) > tol) {
        for (i_t p = cs[acol] + lane; p < cs[acol + 1]; p += 32) { x[gri[p]] -= theta * gvx[p]; }
      }
      __syncwarp(0xffffffffu);
      return;
    }
    const i_t ka = asc ? s0 : num - s1, kb = asc ? s1 : num - s0;
    const i_t E = cs[2 * kb] - cs[2 * ka], n = kb - ka;
    const f_t* svx = buf[b];
    const f_t* smu = svx + E;
    const i_t* sri = reinterpret_cast<const i_t*>(smu + n);
    const i_t* scs = sri + E;
    long long c_dot = 0, c_axpy = 0;
    for (i_t q = 0; q < n; ++q) {
      const long long q0 = clock64();
      const i_t kl   = asc ? q : n - 1 - q;  // local update index
      const i_t dcol = asc ? 2 * kl + 1 : 2 * kl;
      const i_t acol = asc ? 2 * kl : 2 * kl + 1;
      const i_t d0 = scs[dcol], d1 = scs[dcol + 1];
      f_t dot;
      if (d1 - d0 <= kShortDot) {
        // Every lane forms the short dot itself (broadcast reads, no shuffles); the barrier keeps
        // these reads of x ahead of the axpy writes below.
        f_t d[4] = {0.0, 0.0, 0.0, 0.0};
        i_t p    = d0;
        for (; p + 4 <= d1; p += 4) {
#pragma unroll
          for (int u = 0; u < 4; ++u) { d[u] += svx[p + u] * x[sri[p + u]]; }
        }
        for (int u = 0; p < d1; ++p, ++u) { d[u] += svx[p] * x[sri[p]]; }
        dot = (d[0] + d[1]) + (d[2] + d[3]);
        __syncwarp(0xffffffffu);
      } else {
        f_t partial = 0.0;
        for (i_t p = d0 + lane; p < d1; p += 32) { partial += svx[p] * x[sri[p]]; }
        dot = warp_reduce_add(partial);
      }
      const f_t theta = dot * smu[kl];
      const long long q1 = clock64();
      if (fabs(theta) > tol) {
        for (i_t p = scs[acol] + lane; p < scs[acol + 1]; p += 32) { x[sri[p]] -= theta * svx[p]; }
      }
      __syncwarp(0xffffffffu);
      c_dot += q1 - q0, c_axpy += clock64() - q1;
    }
    if (lane == 0) {
      atomicAdd(&g_fstep[S_TL_DOT][0], static_cast<unsigned long long>(c_dot));
      atomicAdd(&g_fstep[S_TL_AXPY][0], static_cast<unsigned long long>(c_axpy));
      atomicAdd(&g_fstep[S_TL_UPD][1], static_cast<unsigned long long>(n));
      atomicAdd(&g_fstep[S_TL_AXPY][1], 1ull);
      atomicAdd(&g_fstep[S_TL_DOT][1], 1ull);
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

// Product-form replay with the thetas in closed form. With M[k][j] = col(2k+1).col(2j) (j < k), the
// ascending replay solves (D + M) theta = d and the descending one (D + M^T) theta = d, D = diag(mu),
// d the dots of the input. R (strictly lower, kept per seat, see the FTRAN tail) gives
//   asc:  theta = D^-1 (d - R d)        desc: theta = e - R^T e,  e = D^-1 d.
// The monolith's drop of |theta| <= tol before later thetas use it is not reproduced; zero thetas skip
// their axpy. scratch: 2K doubles of shared memory; with sm (the block scan storage), 2K + 3K +
// kFissionThreads doubles: the ascending R d is split over idle threads and the axpys go through the
// entry-balanced scatter (CUOPT_EXP_TLOOP=old: thread-per-k sums and warp-per-k axpys).
__device__ __forceinline__ void fission_scatter_atomic(fission_scan_smem_t& sm, const i_t* cp, const i_t* __restrict__ ri,
                                                       const f_t* __restrict__ cx, const i_t* idx, const f_t* val, int n,
                                                       int* pre, int* sb, f_t* acc, int cls);
__device__ __forceinline__ void fission_tloop_m(const gpu_basis_update_t& ft, f_t* x, bool asc, f_t tol,
                                                const f_t* R, int kc, f_t* scratch, size_t,
                                                fission_scan_smem_t* sm = nullptr, f_t* wsave = nullptr,
                                                bool timed = true)
{
  const bool fast = sm != nullptr && c_tloop_old == 0;
  const i_t K = ft.num_updates_;
  if (K == 0) { return; }
  const i_t* cs = ft.S_.col_start.data();
  const i_t* ri = ft.S_.i.data();
  const f_t* vx = ft.S_.x.data();
  const f_t* mu = ft.mu_values_.data();
  f_t* d        = scratch;
  f_t* th       = scratch + K;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  // CUOPT_EXP_SCATSTAT=<k>: ascending replay phase cycles (thread 0) into tl.dot / tl.updates /
  // tl.axpy, for replays of at least k - 1 updates.
  const bool clk = timed && c_scatstat != 0 && asc && threadIdx.x == 0 && K + 1 >= c_scatstat;
  long long tc   = clk ? clock64() : 0;
  const auto tmark = [&](int s) {
    if (clk) {
      const long long c = clock64();
      atomicAdd(&g_fstep[s][0], static_cast<unsigned long long>(c - tc));
      atomicAdd(&g_fstep[s][1], 1ull);
      tc = c;
    }
  };
  for (i_t k = warp; k < K; k += nw) {
    const i_t c = asc ? 2 * k + 1 : 2 * k;
    f_t part    = 0.0;
    for (i_t p = cs[c] + lane; p < cs[c + 1]; p += 32) { part += vx[p] * x[ri[p]]; }
    part = warp_reduce_add(part);
    if (lane == 0) {
      d[k] = asc ? part : part / mu[k];
      // wsave (descending, BTRAN): d_k = sgn w_k for the FTRAN tail's row K of R, stride kc.
      if (wsave != nullptr) { wsave[static_cast<size_t>(k) * kc] = d[k]; }
    }
  }
  __syncthreads();
  tmark(S_TL_DOT);
  if (c_scatstat != 0) {
    // Density of d: replay calls, sum of K, sum of nonzero d (ascending / descending).
    __shared__ int s_nzd;
    if (threadIdx.x == 0) { s_nzd = 0; }
    __syncthreads();
    int nzd = 0;
    for (i_t k = threadIdx.x; k < K; k += blockDim.x) { nzd += d[k] != 0.0 ? 1 : 0; }
    atomicAdd(&s_nzd, nzd);
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&g_ftl[asc ? 0 : 1][0], 1ull);
      atomicAdd(&g_ftl[asc ? 0 : 1][1], static_cast<unsigned long long>(K));
      atomicAdd(&g_ftl[asc ? 0 : 1][2], static_cast<unsigned long long>(s_nzd));
    }
  }
  if (fast && K <= kFissionThreads) {
    // Only the nonzero d_i contribute: their indices, ascending, then R reads of those columns only.
    int* nzl = reinterpret_cast<int*>(scratch + 2 * K);
    __shared__ int s_nl;
    {
      const int flag = (static_cast<i_t>(threadIdx.x) < K && d[threadIdx.x] != 0.0) ? 1 : 0;
      int pos = 0, L = 0;
      fission_scan_t(sm->tmp).ExclusiveSum(flag, pos, L);
      if (flag) { nzl[pos] = threadIdx.x; }
      if (threadIdx.x == 0) { s_nl = L; }
    }
    __syncthreads();
    const int L = s_nl;
    if (asc) {
      // theta_k = (d_k - sum_{i<k, d_i != 0} R[k][i] d_i) / mu_k; lanes over k (coalesced R reads).
      for (i_t k = threadIdx.x; k < K; k += blockDim.x) {
        f_t acc[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll 4
        for (int q = 0; q < L; ++q) {
          const int i = nzl[q];
          if (i < k) { acc[q & 3] += R[static_cast<size_t>(i) * kc + k] * d[i]; }
        }
        const f_t t = (d[k] - ((acc[0] + acc[1]) + (acc[2] + acc[3]))) / mu[k];
        th[k]       = fabs(t) <= tol ? f_t{0.0} : t;
      }
    } else {
      // theta_k = e_k - sum_{i>k, e_i != 0} R[i][k] e_i: a warp per k, lanes over the list.
      for (i_t k = warp; k < K; k += nw) {
        f_t part = 0.0;
        for (int q = lane; q < L; q += 32) {
          const int i = nzl[q];
          if (i > k) { part += R[static_cast<size_t>(k) * kc + i] * d[i]; }
        }
        part = warp_reduce_add(part);
        if (lane == 0) {
          const f_t t = d[k] - part;
          th[k]       = fabs(t) <= tol ? f_t{0.0} : t;
        }
      }
    }
  } else if (asc) {
    // theta_k = (d_k - sum_{i<k} R[k][i] d_i) / mu_k; R[k][i] at i * kc + k: coalesced over k.
    for (i_t k = threadIdx.x; k < K; k += blockDim.x) {
      f_t acc[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll 8
      for (i_t i = 0; i < k; ++i) { acc[i & 3] += R[static_cast<size_t>(i) * kc + k] * d[i]; }
      const f_t t = (d[k] - ((acc[0] + acc[1]) + (acc[2] + acc[3]))) / mu[k];
      th[k]       = fabs(t) <= tol ? f_t{0.0} : t;
    }
  } else {
    // theta_k = e_k - sum_{i>k} R[i][k] e_i; column k of R is contiguous: a warp per k.
    for (i_t k = warp; k < K; k += nw) {
      f_t part = 0.0;
      for (i_t i = k + 1 + lane; i < K; i += 32) { part += R[static_cast<size_t>(k) * kc + i] * d[i]; }
      part = warp_reduce_add(part);
      if (lane == 0) {
        const f_t t = d[k] - part;
        th[k]       = fabs(t) <= tol ? f_t{0.0} : t;
      }
    }
  }
  __syncthreads();
  tmark(S_TL_UPD);
  if (fast && K <= kFissionThreads) {
    // x -= sum_k theta_k col(c_k) over the nonzero thetas, as segments of the balanced scatter.
    f_t* nv  = scratch + 2 * K + kFissionThreads;
    int* ci  = reinterpret_cast<int*>(nv + K);
    int* pre = ci + K;
    int* sb  = pre + K;
    __shared__ int s_nt;
    {
      const i_t k    = threadIdx.x;
      const int flag = (k < K && th[k] != 0.0) ? 1 : 0;
      int pos = 0, nt = 0;
      fission_scan_t(sm->tmp).ExclusiveSum(flag, pos, nt);
      if (flag) {
        ci[pos] = asc ? 2 * k : 2 * k + 1;
        nv[pos] = -th[k];
      }
      if (threadIdx.x == 0) { s_nt = nt; }
    }
    __syncthreads();
    fission_scatter_atomic(*sm, cs, ri, vx, ci, nv, s_nt, pre, sb, x, -1);
    tmark(S_TL_AXPY);
    return;
  }
  for (i_t k = warp; k < K; k += nw) {
    const f_t t = th[k];
    if (t == 0.0) { continue; }
    const i_t c = asc ? 2 * k : 2 * k + 1;
    for (i_t p = cs[c] + lane; p < cs[c + 1]; p += 32) { atomicAdd(&x[ri[p]], -t * vx[p]); }
  }
  __syncthreads();
  tmark(S_TL_AXPY);
}

// In-place level-scheduled triangular solve over the transposed factor (row j's entries in column j
// of `T`), schedule op `o` built by setup: level l's rows are co[ls[l-1] .. ls[l]); rows absent from
// the schedule have only a unit diagonal. One thread per row, a block barrier between levels.
__device__ __forceinline__ void fission_level_solve(const gpu_csc_view& T, const i_t* ls, const i_t* co,
                                                    int nlev, bool diag_first, f_t* x)
{
  const i_t* cs = T.col_start.data();
  const i_t* ri = T.i.data();
  const f_t* vx = T.x.data();
  for (int l = 1; l <= nlev; ++l) {
    for (i_t q = ls[l - 1] + threadIdx.x; q < ls[l]; q += blockDim.x) {
      const i_t j  = co[q] & ~kColNoop;
      const i_t c0 = cs[j], c1 = cs[j + 1];
      const i_t lo = diag_first ? c0 + 1 : c0, hi = diag_first ? c1 : c1 - 1;
      f_t v        = x[j];
      for (i_t p = lo; p < hi; ++p) { v -= vx[p] * x[ri[p]]; }
      x[j] = v / (diag_first ? vx[c0] : vx[c1 - 1]);
    }
    __syncthreads();
  }
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
constexpr size_t kFissionStage = 32768;  // CHOSEN: 2 blocks/SM at m = 5236 (8m + 32 KB + static)
static_assert(kFissionStage >= kFissionCapA * (sizeof(f_t) + 2 * sizeof(i_t)), "lists live in the stage"); 

// DEBUG (CUOPT_EXP_XBCHECK): count of iteration-0 choose exits with no entering column, then the
// first one's seat, leaving, direction, x, lower, upper of the leaving column, ncand, nidx, nzdy.
__device__ double g_unb0[16];

// Compaction of a dense shared vector into (index, value) lists by an atomic counter. Returns the
// nonzero count. Up to cap: the lists hold them and the vector is cleared. Over cap: the lists are
// partial and the vector is left intact for a dense fallback.
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
  }
  __syncthreads();
  const int c = *cnt;
  if (c <= cap) {
    for (int q = threadIdx.x; q < c; q += blockDim.x) { dense[li[q]] = 0.0; }
    __syncthreads();
  }
  return c;
}

// acc[ri[p]] += cx[p] * val[t] over the segments idx[t]. Segments of up to kShortSeg entries: one
// thread; longer: one warp, lane-strided (coalesced, and a segment's rows are distinct, so the lanes
// of one atomic never collide). No scan or per-entry search.
__device__ __forceinline__ void fission_scatter_seg(const i_t* cp, const i_t* __restrict__ ri,
                                                    const f_t* __restrict__ cx, const i_t* idx, const f_t* val,
                                                    int n, f_t* acc)
{
  for (int t = threadIdx.x; t < n; t += kFissionThreads) {
    const i_t c = idx[t];
    const i_t b = __ldg(cp + c), e = __ldg(cp + c + 1);
    if (e - b > kShortSeg) { continue; }
    const f_t v = val[t];
    for (i_t p = b; p < e; ++p) { atomicAdd(&acc[__ldg(ri + p)], __ldg(cx + p) * v); }
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int t = warp; t < n; t += kFissionThreads / 32) {
    const i_t c = idx[t];
    const i_t b = __ldg(cp + c), e = __ldg(cp + c + 1);
    if (e - b <= kShortSeg) { continue; }
    const f_t v = val[t];
    for (i_t p = b + lane; p < e; p += 32) { atomicAdd(&acc[__ldg(ri + p)], __ldg(cx + p) * v); }
  }
  __syncthreads();
}

// out[i] = sum over row i of (rp, rj, rx) of rx[p] * in[rj[p]] for every row i < m: no atomics. Rows up to
// kLongRow entries: one thread; longer: queued in q (qcap entries; overflow rows stay with their thread),
// then one warp each.
constexpr int kLongRow = 32;  // CHOSEN
__device__ __forceinline__ void fission_gather_rows(const i_t* __restrict__ rp, const i_t* __restrict__ rj,
                                                    const f_t* __restrict__ rx, int m, const f_t* in, f_t* out,
                                                    int* q, int qcap, int* qn)
{
  if (threadIdx.x == 0) { *qn = 0; }
  __syncthreads();
  for (int i = threadIdx.x; i < m; i += kFissionThreads) {
    const i_t b = __ldg(rp + i), e = __ldg(rp + i + 1);
    if (e - b > kLongRow) {
      const int k = atomicAdd(qn, 1);
      if (k < qcap) {
        q[k] = i;
        continue;
      }
    }
    f_t s0 = 0.0, s1 = 0.0;
    i_t p = b;
    for (; p + 1 < e; p += 2) {
      s0 += __ldg(rx + p) * in[__ldg(rj + p)];
      s1 += __ldg(rx + p + 1) * in[__ldg(rj + p + 1)];
    }
    if (p < e) { s0 += __ldg(rx + p) * in[__ldg(rj + p)]; }
    out[i] = s0 + s1;
  }
  __syncthreads();
  const int nq = min(*qn, qcap);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int k = warp; k < nq; k += kFissionThreads / 32) {
    const i_t i = q[k];
    const i_t b = __ldg(rp + i), e = __ldg(rp + i + 1);
    f_t s = 0.0;
    for (i_t p = b + lane; p < e; p += 32) { s += __ldg(rx + p) * in[__ldg(rj + p)]; }
    s = warp_reduce_add(s);
    if (lane == 0) { out[i] = s; }
  }
  __syncthreads();
}

// Same sum, entries balanced across the block (default; CUOPT_EXP_SCATTER=segment selects the above).
// Each thread takes kScatBatch entries per pass and runs their searches, loads and atomics side by
// side, so one pass costs about one entry's latency chain. sb: the segments' start offsets (n ints).
constexpr int kScatBatch = 4;  // CHOSEN
__device__ __forceinline__ void fission_scatter_atomic(fission_scan_smem_t& sm, const i_t* cp, const i_t* __restrict__ ri,
                                                       const f_t* __restrict__ cx, const i_t* idx, const f_t* val, int n,
                                                       int* pre, int* sb, f_t* acc, int cls)
{
  if (c_scatter_seg != 0) {
    fission_scatter_seg(cp, ri, cx, idx, val, n, acc);
    return;
  }
  __shared__ int total;
  constexpr int kItems = (kFissionCapA + kFissionThreads - 1) / kFissionThreads;
  int len[kItems];
  int ml = 0;
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    const int t = threadIdx.x * kItems + k;
    len[k]      = 0;
    if (t < n) {
      const i_t c = idx[t];
      const i_t b = __ldg(cp + c);
      sb[t]       = b;
      len[k]      = __ldg(cp + c + 1) - b;
    }
    ml = max(ml, len[k]);
  }
  int agg = 0;
  fission_scan_t(sm.tmp).ExclusiveSum(len, len, agg);
#pragma unroll
  for (int k = 0; k < kItems; ++k) {
    const int t = threadIdx.x * kItems + k;
    if (t < n) { pre[t] = len[k]; }
  }
  if (threadIdx.x == 0) { total = agg; }
  __syncthreads();
  const int E = total;
  if (c_scatstat != 0 && cls >= 0) {
    __shared__ int s_ml;
    if (threadIdx.x == 0) { s_ml = 0; }
    __syncthreads();
    atomicMax(&s_ml, ml);
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(&g_fscat[cls][0], 1ull);
      atomicAdd(&g_fscat[cls][1], static_cast<unsigned long long>(n));
      atomicAdd(&g_fscat[cls][2], static_cast<unsigned long long>(E));
      atomicAdd(&g_fscat[cls][3], static_cast<unsigned long long>(s_ml));
      atomicAdd(&g_fscat_h[cls][0][min(19, 32 - __clz(n))], 1ull);
      atomicAdd(&g_fscat_h[cls][1][min(19, 32 - __clz(E))], 1ull);
    }
  }
  // Segment of entry e: the largest lo with pre[lo] <= e (fixed-trip, branch-free over the batch).
  const int top = n > 1 ? 1 << (31 - __clz(n - 1)) : 0;
  for (int e0 = threadIdx.x; e0 < E; e0 += kScatBatch * kFissionThreads) {
    int e[kScatBatch], lo[kScatBatch];
#pragma unroll
    for (int k = 0; k < kScatBatch; ++k) { e[k] = e0 + k * kFissionThreads, lo[k] = 0; }
    for (int step = top; step > 0; step >>= 1) {
#pragma unroll
      for (int k = 0; k < kScatBatch; ++k) {
        const int c = lo[k] + step;
        if (c < n && pre[c] <= e[k]) { lo[k] = c; }
      }
    }
    i_t row[kScatBatch];
    f_t prod[kScatBatch];
#pragma unroll
    for (int k = 0; k < kScatBatch; ++k) {
      if (e[k] < E) {
        const int p = sb[lo[k]] + (e[k] - pre[lo[k]]);
        row[k]      = __ldg(ri + p);
        prod[k]     = __ldg(cx + p) * val[lo[k]];
      }
    }
#pragma unroll
    for (int k = 0; k < kScatBatch; ++k) {
      if (e[k] < E) { atomicAdd(&acc[row[k]], prod[k]); }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void fission_btran_atomic_body(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
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
  f_t* R   = fsh + m;  // replay staging, then the lists
  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { w[k] = 0.0; }
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
  if (st.f_mrows == st.ft.num_updates_) {
    const i_t K = st.ft.num_updates_;
    fission_tloop_m(st.ft, w, false, 1e-13, a.fm, a.fm_kcap, R, kFissionStage / sizeof(f_t), &scs,
                    K < a.fm_kcap ? a.fm + K : nullptr);
  } else {
    fission_tloop_chunked(st.ft, w, false, 1e-13, R, kFissionStage);
  }
  FSTEP(S_B_TLOOP)
  fission_fb_note(I.fb, FB_BTRAN_CALLS, 0);
  const int capA = c_fb_force != 0 ? 0 : kFissionCapA;  // CUOPT_EXP_FB_FORCE=1: every list falls back
  const int nw   = fission_compact_atomic(w, m, li, lx, capA, &cnt);
  FSTEP(S_B_COMPACT)
  if (nw <= capA) {
    fission_scatter_atomic(scs, I.Lr_p, I.Lr_i, I.Lr_x, li, lx, nw, pre, pre + kFissionCapA, w, 4);
  } else {
    // Dense fallback: (w' Linv)_j = sum over column j of Linv of Lc_x * w, into the per-seat scratch.
    fission_fb_note(I.fb, FB_BTRAN_DENSE, nw);
    f_t* out = st.ft.dense_scratch2_.data();
    fission_gather_rows(I.Lc_p, I.Lc_i, I.Lc_x, m, w, out, li, 2 * kFissionCapA, &cnt);
    for (int k = threadIdx.x; k < m; k += blockDim.x) { w[k] = out[k]; }
    __syncthreads();
  }
  FSTEP(S_B_LSCATTER)
  auto& dy = st.delta_y_sparse;
  if (threadIdx.x == 0) { cnt = 0; }
  __syncthreads();
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    const f_t v = w[k];
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
  fission_btran_atomic(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  fission_btran_atomic_body(args, inv, n_problems);
}

__device__ __forceinline__ void fission_ftran_atomic_body(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems);
__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_ftran_atomic(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  __shared__ fission_utrace_t r;
  const bool tr = g_ftrace != nullptr;
  if (tr && threadIdx.x == 0) {
    r      = fission_utrace_t{};
    r.t0   = fission_gtime();
    r.args = reinterpret_cast<unsigned long long>(args);
    r.seat = blockIdx.x;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r.sm));
    if (blockIdx.x < n_problems && args[blockIdx.x].done == 0 && args[blockIdx.x].slot->f_state == F_FTRAN) {
      r.live = 1;
      r.nzx  = args[blockIdx.x].slot->ft.num_updates_;
      r.unz  = static_cast<int>(args[blockIdx.x].slot->rhs_sparse.i.size());
      r.enz  = static_cast<int>(args[blockIdx.x].slot->delta_y_sparse.i.size());
    }
  }
  __syncthreads();
  fission_ftran_atomic_body(args, inv, n_problems);
  __syncthreads();
  if (tr && threadIdx.x == 0) {
    r.t1 = fission_gtime();
    g_ftrace[atomicAdd(&g_ftrace_n, 1ull) & (kUtraceCap - 1)] = r;
  }
}
__device__ __forceinline__ void fission_ftran_atomic_body(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_FTRAN) { return; }
  __shared__ fission_scan_smem_t scs;
  __shared__ int cnt;
  const int m   = a.m;
  const auto& I = inv[blockIdx.x];
  f_t* z   = fsh;
  f_t* R   = fsh + m;  // replay staging, then the lists
  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  f_t* acc = z;      // the U step scatters back into z, zero after the compaction
  const int capU = c_fb_force != 0 ? 0 : kFissionCapA;  // CUOPT_EXP_FB_FORCE=1: every list falls back
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { z[k] = 0.0; }
  __syncthreads();
  FSTEP(S_F_ZERO)
  for (int r = 0; r < 2; ++r) {
    const gpu_sparse_vector& b = r == 0 ? st.rhs_sparse : st.delta_y_sparse;
    if (r == 0) { fission_fb_note(I.fb, FB_FTRAN_CALLS, 0); }
    const int nb               = static_cast<int>(b.i.size());
    const bool direct = c_ftran_direct != 0;
    if (direct) {
      const i_t* ip = st.ft.inverse_row_permutation_.data();
      for (int p = threadIdx.x; p < nb; p += blockDim.x) { z[ip[b.i[p]]] = b.x[p]; }
      __syncthreads();
      fission_level_solve(st.ft.L0_transpose_, st.lvl_ls[0].data(), st.lvl_co[0].data(), st.cfg_nlev[0], false, z);
    } else {
      // The L step is additive in its input: a list over capacity goes through in chunks.
      const int chunk = c_fb_force != 0 ? 64 : kFissionCapA;
      if (nb > chunk) { fission_fb_note(I.fb, FB_FTRAN_IN_CHUNKED, nb); }
      for (int o = 0; o < nb || o == 0; o += chunk) {
        fission_scatter_atomic(scs, I.Lc_p, I.Lc_i, I.Lc_x, b.i.data() + o, b.x.data() + o,
                               min(chunk, nb - o), pre, pre + kFissionCapA, z, r);
      }
    }
    FSTEP(r == 0 ? S_F_LSCATTER : S_F1_LSCATTER)
    if (st.f_mrows == st.ft.num_updates_) {
      fission_tloop_m(st.ft, z, true, 1e-16, a.fm, a.fm_kcap, R, kFissionStage / sizeof(f_t), &scs, nullptr,
                      c_scatstat_r < 0 || c_scatstat_r == r);
    } else {
      fission_tloop_chunked(st.ft, z, true, 1e-16, R, kFissionStage);
    }
    FSTEP(r == 0 ? S_F_TLOOP : S_F1_TLOOP)
    // The dy solve's output v is read only at the rows of the rhs solve's result (the steepest-edge
    // update over scaled_dxB): by default gather just those rows of U0^-1 from z; CUOPT_EXP_FTRAN_DY=
    // gather: every row; =scatter: the full scatter.
    const bool dy_gather = r == 1 && !direct && c_ftran_dy != 1;
    int nz = 0;
    if (dy_gather) {
    } else if (direct) {
      // utilde from the dense L-space vector, which the U solve then overwrites in place.
      if (r == 0) {
        if (threadIdx.x == 0) { cnt = 0; }
        __syncthreads();
        auto& ut = st.utilde_sparse;
        for (int k = threadIdx.x; k < m; k += blockDim.x) {
          if (z[k] != 0.0) {
            const int q = atomicAdd(&cnt, 1);
            ut.i[q] = k, ut.x[q] = z[k];
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          ut.n = m;
          ut.i.set_size(cnt);
          ut.x.set_size(cnt);
        }
      }
    } else {
      nz = fission_compact_atomic(z, m, li, lx, capU, &cnt);
    }
    const bool u_dense = nz > capU;  // z left intact: dense fallback for the U step
    if (!direct && r == 0) {
      auto& ut = st.utilde_sparse;
      if (!u_dense) {
        for (int k = threadIdx.x; k < nz; k += blockDim.x) { ut.i[k] = li[k], ut.x[k] = lx[k]; }
      } else {
        __syncthreads();
        if (threadIdx.x == 0) { cnt = 0; }
        __syncthreads();
        for (int k = threadIdx.x; k < m; k += blockDim.x) {
          if (z[k] != 0.0) {
            const int q = atomicAdd(&cnt, 1);
            ut.i[q] = k, ut.x[q] = z[k];
          }
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        ut.n = m;
        ut.i.set_size(nz);
        ut.x.set_size(nz);
      }
    }
    FSTEP(S_F_COMPACT)
    if (dy_gather && c_ftran_dy == 2) {
      fission_gather_rows(I.Ur_p, I.Ur_i, I.Ur_x, m, z, st.ft.dense_scratch2_.data(), li, 2 * kFissionCapA, &cnt);
    } else if (dy_gather) {
      const auto& dx = st.scaled_dxB_sparse;
      if (c_scatstat != 0) {
        // Stats class 3 (ftran.U dy): rows gathered, their total length, the longest.
        __shared__ unsigned long long s_e;
        __shared__ int s_l;
        if (threadIdx.x == 0) { s_e = 0ull, s_l = 0; }
        __syncthreads();
        const int nr = static_cast<int>(dx.i.size());
        unsigned long long e = 0;
        int l = 0;
        for (int q = threadIdx.x; q < nr; q += blockDim.x) {
          const int len = I.Ur_p[dx.i[q] + 1] - I.Ur_p[dx.i[q]];
          e += len, l = max(l, len);
        }
        atomicAdd(&s_e, e);
        atomicMax(&s_l, l);
        __syncthreads();
        if (threadIdx.x == 0) {
          atomicAdd(&g_fscat[3][0], 1ull);
          atomicAdd(&g_fscat[3][1], static_cast<unsigned long long>(nr));
          atomicAdd(&g_fscat[3][2], s_e);
          atomicAdd(&g_fscat[3][3], static_cast<unsigned long long>(s_l));
          atomicAdd(&g_fscat_h[3][0][min(19, 32 - __clz(nr))], 1ull);
          atomicAdd(&g_fscat_h[3][1][min(19, 32 - __clzll(static_cast<long long>(s_e)))], 1ull);
        }
      }
      fission_gather(I.Ur_p, I.Ur_i, I.Ur_x, dx.i.data(), static_cast<int>(dx.i.size()), z,
                     st.ft.dense_scratch2_.data(), li, 2 * kFissionCapA, &cnt, kLongRow);
    } else if (direct) {
      fission_level_solve(st.ft.U0_transpose_, st.lvl_ls[1].data(), st.lvl_co[1].data(), st.cfg_nlev[1], true, acc);
    } else if (!u_dense) {
      fission_scatter_atomic(scs, I.Uc_p, I.Uc_i, I.Uc_x, li, lx, nz, pre, pre + kFissionCapA, acc, 2 + r);
    } else {
      // Dense fallback: (Uinv z)_i over row i of Uinv, into the per-seat scratch, then back into acc.
      fission_fb_note(I.fb, r == 0 ? FB_FTRAN_U0_DENSE : FB_FTRAN_U1_DENSE, nz);
      f_t* out = st.ft.dense_scratch2_.data();
      fission_gather_rows(I.Ur_p, I.Ur_i, I.Ur_x, m, z, out, li, 2 * kFissionCapA, &cnt);
      for (int k = threadIdx.x; k < m; k += blockDim.x) { acc[k] = out[k]; }
      __syncthreads();
    }
    FSTEP(r == 0 ? S_F_USCATTER : S_F1_USCATTER)
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
      if (dy_gather) {
        for (int k = threadIdx.x; k < m; k += blockDim.x) { z[k] = 0.0; }
      } else {
        for (int k = threadIdx.x; k < m; k += blockDim.x) { v[k] = acc[k], acc[k] = 0.0; }
      }
      if (threadIdx.x == 0) {
        v.set_size(m);
        st.v_sparse.clear();
      }
    }
    __syncthreads();
    FSTEP(S_F_OUT)
  }
  // Row K = num_updates_ of R for the coming update. With w_j = M[K][j] / mu_j, M[K][j] = UTsol.col(2j):
  //   R[K][i] = w_i - sum_{i<j<K} w_j R[j][i]   (R[j][i] at i * kc + j). acc is zero; R region free.
  {
    const i_t K = st.ft.num_updates_;
    if (st.f_mrows == K && K < a.fm_kcap) {
      f_t* w         = R;
      const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
      f_t* Rm         = a.fm;
      const size_t kc = static_cast<size_t>(a.fm_kcap);
      if (c_tail_old == 0 && K <= kFissionThreads) {
        // BTRAN's replay left d_j = sgn w_j in row K of R (fission_tloop_m's wsave): no dots here,
        // and the sums run over the nonzero w only.
        const f_t sgn = static_cast<f_t>(-st.direction);
        int* nzl      = reinterpret_cast<int*>(w + K);
        {
          const i_t j    = threadIdx.x;
          const f_t wj   = j < K ? sgn * Rm[static_cast<size_t>(j) * kc + K] : f_t{0.0};
          if (j < K) { w[j] = wj; }
          const int flag = wj != 0.0 ? 1 : 0;
          int pos = 0, L = 0;
          fission_scan_t(scs.tmp).ExclusiveSum(flag, pos, L);
          if (flag) { nzl[pos] = j; }
          if (threadIdx.x == 0) { cnt = L; }
        }
        __syncthreads();
        const int L = cnt;
        for (i_t iq = warp; iq < K; iq += blockDim.x >> 5) {
          f_t part = 0.0;
          for (int q = lane; q < L; q += 32) {
            const int jq = nzl[q];
            if (jq > iq) { part += w[jq] * Rm[iq * kc + jq]; }
          }
          part = warp_reduce_add(part);
          if (lane == 0) { Rm[iq * kc + K] = w[iq] - part; }
        }
        __syncthreads();
      } else {
        const auto& UT = st.UTsol_sparse;
        const i_t nu   = static_cast<i_t>(UT.i.size());
        for (i_t p = threadIdx.x; p < nu; p += blockDim.x) { acc[UT.i[p]] = UT.x[p]; }
        __syncthreads();
        const i_t* cs = st.ft.S_.col_start.data();
        const i_t* ri = st.ft.S_.i.data();
        const f_t* vx = st.ft.S_.x.data();
        const f_t* mu = st.ft.mu_values_.data();
        for (i_t jq = warp; jq < K; jq += blockDim.x >> 5) {
          f_t part = 0.0;
          for (i_t p = cs[2 * jq] + lane; p < cs[2 * jq + 1]; p += 32) { part += vx[p] * acc[ri[p]]; }
          part = warp_reduce_add(part);
          if (lane == 0) { w[jq] = part / mu[jq]; }
        }
        __syncthreads();
        for (i_t p = threadIdx.x; p < nu; p += blockDim.x) { acc[UT.i[p]] = 0.0; }
        for (i_t iq = warp; iq < K; iq += blockDim.x >> 5) {
          f_t part = 0.0;
          for (i_t jq = iq + 1 + lane; jq < K; jq += 32) { part += w[jq] * Rm[iq * kc + jq]; }
          part = warp_reduce_add(part);
          if (lane == 0) { Rm[iq * kc + K] = w[iq] - part; }
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) { st.f_mrow_k = K; }
    } else if (threadIdx.x == 0) {
      st.f_mrow_k = -1;
    }
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
  f_t* crat = fsh + m;          // candidate ratio
  f_t* caz  = crat + nn;        // candidate |z_j|
  i_t* cpos = reinterpret_cast<i_t*>(caz + nn);  // candidate nonbasic position k
  i_t* cj   = cpos + nn;        // candidate column j
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
    if (a.fstats != nullptr) {
      atomicAdd(a.fstats + FB_CHOOSE_CALLS, 1ull);
      const int kind = a.command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE) ? 1
                       : a.refactor_resume != 0                                              ? 2
                                                                                             : 0;
      const int it   = st.iter, bucket = it < 10 ? 3 : it < 50 ? 4 : it < 200 ? 5 : 6;
      atomicAdd(a.fstats + FB_CHOOSE_BY + kind, 1ull);
      atomicAdd(a.fstats + FB_CHOOSE_BY + bucket, 1ull);
      if (s_reset) {
        atomicAdd(a.fstats + FB_SE_RESETS, 1ull);
        atomicAdd(a.fstats + FB_RESET_BY + kind, 1ull);
        atomicAdd(a.fstats + FB_RESET_BY + bucket, 1ull);
      }
    }
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
  const bool pre_dz = c_choose_old == 0;
  if (pre_dz) {
    // dz first, so the compaction scans below do not wait on column dots: columns up to kLongRow
    // nonzeros one thread each, longer ones queued (in cj) for a warp each.
    __shared__ int s_nq;
    if (tid == 0) { s_nq = 0; }
    __syncthreads();
    for (i_t k = tid; k < nn; k += kFissionThreads) {
      const i_t j = a.nonbasic_list[k];
      const i_t b = Acs[j], e = Acs[j + 1];
      if (e - b > kLongRow) {
        cj[atomicAdd(&s_nq, 1)] = k;
        continue;
      }
      f_t dot = 0.0;
      for (i_t p = b; p < e; ++p) { dot += Ax[p] * dyd[Ai[p]]; }
      dz[j] = -dot;
    }
    __syncthreads();
    const int nq = s_nq, lane = tid & 31, warp = tid >> 5;
    for (int q = warp; q < nq; q += kFissionThreads / 32) {
      const i_t j = a.nonbasic_list[cj[q]];
      f_t dot     = 0.0;
      for (i_t p = Acs[j] + lane; p < Acs[j + 1]; p += 32) { dot += Ax[p] * dyd[Ai[p]]; }
      dot = warp_reduce_add(dot);
      if (lane == 0) { dz[j] = -dot; }
    }
    __syncthreads();
  }
  for (i_t t0 = 0; t0 < nn; t0 += kFissionThreads) {
    const i_t k  = t0 + tid;
    i_t j        = -1;
    f_t dzj      = 0.0;
    f_t ratio    = 0.0;
    int touched  = 0, eligible = 0;
    if (k < nn) {
      j = a.nonbasic_list[k];
      if (pre_dz) {
        dzj = dz[j];
      } else {
        f_t dot = 0.0;
        for (i_t p = Acs[j]; p < Acs[j + 1]; ++p) { dot += Ax[p] * dyd[Ai[p]]; }
        dzj   = -dot;
        dz[j] = dzj;
      }
      touched = dzj != 0.0 ? 1 : 0;
      const variable_status_t sj = a.vstatus[j];
      if (sj == variable_status_t::NONBASIC_LOWER && dzj < -ptol) {
        ratio = fmax((-dtol - z[j]) / dzj, f_t{0.0}), eligible = 1;
      } else if (sj == variable_status_t::NONBASIC_UPPER && dzj > ptol) {
        ratio = fmax((dtol - z[j]) / dzj, f_t{0.0}), eligible = 1;
      }
    }
    // One scan for both lists: touched in the low 16 bits, eligible in the high 16 (each <= 512).
    int pos = 0, agg = 0;
    fission_scan_t(scs.tmp).ExclusiveSum(touched | (eligible << 16), pos, agg);
    if (touched) {
      dzi[nidx + (pos & 0xffff)] = j;
      mark[j]                    = 1;
    }
    if (eligible) {
      const int q = ncand + (pos >> 16);
      cpos[q]     = k;
      crat[q]     = ratio;
      cj[q]       = j;
      caz[q]      = fabs(z[j]);
    }
    nidx += agg & 0xffff;
    ncand += agg >> 16;
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
    // Shared memory only: |z| of the current choice is carried in a register.
    f_t az_e = 0.0;
    for (int c = 0; c < ncand; ++c) {
      const f_t rc = crat[c];
      const f_t az = caz[c];
      if (rc < min_val || (rc < min_val + ztol && az > az_e)) {
        min_val = rc, step = rc, az_e = az, entering = cj[c], nonbasic_entering = cpos[c];
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
    if (tid == 0 && st.iter == 0) {  // DEBUG (CUOPT_EXP_XBCHECK): iteration-0 "infeasible" exits
      if (atomicAdd(&g_unb0[0], 1.0) == 0.0) {
        const i_t jl = a.basic_list[leaving];
        g_unb0[1] = blockIdx.x, g_unb0[2] = leaving, g_unb0[3] = direction, g_unb0[4] = a.x[jl];
        g_unb0[5] = a.lower[jl], g_unb0[6] = a.upper[jl], g_unb0[7] = ncand, g_unb0[8] = nidx;
        g_unb0[9] = nzdy;
        g_unb0[10] = st.f_lean, g_unb0[11] = a.command, g_unb0[12] = a.refactor_resume;
        g_unb0[13] = st.squared_infeas[jl], g_unb0[14] = st.infeas_idx.size(), g_unb0[15] = a.transpose_on_device;
      }
    }
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
template <int T>
__device__ __forceinline__ void fission_primal_sync() { asm volatile("bar.sync 1, %0;" ::"n"(T - 32) : "memory"); }

template <int T>
__device__ __forceinline__ void fission_update_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r,
                                                          int* s_j = nullptr);
// Dynamic shared memory: m + 1 ints, the basic variable at each position of scaled_dxB.
__global__ void __launch_bounds__(kUpdThreads) fission_update_block(kernel_args* args, i_t n_problems)
{
  extern __shared__ f_t fsh[];
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
  fission_update_block_body<kUpdThreads>(args, n_problems, r,
                                         c_update_old != 0 ? nullptr : reinterpret_cast<int*>(fsh));
  __syncthreads();
  if (tr && threadIdx.x == 0) {
    r.t1 = fission_gtime();
    g_utrace[atomicAdd(&g_utrace_n, 1ull) & (kUtraceCap - 1)] = r;
  }
}

template <int T>
__device__ __forceinline__ void fission_update_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r,
                                                          int* s_j)
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
  __shared__ typename cub::BlockScan<int, T, cub::BLOCK_SCAN_WARP_SCANS>::TempStorage scan_tmp;
  __shared__ f_t s_part[T / 32];
  __shared__ int s_cut, s_fu;
  __shared__ i_t s_priced;
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
  __shared__ typename cub::BlockReduce<f_t, T>::TempStorage otmp;
  if (st.f_pivot_found == 0) {
    if (st.delta_z[entering] == 0.0) {
      const f_t obj = fission_objective<T>(lp, x, otmp);
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
  if (tid == 0) { s_cut = 0, s_fu = 0, s_priced = -1; }
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
    f_t part                     = 0.0;
    if (s_j != nullptr) {
      // One pass: each position's basic variable is looked up once and kept in s_j for the later
      // passes; the leaving row takes delta_x_leaving exactly, as the separate passes did.
      for (i_t k = t; k < nzx; k += (T - 32)) {
        const i_t ik  = sdx.i[k];
        const i_t j   = a.basic_list[ik];
        s_j[k]        = j;
        const f_t dxj = ik == basic_leaving ? delta_x_leaving : primal_step_length * sdx.x[k];
        delta_x[j]    = dxj;
        x[j] += dxj;
        part += dxj * lp.objective[j];
      }
      if (t == 0) {
        delta_x[leaving]  = delta_x_leaving;
        delta_x[entering] = primal_step_length;
        x[entering] += primal_step_length;
      }
      FSTEP_AT(S_UP_DX, 32)
    } else {
      for (i_t k = t; k < nzx; k += (T - 32)) { delta_x[a.basic_list[sdx.i[k]]] = primal_step_length * sdx.x[k]; }
      fission_primal_sync<T>();
      if (t == 0) {
        delta_x[leaving]  = delta_x_leaving;
        delta_x[entering] = primal_step_length;
      }
      fission_primal_sync<T>();
      FSTEP_AT(S_UP_DX, 32)
      for (i_t k = t; k < nzx; k += (T - 32)) {
        const i_t j = a.basic_list[sdx.i[k]];
        x[j] += delta_x[j];
        part += delta_x[j] * lp.objective[j];
      }
      if (t == 0) { x[entering] += delta_x[entering]; }
    }
    FSTEP_AT(S_UP_PRIMAL, 32)
    part = warp_reduce_add(part);
    if ((t & 31) == 0) { s_part[t / 32] = part; }
    fission_primal_sync<T>();
    if (t == 0) {
      f_t total = 0.0;
      for (int w = 0; w < (T - 32) / 32; ++w) { total += s_part[w]; }
      st.obj += total;
      st.obj += delta_x[entering] * lp.objective[entering];
    }
    fission_primal_sync<T>();
    FSTEP_AT(S_UP_OBJ, 32)
    if (isfinite(settings.cut_off) && st.obj >= settings.cut_off) {
      int bad = 0;
      for (i_t j = t; j < a.n; j += (T - 32)) {
        const variable_status_t sj = a.vstatus[j];
        const f_t zj               = a.z[j];
        if (sj != variable_status_t::BASIC && !isfinite(zj)) { bad = 1; }
        else if (sj == variable_status_t::NONBASIC_LOWER && zj < -settings.dual_tol) { bad = 1; }
        else if (sj == variable_status_t::NONBASIC_UPPER && zj > settings.dual_tol) { bad = 1; }
        else if (sj == variable_status_t::NONBASIC_FREE && fabs(zj) > settings.dual_tol) { bad = 1; }
      }
      if (t == 0) { s_cut = 1; }
      fission_primal_sync<T>();
      if (bad) { s_cut = 0; }
      fission_primal_sync<T>();
    }
    FSTEP_AT(S_UP_CUTOFF, 32)
    if (s_cut == 0) {
      // update_steepest_edge_norms; the leaving coefficient of scaled_dxB is `scale`.
      f_t* v = st.v.data();
      const bool v_dense = ft.dense_scratch2_.size() == static_cast<size_t>(st.delta_y_sparse.n);
      if (v_dense) {
        v = ft.dense_scratch2_.data();
      } else {
        for (i_t k = t; k < static_cast<i_t>(st.v_sparse.i.size()); k += (T - 32)) { v[st.v_sparse.i[k]] += st.v_sparse.x[k]; }
        fission_primal_sync<T>();
      }
      const f_t wr = -scale;
      if (wr != 0) {
        const f_t dyn    = st.dy_norm_sq;
        const f_t omegar = dyn / (wr * wr);
        f_t* dyse        = a.dy_se;
        for (i_t h = t; h < nzx; h += (T - 32)) {
          const i_t k = sdx.i[h];
          const i_t j = s_j != nullptr ? s_j[h] : a.basic_list[k];
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
        fission_primal_sync<T>();
        for (i_t k = t; k < static_cast<i_t>(st.v_sparse.i.size()); k += (T - 32)) { v[st.v_sparse.i[k]] = 0.0; }
      }
      fission_primal_sync<T>();
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
    const f_t obj = s_cut != 0 ? fission_objective<T>(lp, x, otmp) : f_t{0.0};
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
    // Row K of M was written by FTRAN from UTsol (the column 2K+1 just appended).
    const i_t K = ft.num_updates_ - 1;
    if (tid == 0) { st.f_mrows = (K >= 0 && st.f_mrows == K && st.f_mrow_k == K) ? K + 1 : -1; }
  }
  {
    // fission_infeas_update over the block, same list order.
    f_t* sq   = st.squared_infeas.data();
    i_t* idx  = st.infeas_idx.data();
    if (tid == 0) { sq[leaving] = 0.0; }
    __syncthreads();
    i_t count     = static_cast<i_t>(st.infeas_idx.size());
    const i_t nzx = static_cast<i_t>(sdx.i.size());
    for (i_t base = 0; base < nzx; base += T) {
      const i_t k = base + tid;
      int add     = 0;
      i_t j       = -1;
      if (k < nzx) {
        // After the swap: the leaving position now holds the entering variable.
        if (s_j != nullptr) {
          j = sdx.i[k] == basic_leaving ? entering : s_j[k];
        } else {
          j = a.basic_list[sdx.i[k]];
        }
        const f_t infeas = fmax(lp.lower[j] - x[j], x[j] - lp.upper[j]);
        if (infeas > settings.primal_tol) {
          add   = sq[j] == 0.0 ? 1 : 0;
          sq[j] = infeas * infeas;
        } else {
          sq[j] = 0.0;
        }
      }
      int pos = 0, agg = 0;
      cub::BlockScan<int, T, cub::BLOCK_SCAN_WARP_SCANS>(scan_tmp).ExclusiveSum(add, pos, agg);
      if (add) { idx[count + pos] = j; }
      count += agg;
      __syncthreads();
    }
    // The next pricing rides on this pass (CUOPT_EXP_UPDATE_PRICE=0: the price kernel does it): the
    // argmax of sq / dy_se under fission_price's total order, so the same leaving row is chosen.
    const bool pricing = c_update_price != 0;
    f_t bv = 0.0;
    i_t bj = -1;
    i_t out = 0;
    for (i_t base = 0; base < count; base += T) {
      const i_t k   = base + tid;
      const i_t j   = k < count ? idx[k] : -1;
      const int keep = (j >= 0 && sq[j] != 0.0) ? 1 : 0;
      if (keep && pricing) {
        const f_t val = sq[j] / a.dy_se[j];
        if (val > bv || (val == bv && j > bj)) { bv = val, bj = j; }
      }
      int pos = 0, agg = 0;
      cub::BlockScan<int, T, cub::BLOCK_SCAN_WARP_SCANS>(scan_tmp).ExclusiveSum(keep, pos, agg);
      __syncthreads();  // all reads of this stride before any write into it
      if (keep) { idx[out + pos] = j; }
      out += agg;
      __syncthreads();
    }
    if (tid == 0) { st.infeas_idx.set_size(out); }
    if (pricing) {
      __shared__ f_t s_bv[T / 32];
      __shared__ i_t s_bj[T / 32];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const f_t ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const i_t oj = __shfl_xor_sync(0xffffffffu, bj, off);
        if (ov > bv || (ov == bv && oj > bj)) { bv = ov, bj = oj; }
      }
      if ((tid & 31) == 0) { s_bv[tid >> 5] = bv, s_bj[tid >> 5] = bj; }
      __syncthreads();
      if (tid == 0) {
        for (int w = 1; w < T / 32; ++w) {
          if (s_bv[w] > bv || (s_bv[w] == bv && s_bj[w] > bj)) { bv = s_bv[w], bj = s_bj[w]; }
        }
        s_priced = bj;
        if (bj >= 0) {
          st.leaving       = bj;
          st.direction     = (lp.lower[bj] - x[bj] >= x[bj] - lp.upper[bj]) ? 1 : -1;
          st.basic_leaving = st.basic_mark[bj];
        }
      }
    }
  }
  FSTEP(S_U_INFEAS)
  {
    const i_t nz = st.delta_z_indices.size();
    for (i_t k = tid; k < nz; k += T) {
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
      st.f_state = s_priced >= 0 ? F_BTRAN : F_PRICE;
    }
  }
  FSTEP(S_U_TAIL)
}

// Fused rounds (CUOPT_EXP_FUSED): one block per seat runs `rounds` rounds of price, BTRAN, choose,
// FTRAN and update with block barriers only, so seats do not wait for each other between phases.
__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_fused(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems, int rounds)
{
  extern __shared__ f_t fsh[];
  __shared__ fission_utrace_t r;
  if (blockIdx.x >= n_problems) { return; }
  for (int k = 0; k < rounds; ++k) {
    if (args[blockIdx.x].done != 0 || args[blockIdx.x].slot->f_state == F_DONE) { return; }
    if (threadIdx.x < 32) { fission_price_body(args, n_problems); }
    __syncthreads();
    fission_btran_atomic_body(args, inv, n_problems);
    __syncthreads();
    fission_choose_block_body(args, n_problems, r);
    __syncthreads();
    fission_ftran_atomic_body(args, inv, n_problems);
    __syncthreads();
    // FTRAN's shared memory is free here: the update's position cache (m + 1 ints) goes there.
    fission_update_block_body<kFissionThreads>(args, n_problems, r,
                                               c_update_old != 0 ? nullptr : reinterpret_cast<int*>(fsh));
    __syncthreads();
  }
}

__global__ void fission_count(const kernel_args* args, i_t n_problems, int* live)
{
  const i_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_problems) { return; }
  const kernel_args& a = args[s];
  if (a.done == 0 && a.slot != nullptr && a.slot->f_state != F_DONE) { atomicAdd(live, 1); }
}

// The rounds graph's WHILE condition: repeat while more than yield_at seats run.
__global__ void fission_set_cond(cudaGraphConditionalHandle cond, const int* live, unsigned long long* fstats,
                                 int yield_at)
{
  if (fstats != nullptr) { ++fstats[FB_GROUPS]; }
  cudaGraphSetConditional(cond, *live > yield_at ? 1u : 0u);
}

// CUOPT_EXP_FISSION_YIELD: the seats still running when the pass stops (at a pivot boundary: the
// last group ended with the update) exit as REFACTOR_NEEDED and resume in a later launch.
__global__ void fission_yield(kernel_args* args, i_t n_problems, unsigned long long* fstats, int reason)
{
  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0 || a.slot == nullptr || a.slot->f_state == F_DONE) { return; }
  auto& st = *a.slot;
  gpu_lp lp;
  gpu_settings settings;
  gpu_solution sol;
  fission_views(a, lp, settings, sol);
  if (threadIdx.x == 0) {
    if (fstats != nullptr) { atomicAdd(fstats + FB_YIELDED, 1ull); }
    st.f_yield_state = st.f_state;
  }
  __syncwarp(0xffffffffu);
  fission_exit(a, st, lp, sol, dual::status_t::REFACTOR_NEEDED, st.iter, true, reason);
}

// ---- Explicit inverses built on the device every relaunch (see benchmarks/.../inv_bench.cu).
// L0^{-1} P and U0^{-1} of every seat, column (CSC) and row (CSR) forms, from the staged factors.
// Entries live in shared pools per factor, not per-seat slices: the column solve takes scratch space
// from one cursor (gcur), finv_offsets turns per-seat counts (cur) into packed offsets (off). A pool
// that overflows sets err; the host grows it and reruns the build (grow on overflow).
struct finv_build_t {
  int N{0}, m{0}, warps{0}, grid{0};
  long long scap[2]{0, 0}, pcap[2]{0, 0};  // scratch / packed pool entries per factor
  int *cur[2], *cstart[2], *clen[2], *si[2], *cp[2], *ci[2], *rp[2], *rj[2];
  double *sx[2], *cv[2], *rv[2];
  int *tasks, *ntasks, *next, *err;
  int* gcur;            // [2] scratch cursors
  long long* off[2];    // N + 1 packed offsets per factor (off[f][N]: total)
  gpu_fission_inv* inv; // per-seat views, written by finv_offsets
  unsigned long long* fb;  // fission_fb_t counters, copied into every view
  double* ws_y;
  unsigned long long* ws_b;
  int* ws_l;
};

__device__ __forceinline__ void finv_factor(const kernel_args& a, int f, const i_t*& cp, const i_t*& ri, const f_t*& cx)
{
  cp = f == 0 ? a.L0_col_start : a.U0_col_start;
  ri = f == 0 ? a.L0_i : a.U0_i;
  cx = f == 0 ? a.L0_x : a.U0_x;
}

// Thread per (seat, factor, column): trivial columns written directly, the rest listed.
__global__ void finv_trivial(const kernel_args* args, finv_build_t b)
{
  const long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const int m = b.m;
  if (t >= 2LL * b.N * m) { return; }
  const int k = static_cast<int>(t % m), f = static_cast<int>((t / m) % 2), s = static_cast<int>(t / (2LL * m));
  const kernel_args& a = args[s];
  if (a.done != 0) { return; }
  const i_t *cp, *ri;
  const f_t* cx;
  finv_factor(a, f, cp, ri, cx);
  if (cp[k + 1] - cp[k] != 1) {
    b.tasks[atomicAdd(b.ntasks, 1)] = (s * 2 + f) * m + k;
    return;
  }
  // No scratch: finv_pack writes row k with 1 / diagonal (cstart = -(k + 1)).
  atomicAdd(&b.cur[f][s], 1);
  const size_t c = static_cast<size_t>(s) * m + (f == 0 ? a.row_perm[k] : k);
  b.cstart[f][c] = -(k + 1);
  b.clen[f][c]   = 1;
}

// Warp per non-trivial column: T y = e_k with rows finalized in index order (ascending for L0,
// descending for U0), discovered rows in a bitset; per-warp workspace left zero after each task.
// Dynamic shared memory: the warps' bitsets, 8 * W64 words (CUOPT_EXP_FINV_GBITS=1: global, as before).
__global__ void __launch_bounds__(256) finv_columns(const kernel_args* args, finv_build_t b)
{
  extern __shared__ unsigned long long fbits[];
  const int lane = threadIdx.x & 31;
  const int gw   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int m = b.m, W64 = (m + 63) / 64;
  double* y = b.ws_y + static_cast<size_t>(gw) * m;
  unsigned long long* bits = b.ws_b + static_cast<size_t>(gw) * W64;
  if (c_finv_gbits == 0) {
    bits = fbits + static_cast<size_t>(threadIdx.x >> 5) * W64;
    for (int w = lane; w < W64; w += 32) { bits[w] = 0ull; }
    __syncwarp();
  }
  int* lst = b.ws_l + static_cast<size_t>(gw) * m;
  const int nt = *b.ntasks;
  for (;;) {
    int t = 0;
    if (lane == 0) { t = atomicAdd(b.next, 1); }
    t = __shfl_sync(0xffffffffu, t, 0);
    if (t >= nt) { return; }
    const int code = b.tasks[t];
    const int k = code % m, f = (code / m) % 2, s = code / (2 * m);
    const bool asc = f == 0;
    const kernel_args& a = args[s];
    const i_t *cp, *ri;
    const f_t* cx;
    finv_factor(a, f, cp, ri, cx);
    if (lane == 0) {
      y[k] = 1.0;
      bits[k >> 6] |= 1ull << (k & 63);
    }
    __syncwarp();
    int cnt = 0, cur = k;
    for (;;) {
      int r = -1;
      const int w0 = cur >> 6;
      for (int base = 0; r < 0; base += 32) {
        const int w = asc ? w0 + base + lane : w0 - base - lane;
        unsigned long long word = 0;
        if (w >= 0 && w < W64) {
          word = bits[w];
          if (w == w0) {
            const int bb = cur & 63;
            word &= asc ? (~0ull << bb) : (bb == 63 ? ~0ull : ((2ull << bb) - 1));
          }
        }
        const unsigned hit = __ballot_sync(0xffffffffu, word != 0);
        if (hit != 0) {
          const int src = __ffs(hit) - 1;
          const unsigned long long ww = __shfl_sync(0xffffffffu, word, src);
          const int wi = asc ? w0 + base + src : w0 - base - src;
          r = wi * 64 + (asc ? __ffsll(static_cast<long long>(ww)) - 1 : 63 - __clzll(static_cast<long long>(ww)));
        } else if ((asc && w0 + base + 31 >= W64 - 1) || (!asc && w0 - base - 31 <= 0)) {
          break;
        }
      }
      if (r < 0) { break; }
      const double yr = y[r] / (asc ? cx[cp[r]] : cx[cp[r + 1] - 1]);
      __syncwarp();
      if (lane == 0) {
        y[r] = yr;
        lst[cnt] = r;
        bits[r >> 6] &= ~(1ull << (r & 63));
      }
      ++cnt;
      __syncwarp();
      const int p0 = asc ? cp[r] + 1 : cp[r];
      const int p1 = asc ? cp[r + 1] : cp[r + 1] - 1;
      for (int p = p0 + lane; p < p1; p += 32) {
        const int i = ri[p];
        y[i] -= cx[p] * yr;
        atomicOr(&bits[i >> 6], 1ull << (i & 63));
      }
      __syncwarp();
      cur = r;
    }
    int nz = 0;
    for (int q = lane; q < cnt; q += 32) { nz += y[lst[q]] != 0.0 ? 1 : 0; }
    for (int o = 16; o > 0; o >>= 1) { nz += __shfl_xor_sync(0xffffffffu, nz, o); }
    int pos = 0;
    if (lane == 0) {
      atomicAdd(&b.cur[f][s], nz);
      pos = atomicAdd(b.gcur + f, nz);
    }
    pos = __shfl_sync(0xffffffffu, pos, 0);
    const bool fit = static_cast<long long>(pos) + nz <= b.scap[f];
    if (!fit && lane == 0) { atomicExch(b.err, 1); }
    const size_t base = 0;
    int w = 0;
    for (int q0 = 0; q0 < cnt; q0 += 32) {
      const int q = q0 + lane;
      const int row = q < cnt ? lst[q] : 0;
      const double v = q < cnt ? y[row] : 0.0;
      const bool keep = q < cnt && v != 0.0;
      const unsigned bal = __ballot_sync(0xffffffffu, keep);
      if (keep && fit) {
        const int o = pos + w + __popc(bal & ((1u << lane) - 1u));
        b.si[f][base + o] = row;
        b.sx[f][base + o] = v;
      }
      w += __popc(bal);
      if (q < cnt) { y[row] = 0.0; }
    }
    __syncwarp();
    if (lane == 0) {
      const size_t c = static_cast<size_t>(s) * m + (f == 0 ? a.row_perm[k] : k);
      b.cstart[f][c] = pos;
      b.clen[f][c]   = nz;
    }
  }
}

// One block: per-seat packed offsets from the per-seat counts (seats in index order), the packed
// pools' fit (err), and every seat's inverse views.
constexpr int kFinvOff = 1024;
__global__ void __launch_bounds__(kFinvOff) finv_offsets(finv_build_t b)
{
  using scan_t = cub::BlockScan<long long, kFinvOff, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename scan_t::TempStorage tmp;
  const int N = b.N, m = b.m;
  for (int f = 0; f < 2; ++f) {
    long long carry = 0;
    for (int s0 = 0; s0 < N; s0 += kFinvOff) {
      const int s       = s0 + threadIdx.x;
      const long long v = s < N ? b.cur[f][s] : 0;
      long long o = 0, tot = 0;
      scan_t(tmp).ExclusiveSum(v, o, tot);
      if (s < N) { b.off[f][s] = carry + o; }
      carry += tot;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      b.off[f][N] = carry;
      if (carry > b.pcap[f]) { atomicExch(b.err, 1); }
    }
  }
  __syncthreads();
  for (int s = threadIdx.x; s < N; s += kFinvOff) {
    gpu_fission_inv v;
    const size_t pb = static_cast<size_t>(s) * (m + 1), e0 = b.off[0][s], e1 = b.off[1][s];
    v.Lc_p = b.cp[0] + pb, v.Lc_i = b.ci[0] + e0, v.Lc_x = b.cv[0] + e0;
    v.Lr_p = b.rp[0] + pb, v.Lr_i = b.rj[0] + e0, v.Lr_x = b.rv[0] + e0;
    v.Uc_p = b.cp[1] + pb, v.Uc_i = b.ci[1] + e1, v.Uc_x = b.cv[1] + e1;
    v.Ur_p = b.rp[1] + pb, v.Ur_i = b.rj[1] + e1, v.Ur_x = b.rv[1] + e1;
    v.fb     = b.fb;
    b.inv[s] = v;
  }
}

// Block per (seat, factor): CSC column pointers by scan, entries copied in.
constexpr int kFinvPack = 1024;
__global__ void __launch_bounds__(kFinvPack) finv_pack(const kernel_args* args, finv_build_t b)
{
  using scan_t = cub::BlockScan<int, kFinvPack, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename scan_t::TempStorage tmp;
  const int s = blockIdx.x / 2, f = blockIdx.x % 2, m = b.m;
  if (args[s].done != 0 || *b.err != 0) { return; }
  const int per = (m + kFinvPack - 1) / kFinvPack;
  const int c0 = threadIdx.x * per, c1 = min(m, c0 + per);
  const int* clen = b.clen[f] + static_cast<size_t>(s) * m;
  const int* cst  = b.cstart[f] + static_cast<size_t>(s) * m;
  int* cp         = b.cp[f] + static_cast<size_t>(s) * (m + 1);
  int sum = 0;
  for (int c = c0; c < c1; ++c) { sum += clen[c]; }
  int off = 0, tot = 0;
  scan_t(tmp).ExclusiveSum(sum, off, tot);
  for (int c = c0; c < c1; ++c) {
    cp[c] = off;
    off += clen[c];
  }
  if (threadIdx.x == 0) { cp[m] = tot; }
  __syncthreads();
  const size_t base = static_cast<size_t>(b.off[f][s]);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const i_t *fcp, *fri;
  const f_t* fcx;
  finv_factor(args[s], f, fcp, fri, fcx);
  for (int c = warp; c < m; c += kFinvPack / 32) {
    const int src = cst[c], dst = cp[c], len = clen[c];
    if (src < 0) {  // trivial column: row k, 1 / diagonal
      if (lane == 0) {
        const int k              = -src - 1;
        b.ci[f][base + dst]      = k;
        b.cv[f][base + dst]      = 1.0 / fcx[fcp[k]];
      }
      continue;
    }
    for (int q = lane; q < len; q += 32) {
      b.ci[f][base + dst + q] = b.si[f][static_cast<size_t>(src) + q];
      b.cv[f][base + dst + q] = b.sx[f][static_cast<size_t>(src) + q];
    }
  }
}

// Block per (seat, factor): CSR by count, scan and a scatter with shared atomic row cursors (order
// within a row varies run to run). The scatter: columns of up to 32 entries one thread each, longer
// ones queued for a warp each. Dynamic shared memory: 2m ints (row cursors, queue).
constexpr int kFinvTr = 256;
__global__ void __launch_bounds__(kFinvTr) finv_transpose(const kernel_args* args, finv_build_t b)
{
  extern __shared__ int rc[];
  using scan_t = cub::BlockScan<int, kFinvTr, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename scan_t::TempStorage tmp;
  __shared__ int nq;
  const int s = blockIdx.x / 2, f = blockIdx.x % 2, m = b.m;
  if (args[s].done != 0 || *b.err != 0) { return; }
  if (threadIdx.x == 0) { nq = 0; }
  const size_t base = static_cast<size_t>(b.off[f][s]);
  const int* cp     = b.cp[f] + static_cast<size_t>(s) * (m + 1);
  const int* ci     = b.ci[f] + base;
  const double* cv  = b.cv[f] + base;
  int* rp           = b.rp[f] + static_cast<size_t>(s) * (m + 1);
  for (int r = threadIdx.x; r < m; r += kFinvTr) { rc[r] = 0; }
  __syncthreads();
  const int nnz = cp[m];
  for (int q = threadIdx.x; q < nnz; q += kFinvTr) { atomicAdd(&rc[ci[q]], 1); }
  __syncthreads();
  const int per = (m + kFinvTr - 1) / kFinvTr;
  const int r0 = threadIdx.x * per, r1 = min(m, r0 + per);
  int sum = 0;
  for (int r = r0; r < r1; ++r) { sum += rc[r]; }
  int off = 0, tot = 0;
  scan_t(tmp).ExclusiveSum(sum, off, tot);
  __syncthreads();
  for (int r = r0; r < r1; ++r) {
    const int c = rc[r];
    rp[r] = off, rc[r] = off;
    off += c;
  }
  if (threadIdx.x == 0) { rp[m] = tot; }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int* lq = rc + m;
  for (int c = threadIdx.x; c < m; c += kFinvTr) {
    const int q0 = cp[c], q1 = cp[c + 1];
    if (q1 - q0 > 32) {
      lq[atomicAdd(&nq, 1)] = c;
      continue;
    }
    for (int q = q0; q < q1; ++q) {
      const int o = atomicAdd(&rc[ci[q]], 1);
      b.rj[f][base + o] = c;
      b.rv[f][base + o] = cv[q];
    }
  }
  __syncthreads();
  for (int k = warp; k < nq; k += kFinvTr / 32) {
    const int c = lq[k];
    for (int q = cp[c] + lane; q < cp[c + 1]; q += 32) {
      const int o = atomicAdd(&rc[ci[q]], 1);
      b.rj[f][base + o] = c;
      b.rv[f][base + o] = cv[q];
    }
  }
}


inline bool gpu_fission()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FISSION");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}

inline int gpu_fission_rounds()
{
  static const int k = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FISSION_ROUNDS");
    return e != nullptr ? std::max(1, std::atoi(e)) : 8;
  }();
  return k;
}

// CUOPT_EXP_FISSION_YIELD=f: a pass stops once at most f * seats still run (0: run to the end).
inline double gpu_fission_yield()
{
  static const double f = [] {
    const char* e = std::getenv("CUOPT_EXP_FISSION_YIELD");
    return e != nullptr ? std::max(0.0, std::atof(e)) : 0.0;
  }();
  return f;
}

// Extra price/BTRAN/choose passes per round for seats whose steepest-edge weight was reset.
inline int gpu_fission_reset_passes()
{
  static const int k = [] {
    const char* e = std::getenv("CUOPT_EXP_RESET_PASSES");
    return e != nullptr ? std::max(0, std::atoi(e)) : 0;
  }();
  return k;
}
