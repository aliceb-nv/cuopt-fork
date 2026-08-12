/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// ===========================================================================
// gpu_simplified_dual_phase2 — single-thread GPU port of simplified_dual_phase2
//
// This is a faithful, line-for-line port of the CPU dual-simplex driver and its
// entire (pure-MPF-stacking) call tree. The algorithm is UNCHANGED; only the
// data structures differ:
//   * std::vector<T> created inside the kernel        -> gpu_vector<T> (owning,
//     device-side malloc/free with size/capacity tracking and geometric growth)
//   * data that exists before the kernel (LP, the LU  -> cuda::std::span views
//     factors, permutations, the solution/state)         (non-owning), uploaded
//                                                         host-side before launch
//   * std::sort                                        -> gpu_sort (insertion)
//   * NVTX ranges / settings.log logging               -> dropped
//   * CHECK_*/PRINT_* debug blocks                      -> not compiled
//
// Everything runs in a single GPU thread (<<<1,1>>>); no host<->device traffic
// happens inside the solve. See gpu_phase2/DIFF_REPORT.md for the per-function
// diff against the CPU source.
// ===========================================================================

#include "gpu_phase2.cuh"

#include "dense_triangle_solve.cuh"

#include <linear_algebra/sparse_matrix.hpp>
#include <linear_algebra/sparse_vector.hpp>
#include <math_optimization/tic_toc.hpp>

#include <utilities/macros.cuh>
#include <utilities/manual_cuda_graph.cuh>   // run_while: the device-controlled super-step loop

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/limits>
#include <cuda/std/span>

#include <new>  // placement new: construct the per-slot persistent state in host-allocated memory

#include <chrono>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {
// Shared simplex primitives live in mathematical_optimization::simplex; tic/toc
// live in the parent mathematical_optimization namespace.
using namespace cuopt::mathematical_optimization;
using namespace cuopt::mathematical_optimization::simplex;
namespace gpu_phase2_detail {

using i_t = int;
using f_t = double;
namespace dts = dense_triangle_solve;  // shared dense-solve cores

template <typename T>
using gpu_span = cuda::std::span<T>;

// Set if any device-side allocation fails; the driver bails with NUMERICAL.
// Shared across all blocks in a batch launch (a single failing block poisons the
// whole batch by design → surfaces as a loud verification mismatch). Reset from
// the host before each launch; set with atomicOr so concurrent blocks don't tear.
__device__ int g_gpu_alloc_failed;

// Steepest-edge drift reprice switch. **DEFAULT: reprice OFF** (g_serep_off = 1).
//   CUOPT_SEREP_ON=1  -> restore the reprice
//   CUOPT_SEREP_OFF   -> still accepted; it now only reaffirms the default (kept so existing
//                        harnesses and the B200 benchmark scripts keep working unchanged)
//
// WHY OFF BY DEFAULT (measured 2026-08-11, 100 v6 nodes, after the Direct conversion):
//   reprice ON   3.410 s / 3866 pivots      reprice OFF   2.863 s / 5148 pivots
// It fires 7285 times to save 1282 pivots — ~1.9 firings per pivot saved, i.e. it has ALWAYS been a
// near-break-even trade. A reprice pass is pricing + BTran ~= 16% of a full pivot, so 7285 of them
// cost roughly the 1282 pivots they buy back. Before the Direct conversion the trade landed just on
// the winning side (4.792 vs 5.515 s); afterwards it lands just on the losing side. Nothing about
// the reprice regressed — a coin-flip trade flipped.
//
// ⚠ IT IS A NUMERICAL SAFEGUARD, and off costs 33% MORE PIVOTS (5148 vs 3866) — i.e. measurably
// worse pivot choices. This default is a speed decision taken on ONE 100-node laptop benchmark; it
// is not evidence that the safeguard is unnecessary on harder or larger problems. Re-check it on the
// large-batch B200 run before treating it as settled.
//
// ⚠ CPU-PARITY CONSEQUENCE: this tree's CPU driver (phase2.cpp) does NOT honour CUOPT_SEREP_OFF at
// all, so the CPU always reprices. With the GPU now defaulting OFF, the single-thread GPU no longer
// matches the CPU bit-for-bit unless you pass CUOPT_SEREP_ON=1. Use CUOPT_SEREP_ON=1 for the
// 100/100 bit-exactness gate (dual_simplex/CLAUDE.md rules 2 and 8).
//
// It also remains the mechanism that lets the monolith run in the SAME reprice-free configuration as
// the fissioned pipeline, which structurally cannot express the `continue` back-edge:
//     GPU monolith (default, reprice off)  0-diff vs  GPU fission   [the Direct-conversion gate]
__device__ int g_serep_off;

// Force the SERIAL (walking-Harris-band) ratio test even on the warp path (host: CUOPT_RT_SERIAL).
// Default 0 = parallel warp_collective_rt (the MPF production choice). See phase2_ratio_test.
__device__ int g_rt_serial;

// Kernel fission: set by the host when ANY of the pivot kernels runs on a warp AND needs the
// elimination-level schedules (K1/K3/K4 — the solve kernels). Ksetup then builds warp_build_config
// into slot_state and wires ft.warp_cfg_. K2 does NOT need it (its collectives take explicit
// arguments), so a K2-only warp run leaves this 0 and skips the (per-node, O(m + nnz)) build.
__device__ int g_fission_warp;

// ===== Dev-loop build lever: SKIP THE MONOLITH WARP KERNEL ==================================
// `gpu_phase2_kernel<Warp=true>` is one giant fused kernel (whole driver inlined + the 10-branch
// warp_collective op-code dispatch, ~212 regs, 1632 B stack). ptxas cost is superlinear in function
// size, so it dominates this translation unit completely.
//
// MEASURED 2026-08-07 (`--ptxas-options=-v`, per-entry "Compile time", sm_89):
//     gpu_phase2_kernel<Warp=true>    131,449 ms      <-- 99.95% of the TU
//     gpu_phase2_kernel<Warp=false>        43 ms
//     gpu_fission_setup                    17 ms
//     the 11 gpu_fission_k* kernels     2-12 ms each  (~0.09 s for ALL of them)
//     TOTAL WALL for gpu_phase2.cu     131.51 s
// i.e. every other kernel in the file COMBINED is ~0.13 s of ptxas.
//
// END-TO-END effect of the flag, measured on the same command (not just the ptxas share, which would
// wrongly predict ~1000x — the front end does NOT disappear):
//     default                         134.64 s
//     -DCUOPT_SKIP_MONOLITH_WARP=1     21.20 s      -> 6.35x, ~113 s saved per rebuild
// The residual ~21 s is cicc/front-end + host compilation for everything else in the TU, which is
// the floor. (The monolith warp costs more than its 131 s of ptxas: cicc must also inline the whole
// fused driver to PTX first, so the flag saves front-end time too.)
//
// ⚠ WHAT IT DOES **NOT** DISABLE: the FISSION warp kernels (gpu_fission_k1..k4<true>) are launched
// outside these guards, so `CUOPT_GPU_FISSION=1 CUOPT_FISSION_K<N>_WARP=1` still works fully. Only
// the MONOLITH warp path becomes unavailable; CUOPT_GPU_WARP on the monolith then falls back to the
// single-thread kernel with a one-time warning. So this is the right flag for any fission-side dev
// loop, and must be OFF for the monolith-vs-fission warp comparisons (sweep legs 2 and 4) and for
// any production build that ships CUOPT_GPU_WARP.
//
// Set it from the build, no source edit needed:   -DCUOPT_SKIP_MONOLITH_WARP=1
#ifndef CUOPT_SKIP_MONOLITH_WARP
#define CUOPT_SKIP_MONOLITH_WARP 0
#endif
// Backward-compatible alias for the old (misleadingly named) switch.
#ifdef CUOPT_GPU_SINGLE_THREAD_ONLY
#undef CUOPT_SKIP_MONOLITH_WARP
#define CUOPT_SKIP_MONOLITH_WARP 1
#endif




// Occupancy target for the Direct warp monolith.
//   0  (default) -> NO __launch_bounds__; ptxas picks 182 regs / 11 blocks/SM
//   16           -> __launch_bounds__(32,16) -> 128 regs / 16 blocks/SM, 8 B spill
// MEASURED 2026-08-11 on RTX 3500 Ada (40 SMs), 100-node batch: capping is 2-3% SLOWER. That is not
// evidence against the cap — it is evidence the benchmark cannot test it. 40 SMs x 11 blocks/SM =
// 440 concurrent block slots vs a 100-BLOCK launch: occupancy never binds, so the cap buys nothing
// and the spill costs a little. It must be A/B'd on the large-batch B200 run (20k nodes >> slots),
// where higher occupancy is exactly the lever. Switch with -DCUOPT_MONO_DIRECT_MINBLOCKS=16.
#ifndef CUOPT_MONO_DIRECT_MINBLOCKS
#define CUOPT_MONO_DIRECT_MINBLOCKS 0
#endif

// ===== __launch_bounds__: only on the Direct warp monolith (see MinBlocks note at the kernel) ====================================
// Removed 2026-08-11. Challenge #13 measured that `__launch_bounds__(maxThreadsPerBlock=1, N)`
// COSTS ~40 registers per kernel: it tells ptxas there is no occupancy pressure, so it spends
// registers freely toward the 255 cap. The fission kernels never carried it for that reason; the
// monolith did, and now does not — ptxas's own heuristic is used everywhere.
//
// Consequence: `MinBlocks` is now an inert template parameter, and `CUOPT_GPU_REGCAP` (which only
// ever steered it through the launch bound) no longer caps anything. Both are kept so the sweep
// scaffolding can be restored in one line if a cap is ever wanted again.

// ================= Per-function kernel timers (COMPILE-TIME GATED) ================================
// Accumulate clock64 ticks per driver function on lane 0 and print one [KTIMER ...] line per solve
// at kernel exit. Enable by uncommenting the #define below (or building with -DCUOPT_KTIMER);
// when undefined every KT_* macro expands to ((void)0) → ZERO overhead → production default.
// Single-problem launches only (the accumulators are __device__ globals; a multi-block batch would
// race them, so the emit is gated on gridDim.x==1). Pair with the harness's [SENODE <name>] line
// (TEST_NETLIB_SE_STATS=1) to group per node. "setup" = kernel entry → first pivot (arg unpack +
// warp_build_config + the driver's one-time init: bounds/primal-vars/SE-norm-init/Arow/…); "pinf"
// counts only the per-pivot recompute (the one-time setup call is inside "setup").
// #define CUOPT_KTIMER 1   <- uncomment (or build with -DCUOPT_KTIMER) to re-enable the profilers
// below: [KTIMER]/[KTIMER2] per-function ticks + per-branch call counts, [STAT] per-stage histograms
// (ST_TIC/ST_TOC), [ESTSTAT]/[R2HIST] routing-estimate distributions, [QPFILL] Q/P fill.
// Left OFF for production: every KT_*/ST_*/ES_* macro expands to ((void)0), so the overhead is zero.
// These profilers cost ~4% and, being compiled ON by default, silently inflated EVERY timing in the
// 2026-08 campaign. The host-side [GPUKERNEL] line is not gated by this and always prints; it carries
// prof= so no timing can be quoted from a profiling build without the log saying so.
#ifdef CUOPT_KTIMER
#define CUOPT_PROF_BUILD 1
#else
#define CUOPT_PROF_BUILD 0
#endif

#ifdef CUOPT_KTIMER
enum kt_slot {
  KT_SETUP = 0, KT_PRICING, KT_BTRAN, KT_DZ, KT_RCU, KT_RT, KT_UPDDUAL, KT_FTRAN, KT_SE, KT_SE_SOLVE,
  KT_UPDPRIM, KT_OBJ, KT_AROW, KT_FTUPD, KT_PINF, KT_CLEARDZ,
  KT_N
};
// KT_SE_SOLVE is a SUB-slot of KT_SE (the ft.b_solve inside update_steepest_edge_norms); se_rest =
// se - se_solve. It is NOT part of the top-level partition (would double-count KT_SE).
__device__ unsigned long long g_kt[KT_N];
__device__ unsigned long long g_kt_calls[KT_N];
__device__ long long          g_kt_t0;         // kernel-entry clock (origin for "setup" and "total")
__device__ unsigned long long g_kt_sdxb_sum;   // Σ scaled_delta_xB nnz over pivots (density confirm)
__device__ unsigned long long g_kt_sdxb_cnt;   // pivot count for the above
// Per-CALL statistics for 4 inverse-add op groups (cycle-count distribution, exact cnt/sum/min/max +
// 24-bucket log2 histogram for the median). Groups: 0 hypersparse-SpMV (spa_.multiply, hybrid first solve),
// 1 SpMV (dense gather), 2 hypersparse-DMV (FIRSTMULT, P^T/Q^T rhs first call), 3 DMV (AXPY correction).
// Symmetric per-call stat groups for the LU(MPF)-vs-SpMV(inverse) primitive comparison. Both drivers do
// two factor-applies per direction; each apply picks a HYPersparse or DENse path. Naming:
//   <driver>_<direction>_<factor>_<path> ; B = b_solve, BT = b_transpose_solve.
// Plus the product-form "update application" each driver pays per solve: MPF = T-loop, inverse = DMV.
enum stat_grp {
  // ---- MPF (LU) driver ----
  SG_LU_B_L_HYP = 0, SG_LU_B_L_DEN,     // b_solve      L  triangular solve
  SG_LU_B_U_HYP, SG_LU_B_U_DEN,     // b_solve      U  triangular solve
  SG_LU_BT_U_HYP, SG_LU_BT_U_DEN,   // b_transpose  U^T triangular solve
  SG_LU_BT_L_HYP, SG_LU_BT_L_DEN,   // b_transpose  L^T triangular solve
  SG_LU_TLOOP,        // MPF product-form apply: the T-loop (nested INSIDE the L / L^T solve groups)
  SG_N
};
__device__ unsigned long long g_st_cnt[SG_N], g_st_sum[SG_N], g_st_min[SG_N], g_st_max[SG_N];
__device__ unsigned long long g_st_hist[SG_N][24];
#define ST_TIC(v)  long long v = clock64()
#define ST_TOC(g, v) do { if (threadIdx.x == 0) { const unsigned long long _d = (unsigned long long)(clock64() - (v)); \
    g_st_cnt[g]++; g_st_sum[g] += _d; if (g_st_min[g] == 0ULL || _d < g_st_min[g]) g_st_min[g] = _d; \
    if (_d > g_st_max[g]) g_st_max[g] = _d; int _b = 0; unsigned long long _t = _d; \
    while (_t > 1ULL && _b < 23) { _t >>= 1; ++_b; } g_st_hist[g][_b]++; } } while (0)
#define ST_EMIT() do { if (threadIdx.x == 0 && gridDim.x == 1) { for (int _g = 0; _g < SG_N; ++_g) { \
    printf("[STAT] g=%d cnt=%llu sum=%llu min=%llu max=%llu h=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n", \
      _g, g_st_cnt[_g], g_st_sum[_g], g_st_min[_g], g_st_max[_g], \
      g_st_hist[_g][0],g_st_hist[_g][1],g_st_hist[_g][2],g_st_hist[_g][3],g_st_hist[_g][4],g_st_hist[_g][5], \
      g_st_hist[_g][6],g_st_hist[_g][7],g_st_hist[_g][8],g_st_hist[_g][9],g_st_hist[_g][10],g_st_hist[_g][11], \
      g_st_hist[_g][12],g_st_hist[_g][13],g_st_hist[_g][14],g_st_hist[_g][15],g_st_hist[_g][16],g_st_hist[_g][17], \
      g_st_hist[_g][18],g_st_hist[_g][19],g_st_hist[_g][20],g_st_hist[_g][21],g_st_hist[_g][22],g_st_hist[_g][23]); } } } while (0)
#define KT_RESET()      do { if (threadIdx.x == 0) { for (int _i = 0; _i < KT_N; ++_i) { g_kt[_i] = 0; g_kt_calls[_i] = 0; } g_kt_t0 = clock64(); g_kt_sdxb_sum = 0; g_kt_sdxb_cnt = 0; \
    for (int _g = 0; _g < SG_N; ++_g) { g_st_cnt[_g] = 0; g_st_sum[_g] = 0; g_st_min[_g] = 0; g_st_max[_g] = 0; for (int _b = 0; _b < 24; ++_b) g_st_hist[_g][_b] = 0; } \
    } } while (0)
#define KT_TIC(v)       long long v = clock64()
#define KT_TOC(slot, v) do { if (threadIdx.x == 0) { g_kt[slot] += (unsigned long long)(clock64() - (v)); g_kt_calls[slot]++; } } while (0)
#define KT_BUMP(slot)   do { if (threadIdx.x == 0) { g_kt_calls[slot]++; } } while (0)
#define KT_SETUP_DONE() do { if (threadIdx.x == 0) { g_kt[KT_SETUP] += (unsigned long long)(clock64() - g_kt_t0); g_kt_calls[KT_SETUP]++; } } while (0)
#define KT_SDXB(nz)     do { if (threadIdx.x == 0) { g_kt_sdxb_sum += (unsigned long long)(nz); g_kt_sdxb_cnt++; } } while (0)
#define KT_EMIT(m)      do { if (threadIdx.x == 0 && gridDim.x == 1) { \
    const unsigned long long _tot = (unsigned long long)(clock64() - g_kt_t0); \
    printf("[KTIMER] total=%llu setup=%llu pricing=%llu btran=%llu dz=%llu rcu=%llu rt=%llu " \
           "upddual=%llu ftran=%llu se=%llu se_solve=%llu updprim=%llu obj=%llu arow=%llu ftupd=%llu " \
           "pinf=%llu cleardz=%llu sdxb_sum=%llu sdxb_cnt=%llu m=%d\n", _tot, g_kt[KT_SETUP], \
           g_kt[KT_PRICING], g_kt[KT_BTRAN], g_kt[KT_DZ], g_kt[KT_RCU], g_kt[KT_RT], g_kt[KT_UPDDUAL], \
           g_kt[KT_FTRAN], g_kt[KT_SE], g_kt[KT_SE_SOLVE], g_kt[KT_UPDPRIM], g_kt[KT_OBJ], g_kt[KT_AROW], \
           g_kt[KT_FTUPD], g_kt[KT_PINF], g_kt[KT_CLEARDZ], \
           g_kt_sdxb_sum, g_kt_sdxb_cnt, \
           static_cast<int>(m)); } } while (0)
#else
#define KT_RESET()      ((void)0)
#define KT_TIC(v)       ((void)0)
#define KT_TOC(slot, v) ((void)0)
#define KT_BUMP(slot)   ((void)0)
#define ST_TIC(v)       ((void)0)
#define ST_TOC(g, v)    ((void)0)
#define ST_EMIT()       ((void)0)
#define KT_SETUP_DONE() ((void)0)
#define KT_SDXB(nz)     ((void)0)
#define KT_EMIT(m)      ((void)0)
#endif

__device__ __forceinline__ f_t dev_inf() { return cuda::std::numeric_limits<f_t>::infinity(); }
__device__ __forceinline__ f_t dev_nan() { return cuda::std::numeric_limits<f_t>::quiet_NaN(); }

// Device wall-clock in nanoseconds (PTX %globaltimer). Used for the loop's
// time-limit guard (the CPU uses host toc(), which is unavailable on device).
__device__ __forceinline__ unsigned long long dev_globaltimer()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---------------------------------------------------------------------------
// gpu_vector<T> : owning device container, mirrors the std::vector ops used by
// the CPU code. Allocates with device malloc(), frees with free(). Used only
// for data created dynamically inside the kernel.
// ---------------------------------------------------------------------------
template <typename T>
struct gpu_vector {
  T* data_       = nullptr;
  size_t size_   = 0;
  size_t cap_    = 0;
  bool owns_     = true;   // false => non-owning view over external device memory (never freed here)

  // Adopt external device memory as a fixed-size, mutable, non-owning view (zero-copy). The caller
  // owns/frees the buffer; resize/reserve must not be called on a view.
  __device__ void view(T* ptr, size_t n)
  {
    if (owns_ && data_ != nullptr) { free(data_); }
    data_ = ptr; size_ = n; cap_ = n; owns_ = false;
  }

  __device__ gpu_vector() {}
  __device__ explicit gpu_vector(size_t n) { resize(n); }
  __device__ gpu_vector(size_t n, T val) { resize(n, val); }
  __device__ gpu_vector(const gpu_vector& o)
  {
    reserve(o.size_);
    for (size_t k = 0; k < o.size_; ++k) { data_[k] = o.data_[k]; }
    size_ = o.size_;
  }
  __device__ gpu_vector(gpu_vector&& o)
    : data_(o.data_), size_(o.size_), cap_(o.cap_), owns_(o.owns_)
  {
    o.data_ = nullptr;
    o.size_ = 0;
    o.cap_  = 0;
    o.owns_ = true;
  }
  __device__ gpu_vector& operator=(const gpu_vector& o)
  {
    if (this == &o) { return *this; }
    size_ = 0;
    reserve(o.size_);
    for (size_t k = 0; k < o.size_; ++k) { data_[k] = o.data_[k]; }
    size_ = o.size_;
    return *this;
  }
  __device__ gpu_vector& operator=(gpu_vector&& o)
  {
    if (this == &o) { return *this; }
    if (owns_ && data_ != nullptr) { free(data_); }
    data_   = o.data_;
    size_   = o.size_;
    cap_    = o.cap_;
    owns_   = o.owns_;
    o.data_ = nullptr;
    o.size_ = 0;
    o.cap_  = 0;
    o.owns_ = true;
    return *this;
  }
  __device__ ~gpu_vector()
  {
    if (owns_ && data_ != nullptr) { free(data_); }
  }

  __device__ size_t size() const { return size_; }
  __device__ bool empty() const { return size_ == 0; }
  __device__ T* data() { return data_; }
  __device__ const T* data() const { return data_; }
  __device__ T& operator[](size_t k) { return data_[k]; }
  __device__ const T& operator[](size_t k) const { return data_[k]; }
  __device__ T& back() { return data_[size_ - 1]; }
  __device__ T* begin() { return data_; }
  __device__ T* end() { return data_ + size_; }
  __device__ void clear() { size_ = 0; }
  // Set the logical size directly after a raw write through data() (warp collective ops that fill
  // [0, n) in-place). Caller must ensure capacity >= n (reserve first).
  __device__ void set_size(size_t n) { size_ = n; }
  // Swap buffers with another gpu_vector (preserves both allocations — used by sort() to
  // reuse scratch without move-stealing its buffer).
  __device__ void swap(gpu_vector& o)
  {
    T* d = data_; data_ = o.data_; o.data_ = d;
    size_t s = size_; size_ = o.size_; o.size_ = s;
    size_t c = cap_; cap_ = o.cap_; o.cap_ = c;
  }

  __device__ void grow_to(size_t newcap)
  {
    if (newcap <= cap_) { return; }
    T* p = static_cast<T*>(malloc(newcap * sizeof(T)));
    if (p == nullptr) {
      printf("[gpu_phase2] device malloc FAILED for %llu bytes (cap %llu -> %llu)\n",
             (unsigned long long)(newcap * sizeof(T)),
             (unsigned long long)cap_,
             (unsigned long long)newcap);
      atomicOr(&g_gpu_alloc_failed, 1);
      return;
    }
    for (size_t k = 0; k < size_; ++k) { p[k] = data_[k]; }
    if (data_ != nullptr) { free(data_); }
    data_ = p;
    cap_  = newcap;
  }
  __device__ void reserve(size_t n) { grow_to(n); }
  __device__ void resize(size_t n)
  {
    grow_to(n);
    if (g_gpu_alloc_failed) { return; }
    for (size_t k = size_; k < n; ++k) { data_[k] = T(); }
    size_ = n;
  }
  __device__ void resize(size_t n, T val)
  {
    grow_to(n);
    if (g_gpu_alloc_failed) { return; }
    for (size_t k = size_; k < n; ++k) { data_[k] = val; }
    size_ = n;
  }
  __device__ void assign(size_t n, T val)
  {
    size_ = 0;
    grow_to(n);
    if (g_gpu_alloc_failed) { return; }
    for (size_t k = 0; k < n; ++k) { data_[k] = val; }
    size_ = n;
  }
  __device__ void push_back(const T& v)
  {
    if (size_ == cap_) {
      grow_to(cap_ == 0 ? 4 : 2 * cap_);
      if (g_gpu_alloc_failed) { return; }
    }
    data_[size_++] = v;
  }
};

// Insertion sort over [first, last) with comparator. Replaces std::sort; the
// arrays sorted here (sparse-vector index lists) are short.
template <typename T, typename Cmp>
__device__ void gpu_sort(T* first, T* last, Cmp cmp)
{
  for (T* it = first + 1; it < last; ++it) {
    T key = *it;
    T* j  = it;
    while (j > first && cmp(key, *(j - 1))) {
      *j = *(j - 1);
      --j;
    }
    *j = key;
  }
}

// ---------------------------------------------------------------------------
// Non-owning sparse-matrix view (CSC). Backs the LP matrix and the fixed LU
// factors / their transposes. col_start is a MUTABLE span because reach()/DFS
// temporarily flip-and-restore the column pointers.
// ---------------------------------------------------------------------------
struct gpu_csc_view {
  i_t m{0}, n{0}, nz_max{0};
  gpu_span<i_t> col_start;
  gpu_span<i_t> i;
  gpu_span<f_t> x;
  __device__ i_t nnz() const { return col_start[n]; }
};

// Owning sparse-matrix (CSC). Used only for S_ (the MPF rank-1 update store),
// which grows one (u,v) column-pair per iteration.
struct gpu_csc_matrix {
  i_t m{0}, n{0}, nz_max{0};
  gpu_vector<i_t> col_start;
  gpu_vector<i_t> i;
  gpu_vector<f_t> x;

  __device__ gpu_csc_matrix() {}
  __device__ gpu_csc_matrix(const gpu_csc_matrix&)            = delete;
  __device__ gpu_csc_matrix& operator=(const gpu_csc_matrix&) = delete;
  __device__ ~gpu_csc_matrix()                                = default;

  // Initialize as an m x 0 matrix with col_start = {0}.
  __device__ void init_empty(i_t rows)
  {
    m      = rows;
    n      = 0;
    nz_max = 0;
    col_start.resize(1);
    col_start[0] = 0;
    i.clear();
    x.clear();
  }

  // Append a sparse column held as (nz, index-list, scatter-values). Mirrors
  // csc_matrix_t::append_column(i_t, i_t*, f_t*).
  __device__ void append_column(i_t x_nz, const i_t* idx, const f_t* xs)
  {
    i_t nz = col_start[n];
    for (i_t k = 0; k < x_nz; ++k) {
      const i_t i_val = idx[k];
      const f_t x_val = xs[i_val];
      if (x_val != 0.0) {
        i[nz] = i_val;
        x[nz] = x_val;
        nz++;
      }
    }
    col_start[n + 1] = nz;
    n++;
  }
};

// Owning sparse-matrix (CSR). Backs Arow — the row-wise copy of A used by the
// transpose delta_z path. j/x are swapped in place (nonbasic partition), so this is
// owning/mutable, not a span.
struct gpu_csr_matrix {
  i_t m{0}, n{0};
  gpu_vector<i_t> row_start;
  gpu_vector<i_t> j;
  gpu_vector<f_t> x;

  __device__ gpu_csr_matrix() {}
  __device__ gpu_csr_matrix(const gpu_csr_matrix&)            = delete;
  __device__ gpu_csr_matrix& operator=(const gpu_csr_matrix&) = delete;

  // Build the CSR copy of a CSC matrix. Mirrors csc_matrix_t::to_compressed_row
  // exactly (row entries end up ordered by ascending column index), so the delta_z
  // accumulation order matches the CPU bit-for-bit.
  __device__ void from_csc(const gpu_csc_view& A)
  {
    m            = A.m;
    n            = A.n;
    const i_t nz = A.col_start[n];
    row_start.resize(m + 1);
    j.resize(nz);
    x.resize(nz);
    gpu_vector<i_t> workspace(m, 0);
    for (i_t p = 0; p < nz; ++p) { workspace[A.i[p]]++; }
    i_t sum = 0;  // cumulative_sum: row_start = prefix sums, workspace reset to starts
    for (i_t i = 0; i < m; ++i) {
      row_start[i] = sum;
      sum += workspace[i];
      workspace[i] = row_start[i];
    }
    row_start[m] = sum;
    for (i_t col = 0; col < n; ++col) {
      const i_t cs = A.col_start[col];
      const i_t ce = A.col_start[col + 1];
      for (i_t p = cs; p < ce; ++p) {
        const i_t q = workspace[A.i[p]]++;
        j[q]        = col;
        x[q]        = A.x[p];
      }
    }
  }

  // Zero-copy: adopt a host-built CSR already in device memory as non-owning views. row_start stays
  // read-only; j/x are mutated in place by the nonbasic partition, so the passed buffer must be a
  // fresh per-solve mutable copy (true for the single-problem path — one upload per solve).
  __device__ void view_device(const i_t* rs, const i_t* jj, const f_t* xx, i_t m_, i_t n_, i_t nz)
  {
    m = m_;
    n = n_;
    row_start.view(const_cast<i_t*>(rs), m + 1);
    j.view(const_cast<i_t*>(jj), nz);
    x.view(const_cast<f_t*>(xx), nz);
  }
};

// Forward decl.
struct gpu_basis_update_mpf;

// ---------------------------------------------------------------------------
// gpu_sparse_vector : owning sparse vector (created/grown inside the kernel).
// Mirrors sparse_vector_t.
// ---------------------------------------------------------------------------
struct gpu_sparse_vector {
  i_t n{0};
  gpu_vector<i_t> i;
  gpu_vector<f_t> x;

  __device__ gpu_sparse_vector() {}
  __device__ gpu_sparse_vector(i_t n_, i_t nz) : n(n_), i(nz), x(nz) {}
  __device__ gpu_sparse_vector(const gpu_sparse_vector&)            = default;
  __device__ gpu_sparse_vector& operator=(const gpu_sparse_vector&) = default;
  __device__ gpu_sparse_vector(gpu_sparse_vector&&)                 = default;
  __device__ gpu_sparse_vector& operator=(gpu_sparse_vector&&)      = default;
  __device__ ~gpu_sparse_vector()                                   = default;

  __device__ void clear()
  {
    i.clear();
    x.clear();
  }

  __device__ void from_csc_column(const gpu_csc_view& A, i_t col)
  {
    const i_t col_start = A.col_start[col];
    const i_t col_end   = A.col_start[col + 1];
    n                   = A.m;
    const i_t nz        = col_end - col_start;
    i.clear();
    x.clear();
    i.reserve(nz);
    x.reserve(nz);
    for (i_t k = col_start; k < col_end; ++k) {
      i.push_back(A.i[k]);
      x.push_back(A.x[k]);
    }
  }

  __device__ void from_dense(const gpu_vector<f_t>& in)
  {
    i.clear();
    x.clear();
    n = in.size();
    i.reserve(n);
    x.reserve(n);
    for (i_t k = 0; k < n; ++k) {
      if (in[k] != 0) {
        i.push_back(k);
        x.push_back(in[k]);
      }
    }
  }

  __device__ void to_dense(gpu_vector<f_t>& x_dense) const
  {
    x_dense.clear();
    x_dense.resize(n, 0.0);
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) {
      x_dense[i[k]] = x[k];
    }
  }

  // Assumes x_dense already cleared / preinitialized.
  __device__ void scatter(f_t* x_dense) const
  {
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) {
      x_dense[i[k]] += x[k];
    }
  }

  __device__ void inverse_permute_vector(const i_t* p)
  {
    // In-place: each index is remapped independently (i[k] <- p[i[k]]), so no temp buffer is needed.
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) { i[k] = p[i[k]]; }
  }

  // Sort the sparse vector by index using caller-provided reused scratch (no per-call device malloc):
  // an O(n) bucket sort when dense (>30% nnz), else an n·log·n sort of a permutation then a gather.
  // The reused bucket is re-zeroed via assign (owning buffer, no realloc); the n·log·n path gathers
  // into scratch and swaps (preserving both buffers).
  __device__ void sort(gpu_vector<f_t>& bucket, gpu_vector<i_t>& i_scr,
                       gpu_vector<f_t>& x_scr, gpu_vector<i_t>& perm)
  {
    if (i.size() == 1) { return; }
    const i_t nz = i.size();
    if (i.size() > 0.3 * n) {
      bucket.assign(n, 0.0);  // reused (was gpu_vector<f_t> bucket(n,0.0)); O(n) re-zero, no malloc
      for (i_t k = 0; k < nz; ++k) { bucket[i[k]] = x[k]; }
      i.clear();
      i.reserve(nz);
      x.clear();
      x.reserve(nz);
      for (i_t k = 0; k < n; ++k) {
        if (bucket[k] != 0.0) {
          i.push_back(k);
          x.push_back(bucket[k]);
        }
      }
    } else {
      perm.clear();
      perm.reserve(nz);
      for (i_t k = 0; k < nz; ++k) { perm.push_back(k); }
      i_t* iunsorted = i.data();
      gpu_sort(perm.begin(), perm.end(), [iunsorted] __device__(i_t a, i_t b) {
        return iunsorted[a] < iunsorted[b];
      });
      i_scr.clear();
      i_scr.reserve(nz);
      x_scr.clear();
      x_scr.reserve(nz);
      for (i_t k = 0; k < nz; ++k) {
        i_scr.push_back(i[perm[k]]);
        x_scr.push_back(x[perm[k]]);
      }
      i.swap(i_scr);  // i/x get the sorted buffers; scratch keeps the old ones (capacity reused)
      x.swap(x_scr);
    }
  }

  __device__ f_t norm2_squared() const
  {
    f_t dot      = 0.0;
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) {
      dot += x[k] * x[k];
    }
    return dot;
  }

  __device__ void negate()
  {
    const i_t nz = x.size();
    for (i_t k = 0; k < nz; ++k) {
      x[k] *= -1.0;
    }
  }

  // ---- warp siblings for the fission `Direct` path ----------------------------------------------
  // On the fission path all 32 lanes are already in the kernel body, so these need no op-code and no
  // cfg_shared dispatch — just a strided loop. Both are element-wise maps over DISTINCT destinations
  // (a gpu_sparse_vector's indices are unique), so the strided form is bit-identical to the serial
  // one: no lane pair touches the same slot, and no arithmetic is reordered.
  // Direct=false forwards to the serial body — the monolith MUST take that arm, because lanes 1-31
  // Direct=false is now only the SINGLE-THREAD path (blockDim 1), where the strided form and its
  // __syncwarp would be wrong; the parked-helper warp model that also required it is gone.
  template <bool Direct>
  __device__ void warp_negate()
  {
    if constexpr (Direct) {
      const int lane = static_cast<int>(threadIdx.x);
      const i_t nz   = x.size();
      for (i_t k = lane; k < nz; k += 32) { x[k] *= -1.0; }
      __syncwarp(0xffffffffu);
    } else {
      negate();
    }
  }

  // scatter() accumulates (`+=`), but with distinct indices no two lanes hit the same x_dense slot,
  // so this needs no atomics. Callers must still honour the "x_dense already cleared" precondition.
  template <bool Direct>
  __device__ void warp_scatter(f_t* x_dense) const
  {
    if constexpr (Direct) {
      const int lane = static_cast<int>(threadIdx.x);
      const i_t nz   = i.size();
      for (i_t k = lane; k < nz; k += 32) { x_dense[i[k]] += x[k]; }
      __syncwarp(0xffffffffu);
    } else {
      scatter(x_dense);
    }
  }

  __device__ f_t find_coefficient(i_t index) const
  {
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) {
      if (i[k] == index) { return x[k]; }
    }
    return dev_nan();
  }
};

// ---------------------------------------------------------------------------
// Free helpers (vector_math.cpp / triangle_solve.cpp)
// ---------------------------------------------------------------------------

// Computes x = P*b, i.e. x[k] = b[p[k]].
__device__ i_t permute_vector(const i_t* p, const f_t* b, f_t* x, i_t n)
{
  for (i_t k = 0; k < n; ++k) {
    x[k] = b[p[k]];
  }
  return 0;
}

// Computes x = P'*b, i.e. x[p[k]] = b[k].
__device__ i_t inverse_permute_vector(const i_t* p, const f_t* b, f_t* x, i_t n)
{
  for (i_t k = 0; k < n; ++k) {
    x[p[k]] = b[k];
  }
  return 0;
}

// dot of two sorted sparse vectors held as (index,value) arrays.
__device__ f_t sparse_dot(const i_t* xind, const f_t* xval, i_t nx, const i_t* yind, const f_t* yval, i_t ny)
{
  f_t dot = 0.0;
  for (i_t a = 0, b = 0; a < nx && b < ny;) {
    const i_t p = xind[a];
    const i_t q = yind[b];
    if (p == q) {
      dot += xval[a] * yval[b];
      a++;
      b++;
    } else if (p < q) {
      a++;
    } else if (q < p) {
      b++;
    }
  }
  return dot;
}

// ---- dense triangular solves (triangle_solve.hpp) -------------------------

__device__ void lower_triangular_solve(const gpu_csc_view& L, f_t* x)
{
  dts::lower_solve(L.col_start.data(), L.i.data(), L.x.data(), x, L.n);
}

__device__ void lower_triangular_transpose_solve(const gpu_csc_view& L, f_t* x)
{
  dts::lower_transpose_solve(L.col_start.data(), L.i.data(), L.x.data(), x, L.n);
}

__device__ i_t upper_triangular_solve(const gpu_csc_view& U, f_t* x)
{
  dts::upper_solve(U.col_start.data(), U.i.data(), U.x.data(), x, U.n);
  return 0;
}

__device__ i_t upper_triangular_transpose_solve(const gpu_csc_view& U, f_t* x)
{
  dts::upper_transpose_solve(U.col_start.data(), U.i.data(), U.x.data(), x, U.n);
  return 0;
}

// ---------------------------------------------------------------------------
// Warp-parallel wavefront (level-scheduled) dense triangular solves.
//
// The fixed L0/U0 factors never change (no refactor), so a solve's dependency
// structure is computed ONCE (warp_build_config) into an elimination-level
// schedule: every column is bucketed into a level; all columns in one level are
// mutually independent and are processed concurrently -- ONE LANE PER COLUMN --
// with a __syncwarp() between levels. Each column's own accumulation runs
// serially in a single lane in the SAME order as the <<<1,1>>> solve, so the
// warp result is bit-for-bit equal to the single-thread reference.
//
// All four solves are expressed in race-free GATHER form  x[j]=(rhs[j]-sum)/diag,
// cross-wiring the non-transpose L/U solves to the transpose CSC so every output
// x[j] is written by exactly one lane (no atomics):
//   op          gather matrix     diag   inner-order   dep / level order
//   L-solve  -> L0_transpose_     last   ascending     deps<j, process asc
//   U-solve  -> U0_transpose_     first  descending    deps>j, process desc
//   Lᵀ-solve -> L0_              first  ascending     deps>j, process desc
//   Uᵀ-solve -> U0_              last   ascending     deps<j, process asc
// ---------------------------------------------------------------------------
// The four dense triangular-solve operators the warp path parallelizes. Their values
// are the indices into every per-operator array in warp_solve_config (and the order in
// which warp_build_config fills them). WARP_TLOOP is a non-solve collective (the MPF
// rank-1 replay) handled via the tl_* fields, so it never indexes a per-operator array.
// WARP_DONE is the "no work left" sentinel, placed one past every real op.
static constexpr int kNumWarpSolveOps = 4;  // L, U, Lᵀ, Uᵀ
// Per-operator gather flags for warp_solve_op {L, U, Lᵀ, Uᵀ} = {0, 1, 2, 3}. SINGLE SOURCE OF TRUTH:
// warp_build_config fills cfg.diag_first/descending/incremental from these, the big-path body reads
// them, and dense_triangle_solve.cuh's four named entry points hard-code the SAME triples. Changing
// one without the others silently changes which solve is computed.
//   df  {false, true,  true,  false}
//   dsc {false, true,  false, false}
//   inc {true,  true,  false, false}
// constexpr FUNCTIONS, not arrays: a namespace-scope constexpr array has no device storage, so
// device code cannot index it. These are usable from host and device and inside constant expressions.
__host__ __device__ constexpr bool op_diag_first(int op) { return op == 1 || op == 2; }
__host__ __device__ constexpr bool op_descending(int op) { return op == 1; }
__host__ __device__ constexpr bool op_incremental(int op) { return op == 0 || op == 1; }
enum warp_solve_op {
  WARP_L_SOLVE = 0, WARP_U_SOLVE = 1, WARP_LT_SOLVE = 2, WARP_UT_SOLVE = 3,
  WARP_TLOOP     = kNumWarpSolveOps,     // 4: MPF rank-1 T-loop (uses tl_* fields, not cs[op]/nlev[op])
  WARP_DZ        = kNumWarpSolveOps + 1, // 5: reduced-cost SpMv δz = -A_N'·δy (uses dz_* fields)
  WARP_TODENSE   = kNumWarpSolveOps + 2, // 6: sparse→dense scatter (td_* fields)
  WARP_FROMDENSE = kNumWarpSolveOps + 3, // 7: dense→sparse compaction (fd_* fields)
  WARP_RT        = kNumWarpSolveOps + 4, // 8: dual ratio test min-argmin (rt_* fields)
  WARP_SEUPD     = kNumWarpSolveOps + 5, // 9: SE-norm update loop over scaled_delta_xB (su_* fields)
  WARP_PINF      = kNumWarpSolveOps + 6, // 10: primal infeasibility scan over m (pi_* fields)
  WARP_UPDDUAL   = kNumWarpSolveOps + 7, // 11: update_dual_variables y/z update (ud_* fields)
  WARP_DZSCAT    = kNumWarpSolveOps + 8, // 12: scatter δz (CSR row-wise, inner span warp-parallel; dzs_*)
  WARP_SPTRSV    = kNumWarpSolveOps + 9, // 13: reach-restricted sparse gather solve (sp_* fields)
  WARP_NBE       = kNumWarpSolveOps + 10,// 14: compute_initial_nonbasic_end row partition (nbe_* fields)
  WARP_DONE      = kNumWarpSolveOps + 11 // 15: sentinel — never indexes a per-op array
};

// Trivially constructible (no member initializers) so it can be a __shared__
// variable. Lane 0 fills every field (warp_build_config + warp_dispatch) before
// any lane reads it, ordered by the first __syncwarp.
struct warp_solve_config {
  // ONLY the elimination-level SCHEDULE survives. Every per-dispatch REQUEST field (x/op and the
  // tl_/dz_/dzs_/td_/fd_/rt_/su_/pi_/ud_/nbe_/sp_op/sp_nlev groups, ~95 fields, ~700 of 928 bytes)
  // was staging for the parked-helper dispatch and was deleted with it on 2026-08-11: the Direct
  // path passes those values as call arguments to the warp_*_body collectives instead.
  // Also gone: diag_first/descending/incremental/m, which warp_build_config wrote but nothing ever
  // read — superseded by the compile-time op_diag_first/op_descending/op_incremental constexprs.
  // Built once (monolith: by the kernel prologue on lane 0; fission: by Ksetup), read-only after.
  const i_t* cs[kNumWarpSolveOps];      // gather CSC col_start
  const i_t* ridx[kNumWarpSolveOps];    // gather CSC row indices
  const f_t* val[kNumWarpSolveOps];     // gather CSC values
  const i_t* lstart[kNumWarpSolveOps];  // level offsets [nlev+1]
  const i_t* corder[kNumWarpSolveOps];  // columns by level [m]
  const i_t* level[kNumWarpSolveOps];   // per-column elimination level [m] (reach-restricted solve)
  i_t        nlev[kNumWarpSolveOps];
  // reach-restricted sparse gather solve: sptrsv_prologue buckets the reachable set into sp_ls/sp_co
  // using sp_pos as write-cursors, then the warp runs the wavefront over that sparse schedule.
  i_t* sp_ls;    // sparse level offsets [sp_nlev+1] (built per solve)
  i_t* sp_co;    // reachable columns bucketed by level [|R|] (built per solve)
  i_t* sp_pos;   // scratch write-cursors [sp_nlev+1] (built per solve)
};

// Collective (all-32-lane) dense solve of operator cfg.op on x = cfg.x. Both modes do exactly
// cfg.nlev[op] internal __syncwarps, so the persistent-helper dispatch count is mode-independent.

// ===== Parameterized collective BODIES ======================================================
// Each body takes ONLY what it needs as plain parameters (a_ prefix avoids collisions with locals).
// TWO call paths, ONE copy of the math:
//   * monolith : warp_collective_X(cfg) below is a thin unpacker -> calls the body.
//   * fission  : the warp K1..K5 kernels will call warp_X_body(...) DIRECTLY -- no cfg_shared, no
//                op-code dispatch, so ptxas sees ONE known body per call site instead of the
//                10-branch union that pins the warp monolith at 250 registers.
// __forceinline__ is REQUIRED: it collapses the wrapper's unpack-then-pass back to the original
// code so the monolith must not get slower (verified by A/B benchmark, not assumed).
// ===========================================================================================

// fwd decls: the bodies are hoisted above these helpers' definitions.
__device__ __forceinline__ f_t warp_reduce_add(f_t v);
__device__ __forceinline__ f_t warp_reduce_min(f_t v);

// COMPILE-TIME operator variant, for the fission `Direct` path where the op is a literal at the call
// site. Nothing about the solve is decided at runtime: no op code, no flag loads, no per-column
// branches — ptxas emits one specialized wavefront body per call site.
template <int Op>
__device__ __forceinline__ void warp_solve_body_op(const i_t* a_corder, const i_t* a_cs,
                                                   const i_t* a_lstart, i_t a_nlev,
                                                   const i_t* a_ridx, const f_t* a_val, f_t* a_x)
{
  dts::wavefront_solve_op<Op>(a_cs, a_ridx, a_val, a_x, a_lstart, a_corder, a_nlev);
}

__device__ __forceinline__ void warp_tloop_body(bool a_tl_asc, i_t a_tl_axpyoff, const i_t* a_tl_cs, i_t a_tl_dotoff, const f_t* a_tl_mu, i_t a_tl_num, const i_t* a_tl_ri, f_t a_tl_tol, const f_t* a_tl_val, f_t* a_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t num  = a_tl_num;
  f_t* x         = a_x;
  for (i_t s = 0; s < num; ++s) {
    const i_t k    = a_tl_asc ? s : (num - 1 - s);
    const i_t dcol = 2 * k + a_tl_dotoff;
    const i_t acol = 2 * k + a_tl_axpyoff;
    const f_t mu   = a_tl_mu[k];
    const i_t dc0  = a_tl_cs[dcol];
    const i_t dc1  = a_tl_cs[dcol + 1];
    f_t partial    = 0.0;
    for (i_t p = dc0 + lane; p < dc1; p += 32) { partial += a_tl_val[p] * x[a_tl_ri[p]]; }
    const f_t dot   = warp_reduce_add(partial);   // every lane gets the full sum
    const f_t theta = dot / mu;
    if (fabs(theta) > a_tl_tol) {
      const i_t ac0 = a_tl_cs[acol];
      const i_t ac1 = a_tl_cs[acol + 1];
      for (i_t p = ac0 + lane; p < ac1; p += 32) { x[a_tl_ri[p]] += (-theta) * a_tl_val[p]; }
    }
    __syncwarp(0xffffffffu);   // x writes visible before the next update's dot reads them
  }
}

__device__ __forceinline__ void warp_dz_body(const i_t* a_dz_Acs, const i_t* a_dz_Ari, const f_t* a_dz_Aval, const i_t* a_dz_basic, i_t* a_dz_count, i_t a_dz_dir, const f_t* a_dz_dy, i_t* a_dz_indices, i_t a_dz_leaving, i_t a_dz_m, i_t* a_dz_mark, i_t a_dz_n, const i_t* a_dz_nonbasic, f_t* a_dz_out)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t m = a_dz_m, n = a_dz_n;
  f_t* dz = a_dz_out;
  for (i_t k = lane; k < m; k += 32) { dz[a_dz_basic[k]] = 0.0; }  // clear basic entries
  __syncwarp(0xffffffffu);                                           // clears done before leaving set
  if (lane == 0) { dz[a_dz_leaving] = static_cast<f_t>(a_dz_dir); }
  const i_t num_nb = n - m;
  i_t count      = 0;    // running #appended (identical on all lanes)
  for (i_t base = 0; base < num_nb; base += 32) {
    const i_t k = base + lane;
    i_t j = -1; f_t dzj = 0.0; bool active = false;
    if (k < num_nb) {
      j            = a_dz_nonbasic[k];
      const i_t c0 = a_dz_Acs[j], c1 = a_dz_Acs[j + 1];
      f_t dot      = 0.0;
      for (i_t p = c0; p < c1; ++p) { dot += a_dz_Aval[p] * a_dz_dy[a_dz_Ari[p]]; }
      dzj = -dot; dz[j] = dzj;
      active = (dzj != 0.0);
    }
    const unsigned ball = __ballot_sync(0xffffffffu, active);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (active) { a_dz_indices[count + prefix] = j; a_dz_mark[j] = 1; }  // distinct offsets/rows
    count += __popc(ball);
  }
  if (lane == 0) { *a_dz_count = count; }
}

__device__ __forceinline__ void warp_dz_scatter_body(i_t* a_dz_count, i_t a_dz_dir, i_t* a_dz_indices, i_t a_dz_leaving, i_t* a_dz_mark, f_t* a_dz_out, const i_t* a_dzs_dyi, const f_t* a_dzs_dyx, const i_t* a_dzs_j, const i_t* a_dzs_nbe, i_t a_dzs_nzdy, const i_t* a_dzs_rs, const f_t* a_dzs_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t nz   = a_dzs_nzdy;
  f_t* dz        = a_dz_out;
  i_t count             = 0;   // running #appended (identical on all lanes)
  for (i_t kk = 0; kk < nz; ++kk) {
    const f_t dyi = a_dzs_dyx[kk];
    if (fabs(dyi) < 1e-12) { continue; }         // uniform skip (dyi identical on all lanes)
    const i_t i  = a_dzs_dyi[kk];
    const i_t rs = a_dzs_rs[i];
    const i_t re = a_dzs_nbe[i] + 1;
    for (i_t base = rs; base < re; base += 32) {
      const i_t p = base + lane;
      i_t j = -1; bool newcol = false;
      if (p < re) {
        j = a_dzs_j[p];
        dz[j] -= dyi * a_dzs_x[p];             // distinct j within the row → no collision
        if (!a_dz_mark[j]) { a_dz_mark[j] = 1; newcol = true; }
      }
      const unsigned ball = __ballot_sync(0xffffffffu, newcol);
      const int prefix    = __popc(ball & ((1u << lane) - 1u));
      if (newcol) { a_dz_indices[count + prefix] = j; }  // increasing k/lane == serial append order
      count += __popc(ball);
    }
  }
  if (lane == 0) {
    *a_dz_count      = count;
    dz[a_dz_leaving] = static_cast<f_t>(a_dz_dir);
  }
}

__device__ __forceinline__ void warp_todense_body(f_t* a_td_dense, const i_t* a_td_i, i_t a_td_n, i_t a_td_nz, const f_t* a_td_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t n = a_td_n;
  for (i_t k = lane; k < n; k += 32) { a_td_dense[k] = 0.0; }
  __syncwarp(0xffffffffu);
  const i_t nz = a_td_nz;
  for (i_t k = lane; k < nz; k += 32) { a_td_dense[a_td_i[k]] = a_td_x[k]; }
}

__device__ __forceinline__ void warp_fromdense_body(i_t* a_fd_count, i_t* a_fd_i, const f_t* a_fd_in, i_t a_fd_n, f_t* a_fd_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t n = a_fd_n;
  i_t count = 0;
  for (i_t base = 0; base < n; base += 32) {
    const i_t k = base + lane;
    bool active = false; f_t v = 0.0;
    if (k < n) { v = a_fd_in[k]; active = (v != 0.0); }
    const unsigned ball = __ballot_sync(0xffffffffu, active);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (active) { a_fd_i[count + prefix] = k; a_fd_x[count + prefix] = v; }
    count += __popc(ball);
  }
  if (lane == 0) { *a_fd_count = count; }
}

__device__ __forceinline__ void warp_rt_body(f_t a_rt_dual_tol, const f_t* a_rt_dz, i_t* a_rt_entering, i_t* a_rt_k, i_t a_rt_m, i_t a_rt_n, const i_t* a_rt_nonbasic, f_t a_rt_pivot_tol, f_t* a_rt_step, const variable_status_t* a_rt_vstatus, const f_t* a_rt_z, f_t a_rt_zero_tol)
{
  const int lane   = static_cast<int>(threadIdx.x);
  const i_t num_nb = a_rt_n - a_rt_m;
  const f_t ptol = a_rt_pivot_tol, dtol = a_rt_dual_tol, ztol = a_rt_zero_tol;
  const f_t INF  = dev_inf();

  // Pass 1: global minimum eligible ratio (all lanes get it).
  f_t lmin = INF;
  for (i_t k = lane; k < num_nb; k += 32) {
    const i_t j             = a_rt_nonbasic[k];
    const variable_status_t s = a_rt_vstatus[j];
    f_t ratio = INF;
    if (s == variable_status_t::NONBASIC_LOWER && a_rt_dz[j] < -ptol) {
      ratio = (-dtol - a_rt_z[j]) / a_rt_dz[j];
    } else if (s == variable_status_t::NONBASIC_UPPER && a_rt_dz[j] > ptol) {
      ratio = (dtol - a_rt_z[j]) / a_rt_dz[j];
    }
    if (ratio < lmin) { lmin = ratio; }
  }
  const f_t gmin = warp_reduce_min(lmin);

  // TRUE DUAL_UNBOUNDED: gmin == INF means NO column was eligible, and the serial scan returns -1
  // here. This early-out is REQUIRED because pass 2's candidate test `ratio <= gmin` degenerates to
  // `INF <= INF` — true for EVERY column — so without it the warp picks an arbitrary max-|z| column
  // and returns step_length = INF. The driver then does z[j] += INF * delta_z[j] and the solve
  // detonates: objective -> ~-3.7e15 and the node exits NUMERICAL instead of DUAL_UNBOUNDED.
  // (This is the bug that made csched008/node_d79_id150 report NUMERICAL on every recorded warp run
  // since 2026-08-04; it was invisible because the warp gate only counted bit-exactness.)
  // The `ratio <= gmin` form is still right for the FINITE case — see the note below.
  if (!isfinite(gmin)) {
    if (lane == 0) { *a_rt_entering = -1; *a_rt_k = -1; *a_rt_step = INF; }
    return;
  }

  // Pass 2: among ratio < gmin+ztol, pick max |z[j]| (tie → smaller k).
  f_t best_absz = -1.0; i_t best_k = 0x7fffffff; i_t best_j = -1; f_t best_r = INF;
  for (i_t k = lane; k < num_nb; k += 32) {
    const i_t j             = a_rt_nonbasic[k];
    const variable_status_t s = a_rt_vstatus[j];
    f_t ratio = INF;
    if (s == variable_status_t::NONBASIC_LOWER && a_rt_dz[j] < -ptol) {
      ratio = (-dtol - a_rt_z[j]) / a_rt_dz[j];
    } else if (s == variable_status_t::NONBASIC_UPPER && a_rt_dz[j] > ptol) {
      ratio = (dtol - a_rt_z[j]) / a_rt_dz[j];
    }
    // Candidate = the argmin (always, so entering can't be spuriously -1) plus near-ties within
    // zero_tol. Use the difference (ratio - gmin) not (ratio < gmin + ztol): the latter drops the
    // argmin when gmin is large (gmin+ztol rounds back to gmin) or zero_tol == 0 → false DUAL_UNBOUNDED.
    if (ratio <= gmin || ratio - gmin < ztol) {
      const f_t az = fabs(a_rt_z[j]);
      if (az > best_absz || (az == best_absz && k < best_k)) {
        best_absz = az; best_k = k; best_j = j; best_r = ratio;
      }
    }
  }
  // Reduce (best_absz, best_k, best_j, best_r) to lane 0: max |z|, tie → smaller k.
  for (int off = 16; off > 0; off >>= 1) {
    const f_t o_absz = __shfl_down_sync(0xffffffffu, best_absz, off);
    const i_t o_k    = __shfl_down_sync(0xffffffffu, best_k, off);
    const i_t o_j    = __shfl_down_sync(0xffffffffu, best_j, off);
    const f_t o_r    = __shfl_down_sync(0xffffffffu, best_r, off);
    if (o_absz > best_absz || (o_absz == best_absz && o_k < best_k)) {
      best_absz = o_absz; best_k = o_k; best_j = o_j; best_r = o_r;
    }
  }
  if (lane == 0) { *a_rt_entering = best_j; *a_rt_k = best_j < 0 ? -1 : best_k; *a_rt_step = best_r; }
}

__device__ __forceinline__ void warp_seupd_body(const i_t* a_su_basic, f_t a_su_dynormsq, f_t* a_su_dyse, i_t a_su_entering, i_t a_su_leaving, i_t a_su_nz, f_t a_su_omegar, const i_t* a_su_sdxb_i, const f_t* a_su_sdxb_x, const f_t* a_su_v, f_t a_su_wr)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t nz = a_su_nz;
  for (i_t h = lane; h < nz; h += 32) {
    const i_t k = a_su_sdxb_i[h];
    const i_t j = a_su_basic[k];
    if (k == a_su_leaving) {
      const f_t w = a_su_sdxb_x[h];
      a_su_dyse[a_su_entering] = (1.0 / (w * w)) * a_su_dynormsq;
    } else {
      const f_t wk      = -a_su_sdxb_x[h];
      const f_t new_val = a_su_dyse[j] + wk * (2.0 * a_su_v[k] / a_su_wr + wk * a_su_omegar);
      a_su_dyse[j]    = fmax(new_val, 1e-4);
    }
  }
}

__device__ __forceinline__ void warp_pinf_body(const i_t* a_pi_basic, i_t* a_pi_count, i_t* a_pi_idx, const f_t* a_pi_lower, i_t a_pi_m, i_t a_pi_n, f_t* a_pi_pinf, f_t* a_pi_pinf_sq, f_t* a_pi_sqinf, f_t a_pi_tol, const f_t* a_pi_upper, const f_t* a_pi_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t n = a_pi_n, m = a_pi_m;
  for (i_t k = lane; k < n; k += 32) { a_pi_sqinf[k] = 0.0; }
  __syncwarp(0xffffffffu);
  i_t count = 0; f_t lpinf = 0.0, lpinf_sq = 0.0;
  for (i_t base = 0; base < m; base += 32) {
    const i_t k = base + lane;
    i_t j = -1; bool inf = false;
    if (k < m) {
      j                = a_pi_basic[k];
      const f_t lo     = a_pi_lower[j] - a_pi_x[j];
      const f_t up     = a_pi_x[j] - a_pi_upper[j];
      const f_t infeas = fmax(lo, up);
      if (infeas > a_pi_tol) {
        const f_t sq    = infeas * infeas;
        a_pi_sqinf[j] = sq;   // distinct j → race-free
        inf = true; lpinf += infeas; lpinf_sq += sq;
      }
    }
    const unsigned ball = __ballot_sync(0xffffffffu, inf);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (inf) { a_pi_idx[count + prefix] = j; }  // ordered (increasing k) → matches serial
    count += __popc(ball);
  }
  const f_t pinf    = warp_reduce_add(lpinf);
  const f_t pinf_sq = warp_reduce_add(lpinf_sq);
  if (lane == 0) { *a_pi_count = count; *a_pi_pinf = pinf; *a_pi_pinf_sq = pinf_sq; }
}

__device__ __forceinline__ void warp_upddual_body(const i_t* a_ud_dy_i, i_t a_ud_dy_nz, const f_t* a_ud_dy_x, const f_t* a_ud_dz, const i_t* a_ud_dz_idx, i_t a_ud_dz_nz, i_t a_ud_leaving, f_t a_ud_step, f_t* a_ud_y, f_t* a_ud_z)
{
  const int lane = static_cast<int>(threadIdx.x);
  const f_t step = a_ud_step;
  for (i_t k = lane; k < a_ud_dy_nz; k += 32) {
    const i_t i = a_ud_dy_i[k];
    a_ud_y[i] += step * a_ud_dy_x[k];
  }
  for (i_t k = lane; k < a_ud_dz_nz; k += 32) {
    const i_t j = a_ud_dz_idx[k];
    a_ud_z[j] += step * a_ud_dz[j];
  }
  if (lane == 0) { a_ud_z[a_ud_leaving] += step * a_ud_dz[a_ud_leaving]; }
}

// COMPILE-TIME operator variant of the reach-restricted gather (fission `Direct` path).
template <int Op>
__device__ __forceinline__ void warp_sptrsv_body_op(const i_t* a_cs, const i_t* a_ridx,
                                                    const f_t* a_val, i_t* a_sp_co, i_t* a_sp_ls,
                                                    int a_sp_nlev, f_t* a_x)
{
  dts::wavefront_solve_op<Op>(a_cs, a_ridx, a_val, a_x, a_sp_ls, a_sp_co, a_sp_nlev);
}

__device__ __forceinline__ void warp_nbe_body(const i_t* a_nbe_basic_mark, i_t* a_nbe_end, i_t* a_nbe_j, i_t a_nbe_m, const i_t* a_nbe_rs, f_t* a_nbe_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t m    = a_nbe_m;
  for (i_t i = lane; i < m; i += 32) {
    i_t lo = a_nbe_rs[i];
    i_t hi = a_nbe_rs[i + 1] - 1;
    while (lo <= hi) {
      const i_t j = a_nbe_j[lo];
      if (a_nbe_basic_mark[j] >= 0) {
        const i_t tj = a_nbe_j[lo]; a_nbe_j[lo] = a_nbe_j[hi]; a_nbe_j[hi] = tj;
        const f_t tx = a_nbe_x[lo]; a_nbe_x[lo] = a_nbe_x[hi]; a_nbe_x[hi] = tx;
        hi--;
      } else {
        lo++;
      }
    }
    a_nbe_end[i] = hi;
  }
}

// All-lane butterfly sum: every lane returns the full 32-lane total, so all lanes then compute the
// same theta and take the same branch (branch-uniform => matched __syncwarp counts). NOT bit-exact vs
// the serial left-to-right sum (reordered adds) -> warp path only; the <<<1,1>>> ref never uses it.
__device__ __forceinline__ f_t warp_reduce_add(f_t v)
{
  for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
  return v;
}

// All-lane butterfly min: every lane returns the full 32-lane minimum (used by the ratio test to
// broadcast the global min ratio before the tie-break pass). min is associative/commutative => bit-exact.
__device__ __forceinline__ f_t warp_reduce_min(f_t v)
{
  for (int o = 16; o > 0; o >>= 1) { v = fmin(v, __shfl_xor_sync(0xffffffffu, v, o)); }
  return v;
}

// The parked-helper dispatch layer (warp_collective_* unpackers, warp_collective, warp_helper_loop,
// mono_solve_arm/mono_sptrsv_arm) was DELETED 2026-08-11 when the monolith moved to the Direct model.
// Nothing instantiates Warp=true,Direct=false any more, so every call site now reaches the
// warp_*_body collectives directly and the op-code union is gone. See the Direct banner below.

// Build the elimination-level schedule of one gather operator (lane 0, once).
// `mat` is the gather CSC (its columns are the solve's output rows). Fills
// level_start[nlev+1] and col_order[m]; returns nlev. `level` is m-sized scratch.
__device__ int warp_build_levels(const gpu_csc_view& mat, bool diag_first, bool asc_process,
                                  gpu_vector<i_t>& level, gpu_vector<i_t>& level_start,
                                  gpu_vector<i_t>& col_order)
{
  const i_t m = mat.n;
  level.assign(m, 0);
  for (i_t s = 0; s < m; ++s) {                        // pull: level[j] = 1 + max level[dep]
    const i_t j  = asc_process ? s : (m - 1 - s);
    const i_t c0 = mat.col_start[j];
    const i_t c1 = mat.col_start[j + 1];
    const i_t lo = diag_first ? c0 + 1 : c0;
    const i_t hi = diag_first ? c1 : (c1 - 1);
    i_t lv = 1;
    for (i_t p = lo; p < hi; ++p) {
      const i_t dep = mat.i[p];
      if (level[dep] + 1 > lv) { lv = level[dep] + 1; }
    }
    level[j] = lv;
  }
  i_t nlev = 0;
  for (i_t j = 0; j < m; ++j) { if (level[j] > nlev) { nlev = level[j]; } }
  level_start.assign(nlev + 1, 0);                     // counting sort by level
  for (i_t j = 0; j < m; ++j) { level_start[level[j]]++; }
  for (i_t k = 1; k <= nlev; ++k) { level_start[k] += level_start[k - 1]; }
  col_order.resize(m);
  gpu_vector<i_t> pos(nlev + 1);
  for (i_t lv = 1; lv <= nlev; ++lv) { pos[lv] = level_start[lv - 1]; }
  for (i_t j = 0; j < m; ++j) { const i_t lv = level[j]; col_order[pos[lv]++] = j; }
  return static_cast<int>(nlev);
}


// ---- sparse triangular solve (reach + iterative DFS) ----------------------
// G.col_start is flipped to mark visited nodes and restored on the way out.
#define GP_FLIP(i)       (-(i) - 2)
#define GP_UNFLIP(i)     (((i) < 0) ? GP_FLIP(i) : (i))
#define GP_MARKED(w, j)  ((w)[j] < 0)
#define GP_MARK(w, j)    { (w)[j] = GP_FLIP((w)[j]); }

// pinv == nullptr means no permutation.
__device__ i_t depth_first_search(
  i_t j, const i_t* pinv, const gpu_csc_view& G, i_t top, i_t* xi, i_t* pstack)
{
  i_t head = 0;
  xi[0]    = j;
  i_t done = 0;
  while (head >= 0) {
    j        = xi[head];
    i_t jnew = pinv ? pinv[j] : j;
    if (!GP_MARKED(G.col_start, j)) {
      GP_MARK(G.col_start, j)
      pstack[head] = (jnew < 0) ? 0 : GP_UNFLIP(G.col_start[jnew]);
    }
    done           = 1;
    i_t p2         = (jnew < 0) ? 0 : GP_UNFLIP(G.col_start[jnew + 1]);
    const i_t psav = pstack[head];
    i_t p;
    for (p = psav; p < p2; ++p) {
      i_t i = G.i[p];
      if (GP_MARKED(G.col_start, i)) { continue; }
      pstack[head] = p;
      xi[++head]   = i;
      done         = 0;
      break;
    }
    if (done) {
      pstack[head] = 0;
      xi[head]     = 0;
      head--;
      xi[--top] = j;
    }
  }
  return top;
}

__device__ i_t reach(
  const gpu_sparse_vector& b, const i_t* pinv, const gpu_csc_view& G, i_t* xi)
{
  const i_t m   = G.m;
  i_t top       = m;
  const i_t bnz = b.i.size();
  for (i_t p = 0; p < bnz; ++p) {
    if (!GP_MARKED(G.col_start, b.i[p])) {
      top = depth_first_search(b.i[p], pinv, G, top, xi, xi + m);
    }
  }
  for (i_t p = top; p < m; ++p) {
    GP_MARK(G.col_start, xi[p]);
  }
  return top;
}

// lo == true solves L*x=b, lo == false solves U*x=b. x is a dense scratch
// buffer; xi is a 2*m scratch. Output nonzero pattern is xi[top..m-1].
template <bool lo>
__device__ i_t sparse_triangle_solve(
  const gpu_sparse_vector& b, const i_t* pinv, i_t* xi, const gpu_csc_view& G, f_t* x)
{
  i_t m   = G.m;
  i_t top = reach(b, pinv, G, xi);
  for (i_t p = top; p < m; ++p) {
    x[xi[p]] = 0;
  }

  const i_t bnz = b.i.size();
  for (i_t p = 0; p < bnz; ++p) {
    x[b.i[p]] = b.x[p];
  }

  for (i_t px = top; px < m; ++px) {
    i_t j = xi[px];
    i_t J = pinv ? pinv[j] : j;
    if (J < 0) continue;
    f_t Gjj;
    i_t p;
    i_t end;
    if constexpr (lo) {
      Gjj = G.x[G.col_start[J]];
      p   = G.col_start[J] + 1;
      end = G.col_start[J + 1];
    } else {
      Gjj = G.x[G.col_start[J + 1] - 1];
      p   = G.col_start[J];
      end = G.col_start[J + 1] - 1;
    }
    x[j] /= Gjj;
    for (; p < end; ++p) {
      x[G.i[p]] -= G.x[p] * x[j];
    }
  }
  return top;
}

// ===========================================================================================
// KERNEL FISSION, WARP PATH — the `Direct` execution mode
//
// Two warp execution models coexist in this file, and confusing them is the #1 source of bugs:
//
//   * MONOLITH (Direct == false). Lane 0 runs the ENTIRE driver; lanes 1..31 are parked in
//     warp_helper_loop. Work reaches them only by staging a request into `cfg_shared` and calling
//     warp_collective (a 10-branch runtime op-code dispatch — the thing that pins the warp monolith
//     at ~250 registers). Because only lane 0 is executing driver code here, a __syncwarp or
//     __shfl_sync emitted on this path DEADLOCKS: the other lanes are blocked on the dispatch
//     barrier, not on ours.
//
//   * FISSION  (Direct == true).  ALL 32 lanes execute the kernel body, so the collective BODIES
//     (warp_X_body) are called directly — no cfg staging, no op code, statically-known callees.
//     But every statement the monolith ran on lane 0 alone now runs 32 times. That is fine for pure
//     reads and for stores that are identical from every lane; it is FATAL for
//       (a) anything that ALLOCATES — gpu_vector::reserve/resize/push_back call device malloc(),
//           so 32 lanes malloc 32 buffers, 31 leak, and each lane ends up with a DIFFERENT data_
//           pointer (the collective then writes 32 private buffers instead of one shared one), and
//       (b) any READ-MODIFY-WRITE on shared state — a size counter, an ft routing statistic, a
//           push_back cursor.
//     Those statements go inside lane0_serial<Direct>(), which runs them on lane 0 and re-converges.
//
// `Direct` must be a COMPILE-TIME parameter (if constexpr), never a runtime flag: a runtime branch
// keeps both arms live and ptxas re-allocates the register union we split the driver to escape.
// It is only ever instantiated as Direct == Warp.
// ===========================================================================================

// Run serial glue (allocation / read-modify-write) on lane 0, then re-converge. Under !Direct this
// is a plain call — see the deadlock note above.
template <bool Direct, class Fn>
__device__ __forceinline__ void lane0_serial(Fn&& f)
{
  if constexpr (Direct) {
    if (threadIdx.x == 0) { f(); }
    // Re-converge AND publish lane 0's global writes to the rest of the warp (__syncwarp orders
    // memory among the participating lanes), so the following collective reads what lane 0 wrote.
    __syncwarp(0xffffffffu);
  } else {
    f();
  }
}

// Publish a value computed inside a lane0_serial region (only lane 0 holds it) to every lane.
// Must be called by all 32 lanes, i.e. AFTER the lane0_serial that produced it.
template <bool Direct, class T>
__device__ __forceinline__ T lane0_bcast(T v)
{
  if constexpr (Direct) { return __shfl_sync(0xffffffffu, v, 0); } else { return v; }
}

// ---------------------------------------------------------------------------
// gpu_basis_update_mpf : the MPF LU factorization object. The initial factors
// L0_/U0_ (+ transposes) and permutations are fixed views uploaded from the
// host; S_/mu_values_ grow in-kernel as rank-1 updates are stacked.
// ---------------------------------------------------------------------------
struct gpu_basis_update_mpf {
  // Compile-time tag for the BU switch in the templated driver/kernel (mirrors the CPU
  // `is_mpf = std::is_same<BU, basis_update_mpf_t>` dispatch in simplified_dual_phase2).
  static constexpr bool is_mpf_bu = true;
  i_t num_updates_{0};
  i_t refactor_frequency_{100};
  gpu_csc_view L0_, U0_, L0_transpose_, U0_transpose_;
  gpu_span<i_t> row_permutation_, inverse_row_permutation_;
  gpu_csc_matrix S_;
  gpu_vector<f_t> mu_values_;
  gpu_span<i_t> xi_workspace_;  // size 2*m (scratch)
  gpu_span<f_t> x_workspace_;   // size m   (scratch)
  // Reused solve scratch (owning; grows to m once, then reused every call) so the b_solve /
  // b_transpose_solve dense↔sparse conversions and the sparse-Lᵀ gather no longer device-malloc a
  // fresh temporary per solve. Bit-identical to the per-call locals (same values, same order).
  mutable gpu_vector<f_t> dense_scratch_;    // solution_dense in the dense b_solve/b_transpose branches
  mutable gpu_sparse_vector sparse_scratch_; // gathered `b` (sparse Lᵀ) + throwaway Lsol
  mutable gpu_sparse_vector ei_scratch_;     // unit RHS for compute_delta_y (separate — b_transpose_solve
                                             // uses sparse_scratch_ internally, so ei can't share it)
  mutable gpu_vector<f_t> sort_bucket_;      // reused scratch for etilde.sort() (per-pivot sort
  mutable gpu_vector<i_t> sort_i_;           // in update(): the O(n) bucket + the n·log·n index/value/perm
  mutable gpu_vector<f_t> sort_x_;           // buffers, all reused instead of malloc'd each pivot.
  mutable gpu_vector<i_t> sort_perm_;
  // Statistics that drive the hypersparse-vs-dense path choice.
  mutable i_t num_calls_L_{0}, num_calls_U_{0}, num_calls_L_transpose_{0}, num_calls_U_transpose_{0};
  mutable f_t sum_L_{0.0}, sum_U_{0.0}, sum_L_transpose_{0.0}, sum_U_transpose_{0.0};
  f_t hypersparse_threshold_{0.05};
  // When non-null (warp-optimized kernel), the 4 dense triangular solves are
  // dispatched to the whole warp via this shared config instead of running on
  // one lane. Set by lane 0 before the driver runs.
  warp_solve_config* warp_cfg_ = nullptr;

  __device__ gpu_basis_update_mpf() {}
  gpu_basis_update_mpf(const gpu_basis_update_mpf&)            = delete;
  gpu_basis_update_mpf& operator=(const gpu_basis_update_mpf&) = delete;
  __device__ ~gpu_basis_update_mpf()                          = default;

  __device__ i_t num_updates() const { return num_updates_; }

  // Lane 0 posts a dense-solve request to the warp and joins the collective
  // solve. (Historical: this used to publish (x, op) to lanes parked in warp_helper_loop via
  // (x, op); warp_collective_solve runs nlev internal barriers; __syncwarp B
  // makes the result visible to lane 0 before it reads x.
  // Direct (fission): all 32 lanes are already here, so call the one known body instead of posting
  // an op code. The level SCHEDULE still comes from warp_cfg_ — in fission it lives in slot_state
  // and is built once by Ksetup, not on a kernel stack.
  // `Op` is a template parameter: every caller names its operator as a literal, so on the Direct
  // path there is no op code, no flag load and no branch left anywhere below this point.
  template <bool Direct, int Op>
  __device__ void warp_dispatch(f_t* rhs) const
  {
    static_assert(Direct, "the parked-helper (Direct=false) warp path was deleted");
    const warp_solve_config* c = warp_cfg_;
    warp_solve_body_op<Op>(c->corder[Op], c->cs[Op], c->lstart[Op], c->nlev[Op], c->ridx[Op],
                           c->val[Op], rhs);
  }

  // Post the MPF rank-1 T-loop to the whole warp (op == WARP_TLOOP). One dispatch runs the
  // entire replay collectively (tl_num internal barriers). S_ pointers are re-read each call
  // (they can move on resize). dot col = 2k+dotoff, axpy col = 2k+axpyoff; asc selects
  // l_solve (k up) vs l_transpose (k down).
  template <bool Direct = false>
  __device__ void warp_dispatch_tloop(f_t* rhs, bool asc, i_t dotoff, i_t axpyoff, f_t tol) const
  {
    static_assert(Direct, "the parked-helper (Direct=false) warp path was deleted");
    // S_ pointers are re-read here (they move on resize) — every lane reads the same globals.
    warp_tloop_body(asc, axpyoff, S_.col_start.data(), dotoff, mu_values_.data(), num_updates_,
                    S_.i.data(), tol, S_.x.data(), rhs);
  }



  // Reach-restricted sparse gather solve (op == WARP_SPTRSV). `G` is the SCATTER factor the serial
  // sparse_triangle_solve walks (L0_ / U0_ / their transposes); `op` is the matching dense gather
  // operator (WARP_L/U/LT/UT_SOLVE) whose transpose CSC + flags + precomputed level[] we reuse. Lane 0
  // (the lane that owns the serial glue under Direct) runs the serial reach,
  // preloads the rhs into x on the reachable set, and buckets the reachable columns by their precomputed
  // elimination level into sp_ls/sp_co; then the whole warp runs the wavefront gather over that sparse
  // schedule. Returns `top` (reachable pattern is xi_workspace_[top..m), values in x). Mirrors
  // sparse_triangle_solve's serial pre/post exactly so solve_to_workspace / solve_to_sparse_vector /
  // the T-loop that follow read the same xi_workspace_/x_workspace_ state.
  // The SERIAL half: reach + rhs preload + level-bucketing. Runs on ONE lane in both models (lane 0
  // by construction in the monolith; under a lane0_serial guard in fission). It marks/restores
  // G.col_start in place and rewrites the shared sp_ls/sp_co/sp_pos buckets, so replicating it
  // across 32 lanes would corrupt both. Returns `top`; publishes the reachable level count in nlevR.
  __device__ i_t sptrsv_prologue(int op, const gpu_sparse_vector& b, const gpu_csc_view& G, f_t* x,
                                 i_t& nlevR) const
  {
    const i_t  m  = G.m;
    i_t* const xi = xi_workspace_.data();
    // (1) reach: serial DFS, marks + restores G.col_start in place. xi[top..m) = reachable, topo order.
    const i_t top = reach(b, nullptr, G, xi);
    // (2) preload x on the reachable set, then scatter the rhs (identical to sparse_triangle_solve).
    for (i_t p = top; p < m; ++p) { x[xi[p]] = 0.0; }
    const i_t bnz = b.i.size();
    for (i_t p = 0; p < bnz; ++p) { x[b.i[p]] = b.x[p]; }
    // (3) bucket the reachable columns by their precomputed elimination level (counting sort).
    const warp_solve_config* c = warp_cfg_;
    const i_t* level = c->level[op];
    i_t* sp_ls  = c->sp_ls;
    i_t* sp_co  = c->sp_co;
    i_t* sp_pos = c->sp_pos;
    nlevR = 0;
    for (i_t p = top; p < m; ++p) { const i_t lv = level[xi[p]]; if (lv > nlevR) { nlevR = lv; } }
    for (i_t k = 0; k <= nlevR; ++k) { sp_ls[k] = 0; }
    for (i_t p = top; p < m; ++p) { sp_ls[level[xi[p]]]++; }          // level values ∈ [1, nlevR]
    for (i_t k = 1; k <= nlevR; ++k) { sp_ls[k] += sp_ls[k - 1]; }    // sp_ls[lv]=count(level<=lv)
    for (i_t lv = 1; lv <= nlevR; ++lv) { sp_pos[lv] = sp_ls[lv - 1]; }
    for (i_t p = top; p < m; ++p) { const i_t j = xi[p]; sp_co[sp_pos[level[j]]++] = j; }
    return top;
  }

  template <bool Direct, int Op>
  __device__ i_t warp_dispatch_sptrsv(const gpu_sparse_vector& b, const gpu_csc_view& G,
                                      f_t* x) const
  {
    constexpr int op = Op;
    i_t nlevR = 0;
    static_assert(Direct, "the parked-helper (Direct=false) warp path was deleted");
    i_t top = 0;
    lane0_serial<true>([&] { top = sptrsv_prologue(op, b, G, x, nlevR); });
    top   = lane0_bcast<true>(top);
    // EVERY lane needs nlevR: it is the wavefront's internal __syncwarp count, so a lane that
    // disagreed would desynchronize the whole warp.
    nlevR = lane0_bcast<true>(nlevR);
    const warp_solve_config* c = warp_cfg_;
    warp_sptrsv_body_op<Op>(c->cs[Op], c->ridx[Op], c->val[Op], c->sp_co, c->sp_ls,
                            static_cast<int>(nlevR), x);
    return top;
  }


  // Warp-parallel gpu_sparse_vector::to_dense — lane 0 sizes the dense buffer (no zero-fill), the warp
  // zeros [0,n) + scatters the nnz. Bit-exact (distinct sparse indices). Replaces the O(n) serial tail
  // before each dense solve.
  template <bool Direct = false>
  __device__ void warp_to_dense(const gpu_sparse_vector& sv, gpu_vector<f_t>& dense) const
  {
    const i_t n = sv.n;
    static_assert(Direct, "the parked-helper (Direct=false) warp path was deleted");
    // reserve() may device-malloc and set_size() is a header RMW → lane 0 only. The __syncwarp
    // inside lane0_serial is also what publishes the (possibly relocated) data_ pointer, so
    // dense.data() below must be read AFTER it, not before.
    lane0_serial<true>([&] { dense.reserve(n); dense.set_size(n); });
    warp_todense_body(dense.data(), sv.i.data(), n, static_cast<i_t>(sv.i.size()), sv.x.data());
  }

  // Warp-parallel gpu_sparse_vector::from_dense — lane 0 reserves the sparse buffers, the warp compacts
  // the nonzeros of `dense` in increasing index order, lane 0 sets the final size. Bit-exact.
  template <bool Direct = false>
  __device__ void warp_from_dense(const gpu_vector<f_t>& dense, gpu_sparse_vector& sv) const
  {
    const i_t n = static_cast<i_t>(dense.size());
    static_assert(Direct, "the parked-helper (Direct=false) warp path was deleted");
    lane0_serial<true>([&] { sv.n = n; sv.i.reserve(n); sv.x.reserve(n); });
    i_t count = 0;                   // the body writes it on lane 0 only; each lane has its own
    warp_fromdense_body(&count, sv.i.data(), dense.data(), n, sv.x.data());
    // set_size is a header RMW; lane 0 already holds the right count, and the trailing
    // __syncwarp makes the new size visible to every lane's later sv.i.size() read.
    lane0_serial<true>([&] { sv.i.set_size(count); sv.x.set_size(count); });
  }





  __device__ f_t estimate_solution_density(f_t rhs_nz,
                                           f_t sum,
                                           i_t& num_calls,
                                           bool& use_hypersparse) const
  {
    num_calls++;
    const f_t average_growth    = fmax(1.0, sum / static_cast<f_t>(num_calls));
    const f_t predicted_nz      = rhs_nz * average_growth;
    const f_t predicted_density = predicted_nz / static_cast<f_t>(L0_.m);
    use_hypersparse             = predicted_density < hypersparse_threshold_;
    return predicted_nz;
  }

  // ---- private-helper equivalents ----
  __device__ void gather_into_sparse_vector(i_t nz, gpu_sparse_vector& out) const
  {
    const i_t m = L0_.m;
    out.i.clear();
    out.x.clear();
    out.i.reserve(nz);
    out.x.reserve(nz);
    const f_t zero_tol = 1e-13;
    for (i_t k = 0; k < nz; ++k) {
      const i_t i = xi_workspace_[m + k];
      if (fabs(x_workspace_[i]) > zero_tol) {
        out.i.push_back(i);
        out.x.push_back(x_workspace_[i]);
      }
      xi_workspace_[m + k] = 0;
      xi_workspace_[i]     = 0;
      x_workspace_[i]      = 0.0;
    }
  }

  __device__ void solve_to_workspace(i_t top) const
  {
    const i_t m = L0_.m;
    i_t nz      = 0;
    for (i_t p = top; p < m; ++p) {
      const i_t i           = xi_workspace_[p];
      xi_workspace_[m + nz] = i;
      xi_workspace_[p]      = 0;
      nz++;
    }
    for (i_t k = 0; k < nz; ++k) {
      const i_t i      = xi_workspace_[m + k];
      xi_workspace_[i] = 1;
    }
  }

  __device__ void solve_to_sparse_vector(i_t top, gpu_sparse_vector& out) const
  {
    const i_t m  = L0_.m;
    out.n        = m;
    const i_t nz = m - top;
    out.x.clear();
    out.i.clear();
    out.x.reserve(nz);
    out.i.reserve(nz);
    i_t k              = 0;
    const f_t zero_tol = 1e-13;
    for (i_t p = top; p < m; ++p) {
      const i_t i = xi_workspace_[p];
      if (fabs(x_workspace_[i]) > zero_tol) {
        out.i.push_back(i);
        out.x.push_back(x_workspace_[i]);
      }
      x_workspace_[i]  = 0.0;
      xi_workspace_[p] = 0;
      k++;
    }
  }

  __device__ i_t scatter_into_workspace(const gpu_sparse_vector& in) const
  {
    const i_t m = L0_.m;
    i_t nz      = in.i.size();
    for (i_t k = 0; k < nz; ++k) {
      const i_t i          = in.i[k];
      xi_workspace_[i]     = 1;
      xi_workspace_[m + k] = i;
    }
    for (i_t k = 0; k < nz; ++k) {
      x_workspace_[in.i[k]] = in.x[k];
    }
    return nz;
  }

  __device__ void grow_storage(i_t nz, i_t& S_start, i_t& S_nz)
  {
    const i_t last_S_col     = num_updates_ * 2;
    const i_t new_last_S_col = last_S_col + 2;
    if (new_last_S_col >= static_cast<i_t>(S_.col_start.size())) {
      S_.col_start.resize(new_last_S_col + refactor_frequency_);
    }
    S_nz = S_.col_start[last_S_col];
    if (S_nz + nz > static_cast<i_t>(S_.i.size())) {
      const i_t newsz = (2 * S_nz > S_nz + nz) ? 2 * S_nz : S_nz + nz;
      S_.i.resize(newsz);
      S_.x.resize(newsz);
    }
    S_start = last_S_col;
  }

  // dot = S(:, col)' * x   (dense x)
  __device__ f_t dot_product(i_t col, const f_t* x) const
  {
    f_t dot             = 0.0;
    const i_t col_start = S_.col_start[col];
    const i_t col_end   = S_.col_start[col + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = S_.i[p];
      dot += S_.x[p] * x[i];
    }
    return dot;
  }

  // dot = S(:, col)' * x   (sparse x, restricted to marked rows)
  __device__ f_t dot_product(i_t col, const i_t* mark, const f_t* x) const
  {
    f_t dot             = 0.0;
    const i_t col_start = S_.col_start[col];
    const i_t col_end   = S_.col_start[col + 1];
    i_t nz_mark         = 0;
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = S_.i[p];
      if (mark[i]) {
        dot += S_.x[p] * x[i];
        nz_mark++;
      }
    }
    return dot;
  }

  // x <- x + theta * Mat(:, col)   (dense x). Mat is S_ (owning) or a view.
  template <typename Mat>
  __device__ void add_sparse_column(const Mat& S, i_t col, f_t theta, f_t* x) const
  {
    const i_t col_start = S.col_start[col];
    const i_t col_end   = S.col_start[col + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = S.i[p];
      x[i] += theta * S.x[p];
    }
  }

  // x <- x + theta * Mat(:, col)   (sparse x, tracking fill via mark/nz).
  template <typename Mat>
  __device__ void add_sparse_column(
    const Mat& S, i_t col, f_t theta, i_t* mark, i_t& nz, f_t* x) const
  {
    const i_t m         = L0_.m;
    const i_t col_start = S.col_start[col];
    const i_t col_end   = S.col_start[col + 1];
    i_t nz_start        = nz;
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = S.i[p];
      if (!mark[i]) {
        mark[i]      = 1;
        mark[m + nz] = i;
        nz++;
      }
      x[i] += theta * S.x[p];
    }
  }

  // ---- triangular solves through the MPF form ----
  // `Direct` is only ever true together with Warp (fission warp path) — see the Direct-mode banner
  // above gpu_basis_update_mpf. It selects direct body calls + lane-0 guards on the serial glue.
  template <bool Warp = false, bool Direct = false>
  __device__ i_t u_transpose_solve(f_t* rhs) const
  {
    if constexpr (Warp) { warp_dispatch<Direct, WARP_UT_SOLVE>(rhs); }
    else { upper_triangular_transpose_solve(U0_, rhs); }
    return 0;
  }
  template <bool Warp = false, bool Direct = false>
  __device__ i_t u_transpose_solve(gpu_sparse_vector& rhs) const
  {
    i_t top;
    if constexpr (Warp) {
      top = warp_dispatch_sptrsv<Direct, WARP_UT_SOLVE>(rhs, U0_transpose_, x_workspace_.data());
    } else {
      top = sparse_triangle_solve<true>(rhs, nullptr, xi_workspace_.data(), U0_transpose_,
                                        x_workspace_.data());
    }
    // push_back + workspace RMW -> lane 0 only under Direct.
    lane0_serial<Direct>([&] { solve_to_sparse_vector(top, rhs); });
    return 0;
  }

  template <bool Warp = false, bool Direct = false>
  __device__ i_t l_transpose_solve(f_t* rhs) const
  {
    const f_t zero_tol = 1e-13;
    if constexpr (Warp) {
      // dot col = u_col (2k), axpy col = v_col (2k+1), descending k.
      if (num_updates_ > 0) { ST_TIC(_tl1); warp_dispatch_tloop<Direct>(rhs, /*asc=*/false, /*dotoff=*/0, /*axpyoff=*/1, zero_tol); ST_TOC(SG_LU_TLOOP, _tl1); }
    } else {
      for (i_t k = num_updates_ - 1; k >= 0; --k) {
        const i_t u_col = 2 * k;
        const i_t v_col = 2 * k + 1;
        const f_t mu    = mu_values_[k];
        f_t dot         = dot_product(u_col, rhs);
        const f_t theta = dot / mu;
        if (fabs(theta) > zero_tol) { add_sparse_column(S_, v_col, -theta, rhs); }
      }
    }
    if constexpr (Warp) { warp_dispatch<Direct, WARP_LT_SOLVE>(rhs); }
    else { lower_triangular_transpose_solve(L0_, rhs); }
    return 0;
  }
  template <bool Warp = false, bool Direct = false>
  __device__ i_t l_transpose_solve(gpu_sparse_vector& rhs) const
  {
    const i_t m = L0_.m;
    // One lane-0 region: scatter + the SPARSE MPF T-loop (serial in both warp models) + the gather.
    // All three read-modify-write xi_workspace_/x_workspace_/sparse_scratch_ and carry `nz`.
    lane0_serial<Direct>([&] {
      scatter_into_workspace(rhs);
      i_t nz = rhs.i.size();

      const f_t zero_tol = 1e-13;
      ST_TIC(_tl3);
      for (i_t k = num_updates_ - 1; k >= 0; --k) {   // MPF T-loop kept serial on lane 0 for now
        const i_t u_col = 2 * k;
        const i_t v_col = 2 * k + 1;
        const f_t mu    = mu_values_[k];
        f_t dot         = dot_product(u_col, xi_workspace_.data(), x_workspace_.data());
        const f_t theta = dot / mu;
        if (fabs(theta) > zero_tol) {
          add_sparse_column(S_, v_col, -theta, xi_workspace_.data(), nz, x_workspace_.data());
        }
      }
      ST_TOC(SG_LU_TLOOP, _tl3);

      gpu_sparse_vector& b = sparse_scratch_;  // reused (was a per-hypersparse-solve device malloc)
      b.n = m;
      gather_into_sparse_vector(nz, b);
    });
    gpu_sparse_vector& b = sparse_scratch_;
    i_t top;
    if constexpr (Warp) {
      top = warp_dispatch_sptrsv<Direct, WARP_LT_SOLVE>(b, L0_transpose_, x_workspace_.data());
    } else {
      top = sparse_triangle_solve<false>(b, nullptr, xi_workspace_.data(), L0_transpose_,
                                         x_workspace_.data());
    }
    lane0_serial<Direct>([&] { solve_to_sparse_vector(top, rhs); });
    return 0;
  }

  template <bool Warp = false, bool Direct = false>
  __device__ i_t u_solve(f_t* rhs) const
  {
    if constexpr (Warp) { warp_dispatch<Direct, WARP_U_SOLVE>(rhs); }
    else { upper_triangular_solve(U0_, rhs); }
    return 0;
  }
  template <bool Warp = false, bool Direct = false>
  __device__ i_t u_solve(gpu_sparse_vector& rhs) const
  {
    i_t top;
    if constexpr (Warp) {
      top = warp_dispatch_sptrsv<Direct, WARP_U_SOLVE>(rhs, U0_, x_workspace_.data());
    } else {
      top = sparse_triangle_solve<false>(rhs, nullptr, xi_workspace_.data(), U0_,
                                         x_workspace_.data());
    }
    lane0_serial<Direct>([&] { solve_to_sparse_vector(top, rhs); });
    return 0;
  }

  template <bool Warp = false, bool Direct = false>
  __device__ void l_solve(f_t* rhs) const
  {
    if constexpr (Warp) { warp_dispatch<Direct, WARP_L_SOLVE>(rhs); }
    else { lower_triangular_solve(L0_, rhs); }
    const f_t zero_tol = 1e-16;  // Any higher and pilot_ja fails
    if constexpr (Warp) {
      // dot col = v_col (2k+1), axpy col = u_col (2k), ascending k.
      if (num_updates_ > 0) { ST_TIC(_tl2); warp_dispatch_tloop<Direct>(rhs, /*asc=*/true, /*dotoff=*/1, /*axpyoff=*/0, zero_tol); ST_TOC(SG_LU_TLOOP, _tl2); }
    } else {
      for (i_t k = 0; k < num_updates_; ++k) {
        const f_t mu    = mu_values_[k];
        const i_t u_col = 2 * k;
        const i_t v_col = 2 * k + 1;
        f_t dot         = dot_product(v_col, rhs);
        const f_t theta = dot / mu;
        if (fabs(theta) > zero_tol) { add_sparse_column(S_, u_col, -theta, rhs); }
      }
    }
  }
  template <bool Warp = false, bool Direct = false>
  __device__ void l_solve(gpu_sparse_vector& rhs) const
  {
    const i_t m = L0_.m;
    i_t top;
    if constexpr (Warp) {
      top = warp_dispatch_sptrsv<Direct, WARP_L_SOLVE>(rhs, L0_, x_workspace_.data());
    } else {
      top = sparse_triangle_solve<true>(rhs, nullptr, xi_workspace_.data(), L0_,
                                        x_workspace_.data());
    }
    lane0_serial<Direct>([&] {
      solve_to_workspace(top);
      i_t nz             = m - top;
      const f_t zero_tol = 1e-13;
      ST_TIC(_tl4);
      for (i_t k = 0; k < num_updates_; ++k) {   // MPF T-loop kept serial on lane 0 for now
        const f_t mu    = mu_values_[k];
        const i_t u_col = 2 * k;
        const i_t v_col = 2 * k + 1;
        f_t dot         = dot_product(v_col, xi_workspace_.data(), x_workspace_.data());
        const f_t theta = dot / mu;
        if (fabs(theta) > zero_tol) {
          add_sparse_column(S_, u_col, -theta, xi_workspace_.data(), nz, x_workspace_.data());
        }
      }
      ST_TOC(SG_LU_TLOOP, _tl4);
      gather_into_sparse_vector(nz, rhs);
    });
  }

  // ---- B solves ----
  // NOTE (Direct): `use_hypersparse` decides which SEQUENCE of collectives the warp executes, and
  // estimate_solution_density MUTATES the routing statistics (num_calls_/sum_), so it must run
  // exactly once — on lane 0 — and the resulting route must then be broadcast. If a lane disagreed
  // about the route it would call a different body and the warp would desynchronize.
  template <bool Warp = false, bool Direct = false>
  __device__ i_t b_solve(const gpu_vector<f_t>& rhs, gpu_vector<f_t>& solution) const
  {
    lane0_serial<Direct>(
      [&] { permute_vector(row_permutation_.data(), rhs.data(), solution.data(), L0_.m); });
    l_solve<Warp, Direct>(solution.data());
    u_solve<Warp, Direct>(solution.data());
    return 0;
  }

  template <bool Warp = false, bool Direct = false>
  __device__ i_t b_solve(const gpu_sparse_vector& rhs,
                         gpu_sparse_vector& solution,
                         gpu_sparse_vector& Lsol,
                         bool need_Lsol = true) const
  {
    bool use_hypersparse = false;
    const f_t input_size = static_cast<f_t>(rhs.i.size());
    lane0_serial<Direct>([&] {
      solution = rhs;                 // copy-assign: reserve() -> device malloc
      solution.inverse_permute_vector(inverse_row_permutation_.data());
      estimate_solution_density(input_size, sum_L_, num_calls_L_, use_hypersparse);
    });
    use_hypersparse = lane0_bcast<Direct>(static_cast<int>(use_hypersparse)) != 0;
    if (use_hypersparse) {
      ST_TIC(_slh); l_solve<Warp, Direct>(solution); ST_TOC(SG_LU_B_L_HYP, _slh);
    } else {
      gpu_vector<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      if constexpr (Warp) { warp_to_dense<Direct>(solution, solution_dense); } else { solution.to_dense(solution_dense); }
      ST_TIC(_sld); l_solve<Warp, Direct>(solution_dense.data()); ST_TOC(SG_LU_B_L_DEN, _sld);
      if constexpr (Warp) { warp_from_dense<Direct>(solution_dense, solution); } else { solution.from_dense(solution_dense); }
    }
    lane0_serial<Direct>([&] {
      if (need_Lsol) { Lsol = solution; }
      sum_L_ += static_cast<f_t>(solution.i.size()) / input_size;
    });

    const f_t rhs_size = static_cast<f_t>(solution.i.size());
    lane0_serial<Direct>(
      [&] { estimate_solution_density(rhs_size, sum_U_, num_calls_U_, use_hypersparse); });
    use_hypersparse = lane0_bcast<Direct>(static_cast<int>(use_hypersparse)) != 0;
    if (use_hypersparse) {
      ST_TIC(_suh); u_solve<Warp, Direct>(solution); ST_TOC(SG_LU_B_U_HYP, _suh);
    } else {
      gpu_vector<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      if constexpr (Warp) { warp_to_dense<Direct>(solution, solution_dense); } else { solution.to_dense(solution_dense); }
      ST_TIC(_sud); u_solve<Warp, Direct>(solution_dense.data()); ST_TOC(SG_LU_B_U_DEN, _sud);
      if constexpr (Warp) { warp_from_dense<Direct>(solution_dense, solution); } else { solution.from_dense(solution_dense); }
    }
    lane0_serial<Direct>([&] { sum_U_ += static_cast<f_t>(solution.i.size()) / rhs_size; });
    return 0;
  }

  template <bool Warp = false, bool Direct = false>
  __device__ i_t b_solve(const gpu_sparse_vector& rhs, gpu_sparse_vector& solution) const
  {
    // Lsol is nz=0 => resize(0) => no malloc, and need_Lsol=false means it is never written.
    // Safe to leave as a per-lane local under Direct.
    gpu_sparse_vector Lsol(1, 0);
    return b_solve<Warp, Direct>(rhs, solution, Lsol, false);
  }

  template <bool Warp = false, bool Direct = false>
  __device__ i_t b_transpose_solve(const gpu_sparse_vector& rhs,
                                   gpu_sparse_vector& solution,
                                   gpu_sparse_vector& UTsol) const
  {
    bool use_hypersparse = false;
    const f_t input_size = static_cast<f_t>(rhs.i.size());
    lane0_serial<Direct>([&] {
      estimate_solution_density(
        input_size, sum_U_transpose_, num_calls_U_transpose_, use_hypersparse);
    });
    use_hypersparse = lane0_bcast<Direct>(static_cast<int>(use_hypersparse)) != 0;
    if (use_hypersparse) {
      lane0_serial<Direct>([&] { solution = rhs; });   // copy-assign: allocates
      ST_TIC(_suth); u_transpose_solve<Warp, Direct>(solution); ST_TOC(SG_LU_BT_U_HYP, _suth);
    } else {
      gpu_vector<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      if constexpr (Warp) { warp_to_dense<Direct>(rhs, solution_dense); } else { rhs.to_dense(solution_dense); }
      ST_TIC(_sutd); u_transpose_solve<Warp, Direct>(solution_dense.data()); ST_TOC(SG_LU_BT_U_DEN, _sutd);
      if constexpr (Warp) { warp_from_dense<Direct>(solution_dense, solution); } else { solution.from_dense(solution_dense); }
    }
    lane0_serial<Direct>([&] {
      UTsol = solution;                                 // copy-assign: allocates
      sum_U_transpose_ += static_cast<f_t>(solution.i.size()) / input_size;
    });

    const f_t rhs_size = static_cast<f_t>(solution.i.size());
    lane0_serial<Direct>([&] {
      estimate_solution_density(
        rhs_size, sum_L_transpose_, num_calls_L_transpose_, use_hypersparse);
    });
    use_hypersparse = lane0_bcast<Direct>(static_cast<int>(use_hypersparse)) != 0;
    if (use_hypersparse) {
      ST_TIC(_slth); l_transpose_solve<Warp, Direct>(solution); ST_TOC(SG_LU_BT_L_HYP, _slth);
    } else {
      gpu_vector<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      if constexpr (Warp) { warp_to_dense<Direct>(solution, solution_dense); } else { solution.to_dense(solution_dense); }
      ST_TIC(_sltd); l_transpose_solve<Warp, Direct>(solution_dense.data()); ST_TOC(SG_LU_BT_L_DEN, _sltd);
      if constexpr (Warp) { warp_from_dense<Direct>(solution_dense, solution); } else { solution.from_dense(solution_dense); }
    }
    lane0_serial<Direct>([&] {
      sum_L_transpose_ += static_cast<f_t>(solution.i.size()) / rhs_size;
      solution.inverse_permute_vector(row_permutation_.data());
    });
    return 0;
  }

  // ---- stacking rank-1 update (sparse form) ----
  __device__ i_t update(const gpu_sparse_vector& utilde, gpu_sparse_vector& etilde, i_t leaving_index)
  {
    const i_t m = L0_.m;

    i_t nz = scatter_into_workspace(utilde);

    add_sparse_column(U0_, leaving_index, -1.0, xi_workspace_.data(), nz, x_workspace_.data());

    gpu_sort(xi_workspace_.data() + m,
             xi_workspace_.data() + m + nz,
             [] __device__(i_t a, i_t b) { return a < b; });

    i_t S_start;
    i_t S_nz;
    grow_storage(nz + static_cast<i_t>(etilde.i.size()), S_start, S_nz);

    S_.append_column(nz, xi_workspace_.data() + m, x_workspace_.data());

    etilde.sort(sort_bucket_, sort_i_, sort_x_, sort_perm_);  // reused scratch (no per-pivot malloc)
    {
      // S_.append_column(sparse_vector etilde): append nonzeros directly.
      i_t pos       = S_.col_start[S_.n];
      const i_t enz = etilde.i.size();
      for (i_t k = 0; k < enz; ++k) {
        const f_t v = etilde.x[k];
        if (v != 0.0) {
          S_.i[pos] = etilde.i[k];
          S_.x[pos] = v;
          pos++;
        }
      }
      S_.col_start[S_.n + 1] = pos;
      S_.n++;
    }

    const f_t mu = 1.0 + sparse_dot(S_.i.data() + S_.col_start[S_start],
                                    S_.x.data() + S_.col_start[S_start],
                                    S_.col_start[S_start + 1] - S_.col_start[S_start],
                                    S_.i.data() + S_.col_start[S_start + 1],
                                    S_.x.data() + S_.col_start[S_start + 1],
                                    S_.col_start[S_start + 2] - S_.col_start[S_start + 1]);

    if (fabs(mu) < 1E-8 || fabs(mu) > 1E+8) { return 1; }
    mu_values_.push_back(mu);
    // Clear the workspace.
    for (i_t k = 0; k < nz; ++k) {
      const i_t i          = xi_workspace_[m + k];
      xi_workspace_[i]     = 0;
      x_workspace_[i]      = 0.0;
      xi_workspace_[m + k] = 0;
    }
    num_updates_++;
    return 0;
  }
};

// ---------------------------------------------------------------------------
// LP / settings / solution device structs (non-owning views over uploaded data)
// ---------------------------------------------------------------------------
struct gpu_lp {
  i_t num_rows{0}, num_cols{0};
  gpu_csc_view A;
  gpu_span<f_t> rhs, lower, upper, objective;
  f_t obj_scale{1.0}, obj_constant{0.0};
  // Optional host-built CSR of A (Arow). Null => build on device via from_csc.
  const i_t* Arow_row_start{nullptr};
  const i_t* Arow_j{nullptr};
  const f_t* Arow_x{nullptr};
};

struct gpu_settings {
  f_t primal_tol, dual_tol, pivot_tol, tight_tol, fixed_tol, zero_tol, time_limit, cut_off,
    steepest_edge_ratio;
  i_t refactor_frequency;
};

struct gpu_solution {
  gpu_span<f_t> x, y, z;
  f_t objective{0.0}, user_objective{0.0};
  i_t iterations{0};
};

// ---------------------------------------------------------------------------
// phase2::* helpers (phase2.cpp / solve.cpp).
// ---------------------------------------------------------------------------
__device__ f_t compute_objective(const gpu_lp& problem, const f_t* x)
{
  const i_t n = problem.num_cols;
  f_t obj     = 0.0;
  for (i_t j = 0; j < n; ++j) {
    obj += problem.objective[j] * x[j];
  }
  return obj;
}

__device__ f_t compute_user_objective(const gpu_lp& lp, f_t obj)
{
  return lp.obj_scale * (obj + lp.obj_constant);
}

// Incremental objective update (strict port of phase2::update_objective, phase2.cpp:1864):
// obj += delta_x·objective over the changed basics + the entering variable. Called with the
// OLD basic_list (before the basis swap) and — like the advanced/CPU-simplified driver —
// WITHOUT the bound-flip contribution. changed_basic_indices == scaled_dxB_sparse (its .i are
// the changed positions in basic_list). O(nnz), not O(n).
// Direct=true reduces the sum with a warp butterfly (warp_reduce_add). This is NOT bit-exact vs the
// serial left-to-right accumulation — the adds are reordered — and `obj` is the CUTOFF comparand read
// in K5, so this is one of the two places where the fission warp path deliberately stops being
// bit-identical to the monolith warp. See FISSION_CHALLENGES.md §20.
template <bool Direct = false>
__device__ void update_objective(const i_t* basic_list,
                                 const gpu_sparse_vector& changed_basic_indices,
                                 const gpu_span<f_t>& objective,
                                 const f_t* delta_x,
                                 i_t entering_index,
                                 f_t& obj)
{
  const i_t changed_basic_nz = (i_t)changed_basic_indices.i.size();
  if constexpr (Direct) {
    const int lane = static_cast<int>(threadIdx.x);
    f_t part = 0.0;
    for (i_t k = lane; k < changed_basic_nz; k += 32) {
      const i_t j = basic_list[changed_basic_indices.i[k]];
      part += delta_x[j] * objective[j];
    }
    const f_t total = warp_reduce_add(part);   // every lane leaves with the full sum
    // `obj` aliases st.obj in global memory — one writer only.
    lane0_serial<true>([&] {
      obj += total;
      // Leaving index already included above
      obj += delta_x[entering_index] * objective[entering_index];
    });
  } else {
    for (i_t k = 0; k < changed_basic_nz; ++k) {
      const i_t j = basic_list[changed_basic_indices.i[k]];
      obj += delta_x[j] * objective[j];
    }
    // Leaving index already included above
    obj += delta_x[entering_index] * objective[entering_index];
  }
}

// Direct=true (kernel fission, warp): ALL 32 lanes execute this, so the collective is called
// directly (no cfg_shared / op-code dispatch) and the lane-0-only *dz_count must be broadcast before
// the global-memory set_size. Direct=false (monolith): only lane 0 is here — a __shfl_sync would
// deadlock against the lanes parked in warp_helper_loop. Same contract as compute_delta_z.
template <bool Warp = false, bool Direct = false, class FT>
__device__ void compute_reduced_cost_update(const gpu_lp& lp,
                                            const FT& ft,
                                            const i_t* basic_list,
                                            const i_t* nonbasic_list,
                                            const f_t* delta_y,
                                            i_t leaving_index,
                                            i_t direction,
                                            i_t* delta_z_mark,
                                            gpu_vector<i_t>& delta_z_indices,
                                            f_t* delta_z)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  if constexpr (Warp) {
    // One warp dispatch: parallel gather dots (one lane per nonbasic column, bit-exact) + ordered
    // append of the nonzero columns. delta_z_indices is empty here (cleared each iter) and reserved
    // to n, so the collective raw-writes [0,count) and we set the size back on return.
    i_t dz_count = 0;
    if constexpr (Direct) {
      warp_dz_body(lp.A.col_start.data(), lp.A.i.data(), lp.A.x.data(), basic_list, &dz_count,
                   direction, delta_y, delta_z_indices.data(), leaving_index, m, delta_z_mark, n,
                   nonbasic_list, delta_z);
      // the body writes *dz_count on lane 0 only; every lane needs it for set_size (global memory).
      dz_count = __shfl_sync(0xffffffffu, dz_count, 0);
    }
    delta_z_indices.set_size(dz_count);
    return;
  }

  for (i_t k = 0; k < m; k++) {
    const i_t j = basic_list[k];
    delta_z[j]  = 0;
  }
  delta_z[leaving_index] = direction;
  const i_t num_nonbasic = n - m;
  for (i_t k = 0; k < num_nonbasic; k++) {
    const i_t j         = nonbasic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    f_t dot             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      dot += lp.A.x[p] * delta_y[lp.A.i[p]];
    }

    delta_z[j] = -dot;
    if (dot != 0.0) {
      delta_z_indices.push_back(j);
      delta_z_mark[j] = 1;
    }
  }
}

// Transpose delta_z: delta_z_N = -A_N' delta_y, row-wise on the CSR copy Arow,
// iterating only the nonzero rows of delta_y and only their nonbasic-column portion
// (Arow.row_start[i] .. nonbasic_end[i]). Mirrors the CPU compute_delta_z bit-for-bit.
// Direct=true (kernel fission, warp): ALL 32 lanes execute this function, so the collective is
// called directly (no cfg_shared / op-code dispatch) and the lane-0-only output must be broadcast
// before the global-memory set_size. Direct=false (monolith): only lane 0 is here — a __shfl_sync
// would deadlock waiting for lanes parked in warp_helper_loop, so it must NOT be emitted.
template <bool Warp, bool Direct = false, class FT>
__device__ void compute_delta_z(const FT& ft,
                                const gpu_csr_matrix& Arow,
                                const gpu_sparse_vector& delta_y,
                                i_t leaving_index,
                                i_t direction,
                                const i_t* nonbasic_end,
                                i_t* delta_z_mark,
                                gpu_vector<i_t>& delta_z_indices,
                                f_t* delta_z)
{
  if constexpr (Warp) {
    // Collectively scatter (inner span per row split across the warp). delta_z_indices is empty here
    // (cleared each iter, reserved to n); the collective raw-writes [0,count), we set the size back.
    // Always taken on the warp path (proven ~1.12x overall, bit-exact — was CUOPT_GPU_DZSCAT, now default).
    i_t dz_count = 0;
    if constexpr (Direct) {
      warp_dz_scatter_body(&dz_count, direction, delta_z_indices.data(), leaving_index, delta_z_mark,
                           delta_z, delta_y.i.data(), delta_y.x.data(), Arow.j.data(), nonbasic_end,
                           static_cast<i_t>(delta_y.i.size()), Arow.row_start.data(), Arow.x.data());
      // the body writes *dz_count on lane 0 only; every lane needs it for set_size (global memory).
      dz_count = __shfl_sync(0xffffffffu, dz_count, 0);
    }
    delta_z_indices.set_size(dz_count);
    return;
  }
  const i_t nz_delta_y = (i_t)delta_y.i.size();
  for (i_t k = 0; k < nz_delta_y; k++) {
    const i_t i         = delta_y.i[k];
    const f_t delta_y_i = delta_y.x[k];
    if (fabs(delta_y_i) < 1e-12) { continue; }
    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = nonbasic_end[i] + 1;
    for (i_t p = row_start; p < row_end; ++p) {
      const i_t j = Arow.j[p];
      delta_z[j] -= delta_y_i * Arow.x[p];
      if (!delta_z_mark[j]) {
        delta_z_mark[j] = 1;
        delta_z_indices.push_back(j);
      }
    }
  }
  delta_z[leaving_index] = direction;
}

// Partition each row of Arow so nonbasic-variable coefficients occupy
// Arow.row_start[i] .. nonbasic_end[i]. Mirrors compute_initial_nonbasic_end.
// Warp path: one lane per row (rows independent) via WARP_NBE; the serial loop is the reference.
template <bool Warp = false, bool Direct = false, class FT>
__device__ void compute_initial_nonbasic_end(const FT& ft,
                                             const i_t* basic_mark,
                                             gpu_csr_matrix& Arow,
                                             i_t* nonbasic_end)
{
  const i_t m = Arow.m;
  if constexpr (Warp) {
    if constexpr (Direct) {
      // One lane per row; rows touch disjoint j/x slices and disjoint nonbasic_end[i], so this is
      // bit-exact vs the serial two-pointer partition regardless of lane scheduling. No barrier.
      warp_nbe_body(basic_mark, nonbasic_end, Arow.j.data(), m, Arow.row_start.data(),
                    Arow.x.data());
    }
    return;
  }
  for (i_t i = 0; i < m; i++) {
    i_t lo = Arow.row_start[i];
    i_t hi = Arow.row_start[i + 1] - 1;
    while (lo <= hi) {
      const i_t j = Arow.j[lo];
      if (basic_mark[j] >= 0) {
        const i_t tj = Arow.j[lo]; Arow.j[lo] = Arow.j[hi]; Arow.j[hi] = tj;
        const f_t tx = Arow.x[lo]; Arow.x[lo] = Arow.x[hi]; Arow.x[hi] = tx;
        hi--;
      } else {
        lo++;
      }
    }
    nonbasic_end[i] = hi;
  }
}

// Keep the Arow nonbasic partition in sync after a pivot (entering -> basic, leaving
// -> nonbasic). Mirrors update_Arow. row_mark is scratch.
__device__ void update_Arow(i_t leaving,
                            i_t entering,
                            const gpu_csc_view& A,
                            gpu_vector<i_t>& row_mark,
                            i_t* nonbasic_end,
                            gpu_csr_matrix& Arow)
{
  row_mark.clear();
  const i_t col_start_enter = A.col_start[entering];
  const i_t col_end_enter   = A.col_start[entering + 1];
  for (i_t p = col_start_enter; p < col_end_enter; p++) { row_mark.push_back(A.i[p]); }

  for (i_t t = 0; t < (i_t)row_mark.size(); ++t) {
    const i_t i         = row_mark[t];
    const i_t row_start = Arow.row_start[i];
    const i_t nb_end    = nonbasic_end[i];
    for (i_t p = row_start; p <= nb_end; p++) {
      if (Arow.j[p] == entering) {
        const i_t tj = Arow.j[p]; Arow.j[p] = Arow.j[nb_end]; Arow.j[nb_end] = tj;
        const f_t tx = Arow.x[p]; Arow.x[p] = Arow.x[nb_end]; Arow.x[nb_end] = tx;
        nonbasic_end[i]--;
        break;
      }
    }
  }

  row_mark.clear();
  const i_t col_start_leave = A.col_start[leaving];
  const i_t col_end_leave   = A.col_start[leaving + 1];
  for (i_t p = col_start_leave; p < col_end_leave; p++) { row_mark.push_back(A.i[p]); }

  for (i_t t = 0; t < (i_t)row_mark.size(); ++t) {
    const i_t i       = row_mark[t];
    const i_t nb_end  = nonbasic_end[i];
    const i_t row_end = Arow.row_start[i + 1];
    for (i_t p = nb_end + 1; p < row_end; p++) {
      if (Arow.j[p] == leaving) {
        const i_t tj = Arow.j[p]; Arow.j[p] = Arow.j[nb_end + 1]; Arow.j[nb_end + 1] = tj;
        const f_t tx = Arow.x[p]; Arow.x[p] = Arow.x[nb_end + 1]; Arow.x[nb_end + 1] = tx;
        nonbasic_end[i]++;
        break;
      }
    }
  }
}

// Exact steepest-edge norm init: gamma_j = ||B^-T e_k||^2 for each basic var, via a
// singleton-row fast path (1/alpha^2) or a b_transpose_solve. Mirrors the CPU
// initialize_steepest_edge_norms (the simplified always uses the non-slack/exact path).
// All scratch is CALLER-OWNED (slot-resident). Under Direct every lane executes this, so a
// stack-local gpu_vector would device-malloc PER LANE — 32 private buffers, and the collective
// b_transpose_solve below would then operate on lane-private memory. The original even allocated a
// fresh `sparse_ei` inside the per-column loop.
template <bool Warp = false, bool Direct = false, class FT>
__device__ i_t initialize_steepest_edge_norms(const gpu_lp& lp,
                                              const i_t* basic_list,
                                              FT& ft,
                                              f_t* delta_y_steepest_edge,
                                              gpu_vector<i_t>& row_degree,
                                              gpu_vector<i_t>& mapping,
                                              gpu_vector<f_t>& coeff,
                                              gpu_sparse_vector& sparse_dy,
                                              gpu_sparse_vector& UTsol_scratch,
                                              gpu_sparse_vector& sparse_ei)
{
  const i_t m = lp.num_rows;
  lane0_serial<Direct>([&] {
    row_degree.assign(m, 0);
    mapping.assign(m, -1);
    coeff.assign(m, 0.0);
    for (i_t k = 0; k < m; ++k) {
      const i_t j         = basic_list[k];
      const i_t col_start = lp.A.col_start[j];
      const i_t col_end   = lp.A.col_start[j + 1];
      for (i_t p = col_start; p < col_end; ++p) {
        const i_t i   = lp.A.i[p];
        row_degree[i]++;
        mapping[k] = i;
        coeff[k]   = lp.A.x[p];
      }
    }
    sparse_dy.n = m;      sparse_dy.clear();
    UTsol_scratch.n = m;  UTsol_scratch.clear();
    sparse_ei.n = m;      sparse_ei.i.resize(1);  sparse_ei.x.resize(1);
  });

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    f_t init    = -1.0;
    // row_degree/mapping/coeff are global reads -> identical on every lane, so this branch is
    // warp-uniform and the collective below is entered by all 32 lanes or none.
    if (row_degree[mapping[k]] == 1) {
      const f_t alpha = coeff[k];
      init            = 1.0 / (alpha * alpha);
    } else {
      lane0_serial<Direct>([&] {
        sparse_ei.x[0] = -1.0;
        sparse_ei.i[0] = k;
        sparse_dy.clear();
      });
      ft.template b_transpose_solve<Warp, Direct>(sparse_ei, sparse_dy, UTsol_scratch);
      f_t my_init = 0.0;
      for (i_t p = 0; p < (i_t)sparse_dy.x.size(); ++p) {
        my_init += sparse_dy.x[p] * sparse_dy.x[p];
      }
      init = my_init;
    }
    // Same value from every lane to the same address -> benign. NOT lane0_serial: that would add a
    // __syncwarp per basic column (m of them, up to 25533 here) for no benefit.
    delta_y_steepest_edge[j] = init;
  }
  return 0;
}

// `rhs` is CALLER-OWNED for the same reason as above: a stack-local would be per-lane under Direct.
template <bool Warp = false, bool Direct = false, class FT>
__device__ void compute_primal_variables(const FT& ft,
                                         const gpu_span<f_t>& lp_rhs,
                                         const gpu_csc_view& A,
                                         const i_t* basic_list,
                                         const i_t* nonbasic_list,
                                         f_t tight_tol,
                                         f_t* x,
                                         gpu_vector<f_t>& xB_workspace,
                                         gpu_vector<f_t>& rhs)
{
  const i_t m = A.m;
  const i_t n = A.n;
  lane0_serial<Direct>([&] {
    rhs.assign(m, 0.0);
    for (i_t k = 0; k < m; ++k) { rhs[k] = lp_rhs[k]; }
    for (i_t k = 0; k < n - m; ++k) {
      const i_t j         = nonbasic_list[k];
      const i_t col_start = A.col_start[j];
      const i_t col_end   = A.col_start[j + 1];
      const f_t xj        = x[j];
      if (fabs(xj) < tight_tol * 10) continue;
      for (i_t p = col_start; p < col_end; ++p) {
        rhs[A.i[p]] -= xj * A.x[p];
      }
    }
    xB_workspace.resize(m);
  });

  ft.template b_solve<Warp, Direct>(rhs, xB_workspace);

  lane0_serial<Direct>([&] {
    for (i_t k = 0; k < m; ++k) {
      const i_t j = basic_list[k];
      x[j]        = xB_workspace[k];
    }
  });
}

template <bool Direct = false>
__device__ void clear_delta_z(
  i_t entering_index, i_t leaving_index, i_t* delta_z_mark, gpu_vector<i_t>& delta_z_indices, f_t* delta_z)
{
  const i_t nz = delta_z_indices.size();
  // delta_z_indices is a mark-guarded append list, so each column appears at most once -> the
  // destinations are distinct and the strided form is bit-exact.
  if constexpr (Direct) {
    const int lane = static_cast<int>(threadIdx.x);
    for (i_t k = lane; k < nz; k += 32) {
      const i_t j     = delta_z_indices[k];
      delta_z[j]      = 0.0;
      delta_z_mark[j] = 0;
    }
    __syncwarp(0xffffffffu);
  } else {
    for (i_t k = 0; k < nz; k++) {
      const i_t j     = delta_z_indices[k];
      delta_z[j]      = 0.0;
      delta_z_mark[j] = 0;
    }
  }
  // The two tail writes may duplicate an index already zeroed above, but both store 0.0, so the
  // overlap is idempotent and order-free. clear() is a header RMW -> exactly one lane.
  lane0_serial<Direct>([&] {
    if (entering_index != -1) { delta_z[entering_index] = 0.0; }
    delta_z[leaving_index] = 0.0;
    delta_z_indices.clear();
  });
}

template <bool Warp = false, bool Direct = false, class FT>
__device__ f_t compute_initial_primal_infeasibilities(const gpu_lp& lp,
                                                      const gpu_settings& settings,
                                                      const FT& ft,
                                                      const i_t* basic_list,
                                                      const f_t* x,
                                                      gpu_vector<f_t>& squared_infeasibilities,
                                                      gpu_vector<i_t>& infeasibility_indices,
                                                      f_t& primal_inf)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  if constexpr (Warp) {
    i_t pi_count = 0;
    f_t pi_pinf = 0.0, pi_pinf_sq = 0.0;
    if constexpr (Direct) {
      // reserve/set_size are header RMW (+ possible device malloc) -> lane 0; the __syncwarp inside
      // lane0_serial also publishes the (possibly relocated) data_ pointers before the collective.
      lane0_serial<true>([&] {
        squared_infeasibilities.reserve(n);
        squared_infeasibilities.set_size(n);  // no serial zero; the collective zeros [0,n)
        infeasibility_indices.reserve(n);
      });
      warp_pinf_body(basic_list, &pi_count, infeasibility_indices.data(), lp.lower.data(), m, n,
                     &pi_pinf, &pi_pinf_sq, squared_infeasibilities.data(), settings.primal_tol,
                     lp.upper.data(), x);
      // the body writes the three scalars on lane 0; every lane needs them (set_size is global).
      pi_count   = __shfl_sync(0xffffffffu, pi_count, 0);
      pi_pinf    = __shfl_sync(0xffffffffu, pi_pinf, 0);
      pi_pinf_sq = __shfl_sync(0xffffffffu, pi_pinf_sq, 0);
      lane0_serial<true>([&] { infeasibility_indices.set_size(pi_count); });
      primal_inf = pi_pinf;
      return pi_pinf_sq;
    }
  }
  squared_infeasibilities.resize(n);
  for (i_t k = 0; k < n; ++k) { squared_infeasibilities[k] = 0.0; }
  infeasibility_indices.reserve(n);
  infeasibility_indices.clear();
  f_t primal_inf_squared = 0.0;
  primal_inf             = 0.0;
  for (i_t k = 0; k < m; ++k) {
    const i_t j            = basic_list[k];
    const f_t lower_infeas = lp.lower[j] - x[j];
    const f_t upper_infeas = x[j] - lp.upper[j];
    const f_t infeas       = fmax(lower_infeas, upper_infeas);
    if (infeas > settings.primal_tol) {
      const f_t square_infeas    = infeas * infeas;
      squared_infeasibilities[j] = square_infeas;
      infeasibility_indices.push_back(j);
      primal_inf_squared += square_infeas;
      primal_inf += infeas;
    }
  }
  return primal_inf_squared;
}

__device__ i_t steepest_edge_pricing_with_infeasibilities(const gpu_lp& lp,
                                                          const gpu_settings& settings,
                                                          const f_t* x,
                                                          const f_t* dy_steepest_edge,
                                                          const i_t* basic_mark,
                                                          gpu_vector<f_t>& squared_infeasibilities,
                                                          gpu_vector<i_t>& infeasibility_indices,
                                                          i_t& direction,
                                                          i_t& basic_leaving,
                                                          f_t& max_val)
{
  max_val           = 0.0;
  i_t leaving_index = -1;
  const i_t nz      = infeasibility_indices.size();
  i_t max_count     = 0;
  for (i_t k = 0; k < nz; ++k) {
    const i_t j              = infeasibility_indices[k];
    const f_t squared_infeas = squared_infeasibilities[j];
    const f_t val            = squared_infeas / dy_steepest_edge[j];
    if (val > max_val || (val == max_val && j > leaving_index)) {
      max_val                = val;
      leaving_index          = j;
      const f_t lower_infeas = lp.lower[j] - x[j];
      const f_t upper_infeas = x[j] - lp.upper[j];
      direction              = lower_infeas >= upper_infeas ? 1 : -1;
      max_count++;
    }
  }
  basic_leaving = leaving_index >= 0 ? basic_mark[leaving_index] : -1;
  return leaving_index;
}

template <bool Warp = false, class FT>
__device__ i_t phase2_ratio_test(const gpu_lp& lp,
                                 const gpu_settings& settings,
                                 const FT& ft,
                                 const variable_status_t* vstatus,
                                 const i_t* nonbasic_list,
                                 const f_t* z,
                                 const f_t* delta_z,
                                 f_t& step_length,
                                 i_t& nonbasic_entering)
{
  i_t entering_index  = -1;
  const i_t n         = lp.num_cols;
  const i_t m         = lp.num_rows;
  const f_t pivot_tol = settings.pivot_tol;
  const f_t dual_tol  = settings.dual_tol / 10;
  const f_t zero_tol  = settings.zero_tol;
  f_t min_val         = dev_inf();

  // ============================================================================================
  // WARP RATIO TEST — DELIBERATELY NOT BIT-IDENTICAL TO THE SINGLE-THREAD SCAN. READ BEFORE EDITING.
  //
  // The two paths implement DIFFERENT tie-break rules. This is an algorithmic substitution, NOT a
  // floating-point artifact: the ratio expression is identical in both, and warp_reduce_min is exact
  // (min is associative and introduces no rounding).
  //
  //   serial (below, and the CPU reference) — WALKING Harris band. Note min_val is updated in the
  //     TIE-BREAK branch too, where ratio >= min_val, so accepting a near-tie RAISES the anchor by up
  //     to zero_tol. The band therefore drifts upward as the scan proceeds: whether a candidate is
  //     admitted depends on scan order and on which ties were already accepted.
  //
  //   warp (warp_collective_rt / warp_rt_body) — FIXED band. Pass 1 reduces the TRUE global minimum
  //     gmin over all eligible ratios; pass 2 admits [gmin, gmin+zero_tol], anchored and immovable.
  //
  //   Both then pick max |z[j]|, smallest k on an exact tie — that part DOES match.
  //
  // CONSEQUENCE: on near-degenerate ties the two can choose a different entering variable, hence a
  // different pivot and a divergent iteration path. Any GPU-vs-CPU bitwise check must therefore run
  // the SERIAL scan (CUOPT_RT_SERIAL=1); with the warp band the expected MPF signature is
  // 59/100 cpu==gpu, and that is NOT a bug.
  //
  // MEASURED, 100 v6 nodes, MPF/LU warp (claude_scripts/run_rt_band_ab.sh):
  //     parallel  40905 ms   3877 iters   98/100 match expected status   59/100 cpu==gpu
  //     serial    45896 ms   3870 iters   97/100 match expected status   64/100 cpu==gpu
  //   -> parallel is 1.122x FASTER (faster on 89/100 nodes) at essentially IDENTICAL iteration
  //      count (0.998x), and matches the recorded B&B outcome on one node MORE. No cycling.
  //   -> KEEP THE PARALLEL BAND ON MPF/LU.
  //
  // WHY THE INVERSE-ADD PATH DID NOT: there the fixed band was measured to CYCLE a degenerate node
  // (wachplan/node_d79) to the Q/P product-form cap — 4096 iters, NUMERICAL, 78 s — where the walking
  // band converges, so it kept the serial scan. The LU path has no such cap to hit, and on that very
  // node the parallel band is the one that gets CUTOFF (correct) while serial gets DUAL_UNBOUNDED.
  // (Original rationale: nvidia_cuopt/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu, pre-strip.)
  //
  // CUOPT_RT_SERIAL=1 forces the serial walking band even under Warp — used for bitwise checks and
  // to re-measure this trade on a new workload.
  // ============================================================================================
  // NOTE: the Warp arm that dispatched to warp_rt_body via ft.warp_dispatch_rt was removed with the
  // parked-helper layer (2026-08-11). Warp=true no longer reaches this function at all: the Direct
  // driver calls warp_rt_body inline (see gpu_simplified_dual_phase2_device) and fission K2 does the
  // same, so only the single-thread serial walking band below is live here. CUOPT_RT_SERIAL therefore
  // no longer has anything to switch on this path.

  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
      const f_t ratio = (-dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      } else if (ratio < min_val + zero_tol && fabs(z[j]) > fabs(z[entering_index])) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      }
    }
    if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
      const f_t ratio = (dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      } else if (ratio < min_val + zero_tol && fabs(z[j]) > fabs(z[entering_index])) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      }
    }
  }
  return entering_index;
}

template <bool Warp = false, bool Direct = false, class FT>
__device__ void compute_delta_y(const FT& ft,
                                i_t basic_leaving_index,
                                i_t direction,
                                gpu_sparse_vector& delta_y_sparse,
                                gpu_sparse_vector& UTsol_sparse)
{
  const i_t m = delta_y_sparse.n;
  gpu_sparse_vector& ei_sparse = ft.ei_scratch_;  // reused (was a per-pivot device malloc)
  lane0_serial<Direct>([&] {
    ei_sparse.n = m;
    ei_sparse.i.resize(1);   // may device-malloc on the first pivot -> lane 0 only
    ei_sparse.x.resize(1);
    ei_sparse.i[0] = basic_leaving_index;
    ei_sparse.x[0] = -direction;
  });
  ft.template b_transpose_solve<Warp, Direct>(ei_sparse, delta_y_sparse, UTsol_sparse);

  // `direction` is warp-uniform (read from st.direction), so the branch is uniform and the
  // __syncwarp count inside warp_negate matches across the warp.
  if (direction != -1) { UTsol_sparse.template warp_negate<Direct>(); }
}

template <bool Warp = false, bool Direct = false, class FT>
__device__ void update_dual_variables(const FT& ft,
                                     const gpu_sparse_vector& delta_y_sparse,
                                     const gpu_vector<i_t>& delta_z_indices,
                                     const f_t* delta_z,
                                     f_t step_length,
                                     i_t leaving_index,
                                     f_t* y,
                                     f_t* z)
{
  const i_t delta_y_nz = delta_y_sparse.i.size();
  if constexpr (Warp) {
    // warp_collective_upddual performs the FULL y/z update (both loops + the leaving term below), so
    // return here — mirroring the dispatch;return idiom of compute_delta_z / phase2_ratio_test / pinf.
    // Without this return the serial loops below would double-update y and z (2×step) on the warp path.
    if constexpr (Direct) {
      // fission: all lanes here; the body is internally lane-guarded (distinct-location writes
      // plus a lane-0 guard on z[leaving]) so no broadcast is needed.
      warp_upddual_body(delta_y_sparse.i.data(), static_cast<i_t>(delta_y_sparse.i.size()),
                        delta_y_sparse.x.data(), delta_z, delta_z_indices.data(),
                        static_cast<i_t>(delta_z_indices.size()), leaving_index, step_length, y, z);
    }
    return;
  }
  for (i_t k = 0; k < delta_y_nz; ++k) {
    const i_t i = delta_y_sparse.i[k];
    y[i] += step_length * delta_y_sparse.x[k];
  }
  const i_t delta_z_nz = delta_z_indices.size();
  for (i_t k = 0; k < delta_z_nz; ++k) {
    const i_t j = delta_z_indices[k];
    z[j] += step_length * delta_z[j];
  }
  z[leaving_index] += step_length * delta_z[leaving_index];
}

template <bool Warp = false, bool Direct = false, class FT>
__device__ i_t compute_delta_x(const gpu_lp& lp,
                               const FT& ft,
                               i_t entering_index,
                               i_t leaving_index,
                               i_t basic_leaving_index,
                               i_t direction,
                               const i_t* basic_list,
                               const gpu_sparse_vector& rhs_sparse,
                               const f_t* delta_z,
                               const f_t* x,
                               gpu_sparse_vector& utilde_sparse,
                               gpu_sparse_vector& scaled_delta_xB_sparse,
                               f_t* delta_x)
{
  f_t delta_x_leaving = direction == 1 ? lp.lower[leaving_index] - x[leaving_index]
                                       : lp.upper[leaving_index] - x[leaving_index];
  ft.template b_solve<Warp, Direct>(rhs_sparse, scaled_delta_xB_sparse, utilde_sparse);
  // warp_negate's trailing __syncwarp is what publishes the negated values to the other lanes
  // before find_coefficient below reads them on all 32.
  scaled_delta_xB_sparse.template warp_negate<Direct>();

  // find_coefficient is read-only, so every lane may compute `scale` itself — no guard needed.
  f_t scale     = scaled_delta_xB_sparse.find_coefficient(basic_leaving_index);
  i_t numerical = 0;
  if (scale != scale) {
    // Rare NaN-recovery branch. Under Direct it runs ENTIRELY on lane 0, with the inner b_solve
    // forced to the SERIAL (<<<,1>>>) instantiation: `rhs` / `scaled_delta_xB` are stack-local
    // gpu_vectors, so 32 lanes would each malloc a PRIVATE buffer and the collective would write
    // 32 different ones. Serial here is also exactly what the single-thread gate expects.
    lane0_serial<Direct>([&] {
      gpu_vector<f_t> rhs;
      rhs_sparse.to_dense(rhs);
      const i_t m = lp.num_rows;
      gpu_vector<f_t> scaled_delta_xB(m);
      ft.template b_solve<false, false>(rhs, scaled_delta_xB);
      if (scaled_delta_xB[basic_leaving_index] != 0.0 &&
          !isnan(scaled_delta_xB[basic_leaving_index])) {
        scaled_delta_xB_sparse.from_dense(scaled_delta_xB);
        scaled_delta_xB_sparse.negate();
        scale = -scaled_delta_xB[basic_leaving_index];
      } else if (delta_z[entering_index] != 0.0) {
        scale              = -delta_z[entering_index];
        bool found_leaving = false;
        for (i_t k = 0; k < static_cast<i_t>(scaled_delta_xB_sparse.i.size()); ++k) {
          if (scaled_delta_xB_sparse.i[k] == basic_leaving_index) {
            scaled_delta_xB_sparse.x[k] = scale;
            found_leaving               = true;
            break;
          }
        }
        if (!found_leaving) {
          scaled_delta_xB_sparse.i.push_back(basic_leaving_index);
          scaled_delta_xB_sparse.x.push_back(scale);
        }
      } else {
        numerical = 1;   // the monolith's `return -1` — taken uniformly below
      }
    });
    numerical = lane0_bcast<Direct>(numerical);
    if (numerical) { return -1; }
    scale = lane0_bcast<Direct>(scale);
  }
  lane0_serial<Direct>([&] {
    const f_t primal_step_length = delta_x_leaving / scale;
    const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
    for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
      const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
      delta_x[j]  = primal_step_length * scaled_delta_xB_sparse.x[k];
    }
    delta_x[leaving_index]  = delta_x_leaving;
    delta_x[entering_index] = primal_step_length;
  });
  return 0;
}

template <bool Warp = false, bool Direct = false, class FT>
__device__ i_t update_steepest_edge_norms(const gpu_settings& settings,
                                          const i_t* basic_list,
                                          const FT& ft,
                                          i_t direction,
                                          const gpu_sparse_vector& delta_y_sparse,
                                          f_t dy_norm_squared,
                                          const gpu_sparse_vector& scaled_delta_xB,
                                          i_t basic_leaving_index,
                                          i_t entering_index,
                                          f_t* v,
                                          gpu_sparse_vector& v_sparse,
                                          f_t* delta_y_steepest_edge)
{
  lane0_serial<Direct>([&] { v_sparse.clear(); });
  KT_TIC(_kt_ses);  // [KTIMER] se sub-split: time the b_solve (se_solve); se_rest = se - se_solve
  ft.template b_solve<Warp, Direct>(delta_y_sparse, v_sparse);
  KT_TOC(KT_SE_SOLVE, _kt_ses);
  // `direction` is warp-uniform. The scatter accumulates into the dense `v`, but v_sparse's indices
  // are distinct AND `v` is left all-zero by the v_nz loop at the end of every call below, so no two
  // lanes touch the same slot and no atomics are needed.
  if (direction == -1) { v_sparse.template warp_negate<Direct>(); }
  v_sparse.template warp_scatter<Direct>(v);

  // find_coefficient is read-only: every lane computes the same wr/omegar, so `return -1` is taken
  // uniformly and no broadcast is needed.
  const f_t wr = -scaled_delta_xB.find_coefficient(basic_leaving_index);
  if (wr == 0) { return -1; }
  const f_t omegar             = dy_norm_squared / (wr * wr);
  const i_t scaled_delta_xB_nz = scaled_delta_xB.i.size();
  KT_SDXB(scaled_delta_xB_nz);  // [KTIMER] accumulate scaled_delta_xB nnz (density confirmation)

  if constexpr (Warp) {
    // One lane per nonzero: distinct-location writes (see warp_collective_seupd) → bit-exact.
    if constexpr (Direct) {
      warp_seupd_body(basic_list, dy_norm_squared, delta_y_steepest_edge, entering_index,
                      basic_leaving_index, scaled_delta_xB_nz, omegar, scaled_delta_xB.i.data(),
                      scaled_delta_xB.x.data(), v, wr);
    }
  } else {
    for (i_t h = 0; h < scaled_delta_xB_nz; ++h) {
      const i_t k = scaled_delta_xB.i[h];
      const i_t j = basic_list[k];
      if (k == basic_leaving_index) {
        const f_t w                           = scaled_delta_xB.x[h];
        const f_t w_squared                   = w * w;
        delta_y_steepest_edge[entering_index] = (1.0 / w_squared) * dy_norm_squared;
      } else {
        const f_t wk = -scaled_delta_xB.x[h];
        f_t new_val  = delta_y_steepest_edge[j] + wk * (2.0 * v[k] / wr + wk * omegar);
        new_val      = fmax(new_val, 1e-4);
        delta_y_steepest_edge[j] = new_val;
      }
    }
  }

  // Restore `v` to all-zero for the next pivot. Distinct indices -> strided zeroing is bit-exact,
  // and this is precisely the invariant warp_scatter above relies on.
  {
    const i_t v_nz = v_sparse.i.size();
    if constexpr (Direct) {
      const int lane = static_cast<int>(threadIdx.x);
      for (i_t k = lane; k < v_nz; k += 32) { v[v_sparse.i[k]] = 0.0; }
      __syncwarp(0xffffffffu);
    } else {
      for (i_t k = 0; k < v_nz; ++k) { v[v_sparse.i[k]] = 0.0; }
    }
  }
  return 0;
}

// `x[j] += delta_x[j]` is a read-modify-write, but the destinations are DISTINCT: scaled_delta_xB's
// indices are unique, so basic_list[.i[k]] is unique too. Hence the strided form is bit-exact.
template <bool Direct = false>
__device__ void update_primal_variables(const gpu_sparse_vector& scaled_delta_xB_sparse,
                                        const i_t* basic_list,
                                        const f_t* delta_x,
                                        i_t entering_index,
                                        f_t* x)
{
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  if constexpr (Direct) {
    const int lane = static_cast<int>(threadIdx.x);
    for (i_t k = lane; k < scaled_delta_xB_nz; k += 32) {
      const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
      x[j] += delta_x[j];
    }
    // The entering column is NOT yet in basic_list (K5 does the swap), so this slot is disjoint from
    // every j above — but exactly one lane must own it.
    if (threadIdx.x == 0) { x[entering_index] += delta_x[entering_index]; }
    __syncwarp(0xffffffffu);
  } else {
    for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
      const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
      x[j] += delta_x[j];
    }
    x[entering_index] += delta_x[entering_index];
  }
}

// Every column j is independent: the branch chain reads only column-j data and writes only x[j] and
// vstatus[j], so no slot is touched twice and the strided form is bit-exact. The per-column body is
// a lambda so the two loop shapes share one copy of the chain.
template <bool Direct = false>
__device__ void set_primal_variables_on_bounds(const gpu_lp& lp,
                                               const gpu_settings& settings,
                                               const f_t* z,
                                               variable_status_t* vstatus,
                                               f_t* x)
{
  const i_t n = lp.num_cols;
  const f_t inf = dev_inf();
  f_t tol       = 1e-10;
  auto one_column = [&](i_t j) {
    if (vstatus[j] == variable_status_t::BASIC) { return; }
    const f_t fixed_tolerance = settings.fixed_tol;
    if (fabs(lp.lower[j] - lp.upper[j]) < fixed_tolerance) {
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_FIXED;
    } else if (z[j] >= -tol && lp.lower[j] > -inf &&
               vstatus[j] == variable_status_t::NONBASIC_LOWER) {
      x[j] = lp.lower[j];
    } else if (z[j] <= tol && lp.upper[j] < inf &&
               vstatus[j] == variable_status_t::NONBASIC_UPPER) {
      x[j] = lp.upper[j];
    } else if (z[j] >= 0 && lp.lower[j] > -inf) {
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
    } else if (z[j] <= 0 && lp.upper[j] < inf) {
      x[j]       = lp.upper[j];
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
    } else if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < 0) {
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > 0) {
      x[j]       = lp.upper[j];
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf) {
      x[j]       = 0;
      vstatus[j] = variable_status_t::NONBASIC_FREE;
    }
  };
  if constexpr (Direct) {
    const int lane = static_cast<int>(threadIdx.x);
    for (i_t j = lane; j < n; j += 32) { one_column(j); }
    __syncwarp(0xffffffffu);
  } else {
    for (i_t j = 0; j < n; ++j) { one_column(j); }
  }
}

// ---------------------------------------------------------------------------
// The device driver: a line-for-line copy of simplified_dual_phase2's
// pure-MPF-stacking path.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// slot_state : the PERSISTENT per-slot state of one solve.
//
// Every member here used to be a local in gpu_simplified_dual_phase2_device's stack frame, so it
// died when the driver returned and was published nowhere. Kernel fission splits the per-pivot
// pipeline across SEPARATE kernel launches, so anything that crosses a pivot-step boundary must
// outlive a single launch — it is owned here instead: one instance per batch slot (blockIdx.x),
// allocated once by the host and reused across super-steps AND across the successive B&B nodes
// that pass through the slot.
//
// It deliberately also owns `ft`. gpu_basis_update_mpf carries MUTABLE state that must survive
// across launches: the growing S_/mu_values_ rank-1 store, num_updates_, the reused scratch, and
// — subtlest — the hypersparse-vs-dense routing statistics sum_*_/num_calls_*_ that
// estimate_solution_density consumes and every solve updates. If those reset per launch the
// route decisions would drift from the monolith's, breaking both parity and the timing model.
//
// setup() is the one-time S1-S7 prologue, kept as its own method so the fissioned pipeline can
// later run it as a dedicated setup kernel for a freshly refilled slot.
// ---------------------------------------------------------------------------
template <class FT>
struct slot_state {
  bool inited{false};
  // Kernel fission: terminal state of this slot. dual::status_t::UNSET (8) == still running; any
  // other value means a kernel called fission_finish() and every later kernel must skip this slot.
  // Stored as i_t so the host can read it back as a plain int array.
  i_t status{static_cast<i_t>(dual::status_t::UNSET)};

  FT ft;  // basis update: S_/mu_values_, num_updates_, routing stats, reused scratch

  // Dense per-iteration workspaces.
  gpu_vector<f_t> xB_workspace, delta_y, delta_z, delta_x, v, squared_infeas;
  gpu_vector<i_t> basic_mark, delta_z_mark, delta_z_indices, nonbasic_end, row_mark, infeas_idx;

  // Sparse per-iteration workspaces.
  gpu_sparse_vector delta_y_sparse, UTsol_sparse, utilde_sparse, scaled_dxB_sparse, rhs_sparse,
    v_sparse;

  // CSR row copy of A + its incrementally maintained nonbasic partition.
  gpu_csr_matrix Arow;

  // ---- Kernel fission, WARP path: the elimination-level schedules (challenge #1, again) --------
  // In the monolith these are gpu_phase2_kernel STACK LOCALS (lvl_* + a __shared__ cfg), so they die
  // when the kernel exits — a fission kernel launched later would read freed memory. They therefore
  // live in the slot, built ONCE by Ksetup (which is <<<N,1>>>, i.e. exactly the monolith's lane 0)
  // and read by every warp fission kernel through ft.warp_cfg_. Unused (and unbuilt) when no
  // warp-mode kernel is enabled — see g_fission_warp.
  warp_solve_config wcfg;
  gpu_vector<i_t>      lvl_lev[kNumWarpSolveOps];   // per-column level (drives the sparse reach solve)
  gpu_vector<i_t>      lvl_ls[kNumWarpSolveOps];    // level offsets
  gpu_vector<i_t>      lvl_co[kNumWarpSolveOps];    // columns in level order
  gpu_vector<i_t>      sp_ls_v, sp_co_v, sp_pos_v;  // reach-restricted sparse-solve schedule scratch

  // Kernel fission: the marshaled device views, built ONCE by Ksetup and then read from here.
  // Previously every fission kernel rebuilt these on its stack (fission_bind), which forced ~79
  // registers' worth of pointers/spans/doubles live for the kernel's whole duration before any work
  // — the suspected floor under the measured 96-120. Living in the slot they are read lazily
  // (L1/L2-cached), so the compiler loads only the fields a given kernel touches and still hoists
  // the hot ones (e.g. A.col_start inside a solve loop) back into registers.
  gpu_lp       lp;
  gpu_settings settings;
  gpu_solution sol;

  // SETUP-ONLY scratch, slot-resident because under Direct a stack-local gpu_vector allocates PER
  // LANE. All start empty, so they cost nothing until the path that uses them actually runs (the
  // SE-init group only runs when the node carries no edge norms, i.e. dy_se_empty).
  gpu_vector<f_t>   setup_rhs;                       // compute_primal_variables
  gpu_vector<i_t>   se_row_degree, se_mapping;       // initialize_steepest_edge_norms
  gpu_vector<f_t>   se_coeff;
  gpu_sparse_vector se_dy, se_UTsol, se_ei;

  // Loop-carried scalars.
  f_t obj{0.0}, primal_inf{0.0};
  i_t iter{0};

  // Per-pivot scalars. Locals today (the monolith keeps them in registers); they become part of
  // the cross-kernel interface once the pipeline is split, so they live here from the start.
  i_t leaving{-1}, entering{-1}, direction{0}, basic_leaving{-1}, nonbasic_entering{-1};
  f_t step_length{0.0}, dy_norm_sq{0.0};
  bool used_transpose_dz{false};

  // One-time prologue: S1 bounds -> S2 primal -> S3 SE-init -> S4 workspaces -> S5 Arow ->
  // S6 nonbasic partition -> S7 primal infeasibilities. Byte-for-byte the setup the monolithic
  // driver used to run inline, in the same order (bit-exactness depends on that order).
  // Direct=true: Ksetup runs <<<N,32>>>. The prologue is almost entirely allocation and
  // read-modify-write (assign/reserve/clear, sparse-vector construction, Arow build) so it stays on
  // lane 0; the collectives are opted in one at a time. The serial parts deliberately call the
  // <false> variants so porting one collective cannot silently move any other computation.
  template <bool Warp, bool Direct = false>
  __device__ void setup(const gpu_lp& lp,
                        const gpu_settings& settings,
                        variable_status_t* vstatus,
                        i_t* basic_list,
                        i_t* nonbasic_list,
                        gpu_solution& sol,
                        f_t* delta_y_steepest_edge,
                        bool dy_se_empty)
  {
    const i_t m = lp.num_rows;
    const i_t n = lp.num_cols;
    f_t* x      = sol.x.data();
    f_t* z      = sol.z.data();

    // assign (not resize) throughout: a slot is REUSED by the next B&B node, so every buffer must be
    // fully re-initialized, not just grown — resize would leave the previous node's tail values.
    // assign() may device-malloc, so it stays on lane 0; the O(n) bound-setting scan next to it does
    // not, so the two are split rather than sharing one lane-0 region.
    lane0_serial<Direct>([&] { xB_workspace.assign(m, 0.0); });

    set_primal_variables_on_bounds<Direct>(lp, settings, z, vstatus, x);

    // Both of these solve through the factors, so they follow the kernel's geometry — the monolith
    // has always run them on the warp, and fission now does too via Direct.
    compute_primal_variables<Warp, Direct>(
      ft, lp.rhs, lp.A, basic_list, nonbasic_list, settings.tight_tol, x, xB_workspace, setup_rhs);

    if (dy_se_empty) {
      // Exact steepest-edge init (matches the CPU simplified's unconditional exact path).
      lane0_serial<Direct>([&] {
        for (i_t k = 0; k < n; ++k) { delta_y_steepest_edge[k] = -1.0; }
      });
      initialize_steepest_edge_norms<Warp, Direct>(lp, basic_list, ft, delta_y_steepest_edge,
                                                   se_row_degree, se_mapping, se_coeff, se_dy,
                                                   se_UTsol, se_ei);
    } else {
      // basic_list holds m DISTINCT column indices, so each delta_y_steepest_edge[j] is visited by
      // exactly one lane — bit-exact.
      if constexpr (Direct) {
        const int lane = static_cast<int>(threadIdx.x);
        for (i_t k = lane; k < m; k += 32) {
          const i_t j = basic_list[k];
          if (delta_y_steepest_edge[j] <= 0.0) { delta_y_steepest_edge[j] = 1e-4; }
        }
        __syncwarp(0xffffffffu);
      } else {
        for (i_t k = 0; k < m; ++k) {
          const i_t j = basic_list[k];
          if (delta_y_steepest_edge[j] <= 0.0) { delta_y_steepest_edge[j] = 1e-4; }
        }
      }
    }

    lane0_serial<Direct>([&] {

    delta_y.assign(m, 0.0);
    delta_z.assign(n, 0.0);
    delta_x.assign(n, 0.0);
    v.assign(m, 0.0);
    basic_mark.assign(n, -1);
    });

    // basic_list holds m DISTINCT column indices, so no two lanes write the same slot — bit-exact.
    // data() is read AFTER the lane-0 region above because assign() may have relocated the buffer.
    {
      i_t* bm = basic_mark.data();
      if constexpr (Direct) {
        const int lane = static_cast<int>(threadIdx.x);
        for (i_t k = lane; k < m; k += 32) { bm[basic_list[k]] = k; }
        __syncwarp(0xffffffffu);
      } else {
        for (i_t k = 0; k < m; ++k) { bm[basic_list[k]] = k; }
      }
    }

    lane0_serial<Direct>([&] {
    delta_z_mark.assign(n, 0);
    delta_z_indices.clear();
    delta_z_indices.reserve(n);

    // Transpose delta_z state (Arow CSR copy of A + per-row nonbasic partition). Built
    // once here; nonbasic_end is maintained per pivot by update_Arow (no refactor rebuild
    // needed — the GPU is pure MPF stacking).
    if (lp.Arow_row_start != nullptr) {
      // Zero-copy view over the host-built Arow (uploaded per-solve, mutable) — skips the on-device
      // from_csc transpose. Always used when present (was CUOPT_GPU_HOST_AROW, now default).
      Arow.view_device(lp.Arow_row_start, lp.Arow_j, lp.Arow_x, m, n, lp.A.col_start[n]);
    } else {
      // Arow not uploaded (the RO-shared batch path leaves it null) → build on device off the shared A.
      Arow.from_csc(lp.A);
    }
    nonbasic_end.assign(m, 0);
    row_mark.clear();
    row_mark.reserve(m);
    });  // end lane-0 prologue

    // WARP_NBE: one lane per Arow row. Bit-exact (disjoint slices), 0 internal barriers.
    compute_initial_nonbasic_end<Warp, Direct>(ft, basic_mark.data(), Arow, nonbasic_end.data());

    lane0_serial<Direct>([&] {
      squared_infeas.clear();
      infeas_idx.clear();
      primal_inf = 0.0;
    });
    // WARP_PINF: zero squared_infeas[0,n), scan the m basics one lane per k, ordered ballot/popc
    // compaction of the infeasible indices -> bit-exact on both consumed outputs.
    compute_initial_primal_infeasibilities<Warp, Direct>(
      lp, settings, ft, basic_list, x, squared_infeas, infeas_idx, primal_inf);

    lane0_serial<Direct>([&] {
    delta_y_sparse    = gpu_sparse_vector(m, 0);
    UTsol_sparse      = gpu_sparse_vector(m, 0);  // MPF-only: Uᵀ intermediate from compute_delta_y
    utilde_sparse     = gpu_sparse_vector(m, 0);  // MPF-only: b_solve intermediate from compute_delta_x
    scaled_dxB_sparse = gpu_sparse_vector(m, 0);
    rhs_sparse        = gpu_sparse_vector(m, 0);
    v_sparse          = gpu_sparse_vector(m, 0);

    // Incrementally-tracked objective for the cutoff, mirroring the CPU simplified (and
    // advanced) driver: init fresh once, then update_objective per pivot. Using the true
    // costs lp.objective (the GPU has no perturbation) is a valid dual bound.
    obj  = compute_objective(lp, x);
    iter = 0;
    });  // end lane-0 epilogue
  }
};

// `Direct`: see the KERNEL FISSION / Direct banner above. Direct=false is the parked-helper monolith
// (lane 0 alone runs this body); Direct=true means ALL 32 lanes run it, so every allocation and
// read-modify-write must be inside lane0_serial<Direct> and every `return` must be warp-uniform.
template <bool Warp = false, bool Direct = false, class FT>
__device__ dual::status_t gpu_simplified_dual_phase2_device(const gpu_lp& lp,
                                                            const gpu_settings& settings,
                                                            variable_status_t* vstatus,
                                                            slot_state<FT>& st,
                                                            i_t* basic_list,
                                                            i_t* nonbasic_list,
                                                            gpu_solution& sol,
                                                            f_t* delta_y_steepest_edge,
                                                            bool dy_se_empty)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  f_t* x = sol.x.data();
  f_t* y = sol.y.data();
  f_t* z = sol.z.data();

  // One-time prologue into the persistent slot. Already-inited slots (a fissioned super-step
  // re-entering mid-solve) skip straight to the pivot loop.
  if (!st.inited) {
    st.template setup<Warp, Direct>(
      lp, settings, vstatus, basic_list, nonbasic_list, sol, delta_y_steepest_edge, dy_se_empty);
    lane0_serial<Direct>([&] { st.inited = true; });   // RMW on slot state
  }
  KT_SETUP_DONE();  // [KTIMER] one-time setup (kernel entry → first pivot) done

  // Bind the persistent slot members under their original names, so the pivot loop below is
  // unchanged by the state externalization and stays line-diffable against the CPU simplified
  // driver (dual_simplex/CLAUDE.md rule 2).
  FT& ft                          = st.ft;
  gpu_vector<f_t>& delta_y        = st.delta_y;
  gpu_vector<f_t>& delta_z        = st.delta_z;
  gpu_vector<f_t>& delta_x        = st.delta_x;
  gpu_vector<f_t>& v              = st.v;
  gpu_vector<f_t>& squared_infeas = st.squared_infeas;
  gpu_vector<i_t>& basic_mark     = st.basic_mark;
  gpu_vector<i_t>& delta_z_mark   = st.delta_z_mark;
  gpu_vector<i_t>& delta_z_indices = st.delta_z_indices;
  gpu_vector<i_t>& nonbasic_end   = st.nonbasic_end;
  gpu_vector<i_t>& row_mark       = st.row_mark;
  gpu_vector<i_t>& infeas_idx     = st.infeas_idx;
  gpu_sparse_vector& delta_y_sparse    = st.delta_y_sparse;
  gpu_sparse_vector& UTsol_sparse      = st.UTsol_sparse;
  gpu_sparse_vector& utilde_sparse     = st.utilde_sparse;
  gpu_sparse_vector& scaled_dxB_sparse = st.scaled_dxB_sparse;
  gpu_sparse_vector& rhs_sparse        = st.rhs_sparse;
  gpu_sparse_vector& v_sparse          = st.v_sparse;
  gpu_csr_matrix& Arow                 = st.Arow;
  f_t& obj        = st.obj;
  f_t& primal_inf = st.primal_inf;
  i_t& iter       = st.iter;

  // Every terminal exit writes the same three fields. Under Direct the O(n) objective sum and the
  // stores go on lane 0; every caller of this returns IMMEDIATELY afterwards, and does so uniformly
  // (all terminal conditions below are warp-uniform), so the trailing __syncwarp is safe.
  auto finish = [&](i_t iters) {
    lane0_serial<Direct>([&] {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iters;
    });
  };

  const i_t iter_limit = min(500000, max(20000, 20 * (m + n)));
  while (iter < iter_limit) {
    if (g_gpu_alloc_failed) {
      finish(iter);
      return dual::status_t::NUMERICAL;
    }
    // Time limit is handled on the host side

    // 1. Steepest-edge pricing.
    i_t direction        = 0;
    i_t basic_leaving    = -1;
    f_t max_val;
    KT_TIC(_kt_pr);
    const i_t leaving = steepest_edge_pricing_with_infeasibilities(
      lp, settings, x, delta_y_steepest_edge, basic_mark.data(), squared_infeas, infeas_idx,
      direction, basic_leaving, max_val);
    KT_TOC(KT_PRICING, _kt_pr);
    if (leaving == -1) {
      finish(iter);
      return dual::status_t::OPTIMAL;
    }

    // 2. BTran.
    lane0_serial<Direct>([&] {   // clear() is a header RMW on size_
      delta_y_sparse.clear();
      UTsol_sparse.clear();
    });
    KT_TIC(_kt_bt);
    compute_delta_y<Warp, Direct>(ft, basic_leaving, direction, delta_y_sparse, UTsol_sparse);
    KT_TOC(KT_BTRAN, _kt_bt);
    // Butterfly reduction of ||delta_y||^2 under Direct. Reordered vs the serial sum, and
    // dy_norm_sq feeds the SE norms -> pricing, so this is a deliberate arithmetic change.
    // warp_reduce_add returns the FULL sum on every lane, so dy_norm_sq stays warp-uniform and the
    // reprice branch below is taken uniformly.
    f_t dy_norm_sq;
    if constexpr (Direct) {
      const int lane = static_cast<int>(threadIdx.x);
      const f_t* dyx = delta_y_sparse.x.data();
      const i_t nz   = (i_t)delta_y_sparse.i.size();   // trip count as in norm2_squared()
      f_t part = 0.0;
      for (i_t k = lane; k < nz; k += 32) { part += dyx[k] * dyx[k]; }
      dy_norm_sq = warp_reduce_add(part);
    } else {
      dy_norm_sq = delta_y_sparse.norm2_squared();
    }

    // Steepest-edge norm drift safeguard: dy_norm_sq is the true norm of the leaving
    // row. If the maintained norm used for pricing has drifted below
    // settings.steepest_edge_ratio of it, the leaving choice was made on a stale norm --
    // reset it and re-price (strict port of the CPU simplified, phase2.cpp:4037).
    // g_serep_off (host: CUOPT_SEREP_OFF) disables it, matching the CPU driver's switch, so the
    // monolith can serve as the bit-exact reference for the reprice-free fissioned pipeline.
    if (!g_serep_off && delta_y_steepest_edge[leaving] < settings.steepest_edge_ratio * dy_norm_sq) {
      lane0_serial<Direct>([&] { delta_y_steepest_edge[leaving] = dy_norm_sq; });
      continue;
    }

    // 3. Reduced-cost SpMv: transpose (CSR) when delta_y is sparse (<=30% dense), else
    //    dense CSC. Mirrors the CPU driver.
    i_t delta_y_nz0      = 0;
    const i_t nz_delta_y = (i_t)delta_y_sparse.i.size();
    if constexpr (Direct) {
      // INTEGER count, so ballot/popc is EXACT — nothing is reordered the way an FP sum would be.
      // Every lane accumulates the same running total, keeping delta_y_nz0 (and the
      // used_transpose_dz branch it gates) warp-uniform.
      const int lane = static_cast<int>(threadIdx.x);
      const f_t* dyx = delta_y_sparse.x.data();
      for (i_t base = 0; base < nz_delta_y; base += 32) {
        const i_t k    = base + lane;
        const bool hit = (k < nz_delta_y) && (fabs(dyx[k]) > 1e-12);
        delta_y_nz0 += __popc(__ballot_sync(0xffffffffu, hit));
      }
    } else {
      for (i_t k = 0; k < nz_delta_y; ++k) {
        if (fabs(delta_y_sparse.x[k]) > 1e-12) { delta_y_nz0++; }
      }
    }
    const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
    const bool used_transpose_dz    = delta_y_nz_percentage <= 30.0;
    if (used_transpose_dz) {
      KT_TIC(_kt_dz);
      compute_delta_z<Warp, Direct>(ft, Arow, delta_y_sparse, leaving, direction,
                                    nonbasic_end.data(), delta_z_mark.data(), delta_z_indices,
                                    delta_z.data());
      KT_TOC(KT_DZ, _kt_dz);
    } else {
      // to_dense RESIZES delta_y (header RMW + possible device malloc). warp_to_dense keeps the
      // sizing on lane 0 and warps the O(n) zero-fill + nnz scatter — bit-exact (distinct indices).
      if constexpr (Warp) {
        ft.template warp_to_dense<Direct>(delta_y_sparse, delta_y);
      } else {
        delta_y_sparse.to_dense(delta_y);
      }
      KT_TIC(_kt_rcu);
      compute_reduced_cost_update<Warp, Direct>(lp, ft, basic_list, nonbasic_list, delta_y.data(),
                                                leaving, direction, delta_z_mark.data(),
                                                delta_z_indices, delta_z.data());
      KT_TOC(KT_RCU, _kt_rcu);
    }

    // 4. Simple ratio test.
    f_t step_length       = 0.0;
    i_t nonbasic_entering = -1;
    i_t entering          = -1;
    KT_TIC(_kt_rt);
    if constexpr (Direct) {
      // Same body the dispatcher already invokes (warp_dispatch_rt -> warp_rt_body), so this is a
      // pure plumbing change, not a tie-break change. Its outputs are lane-0 pointer writes, so aim
      // them at slot_state (global) and let the __syncwarp publish them to every lane — locals would
      // only ever be set on lane 0.
      warp_rt_body(settings.dual_tol / 10, delta_z.data(), &st.entering, &st.nonbasic_entering,
                   lp.num_rows, lp.num_cols, nonbasic_list, settings.pivot_tol, &st.step_length,
                   vstatus, z, settings.zero_tol);
      __syncwarp(0xffffffffu);
      entering          = st.entering;
      nonbasic_entering = st.nonbasic_entering;
      step_length       = st.step_length;
    } else {
      entering = phase2_ratio_test<Warp>(
        lp, settings, ft, vstatus, nonbasic_list, z, delta_z.data(), step_length, nonbasic_entering);
    }
    KT_TOC(KT_RT, _kt_rt);
    if (entering == -1) {
      finish(iter);
      return dual::status_t::DUAL_UNBOUNDED;
    }

    KT_TIC(_kt_ud);
    update_dual_variables<Warp, Direct>(
      ft, delta_y_sparse, delta_z_indices, delta_z.data(), step_length, leaving, y, z);
    KT_TOC(KT_UPDDUAL, _kt_ud);

    // FTran.
    lane0_serial<Direct>([&] {   // clear() is a header RMW; from_csc_column reserve+push_back mallocs
      utilde_sparse.clear();
      scaled_dxB_sparse.clear();
      rhs_sparse.from_csc_column(lp.A, entering);
    });
    KT_TIC(_kt_ft);
    i_t dx_status =
      compute_delta_x<Warp, Direct>(lp, ft, entering, leaving, basic_leaving, direction, basic_list, rhs_sparse,
                      delta_z.data(), x, utilde_sparse, scaled_dxB_sparse, delta_x.data());
    KT_TOC(KT_FTRAN, _kt_ft);
    if (dx_status == -1) {
      finish(iter);
      return dual::status_t::NUMERICAL;
    }

    KT_TIC(_kt_se);
    update_steepest_edge_norms<Warp, Direct>(settings, basic_list, ft, direction, delta_y_sparse, dy_norm_sq,
                               scaled_dxB_sparse, basic_leaving, entering, v.data(), v_sparse,
                               delta_y_steepest_edge);
    KT_TOC(KT_SE, _kt_se);

    KT_TIC(_kt_up);
    update_primal_variables<Direct>(scaled_dxB_sparse, basic_list, delta_x.data(), entering, x);
    KT_TOC(KT_UPDPRIM, _kt_up);

    // Incrementally update the objective the SAME way the CPU driver does (phase2.cpp:4259):
    // obj += delta_x·objective over the changed basics + entering, true costs, called with the
    // OLD basic_list (before the swap below), WITHOUT the bound-flip contribution.
    KT_TIC(_kt_ob);
    update_objective<Direct>(basic_list, scaled_dxB_sparse, lp.objective, delta_x.data(), entering, obj);
    KT_TOC(KT_OBJ, _kt_ob);

    // 5. Basis swap + MPF rank-1 stacking update.
    // Every line here is a read-modify-write on shared state (vstatus/basic_list/basic_mark stores,
    // update_Arow's in-place row repartition + row_mark push_back, ft.update's S_ append + sort +
    // allocations), so under Direct the whole block is lane 0's.
    i_t _fu_status = 0;
    lane0_serial<Direct>([&] {
      vstatus[entering] = variable_status_t::BASIC;
      vstatus[leaving]  = (lp.lower[leaving] == lp.upper[leaving])
                           ? variable_status_t::NONBASIC_FIXED
                           : static_cast<variable_status_t>(-direction);
      basic_list[basic_leaving]        = entering;
      nonbasic_list[nonbasic_entering] = leaving;
      basic_mark[leaving]              = -1;
      basic_mark[entering]             = basic_leaving;

      // Keep the CSR nonbasic partition in sync for the next iteration's transpose delta_z.
      KT_TIC(_kt_ar);
      update_Arow(leaving, entering, lp.A, row_mark, nonbasic_end.data(), Arow);
      KT_TOC(KT_AROW, _kt_ar);

      KT_TIC(_kt_fu);
      _fu_status = ft.update(utilde_sparse, UTsol_sparse, basic_leaving);
      KT_TOC(KT_FTUPD, _kt_fu);
    });
    // The NUMERICAL exit must be taken UNIFORMLY, so broadcast lane 0's verdict.
    _fu_status = lane0_bcast<Direct>(_fu_status);
    if (_fu_status == 1) {
      finish(iter);
      return dual::status_t::NUMERICAL;
    }

    KT_TIC(_kt_pi);
    compute_initial_primal_infeasibilities<Warp, Direct>(
      lp, settings, ft, basic_list, x, squared_infeas, infeas_idx, primal_inf);
    KT_TOC(KT_PINF, _kt_pi);
    KT_TIC(_kt_cd);
    clear_delta_z<Direct>(entering, leaving, delta_z_mark.data(), delta_z_indices, delta_z.data());
    KT_TOC(KT_CLEARDZ, _kt_cd);

    // Increment only on a COMPLETED iteration, at the end of the body — matching the CPU
    // simplified driver's iter++ (phase2.cpp:4370). The SE-reset `continue` above jumps
    // back to `while` without counting, so `iter` counts pivots, not loop passes.
    lane0_serial<Direct>([&] { iter++; });   // RMW on st.iter in global memory

    // Objective cutoff (dual bound), checked AFTER the pivot/iter++ — mirrors the CPU
    // simplified driver's placement + comparand (phase2.cpp:4377): the incrementally
    // tracked objective `obj` (true costs, omits bound flips — a valid, conservative lower
    // bound). Gated on a finite cutoff (default +inf skips it).
    if (isfinite(settings.cut_off)) {
      const f_t cutoff_obj = obj;
      if (cutoff_obj >= settings.cut_off) {
        finish(iter);
        return dual::status_t::CUTOFF;
      }
    }
  }
  finish(iter_limit);
  return dual::status_t::ITERATION_LIMIT;
}

// ---------------------------------------------------------------------------
// Kernel argument bundle (all raw device pointers + scalars, passed by value).
// ---------------------------------------------------------------------------
struct kernel_args {
  i_t m, n;
  // LP
  i_t *A_col_start, *A_i;
  f_t* A_x;
  i_t A_nz;
  // Host-built CSR of A (Arow); null when not provided (device builds it via from_csc).
  i_t *Arow_row_start, *Arow_j;
  f_t* Arow_x;
  f_t *rhs, *lower, *upper, *objective;
  f_t obj_scale, obj_constant;
  // factorization
  i_t *L0_col_start, *L0_i;
  f_t* L0_x;
  i_t L0_nz;
  i_t *U0_col_start, *U0_i;
  f_t* U0_x;
  i_t U0_nz;
  i_t *L0t_col_start, *L0t_i;
  f_t* L0t_x;
  i_t L0t_nz;
  i_t *U0t_col_start, *U0t_i;
  f_t* U0t_x;
  i_t U0t_nz;
  i_t *row_perm, *inv_row_perm;
  i_t* xi_workspace;
  f_t* x_workspace;
  i_t num_updates;
  i_t refactor_frequency;
  // Per-slot PERSISTENT solve state (slot_state<FT>): host-allocated, ZEROED blob that the kernel
  // placement-news into on first entry and reuses thereafter. This is what lets the per-pivot
  // workspaces + ft's mutable state (S_, routing stats, scratch) outlive a single kernel launch —
  // the precondition for splitting the driver into per-primitive kernels. `slot_live` is the
  // host-visible "already constructed" flag (0 = fresh node → construct + run setup).
  void* slot;
  i_t   slot_live;
  // Kernel fission: contiguous host-visible status array (one i_t per slot, indexed by blockIdx.x).
  // Same base pointer in every item, so the host reads all N terminal states in ONE D2H copy to
  // decide when the super-step loop is done. Null on the monolith path.
  i_t* status_arr;
  // state (mutable)
  variable_status_t* vstatus;
  i_t* basic_list;
  i_t* nonbasic_list;
  f_t *x, *y, *z;
  f_t* dy_se;
  i_t dy_se_empty;
  // settings
  f_t primal_tol, dual_tol, pivot_tol, tight_tol, fixed_tol, zero_tol, time_limit, cut_off,
    steepest_edge_ratio;
  f_t start_time;
  // outputs
  i_t* out_status;
  f_t* out_objective;
  f_t* out_user_objective;
  i_t* out_iterations;
};

__device__ gpu_csc_view make_view(
  i_t rows, i_t cols, i_t nz, i_t* col_start, i_t* idx, f_t* val)
{
  gpu_csc_view v;
  v.m         = rows;
  v.n         = cols;
  v.nz_max    = nz;
  v.col_start = gpu_span<i_t>(col_start, cols + 1);
  v.i         = gpu_span<i_t>(idx, nz);
  v.x         = gpu_span<f_t>(val, nz);
  return v;
}


// Fill the warp-solve config for all 4 operators (lane 0, once at start). The
// level schedules are written into the caller-owned gpu_vectors ls[o]/co[o]
// (kept alive for the whole kernel); `lvlscratch` is reused m-sized workspace.
__device__ void warp_build_config(const gpu_basis_update_mpf& ft, warp_solve_config& cfg,
                                   gpu_vector<i_t>* lev, gpu_vector<i_t>* ls, gpu_vector<i_t>* co,
                                   gpu_vector<i_t>& sp_ls_v, gpu_vector<i_t>& sp_co_v,
                                   gpu_vector<i_t>& sp_pos_v)
{
  // Reach-restricted sparse-solve scratch (built fresh per solve in warp_dispatch_sptrsv): m(+1)-sized.
  const i_t m = ft.L0_.m;
  sp_co_v.resize(m);
  sp_ls_v.resize(m + 1);
  sp_pos_v.resize(m + 1);
  cfg.sp_co  = sp_co_v.data();
  cfg.sp_ls  = sp_ls_v.data();
  cfg.sp_pos = sp_pos_v.data();
  // op order: L, U, Lᵀ, Uᵀ (see warp_solve_op). Gather matrix + flags per op.
  // Indexed by warp_solve_op: {L, U, Lᵀ, Uᵀ}.
  const gpu_csc_view* mats[kNumWarpSolveOps] = {&ft.L0_transpose_, &ft.U0_transpose_, &ft.L0_, &ft.U0_};
  // df/dsc/inc come from the SHARED op_* constexpr functions so the schedule built here and the four
  // specialized wavefront bodies can never disagree about which solve an op means.
  const bool df[kNumWarpSolveOps]  = {op_diag_first(0),  op_diag_first(1),  op_diag_first(2),  op_diag_first(3)};
  const bool asc[kNumWarpSolveOps] = {true, false, false, true};   // level pull processes j ascending?
  for (int o = 0; o < kNumWarpSolveOps; ++o) {
    cfg.cs[o]         = mats[o]->col_start.data();
    cfg.ridx[o]       = mats[o]->i.data();
    cfg.val[o]        = mats[o]->x.data();
    // df/dsc/inc are no longer stored: the flags are compile-time (op_diag_first/op_descending/
    // op_incremental) at every call site. df[o] is still needed here to BUILD the level schedule.
    cfg.nlev[o]  = warp_build_levels(*mats[o], df[o], asc[o], lev[o], ls[o], co[o]);
    cfg.lstart[o]     = ls[o].data();
    cfg.corder[o]     = co[o].data();
    cfg.level[o]      = lev[o].data();  // per-column level, retained for the reach-restricted sparse solve
  }
}

// One block == one problem. args points to an N-entry device array; block b solves
// args[blockIdx.x] entirely on its own device buffers. Launched <<<1,1>>> for the
// single-thread reference (Warp=false) and <<<1,32>>> for the warp-optimized path
// (Warp=true, lane 0 drives, lanes 1..31 run the collective wavefront solves).
// g_gpu_alloc_failed is reset by the HOST before launch (not here) so a late block
// can't clear a failure flag an earlier block already set.
// MinBlocks = the __launch_bounds__ minBlocksPerMultiprocessor. ptxas caps registers at
// 65536/(blockDim*MinBlocks): warp path (blockDim 32) → MinBlocks 8 = 256-reg budget (default, the
// kernel's natural 255 regs, ~8 resident blocks/SM on B200), 16 = 128-reg cap (~16 blocks, some
// spilling), 32 = 64-reg cap (~32 blocks = the HW max, heavier spilling). The batch launcher picks the
// variant at runtime (CUOPT_GPU_REGCAP) so throughput-vs-register-pressure can be swept in one binary.
// `Direct`: the fission execution model applied to the monolith. Direct=false is the historical
// parked-helper form (lane 0 drives, lanes 1-31 wait in warp_helper_loop for cfg_shared requests);
// Direct=true means all 32 lanes run the driver and every collective is called directly. Only ever
// meaningful together with Warp.
// MinBlocks is an OCCUPANCY TARGET for the Direct warp kernel, reinstated 2026-08-11 after measuring
// that ptxas's unconstrained choice (182 regs, 11 blocks/SM) is a heuristic, not a requirement:
// a cap sweep showed 168 regs at ZERO spill and 128 regs at 4 BYTES of spill (16 blocks/SM, 1.45x).
// ptxas caps registers at 65536/(32*MinBlocks): 8 -> 256 (no cap), 12 -> 170, 16 -> 128, 32 -> 64.
// Applied ONLY to the Direct warp kernel — the parked-helper and single-thread forms keep ptxas's
// own heuristic, which is what the 2026-08-11 __launch_bounds__ removal established as best there.
template <bool Warp, bool Direct, class FT>
__device__ __forceinline__ void gpu_phase2_kernel_body(kernel_args* args)
{
  // NOTE: the CUDA 13 `.pragma "enable_smem_spilling"` (spill regs to shared instead of local) is
  // INCOMPATIBLE here — ptxas fails with "(C7800) Smem spilling should not be enabled when functions
  // use abi": this kernel makes ABI (non-inlined) device calls into the whole driver, and the pragma
  // requires a fully-inlined call tree. Not usable without force-inlining everything.
  // Reject a launch geometry that doesn't match the compiled path (the whole point
  // of the compile-time Warp parameter: no runtime branch selects opt vs unopt).
  if constexpr (Warp) {
    if (blockDim.x != 32) { assert(false); return; }
  } else {
    if (blockDim.x != 1) { assert(false); return; }
  }

  // Shared config. Under Direct it is READ-ONLY after the one-time build below: the Direct arms
  // still fetch the elimination-level schedule through ft.warp_cfg_, only the per-dispatch REQUEST
  // fields are unused. Unused entirely when Warp==false.
  __shared__ warp_solve_config cfg_shared;

  // No parked-helper model any more: on the warp path all 32 lanes fall through and run the driver
  // (Direct). The lanes-1..31 -> warp_helper_loop split was deleted with the dispatcher.

  kernel_args& a = args[blockIdx.x];
  KT_RESET();  // [KTIMER] start-of-solve: zero accumulators + record kernel-entry clock (lane 0)

  const i_t m = a.m;
  const i_t n = a.n;

  gpu_lp lp;
  lp.num_rows     = m;
  lp.num_cols     = n;
  lp.A            = make_view(m, n, a.A_nz, a.A_col_start, a.A_i, a.A_x);
  lp.Arow_row_start = a.Arow_row_start;
  lp.Arow_j         = a.Arow_j;
  lp.Arow_x         = a.Arow_x;
  lp.rhs          = gpu_span<f_t>(a.rhs, m);
  lp.lower        = gpu_span<f_t>(a.lower, n);
  lp.upper        = gpu_span<f_t>(a.upper, n);
  lp.objective    = gpu_span<f_t>(a.objective, n);
  lp.obj_scale    = a.obj_scale;
  lp.obj_constant = a.obj_constant;

  gpu_settings settings;
  settings.primal_tol         = a.primal_tol;
  settings.dual_tol           = a.dual_tol;
  settings.pivot_tol          = a.pivot_tol;
  settings.tight_tol          = a.tight_tol;
  settings.fixed_tol          = a.fixed_tol;
  settings.zero_tol           = a.zero_tol;
  settings.time_limit         = a.time_limit;
  settings.cut_off             = a.cut_off;
  settings.steepest_edge_ratio = a.steepest_edge_ratio;
  settings.refactor_frequency  = a.refactor_frequency;

  // Per-slot PERSISTENT state (see slot_state). The host hands us a zeroed blob sized for
  // slot_state<FT>; construct it in place on first entry, then wire the read-only factorization
  // views into its ft. `slot_live` is the host's per-slot "already constructed" flag: it stays 0
  // for a fresh node (construct + run setup) and is set to 1 below, so a later fissioned
  // super-step re-entering the same slot reuses the live state (S_, routing stats, workspaces)
  // instead of rebuilding it.
  slot_state<FT>& st = *static_cast<slot_state<FT>*>(a.slot);
  // One-time construction: placement-new, ft wiring and S_.init_empty (which resizes -> mallocs) are
  // all one-time RMW on shared state, so under Direct they belong to lane 0 alone. gpu_fission_setup
  // does exactly this.
  lane0_serial<Direct>([&] {
  if (a.slot_live == 0) {
    new (&st) slot_state<FT>();  // placement new: host memory is zeroed, NSDMIs must run
    FT& ft                      = st.ft;
    ft.num_updates_             = a.num_updates;
    ft.refactor_frequency_      = a.refactor_frequency;
    ft.L0_                      = make_view(m, m, a.L0_nz, a.L0_col_start, a.L0_i, a.L0_x);
    ft.U0_                      = make_view(m, m, a.U0_nz, a.U0_col_start, a.U0_i, a.U0_x);
    ft.L0_transpose_            = make_view(m, m, a.L0t_nz, a.L0t_col_start, a.L0t_i, a.L0t_x);
    ft.U0_transpose_            = make_view(m, m, a.U0t_nz, a.U0t_col_start, a.U0t_i, a.U0t_x);
    ft.row_permutation_         = gpu_span<i_t>(a.row_perm, m);
    ft.inverse_row_permutation_ = gpu_span<i_t>(a.inv_row_perm, m);
    ft.xi_workspace_            = gpu_span<i_t>(a.xi_workspace, 2 * m);
    ft.x_workspace_             = gpu_span<f_t>(a.x_workspace, m);
    ft.S_.init_empty(m);
    ft.hypersparse_threshold_ = 0.05;
  }
  });   // end one-time lane-0 construction
  FT& ft = st.ft;

  gpu_solution sol;
  sol.x = gpu_span<f_t>(a.x, n);
  sol.y = gpu_span<f_t>(a.y, m);
  sol.z = gpu_span<f_t>(a.z, n);

  // Warp path: precompute the fixed L0/U0 elimination-level schedules once and wire ft to the
  // shared config.
  //
  // The schedule buffers live in `st` (slot_state), NOT on the kernel stack. They used to be stack
  // locals, which was safe only because just lane 0 ran this code. Under the Direct model every lane
  // executes here, and a stack-local gpu_vector would device-malloc PER LANE — 32 private buffers,
  // 31 leaked, and worse: warp_build_config publishes `lvl_*[o].data()` / `sp_*_v.data()` into the
  // config, so every wavefront solve for the rest of the solve would read a lane-private schedule.
  // slot_state already carries byte-identical members for exactly this reason (the fission Ksetup
  // uses them), so this is a reuse, not a new allocation.
  if constexpr (Warp) {
    // Serial O(m*nnz) schedule build that resizes the slot's gpu_vectors and writes ~90 config
    // fields, plus an RMW on ft.warp_cfg_ — one lane. Every lane reads the result afterwards, which
    // the lane0_serial __syncwarp publishes.
    lane0_serial<Direct>([&] {
      warp_build_config(
        ft, cfg_shared, st.lvl_lev, st.lvl_ls, st.lvl_co, st.sp_ls_v, st.sp_co_v, st.sp_pos_v);
      ft.warp_cfg_ = &cfg_shared;
    });
  }

  dual::status_t status = gpu_simplified_dual_phase2_device<Warp, Direct>(lp,
                                                            settings,
                                                            a.vstatus,
                                                            st,
                                                            a.basic_list,
                                                            a.nonbasic_list,
                                                            sol,
                                                            a.dy_se,
                                                            a.dy_se_empty != 0);

  // `status` is warp-uniform (every terminal condition in the driver is), so these are same-value
  // stores from every lane — but keep a single writer.
  lane0_serial<Direct>([&] {
    *a.out_status         = static_cast<i_t>(status);
    *a.out_objective      = sol.objective;
    *a.out_user_objective = sol.user_objective;
    *a.out_iterations     = sol.iterations;
  });
  KT_EMIT(m);  // [KTIMER] print per-function tick totals + scaled_delta_xB density for this solve
  ST_EMIT();   // [STAT] per-call cycle distribution for the sparse-solve op groups
}

// ---- entry points -----------------------------------------------------------------------------
// Two wrappers over ONE body, because __launch_bounds__ cannot be applied conditionally on a
// template parameter and it must NOT reach the single-thread kernel. Measured 2026-08-11: putting
// `__launch_bounds__(32, 8)` on the <<<N,1>>> instantiation takes it from 129 to 188 registers
// (15 -> 10 blocks/SM) — the classic "the bound says there is no occupancy pressure, so ptxas spends
// registers freely" effect that motivated removing launch bounds in the first place.

// Generic entry: NO launch bound, ptxas's own heuristic. Used by the single-thread path and by the
// parked-helper warp form.
template <bool Warp, bool Direct = false, int MinBlocks = 8, class FT = gpu_basis_update_mpf>
__global__ void gpu_phase2_kernel(kernel_args* args)
{
  gpu_phase2_kernel_body<Warp, Direct, FT>(args);
}

// Direct warp entry, register-capped. ptxas left unconstrained picks 182 regs / 11 blocks/SM, but a
// cap sweep showed that is a heuristic, not a requirement: 168 regs at ZERO spill, 128 regs at 8 BYTES
// of spill for 16 blocks/SM (1.45x occupancy). MinBlocks maps to the cap as 65536/(32*MinBlocks):
// 12 -> 170, 16 -> 128, 32 -> 64. Only instantiated when CUOPT_MONO_DIRECT_MINBLOCKS != 0, because
// merely ATTACHING a launch bound changes ptxas's behaviour — see the single-thread note below.
#if CUOPT_MONO_DIRECT_MINBLOCKS
template <int MinBlocks = CUOPT_MONO_DIRECT_MINBLOCKS, class FT = gpu_basis_update_mpf>
__global__ void __launch_bounds__(32, MinBlocks) gpu_phase2_kernel_direct(kernel_args* args)
{
  gpu_phase2_kernel_body<true, true, FT>(args);
}
#else
// Uncapped Direct warp entry: no attribute at all. NOT the same as __launch_bounds__ with a large
// MinBlocks — the attribute's mere presence tells ptxas there is no occupancy pressure, and it then
// spends registers freely (measured: 214 regs with the bound at minBlocks=1-equivalent, vs 182 with
// no attribute; and 129 -> 188 on the single-thread kernel, which is why the two entry points exist).
template <int MinBlocks = 0, class FT = gpu_basis_update_mpf>
__global__ void gpu_phase2_kernel_direct(kernel_args* args)
{
  gpu_phase2_kernel_body<true, true, FT>(args);
}
#endif

// ===========================================================================================
// KERNEL FISSION — Phase 1a: the per-pivot pipeline split into 5 kernels + a setup kernel.
//
// WHY: gpu_phase2_kernel above carries the register union of EVERY per-pivot primitive (255 regs,
// register-limited to ~8 blocks/SM). Splitting lets each kernel carry only its own registers, so the
// light (ELEM) stages reach far higher occupancy, and every block in a launch runs the SAME
// primitive (less divergence).
//
// The monolith is NOT replaced — it stays as the bit-exact reference and the shipping path. These
// kernels are thin __global__ wrappers that call the SAME __device__ primitives; only the driver's
// control flow is redistributed (loop body -> 5 kernels, `while` -> host super-step loop).
//
// SCOPE (Phase 1a), see gpu_phase2/FISSION_CHALLENGES.md:
//   * SINGLE-THREAD ONLY (<<<N,1>>>): each block runs its primitive in the same serial op order as
//     the monolith, which is what makes the bit-exact gate achievable. Warp comes later.
//   * FIXED BATCH, NO REFILL: N nodes in, super-steps until all terminate, N results out — exactly
//     the semantics of gpu_simplified_dual_phase2_batch, so the two are directly A/B comparable.
//     (Slot recycling, and with it the setup-skew and harvest-cadence questions, come in Phase 1b.)
//   * SE-DRIFT REPRICE OMITTED (#3). It is the worst source of super-step barrier imbalance
//     (K1 barrier 2761 -> 592 BTran-units; every step becomes uniform). The monolith must therefore
//     run with CUOPT_SEREP_OFF=1 to serve as the bit-exact reference — see g_serep_off.
//   * NO DEVICE TIME LIMIT: the host owns it (it bounds the super-step loop), so the driver's
//     start_ns local is not needed here.
//
// TERMINATION: slot_state::status is UNSET while running. Every kernel opens with
// `if (st.status != UNSET) return;` so a finished slot is a no-op for the rest of the run. The
// kernel that DETECTS termination must call fission_finish() and return IMMEDIATELY — relying on
// the next kernel's guard is not enough, because the remainder of the current kernel would keep
// mutating state the monolith never touched (e.g. update_dual_variables after DUAL_UNBOUNDED).
// ===========================================================================================

// Rebuild the device views over kernel_args. Identical to gpu_phase2_kernel's prologue; factored so
// all six fission kernels share one definition instead of copying it.
__device__ inline void fission_bind(kernel_args& a, gpu_lp& lp, gpu_settings& settings,
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

  sol.x = gpu_span<f_t>(a.x, n);
  sol.y = gpu_span<f_t>(a.y, m);
  sol.z = gpu_span<f_t>(a.z, n);
}

// Terminate this slot: write the outputs exactly as the monolith's `return` epilogue does, publish
// the status (both into the slot and into the host-visible contiguous array), and mark it finished.
// Direct (warp fission): the epilogue is an O(n) serial objective sum plus a handful of stores —
// keep it on lane 0. Every caller must `return` immediately afterwards, and does so UNIFORMLY (all
// terminal conditions are computed identically on every lane), so the trailing __syncwarp is safe.
template <bool Direct = false, class FT>
__device__ inline void fission_finish(slot_state<FT>& st, kernel_args& a, dual::status_t s, i_t iters)
{
  lane0_serial<Direct>([&] {
    const gpu_lp& lp  = st.lp;
    gpu_solution& sol = st.sol;
    sol.objective         = compute_objective(lp, sol.x.data());
    sol.user_objective    = compute_user_objective(lp, sol.objective);
    sol.iterations        = iters;
    *a.out_status         = static_cast<i_t>(s);
    *a.out_objective      = sol.objective;
    *a.out_user_objective = sol.user_objective;
    *a.out_iterations     = sol.iterations;
    st.status             = static_cast<i_t>(s);
    if (a.status_arr) { a.status_arr[blockIdx.x] = static_cast<i_t>(s); }
  });
}

// Common opening for K1..K5: fetch the slot; false => skip this block (already terminated).
// NOTE: it does NOT call fission_bind. The views were built once by Ksetup and live in the slot —
// rebuilding them per kernel is exactly the register floor we are trying to remove.
template <bool Warp = false, class FT>
__device__ inline bool fission_enter(kernel_args& a, slot_state<FT>*& st)
{
  // geometry must match the compiled path (same contract as gpu_phase2_kernel)
  if (blockDim.x != (Warp ? 32u : 1u)) { assert(false); return false; }
  st = static_cast<slot_state<FT>*>(a.slot);
  return st->status == static_cast<i_t>(dual::status_t::UNSET);
}

// ---- Ksetup: construct the slot, wire ft, run the S1-S7 prologue (once per node) --------------
template <bool Warp = false, class FT = gpu_basis_update_mpf>
__global__ void gpu_fission_setup(kernel_args* args)
{
  kernel_args& a = args[blockIdx.x];
  const i_t m = a.m;
  // Fission never uses the parked-helper model: if this kernel is a warp, all 32 lanes are HERE.
  constexpr bool Direct = Warp;
  slot_state<FT>& st = *static_cast<slot_state<FT>*>(a.slot);
  // Placement-new, the status stores and the whole ft wiring are one-time construction on shared
  // state — lane 0 only, then re-converge before setup() reads any of it.
  lane0_serial<Direct>([&] {
  new (&st) slot_state<FT>();                       // host memory is zeroed; NSDMIs must run
  st.status = static_cast<i_t>(dual::status_t::UNSET);
  if (a.status_arr) { a.status_arr[blockIdx.x] = static_cast<i_t>(dual::status_t::UNSET); }

  FT& ft                      = st.ft;              // same wiring as gpu_phase2_kernel
  ft.num_updates_             = a.num_updates;
  ft.refactor_frequency_      = a.refactor_frequency;
  ft.L0_                      = make_view(m, m, a.L0_nz, a.L0_col_start, a.L0_i, a.L0_x);
  ft.U0_                      = make_view(m, m, a.U0_nz, a.U0_col_start, a.U0_i, a.U0_x);
  ft.L0_transpose_            = make_view(m, m, a.L0t_nz, a.L0t_col_start, a.L0t_i, a.L0t_x);
  ft.U0_transpose_            = make_view(m, m, a.U0t_nz, a.U0t_col_start, a.U0t_i, a.U0t_x);
  ft.row_permutation_         = gpu_span<i_t>(a.row_perm, m);
  ft.inverse_row_permutation_ = gpu_span<i_t>(a.inv_row_perm, m);
  ft.xi_workspace_            = gpu_span<i_t>(a.xi_workspace, 2 * m);
  ft.x_workspace_             = gpu_span<f_t>(a.x_workspace, m);
  ft.S_.init_empty(m);
  ft.hypersparse_threshold_ = 0.05;

  fission_bind(a, st.lp, st.settings, st.sol);   // built ONCE; K1..K5 read them from the slot

  // Build the fixed L0/U0 elimination-level schedules ONCE, into slot-owned buffers that survive
  // this kernel, and point ft at the slot's config.
  //
  // ⚠ THIS MUST HAPPEN BEFORE setup(). setup() itself now solves through the factors
  // (compute_primal_variables, and initialize_steepest_edge_norms when the node carries no edge
  // norms), so on the warp path it dereferences warp_cfg_. Building the config afterwards left it
  // NULL for the whole prologue -> cudaErrorIllegalAddress. The monolith has always ordered it this
  // way (warp_build_config, then the driver); fission now matches.
  if (g_fission_warp) {
    warp_build_config(st.ft, st.wcfg, st.lvl_lev, st.lvl_ls, st.lvl_co, st.sp_ls_v, st.sp_co_v,
                      st.sp_pos_v);
    st.ft.warp_cfg_ = &st.wcfg;
  }
  });  // end lane-0 construction

  st.template setup<Warp, Direct>(st.lp, st.settings, a.vstatus, a.basic_list, a.nonbasic_list,
                                  st.sol, a.dy_se, a.dy_se_empty != 0);

  lane0_serial<Direct>([&] {
    st.inited   = true;
    a.slot_live = 1;
  });
}

// ---- K1: pricing + BTran  (monolith lines: iter-limit guard, alloc guard, L1, L2) -------------
// Terminal: NUMERICAL (alloc), ITERATION_LIMIT, OPTIMAL.
// The SE-drift reprice that followed BTran is OMITTED in Phase 1 (see banner / CHALLENGES #3).
template <bool Warp = false, class FT = gpu_basis_update_mpf>
__global__ void gpu_fission_k1(kernel_args* args)
{
  kernel_args& a = args[blockIdx.x];
  slot_state<FT>* stp;
  if (!fission_enter<Warp, FT>(a, stp)) { return; }
  slot_state<FT>& st          = *stp;
  const gpu_lp& lp            = st.lp;
  const gpu_settings& settings = st.settings;
  gpu_solution& sol           = st.sol;
  FT& ft = st.ft;
  // Fission never uses the parked-helper model: if this kernel is a warp, all 32 lanes are HERE, so
  // the primitives must take their direct-call arms. See the Direct-mode banner above the MPF class.
  constexpr bool Direct = Warp;

  if (g_gpu_alloc_failed) { fission_finish<Direct>(st, a, dual::status_t::NUMERICAL, st.iter); return; }
  const i_t iter_limit = min(500000, max(20000, 20 * (lp.num_rows + lp.num_cols)));
  if (st.iter >= iter_limit) {
    fission_finish<Direct>(st, a, dual::status_t::ITERATION_LIMIT, iter_limit); return;
  }

  // 1. Steepest-edge pricing. PURE READ (no stores), so under Direct every lane just recomputes the
  // same argmax — no guard, no broadcast, and `leaving == -1` is uniform. Warping the scan is a
  // later step: it would change the tie-break and so cannot be part of the ladder's step 1.
  i_t direction = 0, basic_leaving = -1;
  f_t max_val;
  const i_t leaving = steepest_edge_pricing_with_infeasibilities(
    lp, settings, sol.x.data(), a.dy_se, st.basic_mark.data(), st.squared_infeas, st.infeas_idx,
    direction, basic_leaving, max_val);
  if (leaving == -1) { fission_finish<Direct>(st, a, dual::status_t::OPTIMAL, st.iter); return; }
  lane0_serial<Direct>([&] {
    st.leaving = leaving; st.direction = direction; st.basic_leaving = basic_leaving;
    // 2. BTran.
    st.delta_y_sparse.clear();
    st.UTsol_sparse.clear();
  });
  compute_delta_y<Warp, Direct>(ft, basic_leaving, direction, st.delta_y_sparse, st.UTsol_sparse);
  // Butterfly reduction of ||delta_y||^2. Reordered vs the serial sum, and dy_norm_sq feeds the SE
  // norms -> pricing, so this is the second deliberate break from monolith bit-identity (§20).
  if constexpr (Direct) {
    const int lane = static_cast<int>(threadIdx.x);
    const f_t* dyx = st.delta_y_sparse.x.data();
    const i_t nz   = (i_t)st.delta_y_sparse.i.size();   // trip count as in norm2_squared()
    f_t part = 0.0;
    for (i_t k = lane; k < nz; k += 32) { part += dyx[k] * dyx[k]; }
    const f_t total = warp_reduce_add(part);
    lane0_serial<true>([&] { st.dy_norm_sq = total; });
  } else {
    st.dy_norm_sq = st.delta_y_sparse.norm2_squared();
  }
}

// ---- K2: delta_z + ratio test + dual update  (monolith L3, L4, L5) ----------------------------
// Terminal: DUAL_UNBOUNDED — which MUST short-circuit before update_dual_variables.
template <bool Warp = false, class FT = gpu_basis_update_mpf>
__global__ void gpu_fission_k2(kernel_args* args)
{
  kernel_args& a = args[blockIdx.x];
  slot_state<FT>* stp;
  if (!fission_enter<Warp, FT>(a, stp)) { return; }
  slot_state<FT>& st          = *stp;
  const gpu_lp& lp            = st.lp;
  const gpu_settings& settings = st.settings;
  gpu_solution& sol           = st.sol;
  FT& ft = st.ft;
  const i_t m = lp.num_rows;
  // Fission never uses the parked-helper model: if this kernel is a warp, all 32 lanes are HERE.
  constexpr bool Direct = Warp;

  // 3. Reduced-cost SpMv: transpose (CSR) when delta_y is sparse (<=30% dense), else dense CSC.
  i_t delta_y_nz0      = 0;
  const i_t nz_delta_y = (i_t)st.delta_y_sparse.i.size();
  if constexpr (Direct) {
    // INTEGER count, so a ballot/popc reduction is exact — nothing is reordered the way an FP sum
    // would be. Every lane accumulates the same running total, which keeps delta_y_nz0 (and the
    // used_transpose_dz branch it gates) warp-uniform. nz_delta_y is warp-uniform, so all lanes
    // execute the same number of __ballot_sync calls.
    const int lane = static_cast<int>(threadIdx.x);
    const f_t* dyx = st.delta_y_sparse.x.data();
    for (i_t base = 0; base < nz_delta_y; base += 32) {
      const i_t k    = base + lane;
      const bool hit = (k < nz_delta_y) && (fabs(dyx[k]) > 1e-12);
      delta_y_nz0 += __popc(__ballot_sync(0xffffffffu, hit));
    }
  } else {
    for (i_t k = 0; k < nz_delta_y; ++k) {
      if (fabs(st.delta_y_sparse.x[k]) > 1e-12) { delta_y_nz0++; }
    }
  }
  const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
  st.used_transpose_dz            = delta_y_nz_percentage <= 30.0;
  if (st.used_transpose_dz) {
    // Direct=Warp: on the warp path call the collective body directly (no cfg_shared, no op-code
    // dispatch) — the whole point of fission. Internally lane-guarded + count broadcast.
    compute_delta_z<Warp, Warp>(ft, st.Arow, st.delta_y_sparse, st.leaving, st.direction,
                                st.nonbasic_end.data(), st.delta_z_mark.data(), st.delta_z_indices,
                                st.delta_z.data());
  } else {
    // Dense route. warp_to_dense keeps the sizing (reserve + set_size, a header RMW that may
    // device-malloc) on lane 0 and warps the O(n) zero-fill + the nnz scatter — bit-exact, since the
    // sparse indices are distinct. The delta_z update itself is the WARP_DZ collective: one lane per
    // nonbasic column, gather dots, ordered ballot/popc append -> bit-exact vs the serial scan, so
    // the push_back that used to force this whole branch onto lane 0 is gone.
    if constexpr (Warp) {
      ft.template warp_to_dense<Direct>(st.delta_y_sparse, st.delta_y);
    } else {
      st.delta_y_sparse.to_dense(st.delta_y);
    }
    compute_reduced_cost_update<Warp, Direct>(lp, ft, a.basic_list, a.nonbasic_list,
                                              st.delta_y.data(), st.leaving, st.direction,
                                              st.delta_z_mark.data(), st.delta_z_indices,
                                              st.delta_z.data());
  }

  // 4. Ratio test — SERIAL on lane 0 even under Warp, deliberately (port step 1).
  // The warp band is a DIFFERENT tie-break rule (see phase2_ratio_test) and is NOT bit-identical
  // to the single-thread scan, so enabling it here would make this kernel's port unverifiable
  // against the single-thread fission baseline. Keep it serial until the rest of K2 is proven at
  // exactly 100/100, then switch to warp_rt_body and accept the documented 59/100 signature.
  if constexpr (Warp) {
    // PARALLEL band (warp_rt_body) — measured 1.122x faster than the serial walking band on the
    // MPF/LU path at an identical iteration count (run_rt_band_ab.sh). It is a DIFFERENT tie-break
    // rule, so this kernel is deliberately NOT bit-identical to the single-thread path from here on
    // — expect the documented ~59/100 cpu==gpu MPF-warp signature, not 100/100. See phase2_ratio_test.
    // Outputs are lane-0 writes through pointers; aiming them at slot_state (global) + __syncwarp
    // publishes them to every lane with no shuffle needed.
    warp_rt_body(settings.dual_tol / 10, st.delta_z.data(), &st.entering, &st.nonbasic_entering,
                 lp.num_rows, lp.num_cols, a.nonbasic_list, settings.pivot_tol, &st.step_length,
                 a.vstatus, sol.z.data(), settings.zero_tol);
    __syncwarp(0xffffffffu);
  } else {
    f_t step_length = 0.0;
    i_t nonbasic_entering = -1;
    const i_t entering = phase2_ratio_test<false>(lp, settings, ft, a.vstatus, a.nonbasic_list,
                                                  sol.z.data(), st.delta_z.data(), step_length,
                                                  nonbasic_entering);
    st.entering = entering; st.step_length = step_length; st.nonbasic_entering = nonbasic_entering;
  }
  if (st.entering == -1) {
    // SHORT-CIRCUIT: the monolith returns here, so update_dual_variables below must NOT run.
    fission_finish(st, a, dual::status_t::DUAL_UNBOUNDED, st.iter);
    return;
  }

  update_dual_variables<Warp, Warp>(ft, st.delta_y_sparse, st.delta_z_indices, st.delta_z.data(),
                                    st.step_length, st.leaving, sol.y.data(), sol.z.data());
}

// ---- K3: FTran  (monolith: rhs from A(:,entering) + compute_delta_x) --------------------------
// Terminal: NUMERICAL.
template <bool Warp = false, class FT = gpu_basis_update_mpf>
__global__ void gpu_fission_k3(kernel_args* args)
{
  kernel_args& a = args[blockIdx.x];
  slot_state<FT>* stp;
  if (!fission_enter<Warp, FT>(a, stp)) { return; }
  slot_state<FT>& st          = *stp;
  const gpu_lp& lp            = st.lp;
  const gpu_settings& settings = st.settings;
  gpu_solution& sol           = st.sol;
  FT& ft = st.ft;
  constexpr bool Direct = Warp;

  lane0_serial<Direct>([&] {
    st.utilde_sparse.clear();
    st.scaled_dxB_sparse.clear();
    st.rhs_sparse.from_csc_column(lp.A, st.entering);   // push_back -> may device-malloc
  });
  // dx_status is uniform: compute_delta_x broadcasts its NUMERICAL verdict before returning.
  const i_t dx_status = compute_delta_x<Warp, Direct>(
    lp, ft, st.entering, st.leaving, st.basic_leaving, st.direction, a.basic_list, st.rhs_sparse,
    st.delta_z.data(), sol.x.data(), st.utilde_sparse, st.scaled_dxB_sparse, st.delta_x.data());
  if (dx_status == -1) {
    fission_finish<Direct>(st, a, dual::status_t::NUMERICAL, st.iter);
    return;
  }
}

// ---- K4: SE-norm update + primal update + objective update  (monolith L9, L10, L11) -----------
// No terminal state.
template <bool Warp = false, class FT = gpu_basis_update_mpf>
__global__ void gpu_fission_k4(kernel_args* args)
{
  kernel_args& a = args[blockIdx.x];
  slot_state<FT>* stp;
  if (!fission_enter<Warp, FT>(a, stp)) { return; }
  slot_state<FT>& st          = *stp;
  const gpu_lp& lp            = st.lp;
  const gpu_settings& settings = st.settings;
  gpu_solution& sol           = st.sol;
  FT& ft = st.ft;

  constexpr bool Direct = Warp;

  update_steepest_edge_norms<Warp, Direct>(settings, a.basic_list, ft, st.direction,
                                           st.delta_y_sparse, st.dy_norm_sq, st.scaled_dxB_sparse,
                                           st.basic_leaving, st.entering, st.v.data(), st.v_sparse,
                                           a.dy_se);
  // The two used to share one lane-0 region. They are independent (update_objective reads delta_x
  // and objective, never x), so splitting them is safe.
  update_primal_variables<Direct>(st.scaled_dxB_sparse, a.basic_list, st.delta_x.data(), st.entering,
                                  sol.x.data());
  // Called with the OLD basic_list (before K5's swap), true costs, no bound-flip term.
  // Under Direct this reduces with a butterfly — a reordered sum into the CUTOFF comparand.
  update_objective<Direct>(a.basic_list, st.scaled_dxB_sparse, lp.objective, st.delta_x.data(),
                           st.entering, st.obj);
}

// ---- K5: basis swap + Arow sync + ft.update + infeasibilities + clear + iter++ + cutoff -------
// Terminal: NUMERICAL (ft.update), CUTOFF.
template <bool Warp = false, class FT = gpu_basis_update_mpf>
__global__ void gpu_fission_k5(kernel_args* args)
{
  kernel_args& a = args[blockIdx.x];
  slot_state<FT>* stp;
  if (!fission_enter<Warp, FT>(a, stp)) { return; }
  slot_state<FT>& st          = *stp;
  const gpu_lp& lp            = st.lp;
  const gpu_settings& settings = st.settings;
  gpu_solution& sol           = st.sol;
  FT& ft = st.ft;
  const i_t leaving = st.leaving, entering = st.entering;
  constexpr bool Direct = Warp;

  // 5. Basis swap + MPF rank-1 stacking update. Every line here is a read-modify-write on shared
  // state (vstatus/basic_list/basic_mark stores, update_Arow's in-place row repartition + row_mark,
  // ft.update's S_ append + sort + allocations), so it stays on lane 0 — only `pinf` below is
  // collective-able, which is why K5 is a MIXED kernel rather than a fully warped one.
  i_t fu_status = 0;
  lane0_serial<Direct>([&] {
  a.vstatus[entering] = variable_status_t::BASIC;
  a.vstatus[leaving]  = (lp.lower[leaving] == lp.upper[leaving])
                          ? variable_status_t::NONBASIC_FIXED
                          : static_cast<variable_status_t>(-st.direction);
  a.basic_list[st.basic_leaving]        = entering;
  a.nonbasic_list[st.nonbasic_entering] = leaving;
  st.basic_mark[leaving]                = -1;
  st.basic_mark[entering]               = st.basic_leaving;

  update_Arow(leaving, entering, lp.A, st.row_mark, st.nonbasic_end.data(), st.Arow);

  fu_status = ft.update(st.utilde_sparse, st.UTsol_sparse, st.basic_leaving);
  });
  fu_status = lane0_bcast<Direct>(fu_status);   // the NUMERICAL exit must be taken uniformly
  if (fu_status == 1) {
    fission_finish(st, a, dual::status_t::NUMERICAL, st.iter);
    return;
  }

  compute_initial_primal_infeasibilities<Warp, Direct>(lp, settings, ft, a.basic_list, sol.x.data(),
                                                       st.squared_infeas, st.infeas_idx,
                                                       st.primal_inf);
  clear_delta_z<Direct>(entering, leaving, st.delta_z_mark.data(), st.delta_z_indices,
                        st.delta_z.data());

  lane0_serial<Direct>([&] { st.iter++; });   // counts COMPLETED pivots, matching the monolith

  if (isfinite(settings.cut_off) && st.obj >= settings.cut_off) {
    fission_finish(st, a, dual::status_t::CUTOFF, st.iter);
    return;
  }
}


// ---------------------------------------------------------------------------
// Host-side helpers for marshaling.
// ---------------------------------------------------------------------------
template <typename T>
static T* upload(const std::vector<T>& v)
{
  T* d = nullptr;
  const size_t bytes = (v.empty() ? 1 : v.size()) * sizeof(T);
  RAFT_CUDA_TRY(cudaMalloc(&d, bytes));
  if (!v.empty()) { RAFT_CUDA_TRY(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  return d;
}

template <typename T>
static T* upload_n(const T* p, size_t count)
{
  T* d = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d, (count == 0 ? 1 : count) * sizeof(T)));
  if (count > 0) { RAFT_CUDA_TRY(cudaMemcpy(d, p, count * sizeof(T), cudaMemcpyHostToDevice)); }
  return d;
}

template <typename T>
static T* alloc_zeroed(size_t count)
{
  T* d = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d, (count == 0 ? 1 : count) * sizeof(T)));
  RAFT_CUDA_TRY(cudaMemset(d, 0, (count == 0 ? 1 : count) * sizeof(T)));
  return d;
}

// Loop condition for the CUDA-graph WHILE node. Evaluates the SAME predicate the host loop used to:
// "keep going while at least one slot is still UNSET, and the super-step cap is not reached", and
// writes it to the conditional handle so the loop runs entirely on the device.
//
// One block: N is the batch size, so a grid-stride scan by one block is ample, and it keeps the
// conditional write trivially single-threaded (concurrent writes to a handle are UB).
__global__ void gpu_fission_cond(cudaGraphConditionalHandle handle,
                                 const i_t* status,
                                 i_t n,
                                 unsigned long long* steps,
                                 unsigned long long cap)
{
  __shared__ int any_unset;
  if (threadIdx.x == 0) { any_unset = 0; }
  __syncthreads();
  for (i_t b = static_cast<i_t>(threadIdx.x); b < n; b += static_cast<i_t>(blockDim.x)) {
    if (status[b] == static_cast<i_t>(dual::status_t::UNSET)) { atomicOr(&any_unset, 1); }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long s = ++(*steps);   // counts COMPLETED super-steps
    cudaGraphSetConditional(handle, (any_unset != 0 && s < cap) ? 1u : 0u);
  }
}

// Device heap (in-kernel gpu_vector malloc arena) + stack are per-CONTEXT limits and the heap
// can ONLY be set before the first malloc-using kernel launch, so set them exactly once per
// process. First caller wins: the batch wrapper (the only/first GPU kernel in the throughput
// benchmark) sizes the heap for N concurrent S_ stacks; the single-problem wrapper uses a
// generous default. heap_bytes is the requested malloc-arena size (NOT counting the static
// per-problem buffers, which are separate cudaMallocs).
static bool g_device_limits_set = false;
static void ensure_device_limits(size_t heap_bytes)
{
  if (g_device_limits_set) return;
  g_device_limits_set = true;
  if (heap_bytes < (static_cast<size_t>(256) << 20)) { heap_bytes = static_cast<size_t>(256) << 20; }
  cudaError_t hp = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_bytes);
  if (hp != cudaSuccess) {
    cudaGetLastError();
    std::fprintf(stderr, "[gpu_phase2] warning: could not set device heap size: %s\n",
                 cudaGetErrorString(hp));
  }
  // The deep (non-recursive) call chain wants more than the ~1 KB default stack.
  cudaError_t stk = cudaDeviceSetLimit(cudaLimitStackSize, static_cast<size_t>(32) << 10);
  if (stk != cudaSuccess) {
    cudaGetLastError();
    std::fprintf(stderr, "[gpu_phase2] warning: could not raise device stack size: %s\n",
                 cudaGetErrorString(stk));
  }
}

// Marshal one prepared problem into a kernel_args with fresh device buffers. Both the
// single-problem wrapper and the batch wrapper build kernel_args through here, so the device
// sees byte-identical inputs either way. `ft` must be a fresh factorization (num_updates == 0).
// Free the buffers afterwards with free_kernel_args.
// Read-only device buffers shared across all batch items with the same problem+factor (same `lp`
// pointer). These are never written by the kernel (A CSC, the base LU factor, rhs/bounds/objective,
// row permutation), so N same-problem blocks read ONE resident copy instead of N (VRAM + L2 reuse).
// Arow is intentionally NOT here: its j/x are permuted per node, so each block builds it via from_csc
// off the shared CSC.
// Only the PROBLEM-INVARIANT read-only data is shared across a problem's nodes: the constraint matrix
// A (CSC) plus rhs and objective. Everything basis-dependent (the LU factor L0/U0/L0t/U0t, row_perm),
// the branching bounds (lower/upper), and the per-node-permuted Arow stay PER-NODE — they legitimately
// differ from node to node. (A's col_start is safe to share: unlike the factor, A is never reach-marked.)
struct ro_upload {
  i_t *A_col_start{}, *A_i{}; f_t* A_x{}; i_t A_nz{};
  f_t *rhs{}, *objective{};
};

static ro_upload upload_ro(const lp_problem_t<i_t, f_t>& lp)
{
  const auto& A = lp.A;
  ro_upload r{};
  r.A_col_start = upload(A.col_start); r.A_nz = A.nnz();
  r.A_i = upload_n(A.i.data(), A.nnz()); r.A_x = upload_n(A.x.data(), A.nnz());
  r.rhs = upload(lp.rhs); r.objective = upload(lp.objective);
  return r;
}

static void free_ro(const ro_upload& r)
{
  for (void* p : {(void*)r.A_col_start, (void*)r.A_i, (void*)r.A_x, (void*)r.rhs, (void*)r.objective}) {
    if (p) { cudaFree(p); }
  }
}

static kernel_args marshal_problem(const lp_problem_t<i_t, f_t>& lp,
                                   const simplex_solver_settings_t<i_t, f_t>& settings,
                                   std::vector<variable_status_t>& vstatus,
                                   basis_update_mpf_t<i_t, f_t>& ft,
                                   std::vector<i_t>& basic_list,
                                   std::vector<i_t>& nonbasic_list,
                                   lp_solution_t<i_t, f_t>& sol,
                                   std::vector<f_t>& delta_y_steepest_edge,
                                   const ro_upload* ro = nullptr)
{
  const int m = lp.num_rows;
  const int n = lp.num_cols;
  const auto& A   = lp.A;
  const auto& L0  = ft.L0();
  const auto& U0  = ft.U0();
  const auto& L0t = ft.L0_transpose();
  const auto& U0t = ft.U0_transpose();

  kernel_args a{};
  a.m = m;
  a.n = n;

  a.obj_scale   = lp.obj_scale;
  a.obj_constant = lp.obj_constant;
  // Problem-invariant, read-only: A CSC + rhs + objective — shared across the problem's nodes when a
  // shared `ro` is provided, else uploaded per-node. Arow is always per-node (its j/x get permuted in
  // place), but when sharing it is built on-device via from_csc off the SHARED A → the A reads hit one
  // cached copy instead of N.
  if (ro != nullptr) {
    a.A_col_start = ro->A_col_start; a.A_nz = ro->A_nz; a.A_i = ro->A_i; a.A_x = ro->A_x;
    a.rhs = ro->rhs; a.objective = ro->objective;
    a.Arow_row_start = nullptr; a.Arow_j = nullptr; a.Arow_x = nullptr;
  } else {
    a.A_col_start = upload(A.col_start); a.A_nz = A.nnz();
    a.A_i = upload_n(A.i.data(), A.nnz()); a.A_x = upload_n(A.x.data(), A.nnz());
    a.rhs = upload(lp.rhs); a.objective = upload(lp.objective);
    // Host-built Arow (CSR of A) — the kernel views it directly and skips the on-device from_csc
    // transpose. to_compressed_row orders each row by ascending column, matching from_csc bit-for-bit.
    cuopt::mathematical_optimization::csr_matrix_t<i_t, f_t> Arow_host(m, n, A.nnz());
    A.to_compressed_row(Arow_host);
    a.Arow_row_start = upload(Arow_host.row_start);
    a.Arow_j         = upload(Arow_host.j);
    a.Arow_x         = upload(Arow_host.x);
  }
  // Per-node (basis- or branch-dependent), always uploaded per item: bounds, the LU factor, row perm.
  a.lower       = upload(lp.lower);
  a.upper       = upload(lp.upper);
  a.L0_col_start = upload(L0.col_start); a.L0_nz = L0.nnz();
  a.L0_i = upload_n(L0.i.data(), L0.nnz()); a.L0_x = upload_n(L0.x.data(), L0.nnz());
  a.U0_col_start = upload(U0.col_start); a.U0_nz = U0.nnz();
  a.U0_i = upload_n(U0.i.data(), U0.nnz()); a.U0_x = upload_n(U0.x.data(), U0.nnz());
  a.L0t_col_start = upload(L0t.col_start); a.L0t_nz = L0t.nnz();
  a.L0t_i = upload_n(L0t.i.data(), L0t.nnz()); a.L0t_x = upload_n(L0t.x.data(), L0t.nnz());
  a.U0t_col_start = upload(U0t.col_start); a.U0t_nz = U0t.nnz();
  a.U0t_i = upload_n(U0t.i.data(), U0t.nnz()); a.U0t_x = upload_n(U0t.x.data(), U0t.nnz());
  // DEBUG dump of the LU FACTORS (L0, U0 scatter form + L0t, U0t gather transposes) per node →
  // CUOPT_DUMP_LU_DIR/lu_<k>.bin. Four CSC blocks: L0, U0, L0t, U0t (each n, nnz, col_start[n+1], i, x).
  // Pairs 1:1 (same node order) with inv_<k>.bin for the warp-SpMV-vs-warp-LU microbenchmark.
  if (const char* _dd = std::getenv("CUOPT_DUMP_LU_DIR")) {
    static int _c = 0; char _p[600]; std::snprintf(_p, sizeof(_p), "%s/lu_%d.bin", _dd, _c++);
    auto _w = [](FILE* _f, i_t _n, i_t _nz, const i_t* _cs, const i_t* _i, const f_t* _x) {
      std::fwrite(&_n, sizeof(i_t), 1, _f); std::fwrite(&_nz, sizeof(i_t), 1, _f);
      std::fwrite(_cs, sizeof(i_t), static_cast<size_t>(_n) + 1, _f);
      std::fwrite(_i, sizeof(i_t), _nz, _f); std::fwrite(_x, sizeof(f_t), _nz, _f); };
    if (FILE* _f = std::fopen(_p, "wb")) {
      _w(_f, L0.n, L0.nnz(), L0.col_start.data(), L0.i.data(), L0.x.data());
      _w(_f, U0.n, U0.nnz(), U0.col_start.data(), U0.i.data(), U0.x.data());
      _w(_f, L0t.n, L0t.nnz(), L0t.col_start.data(), L0t.i.data(), L0t.x.data());
      _w(_f, U0t.n, U0t.nnz(), U0t.col_start.data(), U0t.i.data(), U0t.x.data());
      std::fclose(_f);
    }
  }
  a.row_perm     = upload(ft.row_permutation());
  a.inv_row_perm = upload(ft.inverse_row_permutation());
  a.xi_workspace = alloc_zeroed<int>(2 * m);
  a.x_workspace  = alloc_zeroed<double>(m);
  a.num_updates  = 0;
  a.refactor_frequency = settings.refactor_frequency;

  a.vstatus       = upload(vstatus);
  a.basic_list    = upload(basic_list);
  a.nonbasic_list = upload(nonbasic_list);
  a.x             = upload(sol.x);
  a.y             = upload(sol.y);
  a.z             = upload(sol.z);
  if (delta_y_steepest_edge.empty()) {
    a.dy_se       = alloc_zeroed<double>(n);
    a.dy_se_empty = 1;
  } else {
    a.dy_se       = upload(delta_y_steepest_edge);
    a.dy_se_empty = 0;
  }

  a.primal_tol = settings.primal_tol;
  a.dual_tol   = settings.dual_tol;
  a.pivot_tol  = settings.pivot_tol;
  a.tight_tol  = settings.tight_tol;
  a.fixed_tol  = settings.fixed_tol;
  a.zero_tol   = settings.zero_tol;
  a.time_limit = settings.time_limit;
  a.cut_off             = settings.cut_off;
  a.steepest_edge_ratio = settings.steepest_edge_ratio;
  a.start_time          = tic();

  a.out_status         = alloc_zeroed<int>(1);
  a.out_objective      = alloc_zeroed<double>(1);
  a.out_user_objective = alloc_zeroed<double>(1);
  a.out_iterations     = alloc_zeroed<int>(1);

  // Persistent per-slot solve state. Allocated ZEROED (the kernel's placement-new relies on that
  // for the slot_live==0 "fresh" test) and sized for the concrete device type. One per problem
  // here; the fissioned rolling pool will allocate one per SLOT and reuse it across nodes.
  a.slot      = alloc_zeroed<char>(sizeof(slot_state<gpu_basis_update_mpf>));
  a.slot_live = 0;
  return a;
}

static void free_kernel_args(const kernel_args& a, bool skip_ro = false)
{
  // Per-item buffers (always owned by this kernel_args). Arow_* are null when RO-shared (no-op frees).
  // Everything basis-/branch-dependent — the full LU factor (col_start + i + x), row perm, and the
  // branching bounds — is per-node and freed here even when RO-shared.
  for (void* p : {(void*)a.Arow_row_start, (void*)a.Arow_j, (void*)a.Arow_x,
                  (void*)a.L0_col_start, (void*)a.L0_i, (void*)a.L0_x,
                  (void*)a.U0_col_start, (void*)a.U0_i, (void*)a.U0_x,
                  (void*)a.L0t_col_start, (void*)a.L0t_i, (void*)a.L0t_x,
                  (void*)a.U0t_col_start, (void*)a.U0t_i, (void*)a.U0t_x,
                  (void*)a.row_perm, (void*)a.inv_row_perm,
                  (void*)a.lower, (void*)a.upper,
                  (void*)a.xi_workspace, (void*)a.x_workspace, (void*)a.vstatus,
                  (void*)a.basic_list, (void*)a.nonbasic_list, (void*)a.x, (void*)a.y, (void*)a.z,
                  (void*)a.dy_se, (void*)a.out_status, (void*)a.out_objective,
                  (void*)a.out_user_objective, (void*)a.out_iterations,
                  (void*)a.slot}) {
    if (p) { cudaFree(p); }
  }
  if (skip_ro) { return; }  // read-only buffers (A CSC + rhs + objective) shared → freed once via free_ro
  for (void* p : {(void*)a.A_col_start, (void*)a.A_i, (void*)a.A_x,
                  (void*)a.rhs, (void*)a.objective}) {
    if (p) { cudaFree(p); }
  }
}

// Exact device bytes marshal_problem cudaMallocs for one problem's STATIC buffers (read-only
// factors/matrix + mutable state), EXCLUDING the in-kernel gpu_vector heap (S_ growth). Used by
// the batch pre-flight VRAM estimate; mirrors the upload/alloc sizes 1:1 (empty vectors still
// take 1 element, matching upload/alloc_zeroed). Slightly conservative on the small state vectors.
static size_t problem_static_bytes(const lp_problem_t<i_t, f_t>& lp,
                                   const basis_update_mpf_t<i_t, f_t>& ft)
{
  const size_t m = lp.num_rows;
  const size_t n = lp.num_cols;
  const auto& L0  = ft.L0();
  const auto& U0  = ft.U0();
  const auto& L0t = ft.L0_transpose();
  const auto& U0t = ft.U0_transpose();
  auto bi = [](size_t c) { return (c == 0 ? 1 : c) * sizeof(i_t); };
  auto bd = [](size_t c) { return (c == 0 ? 1 : c) * sizeof(f_t); };
  size_t b = 0;
  // A (col_start, i, x) + rhs/lower/upper/objective
  b += bi(lp.A.col_start.size()) + bi(lp.A.nnz()) + bd(lp.A.nnz());
  b += bd(m) + 3 * bd(n);
  // Four factors, each col_start + i + x
  b += bi(L0.col_start.size())  + bi(L0.nnz())  + bd(L0.nnz());
  b += bi(U0.col_start.size())  + bi(U0.nnz())  + bd(U0.nnz());
  b += bi(L0t.col_start.size()) + bi(L0t.nnz()) + bd(L0t.nnz());
  b += bi(U0t.col_start.size()) + bi(U0t.nnz()) + bd(U0t.nnz());
  // perms + workspaces
  b += bi(m) + bi(m) + bi(2 * m) + bd(m);
  // state: vstatus(n) + basic(m) + nonbasic(<=n) + x(n)+y(m)+z(n) + dy_se(n) + 4 out scalars
  b += n * sizeof(variable_status_t) + bi(m) + bi(n) + bd(n) + bd(m) + bd(n) + bd(n);
  b += 2 * bi(1) + 2 * bd(1);
  // persistent per-slot solve state (slot_state<FT> header; its gpu_vector payloads come from the
  // in-kernel device heap, which heap_estimate_bytes covers separately)
  b += sizeof(slot_state<gpu_basis_update_mpf>);
  return b;
}

}  // namespace gpu_phase2_detail

// ---------------------------------------------------------------------------
// Host wrapper. Same signature as simplified_dual_phase2.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
dual::status_t gpu_simplified_dual_phase2(const lp_problem_t<i_t, f_t>& lp,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          std::vector<variable_status_t>& vstatus,
                                          basis_update_mpf_t<i_t, f_t>& ft,
                                          std::vector<i_t>& basic_list,
                                          std::vector<i_t>& nonbasic_list,
                                          lp_solution_t<i_t, f_t>& sol,
                                          std::vector<f_t>& delta_y_steepest_edge)
{
  using namespace gpu_phase2_detail;
  static_assert(sizeof(i_t) == sizeof(int) && sizeof(f_t) == sizeof(double),
                "gpu_simplified_dual_phase2 is only instantiated for <int, double>");

  const int m = lp.num_rows;
  const int n = lp.num_cols;

  // Pure-stacking GPU port: the supplied factorization must be fresh (no stacked
  // updates yet), matching the harness which refactors immediately before the call.
  if (ft.num_updates() != 0) {
    std::fprintf(stderr,
                 "[gpu_phase2] ERROR: ft has %d stacked updates; the GPU port expects a fresh "
                 "factorization.\n",
                 ft.num_updates());
    return dual::status_t::NUMERICAL;
  }

  // Size the per-context device heap/stack once (generous default for a single solve).
  {
    size_t free_b = 0, total_b = 0;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_b, &total_b));
    size_t heap = static_cast<size_t>(0.7 * static_cast<double>(free_b));
    const size_t cap = static_cast<size_t>(8) << 30;  // 8 GiB cap
    if (heap > cap) { heap = cap; }
    // The malloc arena is only actually reserved at the first malloc-using LAUNCH, on top of the
    // per-thread local reservation (cudaLimitStackSize x every resident thread) and the per-problem
    // cudaMallocs. On a small-VRAM card those three together can exceed the device and the launch
    // fails with cudaErrorLaunchOutOfResources -- which reads like a kernel-config problem but is not.
    // CUOPT_GPU_HEAP_MB pins the arena so the margin can be controlled explicitly (4096 works on a
    // 12 GB card where the 0.7*free default does not).
    if (const char* e = std::getenv("CUOPT_GPU_HEAP_MB")) {
      const long long v = std::atoll(e);
      if (v > 0) { heap = static_cast<size_t>(v) << 20; }
    }
    ensure_device_limits(heap);
  }

  kernel_args a = marshal_problem(lp, settings, vstatus, ft, basic_list, nonbasic_list, sol,
                                  delta_y_steepest_edge);

  // Reset the shared alloc-failed flag from the host (the kernel no longer does it).
  const int zero_flag = 0;
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_gpu_alloc_failed, &zero_flag, sizeof(int)));

  // The kernel takes a device pointer to an array of kernel_args (one per block);
  // the single-problem path is just N == 1.
  kernel_args* d_args = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_args, sizeof(kernel_args)));
  RAFT_CUDA_TRY(cudaMemcpy(d_args, &a, sizeof(kernel_args), cudaMemcpyHostToDevice));

  // Compile-time-selected optimized (warp) vs unoptimized (single-thread) kernel.
  // Both templates are instantiated here so the compiler emits code for both; the
  // host picks at runtime (env CUOPT_GPU_WARP) which pre-compiled kernel to launch.
  // CUOPT_GPU_WARP: unset/"0" = single-thread; "1" = plain wavefront; "2" (or higher) = big≥32.
  const char* warp_env  = std::getenv("CUOPT_GPU_WARP");
  const bool  warp_mode = (warp_env != nullptr && std::string(warp_env) != "0");
  // Kernel-fission bit-exact reference: CUOPT_SEREP_OFF=1 disables the SE-drift reprice so the
  // monolith matches the reprice-free fissioned pipeline (FISSION_CHALLENGES.md #3).
  { // DEFAULT: reprice OFF (see the g_serep_off banner). CUOPT_SEREP_ON=1 restores it.
    // CUOPT_SEREP_OFF is still accepted so existing harnesses keep working — it now merely
    // reaffirms the default.
    const char* on      = std::getenv("CUOPT_SEREP_ON");
    const int   serep_off = (on != nullptr && std::string(on) != "0") ? 0 : 1;
    RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_serep_off, &serep_off, sizeof(int))); }
  { const char* e = std::getenv("CUOPT_RT_SERIAL");
    const int rt_serial = (e != nullptr && std::string(e) != "0") ? 1 : 0;
    RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_rt_serial, &rt_serial, sizeof(int))); }
  const auto  _k0 = std::chrono::steady_clock::now();
#if !CUOPT_SKIP_MONOLITH_WARP
  if (warp_mode) {
    gpu_phase2_kernel_direct<><<<1, 32>>>(d_args);
  } else
#endif
  {
#if CUOPT_SKIP_MONOLITH_WARP
    if (warp_mode) {
      std::fprintf(stderr, "[gpu_phase2] CUOPT_GPU_WARP set but the warp kernel is not compiled "
                           "(CUOPT_SKIP_MONOLITH_WARP) — running the single-thread kernel.\n");
    }
#endif
    gpu_phase2_kernel<false><<<1, 1>>>(d_args);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  const double _kms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - _k0).count();
  std::printf("[GPUKERNEL] warp=%d prof=%d ms=%.3f\n", warp_mode ? 1 : 0,
              CUOPT_PROF_BUILD, _kms);
  cudaFree(d_args);

  // Marshal outputs back.
  int h_status = 0;
  double h_obj = 0.0, h_uobj = 0.0;
  int h_iters = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&h_status, a.out_status, sizeof(int), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(&h_obj, a.out_objective, sizeof(double), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(&h_uobj, a.out_user_objective, sizeof(double), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(&h_iters, a.out_iterations, sizeof(int), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(sol.x.data(), a.x, n * sizeof(double), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(sol.y.data(), a.y, m * sizeof(double), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(sol.z.data(), a.z, n * sizeof(double), cudaMemcpyDeviceToHost));

  sol.objective      = h_obj;
  sol.user_objective = h_uobj;
  sol.iterations     = h_iters;

  int alloc_failed = 0;
  RAFT_CUDA_TRY(cudaMemcpyFromSymbol(&alloc_failed, g_gpu_alloc_failed, sizeof(int)));
  if (alloc_failed) {
    std::fprintf(stderr, "[gpu_phase2] device allocation failed during solve.\n");
  }

  free_kernel_args(a);

  return static_cast<dual::status_t>(h_status);
}

// ---------------------------------------------------------------------------
// Batch wrapper: solve N independent problems in ONE launch, one block each.
// Each item must already be set up exactly as a single solve is (fresh factorized
// ft, basis lists, duals in sol.y/z) — i.e. the same steps run_phase2_from_basis
// does before the driver call. Fails LOUD (never a silent OOM): a pre-flight VRAM
// check rejects the whole batch if it will not fit. On success, fills out_status[i]
// and each item->sol (x, objective, user_objective, iterations).
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch(const std::vector<gpu_batch_item<i_t, f_t>>& items,
                                      size_t heap_estimate_bytes,
                                      std::vector<dual::status_t>& out_status,
                                      gpu_batch_timing& timing)
{
  using namespace gpu_phase2_detail;
  static_assert(sizeof(i_t) == sizeof(int) && sizeof(f_t) == sizeof(double),
                "gpu_simplified_dual_phase2_batch is only instantiated for <int, double>");

  const size_t N = items.size();
  out_status.assign(N, dual::status_t::UNSET);
  timing = gpu_batch_timing{};
  if (N == 0) return;

  // Every ft must be a fresh factorization (pure MPF stacking port, no on-device refactor).
  for (size_t b = 0; b < N; ++b) {
    if (items[b].ft->num_updates() != 0) {
      timing.rejected = true;
      timing.message  = "item " + std::to_string(b) + " has a non-fresh factorization (num_updates=" +
                        std::to_string(items[b].ft->num_updates()) + ")";
      std::fprintf(stderr, "[gpu_phase2] batch ERROR: %s\n", timing.message.c_str());
      return;
    }
  }

  // ---- Pre-flight VRAM check (fail loud, before any device allocation) ----
  size_t static_bytes = 0;
  for (size_t b = 0; b < N; ++b) {
    static_bytes += problem_static_bytes(*items[b].lp, *items[b].ft);
  }
  const size_t args_bytes = N * sizeof(kernel_args);
  const size_t need = static_bytes + heap_estimate_bytes + args_bytes;
  size_t free_b = 0, total_b = 0;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_b, &total_b));
  timing.vram_needed_bytes = need;
  timing.vram_free_bytes   = free_b;
  // Keep an 8% safety margin for CUDA context/allocator overhead + fragmentation.
  const size_t budget = static_cast<size_t>(0.92 * static_cast<double>(free_b));
  const double GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
  if (need > budget) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "batch needs ~%.2f GB VRAM (static %.2f + heap %.2f + args %.2f), only %.2f GB "
                  "free (usable budget %.2f GB) for %zu problems — reduce --replicas or use a larger GPU",
                  need * GB, static_bytes * GB, heap_estimate_bytes * GB, args_bytes * GB,
                  free_b * GB, budget * GB, N);
    timing.rejected = true;
    timing.message  = buf;
    std::fprintf(stderr, "[gpu_phase2] batch REJECTED: %s\n", buf);
    return;
  }

  // Size the malloc heap for all N concurrent S_ stacks (first GPU kernel this process).
  ensure_device_limits(heap_estimate_bytes);

  // ---- Marshal every problem + upload (host-timed) ----
  // CUOPT_GPU_SHARE_RO: dedup the problem-invariant read-only device buffers (A CSC + rhs + objective)
  // across items with the same `lp` pointer — N same-problem blocks read ONE resident copy of A instead
  // of N (VRAM + L2 reuse). The basis-dependent factor L/U, row perm, and branching bounds stay per-node
  // (they differ per B&B node). Arow is per-node too but built on-device off the shared A. Default ON (=0 opts out).
  const char* share_env  = std::getenv("CUOPT_GPU_SHARE_RO");
  const bool  share_ro   = (share_env == nullptr || std::string(share_env) != "0");  // default ON, opt out =0
  const double up_t0 = tic();
  std::vector<kernel_args> h_args(N);
  std::unordered_map<const void*, ro_upload> ro_cache;
  for (size_t b = 0; b < N; ++b) {
    const auto& it = items[b];
    const ro_upload* ro = nullptr;
    if (share_ro) {
      auto res = ro_cache.try_emplace(static_cast<const void*>(it.lp));
      if (res.second) { res.first->second = upload_ro(*it.lp); }
      ro = &res.first->second;
    }
    h_args[b] = marshal_problem(*it.lp, *it.settings, *it.vstatus, *it.ft, *it.basic_list,
                                *it.nonbasic_list, *it.sol, *it.dy_se, ro);
  }
  kernel_args* d_args = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_args, N * sizeof(kernel_args)));
  RAFT_CUDA_TRY(cudaMemcpy(d_args, h_args.data(), N * sizeof(kernel_args), cudaMemcpyHostToDevice));
  const int zero_flag = 0;
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_gpu_alloc_failed, &zero_flag, sizeof(int)));
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  timing.upload_s = toc(up_t0);

  // ---- Launch one block per problem, timed with CUDA events (device compute only) ----
  // CUOPT_GPU_WARP (unset/"0" = single-thread <<<N,1>>>; else the warp path <<<N,32>>>, one 32-lane
  // warp per problem — same env mapping as the single-problem launch: "1" plain / "2" big≥32).
  // Warp results are numeric-equivalent, NOT bit-exact, so the caller must relax its
  // GPU-vs-reference verification when CUOPT_GPU_WARP is set.
  const char* warp_env  = std::getenv("CUOPT_GPU_WARP");
  const bool  warp_mode = (warp_env != nullptr && std::string(warp_env) != "0");
  // Kernel-fission bit-exact reference: CUOPT_SEREP_OFF=1 disables the SE-drift reprice so the
  // monolith matches the reprice-free fissioned pipeline (FISSION_CHALLENGES.md #3).
  { // DEFAULT: reprice OFF (see the g_serep_off banner). CUOPT_SEREP_ON=1 restores it.
    // CUOPT_SEREP_OFF is still accepted so existing harnesses keep working — it now merely
    // reaffirms the default.
    const char* on      = std::getenv("CUOPT_SEREP_ON");
    const int   serep_off = (on != nullptr && std::string(on) != "0") ? 0 : 1;
    RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_serep_off, &serep_off, sizeof(int))); }
  { const char* e = std::getenv("CUOPT_RT_SERIAL");
    const int rt_serial = (e != nullptr && std::string(e) != "0") ? 1 : 0;
    RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_rt_serial, &rt_serial, sizeof(int))); }
  // CUOPT_GPU_REGCAP: warp-path register/occupancy target (throughput sweep). Maps to the kernel's
  // __launch_bounds__ MinBlocks so ptxas caps registers → more resident blocks/SM:
  //   unset / 255  → MinBlocks 8  (255 regs, ~8 blocks/SM   — current default)
  //   128          → MinBlocks 16 (128-reg cap, ~16 blocks/SM)
  //   64           → MinBlocks 32 (64-reg cap,  ~32 blocks/SM = HW max)
  // CUOPT_GPU_REGCAP is gone: it steered __launch_bounds__ MinBlocks on the parked-helper warp
  // kernel, which no longer exists. The Direct kernel's cap is -DCUOPT_MONO_DIRECT_MINBLOCKS.
  // Fix the shared-memory occupancy ceiling: our per-block static shared (~0.6 KB) + the 1 KB driver
  // reservation is tiny, but the DEFAULT carveout (32 KB on B200) only fits ~19 blocks. Prefer a larger
  // shared carveout so shared stops capping occupancy below 32 blocks/SM (leaves the rest as L1). This is
  // a hint the driver rounds to a supported bucket; harmless when occupancy is register-limited.
  auto raise_carveout = [](const void* fn) {
    // 40% of B200's ~228 KB unified cache ≈ 90 KB shared → ~55 blocks worth, well past 32; ~60% stays L1.
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 40);
  };
  cudaEvent_t ev_start, ev_stop;
  RAFT_CUDA_TRY(cudaEventCreate(&ev_start));
  RAFT_CUDA_TRY(cudaEventCreate(&ev_stop));
  RAFT_CUDA_TRY(cudaEventRecord(ev_start));
#if !CUOPT_SKIP_MONOLITH_WARP
  if (warp_mode) {
    const unsigned nb = static_cast<unsigned>(N);
    // The register/occupancy target is now a BUILD-time knob on the Direct kernel
    // (-DCUOPT_MONO_DIRECT_MINBLOCKS=16 -> 128 regs / 16 blocks/SM); see the note at
    // gpu_phase2_kernel_direct. The old runtime CUOPT_GPU_REGCAP sweep selected between separate
    // MinBlocks instantiations of the PARKED-HELPER warp kernel, which no longer exists.
    raise_carveout(reinterpret_cast<const void*>(&gpu_phase2_kernel_direct<>));
    gpu_phase2_kernel_direct<><<<nb, 32>>>(d_args);
  } else
#endif
  {
#if CUOPT_SKIP_MONOLITH_WARP
    (void)raise_carveout;  // warp block compiled out (single-thread-only build)
    if (warp_mode) {
      std::fprintf(stderr, "[gpu_phase2] CUOPT_GPU_WARP set but the warp kernel is not compiled "
                           "(CUOPT_SKIP_MONOLITH_WARP) — running the single-thread kernel.\n");
    }
#endif
    gpu_phase2_kernel<false><<<static_cast<unsigned>(N), 1>>>(d_args);
  }
  RAFT_CUDA_TRY(cudaEventRecord(ev_stop));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaEventSynchronize(ev_stop));
  float kernel_ms = 0.0f;
  RAFT_CUDA_TRY(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
  timing.kernel_s = kernel_ms / 1000.0;
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  // ---- Copy each problem's result back (host-timed) ----
  const double cb_t0 = tic();
  for (size_t b = 0; b < N; ++b) {
    const auto& a  = h_args[b];
    auto& sol      = *items[b].sol;
    const int n    = a.n;
    const int m    = a.m;
    int h_status = 0, h_iters = 0;
    double h_obj = 0.0, h_uobj = 0.0;
    RAFT_CUDA_TRY(cudaMemcpy(&h_status, a.out_status, sizeof(int), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(&h_obj, a.out_objective, sizeof(double), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(&h_uobj, a.out_user_objective, sizeof(double), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(&h_iters, a.out_iterations, sizeof(int), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(sol.x.data(), a.x, n * sizeof(double), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(sol.y.data(), a.y, m * sizeof(double), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(sol.z.data(), a.z, n * sizeof(double), cudaMemcpyDeviceToHost));
    sol.objective      = h_obj;
    sol.user_objective = h_uobj;
    sol.iterations     = h_iters;
    out_status[b]      = static_cast<dual::status_t>(h_status);
  }
  timing.copyback_s = toc(cb_t0);

  int alloc_failed = 0;
  RAFT_CUDA_TRY(cudaMemcpyFromSymbol(&alloc_failed, g_gpu_alloc_failed, sizeof(int)));
  if (alloc_failed) {
    timing.message = "a device allocation failed during the batch solve (heap too small) — results "
                     "are unreliable and will show up as verification mismatches";
    std::fprintf(stderr, "[gpu_phase2] batch WARNING: %s\n", timing.message.c_str());
  }

  for (size_t b = 0; b < N; ++b) { free_kernel_args(h_args[b], share_ro); }
  if (share_ro) { for (const auto& kv : ro_cache) { free_ro(kv.second); } }
  cudaFree(d_args);
}

// ---------------------------------------------------------------------------
// KERNEL-FISSION batch orchestrator (Phase 1a).
//
// Same contract as gpu_simplified_dual_phase2_batch above — N independent problems in, N results
// out — but instead of ONE monolithic kernel running each problem's whole pivot loop, the host
// drives a bulk-synchronous super-step loop over five per-primitive kernels:
//
//     Ksetup                        (once, all slots)
//     repeat:  K1 -> K2 -> K3 -> K4 -> K5   (all live slots)   until every slot has terminated
//
// Each kernel is a thin wrapper over the SAME __device__ primitives the monolith calls; only the
// control flow moved. Slots that terminate become no-ops (slot_state::status != UNSET) and the pool
// drains, exactly as the monolith batch does. Phase 1a: single-thread <<<N,1>>>, fixed batch (no
// refill), SE-drift reprice omitted, host owns the time limit.
//
// The monolith path (gpu_simplified_dual_phase2_batch) is untouched and remains the A/B baseline:
// run it with CUOPT_SEREP_OFF=1 to get the matching reprice-free reference for the bit-exact gate.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_fission(const std::vector<gpu_batch_item<i_t, f_t>>& items,
                                              size_t heap_estimate_bytes,
                                              std::vector<dual::status_t>& out_status,
                                              gpu_batch_timing& timing)
{
  using namespace gpu_phase2_detail;
  static_assert(sizeof(i_t) == sizeof(int) && sizeof(f_t) == sizeof(double),
                "gpu_simplified_dual_phase2_batch_fission is only instantiated for <int, double>");

  const size_t N = items.size();
  out_status.assign(N, dual::status_t::UNSET);
  timing = gpu_batch_timing{};
  if (N == 0) return;

  for (size_t b = 0; b < N; ++b) {
    if (items[b].ft->num_updates() != 0) {
      timing.rejected = true;
      timing.message  = "item " + std::to_string(b) + " has a non-fresh factorization";
      std::fprintf(stderr, "[gpu_phase2] fission batch ERROR: %s\n", timing.message.c_str());
      return;
    }
  }

  // ---- Pre-flight VRAM check (fail loud, same policy as the monolith batch) ----
  size_t static_bytes = 0;
  for (size_t b = 0; b < N; ++b) { static_bytes += problem_static_bytes(*items[b].lp, *items[b].ft); }
  const size_t args_bytes = N * sizeof(kernel_args) + N * sizeof(i_t);
  const size_t need = static_bytes + heap_estimate_bytes + args_bytes;
  size_t free_b = 0, total_b = 0;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_b, &total_b));
  timing.vram_needed_bytes = need;
  timing.vram_free_bytes   = free_b;
  const size_t budget = static_cast<size_t>(0.92 * static_cast<double>(free_b));
  if (need > budget) {
    const double GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "fission batch needs ~%.2f GB VRAM, only %.2f GB free (budget %.2f GB) for %zu problems",
                  need * GB, free_b * GB, budget * GB, N);
    timing.rejected = true;
    timing.message  = buf;
    std::fprintf(stderr, "[gpu_phase2] fission batch REJECTED: %s\n", buf);
    return;
  }
  ensure_device_limits(heap_estimate_bytes);

  { // DEFAULT: reprice OFF (see the g_serep_off banner). CUOPT_SEREP_ON=1 restores it.
    // CUOPT_SEREP_OFF is still accepted so existing harnesses keep working — it now merely
    // reaffirms the default.
    const char* on      = std::getenv("CUOPT_SEREP_ON");
    const int   serep_off = (on != nullptr && std::string(on) != "0") ? 0 : 1;
    RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_serep_off, &serep_off, sizeof(int))); }
  { const char* e = std::getenv("CUOPT_RT_SERIAL");
    const int rt_serial = (e != nullptr && std::string(e) != "0") ? 1 : 0;
    RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_rt_serial, &rt_serial, sizeof(int))); }
  { const int zero = 0; RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_gpu_alloc_failed, &zero, sizeof(int))); }

  // ---- Warp geometry for the whole fission pipeline ------------------------------------------
  // Driven by CUOPT_GPU_WARP, exactly like the monolith: every stage is warp or none is. The six
  // per-stage CUOPT_FISSION_*_WARP switches are gone — they existed to port and verify one stage
  // at a time, and every stage is now proven node-for-node identical to the monolith warp
  // (FISSION_CHALLENGES.md §18), so per-stage geometry has no remaining use.
  // Parsed BEFORE Ksetup launches: g_fission_warp gates the level-schedule build inside it.
  const char* fiss_warp_env = std::getenv("CUOPT_GPU_WARP");
  const bool  fiss_warp = (fiss_warp_env != nullptr && std::string(fiss_warp_env) != "0");
  // Ksetup builds the elimination-level schedules into the slot only when the solve stages need them.
  { const int fw = fiss_warp ? 1 : 0;
    RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_fission_warp, &fw, sizeof(int))); }
  // Dense-solve schedule variant, same env mapping as the monolith ("2" = big≥32). Set explicitly so
  // the fission run never inherits a stale value from an earlier monolith launch in this process.
  if (fiss_warp) {
    // Print BEFORE trusting any PASS: a stale binary silently ignoring the setting once produced a
    // meaningless 100/100 here. grep this line in the log.
    std::printf("[fission] WARP <<<N,32>>> — Ksetup + K1..K5\n");
    std::fflush(stdout);
  }

  // ---- Marshal (same as the monolith batch; RO sharing across identical LPs) ----
  const double up_t0 = tic();
  const char* share_env = std::getenv("CUOPT_GPU_SHARE_RO");
  const bool  share_ro  = (share_env == nullptr || std::string(share_env) != "0");
  std::vector<kernel_args> h_args(N);
  std::unordered_map<const void*, ro_upload> ro_cache;
  i_t* d_status = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_status, N * sizeof(i_t)));
  for (size_t b = 0; b < N; ++b) {
    const auto& it = items[b];
    if (share_ro) {
      auto res = ro_cache.try_emplace(static_cast<const void*>(it.lp));
      if (res.second) { res.first->second = upload_ro(*it.lp); }
    }
    h_args[b] = marshal_problem(*it.lp, *it.settings, *it.vstatus, *it.ft, *it.basic_list,
                                *it.nonbasic_list, *it.sol, *it.dy_se,
                                share_ro ? &ro_cache[static_cast<const void*>(it.lp)] : nullptr);
    h_args[b].status_arr = d_status;   // shared, indexed by blockIdx.x
  }
  kernel_args* d_args = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_args, N * sizeof(kernel_args)));
  RAFT_CUDA_TRY(cudaMemcpy(d_args, h_args.data(), N * sizeof(kernel_args), cudaMemcpyHostToDevice));
  timing.upload_s = toc(up_t0);

  // ---- Super-step loop ----
  const unsigned nb = static_cast<unsigned>(N);

  // ---- The super-step loop is a CUDA GRAPH WITH A DEVICE-CONTROLLED WHILE NODE ----------------
  // `manual_cuda_graph_t::run_while` builds a graph whose single conditional node loops on a handle
  // written from the device by gpu_fission_cond. The whole solve is therefore ONE graph launch: the
  // host never sees an iteration, so there is no per-super-step D2H, stream sync, or status poll.
  //
  // Unconditional — there is no opt-out. It is bit-exact against the old host-polled loop (0/100
  // differing on the 100 v6 nodes in every geometry) and never slower, so a fallback would only be
  // dead code. CUDA 12.9+ is a hard requirement of this repo, well above the 12.4 that conditional
  // nodes need, so there is no version guard either.
  //
  // Capture requires a non-default stream, hence the owned stream below.
  // Device-side super-step counter, read back once at the end. Allocated HERE, before the timed
  // region and before Ksetup is issued: cudaMalloc implicitly synchronizes the device, so allocating
  // it later would stall the host on Ksetup and be misattributed to graph capture.
  unsigned long long* d_steps = alloc_zeroed<unsigned long long>(1);

  rmm::cuda_stream stream_owner;
  const rmm::cuda_stream_view stream = stream_owner.view();

  // One iteration = K1..K5, issued in order. Host-idempotent: launches only.
  auto issue_super_step = [&] {
    cudaStream_t s = stream.value();
    if (fiss_warp) {
      gpu_fission_k1<true><<<nb, 32, 0, s>>>(d_args);
      gpu_fission_k2<true><<<nb, 32, 0, s>>>(d_args);
      gpu_fission_k3<true><<<nb, 32, 0, s>>>(d_args);
      gpu_fission_k4<true><<<nb, 32, 0, s>>>(d_args);
      gpu_fission_k5<true><<<nb, 32, 0, s>>>(d_args);
    } else {
      gpu_fission_k1<false><<<nb, 1, 0, s>>>(d_args);
      gpu_fission_k2<false><<<nb, 1, 0, s>>>(d_args);
      gpu_fission_k3<false><<<nb, 1, 0, s>>>(d_args);
      gpu_fission_k4<false><<<nb, 1, 0, s>>>(d_args);
      gpu_fission_k5<false><<<nb, 1, 0, s>>>(d_args);
    }
  };
  cuopt::manual_cuda_graph_t step_graph;

  cudaEvent_t ev_start, ev_stop;
  RAFT_CUDA_TRY(cudaEventCreate(&ev_start));
  RAFT_CUDA_TRY(cudaEventCreate(&ev_stop));
  RAFT_CUDA_TRY(cudaEventRecord(ev_start, stream.value()));

  // Ksetup geometry. Its prologue is nearly all allocation/RMW (lane 0 either way); the warp form
  // exists so the collectives inside it (WARP_NBE, WARP_PINF) can run on 32 lanes.
  // Deliberately OUTSIDE the graph: it runs exactly once, so capturing it would buy nothing, and it
  // is where the per-slot device allocation happens.
  if (fiss_warp) { gpu_fission_setup<true><<<nb, 32, 0, stream.value()>>>(d_args); }
  else            { gpu_fission_setup<false><<<nb, 1, 0, stream.value()>>>(d_args); }

  // Host owns the iteration/wall-clock bound (the device kernels have no time-limit check). The cap
  // mirrors the device iter_limit so a pathological batch cannot spin forever.
  size_t max_super_steps = 500000;
  if (const char* e = std::getenv("CUOPT_FISSION_MAX_STEPS")) { max_super_steps = std::strtoull(e, nullptr, 10); }
  size_t steps = 0;
  // The ENTIRE loop runs inside ONE graph launch: a cudaGraphCondTypeWhile node whose body is
  // K1..K5 + gpu_fission_cond. The condition kernel evaluates the same predicate the host loop used
  // to ("any slot still UNSET, and under the super-step cap") and writes it to the conditional
  // handle, so the host does no per-super-step work at all — no D2H, no stream sync, no poll.
  step_graph.run_while(stream, [&](cudaGraphConditionalHandle handle) {
    issue_super_step();
    gpu_fission_cond<<<1, 128, 0, stream.value()>>>(
      handle, d_status, static_cast<i_t>(N), d_steps, max_super_steps);
  });
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  
  RAFT_CUDA_TRY(cudaEventRecord(ev_stop, stream.value()));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaEventSynchronize(ev_stop));
  unsigned long long dev_steps = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&dev_steps, d_steps, sizeof(dev_steps), cudaMemcpyDeviceToHost));
  steps = static_cast<size_t>(dev_steps);
  RAFT_CUDA_TRY_NO_THROW(cudaFree(d_steps));
  float kernel_ms = 0.0f;
  RAFT_CUDA_TRY(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
  timing.kernel_s = kernel_ms / 1000.0;
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  if (steps >= max_super_steps) {
    timing.message = "fission batch hit the super-step cap (" + std::to_string(max_super_steps) +
                     ") — some problems did not terminate";
    std::fprintf(stderr, "[gpu_phase2] fission batch WARNING: %s\n", timing.message.c_str());
  }

  // ---- Copy results back (identical to the monolith batch) ----
  const double cb_t0 = tic();
  for (size_t b = 0; b < N; ++b) {
    const auto& a = h_args[b];
    auto& sol     = *items[b].sol;
    int h_st = 0, h_it = 0;
    double h_obj = 0.0, h_uobj = 0.0;
    RAFT_CUDA_TRY(cudaMemcpy(&h_st,  a.out_status,         sizeof(int),    cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(&h_obj, a.out_objective,      sizeof(double), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(&h_uobj,a.out_user_objective, sizeof(double), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(&h_it,  a.out_iterations,     sizeof(int),    cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(sol.x.data(), a.x, a.n * sizeof(double), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(sol.y.data(), a.y, a.m * sizeof(double), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(sol.z.data(), a.z, a.n * sizeof(double), cudaMemcpyDeviceToHost));
    sol.objective      = h_obj;
    sol.user_objective = h_uobj;
    sol.iterations     = h_it;
    out_status[b]      = static_cast<dual::status_t>(h_st);
  }
  timing.copyback_s = toc(cb_t0);

  int alloc_failed = 0;
  RAFT_CUDA_TRY(cudaMemcpyFromSymbol(&alloc_failed, g_gpu_alloc_failed, sizeof(int)));
  if (alloc_failed) {
    timing.message = "a device allocation failed during the fission batch solve (heap too small)";
    std::fprintf(stderr, "[gpu_phase2] fission batch WARNING: %s\n", timing.message.c_str());
  }

  for (size_t b = 0; b < N; ++b) { free_kernel_args(h_args[b], share_ro); }
  if (share_ro) { for (const auto& kv : ro_cache) { free_ro(kv.second); } }
  cudaFree(d_status);
  cudaFree(d_args);
}


template dual::status_t gpu_simplified_dual_phase2<int, double>(
  const lp_problem_t<int, double>&,
  const simplex_solver_settings_t<int, double>&,
  std::vector<variable_status_t>&,
  basis_update_mpf_t<int, double>&,
  std::vector<int>&,
  std::vector<int>&,
  lp_solution_t<int, double>&,
  std::vector<double>&);

template void gpu_simplified_dual_phase2_batch<int, double>(
  const std::vector<gpu_batch_item<int, double>>&,
  size_t,
  std::vector<dual::status_t>&,
  gpu_batch_timing&);

template void gpu_simplified_dual_phase2_batch_fission<int, double>(
  const std::vector<gpu_batch_item<int, double>>&,
  size_t,
  std::vector<dual::status_t>&,
  gpu_batch_timing&);

}  // namespace cuopt::linear_programming::dual_simplex
