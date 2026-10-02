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
//   * std::vector<T> created inside the kernel        -> gpu_span<T> (owning,
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
#include "gpu_branch_scores.cuh"
#include "pinned_host_arena.hpp"

#include <climits>
#include "gpu_lu.cuh"

#include "dense_triangle_solve.cuh"

#include <dual_simplex/basis_solves.hpp>

#include <linear_algebra/sparse_matrix.hpp>
#include <linear_algebra/sparse_vector.hpp>
#include <math_optimization/tic_toc.hpp>

#include <utilities/macros.cuh>
#include <utilities/manual_cuda_graph.cuh>   // run_while: the device-controlled super-step loop

#include <raft/core/nvtx.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/stream>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_memcpy.cuh>

#include <new>  // placement new: construct the per-slot persistent state in host-allocated memory

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <sys/mman.h>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <map>
#include <set>
#include <vector>

static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free,
              "the mapped completion word requires lock-free CPU atomics");

// The GPU used to store 1/mu and MULTIPLY when applying each stacked rank-1
// update; the CPU (basis_updates.cpp) stores mu and DIVIDES. That was two roundings against one, on
// every stacked update, on every T-loop pass -- and the result feeds the `utilde` that produces the
// NEXT mu, so it compounded through the stack. Dividing matches the CPU exactly.
// Together with dividing by the triangular diagonal, this removes the wrong answer on nexp node
// 783 without the dual rebuild that used to be needed.


// NO-FLIP FIX, part 1 of 2 -- DEVICE-ONLY (the CPU drivers are deliberately untouched).
//
// THE DEFECT. The ratio test computes (+-dual_tol/10 - z_j)/delta_z_j. A reduced cost that has
// drifted one ULP past the tolerance flips the numerator's sign, so the ratio comes out NEGATIVE.
// Selection is `ratio < min_val`, so a negative ratio beats every legitimate candidate WITH
// CERTAINTY -- it is a sorting bug, not a rare race. The resulting negative dual step is added to
// every nonbasic reduced cost, the pivot element is tiny, and the primal step divides by it. On
// badnode_m4640 that took dinf 1e-07 -> 0.022, |Ax-b| -> 8913, and drove the tracked objective to
// 292.10 past a cutoff of 282.0 -- fathoming a node whose true optimum is 49.58.
//
// THE FIX. Clamp the ratio at zero in warp_rt_body: fmax(ratio, 0). The candidate becomes a
// degenerate step, which is exactly what the bound-flipping breakpoint path has always produced.
// The cost is more degenerate pivots, not a lost candidate -- and that distinction is the whole
// point, because DROPPING such candidates can empty the eligible set, which the driver reports as
// DUAL_UNBOUNDED and branch-and-bound fathoms. See part 2 at the cutoff exit.
//
// The clamp is unconditional; there is no switch. A guard that can be compiled out invites a
// harness to measure the unguarded build by accident.
//
// NOTE: this BREAKS bit-exactness with simplified_dual_phase2 (dual_simplex/CLAUDE.md rule 2). The
// CPU minimal driver still takes negative steps; it is exposed to the same defect and is simply not
// guarded here. The CPU FULL driver is the reference and is not affected.
//
// REJECTED ALTERNATIVES, all measured, do not re-try without new evidence:
//   * reject negative candidates outright -> emptied the set -> false DUAL_UNBOUNDED (csched008
//     node 9304); adding an "emptied by rejection" -> NUMERICAL split fixed that but traded a wrong
//     answer for a large NUMERICAL rate.
//   * DS_RT_MIN_PIVOT (refuse a pivot below a magnitude): dead below pivot_tol=1e-7 by construction,
//     and at 1e-5 it refused a good pivot at iteration 29 of 57 on nexp node 783 with dinf=0.
//   * Lagrangian dual-bound certificates at the CUTOFF and OPTIMAL exits: correct, but they only
//     REFUSE to fathom after the iterate is already destroyed, so the solve then spun to 214,299
//     iterations, and most of the resulting NUMERICAL were false positives (node 783 bailed with
//     its objective right to 5.6e-13) because the bound was built from the drifted y.
//   * refuse OPTIMAL while the maintained z is dual infeasible: fires on 53-55% of seats.


#ifndef GPU_DUAL_UNBOUNDED_TRACE
#define GPU_DUAL_UNBOUNDED_TRACE 0
#endif

namespace cuopt::linear_programming::dual_simplex {
// Shared simplex primitives live in mathematical_optimization::simplex; tic/toc
// live in the parent mathematical_optimization namespace.
using namespace cuopt::mathematical_optimization;
using namespace cuopt::mathematical_optimization::simplex;
namespace gpu_phase2_detail {

using i_t = int;
using f_t = double;
namespace dts = dense_triangle_solve;  // shared dense-solve cores




// Next independent problem claimed by the persistent work-stealing grid. Reset by the host before
// every launch; only lane 0 of each block updates it, then publishes the claimed index to its warp.
// (the work-stealing counter is per launch, passed as a kernel argument -- not a device global)

// MEASURED AND REJECTED (2026-08-18): making the hypersparse-vs-dense routing threshold tunable
// per solve entry point. An earlier per-phase profile appeared to say the hard-coded 0.05 is badly
// mistuned (hypersparse l_solve mean 4.88M cycles against dense's 1.80M), but a sweep over four
// independent runtime thresholds says otherwise: all-dense 9.125 s and all-hypersparse 13.816 s
// against 0.759 s at 0.05, and every per-entry-point forcing was neutral-to-worse. Re-swept after
// the hypersparse l_solve tail was made collective, which inverts those per-call means (hypersparse
// now 1.08M vs dense 1.85M): 0.10 -> 5.014, 0.20 -> 5.168, 0.40 -> 5.154, 1.00 -> 5.200, all worse
// than 0.05's 4.725. The per-call means are not a valid basis for routing in EITHER direction --
// the dense rows exclude their to_dense/from_dense conversions, and the hypersparse calls are
// selected for small reach, so the two populations are not comparable. 0.05 stays.
// To re-run either sweep, override the threshold at build time and re-measure; 0.05 stays.



// ===== No-op gather column elision =========================================================
// The four warp triangular solves are all GATHER form (see the operator table below): column j of
// the gather matrix computes x[j] = (x[j] - sum over the OFF-DIAGONAL span) / diag. When that span
// is empty and the diagonal is exactly 1.0 the column writes back exactly what it read, so it can
// be dropped from the wavefront schedule outright. Nothing downstream can tell "wrote the same
// value" from "did not write": the dense path leaves x[j] as it found it, and the hypersparse path
// leaves the value warp_sptrsv_preload_body put there (the scattered rhs, or 0), which is what the
// solve would have produced. Bit-identical -- pivot counts must not move.
//
// Measured over the 100 v6 nodes this is 90.8% of L-solve columns, 83.0% of L-transpose, 73.2% of
// U-transpose and 29.4% of U-solve -- the gather form excludes the diagonal from the inner loop,
// and for most columns of these factors there is nothing else in the column.
//
// The flag is carried in the SIGN BIT of the existing per-column level array
// (warp_solve_config::level[op]). Both consumers already load that word, so the classification
// costs no extra memory traffic. Levels are small (max ~413 over this workload). Every read of the
// array must go through factor_level().
//
// DEFAULT ON since 2026-08-21, after measuring -3.6% (4.186 -> 4.035 s) over three independent
// A/Bs with pivots, bit-exact count, registers (128) and smem (1016 B) all unchanged, racecheck
// clean and the <<<1,1>>> reference still 100/100 bit-exact.
// Consumed ONLY in warp_build_levels -- the lane-0 prologue -- so unlike a flag in the wavefront
// inner loop it costs no registers on the hot path and one binary measures both arms.

// REORDERING THE LEVEL SCHEDULE IS A DEAD END, measured 2026-08-26. Emitting each level by
// descending column length instead of ascending index was built and measured at only -1.0%, and
// removed as not worth an LSD radix pass in the lane-0 prologue. The general reason: a level's cost
// equals `max(longest column, ceil(work/32))` to within 1-5% as scheduled today, so a level costs
// the length of its single longest column and no permutation can beat that. SELL-C-sigma and
// two-columns-per-lane were also built and lost (+7.5%, +7.2%).


static constexpr i_t kColNoop = static_cast<i_t>(0x80000000u);  // level[] sign bit
static constexpr i_t kSparseReachDenseFallbackDivisor = 20;     // dense at >= 5% reach
// Runtime override of the divisor (CUOPT_BNB_GPU_DENSE_FALLBACK_DIV; 0 = never fall back to dense).
__constant__ i_t c_dense_fallback_divisor = kSparseReachDenseFallbackDivisor;
// Runtime override of the hypersparse routing threshold (CUOPT_BNB_GPU_HYPERSPARSE_THRESHOLD).
__constant__ f_t c_hypersparse_threshold = 0.05;
__constant__ int c_reuse_levels = 1;
__constant__ int c_derive_levels = 1;  // CUOPT_BNB_GPU_DERIVE_LEVELS
__constant__ int c_warp_levels = 1;    // CUOPT_BNB_GPU_WARP_LEVELS
__constant__ int c_ftran_direct = 0;   // CUOPT_EXP_FTRAN=direct: fission FTRAN through L0/U0
__constant__ int c_long_col = 32;      // CUOPT_BNB_GPU_LONG_COL: longer gather columns go to 8-lane groups
// Transpose dz route at delta_y density <= this percent (CPU rule 30; CUOPT_BNB_GPU_DZ_DENSE_PCT).
__constant__ f_t c_dz_sparse_pct = 30.0;
__device__ __forceinline__ i_t  factor_level(i_t lv) { return lv & 0x7fffffff; }
__device__ __forceinline__ bool level_is_noop(i_t lv) { return lv < 0; }


// CHASING REGISTERS BELOW 128 IS NOT WORTH IT, measured 2026-08-26. Two routes were built. Hoisting
// the one-time prologue into its own <<<N,32>>> launch reached 102 regs (the largest saving found
// here) but cost +14.3%, because a kernel boundary is a global barrier and all N setups must finish
// before ANY pivot loop starts. And __launch_bounds__(32,17) DOES reach the next occupancy step --
// 96 regs, 20 blocks/SM -- for +23%, as the extra spill (112 -> 176 B) outweighs the 1.25x
// occupancy. The 97..128 register band is one step at 16 blocks/SM, so saving registers inside it
// buys nothing and crossing out of it measures worse.

// ===== __launch_bounds__: only on the Direct warp monolith (see MinBlocks note at the kernel) ====================================
// Removed 2026-08-11. Challenge #13 measured that `__launch_bounds__(maxThreadsPerBlock=1, N)`
// COSTS ~40 registers per kernel: it tells ptxas there is no occupancy pressure, so it spends
// registers freely toward the 255 cap. That is why it is not used here; the
// monolith did, and now does not — ptxas's own heuristic is used everywhere.
//
// Consequence: `MinBlocks` is now an inert template parameter, and `CUOPT_GPU_REGCAP` (which only
// ever steered it through the launch bound) no longer caps anything. Both are kept so the sweep
// scaffolding can be restored in one line if a cap is ever wanted again.


__device__ __forceinline__ f_t dev_inf() { return cuda::std::numeric_limits<f_t>::infinity(); }
__device__ __forceinline__ f_t dev_nan() { return cuda::std::numeric_limits<f_t>::quiet_NaN(); }

// WHY THIS EXISTS. Everything a solve allocates used to come from the in-kernel malloc() heap.
// That heap is a fixed arena the driver caps at ~16 GB on EVERY device (measured by bisection on a
// 12 GB RTX 3500 Ada and on a 178 GB B200; usable is ~0.82x granted, so ~13.1 GB). It is not a
// fraction of device memory, and no environment variable raises it. Ordinary cudaMalloc VRAM on the
// same B200 is 177 GB free. So the batch size was being limited by the smallest pool on the card
// while the largest sat idle.
//
// Most of what a solve allocates is FIXED SIZE -- m-, n- or nz-sized workspaces the host already
// knows the dimensions of, because it computes exactly those sizes for its own VRAM pre-flight.
// ---------------------------------------------------------------------------
// gpu_span<T> : a NON-OWNING, FIXED-CAPACITY device buffer with a mutable length.
//
// This is the only buffer type in the kernel and it NEVER allocates. Storage comes from exactly
// two places:
//   * attach(ptr, capacity) -- a slice of the slot memory the host preallocated
//     (one rmm allocation per buffer; slot_state::attach_all_buffers wires them). Length starts
//     at 0 and is moved by
//     resize / assign / push_back / clear.
//   * view(ptr, n)          -- a zero-copy view over a host-uploaded array; length == capacity.
//
// It is deliberately NOT called a vector and has no sizing constructor. Its predecessor looked
// exactly like std::vector, and `x = gpu_span<T>(n)` -- an utterly ordinary-looking whole-object
// assignment -- silently discarded a preallocated buffer and sent it back to growing from zero.
// There is no allocator left to absorb that, so exceeding cap_ is a hard, loud error rather than a
// quiet regrow: it means the buffer was allocated too small.
// ---------------------------------------------------------------------------
template <typename T>
struct gpu_span {
  T* data_     = nullptr;
  size_t size_ = 0;
  size_t cap_  = 0;

  __host__ __device__ gpu_span() = default;
  // View: length == capacity. Every host-uploaded array arrives through this.
  __host__ __device__ gpu_span(T* ptr, size_t n) : data_(ptr), size_(n), cap_(n) {}

  __device__ void view(T* ptr, size_t n) { data_ = ptr; size_ = n; cap_ = n; }
  __device__ void attach(T* ptr, size_t capacity) { data_ = ptr; size_ = 0; cap_ = capacity; }

  // SHALLOW const, exactly like std::span: a const span is a const HANDLE, not a handle to const
  // elements. Constness of the view does not propagate to what it views, which is what lets the
  // const solve helpers write through their scratch spans. Element constness is spelled in T
  // (gpu_span<const i_t>), not on the span.
  __host__ __device__ T* data() const { return data_; }
  __host__ __device__ size_t size() const { return size_; }
  __host__ __device__ size_t capacity() const { return cap_; }
  __host__ __device__ bool empty() const { return size_ == 0; }
  __host__ __device__ T& operator[](size_t k) const { return data_[k]; }
  __host__ __device__ T* begin() const { return data_; }
  __host__ __device__ T* end() const { return data_ + size_; }
  __device__ T& back() const { return data_[size_ - 1]; }

  __device__ void clear() { size_ = 0; }
  __device__ void set_size(size_t n) { size_ = n; }

  // Capacity is fixed at allocation; these only move the LENGTH. Needing more than was allocated is
  // a sizing bug in the host-side per-slot allocation, not a condition to recover from, so it is a
  // debug-build assert rather than a runtime signal. The callers still skip the write in release so
  // an undersized buffer degrades to a wrong answer rather than to memory corruption in a
  // neighbouring seat.
  __device__ void require_capacity(size_t n) const
  {
    cuopt_assert(n <= cap_, "gpu_span: write exceeds the capacity allocated for this buffer");
  }
  __device__ void reserve(size_t n) { require_capacity(n); }
  __device__ void resize(size_t n)
  {
    if (n > cap_) { require_capacity(n); return; }
    // TODO: is this necessary?
    for (size_t k = size_; k < n; ++k) { data_[k] = T(); }
    size_ = n;
  }
  __device__ void resize(size_t n, T val)
  {
    if (n > cap_) { require_capacity(n); return; }
    for (size_t k = size_; k < n; ++k) { data_[k] = val; }
    size_ = n;
  }
  __device__ void assign(size_t n, T val)
  {
    // TODO: all those n > cap should be debug only
    if (n > cap_) { require_capacity(n); return; }
    for (size_t k = 0; k < n; ++k) { data_[k] = val; }
    size_ = n;
  }
  __device__ void push_back(const T& v)
  {
    if (size_ + 1 > cap_) { require_capacity(size_ + 1); return; }
    data_[size_++] = v;
  }
  // Exchange storage with another span (the sparse-vector sort swaps a vector with its scratch).
  // Nothing owns memory, so this just moves pointers.
  __device__ void swap(gpu_span& o)
  {
    T* d = data_;     data_ = o.data_; o.data_ = d;
    size_t z = size_; size_ = o.size_; o.size_ = z;
    size_t c = cap_;  cap_  = o.cap_;  o.cap_  = c;
  }
};

// TODO: not SoL but needs dynamic size sort
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
// Non-owning sparse-matrix view (CSC). Backs the LP matrix and the fixed LU factors / their
// transposes. col_start is read-only because reach()/DFS keeps visited state in a private bitset
// workspace instead of temporarily modifying the factor.
// ---------------------------------------------------------------------------
// All matrix row indices, including the four immutable LU factors, use i_t.
struct gpu_csc_view {
  i_t m{0}, n{0}, nz_max{0};
  gpu_span<const i_t> col_start;
  gpu_span<i_t> i;
  gpu_span<f_t> x;
  __device__ i_t nnz() const { return col_start[n]; }
};

// Owning sparse-matrix (CSC). Used only for S_ (the MPF rank-1 update store),
// which grows one (u,v) column-pair per iteration.
struct gpu_csc_matrix {
  i_t m{0}, n{0}, nz_max{0};
  gpu_span<i_t> col_start;
  gpu_span<i_t> i;
  gpu_span<f_t> x;

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

};

// Owning sparse-matrix (CSR). Backs Arow — the row-wise copy of A used by the
// transpose delta_z path. j/x are swapped in place (nonbasic partition), so this is
// owning/mutable, not a span.
struct gpu_csr_matrix {
  i_t m{0}, n{0};
  gpu_span<i_t> row_start;
  gpu_span<i_t> j;
  gpu_span<f_t> x;

  __device__ gpu_csr_matrix() {}
  __device__ gpu_csr_matrix(const gpu_csr_matrix&)            = delete;
  __device__ gpu_csr_matrix& operator=(const gpu_csr_matrix&) = delete;

  // Build the CSR copy of a CSC matrix. Mirrors csc_matrix_t::to_compressed_row
  // exactly (row entries end up ordered by ascending column index), so the delta_z
  // accumulation order matches the CPU bit-for-bit.
  // Zero-copy: adopt a host-built CSR already in device memory as non-owning views. row_start stays
  // read-only and may be shared; j/x are mutated in place by the nonbasic partition, so those two
  // buffers must be fresh per-solve mutable copies.
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
struct gpu_basis_update_t;

// ---------------------------------------------------------------------------
// gpu_sparse_vector : owning sparse vector (created/grown inside the kernel).
// Mirrors sparse_vector_t.
// ---------------------------------------------------------------------------
struct gpu_sparse_vector {
  i_t n{0};
  gpu_span<i_t> i;
  gpu_span<f_t> x;

  __device__ void attach(i_t* pi, f_t* px, size_t capacity)
  {
    i.attach(pi, capacity); x.attach(px, capacity);
  }

  __device__ gpu_sparse_vector() {}
  // Dimension only. Storage is never allocated here -- it arrives via attach().
  __device__ gpu_sparse_vector(i_t n_, i_t /*nz*/) : n(n_) {}
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

  // Sort the sparse vector by index using caller-provided reused scratch (no per-call device malloc):
  // an O(n) bucket sort when dense (>30% nnz), else an n·log·n sort of a permutation then a gather.
  // The reused bucket is re-zeroed via assign (owning buffer, no realloc); the n·log·n path gathers
  // into scratch and swaps (preserving both buffers).
  // TODO: not SoL but needs dynamic size sort
  __device__ void sort(gpu_span<f_t>& bucket, gpu_span<i_t>& i_scr,
                       gpu_span<f_t>& x_scr, gpu_span<i_t>& perm)
  {
    if (i.size() == 1) { return; }
    const i_t nz = i.size();
    if (i.size() > 0.3 * n) {
      bucket.assign(n, 0.0);  // reused (was gpu_span<f_t> bucket(n,0.0)); O(n) re-zero, no malloc
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

  // ---- warp siblings for the `Direct` path ----------------------------------------------
  // On the Direct path all 32 lanes are already in the kernel body, so these need no op-code and no
  // cfg_shared dispatch — just a strided loop. Both are element-wise maps over DISTINCT destinations
  // (a gpu_sparse_vector's indices are unique), so the strided form is bit-identical to the serial
  // one: no lane pair touches the same slot, and no arithmetic is reordered.
  // Direct=false forwards to the serial body — the monolith MUST take that arm, because lanes 1-31
  // Direct=false is now only the SINGLE-THREAD path (blockDim 1), where the strided form and its
  // __syncwarp would be wrong; the parked-helper warp model that also required it is gone.
  __device__ void warp_negate()
  {
    const int lane = static_cast<int>(threadIdx.x);
    const i_t nz   = x.size();
    for (i_t k = lane; k < nz; k += 32) { x[k] *= -1.0; }
    __syncwarp(0xffffffffu);
  }

  // scatter() accumulates (`+=`), but with distinct indices no two lanes hit the same x_dense slot,
  // so this needs no atomics. Callers must still honour the "x_dense already cleared" precondition.
  __device__ void warp_scatter(f_t* x_dense) const
  {
    const int lane = static_cast<int>(threadIdx.x);
    const i_t nz   = i.size();
    for (i_t k = lane; k < nz; k += 32) { x_dense[i[k]] += x[k]; }
    __syncwarp(0xffffffffu);
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
// The four dense triangular-solve operators the warp path parallelizes, {L, U, Lt, Ut} = {0,1,2,3}.
// Their values are the indices into every per-operator array in warp_solve_config, and the order in
// which warp_build_config fills them. They are compile-time template arguments
// (warp_dispatch<Op> -> dts::wavefront_solve_op<Op>), never runtime values.
//
// These were part of a 16-entry `enum warp_solve_op` naming every collective, because the
// parked-helper model posted an OP CODE to the waiting lanes. That dispatch is gone -- each
// collective is now called directly -- so the other twelve enumerators had no use left anywhere.
static constexpr int kNumWarpSolveOps = 4;
static constexpr int WARP_L_SOLVE = 0, WARP_U_SOLVE = 1, WARP_LT_SOLVE = 2, WARP_UT_SOLVE = 3;

// The two schedule flags warp_build_levels needs, for op {L, U, Lt, Ut} = {0, 1, 2, 3}.
// diag_first: the column's diagonal is the FIRST entry of its CSC column rather than the last.
// ascending : the level pull walks columns j = 0..m-1 rather than m-1..0.
// constexpr FUNCTIONS, not arrays: a namespace-scope constexpr array has no device storage, so
// device code cannot index it. These are usable from host and device and in constant expressions.
__host__ __device__ constexpr bool op_diag_first(int op) { return op == 1 || op == 2; }
__host__ __device__ constexpr bool op_ascending(int op) { return op == 0 || op == 3; }

// Trivially constructible (no member initializers) so it can be a __shared__
// variable. Lane 0 fills every field (warp_build_config + warp_dispatch) before
// any lane reads it, ordered by the first __syncwarp.
struct warp_solve_config_t {
  // ONLY the elimination-level SCHEDULE survives. Every per-dispatch REQUEST field (x/op and the
  // tl_/dz_/dzs_/td_/fd_/rt_/su_/pi_/ud_/nbe_/sp_op/sp_nlev groups, ~95 fields, ~700 of 928 bytes)
  // was staging for the parked-helper dispatch and was deleted with it on 2026-08-11: the Direct
  // path passes those values as call arguments to the warp_*_body collectives instead.
  // Also gone: diag_first/descending/incremental/m, which warp_build_config wrote but nothing ever
  // read — the four wavefront bodies encode their own direction, so nothing consumes them.
  // Built once (by the kernel prologue on lane 0), read-only after.
  const i_t* cs[kNumWarpSolveOps];      // gather CSC col_start
  const i_t* ridx[kNumWarpSolveOps];    // gather CSC row indices
  const f_t* val[kNumWarpSolveOps];     // gather CSC values
  const i_t* lstart[kNumWarpSolveOps];  // level offsets [nlev+1]
  const i_t* corder[kNumWarpSolveOps];  // columns by level [m]
  const i_t* level[kNumWarpSolveOps];   // per-column elimination level [m] (reach-restricted solve)
  i_t        nlev[kNumWarpSolveOps];
  // reach-restricted sparse gather solve: warp_dispatch_sptrsv buckets the reachable set into
  // sp_ls/sp_co using sp_pos as write-cursors, then runs the wavefront over that sparse schedule.
  i_t* sp_ls;    // sparse level offsets [sp_nlev+1] (built per solve)
  i_t* sp_co;    // reachable columns bucketed by level [|R|] (built per solve)
  i_t* sp_pos;   // scratch write-cursors [sp_nlev+1] (built per solve)
  // Lane-0 DFS hot stack. In the monolith this config is shared memory, so the active DFS frames
  // avoid round trips through the global xi/pstack workspace. The measured maximum depth is 47;
  // depth 64 safely falls back to the general global-stack DFS below.
  i_t dfs_node[64];
  i_t dfs_next[64];
  i_t dfs_end[64];
};

// Collective (all-32-lane) dense solve of operator cfg.op on x = cfg.x. Both modes do exactly
// cfg.nlev[op] internal __syncwarps, so the persistent-helper dispatch count is mode-independent.

// ===== Parameterized collective BODIES ======================================================
// Each body takes ONLY what it needs as plain parameters (a_ prefix avoids collisions with locals).
// TWO call paths, ONE copy of the math:
//   * monolith : warp_collective_X(cfg) below is a thin unpacker -> calls the body.
//                op-code dispatch, so ptxas sees ONE known body per call site instead of the
//                10-branch union that pins the warp monolith at 250 registers.
// __forceinline__ is REQUIRED: it collapses the wrapper's unpack-then-pass back to the original
// code so the monolith must not get slower (verified by A/B benchmark, not assumed).
// ===========================================================================================

// fwd decls: the bodies are hoisted above these helpers' definitions.
__device__ __forceinline__ f_t warp_reduce_add(f_t v);
__device__ __forceinline__ f_t warp_reduce_min(f_t v);

// COMPILE-TIME operator variant, for the `Direct` path where the op is a literal at the call
// site. Nothing about the solve is decided at runtime: no op code, no flag loads, no per-column
// branches — ptxas emits one specialized wavefront body per call site.
template <int Op>
__device__ __forceinline__ void warp_solve_body_op(const i_t* a_corder, const i_t* a_cs,
                                                   const i_t* a_lstart, i_t a_nlev,
                                                   const i_t* a_ridx,
                                                   const f_t* a_val, f_t* a_x)
{
  dts::wavefront_solve_op<Op, true>(a_cs, a_ridx, a_val, a_x, a_lstart, a_corder, a_nlev);
}


template <int Op>
__device__ __forceinline__ void warp_solve2_body_op(const i_t* a_corder, const i_t* a_cs,
                                                    const i_t* a_lstart, i_t a_nlev,
                                                    const i_t* a_ridx,
                                                    const f_t* a_val, f_t* a_x0, f_t* a_x1)
{
  dts::wavefront_solve2_op<Op>(
    a_cs, a_ridx, a_val, a_x0, a_x1, a_lstart, a_corder, a_nlev);
}

__device__ __forceinline__ void warp_tloop_body(bool a_tl_asc, i_t a_tl_axpyoff, const i_t* a_tl_cs, i_t a_tl_dotoff, const f_t* a_tl_inv_mu, i_t a_tl_num, const i_t* a_tl_ri, f_t a_tl_tol, const f_t* a_tl_val, f_t* a_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t num  = a_tl_num;
  f_t* x         = a_x;
  for (i_t s = 0; s < num; ++s) {
    const i_t k    = a_tl_asc ? s : (num - 1 - s);
    const i_t dcol = 2 * k + a_tl_dotoff;
    const i_t acol = 2 * k + a_tl_axpyoff;
    const f_t inv_mu = a_tl_inv_mu[k];
    const i_t dc0  = a_tl_cs[dcol];
    const i_t dc1  = a_tl_cs[dcol + 1];
    // Short-column fast path. The T-loop is 16.2% of all block cycles and its dot products run on
    // the product-form update columns, which are mostly 0-2 entries long. For those the 5-level
    // butterfly (10 shfl instructions for a double) costs far more than the arithmetic it reduces,
    // and it sits on the loop's serial dependency chain: update k+1's dot cannot start until update
    // k's axpy has landed, so this latency is not hidden by other work.
    //
    // BIT-IDENTICAL, not merely equivalent, and only for len <= 2. With len entries only lanes
    // 0..len-1 hold a nonzero partial, and the butterfly's accumulation order for len <= 2 is
    // exactly v0 (len==1) or v0 + v1 (len==2) -- the same order every lane computes directly here.
    // At len == 3 the butterfly folds as (v0 + v2) + v1, which a serial v0 + v1 + v2 does NOT
    // reproduce, so the fast path stops at 2 and everything longer takes the original butterfly.
    // The redundant loads are warp-uniform, so they broadcast out of L1.
    const i_t dlen = dc1 - dc0;
    f_t dot;
    if (dlen <= 2) {
      dot = 0.0;
      if (dlen >= 1) { dot = a_tl_val[dc0] * x[a_tl_ri[dc0]]; }
      if (dlen == 2) { dot += a_tl_val[dc0 + 1] * x[a_tl_ri[dc0 + 1]]; }
      // REQUIRED. In the butterfly arm warp_reduce_add's __shfl_xor_sync is what separates this
      // update's dot READS of x from its axpy WRITES to x; the u_col and v_col row sets overlap, so
      // without a convergence point here a lane that finished its dot could store into x[i] while
      // another lane is still loading x[i] for the same dot. Independent thread scheduling makes
      // that reachable on sm_100. Dropping this is the missing-__syncwarp bug class.
      __syncwarp(0xffffffffu);
    } else {
      f_t partial = 0.0;
      for (i_t p = dc0 + lane; p < dc1; p += 32) { partial += a_tl_val[p] * x[a_tl_ri[p]]; }
      dot = warp_reduce_add(partial);   // every lane gets the full sum
    }
    const f_t theta = dot / inv_mu;
    if (fabs(theta) > a_tl_tol) {
      const i_t ac0 = a_tl_cs[acol];
      const i_t ac1 = a_tl_cs[acol + 1];
      for (i_t p = ac0 + lane; p < ac1; p += 32) { x[a_tl_ri[p]] += (-theta) * a_tl_val[p]; }
    }
    __syncwarp(0xffffffffu);   // x writes visible before the next update's dot reads them
  }
}


// Two dense RHS replay the same product-form update stack together. Factor metadata and values are
// loaded once; the two reductions preserve the one-RHS accumulation order independently.
template <bool Ascending, i_t AxpyOffset, i_t DotOffset>
__device__ __forceinline__ void warp_tloop2_body(const i_t* a_cs,
                                                 const f_t* a_mu,
                                                 i_t a_num,
                                                 const i_t* a_ri,
                                                 f_t a_tol,
                                                 const f_t* a_val,
                                                 f_t* a_x0,
                                                 f_t* a_x1)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t step = 0; step < a_num; ++step) {
    const i_t k    = Ascending ? step : (a_num - 1 - step);
    const i_t dcol = 2 * k + DotOffset;
    const i_t acol = 2 * k + AxpyOffset;
    const i_t d0   = a_cs[dcol];
    const i_t d1   = a_cs[dcol + 1];
    const i_t dlen = d1 - d0;
    f_t dot0, dot1;
    if (dlen <= 2) {
      dot0 = 0.0;
      dot1 = 0.0;
      if (dlen >= 1) {
        const i_t i = a_ri[d0];
        const f_t a = a_val[d0];
        dot0 = a * a_x0[i];
        dot1 = a * a_x1[i];
      }
      if (dlen == 2) {
        const i_t i = a_ri[d0 + 1];
        const f_t a = a_val[d0 + 1];
        dot0 += a * a_x0[i];
        dot1 += a * a_x1[i];
      }
      __syncwarp(0xffffffffu);
    } else {
      f_t part0 = 0.0;
      f_t part1 = 0.0;
      for (i_t q = d0 + lane; q < d1; q += 32) {
        const i_t i = a_ri[q];
        const f_t a = a_val[q];
        part0 += a * a_x0[i];
        part1 += a * a_x1[i];
      }
      dot0 = warp_reduce_add(part0);
      dot1 = warp_reduce_add(part1);
    }
    const f_t mu = a_mu[k];
    const f_t theta0 = dot0 / mu;
    const f_t theta1 = dot1 / mu;
    const i_t a0     = a_cs[acol];
    const i_t a1     = a_cs[acol + 1];
    if (fabs(theta0) > a_tol) {
      for (i_t q = a0 + lane; q < a1; q += 32) {
        a_x0[a_ri[q]] += (-theta0) * a_val[q];
      }
      __syncwarp(0xffffffffu);
    }
    if (fabs(theta1) > a_tol) {
      for (i_t q = a0 + lane; q < a1; q += 32) {
        a_x1[a_ri[q]] += (-theta1) * a_val[q];
      }
      __syncwarp(0xffffffffu);
    }
  }
}

// `dz` stays zero for every basic column between pivots: setup zeroes the whole vector, and
// clear_delta_z zeroes both columns whose basic/nonbasic status changes. Every other basic column
// is untouched by the nonbasic loop below, so sweeping all m basics here only rewrites zeros.
__device__ __forceinline__ void warp_dz_body(const i_t* a_dz_Acs, const i_t* a_dz_Ari, const f_t* a_dz_Aval, i_t* a_dz_count, i_t a_dz_dir, const f_t* a_dz_dy, i_t* a_dz_indices, i_t a_dz_leaving, i_t a_dz_m, i_t* a_dz_mark, i_t a_dz_n, const i_t* a_dz_nonbasic, f_t* a_dz_out)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t m = a_dz_m, n = a_dz_n;
  f_t* dz = a_dz_out;
  // `leaving` is the only basic column touched here and no other lane accesses it before the
  // trailing __syncwarp publishes the store.
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
  *a_dz_count = count;  // every lane: count is warp-uniform, so no broadcast is needed (see lane0_bcast)
  // Publish every global write above (dz, a_dz_indices, a_dz_mark, and the lane-0 dz[leaving]) to
  // the rest of the warp. The caller's next collective reads them from ALL lanes, so without this
  // the reader may see a slot its writer has not stored yet -- a plain cross-lane RAW hazard.
  __syncwarp(0xffffffffu);
}

__device__ __forceinline__ void warp_dz_scatter_body(i_t* a_dz_count, i_t a_dz_dir, i_t* a_dz_indices, i_t a_dz_leaving, i_t* a_dz_mark, f_t* a_dz_out, const i_t* a_dzs_dyi, const f_t* a_dzs_dyx, const i_t* a_dzs_j, const i_t* a_dzs_basic_mark, i_t a_dzs_nzdy, const i_t* a_dzs_rs, const f_t* a_dzs_x)
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
    const i_t re = a_dzs_rs[i + 1];
    for (i_t base = rs; base < re; base += 32) {
      const i_t p = base + lane;
      i_t j = -1; bool newcol = false;
      if (p < re) {
        const i_t jj = a_dzs_j[p];
        if (a_dzs_basic_mark[jj] < 0) {
          j = jj;
          dz[j] -= dyi * a_dzs_x[p];           // distinct j within the row -> no collision
          if (!a_dz_mark[j]) { a_dz_mark[j] = 1; newcol = true; }
        }
      }
      const unsigned ball = __ballot_sync(0xffffffffu, newcol);
      const int prefix    = __popc(ball & ((1u << lane) - 1u));
      if (newcol) { a_dz_indices[count + prefix] = j; }  // increasing k/lane == serial append order
      count += __popc(ball);
    }
  }
  *a_dz_count = count;  // every lane: count is warp-uniform, so no broadcast is needed (see lane0_bcast)
  // Converge before the single-lane store: the loop above wrote dz from all 32 lanes.
  __syncwarp(0xffffffffu);
  if (lane == 0) { dz[a_dz_leaving] = static_cast<f_t>(a_dz_dir); }
  // ...and publish it, plus a_dz_indices/a_dz_mark, before the caller's next collective reads them
  // from every lane. Without this the reader can race the writer (cross-lane RAW).
  __syncwarp(0xffffffffu);
}

__device__ __forceinline__ void warp_todense_body(f_t* a_td_dense, const i_t* a_td_i, i_t a_td_n, i_t a_td_nz, const f_t* a_td_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t n = a_td_n;
  for (i_t k = lane; k < n; k += 32) { a_td_dense[k] = 0.0; }
  __syncwarp(0xffffffffu);
  const i_t nz = a_td_nz;
  for (i_t k = lane; k < nz; k += 32) { a_td_dense[a_td_i[k]] = a_td_x[k]; }
  // Publish the dense buffer before the solve that consumes it reads it from every lane.
  __syncwarp(0xffffffffu);
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
  *a_fd_count = count;  // every lane: count is warp-uniform (same ballot on every lane)
  // Publish a_fd_i / a_fd_x before the caller's next collective reads them from every lane.
  __syncwarp(0xffffffffu);
}


// Count-only form for a dense result that will be consumed in place. It preserves from_dense's
// exact v != 0.0 definition but avoids prefix computation and sparse index/value stores.
__device__ __forceinline__ i_t warp_dense_nnz(const f_t* x, i_t n)
{
  const int lane = static_cast<int>(threadIdx.x);
  i_t count = 0;
  for (i_t base = 0; base < n; base += 32) {
    const i_t k = base + lane;
    count += __popc(__ballot_sync(0xffffffffu, k < n && x[k] != 0.0));
  }
  return count;
}

__device__ __forceinline__ void warp_rt_body(f_t a_rt_dual_tol,
                                                     const f_t* a_rt_dz,
                                                     i_t* a_rt_entering,
                                                     i_t* a_rt_k,
                                                     f_t a_rt_pivot_tol,
                                                     f_t* a_rt_step,
                                                     const variable_status_t* a_rt_vstatus,
                                                     const f_t* a_rt_z,
                                                     f_t a_rt_zero_tol,
                                                     const i_t* a_rt_nonbasic,
                                                     i_t a_rt_num_nonbasic)
{
  const int lane = static_cast<int>(threadIdx.x);
  const f_t ptol = a_rt_pivot_tol, dtol = a_rt_dual_tol, ztol = a_rt_zero_tol;
  const f_t INF  = dev_inf();

  // The CPU near-tie rule updates min_val even when it accepts a slightly larger ratio, making the
  // result order-dependent. Evaluate consecutive nonbasic positions in parallel, then visit the
  // eligible lanes in ascending order. The shuffles avoid materializing and rereading a dense
  // ratio array while keeping the lane-0 state transitions identical to the CPU scan.
  i_t entering = -1, nonbasic_entering = -1;
  f_t step_length = INF, min_val = INF;
  for (i_t base = 0; base < a_rt_num_nonbasic; base += 32) {
    const i_t k = base + lane;
    i_t j       = -1;
    f_t ratio   = INF;
    if (k < a_rt_num_nonbasic) {
      j                         = a_rt_nonbasic[k];
      const variable_status_t s = a_rt_vstatus[j];
      // fmax(.., 0). The ratio is (+-dtol - z_j)/dz_j, so a reduced cost that has drifted one ULP
      // past the tolerance flips the numerator's sign and yields a NEGATIVE ratio. Selection is
      // `ratio < min_val`, so such a candidate beats every legitimate one with certainty, and the
      // resulting negative dual step is added to every nonbasic reduced cost -- which is how a node
      // with a true optimum of 49.58 got fathomed against a cutoff of 282.
      // Clamping makes it a degenerate step, which is exactly what the bound-flipping breakpoint
      // path has always produced. The cost is more degenerate pivots, not a lost candidate:
      // clamping keeps it eligible, where DROPPING it could empty the set and be misreported as
      // DUAL_UNBOUNDED.
      if (s == variable_status_t::NONBASIC_LOWER && a_rt_dz[j] < -ptol) {
        ratio = fmax((-dtol - a_rt_z[j]) / a_rt_dz[j], f_t{0.0});
      } else if (s == variable_status_t::NONBASIC_UPPER && a_rt_dz[j] > ptol) {
        ratio = fmax((dtol - a_rt_z[j]) / a_rt_dz[j], f_t{0.0});
      }
    }
    const unsigned mask = __ballot_sync(0xffffffffu, ratio < INF);
    if (mask == 0u) { continue; }
    unsigned remaining = mask;
    while (remaining != 0u) {
      const int candidate_lane = __ffs(static_cast<int>(remaining)) - 1;
      remaining &= remaining - 1u;
      const f_t candidate_ratio = __shfl_sync(0xffffffffu, ratio, candidate_lane);
      const i_t candidate_j     = __shfl_sync(0xffffffffu, j, candidate_lane);
      if (lane == 0) {
        if (candidate_ratio < min_val) {
          min_val           = candidate_ratio;
          entering          = candidate_j;
          step_length       = candidate_ratio;
          nonbasic_entering = base + candidate_lane;
        } else if (candidate_ratio < min_val + ztol &&
                   fabs(a_rt_z[candidate_j]) > fabs(a_rt_z[entering])) {
          min_val           = candidate_ratio;
          entering          = candidate_j;
          step_length       = candidate_ratio;
          nonbasic_entering = base + candidate_lane;
        }
      }
    }
  }
  if (lane == 0) {
    *a_rt_entering = entering;
    *a_rt_k        = nonbasic_entering;
    *a_rt_step     = step_length;
  }
  __syncwarp(0xffffffffu);
}

__device__ __forceinline__ void warp_seupd_body(const i_t* a_su_basic, f_t a_su_dynormsq, f_t* a_su_dyse, i_t a_su_entering, i_t a_su_leaving, i_t a_su_nz, f_t a_su_omegar, const i_t* a_su_sdxb_i, const f_t* a_su_sdxb_x, const f_t* a_su_v, f_t a_su_wr, i_t a_su_direction)
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
      const f_t vk      = a_su_direction == -1 ? -a_su_v[k] : a_su_v[k];
      const f_t new_val = a_su_dyse[j] + wk * (2.0 * vk / a_su_wr + wk * a_su_omegar);
      a_su_dyse[j]    = fmax(new_val, 1e-4);
    }
  }
  // Publish the updated steepest-edge norms. steepest_edge_pricing_with_infeasibilities runs
  // SERIALLY ON EVERY LANE, so each lane reads norms that other lanes wrote here. Without this
  // barrier a lane can price on a stale norm and pick a different leaving row -- which is exactly
  // how this showed up on B200: identical iteration 0, then a different `leaving` at iteration 1.
  __syncwarp(0xffffffffu);
}

// Only slots written below are appended to a_pi_idx, and pricing reads a_pi_sqinf only through that
// freshly rebuilt list. Values left from an earlier pivot are therefore unreachable, so the old
// n-wide zero-fill was dead work.
__device__ __noinline__ void warp_pinf_body(const i_t* a_pi_basic, i_t* a_pi_count, i_t* a_pi_idx, const f_t* a_pi_lower, i_t a_pi_m, f_t* a_pi_pinf, f_t* a_pi_pinf_sq, f_t* a_pi_sqinf, f_t a_pi_tol, const f_t* a_pi_upper, const f_t* a_pi_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t m = a_pi_m;
  i_t count = 0; f_t lpinf = 0.0, lpinf_sq = 0.0;
  // kB strides load together (the loads cannot move past the previous stride's stores otherwise).
  constexpr int kB = 4;
  for (i_t base = 0; base < m; base += 32 * kB) {
    i_t jb[kB];
    f_t lb[kB], xb[kB], ub[kB];
#pragma unroll
    for (int u = 0; u < kB; ++u) {
      const i_t k = base + 32 * u + lane;
      jb[u]       = k < m ? a_pi_basic[k] : -1;
    }
#pragma unroll
    for (int u = 0; u < kB; ++u) {
      if (jb[u] >= 0) { lb[u] = a_pi_lower[jb[u]], xb[u] = a_pi_x[jb[u]], ub[u] = a_pi_upper[jb[u]]; }
    }
#pragma unroll
    for (int u = 0; u < kB; ++u) {
      if (base + 32 * u >= m) { break; }
      const i_t j = jb[u];
      bool inf    = false;
      if (j >= 0) {
        const f_t lo     = lb[u] - xb[u];
        const f_t up     = xb[u] - ub[u];
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
  }
  const f_t pinf    = warp_reduce_add(lpinf);
  const f_t pinf_sq = warp_reduce_add(lpinf_sq);
  // Written by EVERY lane, not just lane 0. All three are warp-uniform by construction (count is
  // accumulated from the same ballot on every lane; pinf/pinf_sq come from warp_reduce_add), so
  // each lane stores its own copy and the caller needs no broadcast at all. These pointers address
  // the CALLER'S LOCALS, so this is 32 independent per-lane stores, not a race.
  *a_pi_count = count; *a_pi_pinf = pinf; *a_pi_pinf_sq = pinf_sq;
  // Publish a_pi_idx / a_pi_sqinf before the caller's next collective reads them from every lane.
  // The pricing scan that follows is SERIAL ON EVERY LANE, so every lane reads slots that other
  // lanes wrote -- exactly the cross-lane RAW this barrier closes.
  __syncwarp(0xffffffffu);
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
  // Converge before the single-lane update: the loop above wrote a_ud_z from all 32 lanes.
  __syncwarp(0xffffffffu);
  if (lane == 0) { a_ud_z[a_ud_leaving] += step * a_ud_dz[a_ud_leaving]; }
  // Publish y and z before the caller's next collective reads them from every lane.
  __syncwarp(0xffffffffu);
}

// Warp compaction of a reach-restricted solve result. Replaces solve_to_sparse_vector's serial
// O(reach) tail, which ran on lane 0 after EVERY hypersparse solve -- and hypersparse is 59.2% of
// all solve cycles, so this tail was paid on the majority of the kernel's work.
//
// BIT-IDENTICAL to the serial loop. The output order is by increasing p (the ballot prefix orders
// within a step by lane == by p, and `count` accumulates across steps), which is the order the
// serial push_back produced. No arithmetic is performed at all -- values are copied, and the
// filter `fabs(v) > zero_tol` is evaluated per element exactly as before.
//
// The in-place clears are race-free because xi[top..m) is the reach SET: its entries are distinct,
// so no two lanes ever touch the same a_x[i]. Each lane also reads a_x[i] before clearing it itself.
__device__ __forceinline__ void warp_gather_reach_body(
  i_t* a_count, i_t* a_out_i, f_t* a_out_x, i_t a_top, i_t a_m, i_t* a_xi, f_t* a_x)
{
  const int lane     = static_cast<int>(threadIdx.x);
  const f_t zero_tol = 1e-13;
  i_t count          = 0;
  for (i_t base = a_top; base < a_m; base += 32) {
    const i_t p = base + lane;
    bool active = false;
    i_t  i      = 0;
    f_t  v      = 0.0;
    if (p < a_m) {
      i      = a_xi[p];
      v      = a_x[i];
      active = (fabs(v) > zero_tol);
    }
    const unsigned ball = __ballot_sync(0xffffffffu, active);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (active) {
      a_out_i[count + prefix] = i;
      a_out_x[count + prefix] = v;
    }
    if (p < a_m) {
      a_x[i]  = 0.0;   // same index this lane just read; distinct across lanes
      a_xi[p] = 0;
    }
    count += __popc(ball);   // warp-uniform: same ballot on every lane
  }
  *a_count = count;
  // Publish the compacted arrays and the cleared workspaces before the caller's set_size and before
  // the next solve reads x_workspace_/xi_workspace_ from every lane.
  __syncwarp(0xffffffffu);
}

// ---- warp forms of the HYPERSPARSE l_solve tail --------------------------------------------------
// l_solve(gpu_sparse_vector&) ran its whole tail -- solve_to_workspace, the MPF T-loop, and
// gather_into_sparse_vector -- serially on lane 0 ("kept serial on lane 0 for now"), while the DENSE
// l_solve got the warp T-loop. That is the single largest serial site left in the kernel: the
// b_solve/l_solve/hyper_sparse row is 20.47% of all solve cycles at a 4.88M mean against the dense
// path's 1.80M, and this is why -- same algorithm, 1 lane instead of 32.
//
// The workspace representation the tail operates on is:
//   x [i]                values, i in [0,m)
//   mark[i]              0/1 membership flag, i in [0,m)
//   mark[m + k]          the index list, k in [0,nz)
// add_sparse_column appends to that list as fill-in appears, which is what made it look serial. A
// ballot prefix-sum appends in the identical order, so the list -- and hence the output sparse
// vector's index order -- is unchanged.

// Compaction of the reach into the workspace index list, then the membership marks.
// Bit-identical. The two phases MUST stay separated by a barrier: phase 1 clears positions
// [top,m) of mark[] and phase 2 sets mark[i] for the reach indices, and those two index sets
// overlap arbitrarily (a position and an index are different things over the same array).
__device__ __forceinline__ void warp_solve_to_workspace_body(i_t a_m, i_t* a_mark, i_t a_top)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t nz   = a_m - a_top;
  for (i_t k = lane; k < nz; k += 32) { a_mark[a_m + k] = a_mark[a_top + k]; }
  __syncwarp(0xffffffffu);
  for (i_t k = lane; k < nz; k += 32) { a_mark[a_top + k] = 0; }
  __syncwarp(0xffffffffu);
  for (i_t k = lane; k < nz; k += 32) { a_mark[a_mark[a_m + k]] = 1; }
  __syncwarp(0xffffffffu);
}

// The MPF T-loop over the sparse workspace form.
//
// NOT bit-identical, and deliberately so: the dot product is reassociated by the butterfly exactly
// as the already-accepted dense warp T-loop (warp_tloop_body) reassociates its own. Everything else
// IS exact -- the fill-in append order is preserved by the ballot prefix, and each x[i] takes one
// += per column (a CSC column has distinct row indices), so no accumulation is reordered.
// The <=2 dot fast path is bit-identical for the same reason as in warp_tloop_body.
template <bool Ascending, int DotOffset, int AxpyOffset>
__device__ __forceinline__ void warp_tloop_sparse_body(const i_t* a_cs,
                                                       const f_t* a_mu,
                                                       i_t a_num,
                                                       const i_t* a_ri,
                                                       f_t a_tol,
                                                       const f_t* a_val,
                                                       i_t a_m,
                                                       i_t* a_mark,
                                                       i_t* a_nz,
                                                       f_t* a_x)
{
  static_assert(DotOffset == 0 || DotOffset == 1);
  static_assert(AxpyOffset == 0 || AxpyOffset == 1);
  static_assert(DotOffset != AxpyOffset);
  const int lane = static_cast<int>(threadIdx.x);
  constexpr unsigned full = 0xffffffffu;
  i_t nz         = *a_nz;   // warp-uniform on entry, and every update below keeps it uniform
  // Each lane prefetches one step of the next 32: mu, both column ranges and the first two entries
  // of each column. None of them depends on x, so the per-step chain keeps only the x accesses.
  for (i_t base = 0; base < a_num; base += 32) {
    i_t p_d0 = 0, p_d1 = 0, p_a0 = 0, p_a1 = 0, p_dr0 = 0, p_dr1 = 0, p_ar0 = 0, p_ar1 = 0;
    f_t p_mu = 1.0, p_dv0 = 0.0, p_dv1 = 0.0, p_av0 = 0.0, p_av1 = 0.0;
    if (base + lane < a_num) {
      const i_t k    = Ascending ? base + lane : (a_num - 1 - (base + lane));
      const i_t dcol = 2 * k + DotOffset;
      p_mu           = a_mu[k];
      p_d0           = a_cs[dcol];
      p_d1           = a_cs[dcol + 1];
      p_a0           = (AxpyOffset > DotOffset) ? p_d1 : a_cs[dcol - 1];
      p_a1           = (AxpyOffset > DotOffset) ? a_cs[dcol + 2] : p_d0;
      if (p_d1 - p_d0 >= 1) { p_dr0 = a_ri[p_d0]; p_dv0 = a_val[p_d0]; }
      if (p_d1 - p_d0 >= 2) { p_dr1 = a_ri[p_d0 + 1]; p_dv1 = a_val[p_d0 + 1]; }
      if (p_a1 - p_a0 >= 1) { p_ar0 = a_ri[p_a0]; p_av0 = a_val[p_a0]; }
      if (p_a1 - p_a0 >= 2) { p_ar1 = a_ri[p_a0 + 1]; p_av1 = a_val[p_a0 + 1]; }
    }
    const int steps = static_cast<int>(a_num - base < 32 ? a_num - base : 32);
    for (int t = 0; t < steps; ++t) {
    const f_t mu   = __shfl_sync(full, p_mu, t);
    const i_t d0   = __shfl_sync(full, p_d0, t);
    const i_t d1   = __shfl_sync(full, p_d1, t);
    const i_t dlen = d1 - d0;
    f_t dot;
    if (dlen <= 2) {
      const i_t r0 = __shfl_sync(full, p_dr0, t), r1 = __shfl_sync(full, p_dr1, t);
      const f_t v0 = __shfl_sync(full, p_dv0, t), v1 = __shfl_sync(full, p_dv1, t);
      dot = 0.0;
      if (dlen >= 1) { if (a_mark[r0]) { dot = v0 * a_x[r0]; } }
      if (dlen == 2) { if (a_mark[r1]) { dot += v1 * a_x[r1]; } }
      __syncwarp(full);   // separates these x READS from the axpy WRITES below
    } else {
      f_t partial = 0.0;
      for (i_t p = d0 + lane; p < d1; p += 32) {
        const i_t i = a_ri[p];
        if (a_mark[i]) { partial += a_val[p] * a_x[i]; }
      }
      dot = warp_reduce_add(partial);
    }
    const f_t theta = dot / mu;
    const i_t a0    = __shfl_sync(full, p_a0, t);
    const i_t a1    = __shfl_sync(full, p_a1, t);
    const i_t ar0 = __shfl_sync(full, p_ar0, t), ar1 = __shfl_sync(full, p_ar1, t);
    const f_t av0 = __shfl_sync(full, p_av0, t), av1 = __shfl_sync(full, p_av1, t);
    if (fabs(theta) > a_tol) {
      for (i_t base2 = a0; base2 < a1; base2 += 32) {
        const i_t p  = base2 + lane;
        const bool have = (p < a1);
        i_t i        = 0;
        f_t av       = 0.0;
        bool isnew   = false;
        if (have) {
          // The first two entries of the column come from the prefetch.
          i     = p == a0 ? ar0 : p == a0 + 1 ? ar1 : a_ri[p];
          av    = p == a0 ? av0 : p == a0 + 1 ? av1 : a_val[p];
          isnew = (a_mark[i] == 0);   // distinct i across lanes: a CSC column has distinct rows
        }
        const unsigned ball = __ballot_sync(full, isnew);
        const int prefix    = __popc(ball & ((1u << lane) - 1u));
        if (isnew) {
          a_mark[i]                 = 1;
          a_mark[a_m + nz + prefix] = i;   // append, in increasing p == the serial append order
        }
        nz += __popc(ball);                // warp-uniform: identical ballot on every lane
        if (have) { a_x[i] += (-theta) * av; }
        __syncwarp(full);                  // publish marks/x before the next chunk reads them
      }
    }
    }
  }
  *a_nz = nz;
}

// Warp form of gather_into_sparse_vector: compact the workspace index list into a sparse vector and
// clear all three workspaces. Bit-identical -- output order is increasing k, as the serial
// push_back produced, and the list entries are distinct so the clears never collide.
__device__ __forceinline__ void warp_gather_workspace_body(
  i_t* a_count, i_t* a_out_i, f_t* a_out_x, i_t a_m, i_t* a_mark, i_t a_nz, f_t* a_x)
{
  const int lane     = static_cast<int>(threadIdx.x);
  const f_t zero_tol = 1e-13;
  i_t count          = 0;
  for (i_t base = 0; base < a_nz; base += 32) {
    const i_t k = base + lane;
    bool active = false;
    i_t  i      = 0;
    f_t  v      = 0.0;
    if (k < a_nz) {
      i      = a_mark[a_m + k];
      v      = a_x[i];
      active = (fabs(v) > zero_tol);
    }
    const unsigned ball = __ballot_sync(0xffffffffu, active);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (active) {
      a_out_i[count + prefix] = i;
      a_out_x[count + prefix] = v;
    }
    if (k < a_nz) {
      a_mark[a_m + k] = 0;
      a_mark[i]       = 0;
      a_x[i]          = 0.0;
    }
    count += __popc(ball);
  }
  *a_count = count;
  __syncwarp(0xffffffffu);
}

// ---- warp forms of the per-pivot MPF update (ft.update) ------------------------------------------
// ft.update ran entirely inside one lane0_serial and is 8.54% of all block-resident cycles. Its two
// sorts and its fill-tracking append genuinely need one lane, but the scatter, the two column
// appends and the workspace clear are data-parallel. All three below are bit-identical: the appends
// keep source order via the ballot prefix, and the scatter/clear touch distinct indices.
__device__ __forceinline__ void warp_scatter_ws_body(const i_t* a_in_i, const f_t* a_in_x, i_t a_m,
                                                     i_t a_nz, i_t* a_mark, f_t* a_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < a_nz; k += 32) {
    const i_t i     = a_in_i[k];
    a_mark[i]       = 1;
    a_mark[a_m + k] = i;
    a_x[i]          = a_in_x[k];
  }
  __syncwarp(0xffffffffu);
}

// csc append of a column held as (index list, scatter values); drops exact zeros. Returns the new
// end cursor, warp-uniform.
__device__ __forceinline__ i_t warp_append_scatter_body(i_t a_base, i_t* a_dst_i, f_t* a_dst_x,
                                                        const i_t* a_idx, i_t a_n, const f_t* a_xs)
{
  const int lane = static_cast<int>(threadIdx.x);
  i_t pos        = a_base;
  for (i_t b = 0; b < a_n; b += 32) {
    const i_t k = b + lane;
    bool active = false;
    i_t  iv     = 0;
    f_t  xv     = 0.0;
    if (k < a_n) {
      iv     = a_idx[k];
      xv     = a_xs[iv];
      active = (xv != 0.0);
    }
    const unsigned ball = __ballot_sync(0xffffffffu, active);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (active) {
      a_dst_i[pos + prefix] = iv;
      a_dst_x[pos + prefix] = xv;
    }
    pos += __popc(ball);
  }
  __syncwarp(0xffffffffu);
  return pos;
}

// Same, for a column held as an explicit (index, value) sparse vector.
__device__ __forceinline__ i_t warp_append_sv_body(i_t a_base, i_t* a_dst_i, f_t* a_dst_x,
                                                   const i_t* a_src_i, const f_t* a_src_x, i_t a_n)
{
  const int lane = static_cast<int>(threadIdx.x);
  i_t pos        = a_base;
  for (i_t b = 0; b < a_n; b += 32) {
    const i_t k = b + lane;
    bool active = false;
    i_t  iv     = 0;
    f_t  xv     = 0.0;
    if (k < a_n) {
      iv     = a_src_i[k];
      xv     = a_src_x[k];
      active = (xv != 0.0);
    }
    const unsigned ball = __ballot_sync(0xffffffffu, active);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (active) {
      a_dst_i[pos + prefix] = iv;
      a_dst_x[pos + prefix] = xv;
    }
    pos += __popc(ball);
  }
  __syncwarp(0xffffffffu);
  return pos;
}

__device__ __forceinline__ void warp_clear_ws_body(i_t a_m, i_t* a_mark, i_t a_nz, f_t* a_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < a_nz; k += 32) {
    const i_t i     = a_mark[a_m + k];
    a_mark[i]       = 0;
    a_x[i]          = 0.0;
    a_mark[a_m + k] = 0;
  }
  __syncwarp(0xffffffffu);
}

// ---- warp forms of the per-solve data movement ---------------------------------------------------
// b_solve / b_transpose_solve wrap several O(nnz) and O(m) copies in lane0_serial: the sparse-vector
// copy-assigns (solution = rhs, Lsol = solution, UTsol = solution), the in-place index permutes, and
// the dense row permute. They are pure data movement -- no arithmetic, distinct destinations -- so
// they are bit-identical when spread across the warp, and they run on every pivot.
__device__ __forceinline__ void warp_copy_i_body(i_t* a_dst, const i_t* a_src, i_t a_n)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < a_n; k += 32) { a_dst[k] = a_src[k]; }
  __syncwarp(0xffffffffu);
}

__device__ __forceinline__ void warp_copy_f_body(f_t* a_dst, const f_t* a_src, i_t a_n)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < a_n; k += 32) { a_dst[k] = a_src[k]; }
  __syncwarp(0xffffffffu);
}

// In-place i[k] <- p[i[k]]. Each entry is remapped independently, so order is irrelevant.
__device__ __forceinline__ void warp_permute_idx_body(i_t* a_idx, i_t a_n, const i_t* a_p)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < a_n; k += 32) { a_idx[k] = a_p[a_idx[k]]; }
  __syncwarp(0xffffffffu);
}

// x[k] = b[p[k]] over a dense length-n vector (permute_vector).
__device__ __forceinline__ void warp_permute_dense_body(const f_t* a_b, i_t a_n, const i_t* a_p,
                                                        f_t* a_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < a_n; k += 32) { a_x[k] = a_b[a_p[k]]; }
  __syncwarp(0xffffffffu);
}

// x[p[k]] = b[k] over a dense length-n vector (inverse_permute_vector).
__device__ __forceinline__ void warp_inverse_permute_dense_body(const f_t* a_b,
                                                                i_t a_n,
                                                                const i_t* a_p,
                                                                f_t* a_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < a_n; k += 32) { a_x[a_p[k]] = a_b[k]; }
  __syncwarp(0xffffffffu);
}

// Warp forms of the sptrsv prologue's data-parallel passes. The reach DFS itself is a pointer-chase
// and stays on lane 0, but the O(reach) sweeps around it do not have to.
//
// All bit-identical: stores of the same values to the same distinct addresses, and an integer max.
//
// The barrier between the two phases is REQUIRED: b.i is a subset of the reach, so zeroing x over
// the reach after scattering the rhs would silently erase the rhs.
__device__ __forceinline__ void warp_sptrsv_preload_body(const i_t* a_bi, i_t a_bnz, const f_t* a_bx,
                                                         i_t a_m, i_t a_top, const i_t* a_xi,
                                                         f_t* a_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t p = a_top + lane; p < a_m; p += 32) { a_x[a_xi[p]] = 0.0; }
  __syncwarp(0xffffffffu);
  for (i_t p = lane; p < a_bnz; p += 32) { a_x[a_bi[p]] = a_bx[p]; }
  __syncwarp(0xffffffffu);
}

// Largest elimination level over the reach, on every lane. Integer max => exact and order-free.
// No-op columns are excluded: they are also filtered out of the sp_co work list below, so counting
// them here would leave trailing levels that cost a __syncwarp each and process nothing. `best`
// starting at 0 already handles "every reachable column is a no-op" -> zero wavefront levels.
__device__ __forceinline__ i_t warp_sptrsv_maxlevel_body(const i_t* a_level, i_t a_m, i_t a_top,
                                                         const i_t* a_xi)
{
  const int lane = static_cast<int>(threadIdx.x);
  i_t best       = 0;
  for (i_t p = a_top + lane; p < a_m; p += 32) {
    const i_t w = a_level[a_xi[p]];
    if (level_is_noop(w)) { continue; }
    const i_t lv = factor_level(w);
    if (lv > best) { best = lv; }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const i_t o = __shfl_xor_sync(0xffffffffu, best, off);
    if (o > best) { best = o; }
  }
  return best;
}

// COMPILE-TIME operator variant of the reach-restricted gather (`Direct` path).
template <int Op>
__device__ __forceinline__ void warp_sptrsv_body_op(const i_t* a_cs,
                                                    const i_t* a_ridx,
                                                    const f_t* a_val, i_t* a_sp_co, i_t* a_sp_ls,
                                                    int a_sp_nlev, f_t* a_x)
{
  dts::wavefront_solve_op<Op>(a_cs, a_ridx, a_val, a_x, a_sp_ls, a_sp_co, a_sp_nlev);
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
// rev_level != nullptr: the operator's DAG is the reverse of the one rev_level (tagged, rev_nlev
// levels) was built for, so nlev + 1 - level is a schedule with the same level count.
__device__ int warp_build_levels(const gpu_csc_view& mat, bool diag_first, bool asc_process,
                                  gpu_span<i_t>& level, gpu_span<i_t>& level_start,
                                  gpu_span<i_t>& col_order, gpu_span<i_t>& pos,
                                  const i_t* rev_level = nullptr, i_t rev_nlev = 0)
{
  const i_t m = mat.n;
  level.assign(m, 0);
  if (rev_level != nullptr) {
    for (i_t j = 0; j < m; ++j) { level[j] = rev_nlev + 1 - (rev_level[j] & 0x7fffffff); }
  }
  for (i_t s = 0; rev_level == nullptr && s < m; ++s) {  // pull: level[j] = 1 + max level[dep]
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
  auto is_noop = [&](i_t j) {
    return mat.col_start[j + 1] - mat.col_start[j] == 1 && mat.x[mat.col_start[j]] == 1.0;
  };
  col_order.resize(m);                                 // sized m; the tail is unused when eliding
  // One counting sort by level, ascending j within a level.
  level_start.assign(nlev + 1, 0);
  for (i_t j = 0; j < m; ++j) { if (!is_noop(j)) { level_start[level[j]]++; } }
  for (i_t k = 1; k <= nlev; ++k) { level_start[k] += level_start[k - 1]; }
  // `pos` IS now the 16th (m+1) arena array the note below warned about. History, so the reversal is
  // legible: hoisting it out of a per-call device malloc was measured bit-exact and worth ~0.7% of
  // block time, and was rejected ONLY because the extra static VRAM broke the lazily reserved malloc
  // heap at 1,200 problems (cudaErrorLaunchOutOfResources). Two things changed. It is now the LAST
  // device malloc in the kernel -- measured at exactly 4 per problem, one per solve op, ~105 B each
  // -- so removing it retires the whole malloc path. And the VRAM
  // objection is gone: one (m+1) array shared by all four ops costs 4(m+1) B/slot against the
  // iteration-driven S_ heap term it replaces, which is ~36 MB/slot at 200 iterations.
  // Deliberately NOT aliased onto sp_pos_v, which is live for the whole solve as cfg.sp_pos.
  pos.assign(nlev + 1, 0);
  for (i_t lv = 1; lv <= nlev; ++lv) { pos[lv] = level_start[lv - 1]; }
  for (i_t j = 0; j < m; ++j) {
    if (is_noop(j)) { continue; }
    col_order[pos[level[j]]++] = j;
  }
  // Tag AFTER the level pull and the counting sort, both of which read PLAIN level values. From
  // here on level[] is a tagged word: read it through factor_level().
  // Unconditional, and it must stay in step with the counting sort above: that sort already drops
  // no-op columns from col_order, so leaving them untagged here would let the wavefront walk columns
  // the schedule never emitted.
  for (i_t j = 0; j < m; ++j) { if (is_noop(j)) { level[j] |= kColNoop; } }
  return static_cast<int>(nlev);
}

// All 32 lanes; the same schedule as warp_build_levels (ascending j within a level). Only the level
// pull runs on lane 0. Spans are sized without their serial zero-fill.
__device__ int warp_build_levels_all(const gpu_csc_view& mat, bool diag_first, bool asc_process,
                                     gpu_span<i_t>& level, gpu_span<i_t>& level_start,
                                     gpu_span<i_t>& col_order, gpu_span<i_t>& pos,
                                     const i_t* rev_level, i_t rev_nlev, i_t long_len)
{
  constexpr unsigned full = 0xffffffffu;
  const int lane = static_cast<int>(threadIdx.x);
  const i_t m    = mat.n;
  __syncwarp(full);
  if (lane == 0) {
    level.reserve(m), level.set_size(m);
    col_order.reserve(m), col_order.set_size(m);
  }
  __syncwarp(full);
  i_t* lv        = level.data();
  const i_t* cs  = mat.col_start.data();
  const i_t* ri  = mat.i.data();
  const f_t* vx  = mat.x.data();
  if (rev_level != nullptr) {
    for (i_t j = lane; j < m; j += 32) { lv[j] = rev_nlev + 1 - (rev_level[j] & 0x7fffffff); }
  } else {
    for (i_t j = lane; j < m; j += 32) { lv[j] = 0; }
    __syncwarp(full);
    if (lane == 0) {
      for (i_t s = 0; s < m; ++s) {
        const i_t j  = asc_process ? s : (m - 1 - s);
        const i_t c0 = cs[j], c1 = cs[j + 1];
        const i_t lo = diag_first ? c0 + 1 : c0;
        const i_t hi = diag_first ? c1 : (c1 - 1);
        i_t l        = 1;
        for (i_t p = lo; p < hi; ++p) {
          const i_t d = lv[ri[p]] + 1;
          if (d > l) { l = d; }
        }
        lv[j] = l;
      }
    }
  }
  __syncwarp(full);
  i_t nlev = 0;
  for (i_t j = lane; j < m; j += 32) { nlev = lv[j] > nlev ? lv[j] : nlev; }
  for (int off = 16; off > 0; off >>= 1) {
    const i_t o = __shfl_xor_sync(full, nlev, off);
    nlev        = o > nlev ? o : nlev;
  }
  if (lane == 0) { level_start.reserve(nlev + 1), level_start.set_size(nlev + 1); }
  __syncwarp(full);
  i_t* ls = level_start.data();
  i_t* ps = pos.data();
  i_t* co = col_order.data();
  // Long columns (gather length > long_len) go last in their level, tagged: cursors ps[l] (short)
  // and ps[nlev + 1 + l] (long).
  if (2 * (nlev + 1) > m + 1) { long_len = 0; }
  const i_t lb = nlev + 1;
  for (i_t k = lane; k <= nlev; k += 32) {
    ls[k] = 0;
    if (long_len > 0) { ps[lb + k] = 0; }
  }
  __syncwarp(full);
  auto is_noop = [&](i_t j) {
    const i_t c0 = cs[j];
    return cs[j + 1] - c0 == 1 && vx[c0] == 1.0;
  };
  auto is_long = [&](i_t j) { return long_len > 0 && cs[j + 1] - cs[j] - 1 > long_len; };
  for (i_t base = 0; base < m; base += 32) {
    const i_t j        = base + lane;
    const bool on      = j < m && !is_noop(j);
    const i_t key      = on ? (is_long(j) ? lb + lv[j] : lv[j]) : -1 - lane;
    const unsigned grp = __match_any_sync(full, key);
    if (on && lane == __ffs(grp) - 1) {
      atomicAdd(&ls[key < lb ? key : key - lb], __popc(grp));
      if (key >= lb) { atomicAdd(&ps[key], __popc(grp)); }
    }
  }
  __syncwarp(full);
  if (lane == 0) {
    for (i_t k = 1; k <= nlev; ++k) { ls[k] += ls[k - 1]; }
    ps[0] = 0;
    for (i_t l = 1; l <= nlev; ++l) {
      ps[l] = ls[l - 1];
      if (long_len > 0) { ps[lb + l] = ls[l] - ps[lb + l]; }
    }
  }
  __syncwarp(full);
  const unsigned below = (1u << lane) - 1u;
  for (i_t base = 0; base < m; base += 32) {
    const i_t j        = base + lane;
    const bool on      = j < m && !is_noop(j);
    const bool lng     = on && is_long(j);
    const i_t key      = on ? (lng ? lb + lv[j] : lv[j]) : -1 - lane;
    const unsigned grp = __match_any_sync(full, key);
    const int leader   = __ffs(grp) - 1;
    i_t at             = 0;
    if (on && lane == leader) { at = atomicAdd(&ps[key], static_cast<i_t>(__popc(grp))); }
    at = __shfl_sync(full, at, leader);
    if (on) { co[at + __popc(grp & below)] = lng ? (j | kColNoop) : j; }
  }
  __syncwarp(full);
  for (i_t j = lane; j < m; j += 32) {
    if (is_noop(j)) { lv[j] |= kColNoop; }
  }
  __syncwarp(full);
  return static_cast<int>(nlev);
}


// ---- sparse triangular solve (reach + iterative DFS) ----------------------
// Visited state lives in a private byte buffer so the factor CSC remains read-only. The warp
// reach path views this buffer as 32-bit words; small matrices use one byte per row.
using reach_mark_word = std::uint32_t;
static_assert(sizeof(reach_mark_word) == 4);

// Keep m addressable bytes per seat while rounding only this buffer up to word alignment.
__host__ __device__ constexpr i_t reach_mark_storage_words(i_t rows)
{
  const i_t words = rows / 4 + (rows % 4 != 0);
  return words == 0 ? 1 : words;
}

// pinv == nullptr means no permutation.
__device__ i_t depth_first_search(
  i_t j, const i_t* pinv, const gpu_csc_view& G, unsigned char* marked, i_t top, i_t* xi,
  i_t* pstack)
{
  i_t head = 0;
  xi[0]    = j;
  i_t done = 0;
  while (head >= 0) {
    j        = xi[head];
    i_t jnew = pinv ? pinv[j] : j;
    if (marked[j] == 0) {
      marked[j]    = 1;
      pstack[head] = (jnew < 0) ? 0 : G.col_start[jnew];
    }
    done           = 1;
    i_t p2         = (jnew < 0) ? 0 : G.col_start[jnew + 1];
    const i_t psav = pstack[head];
    i_t p;
    for (p = psav; p < p2; ++p) {
      i_t i = G.i[p];
      if (marked[i] != 0) { continue; }
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

__device__ __forceinline__ bool reach_bit_marked(const reach_mark_word* marked, i_t row)
{
  return (marked[row >> 5] & (1u << (row & 31))) != 0;
}

__device__ __forceinline__ void reach_bit_set(reach_mark_word* marked, i_t row)
{
  marked[row >> 5] |= 1u << (row & 31);
}

__device__ __forceinline__ void reach_bit_clear(reach_mark_word* marked, i_t row)
{
  marked[row >> 5] &= ~(1u << (row & 31));
}

// General global-stack fallback for a bit-packed reach whose active depth exceeds the shared hot
// stack. Lane 0 is the only accessor, so ordinary word read-modify-write operations are sufficient.
__device__ i_t depth_first_search_bits(
  i_t root, const gpu_csc_view& G, reach_mark_word* marked, i_t top, i_t* xi, i_t* pstack)
{
  i_t head = 0;
  xi[0]    = root;
  while (head >= 0) {
    const i_t row = xi[head];
    if (!reach_bit_marked(marked, row)) {
      reach_bit_set(marked, row);
      pstack[head] = G.col_start[row];
    }
    const i_t end = G.col_start[row + 1];
    i_t p         = pstack[head];
    while (p < end && reach_bit_marked(marked, G.i[p])) { ++p; }
    if (p < end) {
      pstack[head] = p;
      xi[++head]   = G.i[p];
    } else {
      pstack[head] = 0;
      xi[head--]   = 0;
      xi[--top]    = row;
    }
  }
  return top;
}

// pinv-free DFS specialized for the warp hypersparse path. Keep the live frames in the config's
// 64-entry hot stack (shared memory in the monolith), cache both CSC bounds once per discovered
// vertex, and resume at p+1 instead of reloading the edge that led to the child. If an unusually
// deep factor exceeds the hot stack, the general bitset DFS handles that subtree in xi/xi+m.
__device__ i_t depth_first_search_hot(
  i_t root, const gpu_csc_view& G, reach_mark_word* marked, i_t top, i_t* xi,
  i_t* dfs_node, i_t* dfs_next, i_t* dfs_end)
{
  constexpr i_t hot_depth = 64;
  i_t head                = 0;
  reach_bit_set(marked, root);
  dfs_node[0]              = root;
  dfs_next[0]              = G.col_start[root];
  dfs_end[0]               = G.col_start[root + 1];

  while (head >= 0) {
    i_t p        = dfs_next[head];
    const i_t p2 = dfs_end[head];
    while (p < p2 && reach_bit_marked(marked, G.i[p])) { ++p; }

    if (p < p2) {
      const i_t child = G.i[p];
      dfs_next[head]   = p + 1;
      if (head + 1 < hot_depth) {
        reach_bit_set(marked, child);
        ++head;
        dfs_node[head] = child;
        dfs_next[head] = G.col_start[child];
        dfs_end[head]  = G.col_start[child + 1];
      } else {
        top = depth_first_search_bits(child, G, marked, top, xi, xi + G.m);
      }
    } else {
      xi[--top] = dfs_node[head--];
    }
  }
  return top;
}

__device__ i_t reach_hot(
  const gpu_sparse_vector& b, const gpu_csc_view& G, reach_mark_word* marked, i_t* xi,
  i_t* dfs_node, i_t* dfs_next, i_t* dfs_end)
{
  const i_t m   = G.m;
  i_t top       = m;
  const i_t bnz = b.i.size();
  for (i_t p = 0; p < bnz; ++p) {
    const i_t root = b.i[p];
    if (!reach_bit_marked(marked, root)) {
      top = depth_first_search_hot(root, G, marked, top, xi, dfs_node, dfs_next, dfs_end);
    }
  }
  for (i_t p = top; p < m; ++p) { reach_bit_clear(marked, xi[p]); }
  return top;
}

__device__ i_t reach(
  const gpu_sparse_vector& b, const i_t* pinv, const gpu_csc_view& G, unsigned char* marked,
  i_t* xi)
{
  const i_t m   = G.m;
  i_t top       = m;
  const i_t bnz = b.i.size();
  for (i_t p = 0; p < bnz; ++p) {
    if (marked[b.i[p]] == 0) {
      top = depth_first_search(b.i[p], pinv, G, marked, top, xi, xi + m);
    }
  }
  for (i_t p = top; p < m; ++p) {
    marked[xi[p]] = 0;
  }
  return top;
}

// ===========================================================================================

// Run serial glue (allocation / read-modify-write) on lane 0, then re-converge. Under !Direct this
// is a plain call — see the deadlock note above.
template <class Fn>
__device__ __forceinline__ void lane0_serial(Fn&& f)
{
  // Converge BEFORE the lane-0 region. Under independent thread scheduling (sm_70+) the lanes of
  // a warp are NOT implicitly in lockstep, so without this lane 0 can run ahead into `f()` while
  // the other lanes are still in the PRECEDING collective -- and `f()` is exactly the code that
  // reallocates and rewrites the buffers those lanes are still reading. Omitting it produced
  // wild reads on B200 (garbage vector sizes, a column index of -533,214,976) that moved with
  // every codegen change, and racecheck reported the hazard directly. sm_89 tolerated it; sm_100
  // does not.
  __syncwarp(0xffffffffu);
  if (threadIdx.x == 0) { f(); }
  // Re-converge AND publish lane 0's global writes to the rest of the warp (__syncwarp orders
  // memory among the participating lanes), so the following collective reads what lane 0 wrote.
  __syncwarp(0xffffffffu);
}

// Publish a value computed inside a lane0_serial region (only lane 0 holds it) to every lane.
// Must be called by all 32 lanes, i.e. AFTER the lane0_serial that produced it.
//
// This publishes through SHARED memory rather than __shfl_sync. Both are correct now that
// lane0_serial converges the warp on entry; shared memory is used because it does not depend on
// lane participation at all, and this is the site where the convergence bug first showed itself --
// on B200, __shfl_sync(mask, top, 0) returned 8191 while lane 0 held top=760, ON LANE 0 ITSELF.
// That looks impossible, and it is, for a converged warp: a shuffle whose named lanes have not all
// arrived is undefined, and the lanes had not arrived because lane 0 had already run ahead into the
// preceding lane-0 region. The garbage then indexed sp_co / delta_z_indices far out of bounds and
// the kernel died with an illegal address (csched008/node_d24_id239).
// Related cleanup, not a fix: warp_dz_body / warp_fromdense_body / warp_pinf_body produce
// warp-uniform results, so they now write them on every lane and need no broadcast at all.
template <class T>
__device__ __forceinline__ T lane0_bcast(T v)
{
  __shared__ T s_bcast;
  if (threadIdx.x == 0) { s_bcast = v; }
  __syncwarp(0xffffffffu);
  const T r = s_bcast;
  __syncwarp(0xffffffffu);  // no lane may overwrite s_bcast before every lane has read it
  return r;
}

// ---------------------------------------------------------------------------
// gpu_basis_update_mpf : the MPF LU factorization object. The initial factors
// L0_/U0_ (+ transposes) and permutations are fixed views uploaded from the
// host; S_/mu_values_ grow in-kernel as rank-1 updates are stacked.
// ---------------------------------------------------------------------------
// Kernel path counters (CUOPT_BNB_GPU_SEAT_CLOCK report [GPUBATCH-PATHS]).
enum path_t : int {
  P_SE_RESET, P_DZ_ROWWISE, P_DZ_COLDOT, P_UT_SPARSE, P_UT_DENSE, P_LT_SPARSE, P_LT_DENSE,
  P_L_PAIRED, P_L_SPLIT, P_U_PAIRED, P_U_SPLIT, P_FTRAN_SPARSE_RHS, P_MPF_LEN, P_UPD_OK,
  P_UPD_FULL, P_UPD_MU, P_ETILDE_BUCKET, P_ETILDE_PERM, P_ETILDE_NZ, P_UTILDE_NZ, P_NAN_RECOVERY,
  P_CUTOFF_SCAN, P_V_DENSE, P_V_SCATTER, P_REFACTOR, P_LEVEL_REUSE, P_LEVEL_BUILD, kPaths
};

struct gpu_basis_update_t {
  // Compile-time tag for the BU switch in the templated driver/kernel (mirrors the CPU
  // `is_mpf = std::is_same<BU, basis_update_mpf_t>` dispatch in simplified_dual_phase2).
  static constexpr bool is_mpf_bu = true;
  i_t num_updates_{0};
  // TODO: should that value really be 100 and not 200/300?
  i_t refactor_frequency_{100};
  // Caps on the rank-1 store, so it can be PREALLOCATED instead of grown in-kernel (which is what
  // made a slot un-reusable across launches, and what kept device malloc() alive). Both are set by
  // the host.
  //   update_cap_  : how many rank-1 updates the store holds.
  //   eta_nnz_cap_ : capacity of S_.i / S_.x in entries. A budget rather than a worst case: sizing
  //                  every column densely at m costs 58 MB/seat on the v6 set against a measured
  //                  median of 0.27% of m. Defaults clear every measured node. See PHASE0_SIZING.md.
  i_t update_cap_{0};
  i_t eta_nnz_cap_{0};
  // Set by grow_storage when the store cannot take another update. Sticky until the next reset.
  bool store_exhausted_{false};
  using factor_view_type  = gpu_csc_view;
  using warp_config_type  = warp_solve_config_t;

  factor_view_type L0_, U0_, L0_transpose_, U0_transpose_;
  gpu_span<i_t> row_permutation_, inverse_row_permutation_;
  gpu_csc_matrix S_;
  gpu_span<f_t> mu_values_;
  gpu_span<i_t> xi_workspace_;  // size 2*m (scratch)
  gpu_span<f_t> x_workspace_;   // size m   (scratch)
  gpu_span<unsigned char> reach_mark_;  // m bytes; the warp reach path uses aligned 32-bit words
  // Reused solve scratch (owning; grows to m once, then reused every call) so the b_solve /
  // b_transpose_solve dense↔sparse conversions and the sparse-Lᵀ gather no longer device-malloc a
  // fresh temporary per solve. Bit-identical to the per-call locals (same values, same order).
  mutable gpu_span<f_t> dense_scratch_;     // first dense solve RHS
  mutable gpu_span<f_t> dense_scratch2_;    // second RHS for fused two-RHS FTran
  mutable gpu_sparse_vector sparse_scratch_; // gathered `b` (sparse Lᵀ) + throwaway Lsol
  mutable gpu_sparse_vector ei_scratch_;     // unit RHS for compute_delta_y (separate — b_transpose_solve
                                             // uses sparse_scratch_ internally, so ei can't share it)
  mutable gpu_span<f_t> sort_bucket_;      // reused scratch for etilde.sort() (per-pivot sort
  mutable gpu_span<i_t> sort_i_;           // in update(): the O(n) bucket + the n·log·n index/value/perm
  mutable gpu_span<f_t> sort_x_;           // buffers, all reused instead of malloc'd each pivot.
  mutable gpu_span<i_t> sort_perm_;
  // Dense solve scratch used by NaN recovery in compute_delta_x and by fresh terminal pricing.
  // These are slices of the preallocated slot buffer: the operations are warp collectives, so
  // stack-local gpu_spans would give each lane a different buffer.
  mutable gpu_span<f_t> recover_rhs_;
  mutable gpu_span<f_t> recover_sol_;
  // Statistics that drive the hypersparse-vs-dense path choice.
  mutable i_t num_calls_L_{0}, num_calls_U_{0}, num_calls_L_transpose_{0}, num_calls_U_transpose_{0};
  mutable f_t sum_L_{0.0}, sum_U_{0.0}, sum_L_transpose_{0.0}, sum_U_transpose_{0.0};
  // Blend each average with its observed tail so routing misses favor the bounded dense path.
  mutable f_t max_L_{1.0}, max_U_{1.0}, max_L_transpose_{1.0}, max_U_transpose_{1.0};
  f_t hypersparse_threshold_{0.05};
  // When non-null (warp-optimized kernel), the 4 dense triangular solves are
  // dispatched to the whole warp via this shared config instead of running on
  // one lane. Set by lane 0 before the driver runs.
  warp_config_type* warp_cfg_ = nullptr;
  // Solve routing counts (lane 0): dense wavefront, reach-restricted sparse, sparse -> dense fallback.
  mutable unsigned route_[3] = {0u, 0u, 0u};
  // Solve sub-step cycles (lane 0): dense wavefront, reach DFS, bucketing, sparse wavefront, MPF
  // replay, reach preload.
  mutable long long sub_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  mutable long long wop_[4] = {0, 0, 0, 0};   // dense wavefront cycles per operator (L, U, Lt, Ut)
  mutable unsigned wn_[4]   = {0u, 0u, 0u, 0u};
  // Path counts (lane 0), see path_t; reported by the seat clock.
  mutable unsigned cnt_[kPaths] = {};
  __device__ __forceinline__ void path(int k, unsigned v = 1u) const
  {
    if (threadIdx.x == 0) { cnt_[k] += v; }
  }

  // All of these are slices of the slot memory, wired once by attach_all_buffers.
  //
  // S_ and mu_values_ are preallocated too. They used to be the one genuinely dynamic allocation --
  // one column per pivot, unbounded within a refactor window -- which is what forced device
  // malloc() to stay alive and what made a slot impossible to reuse across launches. With
  // update_cap_/eta_nnz_cap_ set they are preallocated once by reserve_store() and never grow, so
  // they are now fixed-size preallocated buffers; overflow is reported, never allocated.

  __device__ gpu_basis_update_t() {}
  gpu_basis_update_t(const gpu_basis_update_t&)            = delete;
  gpu_basis_update_t& operator=(const gpu_basis_update_t&) = delete;
  __device__ ~gpu_basis_update_t()                             = default;

  __device__ i_t num_updates() const { return num_updates_; }

  // Lane 0 posts a dense-solve request to the warp and joins the collective
  // solve. (Historical: this used to publish (x, op) to lanes parked in warp_helper_loop via
  // (x, op); warp_collective_solve runs nlev internal barriers; __syncwarp B
  // makes the result visible to lane 0 before it reads x.
  // Direct: all 32 lanes are already here, so call the one known body instead of posting
  // an op code. The level SCHEDULE still comes from warp_cfg_.
  // `Op` is a template parameter: every caller names its operator as a literal, so on the Direct
  // path there is no op code, no flag load and no branch left anywhere below this point.
  template <int Op>
  __device__ void warp_dispatch(f_t* rhs) const
  {
    const long long t_ = clock64();
    if (threadIdx.x == 0) { ++route_[0]; }
    const warp_config_type* c = warp_cfg_;
    warp_solve_body_op<Op>(c->corder[Op], c->cs[Op], c->lstart[Op], c->nlev[Op], c->ridx[Op],
                           c->val[Op], rhs);
    if (threadIdx.x == 0) {
      const long long d = clock64() - t_;
      sub_[0] += d, wop_[Op] += d, ++wn_[Op];
    }
  }

  template <int Op>
  __device__ void warp_dispatch2(f_t* rhs0, f_t* rhs1) const
  {
    const long long t_ = clock64();
    if (threadIdx.x == 0) { route_[0] += 2; }
    const warp_config_type* c = warp_cfg_;
    warp_solve2_body_op<Op>(c->corder[Op], c->cs[Op], c->lstart[Op], c->nlev[Op], c->ridx[Op],
                            c->val[Op], rhs0, rhs1);
    if (threadIdx.x == 0) {
      const long long d = clock64() - t_;
      sub_[0] += d, wop_[Op] += d, wn_[Op] += 2;
    }
  }

  // Post the MPF rank-1 T-loop to the whole warp (op == WARP_TLOOP). One dispatch runs the
  // entire replay collectively (tl_num internal barriers). S_ pointers are re-read each call
  // (they can move on resize). dot col = 2k+dotoff, axpy col = 2k+axpyoff; asc selects
  // l_solve (k up) vs l_transpose (k down).
  __device__ void warp_dispatch_tloop(f_t* rhs, bool asc, i_t dotoff, i_t axpyoff, f_t tol) const
  {
    const long long t_ = clock64();
    // S_ pointers are re-read here (they move on resize) — every lane reads the same globals.
    warp_tloop_body(asc, axpyoff, S_.col_start.data(), dotoff, mu_values_.data(), num_updates_,
                    S_.i.data(), tol, S_.x.data(), rhs);
    if (threadIdx.x == 0) { sub_[4] += clock64() - t_; }
  }



  // Reach-restricted sparse gather solve (op == WARP_SPTRSV). `G` is the SCATTER factor the serial
  // sparse_triangle_solve walks (L0_ / U0_ / their transposes); `op` is the matching dense gather
  // operator (WARP_L/U/LT/UT_SOLVE) whose transpose CSC + flags + precomputed level[] we reuse.
  // Lane 0 runs the serial reach and the order-dependent counting sort; the O(reach) sweeps between
  // them are collective. Then the whole warp runs the wavefront gather over that sparse schedule.
  // Returns `top` (reachable pattern is xi_workspace_[top..m), values in x). Mirrors
  // sparse_triangle_solve's serial pre/post exactly so solve_to_workspace / solve_to_sparse_vector /
  // the T-loop that follow read the same xi_workspace_/x_workspace_ state.

  template <int Op>
  __device__ i_t warp_dispatch_sptrsv(const gpu_sparse_vector& b, const factor_view_type& G,
                                      f_t* x) const
  {
    constexpr int op = Op;
    warp_config_type* c = warp_cfg_;
    const i_t  m  = G.m;
    i_t* const xi = xi_workspace_.data();

    // The prologue used to run end-to-end inside one lane0_serial. Only the DFS and the counting
    // sort actually need a single lane; the O(reach) sweeps around them are data-parallel, and with
    // the T-loop now collective they were a visible share of the hypersparse solve.

    // (1) reach: serial DFS (pointer-chase), using bit-packed marks and the shared hot stack.
    long long t_ = clock64();
    i_t top = 0;
    lane0_serial([&] {
      if (m >= static_cast<i_t>(sizeof(reach_mark_word))) {
        auto* marked = reinterpret_cast<reach_mark_word*>(reach_mark_.data());
        top = reach_hot(b, G, marked, xi, c->dfs_node, c->dfs_next, c->dfs_end);
      } else {
        top = reach(b, nullptr, G, reach_mark_.data(), xi);
      }
    });
    top = lane0_bcast(top);
    if (threadIdx.x == 0) { const long long t1 = clock64(); sub_[1] += t1 - t_; t_ = t1; }

    // (2) zero x over the reach, then scatter the rhs — collective, barrier-separated.
    warp_sptrsv_preload_body(b.i.data(), static_cast<i_t>(b.i.size()), b.x.data(), m, top, xi, x);
    if (threadIdx.x == 0) { const long long t1 = clock64(); sub_[5] += t1 - t_; t_ = t1; }

    // Once the reach exceeds the routing budget, skip the serial level bucketing and use the fixed
    // dense schedule. The reach closure guarantees entries outside the preloaded pattern stay zero.
    const i_t div                  = c_dense_fallback_divisor;
    const i_t dense_fallback_reach = div > 0 ? (m + div - 1) / div : m + 1;
    if (threadIdx.x == 0) { ++route_[m - top >= dense_fallback_reach ? 2 : 1]; }
    if (m - top >= dense_fallback_reach) {
      warp_solve_body_op<Op>(c->corder[op], c->cs[op], c->lstart[op], c->nlev[op], c->ridx[op],
                             c->val[op], x);
      if (threadIdx.x == 0) {
        const long long d = clock64() - t_;
        sub_[0] += d, wop_[op] += d, ++wn_[op];
      }
      return top;
    }

    // (3) level bucketing. EVERY lane needs nlevR: it is the wavefront's internal __syncwarp count,
    // so a lane that disagreed would desynchronize the whole warp. The max-reduce already leaves it
    // warp-uniform, so no broadcast is needed.
    const i_t* level = c->level[op];
    const i_t nlevR  = warp_sptrsv_maxlevel_body(level, m, top, xi);
    {
      const int lane = static_cast<int>(threadIdx.x);
      i_t* sp_ls     = c->sp_ls;
      for (i_t k = lane; k <= nlevR; k += 32) { sp_ls[k] = 0; }
      __syncwarp(0xffffffffu);
    }
    // The histogram, its prefix sum and the counting-sort scatter are order-dependent (sp_co's
    // within-level order comes from the running sp_pos cursor), so they stay on one lane.
    lane0_serial([&] {
      i_t* sp_ls  = c->sp_ls;
      i_t* sp_co  = c->sp_co;
      i_t* sp_pos = c->sp_pos;
      // No-op columns are skipped: x[j] already holds what the solve would have written (the
      // scattered rhs, or the zero warp_sptrsv_preload_body left there). `top` and xi[top..m) are
      // deliberately NOT filtered -- the reachable PATTERN is unchanged, only the work list.
      for (i_t p = top; p < m; ++p) {
        const i_t w = level[xi[p]];
        if (level_is_noop(w)) { continue; }
        sp_ls[factor_level(w)]++;                                       // level values ∈ [1, nlevR]
      }
      for (i_t k = 1; k <= nlevR; ++k) { sp_ls[k] += sp_ls[k - 1]; }    // sp_ls[lv]=count(level<=lv)
      for (i_t lv = 1; lv <= nlevR; ++lv) { sp_pos[lv] = sp_ls[lv - 1]; }
      for (i_t p = top; p < m; ++p) {
        const i_t j = xi[p];
        const i_t w = level[j];
        if (level_is_noop(w)) { continue; }
        sp_co[sp_pos[factor_level(w)]++] = j;
      }
    });

    if (threadIdx.x == 0) { const long long t1 = clock64(); sub_[2] += t1 - t_; t_ = t1; }
    warp_sptrsv_body_op<Op>(c->cs[Op], c->ridx[Op], c->val[Op], c->sp_co, c->sp_ls,
                            static_cast<int>(nlevR), x);
    if (threadIdx.x == 0) { sub_[3] += clock64() - t_; }
    return top;
  }


  // Warp-parallel gpu_sparse_vector::to_dense — lane 0 sizes the dense buffer (no zero-fill), the warp
  // zeros [0,n) + scatters the nnz. Bit-exact (distinct sparse indices). Replaces the O(n) serial tail
  // before each dense solve.
  __device__ void warp_to_dense(const gpu_sparse_vector& sv, gpu_span<f_t>& dense) const
  {
    const i_t n = sv.n;
    // reserve() may device-malloc and set_size() is a header RMW → lane 0 only. The __syncwarp
    // inside lane0_serial is also what publishes the (possibly relocated) data_ pointer, so
    // dense.data() below must be read AFTER it, not before.
    lane0_serial([&] { dense.reserve(n); dense.set_size(n); });
    warp_todense_body(dense.data(), sv.i.data(), n, static_cast<i_t>(sv.i.size()), sv.x.data());
  }

  // Warp-parallel gpu_sparse_vector::from_dense — lane 0 reserves the sparse buffers, the warp compacts
  // the nonzeros of `dense` in increasing index order, lane 0 sets the final size. Bit-exact.
  __device__ void warp_from_dense(const gpu_span<f_t>& dense, gpu_sparse_vector& sv) const
  {
    const i_t n = static_cast<i_t>(dense.size());
    lane0_serial([&] { sv.n = n; sv.i.reserve(n); sv.x.reserve(n); });
    i_t count = 0;                   // the body writes it on lane 0 only; each lane has its own
    warp_fromdense_body(&count, sv.i.data(), dense.data(), n, sv.x.data());
    // set_size is a header RMW; lane 0 already holds the right count, and the trailing
    // __syncwarp makes the new size visible to every lane's later sv.i.size() read.
    lane0_serial([&] { sv.i.set_size(count); sv.x.set_size(count); });
  }





  __device__ f_t estimate_solution_density(f_t rhs_nz,
                                           f_t sum,
                                           i_t& num_calls,
                                           f_t max_growth,
                                           bool& use_hypersparse) const
  {
    num_calls++;
    const f_t average_growth    = fmax(1.0, sum / static_cast<f_t>(num_calls));
    const f_t blended_growth    = 0.75 * average_growth + 0.25 * fmax(average_growth, max_growth);
    const f_t predicted_nz      = rhs_nz * blended_growth;
    const f_t predicted_density = predicted_nz / static_cast<f_t>(L0_.m);
    use_hypersparse             = predicted_density < hypersparse_threshold_;
    return predicted_nz;
  }

  // Warp-parallel `dst = src` for a sparse vector. Same result as the defaulted copy-assign
  // (memberwise gpu_span::operator=, which is size_=0; reserve; copy; size_=n).
  // Callers never self-assign or alias here; both call sites copy between distinct vectors.
  __device__ void warp_sv_copy(const gpu_sparse_vector& src, gpu_sparse_vector& dst) const
  {
    const i_t nz = static_cast<i_t>(src.i.size());
    // reserve() may device-malloc; read dst.i.data()/dst.x.data() only after lane0_serial's
    // trailing __syncwarp has published the (possibly relocated) pointers.
    lane0_serial([&] {
      dst.n = src.n;
      dst.i.clear();
      dst.x.clear();
      dst.i.reserve(nz);
      dst.x.reserve(nz);
    });
    warp_copy_i_body(dst.i.data(), src.i.data(), nz);
    warp_copy_f_body(dst.x.data(), src.x.data(), nz);
    lane0_serial([&] { dst.i.set_size(nz); dst.x.set_size(nz); });
  }

  // Warp-parallel solve_to_sparse_vector. Same contract, same output, same cleared workspaces.
  // See warp_gather_reach_body for why it is bit-identical.
  __device__ void warp_solve_to_sparse_vector(i_t top, gpu_sparse_vector& out) const
  {
    const i_t m  = L0_.m;
    const i_t nz = m - top;
    // reserve() may device-malloc; the __syncwarp inside lane0_serial publishes the (possibly
    // relocated) data_ pointers, so out.i.data()/out.x.data() must be read AFTER it.
    lane0_serial([&] {
      out.n = m;
      out.x.clear();
      out.i.clear();
      out.x.reserve(nz);
      out.i.reserve(nz);
    });
    i_t count = 0;   // every lane keeps its own; the body makes it warp-uniform
    warp_gather_reach_body(&count, out.i.data(), out.x.data(), top, m, xi_workspace_.data(),
                           x_workspace_.data());
    lane0_serial([&] { out.i.set_size(count); out.x.set_size(count); });
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

  // Reserve the rank-1 store to its caps. Called once per slot, after attach_all_buffers and before any
  // update. Nothing here allocates: the spans already point at their slot slices.
  // `col_start` needs 2 entries per update plus the leading 0: the update appends TWO columns.
  __device__ void reserve_store(i_t m)
  {
    if (update_cap_ > 0 && eta_nnz_cap_ > 0) {
      // Capacity FIRST, so init_empty's resize(1) fits inside it and never reallocates (a realloc
      // capacity is fixed by the slot layout, so this only publishes the length).
      S_.col_start.reserve(2 * update_cap_ + 1);
      S_.i.reserve(eta_nnz_cap_);
      S_.x.reserve(eta_nnz_cap_);
      mu_values_.reserve(update_cap_);   // push_back only; capacity is enough to stop it doubling
      S_.init_empty(m);
      // grow_storage compares against size(), not capacity, so publish the caps as the size. Same
      // reserve()+set_size() idiom as warp_to_dense; the capacity is already in place above, so
      // set_size only moves size_. NOT resize(), which would value-fill ~10^5 entries on one lane
      // per slot for entries that are only ever reached through S_.col_start, i.e. only where an
      // append already wrote.
      // col_start needs 2 entries per update plus the leading 0; that exact length is what makes
      // the bounds check fire on update index == update_cap_ and not before.
      S_.col_start.set_size(2 * update_cap_ + 1);
      S_.i.set_size(eta_nnz_cap_);
      S_.x.set_size(eta_nnz_cap_);
      S_.col_start[0] = 0;   // init_empty set this, but the size bump above must not lose it
    } else {
      S_.init_empty(m);
    }
    store_exhausted_ = false;
  }

  // Bounds-check the store for one more update. Returns false (and sets store_exhausted_) when the
  // update would not fit; the caller must then stop and hand the node back for refactorization.
  //
  // This used to resize S_ by doubling, which is exactly the dynamic allocation that made a slot
  // un-reusable across launches. With the caps set it only checks; with them unset it falls back to
  // the old growth so an uncapped caller behaves as before.
  __device__ bool grow_storage(i_t nz, i_t& S_start, i_t& S_nz)
  {
    const i_t last_S_col     = num_updates_ * 2;
    const i_t new_last_S_col = last_S_col + 2;
    const bool capped        = (update_cap_ > 0 && eta_nnz_cap_ > 0);

    if (new_last_S_col >= static_cast<i_t>(S_.col_start.size())) {
      if (capped) { store_exhausted_ = true; return false; }
      S_.col_start.resize(new_last_S_col + refactor_frequency_);
    }
    S_nz = S_.col_start[last_S_col];
    if (S_nz + nz > static_cast<i_t>(S_.i.size())) {
      if (capped) { store_exhausted_ = true; return false; }
      const i_t newsz = (2 * S_nz > S_nz + nz) ? 2 * S_nz : S_nz + nz;
      S_.i.resize(newsz);
      S_.x.resize(newsz);
    }
    S_start = last_S_col;
    return true;
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
  // `Direct` is only ever true together with Warp — all 32 lanes execute the body
  // above gpu_basis_update_mpf. It selects direct body calls + lane-0 guards on the serial glue.
  __device__ void u_transpose_solve(f_t* rhs) const
  {
    warp_dispatch<WARP_UT_SOLVE>(rhs);
  }
  __device__ void u_transpose_solve(gpu_sparse_vector& rhs) const
  {
    i_t top;
    top = warp_dispatch_sptrsv<WARP_UT_SOLVE>(rhs, U0_transpose_, x_workspace_.data());
    // push_back + workspace RMW -> lane 0 only under Direct.
    warp_solve_to_sparse_vector(top, rhs);
  }

  __device__ void l_transpose_solve(f_t* rhs) const
  {
    const f_t zero_tol = 1e-13;
    // dot col = u_col (2k), axpy col = v_col (2k+1), descending k.
    if (num_updates_ > 0) {
      warp_dispatch_tloop(rhs, /*asc=*/false, /*dotoff=*/0, /*axpyoff=*/1, zero_tol);
    }
    warp_dispatch<WARP_LT_SOLVE>(rhs);
  }
  __device__ void l_transpose_solve(gpu_sparse_vector& rhs) const
  {
    const i_t m = L0_.m;
    gpu_sparse_vector& b = sparse_scratch_;
    // Keep the sparse representation through the MPF T^T replay. The historical path ran this
    // whole section on lane 0; the forward sparse L solve already proved the same membership-list
    // representation can preserve fill order while distributing dot/AXPY work over the warp.
    i_t nz = static_cast<i_t>(rhs.i.size());
    lane0_serial([&] { scatter_into_workspace(rhs); });
    const f_t zero_tol = 1e-13;
    if (num_updates_ > 0) {
      const long long t_ = clock64();
      // T^T: updates descend; dot u_col (2k), then AXPY v_col (2k+1).
      warp_tloop_sparse_body<false, 0, 1>(
        S_.col_start.data(), mu_values_.data(), num_updates_, S_.i.data(), zero_tol,
        S_.x.data(), m, xi_workspace_.data(), &nz, x_workspace_.data());
      if (threadIdx.x == 0) { sub_[7] += clock64() - t_; }
    }

    // Warp-compacted equivalent of gather_into_sparse_vector(nz, b), including workspace clears.
    lane0_serial([&] {
      b.n = m;
      b.i.clear();
      b.x.clear();
      b.i.reserve(nz);
      b.x.reserve(nz);
    });
    i_t count = 0;
    warp_gather_workspace_body(&count, b.i.data(), b.x.data(), m, xi_workspace_.data(), nz,
                               x_workspace_.data());
    lane0_serial([&] { b.i.set_size(count); b.x.set_size(count); });
    i_t top;
    top = warp_dispatch_sptrsv<WARP_LT_SOLVE>(b, L0_transpose_, x_workspace_.data());
    warp_solve_to_sparse_vector(top, rhs);
  }

  __device__ void u_solve(f_t* rhs) const
  {
    warp_dispatch<WARP_U_SOLVE>(rhs);
  }
  __device__ void u_solve2(f_t* rhs0, f_t* rhs1) const
  {
    warp_dispatch2<WARP_U_SOLVE>(rhs0, rhs1);
  }
  __device__ void u_solve(gpu_sparse_vector& rhs) const
  {
    i_t top;
    top = warp_dispatch_sptrsv<WARP_U_SOLVE>(rhs, U0_, x_workspace_.data());
    warp_solve_to_sparse_vector(top, rhs);
  }

  __device__ void l_solve(f_t* rhs) const
  {
    warp_dispatch<WARP_L_SOLVE>(rhs);
    const f_t zero_tol = 1e-16;  // Any higher and pilot_ja fails
    // dot col = v_col (2k+1), axpy col = u_col (2k), ascending k.
    if (num_updates_ > 0) {
      warp_dispatch_tloop(rhs, /*asc=*/true, /*dotoff=*/1, /*axpyoff=*/0, zero_tol);
    }
  }
  __device__ void l_solve2(f_t* rhs0, f_t* rhs1) const
  {
    warp_dispatch2<WARP_L_SOLVE>(rhs0, rhs1);
    if (num_updates_ > 0) {
      const long long t_ = clock64();
      warp_tloop2_body<true, 0, 1>(S_.col_start.data(), mu_values_.data(), num_updates_,
                                   S_.i.data(), f_t{1e-16}, S_.x.data(), rhs0, rhs1);
      if (threadIdx.x == 0) { sub_[6] += clock64() - t_; }
    }
  }
  __device__ void l_solve(gpu_sparse_vector& rhs) const
  {
    const i_t m = L0_.m;
    i_t top;
    top = warp_dispatch_sptrsv<WARP_L_SOLVE>(rhs, L0_, x_workspace_.data());
    // The whole tail is collective now. This used to be one lane0_serial block containing the
    // MPF T-loop, which is why the hypersparse l_solve cost ~2.7x the dense one per call.
    const f_t zero_tol = 1e-13;
    i_t nz             = m - top;
    warp_solve_to_workspace_body(m, xi_workspace_.data(), top);
    if (num_updates_ > 0) {
      const long long t_ = clock64();
      // S_ pointers are re-read here (they move on resize) — every lane reads the same globals.
      warp_tloop_sparse_body<true, 1, 0>(
        S_.col_start.data(), mu_values_.data(), num_updates_, S_.i.data(), zero_tol,
        S_.x.data(), m, xi_workspace_.data(), &nz, x_workspace_.data());
      if (threadIdx.x == 0) { sub_[7] += clock64() - t_; }
    }
    // gather_into_sparse_vector, warp form. reserve() may device-malloc, so the data() pointers
    // must be read after lane0_serial's __syncwarp. n is deliberately left alone: the serial
    // gather_into_sparse_vector did not set it either.
    lane0_serial([&] {
      rhs.i.clear();
      rhs.x.clear();
      rhs.i.reserve(nz);
      rhs.x.reserve(nz);
    });
    i_t count = 0;
    warp_gather_workspace_body(&count, rhs.i.data(), rhs.x.data(), m, xi_workspace_.data(), nz,
                               x_workspace_.data());
    lane0_serial([&] { rhs.i.set_size(count); rhs.x.set_size(count); });
  }

  // ---- B solves ----
  // NOTE (Direct): `use_hypersparse` decides which SEQUENCE of collectives the warp executes, and
  // estimate_solution_density MUTATES the routing statistics (num_calls_/sum_), so it must run
  // exactly once — on lane 0 — and the resulting route must then be broadcast. If a lane disagreed
  // about the route it would call a different body and the warp would desynchronize.
  __device__ i_t b_solve(const gpu_span<f_t>& rhs, gpu_span<f_t>& solution) const
  {
    warp_permute_dense_body(rhs.data(), L0_.m, row_permutation_.data(), solution.data());
    l_solve(solution.data());
    u_solve(solution.data());
    return 0;
  }

  __device__ i_t b_solve(const gpu_sparse_vector& rhs,
                         gpu_sparse_vector& solution,
                         gpu_sparse_vector& Lsol,
                         bool need_Lsol = true) const
  {
    bool use_hypersparse = false;
    const f_t input_size = static_cast<f_t>(rhs.i.size());
    warp_sv_copy(rhs, solution);
    warp_permute_idx_body(solution.i.data(), static_cast<i_t>(solution.i.size()),
                          inverse_row_permutation_.data());
    lane0_serial([&] {
      estimate_solution_density(input_size, sum_L_, num_calls_L_, max_L_, use_hypersparse);
    });
    use_hypersparse = lane0_bcast(static_cast<int>(use_hypersparse)) != 0;
    // TRIED AND REVERTED: eliding the U leg's warp_to_dense when the L leg already routed dense.
    // dense_scratch_ still holds dense(solution) at that point (from_dense keeps exactly the
    // v != 0.0 entries, to_dense zeroes and re-scatters those same entries), so skipping it is
    // bit-identical and removes an O(m) zero + scatter on the ~618k calls per run where both legs
    // are dense. But carrying the one bool it needs cost one live register: the monolith went
    // 128 -> 129, and blk/SM = 4*floor(16384/(32*roundup8(regs))) then drops 16 -> 12.
    // Measured 4.701 -> 5.151 s. On this kernel a single register is worth more than the work.
    if (use_hypersparse) {
      l_solve(solution);
    } else {
      gpu_span<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      warp_to_dense(solution, solution_dense);
      l_solve(solution_dense.data());
      warp_from_dense(solution_dense, solution);
    }
    if (need_Lsol) { warp_sv_copy(solution, Lsol); }
    lane0_serial([&] {
      const f_t growth = static_cast<f_t>(solution.i.size()) / input_size;
      sum_L_ += growth;
      max_L_ = fmax(max_L_, growth);
    });

    const f_t rhs_size = static_cast<f_t>(solution.i.size());
    lane0_serial([&] {
      estimate_solution_density(rhs_size, sum_U_, num_calls_U_, max_U_, use_hypersparse);
    });
    use_hypersparse = lane0_bcast(static_cast<int>(use_hypersparse)) != 0;
    if (use_hypersparse) {
      u_solve(solution);
    } else {
      gpu_span<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      warp_to_dense(solution, solution_dense);
      u_solve(solution_dense.data());
      warp_from_dense(solution_dense, solution);
    }
    lane0_serial([&] {
      const f_t growth = static_cast<f_t>(solution.i.size()) / rhs_size;
      sum_U_ += growth;
      max_U_ = fmax(max_U_, growth);
    });
    return 0;
  }

  // Solve the entering-column and steepest-edge RHS together. Dense/dense legs share one factor
  // traversal and one level-barrier sequence; a leg whose route differs uses the existing one-RHS
  // sparse/dense implementation.
  __device__ i_t b_solve2(const gpu_sparse_vector& rhs0,
                          const gpu_sparse_vector& rhs1,
                          gpu_sparse_vector& solution0,
                          gpu_sparse_vector& Lsol0,
                          gpu_sparse_vector& solution1) const
  {
    const f_t input0 = static_cast<f_t>(rhs0.i.size());
    const f_t input1 = static_cast<f_t>(rhs1.i.size());
    warp_sv_copy(rhs0, solution0);
    warp_sv_copy(rhs1, solution1);
    warp_permute_idx_body(solution0.i.data(), static_cast<i_t>(solution0.i.size()),
                          inverse_row_permutation_.data());
    warp_permute_idx_body(solution1.i.data(), static_cast<i_t>(solution1.i.size()),
                          inverse_row_permutation_.data());

    bool sparse0 = false;
    bool sparse1 = false;
    lane0_serial([&] {
      estimate_solution_density(input0, sum_L_, num_calls_L_, max_L_, sparse0);
      estimate_solution_density(input1, sum_L_, num_calls_L_, max_L_, sparse1);
      // Size m means the dense buffer still contains this RHS's exact post-L values.
      if (sparse0) { dense_scratch_.set_size(0); }
      if (sparse1) { dense_scratch2_.set_size(0); }
    });
    sparse0 = lane0_bcast(static_cast<int>(sparse0)) != 0;
    sparse1 = lane0_bcast(static_cast<int>(sparse1)) != 0;

    gpu_span<f_t>& dense0 = dense_scratch_;
    gpu_span<f_t>& dense1 = dense_scratch2_;
    path(!sparse0 && !sparse1 ? P_L_PAIRED : P_L_SPLIT);
    path(P_FTRAN_SPARSE_RHS, static_cast<unsigned>(sparse0) + static_cast<unsigned>(sparse1));
    if (!sparse0 && !sparse1) {
      warp_to_dense(solution0, dense0);
      warp_to_dense(solution1, dense1);
      l_solve2(dense0.data(), dense1.data());
      warp_from_dense(dense0, solution0);
      warp_from_dense(dense1, solution1);
    } else {
      if (sparse0) {
        l_solve(solution0);
      } else {
        warp_to_dense(solution0, dense0);
        l_solve(dense0.data());
        warp_from_dense(dense0, solution0);
      }
      if (sparse1) {
        l_solve(solution1);
      } else {
        warp_to_dense(solution1, dense1);
        l_solve(dense1.data());
        warp_from_dense(dense1, solution1);
      }
    }
    lane0_serial([&] {
      const f_t growth0 = static_cast<f_t>(solution0.i.size()) / input0;
      sum_L_ += growth0;
      max_L_ = fmax(max_L_, growth0);
      const f_t growth1 = static_cast<f_t>(solution1.i.size()) / input1;
      sum_L_ += growth1;
      max_L_ = fmax(max_L_, growth1);
    });

    const f_t usize0 = static_cast<f_t>(solution0.i.size());
    const f_t usize1 = static_cast<f_t>(solution1.i.size());
    lane0_serial([&] {
      estimate_solution_density(usize0, sum_U_, num_calls_U_, max_U_, sparse0);
      estimate_solution_density(usize1, sum_U_, num_calls_U_, max_U_, sparse1);
      // A sparse U route invalidates the second dense buffer. update_steepest_edge_norms uses size
      // m to distinguish an in-place dense result from the sparse fallback.
      if (sparse1) { dense_scratch2_.set_size(0); }
    });
    sparse0 = lane0_bcast(static_cast<int>(sparse0)) != 0;
    sparse1 = lane0_bcast(static_cast<int>(sparse1)) != 0;

    i_t output1_nz = 0;
    path(!sparse0 && !sparse1 ? P_U_PAIRED : P_U_SPLIT);
    path(P_FTRAN_SPARSE_RHS, static_cast<unsigned>(sparse0) + static_cast<unsigned>(sparse1));
    if (!sparse0 && !sparse1) {
      if (dense0.size() != static_cast<size_t>(L0_.m)) { warp_to_dense(solution0, dense0); }
      if (dense1.size() != static_cast<size_t>(L0_.m)) { warp_to_dense(solution1, dense1); }
      // U consumes dense0, so preserve its L intermediate with an O(1) span swap.
      lane0_serial([&] {
        solution0.i.swap(Lsol0.i);
        solution0.x.swap(Lsol0.x);
      });
      u_solve2(dense0.data(), dense1.data());
      warp_from_dense(dense0, solution0);
      // The steepest-edge update consumes dense1 in place; routing statistics only need its nnz.
      output1_nz = warp_dense_nnz(dense1.data(), L0_.m);
    } else {
      if (sparse0) {
        warp_sv_copy(solution0, Lsol0);
        u_solve(solution0);
      } else {
        if (dense0.size() != static_cast<size_t>(L0_.m)) { warp_to_dense(solution0, dense0); }
        lane0_serial([&] {
          solution0.i.swap(Lsol0.i);
          solution0.x.swap(Lsol0.x);
        });
        u_solve(dense0.data());
        warp_from_dense(dense0, solution0);
      }
      if (sparse1) {
        u_solve(solution1);
        output1_nz = static_cast<i_t>(solution1.i.size());
      } else {
        if (dense1.size() != static_cast<size_t>(L0_.m)) { warp_to_dense(solution1, dense1); }
        u_solve(dense1.data());
        output1_nz = warp_dense_nnz(dense1.data(), L0_.m);
      }
    }
    lane0_serial([&] {
      const f_t growth0 = static_cast<f_t>(solution0.i.size()) / usize0;
      sum_U_ += growth0;
      max_U_ = fmax(max_U_, growth0);
      const f_t growth1 = static_cast<f_t>(output1_nz) / usize1;
      sum_U_ += growth1;
      max_U_ = fmax(max_U_, growth1);
    });
    return 0;
  }

  __device__ i_t b_solve(const gpu_sparse_vector& rhs, gpu_sparse_vector& solution) const
  {
    // Lsol is nz=0 => resize(0) => no malloc, and need_Lsol=false means it is never written.
    // Safe to leave as a per-lane local under Direct.
    gpu_sparse_vector Lsol(1, 0);
    return b_solve(rhs, solution, Lsol, false);
  }

  __device__ i_t b_transpose_solve(const gpu_sparse_vector& rhs,
                                   gpu_sparse_vector& solution,
                                   gpu_sparse_vector& UTsol) const
  {
    bool use_hypersparse = false;
    const f_t input_size = static_cast<f_t>(rhs.i.size());
    lane0_serial([&] {
      estimate_solution_density(
        input_size, sum_U_transpose_, num_calls_U_transpose_, max_U_transpose_, use_hypersparse);
    });
    use_hypersparse = lane0_bcast(static_cast<int>(use_hypersparse)) != 0;
    path(use_hypersparse ? P_UT_SPARSE : P_UT_DENSE);
    path(P_MPF_LEN, static_cast<unsigned>(num_updates_));
    // NO redundant-to_dense elision here, unlike b_solve. It is valid, but u_transpose_solve routes
    // hypersparse on 99.9% of calls (2,131,200 vs 2,700), so the flag would almost always be false
    // — and carrying it cost one extra live register, which took the monolith 128 -> 129 and so
    // 16 -> 12 blocks/SM (blk/SM = 4*floor(16384/(32*roundup8(regs)))). Measured: 4.701 -> 5.151 s.
    if (use_hypersparse) {
      warp_sv_copy(rhs, solution);
      u_transpose_solve(solution);
    } else {
      gpu_span<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      warp_to_dense(rhs, solution_dense);
      u_transpose_solve(solution_dense.data());
      warp_from_dense(solution_dense, solution);
    }
    warp_sv_copy(solution, UTsol);
    lane0_serial([&] {
      const f_t growth = static_cast<f_t>(solution.i.size()) / input_size;
      sum_U_transpose_ += growth;
      max_U_transpose_ = fmax(max_U_transpose_, growth);
    });

    const f_t rhs_size = static_cast<f_t>(solution.i.size());
    lane0_serial([&] {
      estimate_solution_density(
        rhs_size, sum_L_transpose_, num_calls_L_transpose_, max_L_transpose_, use_hypersparse);
    });
    use_hypersparse = lane0_bcast(static_cast<int>(use_hypersparse)) != 0;
    path(use_hypersparse ? P_LT_SPARSE : P_LT_DENSE);
    if (use_hypersparse) {
      l_transpose_solve(solution);
    } else {
      gpu_span<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      warp_to_dense(solution, solution_dense);
      l_transpose_solve(solution_dense.data());
      warp_from_dense(solution_dense, solution);
    }
    lane0_serial([&] {
      const f_t growth = static_cast<f_t>(solution.i.size()) / rhs_size;
      sum_L_transpose_ += growth;
      max_L_transpose_ = fmax(max_L_transpose_, growth);
    });
    warp_permute_idx_body(solution.i.data(), static_cast<i_t>(solution.i.size()),
                          row_permutation_.data());
    return 0;
  }

  // ---- stacking rank-1 update (sparse form) ----
  // Collective form of update(). Same sequence, same results; only the data-parallel spans are
  // hoisted out of the lane-0 region. The fill-tracking add_sparse_column, the two sorts and the
  // mu dot product stay on lane 0 (order-dependent append cursor / floating-point order).
  // sub (optional, diagnostics): lane 0 adds the cycles of each step to sub[0..8], merged nnz to sub[9].
  __device__ i_t warp_update(const gpu_sparse_vector& utilde, gpu_sparse_vector& etilde,
                             i_t leaving_index, int* sub = nullptr)
  {
    const i_t m = L0_.m;
    i_t nz      = static_cast<i_t>(utilde.i.size());
    long long sub_t = clock64();
    const auto sub_mark = [&](int k) {
      if (sub != nullptr) {
        __syncwarp(0xffffffffu);
        if (threadIdx.x == 0) {
          const long long c = clock64();
          sub[k] += static_cast<int>(c - sub_t);
          sub_t = c;
        }
      }
    };

    warp_scatter_ws_body(utilde.i.data(), utilde.x.data(), m, nz, xi_workspace_.data(),
                         x_workspace_.data());
    sub_mark(0);

    // add_sparse_column appends fill to the index list through a running cursor, and gpu_sort is
    // serial; grow_storage allocates. One lane, one region.
    i_t base = 0;
    lane0_serial([&] {
      add_sparse_column(U0_, leaving_index, -1.0, xi_workspace_.data(), nz, x_workspace_.data());
      if (sub != nullptr) {
        const long long c = clock64();
        sub[1] += static_cast<int>(c - sub_t);
        sub[9] = nz;
        sub_t  = c;
      }
      gpu_sort(xi_workspace_.data() + m, xi_workspace_.data() + m + nz,
               [] __device__(i_t a, i_t b) { return a < b; });
      if (sub != nullptr) {
        const long long c = clock64();
        sub[2] += static_cast<int>(c - sub_t);
        sub_t  = c;
      }
      i_t S_start_l = 0, S_nz_l = 0;
      if (grow_storage(nz + static_cast<i_t>(etilde.i.size()), S_start_l, S_nz_l)) {
        base = S_.col_start[S_.n];
      }
    });
    // The store is full: stop before touching S_, so the stack stays exactly as it was and the host
    // can refactor from the CURRENT basis. Mirrors the bail-out below -- workspace not cleared,
    // num_updates_ not bumped.
    path(P_UTILDE_NZ, static_cast<unsigned>(utilde.i.size()));
    if (lane0_bcast(store_exhausted_ ? 1 : 0)) {
      path(P_UPD_FULL);
      return 2;
    }
    nz   = lane0_bcast(nz);     // add_sparse_column grew it on lane 0 only
    base = lane0_bcast(base);
    sub_mark(3);
    // S_.i/S_.x may have been relocated by grow_storage — data() is read here, after the barrier.
    const i_t end1 =
      warp_append_scatter_body(base, S_.i.data(), S_.x.data(), xi_workspace_.data() + m, nz,
                               x_workspace_.data());
    sub_mark(4);

    i_t base2 = 0;
    lane0_serial([&] {
      S_.col_start[S_.n + 1] = end1;
      S_.n++;
      path(P_ETILDE_NZ, static_cast<unsigned>(etilde.i.size()));
      if (etilde.i.size() > 1) {
        path(etilde.i.size() > 0.3 * etilde.n ? P_ETILDE_BUCKET : P_ETILDE_PERM);
      }
      etilde.sort(sort_bucket_, sort_i_, sort_x_, sort_perm_);   // reused scratch, may reallocate
      base2 = S_.col_start[S_.n];
    });
    base2 = lane0_bcast(base2);
    sub_mark(5);
    const i_t enz  = static_cast<i_t>(etilde.i.size());
    const i_t end2 =
      warp_append_sv_body(base2, S_.i.data(), S_.x.data(), etilde.i.data(), etilde.x.data(), enz);
    sub_mark(6);

    i_t status = 0;
    lane0_serial([&] {
      S_.col_start[S_.n + 1] = end2;
      S_.n++;
      // The two columns just appended are S_.n-2 and S_.n-1; that is grow_storage's S_start.
      const i_t S_start = S_.n - 2;
      const f_t mu      = 1.0 + sparse_dot(S_.i.data() + S_.col_start[S_start],
                                      S_.x.data() + S_.col_start[S_start],
                                      S_.col_start[S_start + 1] - S_.col_start[S_start],
                                      S_.i.data() + S_.col_start[S_start + 1],
                                      S_.x.data() + S_.col_start[S_start + 1],
                                      S_.col_start[S_start + 2] - S_.col_start[S_start + 1]);
      if (fabs(mu) < 1E-8 || fabs(mu) > 1E+8) { status = 1; }
      else { mu_values_.push_back(mu); }            // CPU form: store mu, divide at apply
    });
    status = lane0_bcast(status);
    sub_mark(7);
    // Matches the serial version: on the numerical bail-out the workspace is NOT cleared and
    // num_updates_ is NOT bumped.
    if (status == 1) {
      path(P_UPD_MU);
      return 1;
    }
    path(P_UPD_OK);

    warp_clear_ws_body(m, xi_workspace_.data(), nz, x_workspace_.data());
    lane0_serial([&] { num_updates_++; });
    sub_mark(8);
    return 0;
  }

  // warp_update without sorting (fission): both columns are appended in arrival order, and mu is a
  // gather of etilde against column 2k, still dense in x_workspace_. Every reader of S_ gathers or
  // scatters through a dense vector, so index order changes only summation order.
  __device__ i_t warp_update_unsorted(const gpu_sparse_vector& utilde, const gpu_sparse_vector& etilde,
                                      i_t leaving_index, int* sub = nullptr)
  {
    const i_t m     = L0_.m;
    const int lane  = static_cast<int>(threadIdx.x);
    i_t nz          = static_cast<i_t>(utilde.i.size());
    const i_t enz   = static_cast<i_t>(etilde.i.size());
    long long sub_t = clock64();
    const auto sub_mark = [&](int k) {
      if (sub != nullptr) {
        __syncwarp(0xffffffffu);
        if (threadIdx.x == 0) {
          const long long c = clock64();
          sub[k] += static_cast<int>(c - sub_t);
          sub_t = c;
        }
      }
    };
    warp_scatter_ws_body(utilde.i.data(), utilde.x.data(), m, nz, xi_workspace_.data(),
                         x_workspace_.data());
    sub_mark(0);
    i_t base = 0;
    lane0_serial([&] {
      add_sparse_column(U0_, leaving_index, -1.0, xi_workspace_.data(), nz, x_workspace_.data());
      if (sub != nullptr) { sub[9] = nz; }
    });
    sub_mark(1);
    lane0_serial([&] {
      i_t S_start_l = 0, S_nz_l = 0;
      if (grow_storage(nz + enz, S_start_l, S_nz_l)) { base = S_.col_start[S_.n]; }
    });
    path(P_UTILDE_NZ, static_cast<unsigned>(utilde.i.size()));
    if (lane0_bcast(store_exhausted_ ? 1 : 0)) {
      path(P_UPD_FULL);
      return 2;
    }
    nz   = lane0_bcast(nz);
    base = lane0_bcast(base);
    sub_mark(3);
    const i_t end1 =
      warp_append_scatter_body(base, S_.i.data(), S_.x.data(), xi_workspace_.data() + m, nz,
                               x_workspace_.data());
    sub_mark(4);
    const i_t end2 =
      warp_append_sv_body(end1, S_.i.data(), S_.x.data(), etilde.i.data(), etilde.x.data(), enz);
    sub_mark(6);
    f_t part = 0.0;
    for (i_t k = lane; k < enz; k += 32) { part += x_workspace_[etilde.i[k]] * etilde.x[k]; }
    const f_t mu = 1.0 + warp_reduce_add(part);
    i_t status   = 0;
    lane0_serial([&] {
      path(P_ETILDE_NZ, static_cast<unsigned>(enz));
      S_.col_start[S_.n + 1] = end1;
      S_.col_start[S_.n + 2] = end2;
      S_.n += 2;
      if (fabs(mu) < 1E-8 || fabs(mu) > 1E+8) { status = 1; }
      else { mu_values_.push_back(mu); }
    });
    status = lane0_bcast(status);
    sub_mark(7);
    if (status == 1) {
      path(P_UPD_MU);
      return 1;
    }
    path(P_UPD_OK);
    warp_clear_ws_body(m, xi_workspace_.data(), nz, x_workspace_.data());
    lane0_serial([&] { num_updates_++; });
    sub_mark(8);
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
  // Host-built CSR of A (Arow). Always uploaded.
  const i_t* Arow_row_start{nullptr};
  const i_t* Arow_j{nullptr};
  const f_t* Arow_x{nullptr};
};

struct gpu_settings {
  f_t primal_tol, dual_tol, pivot_tol, tight_tol, fixed_tol, zero_tol, time_limit, cut_off,
    steepest_edge_ratio;
  i_t refactor_frequency;
  // CUOPT_BNB_GPU_YIELD: seats finished in this launch; at yield_at the rest exit to resume later.
  const unsigned int* yield_count{nullptr};
  unsigned int yield_at{0};
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
__device__ void update_objective(const i_t* basic_list,
                                 const gpu_sparse_vector& changed_basic_indices,
                                 const gpu_span<f_t>& objective,
                                 const f_t* delta_x,
                                 i_t entering_index,
                                 f_t& obj)
{
  const i_t changed_basic_nz = (i_t)changed_basic_indices.i.size();
  const int lane = static_cast<int>(threadIdx.x);
  f_t part = 0.0;
  for (i_t k = lane; k < changed_basic_nz; k += 32) {
    const i_t j = basic_list[changed_basic_indices.i[k]];
    part += delta_x[j] * objective[j];
  }
  const f_t total = warp_reduce_add(part);   // every lane leaves with the full sum
  // `obj` aliases st.obj in global memory — one writer only.
  lane0_serial([&] {
    obj += total;
    // Leaving index already included above
    obj += delta_x[entering_index] * objective[entering_index];
  });
}

// Direct=true (warp): ALL 32 lanes execute this, so the collective is called
// directly (no cfg_shared / op-code dispatch) and the lane-0-only *dz_count must be broadcast before
// the global-memory set_size. Direct=false (monolith): only lane 0 is here — a __shfl_sync would
// deadlock against the lanes parked in warp_helper_loop. Same contract as compute_delta_z.
__device__ void compute_reduced_cost_update(const gpu_lp& lp,
                                            const i_t* nonbasic_list,
                                            const f_t* delta_y,
                                            i_t leaving_index,
                                            i_t direction,
                                            i_t* delta_z_mark,
                                            gpu_span<i_t>& delta_z_indices,
                                            f_t* delta_z)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  // One warp dispatch: parallel gather dots (one lane per nonbasic column, bit-exact) + ordered
  // append of the nonzero columns. delta_z_indices is empty here (cleared each iter) and reserved
  // to n, so the collective raw-writes [0,count) and we set the size back on return.
  i_t dz_count = 0;
  warp_dz_body(lp.A.col_start.data(), lp.A.i.data(), lp.A.x.data(), &dz_count, direction,
               delta_y, delta_z_indices.data(), leaving_index, m, delta_z_mark, n,
               nonbasic_list, delta_z);
  // No broadcast: the body writes *dz_count on EVERY lane (the count is warp-uniform), and
  // &dz_count is this lane's own local. The body also ends with __syncwarp, so the global
  // writes it made are visible before set_size below.
  delta_z_indices.set_size(dz_count);
}

// Transpose delta_z: delta_z_N = -A_N' delta_y, row-wise on immutable Arow.
// Scan complete CSR rows and filter basic columns through the node-local basic_mark.
// Direct=true (warp): ALL 32 lanes execute this function, so the collective is
// called directly (no cfg_shared / op-code dispatch) and the lane-0-only output must be broadcast
// before the global-memory set_size. Direct=false (monolith): only lane 0 is here — a __shfl_sync
// would deadlock waiting for lanes parked in warp_helper_loop, so it must NOT be emitted.
template <class FT>
__device__ void compute_delta_z(const FT& ft,
                                const gpu_csr_matrix& Arow,
                                const gpu_sparse_vector& delta_y,
                                i_t leaving_index,
                                i_t direction,
                                const i_t* basic_mark,
                                i_t* delta_z_mark,
                                gpu_span<i_t>& delta_z_indices,
                                f_t* delta_z)
{
  // Collectively scatter (inner span per row split across the warp). delta_z_indices is empty here
  // (cleared each iter, reserved to n); the collective raw-writes [0,count), we set the size back.
  // Always taken on the warp path (proven ~1.12x overall, bit-exact — was CUOPT_GPU_DZSCAT, now default).
  i_t dz_count = 0;
  warp_dz_scatter_body(&dz_count, direction, delta_z_indices.data(), leaving_index, delta_z_mark,
                       delta_z, delta_y.i.data(), delta_y.x.data(), Arow.j.data(), basic_mark,
                       static_cast<i_t>(delta_y.i.size()), Arow.row_start.data(), Arow.x.data());
  // No broadcast: the body writes *dz_count on EVERY lane (the count is warp-uniform), and
  // &dz_count is this lane's own local. The body also ends with __syncwarp, so the global
  // writes it made are visible before set_size below.
  delta_z_indices.set_size(dz_count);
}

// `rhs` is CALLER-OWNED for the same reason as above: a stack-local would be per-lane under Direct.
template <class FT>
__device__ void compute_primal_variables(const FT& ft,
                                         const gpu_span<f_t>& lp_rhs,
                                         const gpu_csc_view& A,
                                         const i_t* basic_list,
                                         const i_t* nonbasic_list,
                                         f_t tight_tol,
                                         f_t* x,
                                         gpu_span<f_t>& xB_workspace,
                                         gpu_span<f_t>& rhs)
{
  const i_t m = A.m;
  const i_t n = A.n;
  const int lane = static_cast<int>(threadIdx.x);

  // reserve/set_size are a header RMW that may device-malloc -> lane 0. The lane0_serial __syncwarp
  // also publishes the (possibly relocated) data_ pointer before the collective copy reads it.
  //
  // NOT assign(m, 0.0): the copy below writes every slot in [0,m), so a zero-fill first is m dead
  // stores on the one serial lane. The "reused slot must be fully re-initialized" rule still holds —
  // the copy is what re-initializes it.
  lane0_serial([&] { rhs.reserve(m); rhs.set_size(m); });
  // rhs[k] = lp_rhs[k] over distinct k: no two lanes touch the same slot, and a copy has no
  // arithmetic to reassociate, so this is bit-exact against the serial loop.
  for (i_t k = lane; k < m; k += 32) { rhs[k] = lp_rhs[k]; }
  __syncwarp(0xffffffffu);

  // The nonbasic scatter stays SERIAL on purpose: two nonbasic columns can hit the same row, so a
  // lane-parallel form needs atomics, and atomic doubles reorder the accumulation into rhs[A.i[p]].
  lane0_serial([&] {
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

  ft.b_solve(rhs, xB_workspace);

  // basic_list holds the m DISTINCT basic variable indices, so the scattered writes to x[] land on
  // distinct slots — parallel and bit-exact, same as the copy above.
  for (i_t k = lane; k < m; k += 32) { x[basic_list[k]] = xB_workspace[k]; }
  __syncwarp(0xffffffffu);
}

__device__ void clear_delta_z(
  i_t entering_index, i_t leaving_index, i_t* delta_z_mark, gpu_span<i_t>& delta_z_indices, f_t* delta_z)
{
  const i_t nz = delta_z_indices.size();
  // delta_z_indices is a mark-guarded append list, so each column appears at most once -> the
  // destinations are distinct and the strided form is bit-exact.
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < nz; k += 32) {
    const i_t j     = delta_z_indices[k];
    delta_z[j]      = 0.0;
    delta_z_mark[j] = 0;
  }
  __syncwarp(0xffffffffu);
  // The two tail writes may duplicate an index already zeroed above, but both store 0.0, so the
  // overlap is idempotent and order-free. clear() is a header RMW -> exactly one lane.
  lane0_serial([&] {
    if (entering_index != -1) { delta_z[entering_index] = 0.0; }
    delta_z[leaving_index] = 0.0;
    delta_z_indices.clear();
  });
}

template <class FT>
__device__ f_t compute_initial_primal_infeasibilities(const gpu_lp& lp,
                                                      const gpu_settings& settings,
                                                      const FT& ft,
                                                      const i_t* basic_list,
                                                      const f_t* x,
                                                      gpu_span<f_t>& squared_infeasibilities,
                                                      gpu_span<i_t>& infeasibility_indices,
                                                      f_t& primal_inf)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  i_t pi_count = 0;
  f_t pi_pinf = 0.0, pi_pinf_sq = 0.0;
  // reserve/set_size are header RMW (+ possible device malloc) -> lane 0; the __syncwarp inside
  // lane0_serial also publishes the (possibly relocated) data_ pointers before the collective.
  // No fill is needed: warp_pinf_body writes every slot exposed through infeasibility_indices.
  lane0_serial([&] {
    squared_infeasibilities.reserve(n);
    squared_infeasibilities.set_size(n);
    infeasibility_indices.reserve(n);
  });
  warp_pinf_body(basic_list, &pi_count, infeasibility_indices.data(), lp.lower.data(), m,
                 &pi_pinf, &pi_pinf_sq, squared_infeasibilities.data(), settings.primal_tol,
                 lp.upper.data(), x);
  // No broadcast: the body writes all three scalars on EVERY lane (all three are warp-uniform),
  // and the pointers address this lane's own locals. The body ends with __syncwarp, so the
  // index/infeasibility arrays it filled are visible to the set_size and the pricing scan.
  lane0_serial([&] { infeasibility_indices.set_size(pi_count); });
  primal_inf = pi_pinf;
  return pi_pinf_sq;
}

// Under <Warp,Direct> this runs as a strided warp scan + butterfly reduction; otherwise it is the
// original serial loop.
//
// The warp form is BIT-IDENTICAL to the serial one, not merely equivalent. The serial loop is a
// maximum over the pairs (val, j) under a total order -- val larger wins, j larger breaks ties --
// with (0.0, -1) as the bottom element (val = squared_infeas / dy_se >= 0 always, and every j >= 0).
// A maximum over a total order is associative and commutative, so partitioning the index set across
// lanes and folding with the SAME comparator returns the same winner regardless of visit order. No
// floating-point arithmetic is reassociated: `val` is computed from the same two loads per element
// either way, and the fold only compares. Hence the pivot sequence must not move -- if it does, this
// is a bug, not an algorithm change.
//
// This was previously left serial on the grounds that warping it would change the tie-break. It does
// not: the `j > leaving_index` tie-break is what makes the reduction order-independent.
__device__ i_t steepest_edge_pricing_with_infeasibilities(const gpu_lp& lp,
                                                          const gpu_settings& settings,
                                                          const f_t* x,
                                                          const f_t* dy_steepest_edge,
                                                          const i_t* basic_mark,
                                                          gpu_span<f_t>& squared_infeasibilities,
                                                          gpu_span<i_t>& infeasibility_indices,
                                                          i_t& direction,
                                                          i_t& basic_leaving,
                                                          f_t& max_val)
{
  const i_t nz      = infeasibility_indices.size();
  i_t leaving_index = -1;
  // Every lane already executes this function (it is not inside lane0_serial), so the serial form
  // did the identical nz-length scan 32 times over. Split it instead.
  const i_t* __restrict__ idx = infeasibility_indices.data();
  const f_t* __restrict__ sq  = squared_infeasibilities.data();
  const i_t lane              = static_cast<i_t>(threadIdx.x);
  f_t best_val                = 0.0;
  i_t best_j                  = -1;
  for (i_t k = lane; k < nz; k += 32) {
    const i_t j   = idx[k];
    const f_t val = sq[j] / dy_steepest_edge[j];
    if (val > best_val || (val == best_val && j > best_j)) { best_val = val; best_j = j; }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const f_t oval = __shfl_xor_sync(0xffffffffu, best_val, off);
    const i_t oj   = __shfl_xor_sync(0xffffffffu, best_j, off);
    if (oval > best_val || (oval == best_val && oj > best_j)) { best_val = oval; best_j = oj; }
  }
  // Butterfly => every lane holds the winner, so no broadcast is needed and the caller's
  // warp-uniform reads of leaving/direction/max_val stay warp-uniform.
  max_val       = best_val;
  leaving_index = best_j;
  if (leaving_index >= 0) {
    const f_t lower_infeas = lp.lower[leaving_index] - x[leaving_index];
    const f_t upper_infeas = x[leaving_index] - lp.upper[leaving_index];
    direction              = lower_infeas >= upper_infeas ? 1 : -1;
  }
  basic_leaving = leaving_index >= 0 ? basic_mark[leaving_index] : -1;
  return leaving_index;
}

template <class FT>
__device__ void compute_delta_y(const FT& ft,
                                i_t basic_leaving_index,
                                i_t direction,
                                gpu_sparse_vector& delta_y_sparse,
                                gpu_sparse_vector& UTsol_sparse)
{
  const i_t m = delta_y_sparse.n;
  gpu_sparse_vector& ei_sparse = ft.ei_scratch_;  // reused (was a per-pivot device malloc)
  lane0_serial([&] {
    ei_sparse.n = m;
    ei_sparse.i.resize(1);   // may device-malloc on the first pivot -> lane 0 only
    ei_sparse.x.resize(1);
    ei_sparse.i[0] = basic_leaving_index;
    ei_sparse.x[0] = -direction;
  });
  ft.b_transpose_solve(ei_sparse, delta_y_sparse, UTsol_sparse);

  // `direction` is warp-uniform (read from st.direction), so the branch is uniform and the
  // __syncwarp count inside warp_negate matches across the warp.
  if (direction != -1) { UTsol_sparse.warp_negate(); }
}

template <class FT>
__device__ void update_dual_variables(const FT& ft,
                                     const gpu_sparse_vector& delta_y_sparse,
                                     const gpu_span<i_t>& delta_z_indices,
                                     const f_t* delta_z,
                                     f_t step_length,
                                     i_t leaving_index,
                                     f_t* y,
                                     f_t* z)
{
  const i_t delta_y_nz = delta_y_sparse.i.size();
  // warp_collective_upddual performs the FULL y/z update (both loops + the leaving term below), so
  // return here — mirroring the dispatch;return idiom of compute_delta_z / phase2_ratio_test / pinf.
  // Without this return the serial loops below would double-update y and z (2×step) on the warp path.
  // All lanes are here; the body is internally lane-guarded (distinct-location writes
  // plus a lane-0 guard on z[leaving]) so no broadcast is needed.
  warp_upddual_body(delta_y_sparse.i.data(), static_cast<i_t>(delta_y_sparse.i.size()),
                    delta_y_sparse.x.data(), delta_z, delta_z_indices.data(),
                    static_cast<i_t>(delta_z_indices.size()), leaving_index, step_length, y, z);
}

template <class FT>
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
  // b_solve2 already produced scaled_delta_xB_sparse and its saved post-L values in
  // utilde_sparse alongside the steepest-edge RHS.
  // warp_negate's trailing __syncwarp is what publishes the negated values to the other lanes
  // before find_coefficient below reads them on all 32.
  scaled_delta_xB_sparse.warp_negate();

  // find_coefficient is read-only, so every lane may compute `scale` itself — no guard needed.
  f_t scale     = scaled_delta_xB_sparse.find_coefficient(basic_leaving_index);
  i_t numerical = 0;
  if (isnan(scale)) {
    ft.path(P_NAN_RECOVERY);
    // We could not find a coefficient for the basic leaving index; it may be very small.
    // Switch to a regular dense solve and try to recover -- mirrors the CPU compute_delta_x.
    //
    // Fully collective. ft.recover_rhs_ / ft.recover_sol_ are members, so all 32 lanes name the
    // SAME buffer; stack-local gpu_spans would give each lane a private malloc and the warp
    // would write 32 different ones. They are lazily allocated, so this costs nothing until the
    // branch first fires.
    const i_t m = lp.num_rows;
    ft.warp_to_dense(rhs_sparse, ft.recover_rhs_);
    lane0_serial([&] { ft.recover_sol_.reserve(m); ft.recover_sol_.set_size(m); });
    __syncwarp(0xffffffffu);          // publish the (possibly relocated) recover_sol_ data pointer
    ft.b_solve(ft.recover_rhs_, ft.recover_sol_);

    const f_t recovered = ft.recover_sol_[basic_leaving_index];   // uniform read, every lane
    if (recovered != 0.0 && !isnan(recovered)) {
      ft.warp_from_dense(ft.recover_sol_, scaled_delta_xB_sparse);
      scaled_delta_xB_sparse.warp_negate();
      scale = -recovered;
    } else if (delta_z[entering_index] != 0.0) {
      scale = -delta_z[entering_index];
      // Warp search for basic_leaving_index. Sparse indices are unique so at most one lane can
      // match; the reduction still takes the MINIMUM k, which reproduces the serial `break`
      // (= first match) even if that invariant were ever broken. Costs no registers over the
      // lane-0 scan it replaces (measured: 129 uncapped either way).
      // Only the tail stays on lane 0: push_back is a header RMW, and the single x[k] write is
      // one element either way.
      // `nz` is the not-found sentinel: every valid k is < nz, so it cannot collide with a real
      // match, and the min-reduction leaves nz behind exactly when no lane matched.
      const i_t nz   = static_cast<i_t>(scaled_delta_xB_sparse.i.size());
      const int lane = static_cast<int>(threadIdx.x);
      i_t found_k    = nz;
      for (i_t k = lane; k < nz; k += 32) {
        if (scaled_delta_xB_sparse.i[k] == basic_leaving_index) { found_k = min(found_k, k); }
      }
      for (int off = 16; off; off >>= 1) {
        found_k = min(found_k, __shfl_xor_sync(0xffffffffu, found_k, off));
      }
      if (found_k < nz) {
        lane0_serial([&] { scaled_delta_xB_sparse.x[found_k] = scale; });
      } else {
        lane0_serial([&] {
          scaled_delta_xB_sparse.i.push_back(basic_leaving_index);
          scaled_delta_xB_sparse.x.push_back(scale);
        });
      }
    } else {
      numerical = 1;   // uniform on every lane: `recovered` and delta_z are uniform reads
    }
    if (numerical) { return -1; }
  }
  // Sparse indices are distinct, so distribute the independent basic-variable updates. The
  // barrier preserves the original ordering when a special-case index also appears in the loop.
  const f_t primal_step_length = delta_x_leaving / scale;
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  const int lane                = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < scaled_delta_xB_nz; k += 32) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    delta_x[j]  = primal_step_length * scaled_delta_xB_sparse.x[k];
  }
  __syncwarp(0xffffffffu);
  if (lane == 0) {
    delta_x[leaving_index]  = delta_x_leaving;
    delta_x[entering_index] = primal_step_length;
  }
  __syncwarp(0xffffffffu);
  return 0;
}

template <class FT>
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
  // b_solve2 leaves the second solution dense when its U route is dense; otherwise v_sparse holds
  // the ordinary sparse result. Applying `direction` inside warp_seupd_body avoids mutating either
  // representation and preserves the original arithmetic for every updated edge norm.
  const bool v_is_dense =
    ft.dense_scratch2_.size() == static_cast<size_t>(delta_y_sparse.n);
  ft.path(v_is_dense ? P_V_DENSE : P_V_SCATTER);
  if (v_is_dense) {
    v = ft.dense_scratch2_.data();
  } else {
    v_sparse.warp_scatter(v);
  }

  // find_coefficient is read-only: every lane computes the same wr/omegar, so `return -1` is taken
  // uniformly and no broadcast is needed.
  const f_t wr = -scaled_delta_xB.find_coefficient(basic_leaving_index);
  if (wr == 0) { return -1; }
  const f_t omegar             = dy_norm_squared / (wr * wr);
  const i_t scaled_delta_xB_nz = scaled_delta_xB.i.size();

  // One lane per nonzero: distinct-location writes (see warp_collective_seupd) → bit-exact.
  warp_seupd_body(basic_list, dy_norm_squared, delta_y_steepest_edge, entering_index,
                  basic_leaving_index, scaled_delta_xB_nz, omegar, scaled_delta_xB.i.data(),
                  scaled_delta_xB.x.data(), v, wr, direction);

  // The dense result lives in factor scratch and is overwritten by the next paired solve. Restore
  // only the sparse fallback's scatter target to zero for the next pivot.
  if (!v_is_dense) {
    const i_t v_nz = v_sparse.i.size();
    const int lane = static_cast<int>(threadIdx.x);
    for (i_t k = lane; k < v_nz; k += 32) { v[v_sparse.i[k]] = 0.0; }
    __syncwarp(0xffffffffu);
  }
  return 0;
}

// `x[j] += delta_x[j]` is a read-modify-write, but the destinations are DISTINCT: scaled_delta_xB's
// indices are unique, so basic_list[.i[k]] is unique too. Hence the strided form is bit-exact.
__device__ void update_primal_variables(const gpu_sparse_vector& scaled_delta_xB_sparse,
                                        const i_t* basic_list,
                                        const f_t* delta_x,
                                        i_t entering_index,
                                        f_t* x)
{
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t k = lane; k < scaled_delta_xB_nz; k += 32) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    x[j] += delta_x[j];
  }
  // The entering column is NOT yet in basic_list (K5 does the swap), so this slot is disjoint from
  // every j above — but exactly one lane must own it.
  if (threadIdx.x == 0) { x[entering_index] += delta_x[entering_index]; }
  __syncwarp(0xffffffffu);
}

// Every column j is independent: the branch chain reads only column-j data and writes only x[j] and
// vstatus[j], so no slot is touched twice and the strided form is bit-exact. The per-column body is
// a lambda so the two loop shapes share one copy of the chain.
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
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t j = lane; j < n; j += 32) { one_column(j); }
  __syncwarp(0xffffffffu);
}

// ---------------------------------------------------------------------------
// The device driver: a line-for-line copy of simplified_dual_phase2's
// pure-MPF-stacking path.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// slot_state : the PERSISTENT per-slot state of one solve.
//
// Every member here used to be a local in gpu_simplified_dual_phase2_device's stack frame, so it
// died when the driver returned and was published nowhere. The per-pivot state therefore
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
// setup() is the one-time S1-S7 prologue, kept as its own method so a persistent block can re-run
// it in place for a freshly refilled slot.
// ---------------------------------------------------------------------------
// ---- Per-slot buffers -----------------------------------------------------------------------
// Every buffer the kernel uses is its own allocation, exactly like every other array the marshal
// uploads. No slab and no offsets: the pointer IS the buffer.
//
// This replaced a single slab that the host sized and the device re-walked to compute offsets, an
// arrangement that required the two walks to agree in order AND in alignment -- and when they did
// not, the device ran off the end of the slot. Here a miscount can only make the VRAM pre-flight
// estimate slightly wrong, which is an estimate, not memory corruption.
struct slot_buffers {
  f_t* xB_workspace;
  f_t* delta_y;
  f_t* delta_z;
  f_t* delta_x;
  f_t* v;
  f_t* squared_infeas;
  f_t* setup_rhs;
  i_t* basic_mark;
  i_t* delta_z_mark;
  i_t* delta_z_indices;
  i_t* nonbasic_pos;
  i_t* infeas_idx;
  f_t* ft_dense_scratch;
  f_t* ft_dense_scratch2;
  f_t* ft_sort_bucket;
  f_t* ft_sort_x;
  i_t* ft_sort_i;
  i_t* ft_sort_perm;
  i_t* lvl_lev0;
  i_t* lvl_ls0;
  i_t* lvl_co0;
  i_t* lvl_lev1;
  i_t* lvl_ls1;
  i_t* lvl_co1;
  i_t* lvl_lev2;
  i_t* lvl_ls2;
  i_t* lvl_co2;
  i_t* lvl_lev3;
  i_t* lvl_ls3;
  i_t* lvl_co3;
  i_t* sp_ls_v;
  i_t* sp_co_v;
  i_t* sp_pos_v;
  i_t* lvl_pos_v;
  f_t* ft_recover_rhs;
  f_t* ft_recover_sol;
  i_t* S_col_start;
  i_t* S_i;
  f_t* S_x;
  f_t* mu_values;
  i_t* delta_y_sparse_i; f_t* delta_y_sparse_x;
  i_t* UTsol_sparse_i; f_t* UTsol_sparse_x;
  i_t* utilde_sparse_i; f_t* utilde_sparse_x;
  i_t* scaled_dxB_sparse_i; f_t* scaled_dxB_sparse_x;
  i_t* rhs_sparse_i; f_t* rhs_sparse_x;
  i_t* v_sparse_i; f_t* v_sparse_x;
  i_t* se_dy_i; f_t* se_dy_x;
  i_t* se_UTsol_i; f_t* se_UTsol_x;
  i_t* se_ei_i; f_t* se_ei_x;
  i_t* ft_sparse_scratch_i; f_t* ft_sparse_scratch_x;
  i_t* ft_ei_scratch_i; f_t* ft_ei_scratch_x;
};

// Pivot-loop phases timed per solve on lane 0 (clock64), reported with the seat clock.
constexpr int kSolvePhases = 11;
template <class FT>
struct slot_state {
  // Stored as i_t so the host can read it back as a plain int array.
  i_t status{static_cast<i_t>(dual::status_t::UNSET)};
  long long ph[kSolvePhases]{}, ph_t{0};
  long long pre[6]{};  // leg set-up cycles: bounds, strengthen, transposes, slot init, levels, prologue
  // Level schedules of this seat's base factor are built (lvl_* spans below hold them); a child
  // continuation keeps the factor, so it re-wires the block's config instead of rebuilding.
  bool cfg_built{false};
  i_t cfg_nlev[4]{};
  // A FRESH seat factorized on the device has no host-priced duals: setup prices them.
  bool price_fresh{false};

  FT ft;  // basis update: S_/mu_values_, num_updates_, routing stats, reused scratch

  // Dense per-iteration workspaces.
  gpu_span<f_t> xB_workspace, delta_y, delta_z, delta_x, v, squared_infeas;
  gpu_span<i_t> basic_mark, delta_z_mark, delta_z_indices, nonbasic_pos, infeas_idx;

  // Sparse per-iteration workspaces.
  gpu_sparse_vector delta_y_sparse, UTsol_sparse, utilde_sparse, scaled_dxB_sparse, rhs_sparse,
    v_sparse;

  // Immutable CSR row copy of A; basic membership is filtered through basic_mark at use time.
  gpu_csr_matrix Arow;

  // In the monolith these are gpu_phase2_kernel STACK LOCALS (lvl_* + a __shared__ cfg), so they die
  typename FT::warp_config_type wcfg;
  gpu_span<i_t>      lvl_lev[kNumWarpSolveOps];   // per-column level (drives the sparse reach solve)
  gpu_span<i_t>      lvl_ls[kNumWarpSolveOps];    // level offsets
  gpu_span<i_t>      lvl_co[kNumWarpSolveOps];    // columns in level order
  gpu_span<i_t>      sp_ls_v, sp_co_v, sp_pos_v;  // reach-restricted sparse-solve schedule scratch
  gpu_span<i_t>      lvl_pos_v;                  // warp_build_levels counting-sort cursor (m+1)

  // registers' worth of pointers/spans/doubles live for the kernel's whole duration before any work
  // — the suspected floor under the measured 96-120. Living in the slot they are read lazily
  // (L1/L2-cached), so the compiler loads only the fields a given kernel touches and still hoists
  // the hot ones (e.g. A.col_start inside a solve loop) back into registers.
  gpu_lp       lp;
  gpu_settings settings;
  gpu_solution sol;

  // SETUP-ONLY scratch, slot-resident because under Direct a stack-local gpu_span allocates PER
  // LANE. The sparse vectors are also reused to price child duals.
  gpu_span<f_t> setup_rhs;  // compute_primal_variables
  gpu_sparse_vector se_dy, se_UTsol, se_ei;


  // Wire every span to its own buffer. No offsets to get wrong.
  // TODO: find a more efficient way to do this
  __device__ void attach_all_buffers(const slot_buffers& b, i_t m, i_t n, i_t ucap, i_t eta)
  {
    const size_t mn = (m > n) ? (size_t)m : (size_t)n;
    const size_t m1 = (size_t)m + 1;
    xB_workspace.attach(b.xB_workspace, mn);
    delta_y.attach(b.delta_y, mn);
    delta_z.attach(b.delta_z, mn);
    delta_x.attach(b.delta_x, mn);
    v.attach(b.v, mn);
    squared_infeas.attach(b.squared_infeas, mn);
    setup_rhs.attach(b.setup_rhs, mn);
    basic_mark.attach(b.basic_mark, mn);
    delta_z_mark.attach(b.delta_z_mark, mn);
    delta_z_indices.attach(b.delta_z_indices, mn);
    nonbasic_pos.attach(b.nonbasic_pos, mn);
    infeas_idx.attach(b.infeas_idx, mn);
    ft.dense_scratch_.attach(b.ft_dense_scratch, mn);
    ft.dense_scratch2_.attach(b.ft_dense_scratch2, mn);
    ft.sort_bucket_.attach(b.ft_sort_bucket, mn);
    ft.sort_x_.attach(b.ft_sort_x, mn);
    ft.sort_i_.attach(b.ft_sort_i, mn);
    ft.sort_perm_.attach(b.ft_sort_perm, mn);
    lvl_lev[0].attach(b.lvl_lev0, m1);
    lvl_ls[0].attach(b.lvl_ls0, m1);
    lvl_co[0].attach(b.lvl_co0, m1);
    lvl_lev[1].attach(b.lvl_lev1, m1);
    lvl_ls[1].attach(b.lvl_ls1, m1);
    lvl_co[1].attach(b.lvl_co1, m1);
    lvl_lev[2].attach(b.lvl_lev2, m1);
    lvl_ls[2].attach(b.lvl_ls2, m1);
    lvl_co[2].attach(b.lvl_co2, m1);
    lvl_lev[3].attach(b.lvl_lev3, m1);
    lvl_ls[3].attach(b.lvl_ls3, m1);
    lvl_co[3].attach(b.lvl_co3, m1);
    sp_ls_v.attach(b.sp_ls_v, m1);
    sp_co_v.attach(b.sp_co_v, m1);
    sp_pos_v.attach(b.sp_pos_v, m1);
    lvl_pos_v.attach(b.lvl_pos_v, m1);
    ft.recover_rhs_.attach(b.ft_recover_rhs, m);
    ft.recover_sol_.attach(b.ft_recover_sol, m);
    ft.S_.col_start.attach(b.S_col_start, 2 * ucap + 1);
    ft.S_.i.attach(b.S_i, eta);
    ft.S_.x.attach(b.S_x, eta);
    ft.mu_values_.attach(b.mu_values, ucap);
    delta_y_sparse.attach(b.delta_y_sparse_i, b.delta_y_sparse_x, mn);
    UTsol_sparse.attach(b.UTsol_sparse_i, b.UTsol_sparse_x, mn);
    utilde_sparse.attach(b.utilde_sparse_i, b.utilde_sparse_x, mn);
    scaled_dxB_sparse.attach(b.scaled_dxB_sparse_i, b.scaled_dxB_sparse_x, mn);
    rhs_sparse.attach(b.rhs_sparse_i, b.rhs_sparse_x, mn);
    v_sparse.attach(b.v_sparse_i, b.v_sparse_x, mn);
    se_dy.attach(b.se_dy_i, b.se_dy_x, mn);
    se_UTsol.attach(b.se_UTsol_i, b.se_UTsol_x, mn);
    se_ei.attach(b.se_ei_i, b.se_ei_x, mn);
    ft.sparse_scratch_.attach(b.ft_sparse_scratch_i, b.ft_sparse_scratch_x, mn);
    ft.ei_scratch_.attach(b.ft_ei_scratch_i, b.ft_ei_scratch_x, mn);
  }

  // Loop-carried scalars.
  f_t obj{0.0}, primal_inf{0.0};
  i_t iter{0};

  // Per-pivot scalars. Locals today (the monolith keeps them in registers); they become part of
  // the cross-kernel interface once the pipeline is split, so they live here from the start.
  i_t leaving{-1}, entering{-1}, direction{0}, basic_leaving{-1}, nonbasic_entering{-1};
  f_t step_length{0.0}, dy_norm_sq{0.0};
  bool used_transpose_dz{false};
  i_t f_state{0};  // fissioned pivot loop (fission.cuh): next phase
  i_t f_lean{0};   // fission fresh seat: x_B, objective and infeasibilities come from fission_xb
  i_t f_mrows{-1}; // rows of the fission replay matrix M held (== num_updates_ when usable)
  i_t f_mrow_k{-1}; // row of M written by the last FTRAN (from UTsol), pending the update
  f_t f_pivot{0.0};    // FTRAN: (B^-1 a_q)[basic_leaving], 0 when absent
  i_t f_pivot_found{0};

  // Price a plunge child from the LIVE factorization. A fresh GPU node is priced on the host and
  // uploads y/z along with its factorization. CHILD_CONTINUE deliberately avoids that host round
  // trip and keeps the parent's resident factorization, so it must reproduce the CPU node-entry
  // pricing here: B' y = c_B followed by z_N = c_N - N' y. This runs even though the child inherits
  // the parent's basis because the CPU advanced driver reprices every node when
  // initialize_basis=false too, before assigning nonbasics to the new bounds.
  __device__ __noinline__ void price_child_duals(const gpu_lp& lp,
                                                 const i_t* basic_list,
                                                 const i_t* nonbasic_list,
                                                 gpu_solution& sol)
  {
    const i_t m = lp.num_rows;
    const i_t n = lp.num_cols;
    lane0_serial([&] {
      se_ei.n = m;
      se_ei.clear();
      se_dy.n = m;
      se_dy.clear();
      se_UTsol.n = m;
      se_UTsol.clear();
      for (i_t k = 0; k < m; ++k) {
        const f_t c = lp.objective[basic_list[k]];
        if (c != 0.0) {
          se_ei.i.push_back(k);
          se_ei.x.push_back(c);
        }
      }
    });
    // Avoid feeding an empty sparse RHS to the routing-density estimator (its historical formula
    // divides by the input nnz). An all-zero c_B simply means y=0.
    if (!se_ei.i.empty()) { ft.b_transpose_solve(se_ei, se_dy, se_UTsol); }

    f_t* y = sol.y.data();
    f_t* z = sol.z.data();
    const int lane = static_cast<int>(threadIdx.x);
    for (i_t i = lane; i < m; i += 32) { y[i] = 0.0; }
    __syncwarp(0xffffffffu);
    // Sparse solve indices are unique, so the scatter has no write conflicts.
    for (i_t p = lane; p < static_cast<i_t>(se_dy.i.size()); p += 32) {
      y[se_dy.i[p]] = se_dy.x[p];
    }
    __syncwarp(0xffffffffu);
    for (i_t k = lane; k < n - m; k += 32) {
      const i_t j = nonbasic_list[k];
      f_t dot      = 0.0;
      for (i_t p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
        dot += lp.A.x[p] * y[lp.A.i[p]];
      }
      z[j] = lp.objective[j] - dot;
    }
    for (i_t k = lane; k < m; k += 32) { z[basic_list[k]] = 0.0; }
    __syncwarp(0xffffffffu);
  }

  // One-time prologue: bounds -> primal -> SE validation -> workspaces -> Arow -> nonbasic
  // partition -> primal infeasibilities. Byte-for-byte the setup the monolithic
  // driver used to run inline, in the same order (bit-exactness depends on that order).
  // Direct=true: the prologue is almost entirely allocation and
  // read-modify-write (assign/reserve/clear, sparse-vector construction, Arow build) so it stays on
  // lane 0; the collectives are opted in one at a time. The serial parts deliberately call the
  // <false> variants so porting one collective cannot silently move any other computation.
  template <bool PersistentSeats = false>
  __device__ __noinline__ void setup(const gpu_lp& lp,
                                     const gpu_settings& settings,
                                     variable_status_t* vstatus,
                                     i_t* basic_list,
                                     i_t* nonbasic_list,
                                     gpu_solution& sol,
                                     f_t* delta_y_steepest_edge,
                                     bool refactor_resume,
                                     bool child_continue,
                                     f_t resume_obj,
                                     i_t resume_iter)
  {
    const i_t m = lp.num_rows;
    const i_t n = lp.num_cols;
    f_t* x      = sol.x.data();
    f_t* z      = sol.z.data();

    if constexpr (PersistentSeats) {
      if (child_continue) {
        lane0_serial([&] { status = static_cast<i_t>(dual::status_t::UNSET); });
      }
    }

    // assign (not resize) throughout: a slot is REUSED by the next B&B node, so every buffer must be
    // fully re-initialized, not just grown — resize would leave the previous node's tail values.
    // assign() may device-malloc, so it stays on lane 0; the O(n) bound-setting scan next to it does
    // not, so the two are split rather than sharing one lane-0 region.
    lane0_serial([&] { xB_workspace.assign(m, 0.0); });

    // On a REFACTOR_NEEDED resume the host has already rebuilt x with
    // phase2::compute_primal_solution_from_basis. Redoing it here would use a different formula
    // -- these two set the nonbasic values from the sign of z and skip columns with
    // |x_j| < 10*tight_tol -- and that difference moves the trajectory from the first refactor on.
    if constexpr (PersistentSeats) {
      if (child_continue || price_fresh) { price_child_duals(lp, basic_list, nonbasic_list, sol); }
    }

    if (!refactor_resume) {
      set_primal_variables_on_bounds(lp, settings, z, vstatus, x);
    }
    if (!refactor_resume && f_lean == 0) {
      // Both of these solve through the factors, so they follow the kernel's geometry — the
      // monolith has always run them on the warp via Direct.
      compute_primal_variables(ft,
                               lp.rhs,
                               lp.A,
                               basic_list,
                               nonbasic_list,
                               settings.tight_tol,
                               x,
                               xB_workspace,
                               setup_rhs);
    }

    if (!refactor_resume && (!PersistentSeats || !child_continue)) {
      // Repair for a WARM START only: norms inherited from a parent B&B node can carry entries that
      // are stale or non-positive for this node's basis. A RESUME is not a warm start -- these are
      // the device's own maintained norms from the previous leg, and the CPU's in-place
      // refactorization does not touch them, so neither may this.
      // basic_list holds m DISTINCT column indices, so each delta_y_steepest_edge[j] is visited by
      // exactly one lane — bit-exact.
      const int lane = static_cast<int>(threadIdx.x);
      for (i_t k = lane; k < m; k += 32) {
        const i_t j = basic_list[k];
        if (delta_y_steepest_edge[j] <= 0.0) { delta_y_steepest_edge[j] = 1e-4; }
      }
      __syncwarp(0xffffffffu);
    }

    lane0_serial([&] {

    delta_y.assign(m, 0.0);
    delta_z.assign(n, 0.0);
    delta_x.assign(n, 0.0);
    v.assign(m, 0.0);
    basic_mark.assign(n, -1);
    nonbasic_pos.assign(n, -1);
    });

    // basic_list holds m DISTINCT column indices, so no two lanes write the same slot — bit-exact.
    // data() is read AFTER the lane-0 region above because assign() may have relocated the buffer.
    {
      i_t* bm = basic_mark.data();
      i_t* np = nonbasic_pos.data();
      const int lane = static_cast<int>(threadIdx.x);
      for (i_t k = lane; k < m; k += 32) { bm[basic_list[k]] = k; }
      for (i_t k = lane; k < n - m; k += 32) { np[nonbasic_list[k]] = k; }
      __syncwarp(0xffffffffu);
    }

    lane0_serial([&] {
    delta_z_mark.assign(n, 0);
    delta_z_indices.clear();
    delta_z_indices.reserve(n);

    // Immutable host-built CSR view. All nodes sharing the original A share these arrays;
    // node-local basic_mark supplies the changing basis membership.
    Arow.view_device(lp.Arow_row_start, lp.Arow_j, lp.Arow_x, m, n, lp.A.col_start[n]);
    });  // end lane-0 prologue

    lane0_serial([&] {
      squared_infeas.clear();
      infeas_idx.clear();
      primal_inf = 0.0;
    });
    // WARP_PINF: scan the m basics one lane per k and compact the infeasible indices in order.
    // squared_infeas is read only through that freshly written index list.
    if (f_lean == 0) {
      compute_initial_primal_infeasibilities(
        lp, settings, ft, basic_list, x, squared_infeas, infeas_idx, primal_inf);
    }

    lane0_serial([&] {
    // Set the dimension and empty them IN PLACE. These used to be whole-object assignments
    // (`= gpu_sparse_vector(m, 0)`), which move-assigned a fresh cap_=0 temporary over the buffer
    // and so DISCARDED its preallocated storage -- the vector then grew again from zero by
    // push_back doubling, and `sort()` swapped the empty buffer into ft.sort_i_/sort_x_ as well.
    // Anything that replaces a whole buffer object undoes its attachment; only clear() is safe.
    delta_y_sparse.n    = m; delta_y_sparse.clear();
    UTsol_sparse.n      = m; UTsol_sparse.clear();     // MPF-only: Uᵀ from compute_delta_y
    utilde_sparse.n     = m; utilde_sparse.clear();    // MPF-only: b_solve from compute_delta_x
    scaled_dxB_sparse.n = m; scaled_dxB_sparse.clear();
    rhs_sparse.n        = m; rhs_sparse.clear();
    v_sparse.n          = m; v_sparse.clear();

    // Incrementally-tracked objective for the cutoff, mirroring the CPU simplified (and
    // advanced) driver: init fresh once, then update_objective per pivot. Using the true
    // costs lp.objective (the GPU has no perturbation) is a valid dual bound.
    // CONTINUE, do not restart. The CPU carries both across an in-place refactorization; only a
    // fresh solve derives them. Recomputing obj from x also silently changes the cutoff comparand
    // mid-solve, and resetting iter makes the driver's own iteration limit unreachable.
    obj  = refactor_resume ? resume_obj : (f_lean != 0 ? f_t{0.0} : compute_objective(lp, x));
    iter = refactor_resume ? resume_iter : 0;
    });  // end lane-0 epilogue
  }
};

__device__ bool dual_feasible_within_tol(const gpu_lp& lp,
                                         const variable_status_t* vstatus,
                                         const f_t* z,
                                         f_t dual_tol)
{
  const i_t n    = lp.num_cols;
  const int lane = static_cast<int>(threadIdx.x);
  int bad        = 0;
  for (i_t j = lane; j < n; j += 32) {
    const variable_status_t s = vstatus[j];
    if (s != variable_status_t::BASIC && !isfinite(z[j])) {
      bad = 1;
      continue;
    }
    if (s == variable_status_t::NONBASIC_LOWER) {
      if (z[j] < -dual_tol) { bad = 1; }
    } else if (s == variable_status_t::NONBASIC_UPPER) {
      if (z[j] > dual_tol) { bad = 1; }
    } else if (s == variable_status_t::NONBASIC_FREE) {
      if (fabs(z[j]) > dual_tol) { bad = 1; }
    }
  }
  return __any_sync(0xffffffffu, bad != 0) == 0;
}


// `Direct`: all 32 lanes execute the body. Direct=false is the historical single-thread monolith
// (lane 0 alone runs this body); Direct=true means ALL 32 lanes run it, so every allocation and
// read-modify-write must be inside lane0_serial and every `return` must be warp-uniform.
template <class FT, bool AllowRefactorExit, bool PersistentSeats = false, bool FissionSetup = false>

__device__ dual::status_t gpu_simplified_dual_phase2_device(const gpu_lp& lp,
                                                            const gpu_settings& settings,
                                                            variable_status_t* vstatus,
                                                            slot_state<FT>& st,
                                                            i_t* basic_list,
                                                            i_t* nonbasic_list,
                                                            gpu_solution& sol,
                                                            f_t* delta_y_steepest_edge,
                                                            bool refactor_resume,
                                                            bool child_continue,
                                                            bool terminal_certificate_retry,
                                                            f_t resume_obj,
                                                            i_t resume_iter,
                                                            i_t& refactor_reason)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  // Why REFACTOR_NEEDED came back, for the host resume loop. 0 = periodic schedule (the update was
  // skipped, the representation is merely long); 1 = the rank-1 update was REJECTED as unsafe
  // (|mu| outside [1e-8,1e8]).
  // Only meaningful when the returned status is REFACTOR_NEEDED.
  // The host needs the distinction because case 1 follows the established rejected-update
  // recovery trajectory: refactorize, but carry the maintained y/z without using fresh pricing as
  // a rejection certificate. Periodic resumes can be certified independently.
  refactor_reason = 0;

  f_t* x = sol.x.data();
  f_t* y = sol.y.data();
  f_t* z = sol.z.data();

  // Prologue into the slot. Unconditional: one launch is one solve, so this always runs. It used to
  // be guarded by st.inited so a re-entering fission super-step could skip it and reuse the live
  // state; there are no super-steps any more.
  const long long ps0 = clock64();
  st.template setup<PersistentSeats>(lp,
           settings,
           vstatus,
           basic_list,
           nonbasic_list,
           sol,
           delta_y_steepest_edge,
           refactor_resume,
           child_continue,
           resume_obj,
           resume_iter);
  if (threadIdx.x == 0) { st.pre[5] = clock64() - ps0; }
  // Fissioned path: setup only; the phase kernels run the loop from slot_state.
  if constexpr (FissionSetup) {
    lane0_serial([&] { st.f_state = 0; });
    return dual::status_t::UNSET;
  }

  // Bind the persistent slot members under their original names, so the pivot loop below is
  // unchanged by the state externalization and stays line-diffable against the CPU simplified
  // driver (dual_simplex/CLAUDE.md rule 2).
  FT& ft                          = st.ft;
  gpu_span<f_t>& delta_y        = st.delta_y;
  gpu_span<f_t>& delta_z        = st.delta_z;
  gpu_span<f_t>& delta_x        = st.delta_x;
  gpu_span<f_t>& v              = st.v;
  gpu_span<f_t>& squared_infeas = st.squared_infeas;
  gpu_span<i_t>& basic_mark     = st.basic_mark;
  gpu_span<i_t>& delta_z_mark   = st.delta_z_mark;
  gpu_span<i_t>& delta_z_indices = st.delta_z_indices;
  gpu_span<i_t>& nonbasic_pos   = st.nonbasic_pos;
  gpu_span<i_t>& infeas_idx     = st.infeas_idx;
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
    lane0_serial([&] {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iters;

    });
  };

  // A refactor exit is a SUSPENSION, not a terminal answer.  The next leg must resume from the
  // incrementally maintained cutoff objective, just as the CPU driver does across its in-place
  // refactorization.  `finish()` deliberately recomputes c'x for a user-facing terminal result;
  // using it here silently replaced the carried objective at every refactor boundary.
  auto suspend = [&](i_t iters) {
    lane0_serial([&] {
      sol.objective      = obj;
      sol.user_objective = compute_user_objective(lp, obj);
      sol.iterations     = iters;
    });
  };

  // NO per-node iteration limit, matching the CPU plunge path, which calls solve_node_lp with its
  // INT_MAX default. A size-derived cap on the device only is a DIFFERENT stopping rule for the same
  // node, so a node could return ITERATION_LIMIT from the GPU where the CPU would have solved it and
  // the tree would depend on which driver took the node. The solve still terminates on the host time
  // limit and on the MPF store cap, so this is not an unbounded loop.
  const i_t iter_limit = cuda::std::numeric_limits<i_t>::max();
  if (threadIdx.x == 0) {
    for (int k = 0; k < kSolvePhases; ++k) { st.ph[k] = 0; }
    st.ph_t = clock64();
  }
#define GPU_PH(k)                                                           \
  if (threadIdx.x == 0) {                                                   \
    const long long c_ = clock64();                                         \
    st.ph[k] += c_ - st.ph_t;                                               \
    st.ph_t = c_;                                                           \
  }
  while (iter < iter_limit) {
    // Time limit is handled on the host side

    // 1. Steepest-edge pricing.
    i_t direction        = 0;
    i_t basic_leaving    = -1;
    f_t max_val;
    const i_t leaving = steepest_edge_pricing_with_infeasibilities(
      lp, settings, x, delta_y_steepest_edge, basic_mark.data(), squared_infeas, infeas_idx,
      direction, basic_leaving, max_val);
    GPU_PH(0)
    if (leaving == -1) {
      finish(iter);
      return dual::status_t::OPTIMAL;
    }

    // 2. BTran.
    lane0_serial([&] {   // clear() is a header RMW on size_
      delta_y_sparse.clear();
      UTsol_sparse.clear();
    });
    compute_delta_y(ft, basic_leaving, direction, delta_y_sparse, UTsol_sparse);
    // Butterfly reduction of ||delta_y||^2 under Direct. Reordered vs the serial sum, and
    // dy_norm_sq feeds the SE norms -> pricing, so this is a deliberate arithmetic change.
    // warp_reduce_add returns the FULL sum on every lane, so dy_norm_sq stays warp-uniform and the
    // reprice branch below is taken uniformly.
    f_t dy_norm_sq;
    const int lane = static_cast<int>(threadIdx.x);
    const f_t* dyx = delta_y_sparse.x.data();
    const i_t nz   = (i_t)delta_y_sparse.i.size();   // trip count as in norm2_squared()
    f_t part = 0.0;
    for (i_t k = lane; k < nz; k += 32) { part += dyx[k] * dyx[k]; }
    dy_norm_sq = warp_reduce_add(part);
    GPU_PH(1)

    // Steepest-edge norm drift safeguard: dy_norm_sq is the true norm of the leaving
    // row. If the maintained norm used for pricing has drifted below
    // settings.steepest_edge_ratio of it, the leaving choice was made on a stale norm --
    // reset it and re-price (strict port of the CPU simplified, phase2.cpp:4037).
    // monolith is the bit-exact reference.
    if (delta_y_steepest_edge[leaving] < settings.steepest_edge_ratio * dy_norm_sq) {
      ft.path(P_SE_RESET);
      lane0_serial([&] { delta_y_steepest_edge[leaving] = dy_norm_sq; });
      continue;
    }

    // 3. Reduced-cost SpMv: transpose (CSR) when delta_y is sparse (<=30% dense), else
    //    dense CSC. Mirrors the CPU driver.
    i_t delta_y_nz0      = 0;
    const i_t nz_delta_y = (i_t)delta_y_sparse.i.size();
    // INTEGER count, so ballot/popc is EXACT — nothing is reordered the way an FP sum would be.
    // Every lane accumulates the same running total, keeping delta_y_nz0 (and the
    // used_transpose_dz branch it gates) warp-uniform.
    // Own scope: `lane`/`dyx` are also named by the norm reduction above, which used to sit in its
    // own `if constexpr (Direct)` block.
    {
      const int lane = static_cast<int>(threadIdx.x);
      const f_t* dyx = delta_y_sparse.x.data();
      for (i_t base = 0; base < nz_delta_y; base += 32) {
        const i_t k    = base + lane;
        const bool hit = (k < nz_delta_y) && (fabs(dyx[k]) > 1e-12);
        delta_y_nz0 += __popc(__ballot_sync(0xffffffffu, hit));
      }
    }
    const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
    const bool used_transpose_dz    = delta_y_nz_percentage <= c_dz_sparse_pct;
    ft.path(used_transpose_dz ? P_DZ_ROWWISE : P_DZ_COLDOT);
    if (used_transpose_dz) {
      compute_delta_z(ft, Arow, delta_y_sparse, leaving, direction,
                                    basic_mark.data(), delta_z_mark.data(), delta_z_indices,
                                    delta_z.data());
    } else {
      // to_dense RESIZES delta_y (header RMW + possible device malloc). warp_to_dense keeps the
      // sizing on lane 0 and warps the O(n) zero-fill + nnz scatter — bit-exact (distinct indices).
      ft.warp_to_dense(delta_y_sparse, delta_y);
      compute_reduced_cost_update(lp, nonbasic_list, delta_y.data(), leaving, direction,
                                  delta_z_mark.data(), delta_z_indices, delta_z.data());
    }
    GPU_PH(2)

    // 4. Simple ratio test.
    f_t step_length       = 0.0;
    i_t nonbasic_entering = -1;
    i_t entering          = -1;
    // The ordered selector writes its outputs through lane 0, so aim
    // them at slot_state (global) and let the __syncwarp publish them to every lane — locals would
    // only ever be set on lane 0.
    warp_rt_body(settings.dual_tol / 10, delta_z.data(), &st.entering, &st.nonbasic_entering,
                 settings.pivot_tol, &st.step_length, vstatus, z,
                 settings.zero_tol,
                 nonbasic_list, n - m);
    __syncwarp(0xffffffffu);
    entering          = st.entering;
    nonbasic_entering = st.nonbasic_entering;
    step_length       = st.step_length;
    GPU_PH(3)
    if (entering == -1) {
      finish(iter);
      return dual::status_t::DUAL_UNBOUNDED;
    }

    update_dual_variables(
      ft, delta_y_sparse, delta_z_indices, delta_z.data(), step_length, leaving, y, z);
    GPU_PH(4)

    // FTran.
    lane0_serial([&] {   // clear() is a header RMW; from_csc_column reserve+push_back mallocs
      utilde_sparse.clear();
      scaled_dxB_sparse.clear();
      v_sparse.clear();
      rhs_sparse.from_csc_column(lp.A, entering);
    });
    ft.b_solve2(rhs_sparse, delta_y_sparse, scaled_dxB_sparse, utilde_sparse, v_sparse);
    GPU_PH(5)
    i_t dx_status =
      compute_delta_x(lp, ft, entering, leaving, basic_leaving, direction, basic_list, rhs_sparse,
                      delta_z.data(), x, utilde_sparse, scaled_dxB_sparse, delta_x.data());
    if (dx_status == -1) {
      finish(iter);
      return dual::status_t::NUMERICAL;
    }

    update_primal_variables(scaled_dxB_sparse, basic_list, delta_x.data(), entering, x);

    // Incrementally update the objective the SAME way the CPU driver does (phase2.cpp:4259):
    // obj += delta_x·objective over the changed basics + entering, true costs, called with the
    // OLD basic_list (before the swap below), WITHOUT the bound-flip contribution.
    update_objective(basic_list, scaled_dxB_sparse, lp.objective, delta_x.data(), entering, obj);
    GPU_PH(6)

    // A cutoff node is complete after its primal/objective update. The remaining state is only
    // needed by a following pivot, so do not maintain it for a node that is being fathomed.
    // The tracked objective is a valid LOWER BOUND on the node optimum only while the basis stays
    // dual feasible -- that is what weak duality gives, and it is the whole justification for
    // fathoming here. The driver never re-tests the invariant, so test it now against the
    // MAINTAINED z: if some nonbasic reduced cost no longer carries the sign its status demands,
    // `obj` is not a bound on anything and fathoming on it can discard a live subtree.
    // Not a certificate, a precondition: it costs one pass over the columns and it is the exact
    // condition the objective's validity rests on.
    if (isfinite(settings.cut_off) && obj >= settings.cut_off) { ft.path(P_CUTOFF_SCAN); }
    if (isfinite(settings.cut_off) && obj >= settings.cut_off &&
        dual_feasible_within_tol(lp, vstatus, z, settings.dual_tol)) {
      finish(iter + 1);
      return dual::status_t::CUTOFF;
    }
    GPU_PH(7)

    update_steepest_edge_norms(settings, basic_list, ft, direction, delta_y_sparse, dy_norm_sq,
                               scaled_dxB_sparse, basic_leaving, entering, v.data(), v_sparse,
                               delta_y_steepest_edge);
    GPU_PH(8)

    // 5. Basis swap + MPF rank-1 stacking update.
    // Every line here is a read-modify-write on node-local state. Arow stays immutable;
    // basic_mark is the sole source of truth for changing basis membership.
    i_t _fu_status = 0;
    lane0_serial([&] {
      vstatus[entering] = variable_status_t::BASIC;
      vstatus[leaving]  = (lp.lower[leaving] == lp.upper[leaving])
                           ? variable_status_t::NONBASIC_FIXED
                           : static_cast<variable_status_t>(-direction);
      basic_list[basic_leaving]        = entering;
      nonbasic_list[nonbasic_entering] = leaving;
      nonbasic_pos[leaving]            = nonbasic_entering;
      basic_mark[leaving]              = -1;
      basic_mark[entering]             = basic_leaving;
    });

    // ft.update is COLLECTIVE now, so it must sit outside the lane-0 region above. It returns the
    // same status on every lane (its internal lane0_bcast), so the NUMERICAL exit is already
    // uniform; the broadcast below is kept for the !Warp instantiation.
    // PERIODIC REFACTORIZATION, ON THE CPU'S SCHEDULE. The CPU evaluates this BEFORE ft.update and
    // SKIPS that update when it fires (phase2.cpp). Testing after the update instead looks
    // equivalent and is not: it refactorizes one pivot earlier while carrying one update more, so
    // from the first refactor onward the two sides hold different numbers of stacked updates on
    // LUs of DIFFERENT bases.
    if constexpr (AllowRefactorExit) {
      bool yield = false;
      if (settings.yield_count != nullptr) {
        unsigned int done = 0;
        if (threadIdx.x == 0) { done = *reinterpret_cast<const volatile unsigned int*>(settings.yield_count); }
        yield = __shfl_sync(0xffffffffu, done, 0) >= settings.yield_at;
      }
      if (ft.num_updates() > settings.refactor_frequency || yield) {
        ft.path(P_REFACTOR);
        refactor_reason = 0;  // periodic schedule (or yield: same resume)
        suspend(iter + 1);
        return dual::status_t::REFACTOR_NEEDED;
      }
    }

    _fu_status = ft.warp_update(utilde_sparse, UTsol_sparse, basic_leaving);
    _fu_status = lane0_bcast(_fu_status);
    GPU_PH(9)
    if (_fu_status != 0) {
      // The CPU driver responds to an unsafe rank-1 update by refactorizing in place. The
      // single-problem GPU path asks its host resume loop to do the same. The pivot and basis swap
      // already happened, so resume from iter + 1 and do not repeat the pivot.
      if constexpr (AllowRefactorExit) {
        refactor_reason = 1;  // rejected update: host refactor + maintained-dual recovery
        suspend(iter + 1);
        return dual::status_t::REFACTOR_NEEDED;
      }
      finish(iter);
      return dual::status_t::NUMERICAL;
    }

    compute_initial_primal_infeasibilities(
      lp, settings, ft, basic_list, x, squared_infeas, infeas_idx, primal_inf);
    clear_delta_z(entering, leaving, delta_z_mark.data(), delta_z_indices, delta_z.data());
    GPU_PH(10)

    // Increment only on a COMPLETED iteration, at the end of the body — matching the CPU
    // simplified driver's iter++ (phase2.cpp:4370). The SE-reset `continue` above jumps
    // back to `while` without counting, so `iter` counts pivots, not loop passes.
#if GPU_PIVOT_TRACE
    {
      const f_t traced_obj = compute_objective(lp, x);
      if (threadIdx.x == 0) {
        // Mirror of the host's [CDINF]: dual infeasibility of the MAINTAINED z at this pivot. The
        // driver never re-tests this invariant, so its growth is invisible to the solve yet decides
        // whether the CUTOFF test is sound.
        {
          f_t dinf_maintained = 0.0;
          for (i_t q = 0; q < lp.num_cols - lp.num_rows; ++q) {
            const i_t j = nonbasic_list[q];
            if (vstatus[j] == variable_status_t::NONBASIC_LOWER) {
              dinf_maintained = fmax(dinf_maintained, -z[j]);
            } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER) {
              dinf_maintained = fmax(dinf_maintained, z[j]);
            }
          }
          printf("[GDINF] iter=%d dinf=%.17g\n", (int)iter, dinf_maintained);
        }
        auto dbits = [](f_t v) { return (unsigned long long)__double_as_longlong(v); };
        printf("[GCMP] iter=%d leave=%d dir=%d enter=%d step=%016llx obj=%016llx p_inf=%016llx "
               "nInf=%d nupd=%d\n",
               (int)iter, (int)leaving, (int)direction, (int)entering, dbits(step_length),
               dbits(traced_obj), dbits(primal_inf), (int)infeas_idx.size(),
               (int)ft.num_updates());
        // ORDER-INDEPENDENT checksums of the three vectors the pivot is built from, matching the
        // host's [CVEC]. XOR of raw bits does not depend on traversal order, so a mismatch means the
        // VECTOR differs -- which localises the divergence to BTRAN (delta_y), the reduced-cost SpMV
        // (delta_z) or FTRAN (delta_x).
        {
          auto xorbits = [&](const f_t* v, i_t cnt) {
            unsigned long long h = 0;
            for (i_t q = 0; q < cnt; ++q) {
              if (v[q] != 0.0) { h ^= (unsigned long long)__double_as_longlong(v[q]); }
            }
            return h;
          };
          auto nnz = [](const f_t* v, i_t cnt) {
            i_t c = 0; for (i_t q = 0; q < cnt; ++q) { c += (v[q] != 0.0); } return c;
          };
          printf("[GVEC] iter=%d dy=%016llx dynz=%d dz=%016llx dznz=%d dx=%016llx dxnz=%d\n",
                 (int)iter, xorbits(delta_y.data(), m), (int)nnz(delta_y.data(), m),
                 xorbits(delta_z.data(), n), (int)nnz(delta_z.data(), n),
                 xorbits(delta_x.data(), n), (int)nnz(delta_x.data(), n));
        }
      }
      __syncwarp(0xffffffffu);
    }
#endif
    lane0_serial([&] { iter++; });   // RMW on st.iter in global memory

  }
  finish(iter_limit);
  return dual::status_t::ITERATION_LIMIT;
#undef GPU_PH
}

// ---------------------------------------------------------------------------
// Kernel argument bundle (all raw device pointers + scalars, passed by value).
// ---------------------------------------------------------------------------

struct kernel_args {
  // Non-zero for a fixed batch seat that already reached a terminal status in an earlier round.
  // Its block is a no-op. Keeping the seat in the launch makes host-refactor batching a simple
  // synchronous "mark done, refactor requests, relaunch the same array" loop.
  i_t done;
  // gpu_batch_seat_command. Fresh/refactor commands reconstruct the slot header and install the
  // uploaded base factor. CHILD_CONTINUE deliberately leaves the live FT object (including its MPF
  // stack and routing statistics) untouched.
  i_t command;
  i_t m, n;
  // LP
  i_t *A_col_start, *A_i;
  f_t* A_x;
  i_t A_nz;
  // Host-built CSR of A (Arow). Always uploaded; the kernel has no transpose of its own.
  const i_t* Arow_row_start;
  const i_t* Arow_j;
  const f_t* Arow_x;
  f_t *rhs, *lower, *upper, *objective;
  gpu_bound_delta<i_t, f_t>* bound_deltas;
  i_t bound_delta_count;
  const f_t *root_lower, *root_upper;
  i_t bounds_from_root;
  f_t obj_scale, obj_constant;
  // Fixed-factor row indices use i_t throughout.
  i_t* L0_col_start;
  i_t* L0_i;
  f_t* L0_x;
  i_t L0_nz;
  i_t* U0_col_start;
  i_t* U0_i;
  f_t* U0_x;
  i_t U0_nz;
  i_t* L0t_col_start;
  i_t* L0t_i;
  f_t* L0t_x;
  i_t L0t_nz;
  i_t* U0t_col_start;
  i_t* U0t_i;
  f_t* U0t_x;
  i_t U0t_nz;
  i_t *row_perm, *inv_row_perm;
  i_t* xi_workspace;
  f_t* x_workspace;
  unsigned char* reach_mark;
  i_t num_updates;
  i_t refactor_frequency;
  // Caps on the preallocated MPF rank-1 store (see gpu_basis_update_t). 0/0 = uncapped, i.e.
  // the historical grow-in-kernel behaviour. Sized by mpf_store_caps().
  i_t mpf_update_cap;
  f_t* fm{nullptr};  // fission replay matrix M of this seat, fm_kcap^2 (column-major)
  i_t fm_kcap{0};
  i_t mpf_eta_nnz_cap;
  // Per-slot PERSISTENT solve state (slot_state<FT>): host-allocated, ZEROED blob that the kernel
  // placement-news into on first entry and reuses thereafter. This is what lets the per-pivot
  // workspaces + ft's mutable state (S_, routing stats, scratch) outlive a single kernel launch —
  // host-visible "already constructed" flag (0 = fresh node → construct + run setup).
  slot_state<gpu_basis_update_t>* slot;
  // Per-slot flat buffer for every gpu_span payload. Ordinary VRAM,
  // so it is NOT subject to the driver's ~16 GB in-kernel malloc cap -- which is what used to set the
  // batch-size ceiling. Null/0 is legal and simply leaves every buffer on device malloc().
  slot_buffers buf;
  // Same base pointer in every item, so the host reads all N terminal states in ONE D2H copy to
  // decide when the super-step loop is done. Null on the monolith path.
  i_t* status_arr;
  // state (mutable)
  variable_status_t* vstatus;
  i_t* basic_list;
  i_t* nonbasic_list;
  f_t *x, *y, *z;
  f_t* dy_se;
  // Non-zero when this launch RESUMES after a host refactorization: the host already rebuilt the
  // primal point the CPU's way, so setup() must leave x alone.
  i_t refactor_resume;
  // Non-zero only on the one relaunch requested by refactor_reason=2. That launch retries the
  // terminal optimality proof; it must not convert the request into a new terminal CUTOFF.
  i_t terminal_certificate_retry;
  // Carried across a REFACTOR_NEEDED resume so the next launch CONTINUES the solve rather than
  // restarting it. The CPU's in-place refactorization leaves both untouched, so a faithful port
  // must too: recomputing the objective from x and resetting the iteration counter are exactly the
  // two things that turn a continuation into a restart.
  f_t resume_obj;
  i_t resume_iter;
  // settings
  f_t primal_tol, dual_tol, pivot_tol, tight_tol, fixed_tol, zero_tol, time_limit, cut_off,
    steepest_edge_ratio;
  f_t start_time;
  // outputs
  i_t* out_status;
  f_t* out_objective;
  f_t* out_user_objective;
  i_t* out_iterations;
  // Device bound strengthening (nondeterministic pool): integer flags per column, per-seat scratch
  // (2m doubles + 2m + n bytes), and the switch.
  const unsigned char* var_int;
  unsigned char* bs_scratch;
  f_t integer_tol;
  i_t strengthen;
  // Device LU of a FRESH seat: lu_info = {status, L nz, U nz} written by the dense batch.
  const int* lu_info;
  i_t device_lu;
  // Non-zero: the host staged L0/U0 only; setup builds L0t/U0t from them (CUOPT_BNB_GPU_DEVICE_TRANSPOSE).
  i_t transpose_on_device;
};

__device__ gpu_csc_view make_view(
  i_t rows, i_t cols, i_t nz, i_t* col_start, i_t* idx, f_t* val)
{
  gpu_csc_view v;
  v.m         = rows;
  v.n         = cols;
  v.nz_max    = nz;
  v.col_start = gpu_span<const i_t>(col_start, cols + 1);
  v.i         = gpu_span<i_t>(idx, nz);
  v.x         = gpu_span<f_t>(val, nz);
  return v;
}

__device__ gpu_csc_view make_factor_view(
  i_t rows, i_t cols, i_t nz, i_t* col_start, i_t* idx, f_t* val)
{
  gpu_csc_view v;
  v.m         = rows;
  v.n         = cols;
  v.nz_max    = nz;
  v.col_start = gpu_span<const i_t>(col_start, cols + 1);
  v.i         = gpu_span<i_t>(idx, nz);
  v.x         = gpu_span<f_t>(val, nz);
  return v;
}


// Fill the warp-solve config for all 4 operators (lane 0, once at start). The
// level schedules are written into the caller-owned gpu_spans ls[o]/co[o]
// (kept alive for the whole kernel); `lvlscratch` is reused m-sized workspace.
__device__ __noinline__ void warp_build_config(const gpu_basis_update_t& ft,
                                  warp_solve_config_t& cfg,
                                   gpu_span<i_t>* lev, gpu_span<i_t>* ls, gpu_span<i_t>* co,
                                   gpu_span<i_t>& sp_ls_v, gpu_span<i_t>& sp_co_v,
                                   gpu_span<i_t>& sp_pos_v, gpu_span<i_t>& lvl_pos_v)
{
  // Reach-restricted sparse-solve scratch (built fresh per solve in warp_dispatch_sptrsv): m(+1)-sized.
  const i_t m = ft.L0_.m;
  sp_co_v.resize(m);
  sp_ls_v.resize(m + 1);
  sp_pos_v.resize(m + 1);
  // Counting-sort cursor for warp_build_levels, shared by all four ops (they run in sequence).
  // Sized once at the bound so the per-op assign() inside never reallocates.
  lvl_pos_v.resize(m + 1);
  cfg.sp_co  = sp_co_v.data();
  cfg.sp_ls  = sp_ls_v.data();
  cfg.sp_pos = sp_pos_v.data();
  // op order: L, U, Lᵀ, Uᵀ. Gather matrix + flags per op.
  const gpu_csc_view* mats[kNumWarpSolveOps] = {
    &ft.L0_transpose_, &ft.U0_transpose_, &ft.L0_, &ft.U0_};
  for (int o = 0; o < kNumWarpSolveOps; ++o) {
    cfg.cs[o]         = mats[o]->col_start.data();
    cfg.ridx[o]       = mats[o]->i.data();
    cfg.val[o]        = mats[o]->x.data();
    // The flags are NOT stored in cfg: the four wavefront bodies (dts::wavefront_{L,U,LT,UT}_solve)
    // are separate specialized functions that encode their own direction, so building the schedule
    // is the only thing that still needs them. op_* are constexpr FUNCTIONS, hence callable here
    // with a runtime `o` -- no materialized flag array required.
    // Lᵀ (2) and Uᵀ (3) reverse the DAGs of L (0) and U (1).
    const bool rev = o >= 2 && c_derive_levels != 0;
    cfg.nlev[o]  = warp_build_levels(*mats[o], op_diag_first(o), op_ascending(o), lev[o], ls[o],
                                     co[o], lvl_pos_v, rev ? lev[o - 2].data() : nullptr,
                                     rev ? static_cast<i_t>(cfg.nlev[o - 2]) : 0);
    cfg.lstart[o]     = ls[o].data();
    cfg.corder[o]     = co[o].data();
    cfg.level[o]      = lev[o].data();  // per-column level, retained for the reach-restricted sparse solve
  }
}

// warp_build_config on all 32 lanes (CUOPT_BNB_GPU_WARP_LEVELS, default on).
__device__ __noinline__ void warp_build_config_all(const gpu_basis_update_t& ft,
                                                   warp_solve_config_t& cfg,
                                                   gpu_span<i_t>* lev, gpu_span<i_t>* ls, gpu_span<i_t>* co,
                                                   gpu_span<i_t>& sp_ls_v, gpu_span<i_t>& sp_co_v,
                                                   gpu_span<i_t>& sp_pos_v, gpu_span<i_t>& lvl_pos_v)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t m    = ft.L0_.m;
  // resize() zero-fills [size, n) serially; fill all of [0, n) across the warp instead.
  lane0_serial([&] {
    sp_co_v.reserve(m), sp_co_v.set_size(m);
    sp_ls_v.reserve(m + 1), sp_ls_v.set_size(m + 1);
    sp_pos_v.reserve(m + 1), sp_pos_v.set_size(m + 1);
    lvl_pos_v.reserve(m + 1), lvl_pos_v.set_size(m + 1);
  });
  for (i_t k = lane; k <= m; k += 32) {
    if (k < m) { sp_co_v.data()[k] = 0; }
    sp_ls_v.data()[k] = 0, sp_pos_v.data()[k] = 0, lvl_pos_v.data()[k] = 0;
  }
  lane0_serial([&] {
    cfg.sp_co  = sp_co_v.data();
    cfg.sp_ls  = sp_ls_v.data();
    cfg.sp_pos = sp_pos_v.data();
  });
  const gpu_csc_view* mats[kNumWarpSolveOps] = {
    &ft.L0_transpose_, &ft.U0_transpose_, &ft.L0_, &ft.U0_};
  for (int o = 0; o < kNumWarpSolveOps; ++o) {
    const bool rev = o >= 2 && c_derive_levels != 0;
    const int nl   = warp_build_levels_all(*mats[o], op_diag_first(o), op_ascending(o), lev[o], ls[o],
                                         co[o], lvl_pos_v, rev ? lev[o - 2].data() : nullptr,
                                         rev ? static_cast<i_t>(cfg.nlev[o - 2]) : 0, c_long_col);
    lane0_serial([&] {
      cfg.cs[o]     = mats[o]->col_start.data();
      cfg.ridx[o]   = mats[o]->i.data();
      cfg.val[o]    = mats[o]->x.data();
      cfg.nlev[o]   = nl;
      cfg.lstart[o] = ls[o].data();
      cfg.corder[o] = co[o].data();
      cfg.level[o]  = lev[o].data();
    });
  }
}

// Re-wire the block's config to a seat whose schedules warp_build_config already built (same factor).
__device__ void warp_wire_config(const gpu_basis_update_t& ft,
                                 warp_solve_config_t& cfg,
                                 gpu_span<i_t>* lev, gpu_span<i_t>* ls, gpu_span<i_t>* co,
                                 gpu_span<i_t>& sp_ls_v, gpu_span<i_t>& sp_co_v,
                                 gpu_span<i_t>& sp_pos_v, const i_t* nlev)
{
  cfg.sp_co  = sp_co_v.data();
  cfg.sp_ls  = sp_ls_v.data();
  cfg.sp_pos = sp_pos_v.data();
  const gpu_csc_view* mats[kNumWarpSolveOps] = {
    &ft.L0_transpose_, &ft.U0_transpose_, &ft.L0_, &ft.U0_};
  for (int o = 0; o < kNumWarpSolveOps; ++o) {
    cfg.cs[o]     = mats[o]->col_start.data();
    cfg.ridx[o]   = mats[o]->i.data();
    cfg.val[o]    = mats[o]->x.data();
    cfg.nlev[o]   = nlev[o];
    cfg.lstart[o] = ls[o].data();
    cfg.corder[o] = co[o].data();
    cfg.level[o]  = lev[o].data();
  }
}

// Solve one problem selected by idx. A persistent block can call this repeatedly, but each call
// reads and writes only that problem's independently marshalled buffers in args[idx].
// g_gpu_capacity_exceeded is reset by the HOST before launch (not here) so a late block cannot
// clear a failure flag an earlier block already set.
// The host bounds_strengthening_t pass on one warp, from every row or the delta columns' rows.
__device__ __noinline__ void warp_strengthen_bounds(kernel_args& a, bool from_deltas)
{
  constexpr unsigned full = 0xffffffffu;
  const i_t m    = a.m;
  const i_t n    = a.n;
  const i_t lane = static_cast<i_t>(threadIdx.x);
  const f_t inf  = cuda::std::numeric_limits<f_t>::infinity();
  const f_t ptol = a.primal_tol;
  f_t* dmin           = reinterpret_cast<f_t*>(a.bs_scratch);
  f_t* dmax           = dmin + m;
  unsigned char* rc   = reinterpret_cast<unsigned char*>(dmax + m);
  unsigned char* rcn  = rc + m;
  unsigned char* vc   = rcn + m;
  f_t* lower          = a.lower;
  f_t* upper          = a.upper;
  for (i_t i = lane; i < m; i += 32) {
    rc[i]  = from_deltas ? 0 : 1;
    rcn[i] = 0;
  }
  for (i_t j = lane; j < n; j += 32) { vc[j] = 0; }
  __syncwarp(full);
  if (from_deltas) {
    for (i_t q = 0; q < a.bound_delta_count; ++q) {
      const i_t j = a.bound_deltas[q].column;
      for (i_t p = a.A_col_start[j] + lane; p < a.A_col_start[j + 1]; p += 32) { rc[a.A_i[p]] = 1; }
    }
    __syncwarp(full);
  }
  for (int iter = 0; iter < 10; ++iter) {
    bool infeasible = false;
    for (i_t i = lane; i < m; i += 32) {
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
      if (min_a > b + ptol || max_a < b - ptol) { infeasible = true; }
      dmin[i] = b - min_a;
      dmax[i] = b - max_a;
    }
    if (__any_sync(full, infeasible)) { return; }
    bool changed = false;
    for (i_t k = lane; k < n; k += 32) {
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
        infeasible = true;
        continue;
      }
      if (new_lb != old_lb || new_ub != old_ub) {
        for (i_t p = a.A_col_start[k]; p < a.A_col_start[k + 1]; ++p) { rcn[a.A_i[p]] = 1; }
      }
      lower[k] = (new_lb < new_ub) ? new_lb : new_ub;
      upper[k] = (new_lb < new_ub) ? new_ub : new_lb;
      changed |= updated;
    }
    if (__any_sync(full, infeasible)) { return; }
    if (!__any_sync(full, changed)) { return; }
    for (i_t i = lane; i < m; i += 32) {
      rc[i]  = rcn[i];
      rcn[i] = 0;
    }
    for (i_t j = lane; j < n; j += 32) { vc[j] = 0; }
    __syncwarp(full);
  }
}

// CSC transpose of an m x m factor by one warp, laid out as the host's transpose: each transposed
// column lists its entries in ascending original-column order. `cur` is m ints of scratch.
__device__ __noinline__ void warp_csc_transpose(i_t m, const i_t* cs, const i_t* ci, const f_t* cx,
                                                i_t* ts, i_t* ti, f_t* tx, i_t* cur)
{
  constexpr unsigned full = 0xffffffffu;
  const int lane = static_cast<int>(threadIdx.x);
  for (i_t i = lane; i <= m; i += 32) { ts[i] = 0; }
  __syncwarp(full);
  const i_t nz = cs[m];
  for (i_t p = lane; p < nz; p += 32) { atomicAdd(&ts[ci[p] + 1], 1); }
  __syncwarp(full);
  i_t carry = 0;
  for (i_t base = 1; base <= m; base += 32) {
    const i_t i = base + lane;
    i_t v       = i <= m ? ts[i] : 0;
    for (int o = 1; o < 32; o <<= 1) {
      const i_t t = __shfl_up_sync(full, v, o);
      if (lane >= o) { v += t; }
    }
    if (i <= m) { ts[i] = v + carry; }
    carry += __shfl_sync(full, v, 31);
  }
  __syncwarp(full);
  for (i_t i = lane; i < m; i += 32) { cur[i] = ts[i]; }
  __syncwarp(full);
  // Columns in order; the rows inside one column are distinct, so the lanes never share a cursor.
  for (i_t j = 0; j < m; ++j) {
    const i_t e = cs[j + 1];
    for (i_t p = cs[j] + lane; p < e; p += 32) {
      const i_t q = cur[ci[p]]++;
      ti[q]       = j;
      tx[q]       = cx[p];
    }
    __syncwarp(full);
  }
}

template <bool AllowRefactorExit, bool PersistentSeats = false, bool FissionSetup = false>
__device__ __forceinline__ void gpu_phase2_kernel_body(kernel_args* args,
                                                       i_t idx,
                                                       const unsigned int* yield_count = nullptr,
                                                       unsigned int yield_at           = 0)
{
  using FT = gpu_basis_update_t;
  using warp_config_type = FT::warp_config_type;
  // NOTE: the CUDA 13 `.pragma "enable_smem_spilling"` (spill regs to shared instead of local) is
  // INCOMPATIBLE here — ptxas fails with "(C7800) Smem spilling should not be enabled when functions
  // use abi": this kernel makes ABI (non-inlined) device calls into the whole driver, and the pragma
  // requires a fully-inlined call tree. Not usable without force-inlining everything.
  // Reject a launch geometry that doesn't match the compiled path (the whole point
  // of the compile-time Warp parameter: no runtime branch selects opt vs unopt).
  if (blockDim.x != 32) { assert(false); return; }

  // Shared config. Under Direct it is READ-ONLY after the one-time build below: the Direct arms
  // still fetch the elimination-level schedule through ft.warp_cfg_, only the per-dispatch REQUEST
  // fields are unused. Unused entirely when Warp==false.
  __shared__ warp_config_type cfg_shared;

  // No parked-helper model any more: on the warp path all 32 lanes fall through and run the driver
  // (Direct). The lanes-1..31 -> warp_helper_loop split was deleted with the dispatcher.

  kernel_args& a = args[idx];

  // Uniform for the whole warp and deliberately before touching any other argument: done seats
  // are marshalled as this marker alone, with all problem pointers left null.
  if (a.done != 0) { return; }
  const long long pc0 = clock64();

  if constexpr (PersistentSeats) {
    if (a.device_lu != 0) {
      if (a.lu_info[0] != 0) {
        // The device LU failed (singular, tiny pivot, or no room): the node goes to the CPU.
        lane0_serial([&] {
          a.out_status[0]       = static_cast<i_t>(dual::status_t::NUMERICAL);
          a.out_status[1]       = 0;
          *a.out_objective      = 0.0;
          *a.out_user_objective = 0.0;
          *a.out_iterations     = 0;
        });
        return;
      }
      lane0_serial([&] {
        a.L0_nz = a.L0t_nz = a.lu_info[1];
        a.U0_nz = a.U0t_nz = a.lu_info[2];
      });
    }
  }

  bool child_continue = false;
  if constexpr (PersistentSeats) {
    child_continue = a.command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE);
    if (!FissionSetup && a.bounds_from_root != 0) {  // fission: fission_presetup did it
      for (i_t j = static_cast<i_t>(threadIdx.x); j < a.n; j += 32) {
        a.lower[j] = a.root_lower[j];
        a.upper[j] = a.root_upper[j];
      }
      __syncwarp(0xffffffffu);
      for (i_t q = static_cast<i_t>(threadIdx.x); q < a.bound_delta_count; q += 32) {
        const auto& d = a.bound_deltas[q];
        a.lower[d.column] = d.lower;
        a.upper[d.column] = d.upper;
      }
      __syncwarp(0xffffffffu);
    }
  }
  if constexpr (PersistentSeats) {
    if (!FissionSetup && child_continue) {
    // The persistent seat already contains the parent's complete solve representation. Apply the
    // child's sparse bounds before constructing the lp view used by pricing/setup. One lane writes
    // the small ordered list; the warp fence publishes it before any lane reads the bounds.
    lane0_serial([&] {
      for (i_t q = 0; q < a.bound_delta_count; ++q) {
        const auto& d = a.bound_deltas[q];
        if ((d.mask & 1u) != 0u) { a.lower[d.column] = d.lower; }
        if ((d.mask & 2u) != 0u) { a.upper[d.column] = d.upper; }
      }
    });
    }
  }

  const long long pcb = clock64();
  // FRESH (all rows) and CHILD_CONTINUE (rows of the delta columns); a resume keeps its bounds.
  if (!FissionSetup && a.strengthen != 0 && (child_continue || a.refactor_resume == 0)) {
    __syncwarp(0xffffffffu);
    warp_strengthen_bounds(a, child_continue);
    __syncwarp(0xffffffffu);
  }

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
  settings.yield_count         = yield_count;
  settings.yield_at            = yield_at;

  // Per-slot state (see slot_state). Fresh and refactor commands reset it and wire the
  // read-only factorization views into its ft; child continuation retains the live state.
  // RMM allocates typed but uninitialized device storage. Placement new below only calls the
  // constructor in that already-allocated memory; it does not allocate device memory.
  // Construction, ft wiring and S_.init_empty are RMW on shared state and belong to lane 0.
  const long long pcs = clock64();
  // Fission fresh seat: x_B from the inverses. The transposes and level schedules are skipped unless
  // the direct FTRAN (CUOPT_EXP_FTRAN=direct) solves through them.
  const bool lean      = FissionSetup && !child_continue && a.refactor_resume == 0 && a.device_lu == 0;
  const bool skip_tri  = lean && c_ftran_direct == 0;
  if constexpr (PersistentSeats) {
    if (!skip_tri && !child_continue && a.transpose_on_device != 0) {
      warp_csc_transpose(m, a.L0_col_start, a.L0_i, a.L0_x, a.L0t_col_start, a.L0t_i, a.L0t_x, a.xi_workspace);
      warp_csc_transpose(m, a.U0_col_start, a.U0_i, a.U0_x, a.U0t_col_start, a.U0t_i, a.U0t_x, a.xi_workspace);
    }
  }
  const long long pc1 = clock64();
  if (!child_continue) {
    lane0_serial([&] {
      slot_state<FT>& st = *new (a.slot) slot_state<FT>();
    // Wire every buffer into the host-provided slot memory. Nothing allocates: a buffer that
    // outgrows its layout slot is a layout bug and is reported, not silently regrown.
    // Exceeding a span's reserved capacity is a host-side sizing bug; a debug build asserts.
    st.attach_all_buffers(a.buf, m, n, a.mpf_update_cap, a.mpf_eta_nnz_cap);
    FT& ft                      = st.ft;
    ft.num_updates_             = a.num_updates;
    ft.refactor_frequency_      = a.refactor_frequency;
    ft.L0_                      = make_factor_view(m, m, a.L0_nz, a.L0_col_start, a.L0_i, a.L0_x);
    ft.U0_                      = make_factor_view(m, m, a.U0_nz, a.U0_col_start, a.U0_i, a.U0_x);
    ft.L0_transpose_            = make_factor_view(m, m, a.L0t_nz, a.L0t_col_start, a.L0t_i, a.L0t_x);
    ft.U0_transpose_            = make_factor_view(m, m, a.U0t_nz, a.U0t_col_start, a.U0t_i, a.U0t_x);
    ft.row_permutation_         = gpu_span<i_t>(a.row_perm, m);
    ft.inverse_row_permutation_ = gpu_span<i_t>(a.inv_row_perm, m);
    ft.xi_workspace_            = gpu_span<i_t>(a.xi_workspace, 2 * m);
    ft.x_workspace_             = gpu_span<f_t>(a.x_workspace, m);
    ft.reach_mark_              = gpu_span<unsigned char>(a.reach_mark, m);
    ft.update_cap_  = a.mpf_update_cap;
    ft.eta_nnz_cap_ = a.mpf_eta_nnz_cap;
    ft.reserve_store(m);
      ft.hypersparse_threshold_ = c_hypersparse_threshold;
      if constexpr (PersistentSeats) { st.price_fresh = a.device_lu != 0; }
    });   // end lane-0 construction
  }
  slot_state<FT>& st = *a.slot;
  FT& ft = st.ft;

  if (!child_continue) {
    // The factor-solve workspaces are reused across pivots but are not part of a node's state.
    // Some terminal paths deliberately leave them dirty (notably a rejected MPF update). A fresh
    // node or host-refactor resume in the same persistent seat must restore the clean-workspace
    // precondition that the one-shot marshaller gets from alloc_zeroed; otherwise its very first
    // hypersparse BTRAN can consume stale reach/value state and corrupt steepest-edge norms.
    const i_t lane = static_cast<i_t>(threadIdx.x);
    for (i_t k = lane; k < m; k += 32) {
      ft.xi_workspace_[k]     = 0;
      ft.xi_workspace_[m + k] = 0;
      ft.x_workspace_[k]      = 0.0;
      ft.reach_mark_[k]       = 0;
    }
    __syncwarp(0xffffffffu);
  }
  const long long pc2 = clock64();

  gpu_solution sol;
  sol.x = gpu_span<f_t>(a.x, n);
  sol.y = gpu_span<f_t>(a.y, m);
  sol.z = gpu_span<f_t>(a.z, n);

  // Warp path: precompute the fixed L0/U0 elimination-level schedules once and wire ft to the
  // shared config.
  //
  // The schedule buffers live in `st` (slot_state), NOT on the kernel stack. They used to be stack
  // locals, which was safe only because just lane 0 ran this code. Under the Direct model every lane
  // executes here, and a stack-local gpu_span would device-malloc PER LANE — 32 private buffers,
  // 31 leaked, and worse: warp_build_config publishes `lvl_*[o].data()` / `sp_*_v.data()` into the
  // config, so every wavefront solve for the rest of the solve would read a lane-private schedule.
  // slot_state already carries byte-identical members for exactly this reason (the setup
  // uses them), so this is a reuse, not a new allocation.

  // Serial O(m*nnz) schedule build that resizes the slot's gpu_spans and writes the config.
  // The trailing warp sync publishes the completed schedule and ft.warp_cfg_ to every lane.
  // CUOPT_BNB_GPU_REUSE_LEVELS=0: rebuild the schedules for child continuations too.
  const bool reuse_levels = child_continue && st.cfg_built && c_reuse_levels != 0;
  ft.path(reuse_levels ? P_LEVEL_REUSE : P_LEVEL_BUILD);
  if (!skip_tri) {
  if (!reuse_levels && c_warp_levels != 0) {
    warp_build_config_all(ft, cfg_shared, st.lvl_lev, st.lvl_ls, st.lvl_co, st.sp_ls_v, st.sp_co_v,
                          st.sp_pos_v, st.lvl_pos_v);
  }
  lane0_serial([&] {
    if (reuse_levels) {
      warp_wire_config(ft, cfg_shared, st.lvl_lev, st.lvl_ls, st.lvl_co, st.sp_ls_v, st.sp_co_v,
                       st.sp_pos_v, st.cfg_nlev);
    } else if (c_warp_levels != 0) {
      for (int o = 0; o < 4; ++o) { st.cfg_nlev[o] = cfg_shared.nlev[o]; }
      st.cfg_built = true;
    } else {
      warp_build_config(
        ft, cfg_shared, st.lvl_lev, st.lvl_ls, st.lvl_co, st.sp_ls_v, st.sp_co_v, st.sp_pos_v,
        st.lvl_pos_v);
      for (int o = 0; o < 4; ++o) { st.cfg_nlev[o] = cfg_shared.nlev[o]; }
      st.cfg_built = true;
    }
    ft.warp_cfg_ = &cfg_shared;
  });
  }
  lane0_serial([&] { st.f_lean = lean ? 1 : 0; });
  if (threadIdx.x == 0) {
    const long long pc3 = clock64();
    st.pre[0] = pcb - pc0, st.pre[1] = pcs - pcb, st.pre[2] = pc1 - pcs;
    st.pre[3] = pc2 - pc1, st.pre[4] = pc3 - pc2;
  }

  i_t refactor_reason   = 0;
  dual::status_t status =
    gpu_simplified_dual_phase2_device<FT, AllowRefactorExit, PersistentSeats, FissionSetup>(lp,
                                                             settings,
                                                             a.vstatus,
                                                             st,
                                                             a.basic_list,
                                                             a.nonbasic_list,
                                                             sol,
                                                             a.dy_se,
                                                             a.refactor_resume != 0,
                                                             child_continue,
                                                             a.terminal_certificate_retry != 0,
                                                            a.resume_obj,
                                                            a.resume_iter,
                                                            refactor_reason);

  if constexpr (FissionSetup) { return; }
  // `status` is warp-uniform (every terminal condition in the driver is), so these are same-value
  // stores from every lane — but keep a single writer.
  lane0_serial([&] {
    a.out_status[0]       = static_cast<i_t>(status);
    a.out_status[1]       = refactor_reason;  // see gpu_simplified_dual_phase2_device
    *a.out_objective      = sol.objective;
    *a.out_user_objective = sol.user_objective;
    *a.out_iterations     = sol.iterations;
  });
}

struct seat_clock_t {
  unsigned long long t0, t1;  // globaltimer (ns) around the seat's solve
  long long cycles;           // clock64 difference (same SM)
  int iterations, ran;
  long long ph[kSolvePhases];  // pivot-loop phase cycles of this solve
  int nlev[4];                 // elimination levels of the L, U, L^T, U^T wavefront schedules
  unsigned route[3];           // this solve's dense / sparse / sparse->dense solves
  long long sub[8];            // solve sub-step cycles (see gpu_basis_update_t::sub_)
  long long pre[6];            // leg set-up cycles (see slot_state::pre)
  long long wop[4];            // dense wavefront cycles per operator
  unsigned wn[4];              // dense wavefront solves per operator
  unsigned path[kPaths];       // path counts (gpu_basis_update_t::cnt_)
};

__device__ __forceinline__ unsigned long long global_ns()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Device dives: buffers, and (in-kernel mode) the scoring inputs; records == 0 disables.
struct dive_ctx_t {
  int records{0}, pk{0};
  const f_t* root_x{nullptr};
  gpu_bound_delta<i_t, f_t>* deltas{nullptr};
  int* taken{nullptr};
  gpu_dive_record<i_t, f_t>* log{nullptr};
  std::uint8_t* packed{nullptr};
  gpu_scores::seat_score_t* scores{nullptr};
  const unsigned char* var_int{nullptr};
  gpu_scores::pseudocosts_t pc{};
  seat_clock_t* clk{nullptr};  // CUOPT_BNB_GPU_SEAT_CLOCK: [round][seat]
  int clk_seats{0};
  unsigned int* yield_count{nullptr};  // CUOPT_BNB_GPU_YIELD
  unsigned int yield_at{0};
};

// score_kernel's per-seat computation for one warp; every lane returns the seat's score.
__device__ __noinline__ gpu_scores::seat_score_t warp_seat_score(const f_t* x,
                                                                 const unsigned char* is_integer,
                                                                 i_t n,
                                                                 const gpu_scores::pseudocosts_t& pc)
{
  const int lane      = static_cast<int>(threadIdx.x % 32);
  constexpr double eps = 1e-6;
  int num_frac = 0, unreliable = 0;
  double best_score = -1.0, estimate = 0.0;
  int best_var      = INT_MAX;
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
  num_frac   = __shfl_sync(0xffffffffu, num_frac, 0);
  unreliable = __shfl_sync(0xffffffffu, unreliable, 0);
  estimate   = __shfl_sync(0xffffffffu, estimate, 0);
  best_var   = __shfl_sync(0xffffffffu, best_var, 0);
  return {num_frac, num_frac > 0 ? best_var : -1, unreliable, 0, estimate};
}

// Log a seat that solved OPTIMAL with reliable scores and continue it into the Martin child of
// its best column. False: the dive ends here.
__device__ __noinline__ bool dive_advance_seat(kernel_args& a,
                                               int seat,
                                               int round,
                                               const gpu_scores::seat_score_t& sc,
                                               const dive_ctx_t& D)
{
  const int lane = static_cast<int>(threadIdx.x % 32);
  bool go = round < D.records && a.out_status[0] == static_cast<i_t>(dual::status_t::OPTIMAL) &&
            sc.num_frac > 0 && sc.unreliable == 0 && sc.best_var >= 0;
  if (go) {
    // The host's child test: a FIXED column must have tight bounds.
    bool loose = false;
    for (i_t j = lane; j < a.n; j += 32) {
      loose |= a.vstatus[j] == variable_status_t::NONBASIC_FIXED && a.upper[j] - a.lower[j] >= a.fixed_tol;
    }
    go = !__any_sync(0xffffffffu, loose);
  }
  __syncwarp(0xffffffffu);
  if (!go) { return false; }
  const i_t n   = a.n;
  const i_t j   = sc.best_var;
  const f_t v   = a.x[j];
  const f_t rv  = D.root_x[j];
  const bool up = !(v - floor(rv) < ceil(rv) - v + 1e-6);
  std::uint8_t* out = D.packed + (static_cast<size_t>(seat) * D.records + round) * D.pk;
  for (int b = lane; b < D.pk; b += 32) {
    std::uint8_t byte = 0;
    for (int t = 0; t < 4 && 4 * b + t < n; ++t) {
      const auto s = a.vstatus[4 * b + t];
      const std::uint8_t code = s == variable_status_t::BASIC ? 0
                                : (s == variable_status_t::NONBASIC_LOWER ||
                                   s == variable_status_t::NONBASIC_FIXED) ? 1
                                : s == variable_status_t::NONBASIC_UPPER ? 2 : 3;
      byte |= code << (2 * t);
    }
    out[b] = byte;
  }
  __syncwarp(0xffffffffu);
  if (lane == 0) {
    D.log[static_cast<size_t>(seat) * D.records + round] = {*a.out_objective, sc.estimate, v, a.lower[j],
                                                            a.upper[j], *a.out_iterations, sc.num_frac, j,
                                                            up ? 1 : 0};
    D.deltas[seat] = up ? gpu_bound_delta<i_t, f_t>{j, ceil(v), a.upper[j], 1u}
                        : gpu_bound_delta<i_t, f_t>{j, a.lower[j], floor(v), 2u};
    a.command                    = static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE);
    a.bound_deltas               = D.deltas + seat;
    a.bound_delta_count          = 1;
    a.bounds_from_root           = 0;
    a.device_lu                  = 0;
    a.refactor_resume            = 0;
    a.terminal_certificate_retry = 0;
    D.taken[seat]                = round + 1;
  }
  __syncwarp(0xffffffffu);
  return true;
}

// ---- entry points -----------------------------------------------------------------------------
// Two wrappers over ONE body, because __launch_bounds__ cannot be applied conditionally on a
// template parameter and it must NOT reach the single-thread kernel. Measured 2026-08-11: putting
// `__launch_bounds__(32, 8)` on the <<<N,1>>> instantiation takes it from 129 to 188 registers
// (15 -> 10 blocks/SM) — the classic "the bound says there is no occupancy pressure, so ptxas spends
// registers freely" effect that motivated removing launch bounds in the first place.

// Direct warp entry, register-capped. ptxas left unconstrained picks 182 regs / 11 blocks/SM, but a
// cap sweep showed that is a heuristic, not a requirement: 168 regs at ZERO spill, 128 regs at 8 BYTES
// of spill for 16 blocks/SM (1.45x occupancy). MinBlocks maps to the cap as 65536/(32*MinBlocks):
// merely ATTACHING a launch bound changes ptxas's behaviour — see the single-thread note below.
// Uncapped Direct warp entry: no attribute at all. NOT the same as __launch_bounds__ with a large
// MinBlocks — the attribute's mere presence tells ptxas there is no occupancy pressure, and it then
// spends registers freely (measured: 214 regs with the bound at minBlocks=1-equivalent, vs 182 with
// no attribute; and 129 -> 188 on the single-thread kernel, which is why the two entry points exist).
// Keep the cleanup's measured 128-register occupancy target for both factor-index specializations.
// Keep only the resident-block budget in flight. Each block dequeues another independent problem
// after finishing its current one, reducing tail imbalance without sharing any per-problem state.
template <bool AllowRefactorExit = false,
          bool PersistentSeats   = false>
__global__ void __launch_bounds__(32, 16)
// `next_problem` is PER LAUNCH, not a device global. Branch-and-bound runs one batch per GPU
// worker, concurrently, on separate streams; a single global counter would have those launches
// stealing from each other's index space and resetting it under each other mid-flight.
gpu_phase2_kernel_direct(kernel_args* args,
                         i_t n_problems,
                         unsigned int* next_problem,
                         unsigned int* completed_blocks,
                         std::uint32_t* completion_flag,
                         dive_ctx_t dive = dive_ctx_t{})
{
  __shared__ i_t problem_idx;
  while (true) {
    if (threadIdx.x == 0) {
      problem_idx = static_cast<i_t>(atomicAdd(next_problem, 1u));
    }
    __syncwarp(0xffffffffu);
    if (problem_idx >= n_problems) { break; }
    // One body call site (each site inlines the whole solver). In-kernel dives loop here: score
    // after every solve and continue the seat without leaving the block.
    if constexpr (PersistentSeats) {
      if (dive.scores != nullptr && threadIdx.x == 0) { dive.taken[problem_idx] = 0; }
    }
    for (int round = 0;; ++round) {
      seat_clock_t* clk     = nullptr;
      bool ran              = false;
      unsigned long long t0 = 0;
      long long c0          = 0;
      i_t leg0              = 0;   // a resumed leg's iterations start at resume_iter
      if constexpr (PersistentSeats) {
        if (dive.clk != nullptr) {
          clk = dive.clk + static_cast<size_t>(round) * dive.clk_seats + problem_idx;
          ran = args[problem_idx].done == 0;
          t0  = global_ns();
          c0  = clock64();
          leg0 = ran && args[problem_idx].refactor_resume != 0 ? args[problem_idx].resume_iter : 0;
        }
      }
      gpu_phase2_kernel_body<AllowRefactorExit, PersistentSeats>(
        args, problem_idx, dive.yield_count, dive.yield_at);
      if constexpr (!PersistentSeats) {
        break;
      } else {
        if (clk != nullptr) {
          __syncwarp(0xffffffffu);
          const long long c1          = clock64();
          const unsigned long long t1 = global_ns();
          if (threadIdx.x == 0) {
            *clk = ran ? seat_clock_t{t0, t1, c1 - c0, *args[problem_idx].out_iterations - leg0, 1, {}, {}, {}, {}, {}, {}, {}}
                       : seat_clock_t{0, 0, 0, 0, 0, {}, {}, {}, {}, {}, {}, {}};
            if (ran && args[problem_idx].slot != nullptr) {
              for (int k = 0; k < kSolvePhases; ++k) { clk->ph[k] = args[problem_idx].slot->ph[k]; }
              for (int k = 0; k < 6; ++k) { clk->pre[k] = args[problem_idx].slot->pre[k]; }
              const auto* wc = args[problem_idx].slot->ft.warp_cfg_;
              for (int k = 0; k < 4; ++k) { clk->nlev[k] = wc != nullptr ? static_cast<int>(wc->nlev[k]) : -1; }
              auto& rt = args[problem_idx].slot->ft.route_;
              for (int k = 0; k < 3; ++k) {
                clk->route[k] = rt[k];
                rt[k]         = 0u;
              }
              auto& wo = args[problem_idx].slot->ft.wop_;
              auto& wc2 = args[problem_idx].slot->ft.wn_;
              for (int k = 0; k < 4; ++k) {
                clk->wop[k] = wo[k], clk->wn[k] = wc2[k];
                wo[k] = 0, wc2[k] = 0u;
              }
              auto& sb = args[problem_idx].slot->ft.sub_;
              for (int k = 0; k < 8; ++k) {
                clk->sub[k] = sb[k];
                sb[k]       = 0;
              }
              auto& pc = args[problem_idx].slot->ft.cnt_;
              for (int k = 0; k < kPaths; ++k) {
                clk->path[k] = pc[k];
                pc[k]        = 0u;
              }
            }
          }
        }
        if (dive.scores == nullptr) { break; }
        kernel_args& a = args[problem_idx];
        if (a.done != 0) { break; }
        if (dive.yield_count != nullptr) {
          unsigned int done = 0;
          if (threadIdx.x == 0) { done = *reinterpret_cast<volatile unsigned int*>(dive.yield_count); }
          if (__shfl_sync(0xffffffffu, done, 0) >= dive.yield_at) { break; }
        }
        const auto sc = warp_seat_score(a.x, dive.var_int, a.n, dive.pc);
        if (threadIdx.x == 0) { dive.scores[problem_idx] = sc; }
        __syncwarp(0xffffffffu);
        if (!dive_advance_seat(a, problem_idx, round, sc, dive)) { break; }
      }
    }
    if (dive.yield_count != nullptr && threadIdx.x == 0) { atomicAdd(dive.yield_count, 1u); }
  }

  // Every block publishes all result writes before announcing that it has drained its share of
  // the queue. The last block performs a system-scope release store into mapped pinned memory;
  // branch-and-bound polls the same word with an acquire load, avoiding cudaEventQuery in its hot
  // worker loop. The event recorded after this kernel remains the authoritative blocking/timing
  // primitive for harvest and destruction.
  if (completed_blocks != nullptr && completion_flag != nullptr) {
    // Every lane can write its slice of the solve vectors, so every lane must publish before lane
    // zero counts the block as complete.
    __threadfence_system();
    __syncwarp(0xffffffffu);
  }
  if (threadIdx.x == 0 && completed_blocks != nullptr && completion_flag != nullptr) {
    const unsigned int previously_completed = atomicAdd(completed_blocks, 1u);
    if (previously_completed + 1u == static_cast<unsigned int>(gridDim.x)) {
      *next_problem     = 0u;  // ready for the next launch without memsets
      *completed_blocks = 0u;
      if (dive.yield_count != nullptr) { *dive.yield_count = 0u; }
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> ready(*completion_flag);
      ready.store(1u, cuda::memory_order_release);
    }
  }
}

__global__ void set_flag_kernel(std::uint32_t* flag)
{
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> ready(*flag);
  ready.store(1u, cuda::memory_order_release);
}

#include "fission.cuh"

struct compact_hdr_t {
  double x, lower, upper;  // branch column (kind 1)
  int kind;                // 0 not run, 1 branch, 2 status only, 3 full LP state in `slot`
  int slot;
};

// Device dive step after a solve round (separate-kernel mode): seats that do not continue get done = 2.
__global__ void dive_advance_kernel(kernel_args* args,
                                    int count,
                                    int round,
                                    const gpu_scores::seat_score_t* scores,
                                    dive_ctx_t D,
                                    unsigned int* next_problem)
{
  const int seat = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (blockIdx.x == 0 && threadIdx.x == 0) { *next_problem = 0u; }
  if (seat >= count) { return; }
  kernel_args& a = args[seat];
  if (round == 0 && lane == 0) { D.taken[seat] = 0; }
  if (a.done != 0) { return; }
  if (!dive_advance_seat(a, seat, round, scores[seat], D) && lane == 0) { a.done = 2; }
}

__device__ __forceinline__ std::uint8_t pack_code(variable_status_t s)
{
  return s == variable_status_t::BASIC ? 0
         : (s == variable_status_t::NONBASIC_LOWER || s == variable_status_t::NONBASIC_FIXED) ? 1
         : s == variable_status_t::NONBASIC_UPPER ? 2 : 3;
}

// Compact harvest: classify every seat that ran and gather what the host reads from it.
// LP state: x | lower | upper | z | dy_se (n doubles each) | y (m) | basic (m) | nonbasic (n-m) | vstatus.
__global__ void compact_gather_kernel(const kernel_args* args,
                                      int count,
                                      int pk,
                                      size_t lp_state_bytes,
                                      const gpu_scores::seat_score_t* scores,
                                      compact_hdr_t* hdr,
                                      int* full_count,
                                      std::uint8_t* packed,
                                      char* lp_states)
{
  const int seat = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (seat >= count) { return; }
  const kernel_args& a = args[seat];
  if (a.done == 1) {
    if (lane == 0) { hdr[seat] = {0.0, 0.0, 0.0, 0, -1}; }
    return;
  }
  const i_t m = a.m, n = a.n;
  const auto status = static_cast<dual::status_t>(a.out_status[0]);
  const auto& sc    = scores[seat];
  int kind          = 3;
  if (status == dual::status_t::OPTIMAL) {
    if (sc.num_frac > 0 && sc.unreliable == 0 && sc.best_var >= 0) {
      bool loose = false;
      for (i_t j = lane; j < n; j += 32) {
        loose |= a.vstatus[j] == variable_status_t::NONBASIC_FIXED && a.upper[j] - a.lower[j] >= a.fixed_tol;
      }
      kind = __any_sync(0xffffffffu, loose) ? 3 : 1;
    }
  } else if (status == dual::status_t::CUTOFF || status == dual::status_t::DUAL_UNBOUNDED) {
    kind = 2;
  }
  if (kind == 1) {
    std::uint8_t* out = packed + static_cast<size_t>(seat) * pk;
    for (int b = lane; b < pk; b += 32) {
      std::uint8_t byte = 0;
      for (int t = 0; t < 4 && 4 * b + t < n; ++t) { byte |= pack_code(a.vstatus[4 * b + t]) << (2 * t); }
      out[b] = byte;
    }
    if (lane == 0) {
      const i_t j = sc.best_var;
      hdr[seat]   = {a.x[j], a.lower[j], a.upper[j], 1, -1};
    }
    return;
  }
  if (kind == 2) {
    if (lane == 0) { hdr[seat] = {0.0, 0.0, 0.0, 2, -1}; }
    return;
  }
  int slot = 0;
  if (lane == 0) { slot = atomicAdd(full_count, 1); }
  slot    = __shfl_sync(0xffffffffu, slot, 0);
  char* r = lp_states + static_cast<size_t>(slot) * lp_state_bytes;
  f_t* d  = reinterpret_cast<f_t*>(r);
  for (i_t j = lane; j < n; j += 32) {
    d[j]         = a.x[j];
    d[n + j]     = a.lower[j];
    d[2 * n + j] = a.upper[j];
    d[3 * n + j] = a.z[j];
    d[4 * n + j] = a.dy_se[j];
  }
  for (i_t i = lane; i < m; i += 32) { d[5 * n + i] = a.y[i]; }
  i_t* li = reinterpret_cast<i_t*>(d + 5 * n + m);
  for (i_t i = lane; i < m; i += 32) { li[i] = a.basic_list[i]; }
  for (i_t k = lane; k < n - m; k += 32) { li[m + k] = a.nonbasic_list[k]; }
  auto* vs = reinterpret_cast<variable_status_t*>(li + n);
  for (i_t j = lane; j < n; j += 32) { vs[j] = a.vstatus[j]; }
  if (lane == 0) { hdr[seat] = {0.0, 0.0, 0.0, 3, slot}; }
}

// ===========================================================================================

// ---------------------------------------------------------------------------
// Host-side helpers for marshaling.
// ---------------------------------------------------------------------------
// HARDWARE MEMORY COMPRESSION (CU_MEM_ALLOCATION_COMP_GENERIC via the VMM API) was built here and
// REJECTED, 2026-08-26. B200 supports it and it is transparent (results bit-identical), but the cost
// scaled with coverage: 9% of the working set compressed cost +9%, 100% cost +47%. It buys DRAM
// bandwidth, and this kernel is latency-bound, so it pays on the path it is limited by. At 100
// problems the penalty was only +0.7% -- do not re-derive this from a small run.
//
// Unrelated finding left by that work, worth a separate look: this marshaller actually requests
// 85.6 GiB over 525,500 allocations (~5.85 MB/problem) at 15,000 problems, while the pre-flight
// reports `static 52.58 GB` (~3.5 MB/problem) -- a 1.75x under-count in the VRAM budget.
// Marshalling storage. Every per-problem input buffer is an rmm::device_buffer allocated from the
// CURRENT device resource -- the same convention as the rest of cuOpt. Installing and sizing the
// pool is the APPLICATION's job (rmm::mr::set_current_device_resource; see cuopt's grpc_worker, and
// test_netlib for the benchmark). This file deliberately does not install a resource of its own.
//
// Why it matters: the raw path cost 41,300 cudaMallocs (0.234 s) plus ~42,000 cudaFrees per
// 1,200-problem batch -- call-count bound, not bytes bound. device_buffer is also what makes pooling
// possible at all, since RMM's deallocate needs the original size a bare pointer does not carry.
// Owns every device_buffer handed out for one problem. Destroying it returns the memory to the pool,
// so free_kernel_args no longer frees these pointers.
// Pieces packed into one pinned array, uploaded in one copy, scattered by cub::DeviceMemcpy::Batched.
// CUOPT_BNB_GPU_HUGE_ARENA: the pool's pinned host buffers come from memory backed by 2 MB pages and
// registered with cudaHostRegister (1 GiB: ~130 ms vs ~810 ms for cudaHostAlloc on 4 KB pages). One
// arena per handle holds the buffers whose worst-case size follows from the problem (create_huge_arena);
// any other pinned buffer gets its own 2 MB-page mapping. =0: CCCL pinned pool / cudaHostAlloc.
inline bool gpu_huge_arena()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_HUGE_ARENA");
    return e == nullptr || std::string(e) != "0";
  }();
  return on || pinned_host::reserved();   // CUOPT_BNB_GPU_PINNED_ARENA reserves one before pools start
}

using hp_mapping_t = pinned_host::mapping_t;
inline std::atomic<std::uint64_t> g_hp_maps{0}, g_hp_bytes{0}, g_arena_n{0}, g_arena_bytes{0};
inline hp_mapping_t hp_map(size_t bytes) { return pinned_host::map(bytes); }
inline void hp_unmap(const hp_mapping_t& m) { pinned_host::unmap(m); }

// Arena ranges, so a result-slab deallocation can tell an arena slice (no-op) from its own mapping.
struct hp_registry_t {
  std::mutex mu;
  std::vector<std::pair<const char*, size_t>> ranges;
  bool contains(const void* p)
  {
    std::lock_guard<std::mutex> lk(mu);
    for (auto& [a, n] : ranges) {
      if (static_cast<const char*>(p) >= a && static_cast<const char*>(p) < a + n) { return true; }
    }
    return false;
  }
};
inline hp_registry_t g_hp_registry;

// Bump allocation over either its own mapping (CUOPT_BNB_GPU_HUGE_ARENA) or a slice of the process
// arena (CUOPT_BNB_GPU_PINNED_ARENA), which goes back to the free list with the handle.
struct pinned_arena_t {
  hp_mapping_t map;
  int slice{-1};
  int node{-1};   // NUMA node of the slice's region (per-node process arena)
  char* base{nullptr};
  size_t cap{0}, used{0};
  explicit pinned_arena_t(size_t bytes) : map(hp_map(bytes))
  {
    base = static_cast<char*>(map.p);
    cap  = map.bytes;
    std::lock_guard<std::mutex> lk(g_hp_registry.mu);
    g_hp_registry.ranges.push_back({base, cap});
    g_arena_n.fetch_add(1);
    g_arena_bytes.fetch_add(map.bytes);
  }
  pinned_arena_t(int index, int node_, void* p, size_t bytes)
    : slice(index), node(node_), base(static_cast<char*>(p)), cap(bytes)
  {
  }
  ~pinned_arena_t()
  {
    if (slice >= 0) {
      pinned_host::release_slice(slice);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(g_hp_registry.mu);
      auto& r = g_hp_registry.ranges;
      r.erase(std::remove_if(r.begin(), r.end(), [&](auto& x) { return x.first == base; }), r.end());
    }
    hp_unmap(map);
  }
  pinned_arena_t(const pinned_arena_t&)            = delete;
  pinned_arena_t& operator=(const pinned_arena_t&) = delete;
  void* take(size_t bytes)
  {
    const size_t off = (used + 255) / 256 * 256;
    if (off + bytes > cap) { return nullptr; }
    used = off + bytes;
    return base + off;
  }
};
// Result slabs (result_allocator_t) take from this arena while it is set on the thread.
inline thread_local pinned_arena_t* tl_pinned_arena = nullptr;

// A pinned host buffer: a CCCL pinned-pool buffer (switch off), an arena slice, or its own mapping.
template <typename T>
class pinned_buf {
 public:
  pinned_buf() = default;
  pinned_buf(pinned_buf&& o) noexcept : legacy_(std::move(o.legacy_)), p_(o.p_), own_(o.own_)
  {
    o.p_   = nullptr;
    o.own_ = {};
  }
  pinned_buf& operator=(pinned_buf&& o) noexcept
  {
    if (this != &o) {
      hp_unmap(own_);
      legacy_ = std::move(o.legacy_);
      p_      = o.p_;
      own_    = o.own_;
      o.p_    = nullptr;
      o.own_  = {};
    }
    return *this;
  }
  pinned_buf(const pinned_buf&)            = delete;
  pinned_buf& operator=(const pinned_buf&) = delete;
  ~pinned_buf() { hp_unmap(own_); }
  T* data() { return legacy_ ? legacy_->data() : p_; }
  const T* data() const { return legacy_ ? legacy_->data() : p_; }

  template <typename U>
  friend pinned_buf<U> pinned_make(pinned_arena_t* arena, cuda::stream_ref s, size_t n);

 private:
  std::optional<cuda::host_buffer<T>> legacy_;
  T* p_{nullptr};
  hp_mapping_t own_;
};

template <typename T>
pinned_buf<T> pinned_make(pinned_arena_t* arena, cuda::stream_ref s, size_t n)
{
  pinned_buf<T> b;
  if (!gpu_huge_arena()) {
    b.legacy_.emplace(cuda::make_pinned_buffer<T>(s, n, cuda::no_init));
    return b;
  }
  if (arena != nullptr) {
    if (void* p = arena->take(n * sizeof(T))) {
      b.p_ = static_cast<T*>(p);
      return b;
    }
  }
  if (void* p = pinned_host::overflow_take(n * sizeof(T), arena != nullptr ? arena->node : -1)) {
    b.p_ = static_cast<T*>(p);
    return b;
  }
  b.own_ = hp_map(n * sizeof(T));
  b.p_   = static_cast<T*>(b.own_.p);
  g_hp_maps.fetch_add(1);
  g_hp_bytes.fetch_add(b.own_.bytes);
  return b;
}

struct packed_upload_t {
  std::optional<pinned_buf<unsigned char>> h;
  std::vector<unsigned char> hp;  // pageable staging (first launch, CUOPT_BNB_GPU_PAGEABLE_SETUP)
  bool pageable{false};
  unsigned char* hdata() { return pageable ? hp.data() : h->data(); }
  void to_pinned()
  {
    pageable = false;
    std::vector<unsigned char>().swap(hp);
    cap = used = 0;
  }
  std::optional<rmm::device_buffer> d, temp;
  size_t cap{0}, dev_cap{0}, temp_cap{0}, used{0};
  std::vector<size_t> src;
  std::vector<void*> dst;
  std::vector<size_t> bytes;
  std::uint64_t flushes{0}, flushed_bytes{0}, flushed_pieces{0}, grows{0};

  static size_t align(size_t v) { return (v + 15) & ~size_t{15}; }
  void reserve(size_t need, rmm::cuda_stream_view s)
  {
    if (need <= cap) { return; }
    const size_t c = std::max(need, 2 * cap);
    if (pageable) {
      hp.resize(c);
      cap = c;
      ++grows;
      return;
    }
    auto grown = pinned_make<unsigned char>(nullptr, cuda::stream_ref{s.value()}, c);
    if (used != 0) { std::memcpy(grown.data(), h->data(), used); }
    h.emplace(std::move(grown));
    cap = c;
    ++grows;
  }
  void add(void* to, const void* from, size_t n, rmm::cuda_stream_view s)
  {
    if (n == 0) { return; }
    reserve(align(used) + n, s);
    used = align(used);
    std::memcpy(hdata() + used, from, n);
    src.push_back(used);
    dst.push_back(to);
    bytes.push_back(n);
    used += n;
  }
  // `mid` (optional) is recorded between the host-to-device copy and the batched scatter.
  void flush(rmm::cuda_stream_view s, cudaEvent_t mid = nullptr)
  {
    const size_t P = src.size();
    if (P == 0) {
      if (mid != nullptr) { RAFT_CUDA_TRY(cudaEventRecord(mid, s.value())); }
      return;
    }
    const size_t desc  = align(used);
    const size_t total = desc + P * (2 * sizeof(void*) + sizeof(size_t));
    reserve(total, s);
    if (total > dev_cap) {
      d.emplace(cap, s);
      dev_cap = cap;
    }
    auto* base = static_cast<unsigned char*>(d->data());
    auto** hs  = reinterpret_cast<void**>(hdata() + desc);
    auto** hd  = hs + P;
    auto* hn   = reinterpret_cast<size_t*>(hd + P);
    for (size_t k = 0; k < P; ++k) {
      hs[k] = base + src[k];
      hd[k] = dst[k];
      hn[k] = bytes[k];
    }
    RAFT_CUDA_TRY(cudaMemcpyAsync(base, hdata(), total, cudaMemcpyHostToDevice, s.value()));
    if (mid != nullptr) { RAFT_CUDA_TRY(cudaEventRecord(mid, s.value())); }
    auto** ds = reinterpret_cast<void**>(base + desc);
    auto* dn  = reinterpret_cast<size_t*>(ds + 2 * P);
    size_t temp_bytes = 0;
    RAFT_CUDA_TRY(cub::DeviceMemcpy::Batched(
      nullptr, temp_bytes, ds, ds + P, dn, static_cast<::cuda::std::int64_t>(P), s.value()));
    if (temp_bytes > temp_cap) {
      temp.emplace(std::max(temp_bytes, 2 * temp_cap), s);
      temp_cap = temp->size();
    }
    RAFT_CUDA_TRY(cub::DeviceMemcpy::Batched(
      temp->data(), temp_bytes, ds, ds + P, dn, static_cast<::cuda::std::int64_t>(P), s.value()));
    ++flushes;
    flushed_bytes += used;
    flushed_pieces += P;
    src.clear();
    dst.clear();
    bytes.clear();
    used = 0;
  }
};

// Fixed per-seat factor slots, sized from the first launch; a larger LU goes to the CPU.
struct factor_slots_t {
  size_t capL{0}, capU{0}, acceptL{0}, acceptU{0}, bound{0};
  i_t *Li{}, *Ui{}, *Lti{}, *Uti{};
  f_t *Lx{}, *Ux{}, *Ltx{}, *Utx{};
  // Fill of every LU checked against the slots, as nz / cap: 10 bins of 0.1, then > cap.
  std::array<std::uint64_t, 11> histL{}, histU{};
  std::uint64_t lus{0}, overflow_fresh{0}, overflow_resume{0};
  size_t maxL{0}, maxU{0};

  void record(size_t nzL, size_t nzU)
  {
    const auto bin = [](size_t nz, size_t c) {
      return nz > c ? size_t{10} : std::min<size_t>(9, (10 * nz) / std::max<size_t>(c, 1));
    };
    ++histL[bin(nzL, capL)];
    ++histU[bin(nzU, capU)];
    ++lus;
    maxL = std::max(maxL, nzL);
    maxU = std::max(maxU, nzU);
  }
};

struct marshal_storage {
  std::vector<rmm::device_buffer> bufs;
  // The CSR transpose is built here and its host storage must outlive marshal_problem.
  std::vector<cuopt::mathematical_optimization::csr_matrix_t<i_t, f_t>> arow_keep;
  // Every allocation and copy this storage makes is ordered on this stream. It is explicit rather
  // than the default stream because branch-and-bound runs one batch per GPU worker concurrently:
  // on the default stream those batches would serialise against each other.
  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  explicit marshal_storage(rmm::cuda_stream_view s = rmm::cuda_stream_default) : stream(s)
  {
    bufs.reserve(111);  // The single-problem path still uses per-field buffers.
  }
};

// One typed device vector per fixed-capacity field. Only reach_mark rounds its byte-sized seat
// storage up to 32-bit words, because the warp reach path accesses it through a word pointer.
struct fixed_storage {
  rmm::device_uvector<f_t> lower;
  rmm::device_uvector<f_t> upper;
  rmm::device_uvector<i_t> L0_col_start;
  rmm::device_uvector<i_t> U0_col_start;
  rmm::device_uvector<i_t> L0t_col_start;
  rmm::device_uvector<i_t> U0t_col_start;
  rmm::device_uvector<i_t> row_perm;
  rmm::device_uvector<i_t> inv_row_perm;
  rmm::device_uvector<i_t> xi_workspace;
  rmm::device_uvector<f_t> x_workspace;
  rmm::device_uvector<reach_mark_word> reach_mark;
  rmm::device_uvector<variable_status_t> vstatus;
  rmm::device_uvector<i_t> basic_list;
  rmm::device_uvector<i_t> nonbasic_list;
  rmm::device_uvector<f_t> x;
  rmm::device_uvector<f_t> y;
  rmm::device_uvector<f_t> z;
  rmm::device_uvector<f_t> dy_se;
  rmm::device_uvector<i_t> out_status;
  rmm::device_uvector<f_t> out_objective;
  rmm::device_uvector<f_t> out_user_objective;
  rmm::device_uvector<i_t> out_iterations;
  rmm::device_uvector<slot_state<gpu_basis_update_t>> slot;
  rmm::device_uvector<f_t> xB_workspace;
  rmm::device_uvector<f_t> delta_y;
  rmm::device_uvector<f_t> delta_z;
  rmm::device_uvector<f_t> delta_x;
  rmm::device_uvector<f_t> v;
  rmm::device_uvector<f_t> squared_infeas;
  rmm::device_uvector<f_t> setup_rhs;
  rmm::device_uvector<i_t> basic_mark;
  rmm::device_uvector<i_t> delta_z_mark;
  rmm::device_uvector<i_t> delta_z_indices;
  rmm::device_uvector<i_t> nonbasic_pos;
  rmm::device_uvector<i_t> infeas_idx;
  rmm::device_uvector<f_t> ft_dense_scratch;
  rmm::device_uvector<f_t> ft_dense_scratch2;
  rmm::device_uvector<f_t> ft_sort_bucket;
  rmm::device_uvector<f_t> ft_sort_x;
  rmm::device_uvector<i_t> ft_sort_i;
  rmm::device_uvector<i_t> ft_sort_perm;
  rmm::device_uvector<i_t> lvl_lev0;
  rmm::device_uvector<i_t> lvl_ls0;
  rmm::device_uvector<i_t> lvl_co0;
  rmm::device_uvector<i_t> lvl_lev1;
  rmm::device_uvector<i_t> lvl_ls1;
  rmm::device_uvector<i_t> lvl_co1;
  rmm::device_uvector<i_t> lvl_lev2;
  rmm::device_uvector<i_t> lvl_ls2;
  rmm::device_uvector<i_t> lvl_co2;
  rmm::device_uvector<i_t> lvl_lev3;
  rmm::device_uvector<i_t> lvl_ls3;
  rmm::device_uvector<i_t> lvl_co3;
  rmm::device_uvector<i_t> sp_ls_v;
  rmm::device_uvector<i_t> sp_co_v;
  rmm::device_uvector<i_t> sp_pos_v;
  rmm::device_uvector<i_t> lvl_pos_v;
  rmm::device_uvector<f_t> ft_recover_rhs;
  rmm::device_uvector<f_t> ft_recover_sol;
  rmm::device_uvector<i_t> S_col_start;
  rmm::device_uvector<i_t> S_i;
  rmm::device_uvector<f_t> S_x;
  rmm::device_uvector<f_t> mu_values;
  rmm::device_uvector<i_t> delta_y_sparse_i;
  rmm::device_uvector<f_t> delta_y_sparse_x;
  rmm::device_uvector<i_t> UTsol_sparse_i;
  rmm::device_uvector<f_t> UTsol_sparse_x;
  rmm::device_uvector<i_t> utilde_sparse_i;
  rmm::device_uvector<f_t> utilde_sparse_x;
  rmm::device_uvector<i_t> scaled_dxB_sparse_i;
  rmm::device_uvector<f_t> scaled_dxB_sparse_x;
  rmm::device_uvector<i_t> rhs_sparse_i;
  rmm::device_uvector<f_t> rhs_sparse_x;
  rmm::device_uvector<i_t> v_sparse_i;
  rmm::device_uvector<f_t> v_sparse_x;
  rmm::device_uvector<i_t> se_dy_i;
  rmm::device_uvector<f_t> se_dy_x;
  rmm::device_uvector<i_t> se_UTsol_i;
  rmm::device_uvector<f_t> se_UTsol_x;
  rmm::device_uvector<i_t> se_ei_i;
  rmm::device_uvector<f_t> se_ei_x;
  rmm::device_uvector<i_t> ft_sparse_scratch_i;
  rmm::device_uvector<f_t> ft_sparse_scratch_x;
  rmm::device_uvector<i_t> ft_ei_scratch_i;
  rmm::device_uvector<f_t> ft_ei_scratch_x;

  std::vector<f_t> h_lower;
  std::vector<f_t> h_upper;
  std::vector<i_t> h_L0_col_start;
  std::vector<i_t> h_U0_col_start;
  std::vector<i_t> h_L0t_col_start;
  std::vector<i_t> h_U0t_col_start;
  std::vector<i_t> h_row_perm;
  std::vector<i_t> h_inv_row_perm;
  std::vector<variable_status_t> h_vstatus;
  std::vector<i_t> h_basic_list;
  std::vector<i_t> h_nonbasic_list;
  std::vector<f_t> h_x;
  std::vector<f_t> h_y;
  std::vector<f_t> h_z;
  std::vector<f_t> h_dy_se;
  struct upload_descriptor {
    void* device;
    const void* host;
    size_t bytes;
  };
  std::vector<upload_descriptor> uploads;
  size_t seats;
  rmm::cuda_stream_view stream;

  fixed_storage(size_t seat_count, size_t m, size_t n, size_t ucap,
                size_t eta, rmm::cuda_stream_view s)
    :
      // The batch is homogeneous, so every seat has the same field widths.
      lower(seat_count * (n), s),
      upper(seat_count * (n), s),
      L0_col_start(seat_count * (m + 1), s),
      U0_col_start(seat_count * (m + 1), s),
      L0t_col_start(seat_count * (m + 1), s),
      U0t_col_start(seat_count * (m + 1), s),
      row_perm(seat_count * (m), s),
      inv_row_perm(seat_count * (m), s),
      xi_workspace(seat_count * (2 * m), s),
      x_workspace(seat_count * (m), s),
      reach_mark(seat_count * static_cast<size_t>(reach_mark_storage_words(static_cast<i_t>(m))), s),
      vstatus(seat_count * (n), s),
      basic_list(seat_count * (m), s),
      nonbasic_list(seat_count * (n - m), s),
      x(seat_count * (n), s),
      y(seat_count * (m), s),
      z(seat_count * (n), s),
      dy_se(seat_count * (n), s),
      out_status(seat_count * (2), s),
      out_objective(seat_count * (1), s),
      out_user_objective(seat_count * (1), s),
      out_iterations(seat_count * (1), s),
      slot(seat_count * (1), s),
      xB_workspace(seat_count * (std::max(m, n)), s),
      delta_y(seat_count * (std::max(m, n)), s),
      delta_z(seat_count * (std::max(m, n)), s),
      delta_x(seat_count * (std::max(m, n)), s),
      v(seat_count * (std::max(m, n)), s),
      squared_infeas(seat_count * (std::max(m, n)), s),
      setup_rhs(seat_count * (std::max(m, n)), s),
      basic_mark(seat_count * (std::max(m, n)), s),
      delta_z_mark(seat_count * (std::max(m, n)), s),
      delta_z_indices(seat_count * (std::max(m, n)), s),
      nonbasic_pos(seat_count * (std::max(m, n)), s),
      infeas_idx(seat_count * (std::max(m, n)), s),
      ft_dense_scratch(seat_count * (std::max(m, n)), s),
      ft_dense_scratch2(seat_count * (std::max(m, n)), s),
      ft_sort_bucket(seat_count * (std::max(m, n)), s),
      ft_sort_x(seat_count * (std::max(m, n)), s),
      ft_sort_i(seat_count * (std::max(m, n)), s),
      ft_sort_perm(seat_count * (std::max(m, n)), s),
      lvl_lev0(seat_count * (m + 1), s),
      lvl_ls0(seat_count * (m + 1), s),
      lvl_co0(seat_count * (m + 1), s),
      lvl_lev1(seat_count * (m + 1), s),
      lvl_ls1(seat_count * (m + 1), s),
      lvl_co1(seat_count * (m + 1), s),
      lvl_lev2(seat_count * (m + 1), s),
      lvl_ls2(seat_count * (m + 1), s),
      lvl_co2(seat_count * (m + 1), s),
      lvl_lev3(seat_count * (m + 1), s),
      lvl_ls3(seat_count * (m + 1), s),
      lvl_co3(seat_count * (m + 1), s),
      sp_ls_v(seat_count * (m + 1), s),
      sp_co_v(seat_count * (m + 1), s),
      sp_pos_v(seat_count * (m + 1), s),
      lvl_pos_v(seat_count * (m + 1), s),
      ft_recover_rhs(seat_count * (m), s),
      ft_recover_sol(seat_count * (m), s),
      S_col_start(seat_count * (2 * ucap + 1), s),
      S_i(seat_count * (eta), s),
      S_x(seat_count * (eta), s),
      mu_values(seat_count * (ucap), s),
      delta_y_sparse_i(seat_count * (std::max(m, n)), s),
      delta_y_sparse_x(seat_count * (std::max(m, n)), s),
      UTsol_sparse_i(seat_count * (std::max(m, n)), s),
      UTsol_sparse_x(seat_count * (std::max(m, n)), s),
      utilde_sparse_i(seat_count * (std::max(m, n)), s),
      utilde_sparse_x(seat_count * (std::max(m, n)), s),
      scaled_dxB_sparse_i(seat_count * (std::max(m, n)), s),
      scaled_dxB_sparse_x(seat_count * (std::max(m, n)), s),
      rhs_sparse_i(seat_count * (std::max(m, n)), s),
      rhs_sparse_x(seat_count * (std::max(m, n)), s),
      v_sparse_i(seat_count * (std::max(m, n)), s),
      v_sparse_x(seat_count * (std::max(m, n)), s),
      se_dy_i(seat_count * (std::max(m, n)), s),
      se_dy_x(seat_count * (std::max(m, n)), s),
      se_UTsol_i(seat_count * (std::max(m, n)), s),
      se_UTsol_x(seat_count * (std::max(m, n)), s),
      se_ei_i(seat_count * (std::max(m, n)), s),
      se_ei_x(seat_count * (std::max(m, n)), s),
      ft_sparse_scratch_i(seat_count * (std::max(m, n)), s),
      ft_sparse_scratch_x(seat_count * (std::max(m, n)), s),
      ft_ei_scratch_i(seat_count * (std::max(m, n)), s),
      ft_ei_scratch_x(seat_count * (std::max(m, n)), s),
      h_lower(seat_count * (n)),
      h_upper(seat_count * (n)),
      h_L0_col_start(seat_count * (m + 1)),
      h_U0_col_start(seat_count * (m + 1)),
      h_L0t_col_start(seat_count * (m + 1)),
      h_U0t_col_start(seat_count * (m + 1)),
      h_row_perm(seat_count * (m)),
      h_inv_row_perm(seat_count * (m)),
      h_vstatus(seat_count * (n)),
      h_basic_list(seat_count * (m)),
      h_nonbasic_list(seat_count * (n - m)),
      h_x(seat_count * (n)),
      h_y(seat_count * (m)),
      h_z(seat_count * (n)),
      h_dy_se(seat_count * (n)),
      seats(seat_count),
      stream(s)
  {
    const auto zero = [s](auto& field) {
      RAFT_CUDA_TRY(cudaMemsetAsync(
        field.data(), 0, field.size() * sizeof(*field.data()), s.value()));
    };
    zero(xi_workspace);
    zero(x_workspace);
    zero(reach_mark);
    zero(out_status);
    zero(out_objective);
    zero(out_user_objective);
    zero(out_iterations);
    uploads = {
      {lower.data(), h_lower.data(), h_lower.size() * sizeof(f_t)},
      {upper.data(), h_upper.data(), h_upper.size() * sizeof(f_t)},
      {L0_col_start.data(), h_L0_col_start.data(), h_L0_col_start.size() * sizeof(i_t)},
      {U0_col_start.data(), h_U0_col_start.data(), h_U0_col_start.size() * sizeof(i_t)},
      {L0t_col_start.data(), h_L0t_col_start.data(), h_L0t_col_start.size() * sizeof(i_t)},
      {U0t_col_start.data(), h_U0t_col_start.data(), h_U0t_col_start.size() * sizeof(i_t)},
      {row_perm.data(), h_row_perm.data(), h_row_perm.size() * sizeof(i_t)},
      {inv_row_perm.data(), h_inv_row_perm.data(), h_inv_row_perm.size() * sizeof(i_t)},
      {vstatus.data(), h_vstatus.data(), h_vstatus.size() * sizeof(variable_status_t)},
      {basic_list.data(), h_basic_list.data(), h_basic_list.size() * sizeof(i_t)},
      {nonbasic_list.data(), h_nonbasic_list.data(), h_nonbasic_list.size() * sizeof(i_t)},
      {x.data(), h_x.data(), h_x.size() * sizeof(f_t)},
      {y.data(), h_y.data(), h_y.size() * sizeof(f_t)},
      {z.data(), h_z.data(), h_z.size() * sizeof(f_t)},
      {dy_se.data(), h_dy_se.data(), h_dy_se.size() * sizeof(f_t)},
    };
  }

  template <typename T>
  T* seat_ptr(rmm::device_uvector<T> fixed_storage::*member, size_t seat,
              size_t elements)
  {
    if (seat >= seats) { throw std::runtime_error("invalid fixed-storage seat"); }
    auto& field = this->*member;
    if (field.size() != seats * elements) {
      throw std::runtime_error("fixed field length changed within a batch");
    }
    return elements == 0 ? nullptr : field.data() + seat * elements;
  }

  template <typename T>
  T* stage(rmm::device_uvector<T> fixed_storage::*member,
           std::vector<T> fixed_storage::*host_member,
           size_t seat, const T* src, size_t count)
  {
    T* dst = seat_ptr(member, seat, count);
    auto& host = this->*host_member;
    if (host.size() != seats * count) {
      throw std::runtime_error("fixed upload length changed within a batch");
    }
    if (count != 0) { std::memcpy(host.data() + seat * count, src, count * sizeof(T)); }
    return dst;
  }

  void flush_uploads()
  {
    raft::common::nvtx::range nvtx_scope("GpuDS::flush_uploads");
    for (const auto& upload : uploads) {
      if (upload.bytes == 0) { continue; }
      RAFT_CUDA_TRY(cudaMemcpyAsync(upload.device, upload.host, upload.bytes,
                                    cudaMemcpyHostToDevice, stream.value()));
    }
  }
};

// Setup telemetry (gpu_batch_set_setup_telemetry): host wall time inside each class of CUDA call
// made while marshalling. Thread-local because every branch-and-bound worker marshals its own pool
// on its own thread; the launch resets it before marshalling and copies it into gpu_batch_timing.
static std::atomic<bool> g_setup_telemetry{false};
struct setup_stats_t {
  bool on{false};
  double alloc_s{0.0}, h2d_s{0.0}, memset_s{0.0};
  size_t alloc_n{0}, h2d_n{0}, h2d_bytes{0}, memset_n{0};
};
static thread_local setup_stats_t tl_setup;

// Allocate one device_buffer in `st`, timed as an allocation.
static void* setup_alloc(size_t bytes, marshal_storage& st)
{
  const double t0 = tl_setup.on ? tic() : 0.0;
  st.bufs.emplace_back(bytes, st.stream);
  if (tl_setup.on) {
    tl_setup.alloc_s += toc(t0);
    ++tl_setup.alloc_n;
  }
  return st.bufs.back().data();
}

// Host-to-device copy on `st.stream`, timed as an upload. The same call rmm::device_buffer's
// copying constructor makes, split out so the allocation and the copy are measured separately.
static void setup_h2d(void* dst, const void* src, size_t bytes, marshal_storage& st)
{
  const double t0 = tl_setup.on ? tic() : 0.0;
  RAFT_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, st.stream.value()));
  if (tl_setup.on) {
    tl_setup.h2d_s += toc(t0);
    ++tl_setup.h2d_n;
    tl_setup.h2d_bytes += bytes;
  }
}

template <typename T>
static T* upload(const std::vector<T>& v, marshal_storage& st)
{
  const size_t bytes = (v.empty() ? 1 : v.size()) * sizeof(T);
  void* d            = setup_alloc(bytes, st);
  if (!v.empty()) { setup_h2d(d, v.data(), bytes, st); }
  return static_cast<T*>(d);
}

template <typename T>
static T* upload_n(const T* p, size_t count, marshal_storage& st)
{
  const size_t bytes = (count == 0 ? 1 : count) * sizeof(T);
  void* d            = setup_alloc(bytes, st);
  if (count != 0) { setup_h2d(d, p, bytes, st); }
  return static_cast<T*>(d);
}

// Uninitialised: every per-slot buffer is resize/assign-filled before it is read, and zeroing all
// 66 of them per problem would be a multi-GB memset across a large batch.
template <typename T>
static T* alloc_uninit(size_t count, marshal_storage& st)
{
  const size_t bytes = (count == 0 ? 1 : count) * sizeof(T);
  void* d            = setup_alloc(bytes, st);
  // DIAGNOSTIC (CUOPT_GPU_ZERO_WORKSPACE=1). These buffers are left uninitialised on the premise
  // that the kernel writes every element before reading it. If that premise is wrong anywhere, the
  // result depends on whatever the RMM pool last left in that memory -- so the same problem, with
  // bit-identical inputs, answers differently depending on which solve ran before it. Zeroing them
  // makes the pool's history irrelevant: a divergence that disappears under this flag is an
  // uninitialised read, not an input difference.
  static const bool zero_ws = [] {
    const char* e = std::getenv("CUOPT_GPU_ZERO_WORKSPACE");
    return e != nullptr && std::string(e) != "0";
  }();
  if (zero_ws) { RAFT_CUDA_TRY(cudaMemsetAsync(d, 0, bytes, st.stream.value())); }
  return static_cast<T*>(d);
}

template <typename T>
static T* alloc_zeroed(size_t count, marshal_storage& st)
{
  const size_t bytes = (count == 0 ? 1 : count) * sizeof(T);
  void* d            = setup_alloc(bytes, st);
  // The STORAGE's stream, never the legacy default one. The buffer was just allocated on
  // `st.stream`, so zeroing it anywhere else is unordered against that allocation; and
  // branch-and-bound runs one batch per worker concurrently, which the default stream would
  // serialise. It also breaks any CUDA graph capture live on another thread.
  const double t0 = tl_setup.on ? tic() : 0.0;
  RAFT_CUDA_TRY(cudaMemsetAsync(d, 0, bytes, st.stream.value()));
  if (tl_setup.on) {
    tl_setup.memset_s += toc(t0);
    ++tl_setup.memset_n;
  }
  return static_cast<T*>(d);
}

template <typename T>
static T* upload_fixed(const std::vector<T>& src,
                       marshal_storage& st,
                       fixed_storage* fixed,
                       rmm::device_uvector<T> fixed_storage::*member,
                       std::vector<T> fixed_storage::*host_member,
                       size_t seat)
{
  return fixed ? fixed->stage(member, host_member, seat, src.data(), src.size()) : upload(src, st);
}

template <typename T>
static T* alloc_fixed(size_t count,
                      marshal_storage& st,
                      fixed_storage* fixed,
                      rmm::device_uvector<T> fixed_storage::*member,
                      size_t seat,
                      bool zeroed = false)
{
  if (fixed) { return fixed->seat_ptr(member, seat, count); }
  return zeroed ? alloc_zeroed<T>(count, st) : alloc_uninit<T>(count, st);
}



// upload_solve_class_config() is gone: all it did was set g_elide_noop, and that device global was
// removed. Nothing else read it.

// ---- [CONFIG] self-report -------------------------------------------------------------------
// Every launch states its own decisive configuration. A measurement that reports its configuration
// cannot be mislabelled by the operator, the shell history, or another agent reading the output.
static void report_config(const char* path, size_t nprob)
{
  std::printf("[CONFIG] path=%s geometry=persistent-warp<<<resident,32>>> problems=%zu factor_index_bits=%d "
              "mu=divide tri_diag=divide\n",
              path, nprob, 32);
}

// Marshal one prepared problem into a kernel_args with fresh device buffers. Both the
// single-problem wrapper and the batch wrapper build kernel_args through here, so the device
// sees byte-identical inputs either way. `ft` must be a fresh factorization (num_updates == 0).
// Free the buffers afterwards with free_kernel_args.
// Problem-invariant read-only data shared across nodes with the same original matrix.
// A and the full CSR Arow are immutable; bounds and factorization data remain per node.
struct ro_upload {
  i_t *A_col_start{}, *A_i{};
  f_t* A_x{};
  i_t A_nz{};
  f_t *rhs{}, *objective{};
  const i_t* Arow_row_start{};
  const i_t* Arow_j{};
  const f_t* Arow_x{};
};

static ro_upload upload_ro(const lp_problem_t<i_t, f_t>& lp, marshal_storage& st)
{
  raft::common::nvtx::range nvtx_scope("GpuDS::upload_ro");
  const auto& A = lp.A;
  ro_upload r{};
  // Build and upload the immutable CSR transpose once per distinct A.
  st.arow_keep.emplace_back(lp.num_rows, lp.num_cols, A.nnz());
  auto& Arow_host = st.arow_keep.back();
  A.to_compressed_row(Arow_host);
  r.Arow_row_start = upload(Arow_host.row_start, st);
  r.Arow_j         = upload(Arow_host.j, st);
  r.Arow_x         = upload(Arow_host.x, st);
  r.A_col_start = upload(A.col_start, st); r.A_nz = A.nnz();
  r.A_i = upload_n(A.i.data(), A.nnz(), st); r.A_x = upload_n(A.x.data(), A.nnz(), st);
  r.rhs = upload(lp.rhs, st); r.objective = upload(lp.objective, st);
  return r;
}

static i_t mpf_update_cap()
{
  if (const char* e = std::getenv("CUOPT_GPU_MPF_UPDATE_CAP")) { return std::atoi(e); }
  return 256;   // measured max on the v6 nodes is 208 updates; this clears it with margin
}
static i_t mpf_eta_col_budget()
{
  if (const char* e = std::getenv("CUOPT_GPU_MPF_ETA_COL")) { return std::atoi(e); }
  return 256;   // measured max nnz/update is 532; 2*256*256 entries clears every v6 node
}
static i_t mpf_eta_nnz_cap() { return 2 * mpf_update_cap() * mpf_eta_col_budget(); }


// Bytes of per-slot buffer VRAM. Pre-flight estimate only -- nothing depends on it being exact.
static size_t slot_buffer_bytes(size_t m, size_t n, size_t ucap, size_t eta)
{
  const size_t mn = (m > n) ? m : n;
  const size_t m1 = m + 1;
  size_t t = 0;
  t += (mn) * sizeof(i_t);  // nonbasic_pos
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(f_t);
  t += (mn) * sizeof(i_t);
  t += (mn) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m1) * sizeof(i_t);
  t += (m) * sizeof(f_t);
  t += (m) * sizeof(f_t);
  t += (2 * ucap + 1) * sizeof(i_t);
  t += (eta) * sizeof(i_t);
  t += (eta) * sizeof(f_t);
  t += (ucap) * sizeof(f_t);
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * (sizeof(i_t) + sizeof(f_t));
  t += mn * sizeof(f_t);  // ft_dense_scratch2
  // Immutable Arow removes the per-slot nonbasic_end and row_mark buffers.
  t -= 2 * mn * sizeof(i_t);
  return t;
}

static kernel_args marshal_problem(const lp_problem_t<i_t, f_t>& lp,
                                   const simplex_solver_settings_t<i_t, f_t>& settings,
                                   std::vector<variable_status_t>& vstatus,
                                   basis_update_mpf_t<i_t, f_t>& ft,
                                   std::vector<i_t>& basic_list,
                                   std::vector<i_t>& nonbasic_list,
                                   lp_solution_t<i_t, f_t>& sol,
                                   std::vector<f_t>& delta_y_steepest_edge,
                                   const ro_upload* ro,
                                   marshal_storage& st,
                                   bool refactor_resume  = false,
                                   bool terminal_certificate_retry = false,
                                   const std::vector<f_t>* lower_ovr = nullptr,
                                   const std::vector<f_t>* upper_ovr = nullptr,
                                   fixed_storage* fixed = nullptr,
                                   size_t seat = 0,
                                   const factor_slots_t* slots = nullptr,
                                   packed_upload_t* pack = nullptr)
{
  raft::common::nvtx::range nvtx_scope("GpuDS::marshal_problem");
  const int m = lp.num_rows;
  const int n = lp.num_cols;
  assert(delta_y_steepest_edge.size() == static_cast<size_t>(n));

  const auto& A   = lp.A;
  const auto& L0  = ft.L0();
  const auto& U0  = ft.U0();
  const auto& L0t = ft.L0_transpose();
  const auto& U0t = ft.U0_transpose();
  for (const auto* factor : {&L0, &U0, &L0t, &U0t}) {
    for (const auto row : factor->i) {
      if (row < 0 || row >= m) {
        throw std::runtime_error("gpu_phase2: factor row index is outside [0, m)");
      }
    }
  }

  kernel_args a{};
  a.m = m;
  a.n = n;

  a.obj_scale   = lp.obj_scale;
  a.obj_constant = lp.obj_constant;
  // Share A, rhs, objective, and the complete immutable Arow across nodes from one original
  // problem. Bounds, factors, and basis state remain node-local.
  if (ro != nullptr) {
    a.A_col_start    = ro->A_col_start;
    a.A_nz           = ro->A_nz;
    a.A_i            = ro->A_i;
    a.A_x            = ro->A_x;
    a.rhs            = ro->rhs;
    a.objective      = ro->objective;
    a.Arow_row_start = ro->Arow_row_start;
    a.Arow_j         = ro->Arow_j;
    a.Arow_x         = ro->Arow_x;
  } else {
    a.A_col_start = upload(A.col_start, st);
    a.A_nz        = A.nnz();
    a.A_i         = upload_n(A.i.data(), A.nnz(), st);
    a.A_x         = upload_n(A.x.data(), A.nnz(), st);
    a.rhs         = upload(lp.rhs, st);
    a.objective   = upload(lp.objective, st);

    st.arow_keep.emplace_back(m, n, A.nnz());
    auto& Arow_host = st.arow_keep.back();
    A.to_compressed_row(Arow_host);
    a.Arow_row_start = upload(Arow_host.row_start, st);
    a.Arow_j         = upload(Arow_host.j, st);
    a.Arow_x         = upload(Arow_host.x, st);
  }
  // Per-node (basis- or branch-dependent), always uploaded per item: bounds, the LU factor, row perm.
  a.lower       = upload_fixed(lower_ovr != nullptr ? *lower_ovr : lp.lower,
                               st, fixed, &fixed_storage::lower,
                               &fixed_storage::h_lower, seat);
  a.upper       = upload_fixed(upper_ovr != nullptr ? *upper_ovr : lp.upper,
                               st, fixed, &fixed_storage::upper,
                               &fixed_storage::h_upper, seat);
  // Allocated lazily only when the persistent path first continues a child. Keeping it null for
  // one-shot launches preserves their allocation footprint exactly.
  a.bound_deltas      = nullptr;
  a.bound_delta_count = 0;
  a.root_lower        = nullptr;
  a.root_upper        = nullptr;
  a.bounds_from_root  = 0;
  a.L0_col_start = upload_fixed(L0.col_start, st, fixed, &fixed_storage::L0_col_start,
                                &fixed_storage::h_L0_col_start, seat);
  // With slots, the factor goes into the seat's slot through the packed upload.
  const auto factor_arrays = [&](const auto& F, i_t* slot_i, f_t* slot_x, i_t*& d_i, f_t*& d_x) {
    if (slots == nullptr) {
      d_i = upload_n(F.i.data(), F.nnz(), st);
      d_x = upload_n(F.x.data(), F.nnz(), st);
      return;
    }
    d_i = slot_i;
    d_x = slot_x;
    pack->add(d_i, F.i.data(), F.nnz() * sizeof(i_t), st.stream);
    pack->add(d_x, F.x.data(), F.nnz() * sizeof(f_t), st.stream);
  };
  const auto at = [&](auto* base, size_t cap) { return slots ? base + seat * cap : nullptr; };
  a.L0_nz = L0.nnz();
  factor_arrays(L0, at(slots ? slots->Li : nullptr, slots ? slots->capL : 0),
                at(slots ? slots->Lx : nullptr, slots ? slots->capL : 0), a.L0_i, a.L0_x);
  a.U0_col_start = upload_fixed(U0.col_start, st, fixed, &fixed_storage::U0_col_start,
                                &fixed_storage::h_U0_col_start, seat);
  a.U0_nz = U0.nnz();
  factor_arrays(U0, at(slots ? slots->Ui : nullptr, slots ? slots->capU : 0),
                at(slots ? slots->Ux : nullptr, slots ? slots->capU : 0), a.U0_i, a.U0_x);
  a.L0t_col_start = upload_fixed(L0t.col_start, st, fixed, &fixed_storage::L0t_col_start,
                                 &fixed_storage::h_L0t_col_start, seat);
  a.L0t_nz = L0t.nnz();
  factor_arrays(L0t, at(slots ? slots->Lti : nullptr, slots ? slots->capL : 0),
                at(slots ? slots->Ltx : nullptr, slots ? slots->capL : 0), a.L0t_i, a.L0t_x);
  a.U0t_col_start = upload_fixed(U0t.col_start, st, fixed, &fixed_storage::U0t_col_start,
                                 &fixed_storage::h_U0t_col_start, seat);
  a.U0t_nz = U0t.nnz();
  factor_arrays(U0t, at(slots ? slots->Uti : nullptr, slots ? slots->capU : 0),
                at(slots ? slots->Utx : nullptr, slots ? slots->capU : 0), a.U0t_i, a.U0t_x);
  a.row_perm     = upload_fixed(ft.row_permutation(), st, fixed, &fixed_storage::row_perm,
                                &fixed_storage::h_row_perm, seat);
  a.inv_row_perm = upload_fixed(ft.inverse_row_permutation(), st, fixed,
                                &fixed_storage::inv_row_perm,
                                &fixed_storage::h_inv_row_perm, seat);
  a.xi_workspace = alloc_fixed<int>(2 * m, st, fixed, &fixed_storage::xi_workspace, seat, true);
  a.x_workspace  = alloc_fixed<double>(m, st, fixed, &fixed_storage::x_workspace, seat, true);
  const i_t reach_words = reach_mark_storage_words(m);
  a.reach_mark = reinterpret_cast<unsigned char*>(
    fixed ? fixed->seat_ptr(&fixed_storage::reach_mark, seat, reach_words)
          : alloc_zeroed<reach_mark_word>(reach_words, st));
  a.num_updates  = 0;
  a.refactor_frequency = settings.refactor_frequency;
  a.mpf_update_cap     = mpf_update_cap();
  a.mpf_eta_nnz_cap    = mpf_eta_nnz_cap();

  a.refactor_resume = refactor_resume ? 1 : 0;
  a.terminal_certificate_retry = terminal_certificate_retry ? 1 : 0;
  a.resume_obj      = sol.objective;
  a.resume_iter     = static_cast<i_t>(sol.iterations);
  a.vstatus       = upload_fixed(vstatus, st, fixed, &fixed_storage::vstatus, &fixed_storage::h_vstatus, seat);
  a.basic_list    = upload_fixed(basic_list, st, fixed, &fixed_storage::basic_list, &fixed_storage::h_basic_list, seat);
  // REACHABILITY CHECK for the nonbasic_list length contract. get_basis_from_vstatus routes
  // SUPERBASIC variables to superbasic_list, NOT nonbasic_list (basis_solves.cpp:53), so
  // nonbasic_list holds n - m - |superbasic| entries -- while the device's ratio test iterates
  // n - m of them and the copy-back writes n - m back. Any shortfall is an out-of-bounds read on
  // the device and a heap overflow on the host.
  if (static_cast<i_t>(nonbasic_list.size()) != n - m) {
    std::fprintf(stderr,
                 "[NBLEN] nonbasic_list=%zu but n-m=%d (m=%d n=%d) -- short by %d\n",
                 nonbasic_list.size(), (int)(n - m), (int)m, (int)n,
                 (int)(n - m - static_cast<i_t>(nonbasic_list.size())));
  }
  a.nonbasic_list = upload_fixed(nonbasic_list, st, fixed, &fixed_storage::nonbasic_list, &fixed_storage::h_nonbasic_list, seat);
  a.x             = upload_fixed(sol.x, st, fixed, &fixed_storage::x, &fixed_storage::h_x, seat);
  a.y             = upload_fixed(sol.y, st, fixed, &fixed_storage::y, &fixed_storage::h_y, seat);
  a.z             = upload_fixed(sol.z, st, fixed, &fixed_storage::z, &fixed_storage::h_z, seat);
  a.dy_se         = upload_fixed(delta_y_steepest_edge, st, fixed, &fixed_storage::dy_se, &fixed_storage::h_dy_se, seat);

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

  // TWO ints: [0] = status, [1] = the REFACTOR_NEEDED reason (0 periodic / 1 rejected update).
  a.out_status         = alloc_fixed<int>(2, st, fixed, &fixed_storage::out_status, seat, true);
  a.out_objective      = alloc_fixed<double>(1, st, fixed, &fixed_storage::out_objective, seat, true);
  a.out_user_objective = alloc_fixed<double>(1, st, fixed, &fixed_storage::out_user_objective, seat, true);
  a.out_iterations     = alloc_fixed<int>(1, st, fixed, &fixed_storage::out_iterations, seat, true);

  a.slot = alloc_fixed<slot_state<gpu_basis_update_t>>(
    1, st, fixed, &fixed_storage::slot, seat);

  // Each workspace has its own fixed-capacity buffer, wired by attach_all_buffers. Workspaces
  // are not zeroed: consumers initialize elements before reading them.
  {
    const size_t mn   = (static_cast<size_t>(m) > static_cast<size_t>(n)) ? m : n;
    const size_t m1   = static_cast<size_t>(m) + 1;
    const size_t ucap = static_cast<size_t>(mpf_update_cap());
    const size_t eta  = static_cast<size_t>(mpf_eta_nnz_cap());
    slot_buffers& b   = a.buf;
    b.xB_workspace = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::xB_workspace, seat);
    b.delta_y = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::delta_y, seat);
    b.delta_z = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::delta_z, seat);
    b.delta_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::delta_x, seat);
    b.v = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::v, seat);
    b.squared_infeas = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::squared_infeas, seat);
    b.setup_rhs = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::setup_rhs, seat);
    b.basic_mark = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::basic_mark, seat);
    b.delta_z_mark = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::delta_z_mark, seat);
    b.delta_z_indices = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::delta_z_indices, seat);
    b.nonbasic_pos = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::nonbasic_pos, seat);
    b.infeas_idx = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::infeas_idx, seat);
    b.ft_dense_scratch = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::ft_dense_scratch, seat);
    b.ft_dense_scratch2 = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::ft_dense_scratch2, seat);
    b.ft_sort_bucket = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::ft_sort_bucket, seat);
    b.ft_sort_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::ft_sort_x, seat);
    b.ft_sort_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::ft_sort_i, seat);
    b.ft_sort_perm = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::ft_sort_perm, seat);
    b.lvl_lev0 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_lev0, seat);
    b.lvl_ls0 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_ls0, seat);
    b.lvl_co0 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_co0, seat);
    b.lvl_lev1 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_lev1, seat);
    b.lvl_ls1 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_ls1, seat);
    b.lvl_co1 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_co1, seat);
    b.lvl_lev2 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_lev2, seat);
    b.lvl_ls2 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_ls2, seat);
    b.lvl_co2 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_co2, seat);
    b.lvl_lev3 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_lev3, seat);
    b.lvl_ls3 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_ls3, seat);
    b.lvl_co3 = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_co3, seat);
    b.sp_ls_v = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::sp_ls_v, seat);
    b.sp_co_v = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::sp_co_v, seat);
    b.sp_pos_v = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::sp_pos_v, seat);
    b.lvl_pos_v = alloc_fixed<i_t>(m1, st, fixed, &fixed_storage::lvl_pos_v, seat);
    b.ft_recover_rhs = alloc_fixed<f_t>(m, st, fixed, &fixed_storage::ft_recover_rhs, seat);
    b.ft_recover_sol = alloc_fixed<f_t>(m, st, fixed, &fixed_storage::ft_recover_sol, seat);
    b.S_col_start = alloc_fixed<i_t>(2 * ucap + 1, st, fixed, &fixed_storage::S_col_start, seat);
    b.S_i = alloc_fixed<i_t>(eta, st, fixed, &fixed_storage::S_i, seat);
    b.S_x = alloc_fixed<f_t>(eta, st, fixed, &fixed_storage::S_x, seat);
    b.mu_values = alloc_fixed<f_t>(ucap, st, fixed, &fixed_storage::mu_values, seat);
    b.delta_y_sparse_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::delta_y_sparse_i, seat);
    b.delta_y_sparse_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::delta_y_sparse_x, seat);
    b.UTsol_sparse_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::UTsol_sparse_i, seat);
    b.UTsol_sparse_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::UTsol_sparse_x, seat);
    b.utilde_sparse_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::utilde_sparse_i, seat);
    b.utilde_sparse_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::utilde_sparse_x, seat);
    b.scaled_dxB_sparse_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::scaled_dxB_sparse_i, seat);
    b.scaled_dxB_sparse_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::scaled_dxB_sparse_x, seat);
    b.rhs_sparse_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::rhs_sparse_i, seat);
    b.rhs_sparse_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::rhs_sparse_x, seat);
    b.v_sparse_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::v_sparse_i, seat);
    b.v_sparse_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::v_sparse_x, seat);
    b.se_dy_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::se_dy_i, seat);
    b.se_dy_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::se_dy_x, seat);
    b.se_UTsol_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::se_UTsol_i, seat);
    b.se_UTsol_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::se_UTsol_x, seat);
    b.se_ei_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::se_ei_i, seat);
    b.se_ei_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::se_ei_x, seat);
    b.ft_sparse_scratch_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::ft_sparse_scratch_i, seat);
    b.ft_sparse_scratch_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::ft_sparse_scratch_x, seat);
    b.ft_ei_scratch_i = alloc_fixed<i_t>(mn, st, fixed, &fixed_storage::ft_ei_scratch_i, seat);
    b.ft_ei_scratch_x = alloc_fixed<f_t>(mn, st, fixed, &fixed_storage::ft_ei_scratch_x, seat);
  }
  return a;
}

// Bytes uploaded once per distinct immutable matrix block in a batch. marshal_problem deduplicates
// exactly these arrays by A.col_start.data(), so the pre-flight must not charge them once per seat.
static size_t problem_shared_ro_bytes(const lp_problem_t<i_t, f_t>& lp)
{
  const size_t m = lp.num_rows;
  const size_t n = lp.num_cols;
  auto bi = [](size_t c) { return (c == 0 ? 1 : c) * sizeof(i_t); };
  auto bd = [](size_t c) { return (c == 0 ? 1 : c) * sizeof(f_t); };
  // A CSC + rhs/objective + immutable CSR transpose.
  return bi(lp.A.col_start.size()) + bi(lp.A.nnz()) + bd(lp.A.nnz()) + bd(m) + bd(n) +
         bi(m + 1) + bi(lp.A.nnz()) + bd(lp.A.nnz());
}

// Exact device bytes marshal_problem cudaMallocs for one problem's STATIC buffers (read-only
// factors/matrix + mutable state + per-slot buffers; the kernel allocates nothing itself). Used by
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
  // Immutable A/rhs/objective/Arow (deduplicated across seats) plus per-seat lower/upper.
  b += problem_shared_ro_bytes(lp) + 2 * bd(n);
  // Four factors, each col_start + i + x
  b += bi(L0.col_start.size())  + bi(L0.nnz())  + bd(L0.nnz());
  b += bi(U0.col_start.size())  + bi(U0.nnz())  + bd(U0.nnz());
  b += bi(L0t.col_start.size()) + bi(L0t.nnz()) + bd(L0t.nnz());
  b += bi(U0t.col_start.size()) + bi(U0t.nnz()) + bd(U0t.nnz());
  // perms + workspaces
  b += bi(m) + bi(m) + bi(2 * m) + bd(m) +
       static_cast<size_t>(reach_mark_storage_words(static_cast<i_t>(m))) * sizeof(reach_mark_word);
  // state: vstatus(n) + basic(m) + nonbasic(<=n) + x(n)+y(m)+z(n) + dy_se(n) + outputs
  b += n * sizeof(variable_status_t) + bi(m) + bi(n) + bd(n) + bd(m) + bd(n) + bd(n);
  // out_status[2] + iterations + three zero-step diagnostics; objective + user_objective.
  b += 6 * bi(1) + 2 * bd(1);
  // persistent per-slot solve state (slot_state<FT> header only -- its gpu_span payloads live in
  // the arena below, or on the in-kernel heap for the few buffers left there)
  b += sizeof(slot_state<gpu_basis_update_t>);
  // The per-slot buffers. Static VRAM, so it must be in the pre-flight or a large batch passes the
  // check and then fails to allocate.
  b += slot_buffer_bytes(m, n, static_cast<size_t>(mpf_update_cap()),
                         static_cast<size_t>(mpf_eta_nnz_cap()));
  return b;
}


// ---- In-flight batch -------------------------------------------------------------------------
// Everything a launched batch owns until its results are read back. The synchronous entry point is
// launch + wait + finish; branch-and-bound uses the three separately so the summoning worker can
// keep plunging on the CPU while its nodes run.
//
// The compile is --default-stream=per-thread, so each host thread already launches on its own
// stream and two workers do not serialise on the legacy default stream. What is NOT safe to run
// concurrently is the pair of DEVICE GLOBALS this path writes per launch --
// A read-only upload that OUTLIVES a single launch. Every node of a branch-and-bound tree shares the
// same A / rhs / objective and the same Arow master, so re-uploading them on every launch is pure
// repetition -- and with in-seat plunging a cycle now issues many launches. The caller (a pool)
// creates one of these, hands it to every launch it makes, and destroys it when the pool dies.
//
// It holds ONLY the immutable problem block. Per-seat mutable state -- bounds, LU, perms, the Arow
// WORKING copy that warp_nbe_body permutes in place -- stays per launch, because sharing any of that
// across concurrently resident seats is a data race.
template <typename i_t, typename f_t>
struct gpu_ro_cache_t {
  rmm::cuda_stream owned_stream;   // MUST be declared before `store`, which binds to its view
  marshal_storage store;
  // Keyed on A's storage address, as the per-launch cache is. Across a POOL's lifetime that address
  // can be recycled: `original_lp_` is rebuilt when cuts add slack rows, and a freed vector can come
  // back at the same address with different contents. So the shape is recorded with the entry and
  // re-checked -- a mismatch re-uploads instead of handing the kernel a stale matrix.
  struct entry_t {
    ro_upload ro;
    size_t col_start_len{0}, nz{0};
  };
  std::unordered_map<const void*, entry_t> cache;
  // SHARED BY EVERY POOL ON THIS DEVICE, and populated lazily at the first launch that needs it --
  // so several worker threads can reach it at once. Without this lock they race on the map and on
  // `store`, which segfaults. Contention is negligible: one insert per distinct matrix, then
  // lookups.
  std::mutex mtx;
  // OWNS its stream. It used to borrow a pool's, which tied the cache's lifetime to that pool's:
  // destroying the pools first left the cache's device_buffers freeing on a dead stream, faulting
  // inside libcuda (cuMemFreeAsync). A cache shared by several pools cannot belong to any one of
  // them, so it has its own. Declared BEFORE `store` so it is constructed first and destroyed last.
  gpu_ro_cache_t() : owned_stream(), store(owned_stream.view()) {}
  // DIAGNOSTIC ONLY: reproduce the old behaviour where the cache BORROWED a pool's stream. Its
  // device_buffers are then allocated stream-ordered on that pool's stream while kernels on the
  // OTHER pools' streams read them. Used to prove whether that is what produced the rare wrong
  // objective; never for production.
  explicit gpu_ro_cache_t(rmm::cuda_stream_view borrowed) : owned_stream(), store(borrowed) {}
  // Prebuilt block (gpu_ro_cache_prebuild): uploaded once before any pool starts, then read without
  // the map or the lock by every launch whose LP matches the host copy below.
  bool prebuilt{false};
  ro_upload pre{};
  std::vector<i_t> pre_col_start, pre_i;
  std::vector<f_t> pre_x, pre_rhs, pre_objective;
  bool pre_matches(const lp_problem_t<i_t, f_t>& lp) const
  {
    return lp.A.col_start == pre_col_start && lp.A.i == pre_i && lp.A.x == pre_x && lp.rhs == pre_rhs &&
           lp.objective == pre_objective;
  }
};
inline std::atomic<std::uint64_t> g_ro_pre_hits{0}, g_ro_pre_misses{0};


// Result slabs are page-locked unless CUOPT_BNB_GPU_PINNED_RESULTS=0.
inline bool gpu_pinned_results()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PINNED_RESULTS");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// CUOPT_BNB_GPU_LEAN_FRESH: device-LU fresh seats skip the x / y / z slabs.
inline bool gpu_lean_fresh()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LEAN_FRESH");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// CUOPT_BNB_GPU_FEW_CALLS (0 disables): batched result copies, no counter memsets, timing events
// only with CUOPT_BNB_GPU_LOG.
inline bool gpu_few_calls()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FEW_CALLS");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}
// CUOPT_BNB_GPU_YIELD=f (0 = off): once a fraction f of a launch's live seats has finished, the
// remaining seats take the periodic-refactor exit and resume in the next launch.
inline double gpu_yield_frac()
{
  static const double f = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_YIELD");
    return e == nullptr ? 0.0 : std::atof(e);
  }();
  return f;
}
inline bool gpu_timing_events()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LOG");
    return !gpu_few_calls() || (e != nullptr && std::string(e) != "0");
  }();
  return on;
}
struct d2h_list_t {
  std::vector<void*> dst;
  std::vector<const void*> src;
  std::vector<size_t> bytes;
  void add(void* d, const void* s, size_t n)
  {
    if (n == 0) { return; }
    dst.push_back(d);
    src.push_back(s);
    bytes.push_back(n);
  }
  void submit(cudaStream_t cs)
  {
    if (dst.empty()) { return; }
    if (gpu_few_calls()) {
      cudaMemcpyAttributes attr{};
      attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
      size_t first        = 0;
      RAFT_CUDA_TRY(cudaMemcpyBatchAsync(
        dst.data(), src.data(), bytes.data(), dst.size(), &attr, &first, 1, cs));
    } else {
      for (size_t k = 0; k < dst.size(); ++k) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(dst[k], src[k], bytes[k], cudaMemcpyDeviceToHost, cs));
      }
    }
    dst.clear();
    src.clear();
    bytes.clear();
  }
};

// CUOPT_BNB_GPU_PAGEABLE_REST (0 disables): slabs read only by per-seat fetches (y, z, dy_se, basis lists) are
// pageable; pinning is serialized process-wide (~1 GB/s).
inline bool gpu_pageable_rest()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PAGEABLE_REST");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}
struct rest_slab_tag {};
// CUOPT_BNB_GPU_PREALLOC_PINNED: the pinned staging / delta / compact-row / bound buffers get their
// worst-case size once at launch instead of growing (each growth pins new host memory).
inline bool gpu_prealloc_pinned()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PREALLOC_PINNED");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}
// CUOPT_BNB_GPU_SEAT_CLOCK: per-seat solve timers (globaltimer + clock64) per round, summarized per launch.
inline bool gpu_seat_clock()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_SEAT_CLOCK");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}
// CUOPT_BNB_GPU_DIVE_INKERNEL (0 disables): device dives continue inside the solve kernel (no per-round launches).
inline bool gpu_dive_in_kernel()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_DIVE_INKERNEL");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}
// CUOPT_BNB_GPU_COMPACT (0 disables): the harvest returns x / bounds / basis only for seats the host
// cannot branch from device scores; the others send their branch column and packed basis.
inline bool gpu_compact_harvest()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_COMPACT");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}
// CUOPT_BNB_GPU_PAGEABLE_SETUP (0 disables): the first launch's packed upload (every seat's host LU) is staged in
// pageable memory; later generations use a pinned pack sized for them.
inline bool gpu_pageable_setup()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PAGEABLE_SETUP");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

template <typename T, typename Tag = void>
struct result_allocator_t {
  using value_type = T;
  template <typename U>
  struct rebind {
    using other = result_allocator_t<U, Tag>;
  };
  static bool pinned()
  {
    return gpu_pinned_results() && !(std::is_same_v<Tag, rest_slab_tag> && gpu_pageable_rest());
  }
  result_allocator_t() = default;
  template <typename U>
  result_allocator_t(const result_allocator_t<U, Tag>&)
  {
  }
  T* allocate(size_t n)
  {
    if (!pinned()) { return std::allocator<T>{}.allocate(n); }
    if (gpu_huge_arena()) {
      if (tl_pinned_arena != nullptr) {
        if (void* p = tl_pinned_arena->take(n * sizeof(T))) { return static_cast<T*>(p); }
      }
      if (void* p = pinned_host::overflow_take(n * sizeof(T), tl_pinned_arena ? tl_pinned_arena->node : -1)) {
        return static_cast<T*>(p);
      }
      const hp_mapping_t m = hp_map(n * sizeof(T));
      g_hp_maps.fetch_add(1);
      g_hp_bytes.fetch_add(m.bytes);
      std::lock_guard<std::mutex> lk(own_mu());
      own_maps()[m.p] = m;
      return static_cast<T*>(m.p);
    }
    void* p = nullptr;
    RAFT_CUDA_TRY(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable));
    return static_cast<T*>(p);
  }
  void deallocate(T* p, size_t n)
  {
    if (!pinned()) { return std::allocator<T>{}.deallocate(p, n); }
    if (gpu_huge_arena()) {
      if (g_hp_registry.contains(p) || pinned_host::contains(p)) { return; }  // freed with its arena
      hp_mapping_t m;
      {
        std::lock_guard<std::mutex> lk(own_mu());
        auto it = own_maps().find(p);
        if (it == own_maps().end()) { return; }
        m = it->second;
        own_maps().erase(it);
      }
      hp_unmap(m);
      return;
    }
    cudaFreeHost(p);
  }
  // Result-slab mappings owned by this allocator (outside any arena).
  static std::mutex& own_mu()
  {
    static std::mutex mu;
    return mu;
  }
  static std::unordered_map<void*, hp_mapping_t>& own_maps()
  {
    static std::unordered_map<void*, hp_mapping_t> m;
    return m;
  }
  template <typename U>
  bool operator==(const result_allocator_t<U, Tag>&) const
  {
    return true;
  }
  template <typename U>
  bool operator!=(const result_allocator_t<U, Tag>&) const
  {
    return false;
  }
};
template <typename T>
using result_vector_t = std::vector<T, result_allocator_t<T>>;
template <typename T>
using rest_vector_t = std::vector<T, result_allocator_t<T, rest_slab_tag>>;

template <typename i_t, typename f_t>
struct gpu_batch_handle_t {
  // CUOPT_BNB_GPU_HUGE_ARENA: declared first so it outlives every buffer carved from it.
  std::unique_ptr<pinned_arena_t> arena;
  std::vector<gpu_batch_item<i_t, f_t>> items;
  result_vector_t<kernel_args> h_args;   // page-locked with the result slabs (argument upload)
  std::vector<marshal_storage> store;
  std::optional<fixed_storage> fixed;
  marshal_storage ro_store;
  std::unordered_map<const void*, ro_upload> ro_cache;
  kernel_args* d_args{nullptr};
  unsigned int* d_next{nullptr};
  unsigned int* d_completed_blocks{nullptr};
  unsigned int* d_yield{nullptr};  // seats finished in the current launch (CUOPT_BNB_GPU_YIELD)
  // CUOPT_BNB_GPU_EARLY_D2H: the relaunch queues the result copies itself and raises the mapped flag
  // after them (set_flag_kernel); the solve kernel's own flag store goes to this device word instead.
  std::uint32_t* d_flag_sink{nullptr};
  bool early_d2h{false};          // this launch's results are already queued behind the flag
  bool harvest_slim_known{false}, harvest_slim{false};
  std::uint32_t* h_completion_flag{nullptr};
  std::uint32_t* d_completion_flag{nullptr};
  cudaEvent_t ev_start{nullptr}, ev_stop{nullptr};
  cudaEvent_t ev_pre{nullptr};   // start of a relaunch's device work (uploads, dense LU)
  cudaEvent_t ev_lu0{nullptr}, ev_lu1{nullptr};   // around the dense LU
  cudaEvent_t ev_h2d{nullptr}, ev_deltas{nullptr};   // after the packed copy / the delta copy
  cudaEvent_t ev_d2h0{nullptr}, ev_d2h{nullptr};   // around the harvest's result copies
  cudaEvent_t ev_g0{nullptr}, ev_g1{nullptr};       // around the compact gather (timing events)
  bool gather_timed{false};
  bool pre_recorded{false};
  bool timed{false};  // ev_start recorded for the in-flight launch
  unsigned int relaunch_blocks{0};
  std::uint64_t harvests{0};
  int device{-1};  // harvest split skips each handle's first 10 (ramp-up)
  d2h_list_t d2h;
  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  // Non-owning. When set, the immutable problem block lives here instead of in `ro_store`, and
  // survives this launch. `ro_store` still owns the PER-LAUNCH scratch (d_args, the steal counter).
  gpu_ro_cache_t<i_t, f_t>* ro_ext{nullptr};
  const void* ro_pre_lp{nullptr};   // last LP checked against ro_ext's prebuilt block, and the result
  bool ro_pre_ok{false};
  double up_t0{0.0};
  bool launched{false};
  // Set from the launch. _finish needs it to decide whether to copy the duals back -- only a caller
  // with a resume loop consumes them, and for everyone else it is m + n doubles of D2H for nothing.
  bool allow_refactor_exit{false};
  bool persistent_mode{false};
  bool view_mode{false};  // Nondeterministic pool consumes the host slabs without per-seat scatter.
  std::vector<unsigned char> staged;
  // Fissioned path (CUOPT_BNB_GPU_FISSION): per-seat explicit-inverse pointers, set at staging.
  std::vector<gpu_fission_inv> h_finv;
  std::optional<rmm::device_buffer> d_finv;
  std::optional<rmm::device_buffer> d_fm;  // fission replay matrices, one per seat
  std::optional<rmm::device_buffer> d_live;
  // Bound deltas of one generation, uploaded by the relaunch in one copy.
  std::optional<pinned_buf<gpu_bound_delta<i_t, f_t>>> h_deltas;
  std::optional<rmm::device_buffer> d_deltas;
  size_t delta_cap{0}, delta_dev_cap{0}, delta_used{0};
  std::vector<size_t> delta_offset;  // per seat; read only for this generation's CHILD_CONTINUE seats
  std::optional<rmm::device_buffer> d_root;  // root lower | upper | edge norms (path bounds)
  bool root_dy_se_set{false};
  // seat list | packed statuses
  std::optional<rmm::device_buffer> d_expand;
  std::vector<int> expand_seats;
  // sum_down | sum_up | num_down | num_up | seat scores
  const unsigned char* d_var_int{nullptr};
  std::optional<rmm::device_buffer> d_scores;
  std::optional<pinned_buf<gpu_scores::seat_score_t>> h_scores;
  gpu_scores::pseudocosts_t score_pc{};
  bool scores_armed{false}, scores_valid{false};
  // Device dives: root x | deltas | count | records | packed bases; host copy of the last three.
  int dive_rounds{1};
  bool dive_armed{false}, dive_valid{false};
  size_t dive_off[4]{};
  std::optional<rmm::device_buffer> d_dive;
  std::optional<pinned_buf<char>> h_dive;
  // Compact harvest: headers | full count | packed bases | full LP states (device); first three on host.
  bool compact_valid{false};
  size_t cmp_off[3]{}, cmp_lp_state_bytes{0}, cmp_lp_states_cap{0};
  std::optional<rmm::device_buffer> d_compact;
  std::optional<pinned_buf<char>> h_compact, h_lp_states;
  std::vector<unsigned char> seat_kind;
  // CUOPT_BNB_GPU_SEAT_CLOCK: [round][seat] timers of the last launch.
  std::optional<rmm::device_buffer> d_clk;
  std::optional<pinned_buf<seat_clock_t>> h_clk;
  int clk_cap_rounds{0}, clk_rounds{0};
  bool slim_harvest{false};  // last harvest skipped y / z / dy_se / basis lists
  bool use_slots{false};
  factor_slots_t slots;
  packed_upload_t pack;
  bool strengthen{false};              // device bound strengthening; harvest returns the bounds
  // Device LU of FRESH seats (dense cuBLAS batch): per-seat targets, this generation's seats.
  bool device_lu{false};
  void* blas{nullptr};
  std::vector<gpu_lu::dense_seat_t> lu_targets;
  std::vector<gpu_lu::dense_seat_t> lu_batch;
  bool lu_custom{false};  // run_batch_lu (CUOPT_BNB_GPU_LU_KERNEL=custom) instead of cuBLAS
  void* lu_ws{nullptr};   // its workspace
  double *lu_dense{nullptr}, **lu_ptrs{nullptr};
  int *lu_piv{nullptr}, *lu_cinfo{nullptr}, *lu_info{nullptr};
  gpu_lu::dense_seat_t* d_lu_batch{nullptr};
  // Pinned host slabs of the device bounds (with `strengthen`), allocated at the first harvest.
  std::optional<pinned_buf<f_t>> h_lower, h_upper;
  struct factor_caps_t {
    size_t L0{0}, U0{0}, L0t{0}, U0t{0};
  };
  std::vector<factor_caps_t> factor_caps;
  i_t rows{0}, cols{0};
  size_t active_seats{0};
  // Scalars are pinned; large returned vectors remain pageable and are reused across generations.
  std::optional<pinned_buf<i_t>> h_status;
  std::optional<pinned_buf<i_t>> h_iterations;
  std::optional<pinned_buf<f_t>> h_objective;
  std::optional<pinned_buf<f_t>> h_user_objective;
  result_vector_t<f_t> h_x;
  rest_vector_t<f_t> h_y, h_z, h_dy_se;
  result_vector_t<variable_status_t> h_vstatus;
  rest_vector_t<i_t> h_basic_list, h_nonbasic_list;

  ~gpu_batch_handle_t()
  {
    if (h_completion_flag != nullptr) { cudaFreeHost(h_completion_flag); }
    if (ev_start != nullptr) { cudaEventDestroy(ev_start); }
    gpu_lu::destroy_blas_handle(blas);
    if (ev_stop != nullptr) { cudaEventDestroy(ev_stop); }
    if (ev_pre != nullptr) { cudaEventDestroy(ev_pre); }
    if (ev_lu0 != nullptr) { cudaEventDestroy(ev_lu0); }
    if (ev_lu1 != nullptr) { cudaEventDestroy(ev_lu1); }
    if (ev_h2d != nullptr) { cudaEventDestroy(ev_h2d); }
    if (ev_deltas != nullptr) { cudaEventDestroy(ev_deltas); }
    if (ev_d2h != nullptr) { cudaEventDestroy(ev_d2h); }
    if (ev_d2h0 != nullptr) { cudaEventDestroy(ev_d2h0); }
    if (ev_g0 != nullptr) { cudaEventDestroy(ev_g0); }
    if (ev_g1 != nullptr) { cudaEventDestroy(ev_g1); }
  }
};

}  // namespace gpu_phase2_detail

template <typename f_t>
static bool resume_dual_feasible(const std::vector<variable_status_t>& vstatus,
                                 const std::vector<f_t>& z,
                                 f_t dual_tol)
{
  if (z.size() < vstatus.size()) { return false; }
  f_t infeasibility = 0.0;
  for (size_t j = 0; j < vstatus.size(); ++j) {
    if (!std::isfinite(z[j])) { return false; }
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && z[j] < -dual_tol) {
      infeasibility += -z[j];
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && z[j] > dual_tol) {
      infeasibility += z[j];
    } else if (vstatus[j] == variable_status_t::NONBASIC_FREE) {
      infeasibility += std::abs(z[j]);
    }
    if (infeasibility > dual_tol) { return false; }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Host wrapper. Same signature as simplified_dual_phase2.
// ---------------------------------------------------------------------------
// ONE LEG of a solve: marshal, launch, copy back. A leg ends either with a real answer or with
// REFACTOR_NEEDED, in which case gpu_simplified_dual_phase2 below rebuilds the factorization and
// calls this again. `resume` tells the device the host has already rebuilt the primal point.
template <typename i_t, typename f_t>
static dual::status_t gpu_simplified_dual_phase2_leg(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  std::vector<variable_status_t>& vstatus,
  basis_update_mpf_t<i_t, f_t>& ft,
  std::vector<i_t>& basic_list,
  std::vector<i_t>& nonbasic_list,
  lp_solution_t<i_t, f_t>& sol,
  std::vector<f_t>& delta_y_steepest_edge,
  bool resume,
  i_t& refactor_reason)
{
  using namespace gpu_phase2_detail;
  static_assert(sizeof(i_t) == sizeof(int) && sizeof(f_t) == sizeof(double),
                "gpu_simplified_dual_phase2 is only instantiated for <int, double>");

  const int m = lp.num_rows;
  const int n = lp.num_cols;
  assert(delta_y_steepest_edge.size() == static_cast<size_t>(n));

  // These arrays are indexed by the mathematical dimensions with no device-side bounds checks.
  // Reject a malformed first call here instead of turning a short host vector into an unrelated
  // device illegal-address report later in the solve.
  if (m < 0 || n < m || vstatus.size() != static_cast<size_t>(n) ||
      basic_list.size() != static_cast<size_t>(m) ||
      nonbasic_list.size() != static_cast<size_t>(n - m) ||
      sol.x.size() != static_cast<size_t>(n) || sol.y.size() != static_cast<size_t>(m) ||
      sol.z.size() != static_cast<size_t>(n) ||
      delta_y_steepest_edge.size() != static_cast<size_t>(n)) {
    std::fprintf(stderr, "[gpu_phase2] ERROR: first-call state has the wrong dimensions\n");
    return dual::status_t::NUMERICAL;
  }

  // Pure-stacking GPU port: the supplied factorization must be fresh (no stacked
  // updates yet), matching the harness which refactors immediately before the call.
  if (ft.num_updates() != 0) {
    std::fprintf(stderr,
                 "[gpu_phase2] ERROR: ft has %d stacked updates; the GPU port expects a fresh "
                 "factorization.\n",
                 ft.num_updates());
    return dual::status_t::NUMERICAL;
  }

  // Owns every uploaded buffer for this solve; destroyed at scope exit, returning them to the pool.
  marshal_storage store;
  kernel_args a = marshal_problem(lp,
                                  settings,
                                  vstatus,
                                  ft,
                                  basic_list,
                                  nonbasic_list,
                                  sol,
                                  delta_y_steepest_edge,
                                  nullptr,
                                  store,
                                  resume);

  // Work-stealing counter for the persistent kernel, zeroed before the launch. Per launch, not a
  // device global -- see the batch path for why.
  unsigned int* d_next = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_next, sizeof(unsigned int)));
  RAFT_CUDA_TRY(cudaMemset(d_next, 0, sizeof(unsigned int)));

  // The kernel takes a device pointer to an array of kernel_args (one per problem);
  // the single-problem path is just N == 1.
  kernel_args* d_args = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_args, sizeof(kernel_args)));
  RAFT_CUDA_TRY(cudaMemcpy(d_args, &a, sizeof(kernel_args), cudaMemcpyHostToDevice));

  report_config("monolith", 1);
  const auto _k0 = std::chrono::steady_clock::now();
  // AllowRefactorExit = TRUE on the single-problem entry: this path owns a host resume loop that
  // can answer REFACTOR_NEEDED. The batch entry takes the default (false) because it has none.
  gpu_phase2_kernel_direct<true><<<1, 32>>>(
    d_args, i_t{1}, d_next, nullptr, nullptr);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  cudaFree(d_args);
  cudaFree(d_next);

  // Marshal outputs back.
  int h_status[2] = {0, 0};
  double h_obj = 0.0, h_uobj = 0.0;
  int h_iters = 0;
  RAFT_CUDA_TRY(cudaMemcpy(h_status, a.out_status, sizeof(h_status), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(&h_obj, a.out_objective, sizeof(double), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(&h_uobj, a.out_user_objective, sizeof(double), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(&h_iters, a.out_iterations, sizeof(int), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(sol.x.data(), a.x, n * sizeof(double), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(sol.y.data(), a.y, m * sizeof(double), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(sol.z.data(), a.z, n * sizeof(double), cudaMemcpyDeviceToHost));
  // The FINAL basis and the maintained steepest-edge norms. Both are required to RESUME after a
  // host refactorization: the next leg has to continue from the basis the device stopped on and
  // from the norms it had, or it restarts pricing and takes a different path. y/z above are
  // maintained incrementally by the driver for the same reason.
  RAFT_CUDA_TRY(
    cudaMemcpy(vstatus.data(), a.vstatus, n * sizeof(variable_status_t), cudaMemcpyDeviceToHost));
  // Preserve the exact pivot-history order.  Reconstructing the lists from vstatus sorts them by
  // column index; that changes both the next LU and the order-dependent ratio-test scan after a
  // host refactor resume.
  RAFT_CUDA_TRY(
    cudaMemcpy(basic_list.data(), a.basic_list, m * sizeof(i_t), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(nonbasic_list.data(),
                           a.nonbasic_list,
                           (n - m) * sizeof(i_t),
                           cudaMemcpyDeviceToHost));
  delta_y_steepest_edge.resize(static_cast<size_t>(n));
  RAFT_CUDA_TRY(
    cudaMemcpy(delta_y_steepest_edge.data(), a.dy_se, n * sizeof(double), cudaMemcpyDeviceToHost));

  sol.objective      = h_obj;
  sol.user_objective = h_uobj;
  sol.iterations     = h_iters;
  refactor_reason    = h_status[1];

  return static_cast<dual::status_t>(h_status[0]);
}


// ---------------------------------------------------------------------------
// Batch wrapper: solve N independent problems in ONE launch, one block each.
// Each item must already be set up exactly as a single solve is (fresh factorized
// ft, basis lists, duals in sol.y/z) — i.e. the same steps run_phase2_from_basis
// does before the driver call. Fails LOUD (never a silent OOM): a pre-flight VRAM
// check rejects the whole batch if it will not fit. On success, fills out_status[i]
// and each item->sol (x, objective, user_objective, iterations).
// ---------------------------------------------------------------------------
namespace gpu_phase2_detail {
// Slot = max(first-launch max, min(bound, k x p90)), k = CUOPT_BNB_GPU_FACTOR_CAP_K (2);
// CUOPT_BNB_GPU_FACTOR_CAP_LIMIT lowers the accepted size.
static void size_factor_slots(const std::vector<gpu_batch_item<i_t, f_t>>& items,
                              size_t m,
                              factor_slots_t& fs)
{
  static const double k = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FACTOR_CAP_K");
    const double v = e != nullptr ? std::atof(e) : 2.0;
    return v > 0.0 ? v : 2.0;
  }();
  static const long limit = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_FACTOR_CAP_LIMIT");
    return e != nullptr ? std::atol(e) : 0L;
  }();
  std::vector<size_t> L, U;
  for (const auto& it : items) {
    if (it.terminal_status != dual::status_t::UNSET) { continue; }
    L.push_back(std::max<size_t>(it.ft->L0().nnz(), it.ft->L0_transpose().nnz()));
    U.push_back(std::max<size_t>(it.ft->U0().nnz(), it.ft->U0_transpose().nnz()));
  }
  fs.bound = std::max<size_t>(1, m * (m + 1) / 2);
  const auto cap_of = [&](std::vector<size_t> v) {
    const size_t mx  = *std::max_element(v.begin(), v.end());
    const size_t r   = std::min(v.size() - 1, (9 * v.size() + 9) / 10 - 1);
    std::nth_element(v.begin(), v.begin() + r, v.end());
    const auto want = static_cast<size_t>(std::ceil(k * static_cast<double>(v[r])));
    return std::max<size_t>({std::min(fs.bound, want), mx, 1});
  };
  fs.capL    = cap_of(L);
  fs.capU    = cap_of(U);
  fs.acceptL = limit > 0 ? std::min<size_t>(fs.capL, limit) : fs.capL;
  fs.acceptU = limit > 0 ? std::min<size_t>(fs.capU, limit) : fs.capU;
  for (size_t b = 0; b < L.size(); ++b) { fs.record(L[b], U[b]); }
}
}  // namespace gpu_phase2_detail

template <typename handle_t>
static void prealloc_pinned(handle_t* H)
{
  raft::common::nvtx::range nvtx_scope("GpuDS::prealloc_pinned");
  using namespace gpu_phase2_detail;
  using delta_t  = std::remove_reference_t<decltype(H->h_deltas->data()[0])>;
  using real_t   = std::remove_reference_t<decltype(H->h_lower->data()[0])>;
  const size_t N = H->h_args.size(), n = static_cast<size_t>(H->cols), m = static_cast<size_t>(H->rows);
  const cuda::stream_ref hs{H->stream.value()};
  H->pack.reserve(N * 40 * (n + m) + (size_t{1} << 20), H->stream);
  if (H->delta_cap < N * n) {
    H->h_deltas.emplace(pinned_make<delta_t>(nullptr, hs, N * n));
    H->delta_cap = N * n;
  }
  if (gpu_compact_harvest()) {
    const size_t lp_state = (5 * n * sizeof(real_t) + m * sizeof(real_t) + n * sizeof(int) + n + 15) / 16 * 16;
    if (H->cmp_lp_states_cap < N * lp_state) {
      H->h_lp_states.emplace(pinned_make<char>(H->arena.get(), hs, N * lp_state));
      H->cmp_lp_states_cap = N * lp_state;
    }
  }
  if (H->strengthen && !H->h_lower) {
    H->h_lower.emplace(pinned_make<real_t>(nullptr, hs, N * n));
    H->h_upper.emplace(pinned_make<real_t>(nullptr, hs, N * n));
  }
}

// Worst-case sizes of a pool handle's pinned host buffers (N seats, n columns, m rows).
struct pinned_plan_t {
  size_t staging{0}, deltas{0}, scores{0}, hdr{0}, lp_states{0}, bounds{0}, x{0}, vstatus{0}, rest{0}, small{0};
  size_t lp_state{0};
  bool compact{false}, pinned_x{false}, pinned_rest{false};
  size_t total(bool with_small) const
  {
    const auto a = [](size_t b) { return (b + 255) / 256 * 256; };   // pinned_arena_t::take alignment
    return a(staging) + a(deltas) + a(scores) + a(hdr) + a(lp_states) + 2 * a(bounds) + a(x) + a(vstatus) + rest +
           (with_small ? small : 0);
  }
};

// Staging upload (packed_upload_t), from gpu_simplified_dual_phase2_batch_stage per seat: bounds 16n,
// x 8n, vstatus n, basic + nonbasic lists 4n, y 8m, z + dy_se 16n, row permutations 8m, i.e.
// 45n + 16m bytes in <= 23 pieces; a host-refactored seat adds 4 column-start arrays (4(m+1) each) and
// 8 index / value arrays bounded by the factor slot caps (12 pieces, `factor_seat` bytes). Per launch:
// pseudocosts 24n (set_pseudocosts), the LU batch (N dense_seat_t), the fresh-seat list 4N, 16-byte
// padding per piece and the flush descriptors (2 pointers + 1 size per piece). `small`: the kernel
// arguments and the status / iteration / objective arrays created at the start of the first launch.
// CUOPT_BNB_GPU_DEVICE_TRANSPOSE=0: upload the host's L0t/U0t instead of building them on the device.
inline bool gpu_device_transpose()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_DEVICE_TRANSPOSE");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// Bound deltas reserved per seat. CHOSEN: 1024 (a child adds a few, a fresh path at most its depth);
// a larger generation grows the buffer.
constexpr size_t kPinnedDeltasPerSeat = 1024;
// Factor bytes per seat when the slot caps are not known yet (slice planning): four factor matrices
// (L0, U0 and their transposes, int index + real value per nonzero) with L+U estimated by nnz(A)
// (CHOSEN estimate; lrsa120: L+U ~19k, nnz(A) 20.5k), plus four column-start arrays.
inline size_t pinned_factor_seat_estimate(size_t a_nnz, size_t m)
{
  const size_t mats = gpu_device_transpose() ? 1 : 2;   // L0+U0 (+ their transposes)
  return mats * a_nnz * (sizeof(int) + sizeof(double)) + mats * 8 * (m + 1);
}

static pinned_plan_t pinned_plan(size_t N, size_t n, size_t m, size_t factor_seats, size_t factor_seat,
                                 bool strengthen)
{
  using namespace gpu_phase2_detail;
  const auto a16  = [](size_t b) { return (b + 15) / 16 * 16; };
  const auto a256 = [](size_t b) { return (b + 255) / 256 * 256; };
  pinned_plan_t p;
  const size_t pieces = 23 * N + 12 * factor_seats + 4;
  p.staging = N * (45 * n + 16 * m) + factor_seats * factor_seat + 24 * n + N * sizeof(gpu_lu::dense_seat_t) +
              4 * N + 16 * pieces + pieces * (2 * sizeof(void*) + sizeof(size_t));
  p.deltas  = N * std::min(n, kPinnedDeltasPerSeat) * sizeof(gpu_bound_delta<i_t, f_t>);
  p.scores  = N * sizeof(gpu_scores::seat_score_t);
  p.compact = gpu_compact_harvest();
  p.lp_state = a16(5 * n * sizeof(f_t) + m * sizeof(f_t) + n * sizeof(i_t) + n);   // relaunch cmp_lp_state_bytes
  if (p.compact) {
    p.hdr  = a16(a16(N * sizeof(compact_hdr_t)) + 16 + N * ((n + 3) / 4));        // relaunch cmp_off[2]
    p.lp_states = N * p.lp_state * 5 / 4;   // CHOSEN margin: measured use exceeds N LP states by ~6 % (lrsa120)
  }
  if (strengthen) { p.bounds = N * n * sizeof(f_t); }
  p.pinned_x    = gpu_pinned_results();
  p.pinned_rest = p.pinned_x && !gpu_pageable_rest();
  if (p.pinned_x) {
    p.x       = N * n * sizeof(f_t);
    p.vstatus = N * n * sizeof(variable_status_t);
  }
  if (p.pinned_rest) {   // z, dy_se, y, basic list, nonbasic list
    p.rest = 2 * a256(N * n * sizeof(f_t)) + a256(N * m * sizeof(f_t)) + a256(N * m * sizeof(i_t)) +
             a256(N * (n - m) * sizeof(i_t));
  }
  p.small = a256(N * sizeof(kernel_args)) + a256(2 * N * sizeof(i_t)) + a256(N * sizeof(i_t)) +
            2 * a256(N * sizeof(f_t));
  return p;
}

// Carve the handle's pinned buffers from its arena: a slice of the process arena taken when the handle
// was created (CUOPT_BNB_GPU_PINNED_ARENA), or else its own 2 MB-page region (CUOPT_BNB_GPU_HUGE_ARENA).
template <typename handle_t>
static void create_huge_arena(handle_t* H)
{
  raft::common::nvtx::range nvtx_scope("GpuDS::huge_arena");
  using namespace gpu_phase2_detail;
  using delta_t  = std::remove_reference_t<decltype(H->h_deltas->data()[0])>;
  using real_t   = std::remove_reference_t<decltype(H->h_lower->data()[0])>;
  const size_t N = H->h_args.size(), n = static_cast<size_t>(H->cols), m = static_cast<size_t>(H->rows);
  const cuda::stream_ref hs{H->stream.value()};
  const bool slice_mode = H->arena != nullptr;
  // CHOSEN, not a worst case: own-region mode reserves factor room for N / CUOPT_BNB_GPU_ARENA_FACTOR_DIV
  // host-refactored seats per launch (default 8; 0 = none). Slice mode reserves none (the slot caps are
  // only known after the first launch, the slice was sized before); larger uploads grow the buffer.
  static const size_t factor_div = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_ARENA_FACTOR_DIV");
    return e != nullptr ? static_cast<size_t>(std::max(0, std::atoi(e))) : size_t{8};
  }();
  // Slice mode: every seat may carry factors in one flush; sized from nnz(A) as the slice was.
  const size_t a_nnz        = H->items.empty() ? 0 : static_cast<size_t>(H->items[0].lp->A.nnz());
  const size_t factor_seats = slice_mode ? N : factor_div == 0 ? 0 : (N + factor_div - 1) / factor_div;
  const size_t factor_seat  = slice_mode ? pinned_factor_seat_estimate(a_nnz, m)
                                         : 16 * (m + 1) + 2 * (H->slots.capL + H->slots.capU) * (sizeof(int) + sizeof(real_t));
  const pinned_plan_t P = pinned_plan(N, n, m, factor_seats, factor_seat, H->strengthen);
  static std::once_flag printed;
  std::call_once(printed, [&] {
    std::fprintf(stderr,
                 "[GPUBATCH-ARENA] mode=%s N=%zu n=%zu m=%zu capL=%zu capU=%zu | MB: staging=%.2f (factor room=%.2f) "
                 "deltas=%.2f compact_hdr=%.2f lp_states=%.2f bounds=%.2f x+vstatus=%.2f rest=%.2f scores=%.3f "
                 "small=%.2f total=%.2f\n",
                 slice_mode ? "process-slice" : "own-region", N, n, m, static_cast<size_t>(H->slots.capL),
                 static_cast<size_t>(H->slots.capU), P.staging / 1e6, factor_seats * factor_seat / 1e6,
                 P.deltas / 1e6, P.hdr / 1e6, P.lp_states / 1e6, 2.0 * P.bounds / 1e6, (P.x + P.vstatus) / 1e6,
                 P.rest / 1e6, P.scores / 1e6, P.small / 1e6, P.total(slice_mode) / 1e6);
  });
  if (!slice_mode) { H->arena = std::make_unique<pinned_arena_t>(P.total(false)); }
  auto* A = H->arena.get();
  H->pack.h.emplace(pinned_make<unsigned char>(A, hs, P.staging));
  H->pack.cap = P.staging;
  H->h_deltas.emplace(pinned_make<delta_t>(A, hs, N * std::min(n, kPinnedDeltasPerSeat)));
  H->delta_cap = N * std::min(n, kPinnedDeltasPerSeat);
  H->h_scores.emplace(pinned_make<gpu_scores::seat_score_t>(A, hs, N));
  if (P.compact) {
    H->h_compact.emplace(pinned_make<char>(A, hs, P.hdr));
    H->h_lp_states.emplace(pinned_make<char>(A, hs, P.lp_states));
    H->cmp_lp_states_cap = P.lp_states;
  }
  if (H->strengthen && !H->h_lower) {
    H->h_lower.emplace(pinned_make<real_t>(A, hs, N * n));
    H->h_upper.emplace(pinned_make<real_t>(A, hs, N * n));
  }
  auto* const prev_tl = tl_pinned_arena;
  tl_pinned_arena     = A;   // the result slabs' allocator takes from the arena while set
  H->h_x.reserve(N * n);
  H->h_vstatus.reserve(N * n);
  if (P.pinned_rest) {
    H->h_y.reserve(N * m);
    H->h_z.reserve(N * n);
    H->h_dy_se.reserve(N * n);
    H->h_basic_list.reserve(N * m);
    H->h_nonbasic_list.reserve(N * (n - m));
  }
  tl_pinned_arena = prev_tl;
}

// CUOPT_BNB_GPU_PINNED_ARENA: reserve the process arena before any pool starts. One slice per pool,
// sized for `max_seats` (pinned_plan with the start-of-launch buffers, no factor room, bound slabs
// included), plus an overflow region for buffers sized later: per pool the dive log for `dive_rounds`
// (gpu_simplified_dual_phase2_batch_set_dive) and, with CUOPT_BNB_GPU_SEAT_CLOCK, the seat timers.
template <typename i_t, typename f_t>
bool gpu_pinned_arena_reserve(const std::vector<int>& pool_devices,
                              std::size_t max_seats,
                              i_t n_,
                              i_t m_,
                              int dive_rounds,
                              int mode,
                              std::size_t a_nnz)
{
  using namespace gpu_phase2_detail;
  const size_t N = max_seats, n = static_cast<size_t>(n_), m = static_cast<size_t>(m_);
  const auto a16 = [](size_t b) { return (b + 15) / 16 * 16; };
  const pinned_plan_t P = pinned_plan(N, n, m, N, pinned_factor_seat_estimate(a_nnz, m), /*strengthen=*/true);
  const size_t R1       = dive_rounds > 1 ? static_cast<size_t>(dive_rounds - 1) : 0;
  const size_t dive = R1 ? a16(N * sizeof(int)) + a16(N * R1 * sizeof(gpu_dive_record<i_t, f_t>)) + N * R1 * ((n + 3) / 4) : 0;
  const size_t clk  = gpu_seat_clock() ? static_cast<size_t>(std::max(dive_rounds, 1)) * N * sizeof(seat_clock_t) : 0;
  const size_t per_pool = ((dive + 255) / 256 * 256) + ((clk + 255) / 256 * 256);
  if (mode == 1) { return pinned_host::reserve(P.total(true), pool_devices.size(), pool_devices.size() * per_pool); }
  std::map<int, size_t> by_node;
  for (int d : pool_devices) { ++by_node[mode == 3 ? d : pinned_host::device_node(d)]; }
  // CUOPT_BNB_GPU_ARENA_CHUNK: slices per registered region (0 = one region per node). CHOSEN: 4, so
  // pools can start after the first few registrations instead of the whole reservation.
  static const size_t chunk = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_ARENA_CHUNK");
    return e != nullptr ? static_cast<size_t>(std::max(0, std::atoi(e))) : size_t{4};
  }();
  // CUOPT_BNB_GPU_ARENA_SYNC=k: register k regions per GPU before the tree search starts and the
  // rest in the background (0: all before).
  static const size_t sync_per_gpu = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_ARENA_SYNC");
    return e != nullptr ? static_cast<size_t>(std::max(0, std::atoi(e))) : size_t{0};
  }();
  std::map<int, std::set<int>> gpus;
  for (int d : pool_devices) { gpus[mode == 3 ? d : pinned_host::device_node(d)].insert(d); }
  std::vector<size_t> sync;
  for (const auto& [key, slices] : by_node) { sync.push_back(sync_per_gpu * gpus[key].size()); }
  if (mode == 3) {
    return pinned_host::reserve_devices(P.total(true), {by_node.begin(), by_node.end()}, per_pool, chunk, sync);
  }
  return pinned_host::reserve_nodes(P.total(true), {by_node.begin(), by_node.end()}, per_pool, chunk, sync);
}

std::string gpu_batch_pinned_process_report() { return pinned_host::report(); }

std::string gpu_batch_pinned_report()
{
  using namespace gpu_phase2_detail;
  if (!gpu_huge_arena()) { return {}; }
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "[GPUBATCH-PINNED] huge-page arenas=%llu arena_MB=%.1f own mappings (growth / not sized)=%llu own_MB=%.1f\n",
                (unsigned long long)g_arena_n.load(), g_arena_bytes.load() / 1e6,
                (unsigned long long)g_hp_maps.load(), g_hp_bytes.load() / 1e6);
  return buf;
}

template <typename i_t, typename f_t>
gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* gpu_simplified_dual_phase2_batch_launch(
  const std::vector<gpu_batch_item<i_t, f_t>>& items,
  gpu_batch_timing& timing,
  rmm::cuda_stream_view stream,
  bool allow_refactor_exit,
  gpu_phase2_detail::gpu_ro_cache_t<i_t, f_t>* ro_ext,
  bool persistent_seats,
  bool factor_slots)
{
  raft::common::nvtx::range nvtx_scope("GpuDS::batch_launch");
  using namespace gpu_phase2_detail;
  static_assert(sizeof(i_t) == sizeof(int) && sizeof(f_t) == sizeof(double),
                "gpu_simplified_dual_phase2_batch is only instantiated for <int, double>");

  const size_t N = items.size();
  timing = gpu_batch_timing{};
  if (N == 0) return nullptr;
  const double launch_t0 = tic();

  i_t batch_m = -1, batch_n = -1;
  size_t active_seats = 0;
  // Every ft must be a fresh factorization (pure MPF stacking port, no on-device refactor).
  for (size_t b = 0; b < N; ++b) {
    if (items[b].terminal_status != dual::status_t::UNSET) { continue; }
    const auto reject_item = [&](const char* why) {
      timing.rejected = true;
      timing.message  = "item " + std::to_string(b) + " " + why;
      std::fprintf(stderr, "[gpu_phase2] batch ERROR: %s\n", timing.message.c_str());
    };
    if (items[b].lp == nullptr || items[b].settings == nullptr || items[b].vstatus == nullptr ||
        items[b].ft == nullptr || items[b].basic_list == nullptr ||
        items[b].nonbasic_list == nullptr || items[b].sol == nullptr || items[b].dy_se == nullptr) {
      reject_item("has a null state pointer");
      return nullptr;
    }
    const i_t m = items[b].lp->num_rows;
    const i_t n = items[b].lp->num_cols;
    if (active_seats == 0) {
      batch_m = m;
      batch_n = n;
    } else if (m != batch_m || n != batch_n) {
      reject_item("has dimensions different from the first active batch item");
      return nullptr;
    }
    ++active_seats;
    assert(items[b].dy_se->size() == static_cast<size_t>(n));
    if (m < 0 || n < m || items[b].vstatus->size() != static_cast<size_t>(n) ||
        items[b].basic_list->size() != static_cast<size_t>(m) ||
        items[b].nonbasic_list->size() != static_cast<size_t>(n - m) ||
        items[b].sol->x.size() != static_cast<size_t>(n) ||
        items[b].sol->y.size() != static_cast<size_t>(m) ||
        items[b].sol->z.size() != static_cast<size_t>(n) ||
        (items[b].lower != nullptr && items[b].lower->size() != static_cast<size_t>(n)) ||
        (items[b].upper != nullptr && items[b].upper->size() != static_cast<size_t>(n)) ||
        items[b].dy_se->size() != static_cast<size_t>(n)) {
      reject_item("has incorrectly sized state");
      return nullptr;
    }
    // A seat CONTINUING a plunge arrives with the parent's MPF stack on purpose -- that is the CPU's
    // behaviour and the reason a dived child costs no LU. Only a seat that claims to be starting
    // fresh must be fresh.
    if (!items[b].continue_basis && items[b].ft->num_updates() != 0) {
      timing.rejected = true;
      timing.message  = "item " + std::to_string(b) + " has a non-fresh factorization (num_updates=" +
                        std::to_string(items[b].ft->num_updates()) + ")";
      std::fprintf(stderr, "[gpu_phase2] batch ERROR: %s\n", timing.message.c_str());
      return nullptr;
    }
  }
  // ---- Pre-flight VRAM check (fail loud, before any device allocation) ----
  size_t static_bytes = 0;
  std::unordered_map<const void*, std::pair<size_t, size_t>> counted_ro;
  for (size_t b = 0; b < N; ++b) {
    if (items[b].terminal_status != dual::status_t::UNSET) { continue; }
    const auto& item_lp = *items[b].lp;
    size_t item_bytes = problem_static_bytes(item_lp, *items[b].ft);
    const void* ro_key = static_cast<const void*>(item_lp.A.col_start.data());
    const std::pair<size_t, size_t> ro_shape{item_lp.A.col_start.size(), item_lp.A.i.size()};
    const auto [it, inserted] = counted_ro.emplace(ro_key, ro_shape);
    if (!inserted && it->second == ro_shape) { item_bytes -= problem_shared_ro_bytes(item_lp); }
    static_bytes += item_bytes;
  }
  const size_t args_bytes = N * sizeof(kernel_args);
  factor_slots_t slots{};
  const bool use_slots = persistent_seats && factor_slots && active_seats != 0;
  if (use_slots) { size_factor_slots(items, static_cast<size_t>(batch_m), slots); }
  // Device LU of FRESH seats (m <= 256) unless CUOPT_BNB_GPU_DEVICE_LU=0.
  static const bool device_lu_env = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_DEVICE_LU");
    return e == nullptr || std::string(e) != "0";
  }();
  const bool device_lu = use_slots && device_lu_env && batch_m > 0 && batch_m <= 256;
  if (device_lu) {
    slots.capL = slots.capU = slots.acceptL = slots.acceptU = slots.bound;
  }
  // Batch LU kernel: gpu_lu::run_batch_lu (peeling + dense / sparse nucleus), or cuBLAS (dense
  // form + DgetrfBatched + extract) with CUOPT_BNB_GPU_LU_KERNEL=cublas.
  static const bool lu_custom = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LU_KERNEL");
    return e == nullptr || std::string(e) != "cublas";
  }();
  const size_t dense_bytes =
    !device_lu ? 0
    : lu_custom
      ? gpu_lu::batch_lu_workspace_bytes(int(batch_m), int(N)) +
          N * (4 * sizeof(int) + sizeof(gpu_lu::dense_seat_t))
      : N * (size_t(batch_m) * batch_m * sizeof(f_t) + sizeof(f_t*) + size_t(batch_m) * sizeof(int) +
             4 * sizeof(int) + sizeof(gpu_lu::dense_seat_t));
  const size_t slot_bytes =
    use_slots ? N * 2 * (slots.capL + slots.capU) * (sizeof(i_t) + sizeof(f_t)) : 0;
  const gpu_batch_item<i_t, f_t>* first_active = nullptr;
  for (const auto& it : items) {
    if (it.terminal_status == dual::status_t::UNSET) {
      first_active = &it;
      break;
    }
  }
  const bool device_strengthen =
    use_slots && first_active != nullptr && first_active->var_types != nullptr;
  const size_t bs_stride = device_strengthen
                             ? packed_upload_t::align(static_cast<size_t>(batch_m) * (2 * sizeof(f_t) + 2) +
                                                      static_cast<size_t>(batch_n))
                             : 0;
  const size_t need = static_bytes + args_bytes + slot_bytes + N * bs_stride + dense_bytes;
  size_t free_b = 0, total_b = 0;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_b, &total_b));

  // cudaMemGetInfo excludes memory already reserved by the current RMM pool, even though this
  // function's device_buffers allocate from that pool. Counting only free_b therefore rejects a
  // batch merely because the application installed a pool (test_netlib reserves 40% up front).
  // Add that reusable capacity back for this gate. Clamp to total_b so context/driver accounting
  // cannot make the effective figure exceed physical VRAM.
  size_t pool_b = 0;
  auto current_mr = rmm::mr::get_current_device_resource_ref();
  if (auto* pool = ::cuda::mr::resource_cast<rmm::mr::pool_memory_resource>(&current_mr)) {
    pool_b = pool->pool_size();
  }
  size_t available_b = free_b + pool_b;
  if (available_b < free_b || available_b > total_b) { available_b = total_b; }

  timing.vram_needed_bytes = need;
  timing.vram_free_bytes   = available_b;
  // Keep an 8% safety margin for CUDA context/allocator overhead + fragmentation.
  const size_t budget = static_cast<size_t>(0.92 * static_cast<double>(available_b));
  const double GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
  if (need > budget) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "batch needs ~%.2f GB VRAM (static %.2f + args %.2f), only %.2f GB "
                  "available (%.2f free + %.2f current RMM pool; usable budget %.2f GB) for %zu "
                  "problems — reduce --replicas or use a larger GPU",
                  need * GB, static_bytes * GB, args_bytes * GB,
                  available_b * GB, free_b * GB, pool_b * GB, budget * GB, N);
    timing.rejected = true;
    timing.message  = buf;
    std::fprintf(stderr, "[gpu_phase2] batch REJECTED: %s\n", buf);
    return nullptr;
  }

  const double preflight_s = toc(launch_t0);
  // ---- Marshal every problem + upload (host-timed) ----
  // Deduplicate the immutable A CSC, CSR Arow, rhs, and objective across nodes with the same
  // original matrix. Basis-dependent factors, row permutations, and bounds remain per-node.
  const double up_t0 = tic();
  tl_setup           = setup_stats_t{};
  tl_setup.on        = g_setup_telemetry.load(std::memory_order_relaxed);
  {
    static const int div = [] {
      const char* e = std::getenv("CUOPT_BNB_GPU_DENSE_FALLBACK_DIV");
      return e != nullptr ? std::max(0, std::atoi(e)) : static_cast<int>(kSparseReachDenseFallbackDivisor);
    }();
    if (div != kSparseReachDenseFallbackDivisor) {
      const i_t v = div;
      RAFT_CUDA_TRY(cudaMemcpyToSymbol(c_dense_fallback_divisor, &v, sizeof(v)));
    }
    static const int reuse = [] {
      const char* e = std::getenv("CUOPT_BNB_GPU_REUSE_LEVELS");
      return e == nullptr || std::string(e) != "0" ? 1 : 0;
    }();
    if (reuse != 1) { RAFT_CUDA_TRY(cudaMemcpyToSymbol(c_reuse_levels, &reuse, sizeof(reuse))); }
    static const int derive = [] {
      const char* e = std::getenv("CUOPT_BNB_GPU_DERIVE_LEVELS");
      return e == nullptr || std::string(e) != "0" ? 1 : 0;
    }();
    if (derive != 1) { RAFT_CUDA_TRY(cudaMemcpyToSymbol(c_derive_levels, &derive, sizeof(derive))); }
    static const int wl = [] {
      const char* e = std::getenv("CUOPT_BNB_GPU_WARP_LEVELS");
      return e == nullptr || std::string(e) != "0" ? 1 : 0;
    }();
    if (wl != 1) { RAFT_CUDA_TRY(cudaMemcpyToSymbol(c_warp_levels, &wl, sizeof(wl))); }
    static const int lc = [] {
      const char* e = std::getenv("CUOPT_BNB_GPU_LONG_COL");
      return e != nullptr ? std::max(0, std::atoi(e)) : 32;
    }();
    if (lc != 32) { RAFT_CUDA_TRY(cudaMemcpyToSymbol(c_long_col, &lc, sizeof(lc))); }
    static const double dz_pct = [] {
      const char* e = std::getenv("CUOPT_BNB_GPU_DZ_DENSE_PCT");
      return e != nullptr ? std::atof(e) : 30.0;
    }();
    if (dz_pct != 30.0) {
      const f_t v = dz_pct;
      RAFT_CUDA_TRY(cudaMemcpyToSymbol(c_dz_sparse_pct, &v, sizeof(v)));
    }
    static const double hs = [] {
      const char* e = std::getenv("CUOPT_BNB_GPU_HYPERSPARSE_THRESHOLD");
      return e != nullptr ? std::atof(e) : 0.05;
    }();
    if (hs != 0.05) {
      const f_t v = hs;
      RAFT_CUDA_TRY(cudaMemcpyToSymbol(c_hypersparse_threshold, &v, sizeof(v)));
    }
  }
  auto* H = new gpu_batch_handle_t<i_t, f_t>();
  if (pinned_host::reserved()) {
    int dev = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&dev));
    const int node     = pinned_host::key_for_device(dev);
    void* slice_p      = nullptr;
    size_t slice_bytes = 0;
    const int k        = pinned_host::acquire_slice_wait(&slice_p, &slice_bytes, node);
    if (k >= 0) { H->arena = std::make_unique<pinned_arena_t>(k, node, slice_p, slice_bytes); }
  }
  struct tl_arena_guard_t {
    ~tl_arena_guard_t() { tl_pinned_arena = nullptr; }
  } tl_arena_guard;
  tl_pinned_arena = H->arena.get();   // result slabs allocated during the launch come from the slice
  H->items  = items;
  H->up_t0  = up_t0;
  H->stream = stream;
  H->allow_refactor_exit = allow_refactor_exit;
  H->persistent_mode     = persistent_seats;
  H->rows = active_seats == 0 ? 0 : batch_m;
  H->cols = active_seats == 0 ? 0 : batch_n;
  H->active_seats = active_seats;
  double pre_split[5] = {0, 0, 0, 0, 0};  // fixed slabs, factor slots + pack, LU, pinned, stores
  double pre_t = tic();
  const auto pre_lap = [&](int q) {
    const double t = tic();
    pre_split[q] += t - pre_t;
    pre_t = t;
  };
  if (active_seats != 0) {
    H->fixed.emplace(N, static_cast<size_t>(batch_m), static_cast<size_t>(batch_n),
                     static_cast<size_t>(mpf_update_cap()),
                     static_cast<size_t>(mpf_eta_nnz_cap()), stream);
  }
  H->ro_store.stream = stream;
  pre_lap(0);
  if (use_slots) {
    H->use_slots = true;
    H->slots     = slots;
    auto& fs     = H->slots;
    fs.Li  = alloc_uninit<i_t>(N * fs.capL, H->ro_store);
    fs.Lx  = alloc_uninit<f_t>(N * fs.capL, H->ro_store);
    fs.Lti = alloc_uninit<i_t>(N * fs.capL, H->ro_store);
    fs.Ltx = alloc_uninit<f_t>(N * fs.capL, H->ro_store);
    fs.Ui  = alloc_uninit<i_t>(N * fs.capU, H->ro_store);
    fs.Ux  = alloc_uninit<f_t>(N * fs.capU, H->ro_store);
    fs.Uti = alloc_uninit<i_t>(N * fs.capU, H->ro_store);
    fs.Utx = alloc_uninit<f_t>(N * fs.capU, H->ro_store);
    size_t first = 0;
    for (const auto& it : items) {
      if (it.terminal_status != dual::status_t::UNSET) { continue; }
      for (const auto* F : {&it.ft->L0(), &it.ft->U0(), &it.ft->L0_transpose(), &it.ft->U0_transpose()}) {
        first += 2 * packed_upload_t::align(static_cast<size_t>(F->nnz()) * sizeof(f_t)) +
                 2 * (2 * sizeof(void*) + sizeof(size_t));
      }
      // Bounds, permutations, column starts and slab slices of a later FRESH_NODE staging.
      first += 45 * static_cast<size_t>(batch_n) + 40 * static_cast<size_t>(batch_m) + 16 * 64;
    }
    H->pack.pageable = gpu_pageable_setup();
    H->pack.reserve(first, stream);
    if (tl_setup.on) {
      std::fprintf(stderr, "[GPUBATCH-PACK] seats=%zu first_reserve_MB=%.1f pageable=%d\n", N,
                   static_cast<double>(first) / (1024.0 * 1024.0), (int)H->pack.pageable);
    }
  }
  pre_lap(1);
  if (device_lu) {
    H->device_lu  = true;
    H->lu_custom  = lu_custom;
    H->lu_info    = alloc_uninit<int>(3 * N, H->ro_store);
    H->d_lu_batch = alloc_uninit<gpu_lu::dense_seat_t>(N, H->ro_store);
    if (lu_custom) {
      const size_t ws = gpu_lu::batch_lu_workspace_bytes(int(batch_m), int(N));
      if (ws != 0) { H->lu_ws = alloc_uninit<unsigned char>(ws, H->ro_store); }
    } else {
      H->blas     = gpu_lu::create_blas_handle();
      H->lu_dense = alloc_uninit<f_t>(N * size_t(batch_m) * batch_m, H->ro_store);
      H->lu_ptrs  = alloc_uninit<f_t*>(N, H->ro_store);
      H->lu_piv   = alloc_uninit<int>(N * size_t(batch_m), H->ro_store);
      H->lu_cinfo = alloc_uninit<int>(N, H->ro_store);
      if (H->blas == nullptr) { H->device_lu = false; }
    }
  }
  pre_lap(2);
  const unsigned char* d_var_int = nullptr;
  unsigned char* bs_base         = nullptr;
  if (device_strengthen) {
    const auto& vt = *first_active->var_types;
    std::vector<unsigned char> flags(static_cast<size_t>(batch_n), 0);
    for (i_t j = 0; j < batch_n && j < static_cast<i_t>(vt.size()); ++j) {
      flags[j] = vt[j] == variable_type_t::INTEGER || vt[j] == variable_type_t::BINARY;
    }
    d_var_int     = upload(flags, H->ro_store);
    H->d_var_int  = d_var_int;
    bs_base       = alloc_uninit<unsigned char>(N * bs_stride, H->ro_store);
    H->strengthen = true;
  }
  // One cache-line-aligned mapped word is the asynchronous host/device completion handshake for
  // this handle. It is logically a bool, but a 32-bit word has well-defined CUDA and CPU atomic
  // support. The CPU never polls CUDA runtime state in the worker-loop hot path.
  RAFT_CUDA_TRY(cudaHostAlloc(reinterpret_cast<void**>(&H->h_completion_flag),
                              sizeof(std::uint32_t),
                              cudaHostAllocMapped | cudaHostAllocPortable));
  RAFT_CUDA_TRY(cudaHostGetDevicePointer(
    reinterpret_cast<void**>(&H->d_completion_flag), H->h_completion_flag, 0));
  new (H->h_completion_flag) std::uint32_t{0u};
  const cuda::stream_ref host_stream{stream.value()};
  H->h_status.emplace(pinned_make<i_t>(H->arena.get(), host_stream, N * 2));
  H->h_iterations.emplace(pinned_make<i_t>(H->arena.get(), host_stream, N));
  H->h_objective.emplace(pinned_make<f_t>(H->arena.get(), host_stream, N));
  H->h_user_objective.emplace(pinned_make<f_t>(H->arena.get(), host_stream, N));
  pre_lap(3);
  // marshal_storage owns rmm::device_buffers and is move-only, so construct in place.
  H->store.reserve(N);
  for (size_t k = 0; k < N; ++k) { H->store.emplace_back(stream); }
  H->h_args.resize(N);
  auto& h_args   = H->h_args;
  auto& store    = H->store;
  auto& ro_store = H->ro_store;
  H->ro_ext = ro_ext;
  // With a caller-supplied cache the shared block is uploaded once per POOL; without one it falls
  // back to the handle's own cache and is uploaded once per launch, as before. The two caches hold
  // different entry types, so the choice is made at each use site rather than through a reference.
  // `H->ro_store` keeps the per-launch scratch (d_args, the steal counter) either way.
  // One storage per problem, plus one for the deduped read-only uploads. These own every
  // device_buffer for the batch and MUST outlive the kernel launch -- which is why they live on the
  // HANDLE (bound by reference above) rather than on this stack frame: `_launch` returns before the
  // kernel finishes.

  pre_lap(4);
  const double marshal_t0 = tic();
  for (size_t b = 0; b < N; ++b) {
    // A persistent slot seat with no node yet is marshalled from the first active item (so it can be
    // staged later) and marked done.
    const bool templ = items[b].terminal_status != dual::status_t::UNSET && persistent_seats &&
                       H->use_slots && first_active != nullptr;
    const auto& it = templ ? *first_active : items[b];
    if (it.terminal_status != dual::status_t::UNSET) {
      // The kernel checks this before reading any other field. Avoid re-uploading a completed
      // seat's bounds, factorization and workspaces on every host-refactor round.
      h_args[b]      = kernel_args{};
      h_args[b].done = 1;
      continue;
    }
    // Key on A's storage, not on the lp object. Every node of a branch-and-bound tree shares the
    // SAME constraint matrix while carrying its own bounds and LU, so each node is a distinct
    // lp_problem_t -- keying on `it.lp` would miss the sharing that actually exists.
    const ro_upload* ro = nullptr;
    ro_upload ro_copy{};
    if (ro_ext != nullptr && ro_ext->prebuilt) {
      if (H->ro_pre_lp != static_cast<const void*>(it.lp)) {
        H->ro_pre_lp = it.lp;
        H->ro_pre_ok = ro_ext->pre_matches(*it.lp);
        (H->ro_pre_ok ? g_ro_pre_hits : g_ro_pre_misses).fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (ro_ext != nullptr && ro_ext->prebuilt && H->ro_pre_ok) {
      ro = &ro_ext->pre;
    } else if (ro_ext != nullptr) {
      const size_t cs_len = it.lp->A.col_start.size();
      const size_t nz     = it.lp->A.i.size();
      {
        std::lock_guard<std::mutex> lk(ro_ext->mtx);
        auto res = ro_ext->cache.try_emplace(static_cast<const void*>(it.lp->A.col_start.data()));
        auto& e  = res.first->second;
        if (res.second || e.col_start_len != cs_len || e.nz != nz) {
          e.ro            = upload_ro(*it.lp, ro_ext->store);
          e.col_start_len = cs_len;
          e.nz            = nz;
          // The upload is issued on the CACHE's stream, but the seats that read it run on their own
          // pools' streams. Nothing orders those, so without this the first launch on another stream
          // can read the matrix before it has landed. One sync per distinct matrix, not per launch.
          ro_ext->store.stream.synchronize();
        }
        ro_copy = e.ro;   // copy out under the lock; the map may gain entries afterwards
      }
      ro = &ro_copy;
    } else {
      auto res = H->ro_cache.try_emplace(static_cast<const void*>(it.lp->A.col_start.data()));
      if (res.second) { res.first->second = upload_ro(*it.lp, H->ro_store); }
      ro = &res.first->second;
    }
    h_args[b] = marshal_problem(*it.lp, *it.settings, *it.vstatus, *it.ft, *it.basic_list,
                                *it.nonbasic_list, *it.sol, *it.dy_se, ro, store[b],
                                it.refactor_resume, it.terminal_certificate_retry,
                                it.lower, it.upper, &*H->fixed, b,
                                H->use_slots ? &H->slots : nullptr, &H->pack);
    if (H->device_lu) {
      if (H->lu_targets.empty()) { H->lu_targets.resize(N); }
      const kernel_args& a = h_args[b];
      const int cap_l      = static_cast<int>(H->slots.capL);
      const int cap_u      = static_cast<int>(H->slots.capU);
      H->lu_targets[b]     = gpu_lu::dense_seat_t{a.basic_list, a.L0_col_start, a.L0_i, a.L0_x,
                                                  a.U0_col_start, a.U0_i, a.U0_x, a.L0t_col_start,
                                                  a.L0t_i, a.L0t_x, a.U0t_col_start, a.U0t_i,
                                                  a.U0t_x, cap_l, cap_u, a.row_perm,
                                                  a.inv_row_perm, H->lu_info + 3 * b};
    }
    if (H->strengthen) {
      h_args[b].var_int     = d_var_int;
      h_args[b].bs_scratch  = bs_base + b * bs_stride;
      h_args[b].integer_tol = it.settings->integer_tol;
      h_args[b].strengthen  = 1;
    }
    if (templ) { h_args[b].done = 1; }
  }

  const double marshal_s = toc(marshal_t0);
  if (H->fixed) { H->fixed->flush_uploads(); }
  if (H->use_slots) {
    H->pack.flush(stream);
    if (H->pack.pageable) { H->pack.to_pinned(); }
    if (gpu_huge_arena()) {
      create_huge_arena(H);
    } else if (gpu_prealloc_pinned()) {
      prealloc_pinned(H);
    }
  }

  // d_args comes from the pool on this batch's stream. It used to be a raw cudaMalloc plus a
  // BLOCKING cudaMemcpy, which is a device-wide synchronisation point in the middle of the launch
  // path -- fatal for running one batch per GPU worker.
  kernel_args*& d_args = H->d_args;
  d_args = static_cast<kernel_args*>(setup_alloc(N * sizeof(kernel_args), H->ro_store));
  setup_h2d(d_args, h_args.data(), N * sizeof(kernel_args), H->ro_store);

  // The persistent kernel's work-stealing counter, allocated PER LAUNCH from the pool and zeroed
  // stream-ordered. Per launch rather than a device global because branch-and-bound runs one batch
  // per GPU worker concurrently; a shared counter would have those launches stealing from each
  // other's index space. Zeroing on the stream also keeps the launch path free of any device sync.
  unsigned int* d_next = alloc_zeroed<unsigned int>(1, H->ro_store);
  H->d_next            = d_next;
  H->d_completed_blocks = alloc_zeroed<unsigned int>(1, H->ro_store);
  H->d_yield            = alloc_zeroed<unsigned int>(1, H->ro_store);
  H->d_flag_sink        = alloc_zeroed<std::uint32_t>(1, H->ro_store);
  // No device-wide sync: the copies are async and the caller may keep working.
  timing.upload_s = toc(up_t0);
  if (tl_setup.on) {
    timing.setup_alloc_s   = tl_setup.alloc_s;
    timing.setup_alloc_n   = tl_setup.alloc_n;
    timing.setup_h2d_s     = tl_setup.h2d_s;
    timing.setup_h2d_n     = tl_setup.h2d_n;
    timing.setup_h2d_bytes = tl_setup.h2d_bytes;
    timing.setup_memset_s  = tl_setup.memset_s;
    timing.setup_memset_n  = tl_setup.memset_n;
    tl_setup.on            = false;
    std::fprintf(stderr,
                 "[GPUBATCH-LAUNCH] seats=%zu preflight=%.3f before_marshal=%.3f (fixed=%.3f "
                 "slots_pack=%.3f lu=%.3f pinned=%.3f stores=%.3f) marshal=%.3f after=%.3f total=%.3f\n",
                 N, preflight_s, marshal_t0 - up_t0, pre_split[0], pre_split[1], pre_split[2],
                 pre_split[3], pre_split[4], marshal_s, toc(marshal_t0) - marshal_s, toc(launch_t0));
  }

  // ---- Launch a resident work-stealing grid, timed with CUDA events (device compute only) ----
  report_config("monolith", items.size());
  // (upload_solve_class_config() was dropped with g_elide_noop -- all it did was set that flag.)
  // Resident grid: launch only as many blocks as actually fit, and let them steal the remaining
  // problems. This is what removes the intra-launch tail -- a block finishing a 3-iteration node
  // takes another instead of idling until the 256-iteration node in the same launch finishes.
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  H->device = device;
  int sm_count = 1;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 1;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, gpu_phase2_kernel_direct<>, 32, 0));
  unsigned int nb = static_cast<unsigned int>(blocks_per_sm) * static_cast<unsigned int>(sm_count);
  if (nb == 0 || nb > N) { nb = static_cast<unsigned int>(N); }
  // Events live on the HANDLE, not the stack: _launch returns before the kernel finishes, and
  // _finish reads them later to report kernel time.
  cudaEvent_t& ev_start = H->ev_start;
  cudaEvent_t& ev_stop  = H->ev_stop;
  RAFT_CUDA_TRY(cudaEventCreate(&ev_start));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_pre));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_lu0));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_lu1));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_h2d));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_deltas));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_d2h));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_d2h0));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_g0));
  RAFT_CUDA_TRY(cudaEventCreate(&H->ev_g1));
  RAFT_CUDA_TRY(cudaEventCreate(&ev_stop));
  RAFT_CUDA_TRY(cudaEventRecord(ev_start, stream.value()));
  H->timed = true;
  {
    // The register/occupancy target is a BUILD-time knob on the Direct kernel
    // gpu_phase2_kernel_direct.
    // `true` compiles IN the refactor exits so each problem can come back REFACTOR_NEEDED for a
    // caller that owns a host resume loop; `false` compiles them out for resume-less callers.
    if (allow_refactor_exit) {
      gpu_phase2_kernel_direct<true><<<nb, 32, 0, stream.value()>>>(
        d_args,
        static_cast<i_t>(N),
        d_next,
        H->d_completed_blocks,
        H->d_completion_flag);
    } else {
      gpu_phase2_kernel_direct<><<<nb, 32, 0, stream.value()>>>(
        d_args,
        static_cast<i_t>(N),
        d_next,
        H->d_completed_blocks,
        H->d_completion_flag);
    }
  }
  RAFT_CUDA_TRY(cudaEventRecord(ev_stop, stream.value()));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  H->launched = true;
  return H;
}

// True once the kernel has finished. Non-blocking.
template <typename i_t, typename f_t>
gpu_ro_cache<i_t, f_t>* gpu_ro_cache_create()
{
  return new gpu_phase2_detail::gpu_ro_cache_t<i_t, f_t>();
}

template <typename i_t, typename f_t>
gpu_ro_cache<i_t, f_t>* gpu_ro_cache_create_borrowed(rmm::cuda_stream_view borrowed)
{
  return new gpu_phase2_detail::gpu_ro_cache_t<i_t, f_t>(borrowed);
}

template <typename i_t, typename f_t>
void gpu_ro_cache_destroy(gpu_ro_cache<i_t, f_t>* c)
{
  delete c;
}

template <typename i_t, typename f_t>
void gpu_ro_cache_prebuild(gpu_ro_cache<i_t, f_t>* c, const lp_problem_t<i_t, f_t>& lp)
{
  c->pre = gpu_phase2_detail::upload_ro(lp, c->store);
  c->store.stream.synchronize();
  c->pre_col_start = lp.A.col_start;
  c->pre_i.assign(lp.A.i.begin(), lp.A.i.begin() + lp.A.nnz());
  c->pre_x.assign(lp.A.x.begin(), lp.A.x.begin() + lp.A.nnz());
  c->pre_rhs       = lp.rhs;
  c->pre_objective = lp.objective;
  c->prebuilt      = true;
}

std::string gpu_ro_prebuild_report()
{
  const auto h = gpu_phase2_detail::g_ro_pre_hits.load(), m = gpu_phase2_detail::g_ro_pre_misses.load();
  if (h + m == 0) { return {}; }
  char buf[160];
  std::snprintf(buf, sizeof(buf), "[GPUBATCH-RO] prebuilt block: LP matches=%llu mismatches (cache path)=%llu\n",
                (unsigned long long)h, (unsigned long long)m);
  return buf;
}

void gpu_batch_set_setup_telemetry(bool on)
{
  gpu_phase2_detail::g_setup_telemetry.store(on, std::memory_order_relaxed);
}

static thread_local double tl_relaunch_split[4] = {0.0, 0.0, 0.0, 0.0};
const double* gpu_batch_last_relaunch_split() { return tl_relaunch_split; }
// Harvest enqueue distribution: [0..5] whole enqueue, [6..11] the copy submit call alone; buckets
// <0.1, 0.1-0.5, 0.5-2, 2-10, 10-100, >=100 ms; [12], [13] summed seconds (x1e6, us).
static std::atomic<std::uint64_t> g_enqueue_hist[16];  // [14], [15]: first-harvest slab allocations (count, us)
static void enqueue_hist_add(int base, double s)
{
  const double ms = 1e3 * s;
  const int b = ms < 0.1 ? 0 : ms < 0.5 ? 1 : ms < 2.0 ? 2 : ms < 10.0 ? 3 : ms < 100.0 ? 4 : 5;
  g_enqueue_hist[base + b].fetch_add(1, std::memory_order_relaxed);
  g_enqueue_hist[12 + base / 6].fetch_add(static_cast<std::uint64_t>(1e6 * s), std::memory_order_relaxed);
}
void gpu_batch_enqueue_histogram(std::uint64_t* out)
{
  for (int k = 0; k < 16; ++k) { out[k] = g_enqueue_hist[k].load(); }
}
static thread_local double tl_stage_split[4] = {0.0, 0.0, 0.0, 0.0};
double* gpu_batch_stage_split() { return tl_stage_split; }

void gpu_call_stats_begin()
{
  gpu_phase2_detail::tl_setup    = gpu_phase2_detail::setup_stats_t{};
  gpu_phase2_detail::tl_setup.on = true;
}

gpu_call_stats gpu_call_stats_end()
{
  const auto& s = gpu_phase2_detail::tl_setup;
  gpu_call_stats out;
  out.alloc_s   = s.alloc_s;
  out.alloc_n   = s.alloc_n;
  out.h2d_s     = s.h2d_s;
  out.h2d_n     = s.h2d_n;
  out.h2d_bytes = s.h2d_bytes;
  out.memset_s  = s.memset_s;
  out.memset_n  = s.memset_n;
  gpu_phase2_detail::tl_setup.on = false;
  return out;
}

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_ready(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H)
{
  if (H == nullptr || !H->launched) { return true; }
  return std::atomic_ref<std::uint32_t>(*H->h_completion_flag)
           .load(std::memory_order_acquire) != 0u;
}

// Wait for the kernel, copy every result back, release the batch. Destroys the handle.
template <typename i_t, typename f_t>
static void enqueue_seat_rest(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H, size_t b)
{
  const size_t m = H->rows, n = H->cols;
  const auto get = [&](auto& host, const auto& dev, size_t w) {
    H->d2h.add(host.data() + b * w, dev.data() + b * w, w * sizeof(host[0]));
  };
  get(H->h_basic_list, H->fixed->basic_list, m);
  get(H->h_nonbasic_list, H->fixed->nonbasic_list, n - m);
  get(H->h_dy_se, H->fixed->dy_se, n);
  get(H->h_y, H->fixed->y, m);
  get(H->h_z, H->fixed->z, n);
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_fetch_seat(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
                                                 std::size_t seat)
{
  if (H == nullptr || !H->slim_harvest || H->launched) { return; }
  if (H->compact_valid && H->seat_kind[seat] == 3) { return; }
  enqueue_seat_rest(H, seat);
  H->d2h.submit(H->stream.value());
  H->stream.synchronize();
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_read_seat(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
                                                std::size_t seat,
                                                std::vector<variable_status_t>& vstatus,
                                                std::vector<f_t>& lower,
                                                std::vector<f_t>& upper,
                                                std::vector<f_t>& dy_se)
{
  if (H == nullptr || H->launched || seat >= H->h_args.size()) {
    throw std::runtime_error("read_seat: handle must be harvested and the seat in range");
  }
  const auto& a = H->h_args[seat];
  if (a.vstatus == nullptr || a.lower == nullptr || a.upper == nullptr || a.dy_se == nullptr) {
    throw std::runtime_error("read_seat: seat has no device state");
  }
  const size_t n        = static_cast<size_t>(a.n);
  const cudaStream_t cs = H->stream.value();
  vstatus.resize(n);
  lower.resize(n);
  upper.resize(n);
  dy_se.resize(n);
  RAFT_CUDA_TRY(cudaMemcpyAsync(vstatus.data(), a.vstatus, n * sizeof(variable_status_t),
                                cudaMemcpyDeviceToHost, cs));
  RAFT_CUDA_TRY(cudaMemcpyAsync(lower.data(), a.lower, n * sizeof(f_t), cudaMemcpyDeviceToHost, cs));
  RAFT_CUDA_TRY(cudaMemcpyAsync(upper.data(), a.upper, n * sizeof(f_t), cudaMemcpyDeviceToHost, cs));
  RAFT_CUDA_TRY(cudaMemcpyAsync(dy_se.data(), a.dy_se, n * sizeof(f_t), cudaMemcpyDeviceToHost, cs));
  RAFT_CUDA_TRY(cudaStreamSynchronize(cs));
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_fetch_seats(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
                                                  const std::vector<int>& seats)
{
  if (H == nullptr || !H->slim_harvest || H->launched || seats.empty()) { return; }
  if (H->compact_valid) {
    std::vector<int> rest;
    for (int s : seats) {
      if (H->seat_kind[s] != 3) { rest.push_back(s); }
    }
    if (rest.size() != seats.size()) {
      H->compact_valid = false;  // no recursion into this filter
      gpu_simplified_dual_phase2_batch_fetch_seats(H, rest);
      H->compact_valid = true;
      return;
    }
  }
  const size_t N = H->h_args.size(), m = H->rows, n = H->cols;
  if (seats.size() * 8 > N) {  // whole slabs: 5 copies instead of 5 per seat
    const auto get = [&](auto& host, const auto& dev, size_t w) {
      H->d2h.add(host.data(), dev.data(), N * w * sizeof(host[0]));
    };
    get(H->h_basic_list, H->fixed->basic_list, m);
    get(H->h_nonbasic_list, H->fixed->nonbasic_list, n - m);
    get(H->h_dy_se, H->fixed->dy_se, n);
    get(H->h_y, H->fixed->y, m);
    get(H->h_z, H->fixed->z, n);
  } else {
    for (int s : seats) { enqueue_seat_rest(H, static_cast<size_t>(s)); }
  }
  H->d2h.submit(H->stream.value());
  H->stream.synchronize();
}

// Per-device overlap telemetry (pool telemetry): each launch's device intervals as event times
// relative to one reference event per device, and (with CUOPT_BNB_GPU_SEAT_CLOCK) seat-busy time in
// 50 us bins of the device's global timer.
namespace {
struct dev_overlap_t {
  std::mutex mu;
  cudaEvent_t ref{nullptr};
  std::vector<std::array<float, 4>> iv;  // ms from ref: pre-kernel start, solve start, solve end, done
  unsigned long long origin_ns{0};
  std::vector<double> bins;  // seat-busy ns per bin
};
constexpr int overlap_max_devices = 64;
constexpr double seat_bin_ns      = 50000.0;
dev_overlap_t g_overlap[overlap_max_devices];
}  // namespace

template <typename handle_t>
static void overlap_record(handle_t* H, cudaEvent_t done_ev)
{
  if (H->device < 0 || H->device >= overlap_max_devices || !H->timed) { return; }
  auto& D = g_overlap[H->device];
  std::lock_guard<std::mutex> lk(D.mu);
  if (D.ref == nullptr) {
    RAFT_CUDA_TRY(cudaEventCreate(&D.ref));
    RAFT_CUDA_TRY(cudaEventRecord(D.ref, H->stream.value()));
    RAFT_CUDA_TRY(cudaEventSynchronize(D.ref));
  }
  std::array<float, 4> v{};
  RAFT_CUDA_TRY(cudaEventElapsedTime(&v[0], D.ref, H->ev_pre));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&v[1], D.ref, H->ev_start));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&v[2], D.ref, H->ev_stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&v[3], D.ref, done_ev));
  D.iv.push_back(v);
}

static void seat_bins_add(const gpu_phase2_detail::seat_clock_t* c, int rounds, size_t N, int device)
{
  if (device < 0 || device >= overlap_max_devices) { return; }
  auto& D = g_overlap[device];
  std::lock_guard<std::mutex> lk(D.mu);
  for (size_t k = 0; k < static_cast<size_t>(rounds) * N; ++k) {
    const auto& e = c[k];
    if (e.ran == 0 || e.t1 <= e.t0) { continue; }
    if (D.origin_ns == 0) { D.origin_ns = e.t0; }
    if (e.t1 <= D.origin_ns) { continue; }
    const double a = static_cast<double>(std::max(e.t0, D.origin_ns) - D.origin_ns);
    const double b = static_cast<double>(e.t1 - D.origin_ns);
    const size_t last = static_cast<size_t>(b / seat_bin_ns);
    if (D.bins.size() <= last) { D.bins.resize(last + 1, 0.0); }
    for (size_t i = static_cast<size_t>(a / seat_bin_ns); i <= last; ++i) {
      const double lo = std::max(a, i * seat_bin_ns), hi = std::min(b, (i + 1) * seat_bin_ns);
      if (hi > lo) { D.bins[i] += hi - lo; }
    }
  }
}

// Sweep of [a, b) intervals over the central half of their span: fraction of time with >= 1 open,
// mean and max number open.
static void overlap_sweep(std::vector<std::pair<double, double>> iv, double& busy, double& mean, int& peak)
{
  busy = mean = 0.0;
  peak = 0;
  if (iv.empty()) { return; }
  double lo = iv[0].first, hi = iv[0].second;
  for (auto& p : iv) { lo = std::min(lo, p.first), hi = std::max(hi, p.second); }
  const double w0 = lo + (hi - lo) / 4, w1 = hi - (hi - lo) / 4;
  std::vector<std::pair<double, int>> ev;
  for (auto& p : iv) {
    const double a = std::max(p.first, w0), b = std::min(p.second, w1);
    if (b > a) { ev.push_back({a, 1}), ev.push_back({b, -1}); }
  }
  std::sort(ev.begin(), ev.end());
  int cur = 0;
  double last = w0, on = 0.0, area = 0.0;
  for (auto& [t, d] : ev) {
    if (cur > 0) { on += t - last; }
    area += cur * (t - last);
    last = t;
    cur += d;
    peak = std::max(peak, cur);
  }
  busy = (w1 > w0) ? on / (w1 - w0) : 0.0;
  mean = (w1 > w0) ? area / (w1 - w0) : 0.0;
}

std::string gpu_batch_overlap_report()
{
  std::string out;
  char line[640];
  for (int dev = 0; dev < overlap_max_devices; ++dev) {
    auto& D = g_overlap[dev];
    std::lock_guard<std::mutex> lk(D.mu);
    if (D.iv.empty() && D.bins.empty()) { continue; }
    std::vector<std::pair<double, double>> solve, chain;
    for (auto& v : D.iv) {
      solve.push_back({v[1], v[2]});
      chain.push_back({v[0], v[3]});
    }
    double sb, sm, cb, cm;
    int sp, cp;
    overlap_sweep(solve, sb, sm, sp);
    overlap_sweep(chain, cb, cm, cp);
    int sm_count = 0, bps = 0, prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) == cudaSuccess) {
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, gpu_phase2_detail::gpu_phase2_kernel_direct<true, true>, 32, 0);
      cudaSetDevice(prev);
    }
    std::snprintf(line, sizeof(line),
      "[GPUBATCH-OVERLAP] gpu=%d launches=%zu (central half of the span) solve kernels: busy=%.1f%% "
      "mean_concurrent=%.2f max=%d | launch device chain (pre-kernel start..done): busy=%.1f%% "
      "mean_concurrent=%.2f max=%d",
      dev, D.iv.size(), 100 * sb, sm, sp, 100 * cb, cm, cp);
    out += line;
    if (!D.bins.empty()) {
      size_t a = 0, b = D.bins.size();
      while (a < b && D.bins[a] == 0) { ++a; }
      while (b > a && D.bins[b - 1] == 0) { --b; }
      const size_t w0 = a + (b - a) / 4, w1 = b - (b - a) / 4;
      std::vector<double> r;
      for (size_t i = w0; i < w1; ++i) { r.push_back(D.bins[i] / seat_bin_ns); }
      if (!r.empty()) {
        double sum = 0;
        for (double x : r) { sum += x; }
        std::sort(r.begin(), r.end());
        const auto q = [&](double p) { return r[static_cast<size_t>(p * (r.size() - 1))]; };
        std::snprintf(line, sizeof(line),
          " | resident solving seats (seat clock, 50 us bins): mean=%.0f p10=%.0f p50=%.0f p90=%.0f "
          "max=%.0f capacity=%d",
          sum / r.size(), q(0.1), q(0.5), q(0.9), r.back(), sm_count * bps);
        out += line;
      }
    }
    out += "\n";
  }
  return out;
}

// CUOPT_BNB_GPU_SEAT_CLOCK aggregate over launches with >= 32 seats in a round.
namespace {
struct clock_agg_t {
  std::mutex mu;
  static constexpr int R = 8;
  std::uint64_t launches[R]{}, seats[R]{}, hist[R][6]{};
  double med[R]{}, p90[R]{}, mx[R]{}, span[R]{}, tail[R]{}, idle[R]{}, cyc_ratio[R]{}, it_med[R]{},
    it_max[R]{}, it_slowest[R]{};
  double ph[gpu_phase2_detail::kSolvePhases]{}, cyc_total{0.0}, it_total{0.0};
  double nlev_sum[4]{}, nlev_n{0.0}, route[3]{}, sub[8]{}, pre[6]{}, legs{0.0}, wop[4]{}, wn[4]{};
  double path[gpu_phase2_detail::kPaths]{};
  int nlev_max[4]{};
};
clock_agg_t g_clock;
}  // namespace

static void seat_clock_add(const gpu_phase2_detail::seat_clock_t* c, int rounds, size_t N, int device)
{
  seat_bins_add(c, rounds, N, device);
  {
    double ph[gpu_phase2_detail::kSolvePhases]{}, cyc = 0.0, its = 0.0;
    for (size_t k = 0; k < static_cast<size_t>(rounds) * N; ++k) {
      if (c[k].ran == 0) { continue; }
      cyc += static_cast<double>(c[k].cycles);
      its += static_cast<double>(c[k].iterations);
      if (c[k].nlev[0] >= 0) {
        std::lock_guard<std::mutex> lk(g_clock.mu);
        g_clock.nlev_n += 1;
        for (int q = 0; q < 3; ++q) { g_clock.route[q] += c[k].route[q]; }
        for (int q = 0; q < gpu_phase2_detail::kPaths; ++q) { g_clock.path[q] += c[k].path[q]; }
        for (int q = 0; q < 8; ++q) { g_clock.sub[q] += static_cast<double>(c[k].sub[q]); }
        for (int q = 0; q < 4; ++q) {
          g_clock.wop[q] += static_cast<double>(c[k].wop[q]);
          g_clock.wn[q] += c[k].wn[q];
        }
        for (int q = 0; q < 4; ++q) {
          g_clock.nlev_sum[q] += c[k].nlev[q];
          g_clock.nlev_max[q] = std::max(g_clock.nlev_max[q], c[k].nlev[q]);
        }
      }
      for (int p = 0; p < gpu_phase2_detail::kSolvePhases; ++p) { ph[p] += static_cast<double>(c[k].ph[p]); }
      std::lock_guard<std::mutex> lk(g_clock.mu);
      g_clock.legs += 1;
      for (int q = 0; q < 6; ++q) { g_clock.pre[q] += static_cast<double>(c[k].pre[q]); }
    }
    std::lock_guard<std::mutex> lk(g_clock.mu);
    g_clock.cyc_total += cyc;
    g_clock.it_total += its;
    for (int p = 0; p < gpu_phase2_detail::kSolvePhases; ++p) { g_clock.ph[p] += ph[p]; }
  }
  std::vector<double> dur, ends;
  std::vector<long long> cyc;
  std::vector<int> its;
  for (int r = 0; r < rounds && r < clock_agg_t::R; ++r) {
    dur.clear(), ends.clear(), cyc.clear(), its.clear();
    unsigned long long lo = ~0ull, hi = 0;
    double slow = -1.0;
    int slow_it = 0;
    for (size_t b = 0; b < N; ++b) {
      const auto& e = c[r * N + b];
      if (e.ran == 0 || e.t1 < e.t0) { continue; }
      const double d = 1e-3 * static_cast<double>(e.t1 - e.t0);  // us
      dur.push_back(d);
      cyc.push_back(e.cycles);
      its.push_back(e.iterations);
      lo = std::min(lo, e.t0);
      hi = std::max(hi, e.t1);
      if (d > slow) { slow = d, slow_it = e.iterations; }
    }
    if (dur.size() < 32) { continue; }
    for (size_t b = 0; b < N; ++b) {
      const auto& e = c[r * N + b];
      if (e.ran != 0 && e.t1 >= e.t0) { ends.push_back(1e-3 * static_cast<double>(e.t1 - lo)); }
    }
    const auto q = [](auto& v, double p) {
      const size_t k = static_cast<size_t>(p * (v.size() - 1));
      std::nth_element(v.begin(), v.begin() + k, v.end());
      return v[k];
    };
    double total = 0.0;
    for (double d : dur) { total += d; }
    const double span = 1e-3 * static_cast<double>(hi - lo);
    const double m = q(dur, 0.5), p9 = q(dur, 0.9), x = q(dur, 1.0);
    const double e9 = q(ends, 0.9);
    const double cm = static_cast<double>(q(cyc, 0.5)), cx = static_cast<double>(q(cyc, 1.0));
    const double im = q(its, 0.5), ix = q(its, 1.0);
    const double ratio = m > 0 ? x / m : 0.0;
    const int bucket = ratio < 1.5 ? 0 : ratio < 2 ? 1 : ratio < 3 ? 2 : ratio < 5 ? 3 : ratio < 10 ? 4 : 5;
    std::lock_guard<std::mutex> lk(g_clock.mu);
    ++g_clock.launches[r];
    g_clock.seats[r] += dur.size();
    ++g_clock.hist[r][bucket];
    g_clock.med[r] += m, g_clock.p90[r] += p9, g_clock.mx[r] += x, g_clock.span[r] += span;
    g_clock.tail[r] += span - e9;
    g_clock.idle[r] += span > 0 ? 1.0 - total / (static_cast<double>(dur.size()) * span) : 0.0;
    g_clock.cyc_ratio[r] += cm > 0 ? cx / cm : 0.0;
    g_clock.it_med[r] += im, g_clock.it_max[r] += ix, g_clock.it_slowest[r] += slow_it;
  }
}

std::string gpu_fission_step_report()
{
  using namespace gpu_phase2_detail;
  unsigned long long h[kFissionSteps][2];
  if (cudaMemcpyFromSymbol(h, g_fstep, sizeof(h)) != cudaSuccess) { return {}; }
  static const char* names[kFissionSteps] = {
    "price",
    "btran.zero", "btran.Urow", "btran.tloop", "btran.compact", "btran.Lscatter", "btran.out",
    "choose.norm", "choose.se_reset", "choose.dz", "choose.ratio", "choose.dual", "choose.prep",
    "ftran.zero", "ftran.Lscatter", "ftran.tloop", "ftran.compact", "ftran.Uscatter", "ftran.out",
    "update.pivcheck", "update.dx", "update.primal", "update.obj", "update.cutoff", "update.se_norms",
    "update.swap", "update.refcheck", "update.mpf", "update.infeas", "update.tail", "update.join",
    "update.p.dx", "update.p.primal", "update.p.obj", "update.p.cutoff", "update.p.se_norms",
    "update.p.swap", "tl.dot", "tl.axpy", "tl.updates"};
  const double iters = static_cast<double>(h[S_U_TAIL][1]);
  if (iters == 0) { return {}; }
  if (const char* ut = std::getenv("CUOPT_BNB_GPU_FISSION_UTRACE"); ut != nullptr && *ut != 0) {
    fission_utrace_t* d = nullptr;
    unsigned long long n = 0;
    cudaMemcpyFromSymbol(&d, g_utrace, sizeof(d));
    cudaMemcpyFromSymbol(&n, g_utrace_n, sizeof(n));
    if (d != nullptr) {
      const unsigned long long k = n < kUtraceCap ? n : kUtraceCap;
      std::vector<fission_utrace_t> h_tr(k);
      cudaMemcpy(h_tr.data(), d, k * sizeof(fission_utrace_t), cudaMemcpyDeviceToHost);
      if (FILE* f = std::fopen(ut, "wb")) {
        std::fwrite(h_tr.data(), sizeof(fission_utrace_t), k, f);
        std::fclose(f);
      }
    }
  }
  if (const char* ut = std::getenv("CUOPT_BNB_GPU_FISSION_CTRACE"); ut != nullptr && *ut != 0) {
    fission_utrace_t* d = nullptr;
    unsigned long long n = 0;
    cudaMemcpyFromSymbol(&d, g_ctrace, sizeof(d));
    cudaMemcpyFromSymbol(&n, g_ctrace_n, sizeof(n));
    if (d != nullptr) {
      const unsigned long long k = n < kUtraceCap ? n : kUtraceCap;
      std::vector<fission_utrace_t> h_tr(k);
      cudaMemcpy(h_tr.data(), d, k * sizeof(fission_utrace_t), cudaMemcpyDeviceToHost);
      if (FILE* f = std::fopen(ut, "wb")) {
        std::fwrite(h_tr.data(), sizeof(fission_utrace_t), k, f);
        std::fclose(f);
      }
    }
  }
  if (const char* ut = std::getenv("CUOPT_BNB_GPU_FISSION_STRACE"); ut != nullptr && *ut != 0) {
    fission_utrace_t* d = nullptr;
    unsigned long long n = 0;
    cudaMemcpyFromSymbol(&d, g_strace, sizeof(d));
    cudaMemcpyFromSymbol(&n, g_strace_n, sizeof(n));
    if (d != nullptr) {
      const unsigned long long k = n < kUtraceCap ? n : kUtraceCap;
      std::vector<fission_utrace_t> h_tr(k);
      cudaMemcpy(h_tr.data(), d, k * sizeof(fission_utrace_t), cudaMemcpyDeviceToHost);
      if (FILE* f = std::fopen(ut, "wb")) {
        std::fwrite(h_tr.data(), sizeof(fission_utrace_t), k, f);
        std::fclose(f);
      }
    }
  }
  if (const char* ut = std::getenv("CUOPT_BNB_GPU_FISSION_FTRACE"); ut != nullptr && *ut != 0) {
    fission_utrace_t* d = nullptr;
    unsigned long long n = 0;
    cudaMemcpyFromSymbol(&d, g_ftrace, sizeof(d));
    cudaMemcpyFromSymbol(&n, g_ftrace_n, sizeof(n));
    if (d != nullptr) {
      const unsigned long long k = n < kUtraceCap ? n : kUtraceCap;
      std::vector<fission_utrace_t> h_tr(k);
      cudaMemcpy(h_tr.data(), d, k * sizeof(fission_utrace_t), cudaMemcpyDeviceToHost);
      if (FILE* f = std::fopen(ut, "wb")) {
        std::fwrite(h_tr.data(), sizeof(fission_utrace_t), k, f);
        std::fclose(f);
      }
    }
  }
  double total = 0;
  for (int k = 0; k < S_UP_DX; ++k) { total += static_cast<double>(h[k][0]); }
  std::string out;
  char line[256];
  std::snprintf(line, sizeof(line),
                "[FISSION-STEPS] completed iterations=%.0f, seat cycles per iteration=%.0f (per seat, "
                "excluding waits for other seats)\n",
                iters, total / iters);
  out += line;
  for (int k = 0; k < kFissionSteps; ++k) {
    if (h[k][1] == 0) { continue; }
    std::snprintf(line, sizeof(line),
                  "[FISSION-STEPS] %-16s calls/iteration=%5.2f cycles/call=%8.0f cycles/iteration=%8.0f share=%5.1f%%\n",
                  names[k], h[k][1] / iters, static_cast<double>(h[k][0]) / h[k][1],
                  static_cast<double>(h[k][0]) / iters, 100.0 * h[k][0] / total);
    out += line;
  }
  return out;
}

std::string gpu_batch_seat_clock_report()
{
  std::lock_guard<std::mutex> lk(g_clock.mu);
  std::string out;
  char line[640];
  for (int r = 0; r < clock_agg_t::R; ++r) {
    const double n = static_cast<double>(g_clock.launches[r]);
    if (n == 0) { continue; }
    std::snprintf(line, sizeof(line),
      "[GPUBATCH-CLOCK] round=%d launches=%llu seats/launch=%.0f seat_us med=%.0f p90=%.0f max=%.0f "
      "span_us=%.0f after_p90_end_us=%.0f idle_slot_frac=%.2f max/med(cycles)=%.1f iters med=%.1f "
      "max=%.1f slowest_seat=%.1f max/med<1.5,2,3,5,10,>=10=%llu,%llu,%llu,%llu,%llu,%llu\n",
      r, (unsigned long long)g_clock.launches[r], g_clock.seats[r] / n, g_clock.med[r] / n,
      g_clock.p90[r] / n, g_clock.mx[r] / n, g_clock.span[r] / n, g_clock.tail[r] / n,
      g_clock.idle[r] / n, g_clock.cyc_ratio[r] / n, g_clock.it_med[r] / n, g_clock.it_max[r] / n,
      g_clock.it_slowest[r] / n, (unsigned long long)g_clock.hist[r][0],
      (unsigned long long)g_clock.hist[r][1], (unsigned long long)g_clock.hist[r][2],
      (unsigned long long)g_clock.hist[r][3], (unsigned long long)g_clock.hist[r][4],
      (unsigned long long)g_clock.hist[r][5]);
    out += line;
  }
  if (g_clock.cyc_total > 0) {
    double in_loop = 0.0;
    for (double v : g_clock.ph) { in_loop += v; }
    const auto f = [&](double v) { return 100.0 * v / g_clock.cyc_total; };
    std::snprintf(line, sizeof(line),
      "[GPUBATCH-PHASE] cycles/iteration=%.0f | %% of seat cycles: setup/other=%.1f price=%.1f btran=%.1f dz=%.1f ratio=%.1f "
      "dual_upd=%.1f ftran=%.1f dx+primal=%.1f cutoff=%.1f se_norms=%.1f basis_upd=%.1f infeas=%.1f\n",
      g_clock.it_total > 0 ? g_clock.cyc_total / g_clock.it_total : 0.0, f(g_clock.cyc_total - in_loop), f(g_clock.ph[0]), f(g_clock.ph[1]), f(g_clock.ph[2]),
      f(g_clock.ph[3]), f(g_clock.ph[4]), f(g_clock.ph[5]), f(g_clock.ph[6]), f(g_clock.ph[7]),
      f(g_clock.ph[8]), f(g_clock.ph[9]), f(g_clock.ph[10]));
    out += line;
  }
  if (g_clock.nlev_n > 0) {
    std::snprintf(line, sizeof(line),
      "[GPUBATCH-LEVELS] wavefront levels mean (max): L=%.0f (%d) U=%.0f (%d) Lt=%.0f (%d) Ut=%.0f (%d) | "
      "solves per iteration: dense=%.2f sparse=%.2f sparse->dense=%.2f\n",
      g_clock.nlev_sum[0] / g_clock.nlev_n, g_clock.nlev_max[0], g_clock.nlev_sum[1] / g_clock.nlev_n,
      g_clock.nlev_max[1], g_clock.nlev_sum[2] / g_clock.nlev_n, g_clock.nlev_max[2],
      g_clock.nlev_sum[3] / g_clock.nlev_n, g_clock.nlev_max[3], g_clock.route[0] / std::max(1.0, g_clock.it_total),
      g_clock.route[1] / std::max(1.0, g_clock.it_total), g_clock.route[2] / std::max(1.0, g_clock.it_total));
    out += line;
    const double it = std::max(1.0, g_clock.it_total);
    std::snprintf(line, sizeof(line),
      "[GPUBATCH-SOLVES] cycles per iteration: dense_wavefront=%.0f reach_dfs=%.0f bucket=%.0f "
      "sparse_wavefront=%.0f mpf_replay dense=%.0f pair=%.0f sparse=%.0f preload=%.0f\n",
      g_clock.sub[0] / it, g_clock.sub[1] / it, g_clock.sub[2] / it, g_clock.sub[3] / it,
      g_clock.sub[4] / it, g_clock.sub[6] / it, g_clock.sub[7] / it, g_clock.sub[5] / it);
    out += line;
    std::snprintf(line, sizeof(line),
      "[GPUBATCH-WAVE] dense wavefront per operator L U Lt Ut: solves/iteration=%.2f %.2f %.2f %.2f "
      "cycles/solve=%.0f %.0f %.0f %.0f\n",
      g_clock.wn[0] / it, g_clock.wn[1] / it, g_clock.wn[2] / it, g_clock.wn[3] / it,
      g_clock.wop[0] / std::max(1.0, g_clock.wn[0]), g_clock.wop[1] / std::max(1.0, g_clock.wn[1]),
      g_clock.wop[2] / std::max(1.0, g_clock.wn[2]), g_clock.wop[3] / std::max(1.0, g_clock.wn[3]));
    out += line;
    {
      using namespace gpu_phase2_detail;
      const double* P = g_clock.path;
      const auto pct  = [](double a, double b) { return b > 0 ? 100.0 * a / b : 0.0; };
      const double upd = P[P_UPD_OK] + P[P_UPD_FULL] + P[P_UPD_MU];
      const double ets = P[P_ETILDE_BUCKET] + P[P_ETILDE_PERM];
      std::snprintf(line, sizeof(line),
        "[GPUBATCH-PATHS] per iteration: se_resets=%.4f dz rowwise/coldot=%.1f%%/%.1f%% "
        "btran Ut sparse=%.1f%% Lt sparse=%.1f%% | ftran L paired=%.1f%% U paired=%.1f%% "
        "sparse_rhs=%.2f of 4 | mpf_updates_replayed=%.1f\n",
        P[P_SE_RESET] / it, pct(P[P_DZ_ROWWISE], P[P_DZ_ROWWISE] + P[P_DZ_COLDOT]),
        pct(P[P_DZ_COLDOT], P[P_DZ_ROWWISE] + P[P_DZ_COLDOT]),
        pct(P[P_UT_SPARSE], P[P_UT_SPARSE] + P[P_UT_DENSE]),
        pct(P[P_LT_SPARSE], P[P_LT_SPARSE] + P[P_LT_DENSE]),
        pct(P[P_L_PAIRED], P[P_L_PAIRED] + P[P_L_SPLIT]),
        pct(P[P_U_PAIRED], P[P_U_PAIRED] + P[P_U_SPLIT]),
        P[P_FTRAN_SPARSE_RHS] / std::max(1.0, P[P_L_PAIRED] + P[P_L_SPLIT]),
        P[P_MPF_LEN] / std::max(1.0, P[P_UT_SPARSE] + P[P_UT_DENSE]));
      out += line;
      std::snprintf(line, sizeof(line),
        "[GPUBATCH-PATHS] updates: ok=%.0f store_full=%.0f mu_reject=%.0f | utilde nnz=%.0f etilde "
        "nnz=%.0f sort bucket/perm=%.1f%%/%.1f%% | nan_recovery=%.0f cutoff_scans=%.0f "
        "se_v dense=%.1f%% | refactor_exits=%.0f | legs: level reuse=%.0f build=%.0f\n",
        P[P_UPD_OK], P[P_UPD_FULL], P[P_UPD_MU], P[P_UTILDE_NZ] / std::max(1.0, upd),
        P[P_ETILDE_NZ] / std::max(1.0, P[P_UPD_OK] + P[P_UPD_MU]), pct(P[P_ETILDE_BUCKET], ets),
        pct(P[P_ETILDE_PERM], ets), P[P_NAN_RECOVERY], P[P_CUTOFF_SCAN],
        pct(P[P_V_DENSE], P[P_V_DENSE] + P[P_V_SCATTER]), P[P_REFACTOR], P[P_LEVEL_REUSE],
        P[P_LEVEL_BUILD]);
      out += line;
    }
    const double lg = std::max(1.0, g_clock.legs);
    std::snprintf(line, sizeof(line),
      "[GPUBATCH-SETUP] legs=%.0f iterations/leg=%.1f | cycles per leg: bounds=%.0f strengthen=%.0f "
      "transpose=%.0f slot_init=%.0f levels=%.0f prologue=%.0f | per iteration: %.0f %.0f %.0f %.0f %.0f %.0f\n",
      g_clock.legs, it / lg, g_clock.pre[0] / lg, g_clock.pre[1] / lg, g_clock.pre[2] / lg,
      g_clock.pre[3] / lg, g_clock.pre[4] / lg, g_clock.pre[5] / lg, g_clock.pre[0] / it,
      g_clock.pre[1] / it, g_clock.pre[2] / it, g_clock.pre[3] / it, g_clock.pre[4] / it,
      g_clock.pre[5] / it);
    out += line;
  }
  return out;
}

// Harvest split (pool telemetry): host intervals and device event spans, summed over launches.
enum harvest_stat_t : int {
  HV_N, HV_TOTAL, HV_ENQUEUE, HV_SYNC1, HV_UNPACK, HV_LP_STATES_COPY, HV_LP_STATES_BYTES, HV_REFETCH, HV_NG,
  HV_D_SCORE, HV_D_GATHER, HV_D_GAP, HV_D_COPY, HV_SUBMIT, HV_COUNT
};
static std::atomic<std::uint64_t> g_harvest[HV_COUNT];
static void harvest_add(harvest_stat_t k, double seconds)
{
  g_harvest[k].fetch_add(k == HV_N || k == HV_NG ? static_cast<std::uint64_t>(seconds)
                                                 : static_cast<std::uint64_t>(1e6 * seconds),
                         std::memory_order_relaxed);
}

std::string gpu_batch_harvest_report()
{
  const double n = static_cast<double>(g_harvest[HV_N].load());
  if (n == 0) { return {}; }
  const double ng = std::max(1.0, static_cast<double>(g_harvest[HV_NG].load()));
  const auto ms = [&](harvest_stat_t k, double d) { return 1e-3 * g_harvest[k].load() / d; };
  char buf[512];
  std::snprintf(buf, sizeof(buf),
    "[GPUBATCH-HARVEST] steady launches=%.0f ms/launch host: total=%.3f enqueue=%.3f first_sync=%.3f "
    "(submit=%.3f) unpack=%.3f (lp_states copy+sync=%.3f, lp_states MB=%.3f) refactor_fetch=%.3f | device (%.0f launches): "
    "last_solve_end->gather_start=%.3f gather=%.3f gather_end->copy_start=%.3f copies=%.3f\n",
    n, ms(HV_TOTAL, n), ms(HV_ENQUEUE, n), ms(HV_SYNC1, n), ms(HV_SUBMIT, n), ms(HV_UNPACK, n), ms(HV_LP_STATES_COPY, n),
    1e-6 * g_harvest[HV_LP_STATES_BYTES].load() / n, ms(HV_REFETCH, n), ng, ms(HV_D_SCORE, ng),
    ms(HV_D_GATHER, ng), ms(HV_D_GAP, ng), ms(HV_D_COPY, ng));
  return buf;
}

// After the compact harvest's first copy: fetch the gathered LP states and scatter them (and the branch
// columns of the other seats) into the host slabs the result views read.
template <typename i_t, typename f_t>
static void compact_unpack(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H)
{
  using namespace gpu_phase2_detail;
  const size_t N = H->h_args.size(), n = H->cols, m = H->rows;
  const char* h  = H->h_compact->data();
  const auto* hd = reinterpret_cast<const compact_hdr_t*>(h);
  const int full = *reinterpret_cast<const int*>(h + H->cmp_off[0]);
  if (full > 0) {
    const size_t need = static_cast<size_t>(full) * H->cmp_lp_state_bytes;
    if (need > H->cmp_lp_states_cap) {
      H->cmp_lp_states_cap = std::max(need, std::min(N * H->cmp_lp_state_bytes, 2 * need));
      H->h_lp_states.emplace(pinned_make<char>(H->arena.get(),
        cuda::stream_ref{H->stream.value()}, H->cmp_lp_states_cap));
    }
    const double t_lp_states = tic();
    RAFT_CUDA_TRY(cudaMemcpyAsync(H->h_lp_states->data(),
                                  static_cast<char*>(H->d_compact->data()) + H->cmp_off[2], need,
                                  cudaMemcpyDeviceToHost, H->stream.value()));
    H->stream.synchronize();
    if (gpu_timing_events()) {
      harvest_add(HV_LP_STATES_COPY, toc(t_lp_states));
      harvest_add(HV_LP_STATES_BYTES, static_cast<double>(need) * 1e-6);  // "seconds" slot holds MB
    }
  }
  H->seat_kind.resize(N);
  for (size_t b = 0; b < N; ++b) {
    const auto& e  = hd[b];
    H->seat_kind[b] = static_cast<unsigned char>(e.kind);
    if (e.kind == 1) {
      const size_t j                   = static_cast<size_t>(H->h_scores->data()[b].best_var);
      H->h_x[b * n + j]                = e.x;
      H->h_lower->data()[b * n + j]    = e.lower;
      H->h_upper->data()[b * n + j]    = e.upper;
    } else if (e.kind == 3) {
      const auto* d = reinterpret_cast<const f_t*>(H->h_lp_states->data() + e.slot * H->cmp_lp_state_bytes);
      std::copy_n(d, n, H->h_x.data() + b * n);
      std::copy_n(d + n, n, H->h_lower->data() + b * n);
      std::copy_n(d + 2 * n, n, H->h_upper->data() + b * n);
      std::copy_n(d + 3 * n, n, H->h_z.data() + b * n);
      std::copy_n(d + 4 * n, n, H->h_dy_se.data() + b * n);
      std::copy_n(d + 5 * n, m, H->h_y.data() + b * m);
      const auto* li = reinterpret_cast<const i_t*>(d + 5 * n + m);
      std::copy_n(li, m, H->h_basic_list.data() + b * m);
      std::copy_n(li + m, n - m, H->h_nonbasic_list.data() + b * (n - m));
      std::copy_n(reinterpret_cast<const variable_status_t*>(li + n), n, H->h_vstatus.data() + b * n);
    }
  }
}

// Queue every returned field's copy (contiguous across seats) on the handle's stream.
template <typename i_t, typename f_t>
static double enqueue_results(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H, bool slim)
{
  using namespace gpu_phase2_detail;
  const size_t N  = H->items.size();
  double hv_submit = 0.0;
  if (H->active_seats == 0) { return hv_submit; }
  const size_t m = static_cast<size_t>(H->rows);
  const size_t n = static_cast<size_t>(H->cols);
  const bool first_alloc = H->h_x.size() != N * n;
  const double alloc_t0  = tic();
  H->h_x.resize(N * n);
  H->h_y.resize(N * m);
  H->h_z.resize(N * n);
  H->h_dy_se.resize(N * n);
  H->h_vstatus.resize(N * n);
  H->h_basic_list.resize(N * m);
  H->h_nonbasic_list.resize(N * (n - m));
  if (first_alloc && gpu_timing_events()) {
    const double dt = toc(alloc_t0);
    g_enqueue_hist[14].fetch_add(1);
    g_enqueue_hist[15].fetch_add(static_cast<std::uint64_t>(1e6 * dt));
  }
  const cudaStream_t cs = H->stream.value();
  auto copy = [H](void* dst, const void* src, size_t bytes) { H->d2h.add(dst, src, bytes); };
  if (H->ev_d2h0 != nullptr && gpu_timing_events()) {
    RAFT_CUDA_TRY(cudaEventRecord(H->ev_d2h0, cs));
  }
  copy(H->h_status->data(), H->fixed->out_status.data(), N * 2 * sizeof(i_t));
  copy(H->h_objective->data(), H->fixed->out_objective.data(), N * sizeof(f_t));
  copy(H->h_user_objective->data(), H->fixed->out_user_objective.data(), N * sizeof(f_t));
  copy(H->h_iterations->data(), H->fixed->out_iterations.data(), N * sizeof(i_t));
  const bool compact = H->compact_valid && slim;
  if (compact) {
    copy(H->h_compact->data(), H->d_compact->data(), H->cmp_off[2]);
  } else {
    copy(H->h_x.data(), H->fixed->x.data(), N * n * sizeof(f_t));
    copy(H->h_vstatus.data(), H->fixed->vstatus.data(), N * n * sizeof(variable_status_t));
  }
  if (!slim) {
    copy(H->h_basic_list.data(), H->fixed->basic_list.data(), N * m * sizeof(i_t));
    copy(H->h_nonbasic_list.data(), H->fixed->nonbasic_list.data(), N * (n - m) * sizeof(i_t));
    copy(H->h_dy_se.data(), H->fixed->dy_se.data(), N * n * sizeof(f_t));
    copy(H->h_y.data(), H->fixed->y.data(), N * m * sizeof(f_t));
    copy(H->h_z.data(), H->fixed->z.data(), N * n * sizeof(f_t));
  }
  if (H->scores_valid) {
    copy(H->h_scores->data(),
         static_cast<char*>(H->d_scores->data()) + n * (2 * sizeof(f_t) + 2 * sizeof(i_t)),
         N * sizeof(gpu_scores::seat_score_t));
  }
  if (H->clk_rounds > 0) {
    copy(H->h_clk->data(), H->d_clk->data(), H->clk_rounds * N * sizeof(seat_clock_t));
  }
  if (H->dive_valid) {
    copy(H->h_dive->data(), static_cast<char*>(H->d_dive->data()) + H->dive_off[1],
         H->d_dive->size() - H->dive_off[1]);
  }
  if (H->strengthen) {
    if (!H->h_lower) {
      const cuda::stream_ref hs{cs};
      H->h_lower.emplace(pinned_make<f_t>(H->arena.get(), hs, N * n));
      H->h_upper.emplace(pinned_make<f_t>(H->arena.get(), hs, N * n));
    }
    if (!compact) {
      copy(H->h_lower->data(), H->fixed->lower.data(), N * n * sizeof(f_t));
      copy(H->h_upper->data(), H->fixed->upper.data(), N * n * sizeof(f_t));
    }
  }
  const double submit_t0 = tic();
  const double hv_submit0 = tic();
  H->d2h.submit(cs);
  hv_submit = toc(hv_submit0);
  if (H->ev_d2h != nullptr && gpu_timing_events()) { enqueue_hist_add(6, toc(submit_t0)); }
  return hv_submit;
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_harvest(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
  std::vector<dual::status_t>& out_status,
  gpu_batch_timing& timing,
  bool slim)
{
  raft::common::nvtx::range nvtx_scope("GpuDS::batch_harvest");
  using namespace gpu_phase2_detail;
  if (H == nullptr) { return; }
  const std::vector<gpu_batch_item<i_t, f_t>>& items = H->items;
  const size_t N = items.size();
  out_status.assign(N, dual::status_t::UNSET);
  auto& h_args = H->h_args;

  // All returned fields are contiguous across seats. Queue one copy per field after the kernel on
  // the same stream (already done by the relaunch when early_d2h), synchronize once, and only then
  // inspect statuses or scatter into node state.
  const double cb_t0 = tic();
  H->harvest_slim_known = true;
  H->harvest_slim       = slim;
  const bool early      = H->early_d2h;
  const double hv_submit = early ? 0.0 : enqueue_results(H, slim);
  H->early_d2h           = false;
  // Early mode: the relaunch already recorded ev_d2h after the copies, and a raised flag means the
  // stream is drained -- adding work here would queue behind other streams' kernels.
  const bool drained =
    early && std::atomic_ref<std::uint32_t>(*H->h_completion_flag).load(std::memory_order_acquire) != 0u;
  const bool d2h_event = H->active_seats != 0 && H->ev_d2h != nullptr && gpu_timing_events();
  if (d2h_event && !early) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_d2h, H->stream.value())); }
  timing.copyback_enqueue_s = toc(cb_t0);
  if (d2h_event) { enqueue_hist_add(0, timing.copyback_enqueue_s); }
  const double hv_sync0 = tic();
  const bool hv_steady = ++H->harvests > 10;
  if (!drained) { H->stream.synchronize(); }
  const double hv_sync = toc(hv_sync0);
  H->slim_harvest = slim && H->active_seats != 0;
  H->compact_valid = H->compact_valid && H->slim_harvest;
  if (H->clk_rounds > 0) { seat_clock_add(H->h_clk->data(), H->clk_rounds, N, H->device); }
  const double hv_unpack0 = tic();
  if (H->compact_valid) { compact_unpack(H); }
  const double hv_unpack = toc(hv_unpack0);
  const double hv_refetch0 = tic();
  if (H->slim_harvest) {
    bool any = false;
    for (size_t b = 0; b < N; ++b) {
      if (H->compact_valid && H->seat_kind[b] == 3) { continue; }
      if (H->h_status->data()[2 * b] == static_cast<i_t>(dual::status_t::REFACTOR_NEEDED)) {
        enqueue_seat_rest(H, b);
        any = true;
      }
    }
    if (any) {
      H->d2h.submit(H->stream.value());
      H->stream.synchronize();
    }
  }
  timing.copyback_sync_s = toc(cb_t0);
  if (d2h_event && hv_steady) {
    harvest_add(HV_N, 1);
    harvest_add(HV_SUBMIT, hv_submit);
    harvest_add(HV_TOTAL, timing.copyback_sync_s);
    harvest_add(HV_ENQUEUE, hv_sync0 - cb_t0);
    harvest_add(HV_SYNC1, hv_sync);
    harvest_add(HV_UNPACK, hv_unpack);
    harvest_add(HV_REFETCH, timing.copyback_sync_s - (hv_refetch0 - cb_t0));
    if (H->gather_timed) {
      float a = 0.f, b = 0.f, g = 0.f, cpy = 0.f;
      RAFT_CUDA_TRY(cudaEventElapsedTime(&a, H->ev_stop, H->ev_g0));
      RAFT_CUDA_TRY(cudaEventElapsedTime(&b, H->ev_g0, H->ev_g1));
      RAFT_CUDA_TRY(cudaEventElapsedTime(&g, H->ev_g1, H->ev_d2h0));
      RAFT_CUDA_TRY(cudaEventElapsedTime(&cpy, H->ev_d2h0, H->ev_d2h));
      harvest_add(HV_D_SCORE, a / 1e3);
      harvest_add(HV_D_GATHER, b / 1e3);
      harvest_add(HV_D_GAP, g / 1e3);
      harvest_add(HV_D_COPY, cpy / 1e3);
      harvest_add(HV_NG, 1);
    }
    float d2h_ms = 0.0f;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&d2h_ms, H->ev_d2h0, H->ev_d2h));
    timing.copyback_device_s = d2h_ms / 1000.0;
  }
  float kernel_ms = 0.0f;
  if (H->timed) { RAFT_CUDA_TRY(cudaEventElapsedTime(&kernel_ms, H->ev_start, H->ev_stop)); }
  timing.kernel_s = kernel_ms / 1000.0;
  timing.prekernel_s = 0.0;
  if (H->pre_recorded) {
    float pre_ms = 0.0f;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&pre_ms, H->ev_pre, H->ev_start));
    timing.prekernel_s = pre_ms / 1000.0;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&pre_ms, H->ev_lu0, H->ev_lu1));
    timing.prekernel_lu_s = pre_ms / 1000.0;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&pre_ms, H->ev_pre, H->ev_h2d));
    timing.prekernel_h2d_s = pre_ms / 1000.0;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&pre_ms, H->ev_h2d, H->ev_lu0));
    timing.prekernel_scatter_s = pre_ms / 1000.0;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&pre_ms, H->ev_lu1, H->ev_deltas));
    timing.prekernel_deltas_s = pre_ms / 1000.0;
    cudaEvent_t done_ev = H->gather_timed ? H->ev_g1 : H->ev_stop;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&pre_ms, H->ev_pre, done_ev));
    timing.done_after_pre_s = pre_ms / 1000.0;
    overlap_record(H, done_ev);
    H->pre_recorded    = false;
  }
  H->launched = false;
  for (size_t b = 0; b < N; ++b) {
    if (items[b].terminal_status != dual::status_t::UNSET) {
      out_status[b] = items[b].terminal_status;
      continue;
    }
    out_status[b] = static_cast<dual::status_t>(H->h_status->data()[2 * b]);
    if (H->view_mode) { continue; }
    const size_t m = static_cast<size_t>(H->rows);
    const size_t n = static_cast<size_t>(H->cols);
    auto& sol = *items[b].sol;
    if (n != 0) {
      std::copy_n(H->h_x.data() + b * n, n, sol.x.data());
      std::copy_n(H->h_vstatus.data() + b * n, n, items[b].vstatus->data());
    }
    if (m != 0) {
      std::copy_n(H->h_basic_list.data() + b * m, m, items[b].basic_list->data());
    }
    if (n != m) {
      std::copy_n(H->h_nonbasic_list.data() + b * (n - m), n - m,
                  items[b].nonbasic_list->data());
    }
    auto& dy_se = *items[b].dy_se;
    dy_se.resize(n);
    if (n != 0) {
      std::copy_n(H->h_dy_se.data() + b * n, n, dy_se.data());
      std::copy_n(H->h_z.data() + b * n, n, sol.z.data());
    }
    if (m != 0) { std::copy_n(H->h_y.data() + b * m, m, sol.y.data()); }
    sol.objective      = H->h_objective->data()[b];
    sol.user_objective = H->h_user_objective->data()[b];
    sol.iterations     = H->h_iterations->data()[b];
    if (items[b].refactor_reason != nullptr) {
      *items[b].refactor_reason = H->h_status->data()[2 * b + 1];
    }
  }
  timing.copyback_s = toc(cb_t0);



  std::fill(H->staged.begin(), H->staged.end(), 0);
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_use_views(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H)
{
  if (H == nullptr || !H->persistent_mode) {
    throw std::runtime_error("result views require a persistent GPU batch");
  }
  H->view_mode = true;
}

template <typename i_t, typename f_t>
gpu_batch_result_view<i_t, f_t> gpu_simplified_dual_phase2_batch_result(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H, std::size_t seat)
{
  if (H == nullptr || !H->view_mode || H->launched || seat >= H->items.size()) {
    throw std::runtime_error("GPU batch result view is unavailable");
  }
  const size_t m = static_cast<size_t>(H->rows);
  const size_t n = static_cast<size_t>(H->cols);
  const auto slice = [seat](auto& field, size_t width) {
    using value_t = typename std::remove_reference_t<decltype(field)>::value_type;
    return width == 0 ? std::span<value_t>{}
                      : std::span<value_t>{field.data() + seat * width, width};
  };
  gpu_batch_result_view<i_t, f_t> v{slice(H->h_x, n),
          slice(H->h_y, m),
          slice(H->h_z, n),
          slice(H->h_dy_se, n),
          slice(H->h_vstatus, n),
          slice(H->h_basic_list, m),
          slice(H->h_nonbasic_list, n - m),
          static_cast<dual::status_t>(H->h_status->data()[2 * seat]),
          H->h_objective->data()[seat],
          H->h_user_objective->data()[seat],
          H->h_iterations->data()[seat],
          H->h_status->data()[2 * seat + 1]};
  if (H->strengthen) {
    v.lower = std::span<const f_t>(H->h_lower->data() + seat * n, n);
    v.upper = std::span<const f_t>(H->h_upper->data() + seat * n, n);
  }
  if (H->compact_valid && (H->seat_kind[seat] == 1 || H->seat_kind[seat] == 2)) {
    v.compact = true;
    v.vstatus = {};
    if (H->seat_kind[seat] == 1) {
      const size_t pk  = (n + 3) / 4;
      v.packed_vstatus = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(H->h_compact->data() + H->cmp_off[1]) + seat * pk, pk);
    }
  }
  if (H->scores_valid) {
    const auto& sc   = H->h_scores->data()[seat];
    v.dev_num_frac   = sc.num_frac;
    v.dev_best_var   = sc.best_var;
    v.dev_unreliable = sc.unreliable != 0;
    v.dev_estimate   = sc.estimate;
  }
  return v;
}

template <typename T>
static void persistent_copy(T* dst,
                            const T* src,
                            size_t count,
                            rmm::cuda_stream_view stream)
{
  if (count == 0) { return; }
  auto& stats     = gpu_phase2_detail::tl_setup;
  const double t0 = stats.on ? tic() : 0.0;
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    dst, src, count * sizeof(T), cudaMemcpyHostToDevice, stream.value()));
  if (stats.on) {
    stats.h2d_s += toc(t0);
    ++stats.h2d_n;
    stats.h2d_bytes += count * sizeof(T);
  }
}

template <typename T, typename handle_t>
static void stage_h2d(handle_t* H, T* dst, const T* src, size_t count)
{
  if (H->use_slots) {
    H->pack.add(dst, src, count * sizeof(T), H->stream);
  } else {
    persistent_copy(dst, src, count, H->stream);
  }
}

// CUOPT_BNB_GPU_DUMP_FACTORS=dir: the first 32 staged L0/U0 pairs as dir/f<k>.bin (debug).
template <typename matrix_t>
static void debug_dump_factors(const matrix_t& L, const matrix_t& U)
{
  static const char* dir = std::getenv("CUOPT_BNB_GPU_DUMP_FACTORS");
  static std::atomic<int> next{0};
  if (dir == nullptr) { return; }
  const int id = next++;
  if (id >= 32) { return; }
  FILE* f = std::fopen((std::string(dir) + "/f" + std::to_string(id) + ".bin").c_str(), "wb");
  if (f == nullptr) { return; }
  for (const matrix_t* F : {&L, &U}) {
    const int hdr[2] = {static_cast<int>(F->m), static_cast<int>(F->nnz())};
    std::fwrite(hdr, sizeof(int), 2, f);
    std::fwrite(F->col_start.data(), sizeof(F->col_start[0]), F->m + 1, f);
    std::fwrite(F->i.data(), sizeof(F->i[0]), F->nnz(), f);
    std::fwrite(F->x.data(), sizeof(F->x[0]), F->nnz(), f);
  }
  std::fclose(f);
}

template <typename matrix_t, typename i_t, typename f_t>
static void persistent_stage_factor(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
  size_t seat,
  const matrix_t& F,
  i_t*& d_col,
  i_t*& d_i,
  f_t*& d_x,
  i_t& d_nz,
  size_t& capacity)
{
  raft::common::nvtx::range nvtx_scope("GpuDS::persistent_stage_factor");
  using namespace gpu_phase2_detail;
  const i_t rows = H->h_args[seat].m;
  if (F.m != rows || F.n != rows || F.col_start.size() != static_cast<size_t>(rows + 1)) {
    throw std::runtime_error("persistent factor has the wrong dimensions");
  }
  const size_t nz = static_cast<size_t>(F.nnz());
  // The slot path trusts the host LU's row indices; the check stays on the per-seat buffer path.
  for (size_t p = 0; p < (H->use_slots ? 0 : nz); ++p) {
    if (F.i[p] < 0 || F.i[p] >= rows) {
      throw std::runtime_error("persistent factor row index is outside [0, m)");
    }
  }
  stage_h2d(H, d_col, F.col_start.data(), F.col_start.size());
  if (H->use_slots) {
    if (nz > capacity) { throw std::runtime_error("persistent factor exceeds its seat slot"); }
    H->pack.add(d_i, F.i.data(), nz * sizeof(i_t), H->stream);
    H->pack.add(d_x, F.x.data(), nz * sizeof(f_t), H->stream);
    d_nz = static_cast<i_t>(nz);
    return;
  }
  if (nz > capacity) {
    const size_t grown = std::max(nz, capacity == 0 ? size_t{1} : 2 * capacity);
    d_i      = alloc_uninit<i_t>(grown, H->store[seat]);
    d_x      = alloc_uninit<f_t>(grown, H->store[seat]);
    capacity = grown;
  }
  persistent_copy(d_i, F.i.data(), nz, H->stream);
  persistent_copy(d_x, F.x.data(), nz, H->stream);
  d_nz = static_cast<i_t>(nz);
}

template <typename handle_t>
static void init_staged(handle_t* H)
{
  if (!H->staged.empty()) { return; }
  const size_t seats = H->h_args.size();
  H->factor_caps.resize(seats);
  H->delta_offset.assign(seats, 0);
  for (size_t b = 0; b < seats; ++b) {
    if (H->use_slots) {
      H->factor_caps[b] = {H->slots.capL, H->slots.capU, H->slots.capL, H->slots.capU};
      continue;
    }
    H->factor_caps[b] = {static_cast<size_t>(H->h_args[b].L0_nz),
                         static_cast<size_t>(H->h_args[b].U0_nz),
                         static_cast<size_t>(H->h_args[b].L0t_nz),
                         static_cast<size_t>(H->h_args[b].U0t_nz)};
  }
  H->staged.assign(seats, 0);
}

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_park(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
                                           std::size_t seat)
{
  if (H == nullptr || !H->persistent_mode || H->launched || seat >= H->h_args.size()) {
    return false;
  }
  init_staged(H);
  H->h_args[seat].done           = 1;
  H->items[seat].terminal_status = dual::status_t::NUMERICAL;
  H->staged[seat]                = 1;
  return true;
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_unstage(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
                                              std::size_t lo,
                                              std::size_t hi)
{
  if (H == nullptr || H->staged.empty()) { return; }
  for (size_t b = lo; b < hi && b < H->staged.size(); ++b) { H->staged[b] = 0; }
}

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_launched(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H)
{
  return H != nullptr && H->launched;
}

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_stage(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
  std::size_t seat,
  const gpu_batch_item<i_t, f_t>& item,
  gpu_batch_seat_command command,
  const std::vector<gpu_bound_delta<i_t, f_t>>& deltas,
  std::string& error)
{
  using namespace gpu_phase2_detail;
  double split_t = tic();
  const auto split = [&](int k) {
    const double now = tic();
    tl_stage_split[k] += now - split_t;
    split_t = now;
  };
  error.clear();
  if (H == nullptr || !H->persistent_mode || H->launched) {
    error = "persistent batch must be harvested before staging";
    return false;
  }
  if (seat >= H->h_args.size() || item.lp == nullptr || item.settings == nullptr ||
      item.vstatus == nullptr || item.ft == nullptr || item.basic_list == nullptr ||
      item.nonbasic_list == nullptr || item.sol == nullptr || item.dy_se == nullptr) {
    error = "invalid persistent seat or null item field";
    return false;
  }
  kernel_args& a = H->h_args[seat];
  const i_t m    = a.m;
  const i_t n    = a.n;
  assert(H->view_mode ? H->h_dy_se.size() == H->h_args.size() * static_cast<size_t>(n)
                      : item.dy_se->size() == static_cast<size_t>(n));
  if (item.lp->num_rows != m || item.lp->num_cols != n || item.lower == nullptr ||
      item.upper == nullptr || item.lower->size() != static_cast<size_t>(n) ||
      item.upper->size() != static_cast<size_t>(n)) {
    error = "persistent seat dimensions or bounds changed";
    return false;
  }
  if (command == gpu_batch_seat_command::CHILD_CONTINUE && deltas.size() > static_cast<size_t>(n)) {
    error = "child bound-delta list exceeds the column count";
    return false;
  }
  init_staged(H);
  if (H->staged[seat] != 0) {
    error = "persistent seat was staged more than once in one generation";
    return false;
  }

  H->items[seat]         = item;
  H->items[seat].command = command;
  if (gpu_fission() && command != gpu_batch_seat_command::CHILD_CONTINUE) {
    if (item.fission_inv == nullptr) {
      error = "fissioned pivot loop (CUOPT_BNB_GPU_FISSION) needs the seat's explicit inverses";
      return false;
    }
    if (H->h_finv.size() < H->h_args.size()) { H->h_finv.resize(H->h_args.size()); }
    H->h_finv[seat] = *item.fission_inv;
  }
  a.done                 = 0;
  a.device_lu            = 0;
  a.transpose_on_device  = 0;
  a.command              = static_cast<i_t>(command);
  a.bound_delta_count    = 0;
  a.bounds_from_root     = 0;
  a.primal_tol           = item.settings->primal_tol;
  a.dual_tol             = item.settings->dual_tol;
  a.pivot_tol            = item.settings->pivot_tol;
  a.tight_tol            = item.settings->tight_tol;
  a.fixed_tol            = item.settings->fixed_tol;
  a.zero_tol             = item.settings->zero_tol;
  a.time_limit           = item.settings->time_limit;
  a.cut_off              = item.settings->cut_off;
  a.steepest_edge_ratio = item.settings->steepest_edge_ratio;
  a.refactor_frequency  = item.settings->refactor_frequency;
  a.start_time           = tic();
  a.terminal_certificate_retry = item.terminal_certificate_retry ? 1 : 0;
  a.refactor_resume = command == gpu_batch_seat_command::REFACTOR_RESUME ? 1 : 0;
  a.resume_obj      = H->view_mode && command != gpu_batch_seat_command::FRESH_NODE
                        ? H->h_objective->data()[seat]
                        : item.sol->objective;
  a.resume_iter     = H->view_mode && command != gpu_batch_seat_command::FRESH_NODE
                        ? H->h_iterations->data()[seat]
                        : static_cast<i_t>(item.sol->iterations);
  const cudaStream_t cs = H->stream.value();

  const bool from_root = command == gpu_batch_seat_command::FRESH_NODE && item.root_lower != nullptr;
  if (from_root) {
    if (!gpu_simplified_dual_phase2_batch_lean_fresh(H) || !item.device_lu ||
        item.root_lower->size() != static_cast<size_t>(n) ||
        item.root_upper->size() != static_cast<size_t>(n)) {
      error = "path bounds need lean fresh staging and root bounds";
      return false;
    }
    if (!H->d_root) {
      H->d_root.emplace(3 * n * sizeof(f_t), H->stream);
      auto* d = static_cast<f_t*>(H->d_root->data());
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        d, item.root_lower->data(), n * sizeof(f_t), cudaMemcpyHostToDevice, cs));
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        d + n, item.root_upper->data(), n * sizeof(f_t), cudaMemcpyHostToDevice, cs));
    }
    if (item.root_dy_se != nullptr && !H->root_dy_se_set) {
      if (item.root_dy_se->size() != static_cast<size_t>(n)) {
        error = "root edge norms have the wrong size";
        return false;
      }
      RAFT_CUDA_TRY(cudaMemcpyAsync(static_cast<f_t*>(H->d_root->data()) + 2 * n,
                                    item.root_dy_se->data(), n * sizeof(f_t),
                                    cudaMemcpyHostToDevice, cs));
      H->root_dy_se_set = true;
    }
    a.root_lower       = static_cast<const f_t*>(H->d_root->data());
    a.root_upper       = a.root_lower + n;
    a.bounds_from_root = 1;
  }
  if (command == gpu_batch_seat_command::CHILD_CONTINUE || from_root) {
    const size_t need = H->delta_used + deltas.size();
    if (need > H->delta_cap) {
      const size_t cap = std::max({need, 2 * H->delta_cap, 4 * H->h_args.size()});
      auto grown       = pinned_make<gpu_bound_delta<i_t, f_t>>(H->arena.get(),
        cuda::stream_ref{cs}, cap);
      if (H->delta_used != 0) { std::copy_n(H->h_deltas->data(), H->delta_used, grown.data()); }
      H->h_deltas.emplace(std::move(grown));
      H->delta_cap = cap;
    }
    auto* packed = H->h_deltas ? H->h_deltas->data() + H->delta_used : nullptr;
    for (size_t q = 0; q < deltas.size(); ++q) {
      if (deltas[q].column < 0 || deltas[q].column >= n || (deltas[q].mask & ~3u) != 0u) {
        error = "invalid child bound delta";
        return false;
      }
      packed[q] = deltas[q];
    }
    // The device address is resolved by the relaunch, after the last seat is packed.
    H->delta_offset[seat] = H->delta_used;
    H->delta_used         = need;
    a.bound_deltas        = nullptr;
    a.bound_delta_count   = static_cast<i_t>(deltas.size());
    // x/y/z, lists, dy_se, the base factor and the MPF stack deliberately stay resident.
  }
  if (command != gpu_batch_seat_command::CHILD_CONTINUE) {
    if ((!H->view_mode || command == gpu_batch_seat_command::FRESH_NODE) &&
        (item.vstatus->size() != static_cast<size_t>(n) ||
        item.basic_list->size() != static_cast<size_t>(m) ||
        item.nonbasic_list->size() != static_cast<size_t>(n - m) ||
        item.sol->x.size() != static_cast<size_t>(n) ||
        item.sol->y.size() != static_cast<size_t>(m) ||
        item.sol->z.size() != static_cast<size_t>(n) ||
        item.dy_se->size() != static_cast<size_t>(n))) {
      error = "persistent fresh/refactor state has the wrong size";
      return false;
    }
    split(0);
    // A resumed seat's bounds are the ones the device already holds.
    if (!from_root && (!H->use_slots || command == gpu_batch_seat_command::FRESH_NODE)) {
      stage_h2d(H, a.lower, item.lower->data(), n);
      stage_h2d(H, a.upper, item.upper->data(), n);
    }
    split(1);
    const bool lean = H->view_mode && H->use_slots && H->device_lu && item.device_lu &&
                      command == gpu_batch_seat_command::FRESH_NODE && gpu_lean_fresh();
    if (lean) {
      auto& F        = *H->fixed;
      const auto put = [&](auto& dev, const auto& source, size_t width) {
        if (width != 0) {
          H->pack.add(dev.data() + seat * width, source.data(), width * sizeof(source[0]), H->stream);
        }
      };
      if (item.packed_vstatus != nullptr) {
        if (!from_root || !H->root_dy_se_set) {
          error = "packed fresh staging needs path bounds and root edge norms";
          return false;
        }
        const size_t N  = H->h_args.size();
        const size_t pk = static_cast<size_t>((n + 3) / 4);
        if (item.packed_vstatus->size() != pk) {
          error = "packed basis status has the wrong size";
          return false;
        }
        if (!H->d_expand) { H->d_expand.emplace(N * (sizeof(int) + pk), H->stream); }
        auto* packed = static_cast<std::uint8_t*>(H->d_expand->data()) + N * sizeof(int);
        H->pack.add(packed + seat * pk, item.packed_vstatus->data(), pk, H->stream);
        H->expand_seats.push_back(static_cast<int>(seat));
      } else {
        put(F.vstatus, *item.vstatus, n);
        put(F.basic_list, *item.basic_list, m);
        put(F.nonbasic_list, *item.nonbasic_list, n - m);
        put(F.dy_se, *item.dy_se, n);
      }
      split(3);
    } else if (H->view_mode) {
      // A resume already modified its harvested spans in place. A fresh replacement owns new
      // vectors, so only that seat is written into the host slabs before the full-pool upload.
      // Fresh seats in slot mode upload straight from the node's vectors; the slab rows are
      // rewritten by the next harvest, and nothing reads them in between.
      if (command == gpu_batch_seat_command::FRESH_NODE && H->use_slots) {
        auto& F        = *H->fixed;
        const auto put = [&](auto& dev, const auto& source, size_t width) {
          if (width != 0) {
            H->pack.add(dev.data() + seat * width, source.data(), width * sizeof(source[0]), H->stream);
          }
        };
        // No x: a fresh setup sets the nonbasic values from the bounds and solves for the basics.
        put(F.vstatus, *item.vstatus, n);
        put(F.basic_list, *item.basic_list, m);
        put(F.nonbasic_list, *item.nonbasic_list, n - m);
        put(F.y, item.sol->y, m);
        put(F.z, item.sol->z, n);
        put(F.dy_se, *item.dy_se, n);
        split(2);
      } else {
      if (command == gpu_batch_seat_command::FRESH_NODE) {
        const auto copy_seat = [seat](auto& slab, size_t width, const auto& source) {
          if (width != 0) { std::copy_n(source.data(), width, slab.data() + seat * width); }
        };
        copy_seat(H->h_vstatus, n, *item.vstatus);
        copy_seat(H->h_basic_list, m, *item.basic_list);
        copy_seat(H->h_nonbasic_list, n - m, *item.nonbasic_list);
        copy_seat(H->h_x, n, item.sol->x);
        copy_seat(H->h_y, m, item.sol->y);
        copy_seat(H->h_z, n, item.sol->z);
        copy_seat(H->h_dy_se, n, *item.dy_se);
      }
      split(2);
      if (H->use_slots) {
        // A resume changed x and the basis in place; y, z and dy_se stay the device's.
        auto& F = *H->fixed;
        const auto slice = [&](auto& dev, auto& slab, size_t width) {
          if (width != 0) {
            H->pack.add(dev.data() + seat * width, slab.data() + seat * width,
                        width * sizeof(slab[0]), H->stream);
          }
        };
        slice(F.x, H->h_x, n);
        slice(F.vstatus, H->h_vstatus, n);
        slice(F.basic_list, H->h_basic_list, m);
        slice(F.nonbasic_list, H->h_nonbasic_list, n - m);
        if (command == gpu_batch_seat_command::FRESH_NODE) {
          slice(F.y, H->h_y, m);
          slice(F.z, H->h_z, n);
          slice(F.dy_se, H->h_dy_se, n);
        }
      }
      }
    } else {
      persistent_copy(a.vstatus, item.vstatus->data(), n, H->stream);
      persistent_copy(a.basic_list, item.basic_list->data(), m, H->stream);
      persistent_copy(a.nonbasic_list, item.nonbasic_list->data(), n - m, H->stream);
      persistent_copy(a.x, item.sol->x.data(), n, H->stream);
      persistent_copy(a.y, item.sol->y.data(), m, H->stream);
      persistent_copy(a.z, item.sol->z.data(), n, H->stream);
      persistent_copy(a.dy_se, item.dy_se->data(), n, H->stream);
    }
    a.num_updates = 0;
    a.device_lu   = 0;
    if (H->device_lu && item.device_lu && command == gpu_batch_seat_command::FRESH_NODE) {
      // Factors, permutations and duals come from the device (the relaunch's dense batch).
      a.device_lu = 1;
      a.lu_info   = H->lu_info + 3 * seat;
      H->lu_batch.push_back(H->lu_targets[seat]);
      H->staged[seat] = 1;
      split(3);
      return true;
    }
    stage_h2d(H, a.row_perm, item.ft->row_permutation().data(), m);
    stage_h2d(H, a.inv_row_perm, item.ft->inverse_row_permutation().data(), m);

    auto& caps = H->factor_caps[seat];
    debug_dump_factors(item.ft->L0(), item.ft->U0());
    persistent_stage_factor(
      H, seat, item.ft->L0(), a.L0_col_start, a.L0_i, a.L0_x, a.L0_nz, caps.L0);
    persistent_stage_factor(
      H, seat, item.ft->U0(), a.U0_col_start, a.U0_i, a.U0_x, a.U0_nz, caps.U0);
    if (H->use_slots && gpu_device_transpose()) {
      if (static_cast<size_t>(a.L0_nz) > caps.L0t || static_cast<size_t>(a.U0_nz) > caps.U0t) {
        throw std::runtime_error("persistent factor transpose exceeds its seat slot");
      }
      a.L0t_nz              = a.L0_nz;
      a.U0t_nz              = a.U0_nz;
      a.transpose_on_device = 1;
      H->staged[seat]       = 1;
      return true;
    }
    persistent_stage_factor(H,
                            seat,
                            item.ft->L0_transpose(),
                            a.L0t_col_start,
                            a.L0t_i,
                            a.L0t_x,
                            a.L0t_nz,
                            caps.L0t);
    persistent_stage_factor(H,
                            seat,
                            item.ft->U0_transpose(),
                            a.U0t_col_start,
                            a.U0t_i,
                            a.U0t_x,
                            a.U0t_nz,
                            caps.U0t);
  }

  H->staged[seat] = 1;
  return true;
}

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_relaunch(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
  gpu_batch_timing& timing,
  std::string& error)
{
  raft::common::nvtx::range nvtx_scope("GpuDS::batch_relaunch");
  using namespace gpu_phase2_detail;
  error.clear();
  if (H == nullptr || !H->persistent_mode || H->launched) {
    error = "persistent batch is null or already launched";
    return false;
  }
  const size_t N = H->h_args.size();
  if (H->staged.size() != N ||
      std::find(H->staged.begin(), H->staged.end(), 0) != H->staged.end()) {
    error = "persistent batch relaunch requires every seat to be staged";
    return false;
  }
  // Staging can include expensive host refactors. Read the settings again at the actual launch
  // boundary so an incumbent published while those ran is not hidden behind the earlier snapshot.
  for (size_t b = 0; b < N; ++b) {
    if (H->items[b].settings != nullptr) { H->h_args[b].cut_off = H->items[b].settings->cut_off; }
  }
  const double t0 = tic();
  const bool timed = gpu_timing_events();
  if (timed) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_pre, H->stream.value())); }
  H->pre_recorded = timed;
  H->timed        = timed;
  if (H->view_mode && !H->use_slots) {
    const auto upload = [H](const auto& source, auto member) {
      if (!source.empty()) {
        RAFT_CUDA_TRY(cudaMemcpyAsync((H->fixed.value().*member).data(),
                                      source.data(),
                                      source.size() * sizeof(typename std::remove_reference_t<decltype(source)>::value_type),
                                      cudaMemcpyHostToDevice,
                                      H->stream.value()));
      }
    };
    upload(H->h_x, &fixed_storage::x);
    upload(H->h_y, &fixed_storage::y);
    upload(H->h_z, &fixed_storage::z);
    upload(H->h_dy_se, &fixed_storage::dy_se);
    upload(H->h_vstatus, &fixed_storage::vstatus);
    upload(H->h_basic_list, &fixed_storage::basic_list);
    upload(H->h_nonbasic_list, &fixed_storage::nonbasic_list);
  }
  const int lu_count = static_cast<int>(H->lu_batch.size());
  if (lu_count > 0) {
    H->pack.add(H->d_lu_batch, H->lu_batch.data(), lu_count * sizeof(gpu_lu::dense_seat_t),
                H->stream);
  }
  const int expand_count = static_cast<int>(H->expand_seats.size());
  if (expand_count > 0) {
    H->pack.add(H->d_expand->data(), H->expand_seats.data(), expand_count * sizeof(int), H->stream);
  }
  if (H->use_slots) {
    H->pack.flush(H->stream, timed ? H->ev_h2d : nullptr);
  } else if (timed) {
    RAFT_CUDA_TRY(cudaEventRecord(H->ev_h2d, H->stream.value()));
  }
  double split_t = tic();
  tl_relaunch_split[0] = split_t - t0;
  if (expand_count > 0) {
    auto& F          = *H->fixed;
    const size_t N   = H->h_args.size();
    const int n      = H->cols;
    auto* seat_list  = static_cast<const int*>(H->d_expand->data());
    gpu_lu::expand_fresh_seats(
      {seat_list, expand_count, reinterpret_cast<const std::uint8_t*>(seat_list + N), (n + 3) / 4,
       H->rows, n, reinterpret_cast<std::int8_t*>(F.vstatus.data()), F.basic_list.data(),
       F.nonbasic_list.data(), F.dy_se.data(), static_cast<const f_t*>(H->d_root->data()) + 2 * n},
      H->stream.value());
    H->expand_seats.clear();
  }
  if (timed) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_lu0, H->stream.value())); }
  if (lu_count > 0) {
    const kernel_args& a0 = H->h_args[0];
    gpu_lu::matrix_t A{H->rows, a0.A_col_start, a0.A_i, a0.A_x};
    if (H->lu_custom) {
      gpu_lu::run_batch_lu(A, H->d_lu_batch, lu_count, H->lu_ws, 1e-11, H->stream.value());
    } else {
      gpu_lu::run_dense_batch(H->blas, A, H->d_lu_batch, lu_count, H->lu_dense, H->lu_ptrs,
                              H->lu_piv, H->lu_cinfo, 1e-11, H->stream.value());
    }
    H->lu_batch.clear();
  }
  if (timed) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_lu1, H->stream.value())); }
  tl_relaunch_split[1] = tic() - split_t;
  split_t              = tic();
  if (H->delta_used != 0) {
    using delta_t = gpu_bound_delta<i_t, f_t>;
    if (H->delta_used > H->delta_dev_cap) {
      H->d_deltas.emplace(H->delta_cap * sizeof(delta_t), H->stream);
      H->delta_dev_cap = H->delta_cap;
    }
    auto* d_deltas = static_cast<delta_t*>(H->d_deltas->data());
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_deltas,
                                  H->h_deltas->data(),
                                  H->delta_used * sizeof(delta_t),
                                  cudaMemcpyHostToDevice,
                                  H->stream.value()));
    for (size_t b = 0; b < N; ++b) {
      if (H->items[b].command == gpu_batch_seat_command::CHILD_CONTINUE ||
          H->h_args[b].bounds_from_root != 0) {
        H->h_args[b].bound_deltas = d_deltas + H->delta_offset[b];
      }
    }
    H->delta_used = 0;
  }
  if (timed) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_deltas, H->stream.value())); }
  if (gpu_fission()) {
    const size_t kc = static_cast<size_t>(mpf_update_cap());
    if (!H->d_fm) { H->d_fm.emplace(N * kc * kc * sizeof(f_t), H->stream); }
    for (size_t b = 0; b < N; ++b) {
      H->h_args[b].fm      = static_cast<f_t*>(H->d_fm->data()) + b * kc * kc;
      H->h_args[b].fm_kcap = static_cast<i_t>(kc);
    }
  }
  RAFT_CUDA_TRY(cudaMemcpyAsync(H->d_args,
                                H->h_args.data(),
                                N * sizeof(kernel_args),
                                cudaMemcpyHostToDevice,
                                H->stream.value()));
  if (!gpu_few_calls()) {
    RAFT_CUDA_TRY(cudaMemsetAsync(H->d_next, 0, sizeof(unsigned int), H->stream.value()));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(H->d_completed_blocks, 0, sizeof(unsigned int), H->stream.value()));
    RAFT_CUDA_TRY(cudaMemsetAsync(H->d_yield, 0, sizeof(unsigned int), H->stream.value()));
  }
  std::atomic_ref<std::uint32_t>(*H->h_completion_flag)
    .store(0u, std::memory_order_release);
  timing.upload_s = toc(t0);
  tl_relaunch_split[2] = tic() - split_t;
  split_t              = tic();

  unsigned int nb = H->relaunch_blocks;
  if (nb == 0 || !gpu_few_calls()) {
    int device = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    int sm_count = 1;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int blocks_per_sm = 1;
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, gpu_phase2_kernel_direct<true, true>, 32, 0));
    nb = static_cast<unsigned int>(blocks_per_sm) * static_cast<unsigned int>(sm_count);
    if (nb == 0 || nb > N) { nb = static_cast<unsigned int>(N); }
    H->relaunch_blocks = nb;
  }
  if (timed) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_start, H->stream.value())); }
  int rounds = !gpu_phase2_detail::gpu_fission() && H->scores_armed && H->dive_armed ? H->dive_rounds : 1;
  auto* seat_scores =
    H->scores_armed ? reinterpret_cast<gpu_scores::seat_score_t*>(
                        static_cast<char*>(H->d_scores->data()) +
                        static_cast<size_t>(H->cols) * (2 * sizeof(f_t) + 2 * sizeof(i_t)))
                    : nullptr;
  dive_ctx_t D;
  if (rounds > 1) {
    auto* d  = static_cast<char*>(H->d_dive->data());
    D.records = rounds - 1;
    D.pk      = (H->cols + 3) / 4;
    D.root_x  = reinterpret_cast<const f_t*>(d);
    D.deltas  = reinterpret_cast<gpu_bound_delta<i_t, f_t>*>(d + H->dive_off[0]);
    D.taken   = reinterpret_cast<int*>(d + H->dive_off[1]);
    D.log     = reinterpret_cast<gpu_dive_record<i_t, f_t>*>(d + H->dive_off[2]);
    D.packed  = reinterpret_cast<std::uint8_t*>(d + H->dive_off[3]);
  }
  const bool in_kernel = rounds > 1 && gpu_dive_in_kernel();
  const int launches   = in_kernel ? 1 : rounds;
  H->clk_rounds        = gpu_seat_clock() ? rounds : 0;
  if (H->clk_rounds > 0) {
    if (H->clk_cap_rounds < rounds) {
      H->d_clk.emplace(rounds * N * sizeof(seat_clock_t), H->stream);
      H->h_clk.emplace(pinned_make<seat_clock_t>(H->arena.get(),
        cuda::stream_ref{H->stream.value()}, rounds * N));
      H->clk_cap_rounds = rounds;
    }
    RAFT_CUDA_TRY(cudaMemsetAsync(H->d_clk->data(), 0, rounds * N * sizeof(seat_clock_t),
                                  H->stream.value()));
  }
  static const bool early_env = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_EARLY_D2H");
    return e == nullptr || std::string(e) != "0";
  }();
  const bool early = early_env && H->harvest_slim_known;
  for (int r = 0; r < launches; ++r) {
    const bool last = r + 1 == launches;
    if (r > 0) {
      constexpr int threads = 128;
      dive_advance_kernel<<<(static_cast<int>(N) * 32 + threads - 1) / threads, threads, 0,
                            H->stream.value()>>>(
        H->d_args, static_cast<int>(N), r - 1, seat_scores, D, H->d_next);
    }
    dive_ctx_t K;
    if (in_kernel) {
      K         = D;
      K.scores  = seat_scores;
      K.var_int = H->d_var_int;
      K.pc      = H->score_pc;
    }
    if (H->clk_rounds > 0) {
      K.clk       = static_cast<seat_clock_t*>(H->d_clk->data()) + (in_kernel ? 0 : r * N);
      K.clk_seats = static_cast<int>(N);
    }
    if (launches == 1 && gpu_yield_frac() > 0.0) {
      size_t live = 0;
      for (size_t b = 0; b < N; ++b) { live += H->h_args[b].done == 0; }
      K.yield_count = H->d_yield;
      K.yield_at    = static_cast<unsigned int>(N - live) +
                   static_cast<unsigned int>(std::ceil(gpu_yield_frac() * static_cast<double>(live)));
    }
    if (gpu_phase2_detail::gpu_fission()) {
      using namespace gpu_phase2_detail;
      // Fissioned pivot loop: setup, then rounds of the phase kernels until no seat is running.
      const cudaStream_t cs = H->stream.value();
      const i_t Ni          = static_cast<i_t>(N);
      if (H->h_finv.size() != N) {
        error = "fissioned pivot loop: seats without explicit inverses";
        return false;
      }
      if (!H->d_finv) { H->d_finv.emplace(N * sizeof(gpu_fission_inv), H->stream); }
      if (!H->d_live) { H->d_live.emplace(sizeof(int), H->stream); }
      RAFT_CUDA_TRY(cudaMemcpyAsync(H->d_finv->data(), H->h_finv.data(), N * sizeof(gpu_fission_inv),
                                    cudaMemcpyHostToDevice, cs));
      const auto* finv = static_cast<const gpu_fission_inv*>(H->d_finv->data());
      int* d_live      = static_cast<int*>(H->d_live->data());
      const bool det = gpu_fission_deterministic();
      const size_t smem =
        det ? static_cast<size_t>(H->rows) * (2 * sizeof(f_t) + sizeof(i_t)) +
                2 * sizeof(unsigned) * ((static_cast<size_t>(H->rows) + 31) / 32)
            : static_cast<size_t>(H->rows) * sizeof(f_t) + kFissionStage;
      auto* k_btran = det ? fission_btran : fission_btran_atomic;
      auto* k_ftran = det ? fission_ftran : fission_ftran_atomic;
      RAFT_CUDA_TRY(cudaFuncSetAttribute(k_btran, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)));
      RAFT_CUDA_TRY(cudaFuncSetAttribute(k_ftran, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem)));
      const size_t smem_c = static_cast<size_t>(H->rows) * sizeof(f_t) +
                            static_cast<size_t>(H->cols - H->rows) * 2 * (sizeof(f_t) + sizeof(i_t));
      RAFT_CUDA_TRY(cudaFuncSetAttribute(fission_choose_block, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem_c)));
      if (const char* ut = std::getenv("CUOPT_BNB_GPU_FISSION_UTRACE"); ut != nullptr && *ut != 0) {
        static std::once_flag once;
        std::call_once(once, [] {
          void* p = nullptr;
          RAFT_CUDA_TRY(cudaMalloc(&p, kUtraceCap * sizeof(fission_utrace_t)));
          RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_utrace, &p, sizeof(p)));
        });
      }
      if (const char* ut = std::getenv("CUOPT_BNB_GPU_FISSION_CTRACE"); ut != nullptr && *ut != 0) {
        static std::once_flag once;
        std::call_once(once, [] {
          void* p = nullptr;
          RAFT_CUDA_TRY(cudaMalloc(&p, kUtraceCap * sizeof(fission_utrace_t)));
          RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_ctrace, &p, sizeof(p)));
        });
      }
      if (const char* ut = std::getenv("CUOPT_BNB_GPU_FISSION_STRACE"); ut != nullptr && *ut != 0) {
        static std::once_flag once;
        std::call_once(once, [] {
          void* p = nullptr;
          RAFT_CUDA_TRY(cudaMalloc(&p, kUtraceCap * sizeof(fission_utrace_t)));
          RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_strace, &p, sizeof(p)));
        });
      }
      if (const char* ut = std::getenv("CUOPT_BNB_GPU_FISSION_FTRACE"); ut != nullptr && *ut != 0) {
        static std::once_flag once;
        std::call_once(once, [] {
          void* p = nullptr;
          RAFT_CUDA_TRY(cudaMalloc(&p, kUtraceCap * sizeof(fission_utrace_t)));
          RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_ftrace, &p, sizeof(p)));
        });
      }
      {
        static const int direct = [] {
          const char* e = std::getenv("CUOPT_EXP_FTRAN");
          return (e != nullptr && std::string(e) == "direct") ? 1 : 0;
        }();
        RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(c_ftran_direct, &direct, sizeof(int), 0, cudaMemcpyHostToDevice, cs));
      }
      fission_presetup<<<Ni, kPresetupThreads, 0, cs>>>(H->d_args, Ni);
      fission_setup_kernel<<<Ni, 32, 0, cs>>>(H->d_args, Ni);
      {
        const size_t smem_x = 2 * static_cast<size_t>(H->rows) * sizeof(f_t);
        RAFT_CUDA_TRY(cudaFuncSetAttribute(fission_xb, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(smem_x)));
        fission_xb<<<Ni, kXbThreads, smem_x, cs>>>(H->d_args, finv, Ni);
      }
      int live0 = -1;
      bool yielding = false;
      static const bool fused = !det && [] { const char* e = std::getenv("CUOPT_EXP_FUSED"); return e != nullptr && std::string(e) != "0"; }();  // opt-in
      const size_t smem_f     = std::max(smem, smem_c);
      if (fused) {
        RAFT_CUDA_TRY(cudaFuncSetAttribute(fission_fused, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(smem_f)));
      }
      for (;;) {
        if (fused) {
          fission_fused<<<Ni, kFissionThreads, smem_f, cs>>>(H->d_args, finv, Ni, gpu_fission_rounds(), yielding ? 1 : 0);
        }
        for (int k = 0; k < (fused ? 0 : gpu_fission_rounds()); ++k) {
          if (yielding) { fission_yield<<<Ni, 32, 0, cs>>>(H->d_args, Ni); }
          fission_price<<<Ni, 32, 0, cs>>>(H->d_args, Ni);
          k_btran<<<Ni, kFissionThreads, smem, cs>>>(H->d_args, finv, Ni);
          fission_choose_block<<<Ni, kFissionThreads, smem_c, cs>>>(H->d_args, Ni);
          k_ftran<<<Ni, kFissionThreads, smem, cs>>>(H->d_args, finv, Ni);
          fission_update_block<<<Ni, kUpdThreads, 0, cs>>>(H->d_args, Ni);
        }
        RAFT_CUDA_TRY(cudaMemsetAsync(d_live, 0, sizeof(int), cs));
        fission_count<<<(Ni + 255) / 256, 256, 0, cs>>>(H->d_args, Ni, d_live);
        int live = 0;
        RAFT_CUDA_TRY(cudaMemcpyAsync(&live, d_live, sizeof(int), cudaMemcpyDeviceToHost, cs));
        RAFT_CUDA_TRY(cudaStreamSynchronize(cs));
        static const bool trace = std::getenv("CUOPT_BNB_GPU_FISSION_TRACE") != nullptr;
        if (trace) {
          // Per live seat: state, iteration, num_updates (first few).
          std::vector<kernel_args> ha(N);
          RAFT_CUDA_TRY(cudaMemcpy(ha.data(), H->d_args, N * sizeof(kernel_args), cudaMemcpyDeviceToHost));
          int shown = 0;
          std::string line;
          for (size_t b = 0; b < N && shown < 6; ++b) {
            if (ha[b].done != 0 || ha[b].slot == nullptr) { continue; }
            int fs = 0, it = 0, nu = 0;
            auto* sl = ha[b].slot;
            RAFT_CUDA_TRY(cudaMemcpy(&fs, &sl->f_state, sizeof(int), cudaMemcpyDeviceToHost));
            if (fs == F_DONE) { continue; }
            RAFT_CUDA_TRY(cudaMemcpy(&it, &sl->iter, sizeof(int), cudaMemcpyDeviceToHost));
            RAFT_CUDA_TRY(cudaMemcpy(&nu, &sl->ft.num_updates_, sizeof(int), cudaMemcpyDeviceToHost));
            line += " seat" + std::to_string(b) + ":st" + std::to_string(fs) + ",it" + std::to_string(it) +
                    ",nu" + std::to_string(nu);
            ++shown;
          }
          std::fprintf(stderr, "[FISSION] live=%d%s\n", live, line.c_str());
        }
        if (live == 0) { break; }
        if (live0 < 0) { live0 = live; }
        if (!yielding && live <= gpu_fission_yield() * live0) { yielding = true; }
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      if (last && !early) { set_flag_kernel<<<1, 1, 0, cs>>>(H->d_completion_flag); }
    } else {
    gpu_phase2_kernel_direct<true, true>
      <<<nb, 32, 0, H->stream.value()>>>(
      H->d_args,
      static_cast<i_t>(N),
      H->d_next,
      last ? H->d_completed_blocks : nullptr,
      last ? (early ? H->d_flag_sink : H->d_completion_flag) : nullptr,
      K);
    }
    if (last) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_stop, H->stream.value())); }
    if (seat_scores != nullptr && !in_kernel) {
      gpu_scores::score_seats(H->fixed->x.data(), H->d_var_int, static_cast<int>(N), H->cols,
                              H->score_pc, seat_scores, H->stream.value());
    }
  }
  H->compact_valid = seat_scores != nullptr && gpu_compact_harvest() && H->view_mode &&
                     H->strengthen && H->use_slots;
  if (H->compact_valid) {
    const size_t n = static_cast<size_t>(H->cols), m = static_cast<size_t>(H->rows);
    const size_t pk = (n + 3) / 4;
    if (!H->d_compact) {
      const auto align = [](size_t b) { return (b + 15) / 16 * 16; };
      H->cmp_lp_state_bytes    = align(5 * n * sizeof(f_t) + m * sizeof(f_t) + n * sizeof(i_t) + n);
      H->cmp_off[0] = align(N * sizeof(compact_hdr_t));
      H->cmp_off[1] = H->cmp_off[0] + 16;
      H->cmp_off[2] = align(H->cmp_off[1] + N * pk);
      H->d_compact.emplace(H->cmp_off[2] + N * H->cmp_lp_state_bytes, H->stream);
      if (!H->h_compact) {
        H->h_compact.emplace(pinned_make<char>(H->arena.get(),
          cuda::stream_ref{H->stream.value()}, H->cmp_off[2]));
      }
    }
    auto* d = static_cast<char*>(H->d_compact->data());
    H->gather_timed = gpu_timing_events() && H->ev_g0 != nullptr;
    if (H->gather_timed) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_g0, H->stream.value())); }
    RAFT_CUDA_TRY(cudaMemsetAsync(d + H->cmp_off[0], 0, sizeof(int), H->stream.value()));
    constexpr int threads = 128;
    compact_gather_kernel<<<(static_cast<int>(N) * 32 + threads - 1) / threads, threads, 0,
                            H->stream.value()>>>(
      H->d_args, static_cast<int>(N), static_cast<int>(pk), H->cmp_lp_state_bytes, seat_scores,
      reinterpret_cast<compact_hdr_t*>(d), reinterpret_cast<int*>(d + H->cmp_off[0]),
      reinterpret_cast<std::uint8_t*>(d + H->cmp_off[1]), d + H->cmp_off[2]);
    if (H->gather_timed) { RAFT_CUDA_TRY(cudaEventRecord(H->ev_g1, H->stream.value())); }
  } else {
    H->gather_timed = false;
  }
  H->scores_valid = H->scores_armed;
  H->scores_armed = false;
  H->dive_valid   = rounds > 1;
  H->dive_armed   = false;
  if (early) {
    enqueue_results(H, H->harvest_slim);
    if (H->active_seats != 0 && H->ev_d2h != nullptr && gpu_timing_events()) {
      RAFT_CUDA_TRY(cudaEventRecord(H->ev_d2h, H->stream.value()));
    }
    set_flag_kernel<<<1, 1, 0, H->stream.value()>>>(H->d_completion_flag);
    H->early_d2h = true;
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  tl_relaunch_split[3] = tic() - split_t;
  H->launched = true;
  return true;
}

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_device_lu(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H)
{
  return H != nullptr && H->device_lu;
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_set_pseudocosts(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
  const f_t* sum_down,
  const f_t* sum_up,
  const i_t* num_down,
  const i_t* num_up,
  f_t avg_down,
  f_t avg_up,
  i_t reliable_threshold,
  f_t integer_tol)
{
  if (H == nullptr || !H->use_slots || H->d_var_int == nullptr || H->launched) { return; }
  const size_t n = static_cast<size_t>(H->cols);
  const size_t N = H->h_args.size();
  if (!H->d_scores) {
    H->d_scores.emplace(n * (2 * sizeof(f_t) + 2 * sizeof(i_t)) +
                          N * sizeof(gpu_scores::seat_score_t),
                        H->stream);
    if (!H->h_scores) {
      H->h_scores.emplace(gpu_phase2_detail::pinned_make<gpu_scores::seat_score_t>(H->arena.get(),
        cuda::stream_ref{H->stream.value()}, N));
    }
  }
  auto* base = static_cast<char*>(H->d_scores->data());
  H->pack.add(base, sum_down, n * sizeof(f_t), H->stream);
  H->pack.add(base + n * sizeof(f_t), sum_up, n * sizeof(f_t), H->stream);
  H->pack.add(base + 2 * n * sizeof(f_t), num_down, n * sizeof(i_t), H->stream);
  H->pack.add(base + 2 * n * sizeof(f_t) + n * sizeof(i_t), num_up, n * sizeof(i_t), H->stream);
  H->score_pc = {reinterpret_cast<const double*>(base),
                 reinterpret_cast<const double*>(base + n * sizeof(f_t)),
                 reinterpret_cast<const int*>(base + 2 * n * sizeof(f_t)),
                 reinterpret_cast<const int*>(base + 2 * n * sizeof(f_t) + n * sizeof(i_t)),
                 avg_down, avg_up, reliable_threshold, integer_tol};
  H->scores_armed = true;
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_set_dive(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
                                               int rounds,
                                               const f_t* root_x)
{
  if (H == nullptr || !H->use_slots || !H->view_mode || !H->strengthen || H->launched ||
      rounds < 2) {
    return;
  }
  const size_t n = static_cast<size_t>(H->cols);
  const size_t N = H->h_args.size();
  if (!H->d_dive || H->dive_rounds != rounds) {
    const auto align = [](size_t b) { return (b + 15) / 16 * 16; };
    const size_t R1  = static_cast<size_t>(rounds - 1);
    H->dive_off[0]   = align(n * sizeof(f_t));
    H->dive_off[1]   = H->dive_off[0] + align(N * sizeof(gpu_bound_delta<i_t, f_t>));
    H->dive_off[2]   = H->dive_off[1] + align(N * sizeof(int));
    H->dive_off[3]   = H->dive_off[2] + align(N * R1 * sizeof(gpu_dive_record<i_t, f_t>));
    const size_t total = H->dive_off[3] + N * R1 * ((n + 3) / 4);
    H->d_dive.emplace(total, H->stream);
    H->h_dive.emplace(gpu_phase2_detail::pinned_make<char>(H->arena.get(),
      cuda::stream_ref{H->stream.value()}, total - H->dive_off[1]));
    H->dive_rounds = rounds;
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      H->d_dive->data(), root_x, n * sizeof(f_t), cudaMemcpyHostToDevice, H->stream.value()));
  }
  H->dive_armed = true;
}

template <typename i_t, typename f_t>
gpu_dive_view<i_t, f_t> gpu_simplified_dual_phase2_batch_dive(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H, std::size_t seat)
{
  gpu_dive_view<i_t, f_t> v;
  if (H == nullptr || !H->dive_valid || H->launched) { return v; }
  const size_t R1 = static_cast<size_t>(H->dive_rounds - 1);
  const size_t pk = static_cast<size_t>((H->cols + 3) / 4);
  const char* h   = H->h_dive->data();
  const size_t b  = H->dive_off[1];
  v.count         = reinterpret_cast<const int*>(h)[seat];
  v.records = reinterpret_cast<const gpu_dive_record<i_t, f_t>*>(h + H->dive_off[2] - b) + seat * R1;
  v.packed  = reinterpret_cast<const std::uint8_t*>(h + H->dive_off[3] - b) + seat * R1 * pk;
  v.packed_stride = pk;
  return v;
}

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_lean_fresh(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H)
{
  return H != nullptr && H->view_mode && H->use_slots && H->device_lu &&
         gpu_phase2_detail::gpu_lean_fresh();
}

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_factor_fits(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
  const basis_update_mpf_t<i_t, f_t>& ft,
  bool resume)
{
  if (H == nullptr || !H->use_slots) { return true; }
  auto& fs         = H->slots;
  const size_t nzL = std::max<size_t>(ft.L0().nnz(), ft.L0_transpose().nnz());
  const size_t nzU = std::max<size_t>(ft.U0().nnz(), ft.U0_transpose().nnz());
  fs.record(nzL, nzU);
  if (nzL <= fs.acceptL && nzU <= fs.acceptU) { return true; }
  ++(resume ? fs.overflow_resume : fs.overflow_fresh);
  return false;
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_destroy(
  gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H)
{
  if (H == nullptr) { return; }
  if (H->use_slots && gpu_phase2_detail::g_setup_telemetry.load(std::memory_order_relaxed)) {
    const auto& fs = H->slots;
    const auto& pk = H->pack;
    const auto hist = [](const std::array<std::uint64_t, 11>& h) {
      std::string s;
      for (size_t b = 0; b < h.size(); ++b) { s += (b ? "," : "") + std::to_string(h[b]); }
      return s;
    };
    std::fprintf(stderr,
                 "[GPUBATCH-FACTOR] m=%d seats=%zu capL=%zu capU=%zu acceptL=%zu acceptU=%zu "
                 "bound=%zu lus=%llu overflow_fresh=%llu overflow_resume=%llu maxL=%zu maxU=%zu "
                 "fillL=%s fillU=%s pack_flushes=%llu pack_bytes=%llu pack_pieces=%llu "
                 "pack_grows=%llu\n",
                 (int)H->rows, H->h_args.size(), fs.capL, fs.capU, fs.acceptL, fs.acceptU, fs.bound,
                 (unsigned long long)fs.lus, (unsigned long long)fs.overflow_fresh,
                 (unsigned long long)fs.overflow_resume, fs.maxL, fs.maxU, hist(fs.histL).c_str(),
                 hist(fs.histU).c_str(), (unsigned long long)pk.flushes,
                 (unsigned long long)pk.flushed_bytes, (unsigned long long)pk.flushed_pieces,
                 (unsigned long long)pk.grows);
  }
  // A caller may drain a pool while a launch is still running. Device buffers cannot be released
  // until that stream has finished consuming them.
  if (H->launched && H->ev_stop != nullptr) { RAFT_CUDA_TRY(cudaEventSynchronize(H->ev_stop)); }
  if (!H->staged.empty()) { H->stream.synchronize(); }
  delete H;
}

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_finish(gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>* H,
                                             std::vector<dual::status_t>& out_status,
                                             gpu_batch_timing& timing)
{
  gpu_simplified_dual_phase2_batch_harvest<i_t, f_t>(H, out_status, timing);
  gpu_simplified_dual_phase2_batch_destroy<i_t, f_t>(H);
}

// Synchronous entry point: launch, wait, collect. What the benchmark harness uses.
template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch(const std::vector<gpu_batch_item<i_t, f_t>>& items,
                                      std::vector<dual::status_t>& out_status,
                                      gpu_batch_timing& timing,
                                      bool allow_refactor_exit)
{
  out_status.assign(items.size(), dual::status_t::UNSET);
  auto* H = gpu_simplified_dual_phase2_batch_launch<i_t, f_t>(items, timing,
                                                             rmm::cuda_stream_default,
                                                             allow_refactor_exit);
  if (H == nullptr) { return; }   // rejected; timing.message says why
  gpu_simplified_dual_phase2_batch_finish<i_t, f_t>(H, out_status, timing);
}

// ---------------------------------------------------------------------------
// Public single-problem entry: the host refactor-and-resume loop.
//
// The device stops with REFACTOR_NEEDED at the periodic update limit, when its rank-1 update is
// unsafe (|mu| outside [1e-8, 1e8]), or when its preallocated update store is full. The CPU driver
// responds by refactorizing in place and continuing; the device cannot, because it has no LU of
// its own. So this rebuilds the
// factorization of the basis the device stopped on and calls it straight back, and REFACTOR_NEEDED
// never escapes this function.
//
// The resume is exact, not a restart: the device hands back the basis it reached, its primal point,
// its objective, its maintained duals and its steepest-edge norms, and every one of those goes
// straight back in. Only the factorization is rebuilt. Two details are load-bearing, and both were
// found by a trajectory divergence rather than by reasoning:
//   - The duals are NOT re-priced. The CPU's in-place refactor rebuilds the LU and recomputes x
//     ONLY, leaving the maintained reduced costs alone. A freshly priced z can violate the
//     dual-feasibility signs the ratio test established against the maintained one, which produces
//     negative steps and destroys the iterate.
//   - The primal point is rebuilt with the CPU's post-refactor formula, and `refactor_resume` tells
//     the device not to redo it with its different cold-start one.
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
  const i_t m = lp.num_rows;
  // Bound the loop on progress, not on a total leg count: a long solve may legitimately require
  // many periodic refactorizations, while repeated legs without a pivot indicate a livelock.
  constexpr int kMaxStalledLegs = 8;
  int stalled_legs              = 0;
  const f_t t0                  = cuopt::mathematical_optimization::tic();

  std::vector<f_t> xB_workspace;
  f_t work_estimate = 0.0;
  i_t total_iters   = 0;
  bool resume       = false;

  while (true) {
    i_t refactor_reason = 0;
    const dual::status_t st = gpu_simplified_dual_phase2_leg<i_t, f_t>(
      lp,
      settings,
      vstatus,
      ft,
      basic_list,
      nonbasic_list,
      sol,
      delta_y_steepest_edge,
      resume,
      refactor_reason);

    // ONE solve, however many launches it took: the caller's iteration count must not reset at a
    // refactorization the caller never asked for.
    // ASSIGN, not accumulate: `iter` now continues across a resume (it is carried in via
    // kernel_args::resume_iter), so sol.iterations is already the node's running total.
    total_iters = sol.iterations;
    if (st != dual::status_t::REFACTOR_NEEDED) {
      sol.iterations = total_iters;
      return st;
    }
    stalled_legs = (sol.iterations == 0) ? (stalled_legs + 1) : 0;
    if (stalled_legs > kMaxStalledLegs) {
      sol.iterations = total_iters;
      return dual::status_t::NUMERICAL;
    }

    // Rebuild the factorization from the exact ordered lists returned by the device.  The order is
    // part of the numerical state: refactor_basis factorizes A(:, basic_list), and the ratio test
    // scans nonbasic_list in order.
    // A successful refactorization can still have repaired a rank-deficient basis by replacing
    // columns. The maintained y/z then belong to the pre-repair basis and cannot be resumed.
    bool repaired = false;
    if (ft.refactor_basis(lp.A,
                          settings,
                          lp.lower,
                          lp.upper,
                          t0,
                          basic_list,
                          nonbasic_list,
                          vstatus,
                          &repaired) != 0 ||
        repaired) {
      sol.iterations = total_iters;
      return dual::status_t::NUMERICAL;
    }

    // Refactorization gives us an independent way to price a periodic resume. Use it only as a
    // certificate: replacing the maintained duals changes the trajectory. A rejected rank-1
    // update is deliberately exempt, matching the B&B resume path: its maintained duals are part
    // of the established recovery trajectory and fresh pricing can temporarily violate their
    // signs even though the next leg restores feasibility.
    if (refactor_reason != 1) {
      std::vector<f_t> fresh_y(static_cast<size_t>(m), 0.0);
      std::vector<f_t> fresh_z(static_cast<size_t>(lp.num_cols), 0.0);
      f_t certificate_work = 0.0;
      phase2::compute_dual_solution_from_basis(
        lp, ft, basic_list, nonbasic_list, fresh_y, fresh_z, certificate_work);
      if (!resume_dual_feasible(vstatus, fresh_z, settings.dual_tol)) {
        sol.iterations = total_iters;
        return dual::status_t::NUMERICAL;
      }
    }

    phase2::compute_primal_solution_from_basis(
      lp, ft, basic_list, nonbasic_list, vstatus, sol.x, xB_workspace, work_estimate);
    resume = true;
  }
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
  std::vector<dual::status_t>&,
  gpu_batch_timing&, bool);

template gpu_phase2_detail::gpu_batch_handle_t<int, double>*
gpu_simplified_dual_phase2_batch_launch<int, double>(
  const std::vector<gpu_batch_item<int, double>>&, gpu_batch_timing&, rmm::cuda_stream_view,
  bool, gpu_phase2_detail::gpu_ro_cache_t<int, double>*, bool, bool);
template bool gpu_simplified_dual_phase2_batch_device_lu<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*);
template void gpu_simplified_dual_phase2_batch_set_dive<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, int, const double*);
template bool gpu_pinned_arena_reserve<int, double>(const std::vector<int>&, std::size_t, int, int, int, int, std::size_t);
template gpu_dive_view<int, double> gpu_simplified_dual_phase2_batch_dive<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, std::size_t);
template bool gpu_simplified_dual_phase2_batch_lean_fresh<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*);
template void gpu_simplified_dual_phase2_batch_set_pseudocosts<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, const double*, const double*, const int*,
  const int*, double, double, int, double);
template bool gpu_simplified_dual_phase2_batch_factor_fits<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, const basis_update_mpf_t<int, double>&, bool);
template gpu_phase2_detail::gpu_ro_cache_t<int, double>* gpu_ro_cache_create<int, double>();
template gpu_phase2_detail::gpu_ro_cache_t<int, double>* gpu_ro_cache_create_borrowed<int, double>(
  rmm::cuda_stream_view);
template void gpu_ro_cache_destroy<int, double>(gpu_phase2_detail::gpu_ro_cache_t<int, double>*);
template void gpu_ro_cache_prebuild<int, double>(gpu_phase2_detail::gpu_ro_cache_t<int, double>*,
                                                 const lp_problem_t<int, double>&);
template bool gpu_simplified_dual_phase2_batch_ready<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*);
template void gpu_simplified_dual_phase2_batch_finish<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*,
  std::vector<dual::status_t>&, gpu_batch_timing&);
template void gpu_simplified_dual_phase2_batch_harvest<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*,
  std::vector<dual::status_t>&, gpu_batch_timing&, bool);
template void gpu_simplified_dual_phase2_batch_fetch_seat<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, std::size_t);
template void gpu_simplified_dual_phase2_batch_fetch_seats<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, const std::vector<int>&);
template void gpu_simplified_dual_phase2_batch_read_seat<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, std::size_t, std::vector<variable_status_t>&,
  std::vector<double>&, std::vector<double>&, std::vector<double>&);
template bool gpu_simplified_dual_phase2_batch_park<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, std::size_t);
template void gpu_simplified_dual_phase2_batch_unstage<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, std::size_t, std::size_t);
template bool gpu_simplified_dual_phase2_batch_launched<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*);
template void gpu_simplified_dual_phase2_batch_use_views<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*);
template gpu_batch_result_view<int, double> gpu_simplified_dual_phase2_batch_result<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, std::size_t);
template bool gpu_simplified_dual_phase2_batch_stage<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*,
  std::size_t,
  const gpu_batch_item<int, double>&,
  gpu_batch_seat_command,
  const std::vector<gpu_bound_delta<int, double>>&,
  std::string&);
template bool gpu_simplified_dual_phase2_batch_relaunch<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*, gpu_batch_timing&, std::string&);
template void gpu_simplified_dual_phase2_batch_destroy<int, double>(
  gpu_phase2_detail::gpu_batch_handle_t<int, double>*);


int gpu_max_blocks_per_sm()
{
  using namespace gpu_phase2_detail;
  int blocks = 0;
  const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks, gpu_phase2_kernel_direct<>, 32, 0);
  return (rc == cudaSuccess) ? blocks : 0;
}

}  // namespace cuopt::linear_programming::dual_simplex
