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

#include "dense_triangle_solve.cuh"

#include <dual_simplex/basis_solves.hpp>

#include <linear_algebra/sparse_matrix.hpp>
#include <linear_algebra/sparse_vector.hpp>
#include <math_optimization/tic_toc.hpp>

#include <utilities/macros.cuh>
#include <utilities/manual_cuda_graph.cuh>   // run_while: the device-controlled super-step loop

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/limits>
#include <cuda/std/span>

#include <new>  // placement new: construct the per-slot persistent state in host-allocated memory

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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



// Set when a span is asked to exceed the capacity the host layout reserved for it, or when
// the host and device slot walks disagree. Nothing allocates any more, so this is a LAYOUT
// bug, not a resource shortage; the driver bails with NUMERICAL and the host reports it.
// Shared across all blocks in a batch launch (a single failing block poisons the
// whole batch by design → surfaces as a loud verification mismatch). Reset from
// the host before each launch; set with atomicOr so concurrent blocks don't tear.
__device__ int g_gpu_capacity_exceeded;

// Next independent problem claimed by the persistent work-stealing grid. Reset by the host before
// every launch; only lane 0 of each block updates it, then publishes the claimed index to its warp.
__device__ unsigned int g_next_problem;

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
__device__ int g_elide_noop;

// REORDERING THE LEVEL SCHEDULE IS A DEAD END, measured 2026-08-26. Emitting each level by
// descending column length instead of ascending index was built and measured at only -1.0%, and
// removed as not worth an LSD radix pass in the lane-0 prologue. The general reason: a level's cost
// equals `max(longest column, ceil(work/32))` to within 1-5% as scheduled today, so a level costs
// the length of its single longest column and no permutation can beat that. SELL-C-sigma and
// two-columns-per-lane were also built and lost (+7.5%, +7.2%).


static constexpr i_t kColNoop = static_cast<i_t>(0x80000000u);  // level[] sign bit
static constexpr i_t kSparseReachDenseFallbackDivisor = 20;     // dense at >= 5% reach
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
  // a sizing bug, not a condition to recover from: it raises the flag and the write is skipped.
  __device__ void require_capacity(size_t n) const
  {
    if (n > cap_) { atomicOr(&g_gpu_capacity_exceeded, 1); }
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
// transposes. col_start is read-only because reach()/DFS keeps visited state in a private byte-mark
// workspace instead of temporarily modifying the factor.
// ---------------------------------------------------------------------------
// RI is the row-index storage type. A and S_ retain full-width indices; only the four immutable LU
// factors use uint16_t. Factor dimensions and every uploaded row index are checked before narrowing.
template <typename RI = i_t>
struct gpu_csc_view_t {
  i_t m{0}, n{0}, nz_max{0};
  gpu_span<const i_t> col_start;
  gpu_span<RI> i;
  gpu_span<f_t> x;
  __device__ i_t nnz() const { return col_start[n]; }
};

using gpu_csc_view = gpu_csc_view_t<i_t>;
template <typename FactorIndex>
using gpu_factor_view_t = gpu_csc_view_t<FactorIndex>;

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
template <typename FactorIndex>
struct gpu_basis_update_mpf_t;

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
template <typename FactorIndex>
struct warp_solve_config_t {
  // ONLY the elimination-level SCHEDULE survives. Every per-dispatch REQUEST field (x/op and the
  // tl_/dz_/dzs_/td_/fd_/rt_/su_/pi_/ud_/nbe_/sp_op/sp_nlev groups, ~95 fields, ~700 of 928 bytes)
  // was staging for the parked-helper dispatch and was deleted with it on 2026-08-11: the Direct
  // path passes those values as call arguments to the warp_*_body collectives instead.
  // Also gone: diag_first/descending/incremental/m, which warp_build_config wrote but nothing ever
  // read — the four wavefront bodies encode their own direction, so nothing consumes them.
  // Built once (by the kernel prologue on lane 0), read-only after.
  const i_t* cs[kNumWarpSolveOps];      // gather CSC col_start
  const FactorIndex* ridx[kNumWarpSolveOps];    // gather CSC row indices
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
template <int Op, typename FactorIndex>
__device__ __forceinline__ void warp_solve_body_op(const i_t* a_corder, const i_t* a_cs,
                                                   const i_t* a_lstart, i_t a_nlev,
                                                   const FactorIndex* a_ridx,
                                                   const f_t* a_val, f_t* a_x)
{
  dts::wavefront_solve_op<Op>(a_cs, a_ridx, a_val, a_x, a_lstart, a_corder, a_nlev);
}


template <int Op, typename FactorIndex>
__device__ __forceinline__ void warp_solve2_body_op(const i_t* a_corder, const i_t* a_cs,
                                                    const i_t* a_lstart, i_t a_nlev,
                                                    const FactorIndex* a_ridx,
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
    const f_t theta = dot * inv_mu;
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
                                                 const f_t* a_inv_mu,
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
    const f_t inv_mu = a_inv_mu[k];
    const f_t theta0 = dot0 * inv_mu;
    const f_t theta1 = dot1 * inv_mu;
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
      if (s == variable_status_t::NONBASIC_LOWER && a_rt_dz[j] < -ptol) {
        ratio = (-dtol - a_rt_z[j]) / a_rt_dz[j];
      } else if (s == variable_status_t::NONBASIC_UPPER && a_rt_dz[j] > ptol) {
        ratio = (dtol - a_rt_z[j]) / a_rt_dz[j];
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
__device__ __forceinline__ void warp_pinf_body(const i_t* a_pi_basic, i_t* a_pi_count, i_t* a_pi_idx, const f_t* a_pi_lower, i_t a_pi_m, f_t* a_pi_pinf, f_t* a_pi_pinf_sq, f_t* a_pi_sqinf, f_t a_pi_tol, const f_t* a_pi_upper, const f_t* a_pi_x)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t m = a_pi_m;
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
                                                       const f_t* a_inv_mu,
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
  i_t nz         = *a_nz;   // warp-uniform on entry, and every update below keeps it uniform
  for (i_t step = 0; step < a_num; ++step) {
    const i_t k = Ascending ? step : (a_num - 1 - step);
    const f_t inv_mu = a_inv_mu[k];
    const i_t dcol = 2 * k + DotOffset;
    const i_t d0   = a_cs[dcol];
    const i_t d1   = a_cs[dcol + 1];
    const i_t dlen = d1 - d0;
    f_t dot;
    if (dlen <= 2) {
      dot = 0.0;
      if (dlen >= 1) { const i_t i = a_ri[d0];     if (a_mark[i]) { dot = a_val[d0] * a_x[i]; } }
      if (dlen == 2) { const i_t i = a_ri[d0 + 1]; if (a_mark[i]) { dot += a_val[d0 + 1] * a_x[i]; } }
      __syncwarp(0xffffffffu);   // separates these x READS from the axpy WRITES below
    } else {
      f_t partial = 0.0;
      for (i_t p = d0 + lane; p < d1; p += 32) {
        const i_t i = a_ri[p];
        if (a_mark[i]) { partial += a_val[p] * a_x[i]; }
      }
      dot = warp_reduce_add(partial);
    }
    const f_t theta = dot * inv_mu;
    if (fabs(theta) > a_tol) {
      // The dot and AXPY CSC columns are adjacent, so one AXPY boundary is already in d0/d1.
      // Compile-time offsets let both directions reuse it without adding a runtime branch.
      const i_t a0 = (AxpyOffset > DotOffset) ? d1 : a_cs[dcol - 1];
      const i_t a1 = (AxpyOffset > DotOffset) ? a_cs[dcol + 2] : d0;
      for (i_t base = a0; base < a1; base += 32) {
        const i_t p  = base + lane;
        const bool have = (p < a1);
        i_t i        = 0;
        bool isnew   = false;
        if (have) {
          i     = a_ri[p];
          isnew = (a_mark[i] == 0);   // distinct i across lanes: a CSC column has distinct rows
        }
        const unsigned ball = __ballot_sync(0xffffffffu, isnew);
        const int prefix    = __popc(ball & ((1u << lane) - 1u));
        if (isnew) {
          a_mark[i]                 = 1;
          a_mark[a_m + nz + prefix] = i;   // append, in increasing p == the serial append order
        }
        nz += __popc(ball);                // warp-uniform: identical ballot on every lane
        if (have) { a_x[i] += (-theta) * a_val[p]; }
        __syncwarp(0xffffffffu);           // publish marks/x before the next chunk reads them
      }
    }
    // An active AXPY always ends with the inner __syncwarp above, which publishes its last chunk
    // before the next update reads x. An inactive AXPY performs no writes. A second unconditional
    // barrier here was therefore redundant (and especially costly for short MPF columns).
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
template <int Op, typename FactorIndex>
__device__ __forceinline__ void warp_sptrsv_body_op(const i_t* a_cs,
                                                    const FactorIndex* a_ridx,
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
template <typename RI>
__device__ int warp_build_levels(const gpu_csc_view_t<RI>& mat, bool diag_first, bool asc_process,
                                  gpu_span<i_t>& level, gpu_span<i_t>& level_start,
                                  gpu_span<i_t>& col_order, gpu_span<i_t>& pos)
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
  // No-op columns (see g_elide_noop). With length 1 the diagonal sits at col_start[j] whichever end
  // diag_first puts it, so one test covers all four operators. Length 0 cannot occur in a
  // nonsingular triangular factor and is left to the existing (unconditional val[dp]) path.
  const bool elide = (g_elide_noop != 0);
  auto is_noop = [&](i_t j) {
    return elide && mat.col_start[j + 1] - mat.col_start[j] == 1 && mat.x[mat.col_start[j]] == 1.0;
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
  // -- so removing it retires the whole malloc path and g_gpu_capacity_exceeded with it. And the VRAM
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
  if (elide) {
    for (i_t j = 0; j < m; ++j) { if (is_noop(j)) { level[j] |= kColNoop; } }
  }
  return static_cast<int>(nlev);
}


// ---- sparse triangular solve (reach + iterative DFS) ----------------------
// Visited state lives in a private byte array so the factor CSC remains read-only. Only lane 0
// walks the DFS, so byte loads/stores are sufficient; reached entries are cleared on exit instead
// of sweeping all m rows per solve.

// pinv == nullptr means no permutation.
template <typename RI>
__device__ i_t depth_first_search(
  i_t j, const i_t* pinv, const gpu_csc_view_t<RI>& G, unsigned char* marked, i_t top, i_t* xi,
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

using reach_mark_word = unsigned int;

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
template <typename RI>
__device__ i_t depth_first_search_bits(
  i_t root, const gpu_csc_view_t<RI>& G, reach_mark_word* marked, i_t top, i_t* xi, i_t* pstack)
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
template <typename RI>
__device__ i_t depth_first_search_hot(
  i_t root, const gpu_csc_view_t<RI>& G, reach_mark_word* marked, i_t top, i_t* xi,
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

template <typename RI>
__device__ i_t reach_hot(
  const gpu_sparse_vector& b, const gpu_csc_view_t<RI>& G, reach_mark_word* marked, i_t* xi,
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

template <typename RI>
__device__ i_t reach(
  const gpu_sparse_vector& b, const i_t* pinv, const gpu_csc_view_t<RI>& G, unsigned char* marked,
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
template <typename FactorIndex>
struct gpu_basis_update_mpf_t {
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
  using factor_index_type = FactorIndex;
  using factor_view_type  = gpu_factor_view_t<FactorIndex>;
  using warp_config_type  = warp_solve_config_t<FactorIndex>;

  factor_view_type L0_, U0_, L0_transpose_, U0_transpose_;
  gpu_span<i_t> row_permutation_, inverse_row_permutation_;
  gpu_csc_matrix S_;
  gpu_span<f_t> mu_values_;
  gpu_span<i_t> xi_workspace_;  // size 2*m (scratch)
  gpu_span<f_t> x_workspace_;   // size m   (scratch)
  gpu_span<unsigned char> reach_mark_;  // size m, private visited bytes for immutable factor DFS
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
  // NaN-recovery dense re-solve in compute_delta_x. Deliberately NOT arena-bound and never
  // pre-sized: the branch is rare (0 of the 100 v6 nodes reach it), so these stay empty and cost
  // nothing until the first time it fires. They exist because the recovery must be a WARP
  // collective -- stack-local gpu_spans would give each of the 32 lanes a private malloc, and
  // the collective would then write 32 different buffers.
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

  // All of these are slices of the slot memory, wired once by attach_all_buffers.
  //
  // S_ and mu_values_ are preallocated too. They used to be the one genuinely dynamic allocation --
  // one column per pivot, unbounded within a refactor window -- which is what forced device
  // malloc() to stay alive and what made a slot impossible to reuse across launches. With
  // update_cap_/eta_nnz_cap_ set they are preallocated once by reserve_store() and never grow, so
  // they are now fixed-size preallocated buffers; overflow is reported, never allocated.

  __device__ gpu_basis_update_mpf_t() {}
  gpu_basis_update_mpf_t(const gpu_basis_update_mpf_t&)            = delete;
  gpu_basis_update_mpf_t& operator=(const gpu_basis_update_mpf_t&) = delete;
  __device__ ~gpu_basis_update_mpf_t()                             = default;

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
    const warp_config_type* c = warp_cfg_;
    warp_solve_body_op<Op>(c->corder[Op], c->cs[Op], c->lstart[Op], c->nlev[Op], c->ridx[Op],
                           c->val[Op], rhs);
  }

  template <int Op>
  __device__ void warp_dispatch2(f_t* rhs0, f_t* rhs1) const
  {
    const warp_config_type* c = warp_cfg_;
    warp_solve2_body_op<Op>(c->corder[Op], c->cs[Op], c->lstart[Op], c->nlev[Op], c->ridx[Op],
                            c->val[Op], rhs0, rhs1);
  }

  // Post the MPF rank-1 T-loop to the whole warp (op == WARP_TLOOP). One dispatch runs the
  // entire replay collectively (tl_num internal barriers). S_ pointers are re-read each call
  // (they can move on resize). dot col = 2k+dotoff, axpy col = 2k+axpyoff; asc selects
  // l_solve (k up) vs l_transpose (k down).
  __device__ void warp_dispatch_tloop(f_t* rhs, bool asc, i_t dotoff, i_t axpyoff, f_t tol) const
  {
    // S_ pointers are re-read here (they move on resize) — every lane reads the same globals.
    warp_tloop_body(asc, axpyoff, S_.col_start.data(), dotoff, mu_values_.data(), num_updates_,
                    S_.i.data(), tol, S_.x.data(), rhs);
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

    // (2) zero x over the reach, then scatter the rhs — collective, barrier-separated.
    warp_sptrsv_preload_body(b.i.data(), static_cast<i_t>(b.i.size()), b.x.data(), m, top, xi, x);

    // Once the reach exceeds the routing budget, skip the serial level bucketing and use the fixed
    // dense schedule. The reach closure guarantees entries outside the preloaded pattern stay zero.
    const i_t dense_fallback_reach =
      (m + kSparseReachDenseFallbackDivisor - 1) / kSparseReachDenseFallbackDivisor;
    if (m - top >= dense_fallback_reach) {
      warp_solve_body_op<Op>(c->corder[op], c->cs[op], c->lstart[op], c->nlev[op], c->ridx[op],
                             c->val[op], x);
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

    warp_sptrsv_body_op<Op>(c->cs[Op], c->ridx[Op], c->val[Op], c->sp_co, c->sp_ls,
                            static_cast<int>(nlevR), x);
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
      // T^T: updates descend; dot u_col (2k), then AXPY v_col (2k+1).
      warp_tloop_sparse_body<false, 0, 1>(
        S_.col_start.data(), mu_values_.data(), num_updates_, S_.i.data(), zero_tol,
        S_.x.data(), m, xi_workspace_.data(), &nz, x_workspace_.data());
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
      warp_tloop2_body<true, 0, 1>(S_.col_start.data(), mu_values_.data(), num_updates_,
                                   S_.i.data(), f_t{1e-16}, S_.x.data(), rhs0, rhs1);
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
      // S_ pointers are re-read here (they move on resize) — every lane reads the same globals.
      warp_tloop_sparse_body<true, 1, 0>(
        S_.col_start.data(), mu_values_.data(), num_updates_, S_.i.data(), zero_tol,
        S_.x.data(), m, xi_workspace_.data(), &nz, x_workspace_.data());
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
  __device__ i_t warp_update(const gpu_sparse_vector& utilde, gpu_sparse_vector& etilde,
                             i_t leaving_index)
  {
    const i_t m = L0_.m;
    i_t nz      = static_cast<i_t>(utilde.i.size());

    warp_scatter_ws_body(utilde.i.data(), utilde.x.data(), m, nz, xi_workspace_.data(),
                         x_workspace_.data());

    // add_sparse_column appends fill to the index list through a running cursor, and gpu_sort is
    // serial; grow_storage allocates. One lane, one region.
    i_t base = 0;
    lane0_serial([&] {
      add_sparse_column(U0_, leaving_index, -1.0, xi_workspace_.data(), nz, x_workspace_.data());
      gpu_sort(xi_workspace_.data() + m, xi_workspace_.data() + m + nz,
               [] __device__(i_t a, i_t b) { return a < b; });
      i_t S_start_l = 0, S_nz_l = 0;
      if (grow_storage(nz + static_cast<i_t>(etilde.i.size()), S_start_l, S_nz_l)) {
        base = S_.col_start[S_.n];
      }
    });
    // The store is full: stop before touching S_, so the stack stays exactly as it was and the host
    // can refactor from the CURRENT basis. Mirrors the bail-out below -- workspace not cleared,
    // num_updates_ not bumped.
    if (lane0_bcast(store_exhausted_ ? 1 : 0)) { return 2; }
    nz   = lane0_bcast(nz);     // add_sparse_column grew it on lane 0 only
    base = lane0_bcast(base);
    // S_.i/S_.x may have been relocated by grow_storage — data() is read here, after the barrier.
    const i_t end1 =
      warp_append_scatter_body(base, S_.i.data(), S_.x.data(), xi_workspace_.data() + m, nz,
                               x_workspace_.data());

    i_t base2 = 0;
    lane0_serial([&] {
      S_.col_start[S_.n + 1] = end1;
      S_.n++;
      etilde.sort(sort_bucket_, sort_i_, sort_x_, sort_perm_);   // reused scratch, may reallocate
      base2 = S_.col_start[S_.n];
    });
    base2 = lane0_bcast(base2);
    const i_t enz  = static_cast<i_t>(etilde.i.size());
    const i_t end2 =
      warp_append_sv_body(base2, S_.i.data(), S_.x.data(), etilde.i.data(), etilde.x.data(), enz);

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
      else { mu_values_.push_back(f_t{1.0} / mu); }
    });
    status = lane0_bcast(status);
    // Matches the serial version: on the numerical bail-out the workspace is NOT cleared and
    // num_updates_ is NOT bumped.
    if (status == 1) { return 1; }

    warp_clear_ws_body(m, xi_workspace_.data(), nz, x_workspace_.data());
    lane0_serial([&] { num_updates_++; });
    return 0;
  }

};

using gpu_basis_update_mpf_u16 = gpu_basis_update_mpf_t<std::uint16_t>;
using gpu_basis_update_mpf_i32 = gpu_basis_update_mpf_t<i_t>;

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

// Exact steepest-edge norm init: gamma_j = ||B^-T e_k||^2 for each basic var, via a
// singleton-row fast path (1/alpha^2) or a b_transpose_solve. Mirrors the CPU
// initialize_steepest_edge_norms (the simplified always uses the non-slack/exact path).
// All scratch is CALLER-OWNED (slot-resident). Under Direct every lane executes this, so a
// stack-local gpu_span would device-malloc PER LANE — 32 private buffers, and the collective
// b_transpose_solve below would then operate on lane-private memory. The original even allocated a
// fresh `sparse_ei` inside the per-column loop.
template <class FT>
__device__ i_t initialize_steepest_edge_norms(const gpu_lp& lp,
                                              const i_t* basic_list,
                                              FT& ft,
                                              f_t* delta_y_steepest_edge,
                                              gpu_span<i_t>& row_degree,
                                              gpu_span<i_t>& mapping,
                                              gpu_span<f_t>& coeff,
                                              gpu_sparse_vector& sparse_dy,
                                              gpu_sparse_vector& UTsol_scratch,
                                              gpu_sparse_vector& sparse_ei)
{
  const i_t m = lp.num_rows;
  lane0_serial([&] {
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
      lane0_serial([&] {
        sparse_ei.x[0] = -1.0;
        sparse_ei.i[0] = k;
        sparse_dy.clear();
      });
      ft.b_transpose_solve(sparse_ei, sparse_dy, UTsol_scratch);
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
// setup() is the one-time S1-S7 prologue, kept as its own method so a persistent slot can
// later run it as a dedicated setup kernel for a freshly refilled slot.
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
  f_t* se_coeff;
  i_t* basic_mark;
  i_t* delta_z_mark;
  i_t* delta_z_indices;
  i_t* nonbasic_pos;
  i_t* infeas_idx;
  i_t* se_row_degree;
  i_t* se_mapping;
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

template <class FT>
struct slot_state {
  // Stored as i_t so the host can read it back as a plain int array.
  i_t status{static_cast<i_t>(dual::status_t::UNSET)};


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
  // LANE. All start empty, so they cost nothing until the path that uses them actually runs (the
  // SE-init group only runs when the node carries no edge norms, i.e. dy_se_empty).
  gpu_span<f_t>   setup_rhs;                       // compute_primal_variables
  gpu_span<i_t>   se_row_degree, se_mapping;       // initialize_steepest_edge_norms
  gpu_span<f_t>   se_coeff;
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
    se_coeff.attach(b.se_coeff, mn);
    basic_mark.attach(b.basic_mark, mn);
    delta_z_mark.attach(b.delta_z_mark, mn);
    delta_z_indices.attach(b.delta_z_indices, mn);
    nonbasic_pos.attach(b.nonbasic_pos, mn);
    infeas_idx.attach(b.infeas_idx, mn);
    se_row_degree.attach(b.se_row_degree, mn);
    se_mapping.attach(b.se_mapping, mn);
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

  // One-time prologue: S1 bounds -> S2 primal -> S3 SE-init -> S4 workspaces -> S5 Arow ->
  // S6 nonbasic partition -> S7 primal infeasibilities. Byte-for-byte the setup the monolithic
  // driver used to run inline, in the same order (bit-exactness depends on that order).
  // Direct=true: the prologue is almost entirely allocation and
  // read-modify-write (assign/reserve/clear, sparse-vector construction, Arow build) so it stays on
  // lane 0; the collectives are opted in one at a time. The serial parts deliberately call the
  // <false> variants so porting one collective cannot silently move any other computation.
  __device__ __noinline__ void setup(const gpu_lp& lp,
                                     const gpu_settings& settings,
                                     variable_status_t* vstatus,
                                     i_t* basic_list,
                                     i_t* nonbasic_list,
                                     gpu_solution& sol,
                                     f_t* delta_y_steepest_edge,
                                     bool dy_se_empty,
                                     bool refactor_resume)
  {
    const i_t m = lp.num_rows;
    const i_t n = lp.num_cols;
    f_t* x      = sol.x.data();
    f_t* z      = sol.z.data();

    // assign (not resize) throughout: a slot is REUSED by the next B&B node, so every buffer must be
    // fully re-initialized, not just grown — resize would leave the previous node's tail values.
    // assign() may device-malloc, so it stays on lane 0; the O(n) bound-setting scan next to it does
    // not, so the two are split rather than sharing one lane-0 region.
    lane0_serial([&] { xB_workspace.assign(m, 0.0); });

    // On a REFACTOR_NEEDED resume the host has already rebuilt x with
    // phase2::compute_primal_solution_from_basis. Redoing it here would use a different formula
    // -- these two set the nonbasic values from the sign of z and skip columns with
    // |x_j| < 10*tight_tol -- and that difference moves the trajectory from the first refactor on.
    if (!refactor_resume) {
      set_primal_variables_on_bounds(lp, settings, z, vstatus, x);

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

    if (dy_se_empty) {
      // Exact steepest-edge init (matches the CPU simplified's unconditional exact path).
      lane0_serial([&] {
        for (i_t k = 0; k < n; ++k) { delta_y_steepest_edge[k] = -1.0; }
      });
      initialize_steepest_edge_norms(lp, basic_list, ft, delta_y_steepest_edge,
                                                   se_row_degree, se_mapping, se_coeff, se_dy,
                                                   se_UTsol, se_ei);
    } else {
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
    compute_initial_primal_infeasibilities(
      lp, settings, ft, basic_list, x, squared_infeas, infeas_idx, primal_inf);

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
    obj  = compute_objective(lp, x);
    iter = 0;
    });  // end lane-0 epilogue
  }
};

// `Direct`: all 32 lanes execute the body. Direct=false is the historical single-thread monolith
// (lane 0 alone runs this body); Direct=true means ALL 32 lanes run it, so every allocation and
// read-modify-write must be inside lane0_serial and every `return` must be warp-uniform.
template <class FT, bool AllowRefactorExit>
__device__ dual::status_t gpu_simplified_dual_phase2_device(const gpu_lp& lp,
                                                            const gpu_settings& settings,
                                                            variable_status_t* vstatus,
                                                            slot_state<FT>& st,
                                                            i_t* basic_list,
                                                            i_t* nonbasic_list,
                                                            gpu_solution& sol,
                                                            f_t* delta_y_steepest_edge,
                                                            bool dy_se_empty,
                                                            bool refactor_resume)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  f_t* x = sol.x.data();
  f_t* y = sol.y.data();
  f_t* z = sol.z.data();

  // Prologue into the slot. Unconditional: one launch is one solve, so this always runs. It used to
  // be guarded by st.inited so a re-entering fission super-step could skip it and reuse the live
  // state; there are no super-steps any more.
  st.setup(lp,
           settings,
           vstatus,
           basic_list,
           nonbasic_list,
           sol,
           delta_y_steepest_edge,
           dy_se_empty,
           refactor_resume);

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

  const i_t iter_limit = min(500000, max(20000, 20 * (m + n)));
  while (iter < iter_limit) {
    if (g_gpu_capacity_exceeded) {
      finish(iter);
      return dual::status_t::NUMERICAL;
    }
    // Time limit is handled on the host side

    // 1. Steepest-edge pricing.
    i_t direction        = 0;
    i_t basic_leaving    = -1;
    f_t max_val;
    const i_t leaving = steepest_edge_pricing_with_infeasibilities(
      lp, settings, x, delta_y_steepest_edge, basic_mark.data(), squared_infeas, infeas_idx,
      direction, basic_leaving, max_val);
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

    // Steepest-edge norm drift safeguard: dy_norm_sq is the true norm of the leaving
    // row. If the maintained norm used for pricing has drifted below
    // settings.steepest_edge_ratio of it, the leaving choice was made on a stale norm --
    // reset it and re-price (strict port of the CPU simplified, phase2.cpp:4037).
    // monolith is the bit-exact reference.
    if (delta_y_steepest_edge[leaving] < settings.steepest_edge_ratio * dy_norm_sq) {
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
    const bool used_transpose_dz    = delta_y_nz_percentage <= 30.0;
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

    // 4. Simple ratio test.
    f_t step_length       = 0.0;
    i_t nonbasic_entering = -1;
    i_t entering          = -1;
    // The ordered selector writes its outputs through lane 0, so aim
    // them at slot_state (global) and let the __syncwarp publish them to every lane — locals would
    // only ever be set on lane 0.
    warp_rt_body(settings.dual_tol / 10, delta_z.data(), &st.entering, &st.nonbasic_entering,
                 settings.pivot_tol, &st.step_length, vstatus, z, settings.zero_tol,
                 nonbasic_list, n - m);
    __syncwarp(0xffffffffu);
    entering          = st.entering;
    nonbasic_entering = st.nonbasic_entering;
    step_length       = st.step_length;
    if (entering == -1) {
      finish(iter);
      return dual::status_t::DUAL_UNBOUNDED;
    }

    update_dual_variables(
      ft, delta_y_sparse, delta_z_indices, delta_z.data(), step_length, leaving, y, z);

    // FTran.
    lane0_serial([&] {   // clear() is a header RMW; from_csc_column reserve+push_back mallocs
      utilde_sparse.clear();
      scaled_dxB_sparse.clear();
      v_sparse.clear();
      rhs_sparse.from_csc_column(lp.A, entering);
    });
    ft.b_solve2(rhs_sparse, delta_y_sparse, scaled_dxB_sparse, utilde_sparse, v_sparse);
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

    // A cutoff node is complete after its primal/objective update. The remaining state is only
    // needed by a following pivot, so do not maintain it for a node that is being fathomed.
    if (isfinite(settings.cut_off) && obj >= settings.cut_off) {
      finish(iter + 1);
      return dual::status_t::CUTOFF;
    }

    update_steepest_edge_norms(settings, basic_list, ft, direction, delta_y_sparse, dy_norm_sq,
                               scaled_dxB_sparse, basic_leaving, entering, v.data(), v_sparse,
                               delta_y_steepest_edge);

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
      if (ft.num_updates() > settings.refactor_frequency) {
        finish(iter + 1);
        return dual::status_t::REFACTOR_NEEDED;
      }
    }

    _fu_status = ft.warp_update(utilde_sparse, UTsol_sparse, basic_leaving);
    _fu_status = lane0_bcast(_fu_status);
    if (_fu_status != 0) {
      // The CPU driver responds to an unsafe rank-1 update by refactorizing in place. The
      // single-problem GPU path asks its host resume loop to do the same. The pivot and basis swap
      // already happened, so resume from iter + 1 and do not repeat the pivot.
      if constexpr (AllowRefactorExit) {
        finish(iter + 1);
        return dual::status_t::REFACTOR_NEEDED;
      }
      finish(iter);
      return dual::status_t::NUMERICAL;
    }

    compute_initial_primal_infeasibilities(
      lp, settings, ft, basic_list, x, squared_infeas, infeas_idx, primal_inf);
    clear_delta_z(entering, leaving, delta_z_mark.data(), delta_z_indices, delta_z.data());

    // Increment only on a COMPLETED iteration, at the end of the body — matching the CPU
    // simplified driver's iter++ (phase2.cpp:4370). The SE-reset `continue` above jumps
    // back to `while` without counting, so `iter` counts pivots, not loop passes.
    lane0_serial([&] { iter++; });   // RMW on st.iter in global memory

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
  // Host-built CSR of A (Arow). Always uploaded; the kernel has no transpose of its own.
  const i_t* Arow_row_start;
  const i_t* Arow_j;
  const f_t* Arow_x;
  f_t *rhs, *lower, *upper, *objective;
  f_t obj_scale, obj_constant;
  // Fixed-factor row-index width is selected once per launch.
  i_t* L0_col_start;
  void* L0_i;
  f_t* L0_x;
  i_t L0_nz;
  i_t* U0_col_start;
  void* U0_i;
  f_t* U0_x;
  i_t U0_nz;
  i_t* L0t_col_start;
  void* L0t_i;
  f_t* L0t_x;
  i_t L0t_nz;
  i_t* U0t_col_start;
  void* U0t_i;
  f_t* U0t_x;
  i_t U0t_nz;
  i_t *row_perm, *inv_row_perm;
  i_t* xi_workspace;
  f_t* x_workspace;
  unsigned char* reach_mark;
  i_t num_updates;
  i_t refactor_frequency;
  // Caps on the preallocated MPF rank-1 store (see gpu_basis_update_mpf_t). 0/0 = uncapped, i.e.
  // the historical grow-in-kernel behaviour. Sized by mpf_store_caps().
  i_t mpf_update_cap;
  i_t mpf_eta_nnz_cap;
  // Per-slot PERSISTENT solve state (slot_state<FT>): host-allocated, ZEROED blob that the kernel
  // placement-news into on first entry and reuses thereafter. This is what lets the per-pivot
  // workspaces + ft's mutable state (S_, routing stats, scratch) outlive a single kernel launch —
  // host-visible "already constructed" flag (0 = fresh node → construct + run setup).
  void* slot;
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
  i_t dy_se_empty;
  // Non-zero when this launch RESUMES after a host refactorization: the host already rebuilt the
  // primal point the CPU's way, so setup() must leave x alone.
  i_t refactor_resume;
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
  v.col_start = gpu_span<const i_t>(col_start, cols + 1);
  v.i         = gpu_span<i_t>(idx, nz);
  v.x         = gpu_span<f_t>(val, nz);
  return v;
}

template <typename FactorIndex>
__device__ gpu_factor_view_t<FactorIndex> make_factor_view(
  i_t rows, i_t cols, i_t nz, i_t* col_start, FactorIndex* idx, f_t* val)
{
  gpu_factor_view_t<FactorIndex> v;
  v.m         = rows;
  v.n         = cols;
  v.nz_max    = nz;
  v.col_start = gpu_span<const i_t>(col_start, cols + 1);
  v.i         = gpu_span<FactorIndex>(idx, nz);
  v.x         = gpu_span<f_t>(val, nz);
  return v;
}


// Fill the warp-solve config for all 4 operators (lane 0, once at start). The
// level schedules are written into the caller-owned gpu_spans ls[o]/co[o]
// (kept alive for the whole kernel); `lvlscratch` is reused m-sized workspace.
template <typename FactorIndex>
__device__ __noinline__ void warp_build_config(const gpu_basis_update_mpf_t<FactorIndex>& ft,
                                  warp_solve_config_t<FactorIndex>& cfg,
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
  const gpu_factor_view_t<FactorIndex>* mats[kNumWarpSolveOps] = {
    &ft.L0_transpose_, &ft.U0_transpose_, &ft.L0_, &ft.U0_};
  for (int o = 0; o < kNumWarpSolveOps; ++o) {
    cfg.cs[o]         = mats[o]->col_start.data();
    cfg.ridx[o]       = mats[o]->i.data();
    cfg.val[o]        = mats[o]->x.data();
    // The flags are NOT stored in cfg: the four wavefront bodies (dts::wavefront_{L,U,LT,UT}_solve)
    // are separate specialized functions that encode their own direction, so building the schedule
    // is the only thing that still needs them. op_* are constexpr FUNCTIONS, hence callable here
    // with a runtime `o` -- no materialized flag array required.
    cfg.nlev[o]  = warp_build_levels(*mats[o], op_diag_first(o), op_ascending(o), lev[o], ls[o],
                                     co[o], lvl_pos_v);
    cfg.lstart[o]     = ls[o].data();
    cfg.corder[o]     = co[o].data();
    cfg.level[o]      = lev[o].data();  // per-column level, retained for the reach-restricted sparse solve
  }
}

// Solve one problem selected by idx. A persistent block can call this repeatedly, but each call
// reads and writes only that problem's independently marshalled buffers in args[idx].
// g_gpu_capacity_exceeded is reset by the HOST before launch (not here) so a late block cannot
// clear a failure flag an earlier block already set.
template <class FT, bool AllowRefactorExit>
__device__ __forceinline__ void gpu_phase2_kernel_body(kernel_args* args, i_t idx)
{
  using factor_index_type = typename FT::factor_index_type;
  using warp_config_type  = typename FT::warp_config_type;
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

  // Per-slot state (see slot_state). The host hands us a zeroed blob sized for slot_state<FT>;
  // construct it in place and wire the read-only factorization views into its ft. This is
  // unconditional -- the blob is freshly zeroed per launch. It used to be guarded by the host's
  // that flag to 1 once the super-steps were removed.
  slot_state<FT>& st = *static_cast<slot_state<FT>*>(a.slot);
  // placement-new, ft wiring and S_.init_empty (which resizes -> mallocs) are all RMW on shared
  // state, so under Direct they belong to lane 0 alone.
  lane0_serial([&] {
    new (&st) slot_state<FT>();  // placement new: host memory is zeroed, NSDMIs must run
    // Wire every buffer into the host-provided slot memory. Nothing allocates: a buffer that
    // outgrows its layout slot is a layout bug and is reported, not silently regrown.
    // Exceeding a span's reserved capacity is reported through g_gpu_capacity_exceeded rather than
    // silently regrowing -- there is no allocator left to regrow with.
    st.attach_all_buffers(a.buf, m, n, a.mpf_update_cap, a.mpf_eta_nnz_cap);
    FT& ft                      = st.ft;
    ft.num_updates_             = a.num_updates;
    ft.refactor_frequency_      = a.refactor_frequency;
    ft.L0_                      = make_factor_view(m, m, a.L0_nz, a.L0_col_start, static_cast<factor_index_type*>(a.L0_i), a.L0_x);
    ft.U0_                      = make_factor_view(m, m, a.U0_nz, a.U0_col_start, static_cast<factor_index_type*>(a.U0_i), a.U0_x);
    ft.L0_transpose_            = make_factor_view(m, m, a.L0t_nz, a.L0t_col_start, static_cast<factor_index_type*>(a.L0t_i), a.L0t_x);
    ft.U0_transpose_            = make_factor_view(m, m, a.U0t_nz, a.U0t_col_start, static_cast<factor_index_type*>(a.U0t_i), a.U0t_x);
    ft.row_permutation_         = gpu_span<i_t>(a.row_perm, m);
    ft.inverse_row_permutation_ = gpu_span<i_t>(a.inv_row_perm, m);
    ft.xi_workspace_            = gpu_span<i_t>(a.xi_workspace, 2 * m);
    ft.x_workspace_             = gpu_span<f_t>(a.x_workspace, m);
    ft.reach_mark_              = gpu_span<unsigned char>(a.reach_mark, m);
    ft.update_cap_  = a.mpf_update_cap;
    ft.eta_nnz_cap_ = a.mpf_eta_nnz_cap;
    ft.reserve_store(m);
    ft.hypersparse_threshold_ = 0.05;
  });   // end lane-0 construction
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
  // executes here, and a stack-local gpu_span would device-malloc PER LANE — 32 private buffers,
  // 31 leaked, and worse: warp_build_config publishes `lvl_*[o].data()` / `sp_*_v.data()` into the
  // config, so every wavefront solve for the rest of the solve would read a lane-private schedule.
  // slot_state already carries byte-identical members for exactly this reason (the setup
  // uses them), so this is a reuse, not a new allocation.

  // Serial O(m*nnz) schedule build that resizes the slot's gpu_spans and writes the config.
  // The trailing warp sync publishes the completed schedule and ft.warp_cfg_ to every lane.
  lane0_serial([&] {
    warp_build_config(
      ft, cfg_shared, st.lvl_lev, st.lvl_ls, st.lvl_co, st.sp_ls_v, st.sp_co_v, st.sp_pos_v,
      st.lvl_pos_v);
    ft.warp_cfg_ = &cfg_shared;
  });

  dual::status_t status =
    gpu_simplified_dual_phase2_device<FT, AllowRefactorExit>(lp,
                                                             settings,
                                                             a.vstatus,
                                                             st,
                                                             a.basic_list,
                                                             a.nonbasic_list,
                                                             sol,
                                                             a.dy_se,
                                                             a.dy_se_empty != 0,
                                                             a.refactor_resume != 0);

  // `status` is warp-uniform (every terminal condition in the driver is), so these are same-value
  // stores from every lane — but keep a single writer.
  lane0_serial([&] {
    *a.out_status         = static_cast<i_t>(status);
    *a.out_objective      = sol.objective;
    *a.out_user_objective = sol.user_objective;
    *a.out_iterations     = sol.iterations;
  });
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
template <class FT = gpu_basis_update_mpf_u16, bool AllowRefactorExit = false>
__global__ void __launch_bounds__(32, 16)
  gpu_phase2_kernel_direct(kernel_args* args, i_t n_problems)
{
  __shared__ i_t problem_idx;
  while (true) {
    if (threadIdx.x == 0) {
      problem_idx = static_cast<i_t>(atomicAdd(&g_next_problem, 1u));
    }
    __syncwarp(0xffffffffu);
    if (problem_idx >= n_problems) { break; }
    gpu_phase2_kernel_body<FT, AllowRefactorExit>(args, problem_idx);
  }
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
struct marshal_storage {
  std::vector<rmm::device_buffer> bufs;
  // The CSR transpose is built here and its host storage must outlive marshal_problem.
  std::vector<cuopt::mathematical_optimization::csr_matrix_t<i_t, f_t>> arow_keep;
  marshal_storage() { bufs.reserve(111); }  // ~34 uploads + 67 slot buffers; avoid vector churn
};

template <typename T>
static T* upload(const std::vector<T>& v, marshal_storage& st)
{
  const size_t bytes = (v.empty() ? 1 : v.size()) * sizeof(T);
  if (v.empty()) { st.bufs.emplace_back(bytes, rmm::cuda_stream_default); }
  else { st.bufs.emplace_back(v.data(), bytes, rmm::cuda_stream_default); }
  return static_cast<T*>(st.bufs.back().data());
}

template <typename T>
static T* upload_n(const T* p, size_t count, marshal_storage& st)
{
  const size_t bytes = (count == 0 ? 1 : count) * sizeof(T);
  if (count == 0) { st.bufs.emplace_back(bytes, rmm::cuda_stream_default); }
  else { st.bufs.emplace_back(p, bytes, rmm::cuda_stream_default); }
  return static_cast<T*>(st.bufs.back().data());
}

static constexpr size_t kUint16FactorRowCapacity =
  static_cast<size_t>(std::numeric_limits<std::uint16_t>::max()) + 1;

static bool factors_fit_uint16(i_t rows)
{
  return rows >= 0 && static_cast<size_t>(rows) <= kUint16FactorRowCapacity;
}

template <typename FactorIndex>
static FactorIndex* upload_factor_indices(
  const i_t* idx, size_t count, i_t rows, marshal_storage& st)
{
  std::vector<FactorIndex> converted(count);
  for (size_t k = 0; k < count; ++k) {
    if (idx[k] < 0 || idx[k] >= rows) {
      throw std::runtime_error("gpu_phase2: factor row index is outside [0, m)");
    }
    if (static_cast<size_t>(idx[k]) >
        static_cast<size_t>(std::numeric_limits<FactorIndex>::max())) {
      throw std::runtime_error("gpu_phase2: selected factor-index type cannot represent row index");
    }
    converted[k] = static_cast<FactorIndex>(idx[k]);
  }
  return upload_n(converted.data(), converted.size(), st);
}

// Uninitialised: every per-slot buffer is resize/assign-filled before it is read, and zeroing all
// 66 of them per problem would be a multi-GB memset across a large batch.
template <typename T>
static T* alloc_uninit(size_t count, marshal_storage& st)
{
  const size_t bytes = (count == 0 ? 1 : count) * sizeof(T);
  st.bufs.emplace_back(bytes, rmm::cuda_stream_default);
  return static_cast<T*>(st.bufs.back().data());
}

template <typename T>
static T* alloc_zeroed(size_t count, marshal_storage& st)
{
  const size_t bytes = (count == 0 ? 1 : count) * sizeof(T);
  st.bufs.emplace_back(bytes, rmm::cuda_stream_default);
  void* d = st.bufs.back().data();
  RAFT_CUDA_TRY(cudaMemsetAsync(d, 0, bytes, rmm::cuda_stream_default.value()));
  return static_cast<T*>(d);
}



static void upload_solve_class_config()
{
  const int on = 1;
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_elide_noop, &on, sizeof(int)));
}

// Device heap (in-kernel gpu_span malloc arena) + stack are per-CONTEXT limits and the heap
// can ONLY be set before the first malloc-using kernel launch, so set them exactly once per
// process. First caller wins: the batch wrapper (the only/first GPU kernel in the throughput
// benchmark) sizes the heap for N concurrent S_ stacks; the single-problem wrapper uses a
// generous default. heap_bytes is the requested malloc-arena size (NOT counting the static
// per-problem buffers, which are separate cudaMallocs).
// MEASURED LIMITS OF THE IN-KERNEL malloc HEAP (2026-08-14, probe.cu, confirmed on an RTX 3500 Ada
// with 12 GB and on a B200 with 178 GB):
//   * cudaDeviceGetLimit saturates at ~16.47 GB however much you ask for. It is the SAME number on
//     both cards, so it is a fixed driver cap, not a fraction of device memory.
//   * The arena is reserved LAZILY, at the first malloc-using launch, not by cudaDeviceSetLimit.
//   * If the granted size does not fit in free VRAM at that moment, the reservation fails and the
//     usable arena is ZERO. It is a cliff, not a gradual degradation: a 12 GB request on a card with
//     11.86 GB free gave 0 bytes usable, while an 8 GB request gave 6.56 GB.
//   * Usable is about 0.82 x granted. The rest is allocator overhead.
// So the practical ceiling is ~13.5 GB usable on ANY device. Requesting more is not merely wasteful,
// it is how we crashed: 86.75 GB requested -> capped -> in-kernel malloc returns nullptr ->
// gpu_span::grow_to returns no buffer -> the next write raises cudaErrorIllegalAddress.
// Measured by bisection with a standalone probe kernel: granted tracks the request exactly up
// to 16.40 GB and saturates at 16.47 GB from 16.50 GB upward. Identical on a 12 GB card and a 178 GB
// B200. We clamp to a round 16 GB, safely inside the saturation point.
static constexpr size_t kDeviceHeapDriverCap = static_cast<size_t>(16) << 30;
static constexpr double kDeviceHeapUsableFrac = 0.82;                          // measured overhead
// The caller's heap estimate is deliberately pessimistic. Measured on the 100 v6 nodes with
// -DCUOPT_ALLOC_STATS=1: real demand 1.408 MB/problem against an estimate of 3.874 MB, i.e. 0.363.
// Rounded UP to 0.45 so the guard errs toward letting a run proceed; a run that then exhausts the
// arena still reports it, whereas a wrongly refused run wastes a whole benchmark slot.
static constexpr double kHeapEstimateToActual = 0.45;

static bool g_device_limits_set = false;
static size_t g_device_heap_granted = 0;
static size_t g_device_heap_requested = 0;
// Returns the bytes the driver GRANTED, which is what the kernel can actually draw on.
static size_t ensure_device_limits(size_t heap_bytes)
{
  if (g_device_limits_set) return g_device_heap_granted;
  g_device_limits_set = true;

  if (heap_bytes < (static_cast<size_t>(256) << 20)) { heap_bytes = static_cast<size_t>(256) << 20; }

  // CLAMP. Asking for more than the driver grants, or more than fits, is never useful and is exactly
  // how the arena ends up unusable. Keep a margin below free VRAM for the per-problem cudaMallocs
  // that follow this call.
  const double GBd = 1.0 / (1024.0 * 1024.0 * 1024.0);
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
    const size_t fit = static_cast<size_t>(0.80 * static_cast<double>(free_b));
    if (heap_bytes > fit) {
      std::fprintf(stderr, "[gpu_phase2] clamping malloc arena %.2f -> %.2f GB (80%% of %.2f GB free)\n",
                   heap_bytes * GBd, fit * GBd, free_b * GBd);
      heap_bytes = fit;
    }
  }
  if (heap_bytes > kDeviceHeapDriverCap) {
    std::fprintf(stderr, "[gpu_phase2] clamping malloc arena %.2f -> %.2f GB (driver cap)\n",
                 heap_bytes * GBd, kDeviceHeapDriverCap * GBd);
    heap_bytes = kDeviceHeapDriverCap;
  }
  // Report the arena. A malloc failure inside the kernel is hard to attribute after the fact, so the
  // size that was actually requested must be in the log of every run.
  std::fprintf(stderr, "[gpu_phase2] device malloc arena = %.2f GB (must match the pre-flight heap)\n",
               static_cast<double>(heap_bytes) / (1024.0 * 1024.0 * 1024.0));
  cudaError_t hp = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_bytes);
  if (hp != cudaSuccess) {
    cudaGetLastError();
    std::fprintf(stderr, "[gpu_phase2] warning: could not set device heap size: %s\n",
                 cudaGetErrorString(hp));
  }
  // READ THE LIMIT BACK. cudaDeviceSetLimit reports success even when the driver grants less than
  // was asked for, and nothing here ever checked. A granted value below the request is invisible
  // until in-kernel malloc starts returning nullptr, which then looks like a solver bug.
  {
    size_t granted = 0;
    if (cudaDeviceGetLimit(&granted, cudaLimitMallocHeapSize) == cudaSuccess) {
      g_device_heap_granted   = granted;
      g_device_heap_requested = heap_bytes;
      std::fprintf(stderr, "[gpu_phase2] malloc arena: requested %.2f GB, granted %.2f GB, "
                           "usable ~%.2f GB\n",
                   heap_bytes * GBd, granted * GBd, granted * kDeviceHeapUsableFrac * GBd);
    } else {
      cudaGetLastError();
      g_device_heap_granted   = heap_bytes;
      g_device_heap_requested = heap_bytes;
    }
  }
  // The deep (non-recursive) call chain wants more than the ~1 KB default stack.
  cudaError_t stk = cudaDeviceSetLimit(cudaLimitStackSize, static_cast<size_t>(32) << 10);
  if (stk != cudaSuccess) {
    cudaGetLastError();
    std::fprintf(stderr, "[gpu_phase2] warning: could not raise device stack size: %s\n",
                 cudaGetErrorString(stk));
  }
  return g_device_heap_granted;
}


// Reject a batch whose in-kernel heap demand cannot fit the arena the driver will grant.
// WITHOUT this the kernel launches, malloc returns nullptr thousands of times, and the run dies on
// cudaErrorIllegalAddress -- which looks like a solver bug and costs a whole benchmark slot.
// Returns true when the caller must abort. `demand` is the caller's own heap estimate.
static bool heap_demand_rejected(size_t demand, size_t granted, size_t requested,
                                 size_t n_problems, gpu_batch_timing& timing)
{
  const double GBd    = 1.0 / (1024.0 * 1024.0 * 1024.0);
  const size_t usable = static_cast<size_t>(static_cast<double>(granted) * kDeviceHeapUsableFrac);
  // Compare PREDICTED ACTUAL demand, not the raw estimate. Using the estimate refused batches that
  // run perfectly well, because the estimate carries kHeapSafety plus terms that are never allocated.
  const size_t predicted = static_cast<size_t>(static_cast<double>(demand) * kHeapEstimateToActual);
  if (predicted <= usable) { return false; }
  // The estimate is deliberately pessimistic (kHeapSafety, plus terms that may not be allocated at
  // all), so "estimate > usable" alone is NOT a reason to refuse: 100 problems estimate 0.58 GB,
  // get a 0.47 GB usable arena, and run perfectly well. Refuse only when the DRIVER CAPPED us --
  // then the arena cannot be enlarged by any means and the run would fail for a knowable reason.
  if (granted >= requested) {
    std::fprintf(stderr,
                 "[gpu_phase2] note: heap estimate %.2f GB is above the usable arena %.2f GB, but "
                 "the driver granted everything asked for. Proceeding; the estimate carries a "
                 "safety factor.\n", demand * GBd, usable * GBd);
    return false;
  }
  const size_t per      = n_problems ? (demand / n_problems) : demand;
  const size_t per_pred = n_problems ? (predicted / n_problems) : predicted;
  const size_t max_n    = per_pred ? (usable / per_pred) : 0;
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "predicted in-kernel heap demand %.2f GB exceeds the usable arena %.2f GB (driver "
                "granted %.2f GB; the heap is capped at ~%.2f GB on EVERY device, so no setting "
                "raises it). %zu problems, estimate %.2f MB each, predicted actual %.2f MB each. "
                "At most ~%zu problems fit. Lower the problem count.",
                predicted * GBd, usable * GBd, granted * GBd, kDeviceHeapDriverCap * GBd,
                n_problems, static_cast<double>(per) / (1024.0 * 1024.0),
                static_cast<double>(per_pred) / (1024.0 * 1024.0), max_n);
  timing.rejected = true;
  timing.message  = buf;
  std::fprintf(stderr, "[gpu_phase2] batch REJECTED: %s\n", buf);
  return true;
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

// Nothing to free: upload_ro's buffers are rmm::device_buffers owned by the caller's
// marshal_storage, returned to the pool when that is destroyed.
static void free_ro(const ro_upload&) {}

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

template <typename FactorIndex>
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
                                   bool refactor_resume = false)
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
  a.lower       = upload(lp.lower, st);
  a.upper       = upload(lp.upper, st);
  a.L0_col_start = upload(L0.col_start, st); a.L0_nz = L0.nnz();
  a.L0_i = upload_factor_indices<FactorIndex>(L0.i.data(), L0.nnz(), m, st);
  a.L0_x = upload_n(L0.x.data(), L0.nnz(), st);
  a.U0_col_start = upload(U0.col_start, st); a.U0_nz = U0.nnz();
  a.U0_i = upload_factor_indices<FactorIndex>(U0.i.data(), U0.nnz(), m, st);
  a.U0_x = upload_n(U0.x.data(), U0.nnz(), st);
  a.L0t_col_start = upload(L0t.col_start, st); a.L0t_nz = L0t.nnz();
  a.L0t_i = upload_factor_indices<FactorIndex>(L0t.i.data(), L0t.nnz(), m, st);
  a.L0t_x = upload_n(L0t.x.data(), L0t.nnz(), st);
  a.U0t_col_start = upload(U0t.col_start, st); a.U0t_nz = U0t.nnz();
  a.U0t_i = upload_factor_indices<FactorIndex>(U0t.i.data(), U0t.nnz(), m, st);
  a.U0t_x = upload_n(U0t.x.data(), U0t.nnz(), st);
  a.row_perm     = upload(ft.row_permutation(), st);
  a.inv_row_perm = upload(ft.inverse_row_permutation(), st);
  a.xi_workspace = alloc_zeroed<int>(2 * m, st);
  a.x_workspace  = alloc_zeroed<double>(m, st);
  a.reach_mark   = alloc_zeroed<unsigned char>(m, st);
  a.num_updates  = 0;
  a.refactor_frequency = settings.refactor_frequency;
  a.mpf_update_cap     = mpf_update_cap();
  a.mpf_eta_nnz_cap    = mpf_eta_nnz_cap();

  a.refactor_resume = refactor_resume ? 1 : 0;
  a.vstatus       = upload(vstatus, st);
  a.basic_list    = upload(basic_list, st);
  a.nonbasic_list = upload(nonbasic_list, st);
  a.x             = upload(sol.x, st);
  a.y             = upload(sol.y, st);
  a.z             = upload(sol.z, st);
  if (delta_y_steepest_edge.empty()) {
    a.dy_se       = alloc_zeroed<double>(n, st);
    a.dy_se_empty = 1;
  } else {
    a.dy_se       = upload(delta_y_steepest_edge, st);
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

  a.out_status         = alloc_zeroed<int>(1, st);
  a.out_objective      = alloc_zeroed<double>(1, st);
  a.out_user_objective = alloc_zeroed<double>(1, st);
  a.out_iterations     = alloc_zeroed<int>(1, st);

  // Persistent per-slot solve state. Allocated ZEROED (the kernel's placement-new relies on that
  // here.
  a.slot = alloc_zeroed<char>(sizeof(slot_state<gpu_basis_update_mpf_t<FactorIndex>>), st);

  // The slot's flat buffer: every gpu_span in the kernel is a slice of this, wired at a fixed
  // offset by attach_all_buffers. Deliberately NOT zeroed -- every consumer resize/assign-fills
  // before reading, and zeroing would cost a multi-GB memset across a large batch.
  // One allocation per buffer, from the RMM pool, like every other array uploaded above.
  {
    const size_t mn   = (static_cast<size_t>(m) > static_cast<size_t>(n)) ? m : n;
    const size_t m1   = static_cast<size_t>(m) + 1;
    const size_t ucap = static_cast<size_t>(mpf_update_cap());
    const size_t eta  = static_cast<size_t>(mpf_eta_nnz_cap());
    slot_buffers& b   = a.buf;
    b.xB_workspace = alloc_uninit<f_t>(mn, st);
    b.delta_y = alloc_uninit<f_t>(mn, st);
    b.delta_z = alloc_uninit<f_t>(mn, st);
    b.delta_x = alloc_uninit<f_t>(mn, st);
    b.v = alloc_uninit<f_t>(mn, st);
    b.squared_infeas = alloc_uninit<f_t>(mn, st);
    b.setup_rhs = alloc_uninit<f_t>(mn, st);
    b.se_coeff = alloc_uninit<f_t>(mn, st);
    b.basic_mark = alloc_uninit<i_t>(mn, st);
    b.delta_z_mark = alloc_uninit<i_t>(mn, st);
    b.delta_z_indices = alloc_uninit<i_t>(mn, st);
    b.nonbasic_pos = alloc_uninit<i_t>(mn, st);
    b.infeas_idx = alloc_uninit<i_t>(mn, st);
    b.se_row_degree = alloc_uninit<i_t>(mn, st);
    b.se_mapping = alloc_uninit<i_t>(mn, st);
    b.ft_dense_scratch = alloc_uninit<f_t>(mn, st);
    b.ft_dense_scratch2 = alloc_uninit<f_t>(mn, st);
    b.ft_sort_bucket = alloc_uninit<f_t>(mn, st);
    b.ft_sort_x = alloc_uninit<f_t>(mn, st);
    b.ft_sort_i = alloc_uninit<i_t>(mn, st);
    b.ft_sort_perm = alloc_uninit<i_t>(mn, st);
    b.lvl_lev0 = alloc_uninit<i_t>(m1, st);
    b.lvl_ls0 = alloc_uninit<i_t>(m1, st);
    b.lvl_co0 = alloc_uninit<i_t>(m1, st);
    b.lvl_lev1 = alloc_uninit<i_t>(m1, st);
    b.lvl_ls1 = alloc_uninit<i_t>(m1, st);
    b.lvl_co1 = alloc_uninit<i_t>(m1, st);
    b.lvl_lev2 = alloc_uninit<i_t>(m1, st);
    b.lvl_ls2 = alloc_uninit<i_t>(m1, st);
    b.lvl_co2 = alloc_uninit<i_t>(m1, st);
    b.lvl_lev3 = alloc_uninit<i_t>(m1, st);
    b.lvl_ls3 = alloc_uninit<i_t>(m1, st);
    b.lvl_co3 = alloc_uninit<i_t>(m1, st);
    b.sp_ls_v = alloc_uninit<i_t>(m1, st);
    b.sp_co_v = alloc_uninit<i_t>(m1, st);
    b.sp_pos_v = alloc_uninit<i_t>(m1, st);
    b.lvl_pos_v = alloc_uninit<i_t>(m1, st);
    b.ft_recover_rhs = alloc_uninit<f_t>(m, st);
    b.ft_recover_sol = alloc_uninit<f_t>(m, st);
    b.S_col_start = alloc_uninit<i_t>(2 * ucap + 1, st);
    b.S_i = alloc_uninit<i_t>(eta, st);
    b.S_x = alloc_uninit<f_t>(eta, st);
    b.mu_values = alloc_uninit<f_t>(ucap, st);
    b.delta_y_sparse_i = alloc_uninit<i_t>(mn, st); b.delta_y_sparse_x = alloc_uninit<f_t>(mn, st);
    b.UTsol_sparse_i = alloc_uninit<i_t>(mn, st); b.UTsol_sparse_x = alloc_uninit<f_t>(mn, st);
    b.utilde_sparse_i = alloc_uninit<i_t>(mn, st); b.utilde_sparse_x = alloc_uninit<f_t>(mn, st);
    b.scaled_dxB_sparse_i = alloc_uninit<i_t>(mn, st); b.scaled_dxB_sparse_x = alloc_uninit<f_t>(mn, st);
    b.rhs_sparse_i = alloc_uninit<i_t>(mn, st); b.rhs_sparse_x = alloc_uninit<f_t>(mn, st);
    b.v_sparse_i = alloc_uninit<i_t>(mn, st); b.v_sparse_x = alloc_uninit<f_t>(mn, st);
    b.se_dy_i = alloc_uninit<i_t>(mn, st); b.se_dy_x = alloc_uninit<f_t>(mn, st);
    b.se_UTsol_i = alloc_uninit<i_t>(mn, st); b.se_UTsol_x = alloc_uninit<f_t>(mn, st);
    b.se_ei_i = alloc_uninit<i_t>(mn, st); b.se_ei_x = alloc_uninit<f_t>(mn, st);
    b.ft_sparse_scratch_i = alloc_uninit<i_t>(mn, st); b.ft_sparse_scratch_x = alloc_uninit<f_t>(mn, st);
    b.ft_ei_scratch_i = alloc_uninit<i_t>(mn, st); b.ft_ei_scratch_x = alloc_uninit<f_t>(mn, st);
  }
  return a;
}

// Exact device bytes marshal_problem cudaMallocs for one problem's STATIC buffers (read-only
// factors/matrix + mutable state), EXCLUDING the in-kernel gpu_span heap (S_ growth). Used by
// the batch pre-flight VRAM estimate; mirrors the upload/alloc sizes 1:1 (empty vectors still
// take 1 element, matching upload/alloc_zeroed). Slightly conservative on the small state vectors.
template <typename FactorIndex>
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
  auto bf = [](size_t c) { return (c == 0 ? 1 : c) * sizeof(FactorIndex); };
  size_t b = 0;
  // A (col_start, i, x) + rhs/lower/upper/objective
  b += bi(lp.A.col_start.size()) + bi(lp.A.nnz()) + bd(lp.A.nnz());
  b += bd(m) + 3 * bd(n);
  // Arow (CSR): row_start(m+1) + j(nnz) + x(nnz). Always uploaded now, so it is STATIC VRAM and no
  // longer part of the in-kernel heap demand.
  b += bi(m + 1) + bi(lp.A.nnz()) + bd(lp.A.nnz());
  // Four factors, each col_start + i + x
  b += bi(L0.col_start.size())  + bf(L0.nnz())  + bd(L0.nnz());
  b += bi(U0.col_start.size())  + bf(U0.nnz())  + bd(U0.nnz());
  b += bi(L0t.col_start.size()) + bf(L0t.nnz()) + bd(L0t.nnz());
  b += bi(U0t.col_start.size()) + bf(U0t.nnz()) + bd(U0t.nnz());
  // perms + workspaces
  b += bi(m) + bi(m) + bi(2 * m) + bd(m) + m * sizeof(unsigned char);
  // state: vstatus(n) + basic(m) + nonbasic(<=n) + x(n)+y(m)+z(n) + dy_se(n) + 4 out scalars
  b += n * sizeof(variable_status_t) + bi(m) + bi(n) + bd(n) + bd(m) + bd(n) + bd(n);
  b += 2 * bi(1) + 2 * bd(1);
  // persistent per-slot solve state (slot_state<FT> header only -- its gpu_span payloads live in
  // the arena below, or on the in-kernel heap for the few buffers left there)
  b += sizeof(slot_state<gpu_basis_update_mpf_t<FactorIndex>>);
  // The per-slot buffers. Static VRAM, so it must be in the pre-flight or a large batch passes the
  // check and then fails to allocate.
  b += slot_buffer_bytes(m, n, static_cast<size_t>(mpf_update_cap()),
                         static_cast<size_t>(mpf_eta_nnz_cap()));
  return b;
}


}  // namespace gpu_phase2_detail

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
  bool resume)
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
    // batch path alike (4096 works on a 12 GB card where 0.7*free does not).
    ensure_device_limits(heap);
  }

  // Owns every uploaded buffer for this solve; destroyed at scope exit, returning them to the pool.
  marshal_storage store;
  const bool use_uint16_factors = factors_fit_uint16(m);
  kernel_args a                 = use_uint16_factors ? marshal_problem<std::uint16_t>(lp,
                                                                      settings,
                                                                      vstatus,
                                                                      ft,
                                                                      basic_list,
                                                                      nonbasic_list,
                                                                      sol,
                                                                      delta_y_steepest_edge,
                                                                      nullptr,
                                                                      store,
                                                                      resume)
                                                     : marshal_problem<i_t>(lp,
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

  // Reset the shared alloc-failed flag from the host (the kernel no longer does it).
  const int zero_flag = 0;
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_gpu_capacity_exceeded, &zero_flag, sizeof(int)));
  const unsigned int zero_counter = 0;
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_next_problem, &zero_counter, sizeof(unsigned int)));

  // The kernel takes a device pointer to an array of kernel_args (one per problem);
  // the single-problem path is just N == 1.
  kernel_args* d_args = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_args, sizeof(kernel_args)));
  RAFT_CUDA_TRY(cudaMemcpy(d_args, &a, sizeof(kernel_args), cudaMemcpyHostToDevice));

  upload_solve_class_config();
  if (use_uint16_factors) {
    gpu_phase2_kernel_direct<gpu_basis_update_mpf_u16, true><<<1, 32>>>(d_args, i_t{1});
  } else {
    gpu_phase2_kernel_direct<gpu_basis_update_mpf_i32, true><<<1, 32>>>(d_args, i_t{1});
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
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
  // The FINAL basis and the maintained steepest-edge norms. Both are required to RESUME after a
  // host refactorization: the next leg has to continue from the basis the device stopped on and
  // from the norms it had, or it restarts pricing and takes a different path. y/z above are
  // maintained incrementally by the driver for the same reason.
  RAFT_CUDA_TRY(
    cudaMemcpy(vstatus.data(), a.vstatus, n * sizeof(variable_status_t), cudaMemcpyDeviceToHost));
  delta_y_steepest_edge.resize(static_cast<size_t>(n));
  RAFT_CUDA_TRY(
    cudaMemcpy(delta_y_steepest_edge.data(), a.dy_se, n * sizeof(double), cudaMemcpyDeviceToHost));

  sol.objective      = h_obj;
  sol.user_objective = h_uobj;
  sol.iterations     = h_iters;

  int alloc_failed = 0;
  RAFT_CUDA_TRY(cudaMemcpyFromSymbol(&alloc_failed, g_gpu_capacity_exceeded, sizeof(int)));
  if (alloc_failed) {
    std::fprintf(stderr, "[gpu_phase2] device allocation failed during solve.\n");
  }

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

  // A launch has one compile-time factor-index type. If any node exceeds uint16 capacity,
  // use the int32 specialization for the entire batch.
  bool use_uint16_factors = true;
  // Every ft must be a fresh factorization (pure MPF stacking port, no on-device refactor).
  for (size_t b = 0; b < N; ++b) {
    use_uint16_factors =
      use_uint16_factors && factors_fit_uint16(items[b].lp->num_rows);
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
    static_bytes += use_uint16_factors
                      ? problem_static_bytes<std::uint16_t>(*items[b].lp, *items[b].ft)
                      : problem_static_bytes<int>(*items[b].lp, *items[b].ft);
  }
  const size_t args_bytes = N * sizeof(kernel_args);
  const size_t need = static_bytes + heap_estimate_bytes + args_bytes;
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
                  "batch needs ~%.2f GB VRAM (static %.2f + heap %.2f + args %.2f), only %.2f GB "
                  "available (%.2f free + %.2f current RMM pool; usable budget %.2f GB) for %zu "
                  "problems — reduce --replicas or use a larger GPU",
                  need * GB, static_bytes * GB, heap_estimate_bytes * GB, args_bytes * GB,
                  available_b * GB, free_b * GB, pool_b * GB, budget * GB, N);
    timing.rejected = true;
    timing.message  = buf;
    std::fprintf(stderr, "[gpu_phase2] batch REJECTED: %s\n", buf);
    return;
  }

  // Size the malloc heap for all N concurrent S_ stacks (first GPU kernel this process).
  {
    const size_t granted = ensure_device_limits(heap_estimate_bytes);
    if (heap_demand_rejected(heap_estimate_bytes, granted, g_device_heap_requested, N, timing)) { return; }
  }

  // ---- Marshal every problem + upload (host-timed) ----
  // Deduplicate the immutable A CSC, CSR Arow, rhs, and objective across nodes with the same
  // original matrix. Basis-dependent factors, row permutations, and bounds remain per-node.
  const double up_t0 = tic();
  std::vector<kernel_args> h_args(N);
  std::unordered_map<const void*, ro_upload> ro_cache;
  // One storage per problem, plus one for the deduped read-only uploads. These own every
  // device_buffer for the batch and MUST outlive the kernel launch below.
  std::vector<marshal_storage> store(N);
  marshal_storage ro_store;

  for (size_t b = 0; b < N; ++b) {
    const auto& it = items[b];
    // Key on A's storage, not on the lp object. Every node of a branch-and-bound tree shares the
    // SAME constraint matrix while carrying its own bounds and LU, so each node is a distinct
    // lp_problem_t -- keying on `it.lp` would miss the sharing that actually exists.
    auto res = ro_cache.try_emplace(static_cast<const void*>(it.lp->A.col_start.data()));
    if (res.second) { res.first->second = upload_ro(*it.lp, ro_store); }
    const ro_upload* ro = &res.first->second;
    h_args[b] = use_uint16_factors
                  ? marshal_problem<std::uint16_t>(
                      *it.lp, *it.settings, *it.vstatus, *it.ft, *it.basic_list,
                      *it.nonbasic_list, *it.sol, *it.dy_se, ro, store[b])
                  : marshal_problem<i_t>(
                      *it.lp, *it.settings, *it.vstatus, *it.ft, *it.basic_list,
                      *it.nonbasic_list, *it.sol, *it.dy_se, ro, store[b]);
  }

  kernel_args* d_args = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_args, N * sizeof(kernel_args)));
  RAFT_CUDA_TRY(cudaMemcpy(d_args, h_args.data(), N * sizeof(kernel_args), cudaMemcpyHostToDevice));
  const int zero_flag = 0;
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_gpu_capacity_exceeded, &zero_flag, sizeof(int)));
  const unsigned int zero_counter = 0;
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_next_problem, &zero_counter, sizeof(unsigned int)));
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  timing.upload_s = toc(up_t0);

  // ---- Launch a resident work-stealing grid, timed with CUDA events (device compute only) ----
  upload_solve_class_config();
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  int sm_count = 1;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 1;
  if (use_uint16_factors) {
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, gpu_phase2_kernel_direct<gpu_basis_update_mpf_u16>, 32, 0));
  } else {
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, gpu_phase2_kernel_direct<gpu_basis_update_mpf_i32>, 32, 0));
  }
  unsigned int nb = static_cast<unsigned int>(blocks_per_sm) * static_cast<unsigned int>(sm_count);
  if (nb == 0 || nb > N) { nb = static_cast<unsigned int>(N); }
  cudaEvent_t ev_start, ev_stop;
  RAFT_CUDA_TRY(cudaEventCreate(&ev_start));
  RAFT_CUDA_TRY(cudaEventCreate(&ev_stop));
  RAFT_CUDA_TRY(cudaEventRecord(ev_start));
  {
    // The register/occupancy target is a BUILD-time knob on the Direct kernel
    // gpu_phase2_kernel_direct.
    if (use_uint16_factors) {
      gpu_phase2_kernel_direct<gpu_basis_update_mpf_u16><<<nb, 32>>>(
        d_args, static_cast<i_t>(N));
    } else {
      gpu_phase2_kernel_direct<gpu_basis_update_mpf_i32><<<nb, 32>>>(d_args, static_cast<i_t>(N));
    }
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
  RAFT_CUDA_TRY(cudaMemcpyFromSymbol(&alloc_failed, g_gpu_capacity_exceeded, sizeof(int)));
  if (alloc_failed) {
    timing.message = "a device span exceeded the capacity the slot layout reserved for it — the "
                     "layout is wrong, and results are unreliable";
    std::fprintf(stderr, "[gpu_phase2] batch WARNING: %s\n", timing.message.c_str());
  }

  for (const auto& kv : ro_cache) { free_ro(kv.second); }
  cudaFree(d_args);
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

  std::vector<i_t> superbasic_list;
  std::vector<f_t> xB_workspace;
  f_t work_estimate = 0.0;
  i_t total_iters   = 0;
  bool resume       = false;

  while (true) {
    const dual::status_t st = gpu_simplified_dual_phase2_leg<i_t, f_t>(
      lp, settings, vstatus, ft, basic_list, nonbasic_list, sol, delta_y_steepest_edge, resume);

    // ONE solve, however many launches it took: the caller's iteration count must not reset at a
    // refactorization the caller never asked for.
    total_iters += sol.iterations;
    if (st != dual::status_t::REFACTOR_NEEDED) {
      sol.iterations = total_iters;
      return st;
    }
    stalled_legs = (sol.iterations == 0) ? (stalled_legs + 1) : 0;
    if (stalled_legs > kMaxStalledLegs) {
      sol.iterations = total_iters;
      return dual::status_t::NUMERICAL;
    }

    // Rebuild the factorization of the basis the device stopped on. get_basis_from_vstatus writes
    // basic_list BY INDEX and only pushes back on nonbasic_list, so basic_list must be pre-sized to
    // m -- clearing it instead writes out of bounds -- and nonbasic_list must be cleared, or it
    // grows by n - m every leg and every later consumer reads the FIRST basis forever after.
    basic_list.assign(static_cast<size_t>(m), 0);
    nonbasic_list.clear();
    nonbasic_list.reserve(static_cast<size_t>(lp.num_cols - m));
    superbasic_list.clear();
    get_basis_from_vstatus<i_t>(m, vstatus, basic_list, nonbasic_list, superbasic_list);
    // Same policy as the CPU simplified driver: any non-zero return (singular basis, a repair that
    // did not restore rank, a halt or time-limit code) ends the solve as NUMERICAL.
    if (ft.refactor_basis(
          lp.A, settings, lp.lower, lp.upper, t0, basic_list, nonbasic_list, vstatus) != 0) {
      sol.iterations = total_iters;
      return dual::status_t::NUMERICAL;
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
  size_t,
  std::vector<dual::status_t>&,
  gpu_batch_timing&);


}  // namespace cuopt::linear_programming::dual_simplex
