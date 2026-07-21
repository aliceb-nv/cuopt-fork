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

#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/sparse_vector.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <utilities/macros.cuh>

#include <cuda/std/limits>
#include <cuda/std/span>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {
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

// Warp dense-solve mode (host sets via cudaMemcpyToSymbol from CUOPT_GPU_WARP): 0 = plain wavefront
// (bit-exact, one lane per column), 1 = big≥32 (whole-warp cooperation on large columns).
__device__ int g_warp_solve_mode;
static constexpr int kBigThreshold = 32;  // "large column" cutoff for the big path

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
    : data_(o.data_), size_(o.size_), cap_(o.cap_)
  {
    o.data_ = nullptr;
    o.size_ = 0;
    o.cap_  = 0;
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
    if (data_ != nullptr) { free(data_); }
    data_   = o.data_;
    size_   = o.size_;
    cap_    = o.cap_;
    o.data_ = nullptr;
    o.size_ = 0;
    o.cap_  = 0;
    return *this;
  }
  __device__ ~gpu_vector()
  {
    if (data_ != nullptr) { free(data_); }
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

__device__ void lower_triangular_solve(const gpu_csc_view& L, f_t* x, f_t& work_estimate)
{
  dts::lower_solve(L.col_start.data(), L.i.data(), L.x.data(), x, L.n, &work_estimate);
}

__device__ void lower_triangular_transpose_solve(const gpu_csc_view& L, f_t* x, f_t& work_estimate)
{
  dts::lower_transpose_solve(L.col_start.data(), L.i.data(), L.x.data(), x, L.n, &work_estimate);
}

__device__ i_t upper_triangular_solve(const gpu_csc_view& U, f_t* x, f_t& work_estimate)
{
  dts::upper_solve(U.col_start.data(), U.i.data(), U.x.data(), x, U.n, &work_estimate);
  return 0;
}

__device__ i_t upper_triangular_transpose_solve(const gpu_csc_view& U, f_t* x, f_t& work_estimate)
{
  dts::upper_transpose_solve(U.col_start.data(), U.i.data(), U.x.data(), x, U.n, &work_estimate);
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
enum warp_solve_op {
  WARP_L_SOLVE = 0, WARP_U_SOLVE = 1, WARP_LT_SOLVE = 2, WARP_UT_SOLVE = 3,
  WARP_TLOOP     = kNumWarpSolveOps,     // 4: MPF rank-1 T-loop (uses tl_* fields, not cs[op]/nlev[op])
  WARP_DZ        = kNumWarpSolveOps + 1, // 5: reduced-cost SpMv δz = -A_N'·δy (uses dz_* fields)
  WARP_TODENSE   = kNumWarpSolveOps + 2, // 6: sparse→dense scatter (td_* fields)
  WARP_FROMDENSE = kNumWarpSolveOps + 3, // 7: dense→sparse compaction (fd_* fields)
  WARP_RT        = kNumWarpSolveOps + 4, // 8: dual ratio test min-argmin (rt_* fields)
  WARP_DONE      = kNumWarpSolveOps + 5  // 9: sentinel — never indexes a per-op array
};

// Trivially constructible (no member initializers) so it can be a __shared__
// variable. Lane 0 fills every field (warp_build_config + warp_dispatch) before
// any lane reads it, ordered by the first __syncwarp.
struct warp_solve_config {
  // Per-operator (indexed by warp_solve_op); filled once by warp_build_config.
  const i_t* cs[kNumWarpSolveOps];      // gather CSC col_start
  const i_t* ridx[kNumWarpSolveOps];    // gather CSC row indices
  const f_t* val[kNumWarpSolveOps];     // gather CSC values
  const i_t* lstart[kNumWarpSolveOps];  // level offsets [nlev+1]
  const i_t* corder[kNumWarpSolveOps];  // columns by level [m] (sorted large-first when mode==big)
  i_t  nlev[kNumWarpSolveOps];
  bool diag_first[kNumWarpSolveOps];
  bool descending[kNumWarpSolveOps];
  bool incremental[kNumWarpSolveOps];   // L/U-solve subtract incrementally (match scatter); LᵀUᵀ sum
  // mode 1 (big): per-level base word into bits[] + the large-column bitmask.
  const i_t*      lword[kNumWarpSolveOps];
  const unsigned* bits[kNumWarpSolveOps];
  int  mode;                            // 0 plain / 1 big (= g_warp_solve_mode)
  i_t  m;
  // Per-dispatch request: written by lane 0, read by all lanes after __syncwarp.
  f_t* x;
  int  op;
  // T-loop request (op == WARP_TLOOP): the MPF rank-1 replay over S_. Set fresh each
  // dispatch (S_ pointers can move on resize). dot col = 2k+tl_dotoff, axpy col = 2k+tl_axpyoff.
  const i_t* tl_cs;      // S_.col_start
  const i_t* tl_ri;      // S_.i
  const f_t* tl_val;     // S_.x
  const f_t* tl_mu;      // mu_values_
  f_t*       tl_work;    // &work_estimate_ (lane-0 accumulation only)
  i_t        tl_num;     // num_updates_
  i_t        tl_dotoff;  // 0 (u_col) or 1 (v_col)
  i_t        tl_axpyoff; // 0 (u_col) or 1 (v_col)
  bool       tl_asc;     // ascending k (l_solve) vs descending (l_transpose)
  f_t        tl_tol;     // theta zero-tolerance
  // δz SpMv request (op == WARP_DZ): dense CSC gather δz[j] = -A(:,j)'·δy over nonbasic columns.
  const i_t* dz_Acs;      // A.col_start
  const i_t* dz_Ari;      // A.i
  const f_t* dz_Aval;     // A.x
  const f_t* dz_dy;       // delta_y (dense)
  const i_t* dz_basic;    // basic_list
  const i_t* dz_nonbasic; // nonbasic_list
  f_t*       dz_out;      // delta_z (dense output)
  i_t*       dz_indices;  // raw index buffer, filled [0,count) in nonbasic order
  i_t*       dz_mark;     // delta_z_mark
  i_t*       dz_count;    // out: number of nonzeros appended (lane 0)
  f_t*       dz_work;     // out: work_estimate accumulation (lane 0), or nullptr
  i_t        dz_m;
  i_t        dz_n;
  i_t        dz_leaving;
  i_t        dz_dir;
  // sparse→dense scatter (op == WARP_TODENSE): dense[0,td_n)=0 then dense[td_i[k]]=td_x[k].
  const i_t* td_i;
  const f_t* td_x;
  f_t*       td_dense;
  i_t        td_nz;
  i_t        td_n;
  // dense→sparse compaction (op == WARP_FROMDENSE): append (k, fd_in[k]) for nonzero fd_in[k].
  const f_t* fd_in;
  i_t*       fd_i;
  f_t*       fd_x;
  i_t*       fd_count;
  i_t        fd_n;
  // dual ratio test (op == WARP_RT): min eligible ratio over nonbasic cols, then argmax |z| within tol.
  const i_t*                rt_nonbasic;
  const variable_status_t*  rt_vstatus;
  const f_t*                rt_z;
  const f_t*                rt_dz;
  i_t*                      rt_entering;   // out: entering column j (or -1)
  i_t*                      rt_k;          // out: its nonbasic_list index
  f_t*                      rt_step;       // out: step_length (chosen ratio)
  i_t                       rt_m;
  i_t                       rt_n;
  f_t                       rt_pivot_tol;
  f_t                       rt_dual_tol;
  f_t                       rt_zero_tol;
};

// Collective (all-32-lane) dense solve of operator cfg.op on x = cfg.x. Both modes do exactly
// cfg.nlev[op] internal __syncwarps, so the persistent-helper dispatch count is mode-independent.
__device__ void warp_collective_solve(const warp_solve_config& cfg)
{
  const int op = cfg.op;
  if (cfg.mode == 1) {  // big≥32: whole-warp cooperation on large columns
    dts::wavefront_big_solve(cfg.cs[op], cfg.ridx[op], cfg.val[op], cfg.x, cfg.lstart[op],
                             cfg.corder[op], cfg.lword[op], cfg.bits[op], cfg.nlev[op],
                             cfg.diag_first[op], cfg.descending[op], cfg.incremental[op]);
  } else {              // plain wavefront: one lane per column
    dts::wavefront_gather_solve(cfg.cs[op], cfg.ridx[op], cfg.val[op], cfg.x, cfg.lstart[op],
                                cfg.corder[op], cfg.nlev[op], cfg.diag_first[op], cfg.descending[op],
                                cfg.incremental[op]);
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
// broadcast the global min ratio before the tie-break pass). min is associative/commutative → bit-exact.
__device__ __forceinline__ f_t warp_reduce_min(f_t v)
{
  for (int o = 16; o > 0; o >>= 1) { v = fmin(v, __shfl_xor_sync(0xffffffffu, v, o)); }
  return v;
}

// Collective (all-32-lane) MPF rank-1 T-loop replay: for each update k, dot = S(:,dcol)'·x
// (lane-strided partial + all-reduce), theta = dot/mu, then x += -theta·S(:,acol) (lane-strided
// distinct writes). One __syncwarp per update (uniform), so the internal barrier count = tl_num
// on both lane 0 and the helpers. Only the dot reduction is non-bit-exact (relaxed); the axpy
// writes are distinct rows within a column -> race-free and bit-exact. work_estimate is
// accumulated on lane 0 only (mirrors the serial dot_product / add_sparse_column accounting).
__device__ void warp_collective_tloop(const warp_solve_config& cfg)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t num  = cfg.tl_num;
  f_t* x         = cfg.x;
  f_t work       = 0.0;
  for (i_t s = 0; s < num; ++s) {
    const i_t k    = cfg.tl_asc ? s : (num - 1 - s);
    const i_t dcol = 2 * k + cfg.tl_dotoff;
    const i_t acol = 2 * k + cfg.tl_axpyoff;
    const f_t mu   = cfg.tl_mu[k];
    const i_t dc0  = cfg.tl_cs[dcol];
    const i_t dc1  = cfg.tl_cs[dcol + 1];
    f_t partial    = 0.0;
    for (i_t p = dc0 + lane; p < dc1; p += 32) { partial += cfg.tl_val[p] * x[cfg.tl_ri[p]]; }
    const f_t dot   = warp_reduce_add(partial);   // every lane gets the full sum
    const f_t theta = dot / mu;
    if (lane == 0) { work += 3 * (dc1 - dc0); }
    if (fabs(theta) > cfg.tl_tol) {
      const i_t ac0 = cfg.tl_cs[acol];
      const i_t ac1 = cfg.tl_cs[acol + 1];
      for (i_t p = ac0 + lane; p < ac1; p += 32) { x[cfg.tl_ri[p]] += (-theta) * cfg.tl_val[p]; }
      if (lane == 0) { work += 3 * (ac1 - ac0); }
    }
    __syncwarp(0xffffffffu);   // x writes visible before the next update's dot reads them
  }
  if (lane == 0 && cfg.tl_work != nullptr) { *cfg.tl_work += work; }
}

// Collective (all-32-lane) reduced-cost SpMv: δz[j] = -A(:,j)'·δy for each nonbasic column j
// (one lane per column, gather form → distinct outputs, no atomics, per-column dot in a single lane
// → BIT-EXACT vs the serial dense path). Clears basic δz, sets the leaving entry, then computes the
// nonbasic dots and appends the nonzero columns to dz_indices via an ordered ballot/popc compaction
// that preserves the exact nonbasic_list order (so the index list matches the serial push_back order).
// Exactly ONE internal __syncwarp (after the clear, before the leaving overwrite).
__device__ void warp_collective_dz(const warp_solve_config& cfg)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t m = cfg.dz_m, n = cfg.dz_n;
  f_t* dz = cfg.dz_out;
  for (i_t k = lane; k < m; k += 32) { dz[cfg.dz_basic[k]] = 0.0; }  // clear basic entries
  __syncwarp(0xffffffffu);                                           // clears done before leaving set
  if (lane == 0) { dz[cfg.dz_leaving] = static_cast<f_t>(cfg.dz_dir); }
  const i_t num_nb = n - m;
  i_t count      = 0;    // running #appended (identical on all lanes)
  f_t nnz_local  = 0.0;
  for (i_t base = 0; base < num_nb; base += 32) {
    const i_t k = base + lane;
    i_t j = -1; f_t dzj = 0.0; bool active = false;
    if (k < num_nb) {
      j            = cfg.dz_nonbasic[k];
      const i_t c0 = cfg.dz_Acs[j], c1 = cfg.dz_Acs[j + 1];
      f_t dot      = 0.0;
      for (i_t p = c0; p < c1; ++p) { dot += cfg.dz_Aval[p] * cfg.dz_dy[cfg.dz_Ari[p]]; }
      dzj = -dot; dz[j] = dzj;
      nnz_local += static_cast<f_t>(c1 - c0);
      active = (dzj != 0.0);
    }
    const unsigned ball = __ballot_sync(0xffffffffu, active);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (active) { cfg.dz_indices[count + prefix] = j; cfg.dz_mark[j] = 1; }  // distinct offsets/rows
    count += __popc(ball);
  }
  if (lane == 0) { *cfg.dz_count = count; }
  if (cfg.dz_work != nullptr) {
    const f_t nnz_total = warp_reduce_add(nnz_local);
    if (lane == 0) {
      *cfg.dz_work += 2.0 * m + 3.0 * static_cast<f_t>(num_nb) + 3.0 * nnz_total
                    + 2.0 * static_cast<f_t>(count);
    }
  }
}

// Collective sparse→dense scatter: dense[0,n)=0 then dense[i[k]]=x[k]. One internal __syncwarp
// (zero all before scattering, since scatter targets overlap the zeroed range). Distinct sparse
// indices → distinct writes → bit-exact.
__device__ void warp_collective_todense(const warp_solve_config& cfg)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t n = cfg.td_n;
  for (i_t k = lane; k < n; k += 32) { cfg.td_dense[k] = 0.0; }
  __syncwarp(0xffffffffu);
  const i_t nz = cfg.td_nz;
  for (i_t k = lane; k < nz; k += 32) { cfg.td_dense[cfg.td_i[k]] = cfg.td_x[k]; }
}

// Collective dense→sparse compaction: append (k, in[k]) for every nonzero in[k], in increasing k
// (ordered ballot/popc → matches the serial scan order). Distinct offsets → no internal barrier.
__device__ void warp_collective_fromdense(const warp_solve_config& cfg)
{
  const int lane = static_cast<int>(threadIdx.x);
  const i_t n = cfg.fd_n;
  i_t count = 0;
  for (i_t base = 0; base < n; base += 32) {
    const i_t k = base + lane;
    bool active = false; f_t v = 0.0;
    if (k < n) { v = cfg.fd_in[k]; active = (v != 0.0); }
    const unsigned ball = __ballot_sync(0xffffffffu, active);
    const int prefix    = __popc(ball & ((1u << lane) - 1u));
    if (active) { cfg.fd_i[count + prefix] = k; cfg.fd_x[count + prefix] = v; }
    count += __popc(ball);
  }
  if (lane == 0) { *cfg.fd_count = count; }
}

// Collective dual ratio test: min eligible ratio over nonbasic columns, then argmax |z[j]| among the
// candidates within zero_tol of that min (ties → smaller nonbasic index). Two lane-strided passes +
// two warp reductions (no `__syncwarp` — all via registers/shfl). Bit-exact vs the serial greedy scan
// EXCEPT a rare window-drift cascade (needs many ratios within zero_tol of each other; zero_tol is
// tiny) → relaxed. Eligibility mirrors the serial exactly.
__device__ void warp_collective_rt(const warp_solve_config& cfg)
{
  const int lane   = static_cast<int>(threadIdx.x);
  const i_t num_nb = cfg.rt_n - cfg.rt_m;
  const f_t ptol = cfg.rt_pivot_tol, dtol = cfg.rt_dual_tol, ztol = cfg.rt_zero_tol;
  const f_t INF  = dev_inf();

  // Pass 1: global minimum eligible ratio (all lanes get it).
  f_t lmin = INF;
  for (i_t k = lane; k < num_nb; k += 32) {
    const i_t j             = cfg.rt_nonbasic[k];
    const variable_status_t s = cfg.rt_vstatus[j];
    f_t ratio = INF;
    if (s == variable_status_t::NONBASIC_LOWER && cfg.rt_dz[j] < -ptol) {
      ratio = (-dtol - cfg.rt_z[j]) / cfg.rt_dz[j];
    } else if (s == variable_status_t::NONBASIC_UPPER && cfg.rt_dz[j] > ptol) {
      ratio = (dtol - cfg.rt_z[j]) / cfg.rt_dz[j];
    }
    if (ratio < lmin) { lmin = ratio; }
  }
  const f_t gmin = warp_reduce_min(lmin);

  // Pass 2: among ratio < gmin+ztol, pick max |z[j]| (tie → smaller k).
  f_t best_absz = -1.0; i_t best_k = 0x7fffffff; i_t best_j = -1; f_t best_r = INF;
  for (i_t k = lane; k < num_nb; k += 32) {
    const i_t j             = cfg.rt_nonbasic[k];
    const variable_status_t s = cfg.rt_vstatus[j];
    f_t ratio = INF;
    if (s == variable_status_t::NONBASIC_LOWER && cfg.rt_dz[j] < -ptol) {
      ratio = (-dtol - cfg.rt_z[j]) / cfg.rt_dz[j];
    } else if (s == variable_status_t::NONBASIC_UPPER && cfg.rt_dz[j] > ptol) {
      ratio = (dtol - cfg.rt_z[j]) / cfg.rt_dz[j];
    }
    // Candidate = the argmin (always, so entering can't be spuriously -1) plus near-ties within
    // zero_tol. Use the difference (ratio - gmin) not (ratio < gmin + ztol): the latter drops the
    // argmin when gmin is large (gmin+ztol rounds back to gmin) or zero_tol == 0 → false DUAL_UNBOUNDED.
    if (ratio <= gmin || ratio - gmin < ztol) {
      const f_t az = fabs(cfg.rt_z[j]);
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
  if (lane == 0) { *cfg.rt_entering = best_j; *cfg.rt_k = best_j < 0 ? -1 : best_k; *cfg.rt_step = best_r; }
}

// Route a posted request to the right collective body: op < kNumWarpSolveOps is a dense triangular
// solve, else the tl_/dz_/td_/fd_/rt_ collective selected by op. Both lane 0 (warp_dispatch*) and the
// parked helpers (warp_helper_loop) go through here, so their internal __syncwarp counts always match
// (nlev[op] solve, tl_num T-loop, 1 δz, 1 to_dense, 0 from_dense, 0 ratio test).
__device__ void warp_collective(const warp_solve_config& cfg)
{
  if (cfg.op == WARP_TLOOP) { warp_collective_tloop(cfg); }
  else if (cfg.op == WARP_DZ) { warp_collective_dz(cfg); }
  else if (cfg.op == WARP_TODENSE) { warp_collective_todense(cfg); }
  else if (cfg.op == WARP_FROMDENSE) { warp_collective_fromdense(cfg); }
  else if (cfg.op == WARP_RT) { warp_collective_rt(cfg); }
  else { warp_collective_solve(cfg); }
}

// Lanes 1..31 park here for the whole driver, waking to run each op lane 0 posts,
// and exit when lane 0 posts WARP_DONE. The __syncwarp count per op (A + internal + B)
// matches the dispatch on lane 0 exactly.
__device__ void warp_helper_loop(const warp_solve_config* cfg)
{
  while (true) {
    __syncwarp(0xffffffffu);          // A: (x, op) published by lane 0
    if (cfg->op == WARP_DONE) { break; }
    warp_collective(*cfg);            // internal __syncwarps (nlev[op] or tl_num)
    __syncwarp(0xffffffffu);          // B: results committed
  }
}

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

// Big schedule (mode 1): build the level schedule, then within each level sort columns large-first
// (large = gather span >= kBigThreshold) and emit the per-32-column bitmask + per-level base word into
// bits[]/lword[]. `scratch` is m-sized workspace. Returns nlev. Runs on lane 0, once.
__device__ int warp_build_levels_big(const gpu_csc_view& mat, bool df, bool asc,
                                     gpu_vector<i_t>& level, gpu_vector<i_t>& level_start,
                                     gpu_vector<i_t>& col_order, gpu_vector<i_t>& lword,
                                     gpu_vector<unsigned>& bits, gpu_vector<i_t>& scratch)
{
  const int nlev = warp_build_levels(mat, df, asc, level, level_start, col_order);
  const i_t m    = mat.n;
  scratch.resize(m);
  lword.assign(nlev + 1, 0);
  i_t nwords = 0;                                       // per-level word-aligned bitmask
  for (int L = 0; L < nlev; ++L) {
    lword[L] = nwords;
    nwords += (level_start[L + 1] - level_start[L] + 31) / 32;
  }
  lword[nlev] = nwords;
  bits.assign(nwords, 0u);
  auto span = [&](i_t j) -> i_t {
    const i_t c0 = mat.col_start[j], c1 = mat.col_start[j + 1];
    return df ? (c1 - (c0 + 1)) : ((c1 - 1) - c0);
  };
  for (int L = 0; L < nlev; ++L) {
    const i_t base = level_start[L], end = level_start[L + 1];
    i_t w = base;                                        // stable partition: large first, then small
    for (i_t t = base; t < end; ++t) { if (span(col_order[t]) >= kBigThreshold) { scratch[w++] = col_order[t]; } }
    const i_t nlarge_end = w;
    for (i_t t = base; t < end; ++t) { if (span(col_order[t]) <  kBigThreshold) { scratch[w++] = col_order[t]; } }
    for (i_t t = base; t < end; ++t) { col_order[t] = scratch[t]; }
    const i_t wbase = lword[L], sz = end - base;         // leading nlarge_end-base entries are large
    for (i_t c = 0; c * 32 < sz; ++c) {
      unsigned word = 0;
      for (int i = 0; i < 32; ++i) {
        const i_t t = base + c * 32 + i;
        if (t < nlarge_end) { word |= (1u << i); }
      }
      bits[wbase + c] = word;
    }
  }
  return nlev;
}

// ---- sparse triangular solve (reach + iterative DFS) ----------------------
// G.col_start is flipped to mark visited nodes and restored on the way out.
#define GP_FLIP(i)       (-(i) - 2)
#define GP_UNFLIP(i)     (((i) < 0) ? GP_FLIP(i) : (i))
#define GP_MARKED(w, j)  ((w)[j] < 0)
#define GP_MARK(w, j)    { (w)[j] = GP_FLIP((w)[j]); }

// pinv == nullptr means no permutation.
__device__ i_t depth_first_search(
  i_t j, const i_t* pinv, const gpu_csc_view& G, i_t top, i_t* xi, i_t* pstack, f_t& work_estimate)
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
    work_estimate += 3 * (p - psav) + 10;
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
  const gpu_sparse_vector& b, const i_t* pinv, const gpu_csc_view& G, i_t* xi, f_t& work_estimate)
{
  const i_t m   = G.m;
  i_t top       = m;
  const i_t bnz = b.i.size();
  for (i_t p = 0; p < bnz; ++p) {
    if (!GP_MARKED(G.col_start, b.i[p])) {
      top = depth_first_search(b.i[p], pinv, G, top, xi, xi + m, work_estimate);
    }
  }
  work_estimate += 4 * bnz;
  for (i_t p = top; p < m; ++p) {
    GP_MARK(G.col_start, xi[p]);
  }
  work_estimate += 3 * (m - top);
  return top;
}

// lo == true solves L*x=b, lo == false solves U*x=b. x is a dense scratch
// buffer; xi is a 2*m scratch. Output nonzero pattern is xi[top..m-1].
template <bool lo>
__device__ i_t sparse_triangle_solve(
  const gpu_sparse_vector& b, const i_t* pinv, i_t* xi, const gpu_csc_view& G, f_t* x, f_t& work_estimate)
{
  i_t m   = G.m;
  i_t top = reach(b, pinv, G, xi, work_estimate);
  for (i_t p = top; p < m; ++p) {
    x[xi[p]] = 0;
  }
  work_estimate += 2 * (m - top);

  const i_t bnz = b.i.size();
  for (i_t p = 0; p < bnz; ++p) {
    x[b.i[p]] = b.x[p];
  }
  work_estimate += 3 * bnz;

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
    work_estimate += 4 * (end - p) + 7;
    for (; p < end; ++p) {
      x[G.i[p]] -= G.x[p] * x[j];
    }
  }
  return top;
}

// ---------------------------------------------------------------------------
// gpu_basis_update_mpf : the MPF LU factorization object. The initial factors
// L0_/U0_ (+ transposes) and permutations are fixed views uploaded from the
// host; S_/mu_values_ grow in-kernel as rank-1 updates are stacked.
// ---------------------------------------------------------------------------
struct gpu_basis_update_mpf {
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
  mutable f_t work_estimate_{0.0};
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
  // solve; lanes 1..31 are waiting in warp_helper_loop. __syncwarp A publishes
  // (x, op); warp_collective_solve runs nlev internal barriers; __syncwarp B
  // makes the result visible to lane 0 before it reads x.
  __device__ void warp_dispatch(int op, f_t* rhs) const
  {
    warp_cfg_->x  = rhs;
    warp_cfg_->op = op;
    __syncwarp(0xffffffffu);          // A
    warp_collective(*warp_cfg_);
    __syncwarp(0xffffffffu);          // B
  }

  // Post the MPF rank-1 T-loop to the whole warp (op == WARP_TLOOP). One dispatch runs the
  // entire replay collectively (tl_num internal barriers). S_ pointers are re-read each call
  // (they can move on resize). dot col = 2k+dotoff, axpy col = 2k+axpyoff; asc selects
  // l_solve (k up) vs l_transpose (k down).
  __device__ void warp_dispatch_tloop(f_t* rhs, bool asc, i_t dotoff, i_t axpyoff, f_t tol) const
  {
    warp_solve_config* c = warp_cfg_;
    c->x          = rhs;
    c->tl_cs      = S_.col_start.data();
    c->tl_ri      = S_.i.data();
    c->tl_val     = S_.x.data();
    c->tl_mu      = mu_values_.data();
    c->tl_work    = &work_estimate_;
    c->tl_num     = num_updates_;
    c->tl_dotoff  = dotoff;
    c->tl_axpyoff = axpyoff;
    c->tl_asc     = asc;
    c->tl_tol     = tol;
    c->op         = WARP_TLOOP;
    __syncwarp(0xffffffffu);          // A
    warp_collective(*c);
    __syncwarp(0xffffffffu);          // B
  }

  // Post the reduced-cost SpMv δz = -A_N'·δy to the whole warp (op == WARP_DZ). One dispatch computes
  // every nonbasic column's δz + the nonzero index list; the count comes back through *dz_count.
  __device__ void warp_dispatch_dz(const i_t* Acs, const i_t* Ari, const f_t* Aval, const f_t* dy,
                                   const i_t* basic, const i_t* nonbasic, f_t* dz, i_t* dz_idx,
                                   i_t* dz_mark, i_t* dz_count, f_t* dz_work,
                                   i_t m, i_t n, i_t leaving, i_t dir) const
  {
    warp_solve_config* c = warp_cfg_;
    c->dz_Acs      = Acs;
    c->dz_Ari      = Ari;
    c->dz_Aval     = Aval;
    c->dz_dy       = dy;
    c->dz_basic    = basic;
    c->dz_nonbasic = nonbasic;
    c->dz_out      = dz;
    c->dz_indices  = dz_idx;
    c->dz_mark     = dz_mark;
    c->dz_count    = dz_count;
    c->dz_work     = dz_work;
    c->dz_m        = m;
    c->dz_n        = n;
    c->dz_leaving  = leaving;
    c->dz_dir      = dir;
    c->op          = WARP_DZ;
    __syncwarp(0xffffffffu);          // A
    warp_collective(*c);
    __syncwarp(0xffffffffu);          // B
  }

  // Warp-parallel gpu_sparse_vector::to_dense — lane 0 sizes the dense buffer (no zero-fill), the warp
  // zeros [0,n) + scatters the nnz. Bit-exact (distinct sparse indices). Replaces the O(n) serial tail
  // before each dense solve.
  __device__ void warp_to_dense(const gpu_sparse_vector& sv, gpu_vector<f_t>& dense) const
  {
    const i_t n = sv.n;
    dense.reserve(n);
    dense.set_size(n);                 // capacity n, contents set by the collective (zero + scatter)
    warp_solve_config* c = warp_cfg_;
    c->td_i     = sv.i.data();
    c->td_x     = sv.x.data();
    c->td_dense = dense.data();
    c->td_nz    = static_cast<i_t>(sv.i.size());
    c->td_n     = n;
    c->op       = WARP_TODENSE;
    __syncwarp(0xffffffffu);          // A
    warp_collective(*c);
    __syncwarp(0xffffffffu);          // B
  }

  // Warp-parallel gpu_sparse_vector::from_dense — lane 0 reserves the sparse buffers, the warp compacts
  // the nonzeros of `dense` in increasing index order, lane 0 sets the final size. Bit-exact.
  __device__ void warp_from_dense(const gpu_vector<f_t>& dense, gpu_sparse_vector& sv) const
  {
    const i_t n = static_cast<i_t>(dense.size());
    sv.n = n;
    sv.i.reserve(n);
    sv.x.reserve(n);
    i_t count            = 0;
    warp_solve_config* c = warp_cfg_;
    c->fd_in    = dense.data();
    c->fd_i     = sv.i.data();
    c->fd_x     = sv.x.data();
    c->fd_count = &count;
    c->fd_n     = n;
    c->op       = WARP_FROMDENSE;
    __syncwarp(0xffffffffu);          // A
    warp_collective(*c);
    __syncwarp(0xffffffffu);          // B
    sv.i.set_size(count);
    sv.x.set_size(count);
  }

  // Post the dual ratio test to the whole warp (op == WARP_RT). Returns via the out-pointers:
  // *entering (column j or -1), *kout (nonbasic_list index), *step (chosen ratio / step length).
  __device__ void warp_dispatch_rt(const i_t* nonbasic, const variable_status_t* vstatus, const f_t* z,
                                   const f_t* dz, i_t m, i_t n, f_t ptol, f_t dtol, f_t ztol,
                                   i_t* entering, i_t* kout, f_t* step) const
  {
    warp_solve_config* c = warp_cfg_;
    c->rt_nonbasic  = nonbasic;
    c->rt_vstatus   = vstatus;
    c->rt_z         = z;
    c->rt_dz        = dz;
    c->rt_entering  = entering;
    c->rt_k         = kout;
    c->rt_step      = step;
    c->rt_m         = m;
    c->rt_n         = n;
    c->rt_pivot_tol = ptol;
    c->rt_dual_tol  = dtol;
    c->rt_zero_tol  = ztol;
    c->op           = WARP_RT;
    __syncwarp(0xffffffffu);          // A
    warp_collective(*c);
    __syncwarp(0xffffffffu);          // B
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
    work_estimate_ += 2 * nz;
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
    work_estimate_ += 5 * nz;
    work_estimate_ += 3 * out.i.size();
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
    work_estimate_ += 3 * (m - top);
    for (i_t k = 0; k < nz; ++k) {
      const i_t i      = xi_workspace_[m + k];
      xi_workspace_[i] = 1;
    }
    work_estimate_ += 2 * nz;
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
    work_estimate_ += 2 * nz;
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
    work_estimate_ += 4 * k + 3 * out.i.size();
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
    work_estimate_ += 3 * nz;
    for (i_t k = 0; k < nz; ++k) {
      x_workspace_[in.i[k]] = in.x[k];
    }
    work_estimate_ += 3 * nz;
    return nz;
  }

  __device__ void grow_storage(i_t nz, i_t& S_start, i_t& S_nz)
  {
    const i_t last_S_col     = num_updates_ * 2;
    const i_t new_last_S_col = last_S_col + 2;
    if (new_last_S_col >= static_cast<i_t>(S_.col_start.size())) {
      S_.col_start.resize(new_last_S_col + refactor_frequency_);
      work_estimate_ += new_last_S_col + refactor_frequency_;
    }
    S_nz = S_.col_start[last_S_col];
    if (S_nz + nz > static_cast<i_t>(S_.i.size())) {
      const i_t newsz = (2 * S_nz > S_nz + nz) ? 2 * S_nz : S_nz + nz;
      S_.i.resize(newsz);
      S_.x.resize(newsz);
      work_estimate_ += 2 * newsz;
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
    work_estimate_ += 3 * (col_end - col_start);
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
    work_estimate_ += 2 * nz_mark + (col_end - col_start);
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
    work_estimate_ += 3 * (col_end - col_start);
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
    work_estimate_ += 4 * (col_end - col_start) + 2 * (nz - nz_start);
  }

  // ---- triangular solves through the MPF form ----
  template <bool Warp = false>
  __device__ i_t u_transpose_solve(f_t* rhs) const
  {
    if constexpr (Warp) { warp_dispatch(WARP_UT_SOLVE, rhs); }
    else { upper_triangular_transpose_solve(U0_, rhs, work_estimate_); }
    return 0;
  }
  __device__ i_t u_transpose_solve(gpu_sparse_vector& rhs) const
  {
    i_t top = sparse_triangle_solve<true>(
      rhs, nullptr, xi_workspace_.data(), U0_transpose_, x_workspace_.data(), work_estimate_);
    solve_to_sparse_vector(top, rhs);
    return 0;
  }

  template <bool Warp = false>
  __device__ i_t l_transpose_solve(f_t* rhs) const
  {
    const f_t zero_tol = 1e-13;
    if constexpr (Warp) {
      // dot col = u_col (2k), axpy col = v_col (2k+1), descending k.
      if (num_updates_ > 0) { warp_dispatch_tloop(rhs, /*asc=*/false, /*dotoff=*/0, /*axpyoff=*/1, zero_tol); }
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
    work_estimate_ += 2 * num_updates_;
    if constexpr (Warp) { warp_dispatch(WARP_LT_SOLVE, rhs); }
    else { lower_triangular_transpose_solve(L0_, rhs, work_estimate_); }
    return 0;
  }
  __device__ i_t l_transpose_solve(gpu_sparse_vector& rhs) const
  {
    const i_t m = L0_.m;
    scatter_into_workspace(rhs);
    i_t nz = rhs.i.size();

    const f_t zero_tol = 1e-13;
    for (i_t k = num_updates_ - 1; k >= 0; --k) {
      const i_t u_col = 2 * k;
      const i_t v_col = 2 * k + 1;
      const f_t mu    = mu_values_[k];
      f_t dot         = dot_product(u_col, xi_workspace_.data(), x_workspace_.data());
      const f_t theta = dot / mu;
      if (fabs(theta) > zero_tol) {
        add_sparse_column(S_, v_col, -theta, xi_workspace_.data(), nz, x_workspace_.data());
      }
    }
    work_estimate_ += 2 * num_updates_;

    gpu_sparse_vector& b = sparse_scratch_;  // reused (was a per-hypersparse-solve device malloc)
    b.n = m;
    work_estimate_ += nz;
    gather_into_sparse_vector(nz, b);
    i_t top = sparse_triangle_solve<false>(
      b, nullptr, xi_workspace_.data(), L0_transpose_, x_workspace_.data(), work_estimate_);
    solve_to_sparse_vector(top, rhs);
    return 0;
  }

  template <bool Warp = false>
  __device__ i_t u_solve(f_t* rhs) const
  {
    if constexpr (Warp) { warp_dispatch(WARP_U_SOLVE, rhs); }
    else { upper_triangular_solve(U0_, rhs, work_estimate_); }
    return 0;
  }
  __device__ i_t u_solve(gpu_sparse_vector& rhs) const
  {
    i_t top = sparse_triangle_solve<false>(
      rhs, nullptr, xi_workspace_.data(), U0_, x_workspace_.data(), work_estimate_);
    solve_to_sparse_vector(top, rhs);
    return 0;
  }

  template <bool Warp = false>
  __device__ i_t l_solve(f_t* rhs) const
  {
    if constexpr (Warp) { warp_dispatch(WARP_L_SOLVE, rhs); }
    else { lower_triangular_solve(L0_, rhs, work_estimate_); }
    const f_t zero_tol = 1e-16;  // Any higher and pilot_ja fails
    if constexpr (Warp) {
      // dot col = v_col (2k+1), axpy col = u_col (2k), ascending k.
      if (num_updates_ > 0) { warp_dispatch_tloop(rhs, /*asc=*/true, /*dotoff=*/1, /*axpyoff=*/0, zero_tol); }
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
    work_estimate_ += 2 * num_updates_;
    return 0;
  }
  __device__ i_t l_solve(gpu_sparse_vector& rhs) const
  {
    const i_t m = L0_.m;
    i_t top     = sparse_triangle_solve<true>(
      rhs, nullptr, xi_workspace_.data(), L0_, x_workspace_.data(), work_estimate_);
    solve_to_workspace(top);
    i_t nz             = m - top;
    const f_t zero_tol = 1e-13;
    for (i_t k = 0; k < num_updates_; ++k) {
      const f_t mu    = mu_values_[k];
      const i_t u_col = 2 * k;
      const i_t v_col = 2 * k + 1;
      f_t dot         = dot_product(v_col, xi_workspace_.data(), x_workspace_.data());
      const f_t theta = dot / mu;
      if (fabs(theta) > zero_tol) {
        add_sparse_column(S_, u_col, -theta, xi_workspace_.data(), nz, x_workspace_.data());
      }
    }
    work_estimate_ += 2 * num_updates_;
    gather_into_sparse_vector(nz, rhs);
    return 0;
  }

  // ---- B solves ----
  template <bool Warp = false>
  __device__ i_t b_solve(const gpu_vector<f_t>& rhs, gpu_vector<f_t>& solution) const
  {
    permute_vector(row_permutation_.data(), rhs.data(), solution.data(), L0_.m);
    work_estimate_ += 3 * rhs.size();
    l_solve<Warp>(solution.data());
    u_solve<Warp>(solution.data());
    return 0;
  }

  template <bool Warp = false>
  __device__ i_t b_solve(const gpu_sparse_vector& rhs,
                         gpu_sparse_vector& solution,
                         gpu_sparse_vector& Lsol,
                         bool need_Lsol = true) const
  {
    solution = rhs;
    work_estimate_ += 2 * rhs.i.size();
    solution.inverse_permute_vector(inverse_row_permutation_.data());
    work_estimate_ += 3 * rhs.i.size();

    bool use_hypersparse;
    const f_t input_size = static_cast<f_t>(rhs.i.size());
    estimate_solution_density(input_size, sum_L_, num_calls_L_, use_hypersparse);
    if (use_hypersparse) {
      l_solve(solution);
    } else {
      gpu_vector<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      if constexpr (Warp) { warp_to_dense(solution, solution_dense); } else { solution.to_dense(solution_dense); }
      work_estimate_ += solution_dense.size();
      l_solve<Warp>(solution_dense.data());
      if constexpr (Warp) { warp_from_dense(solution_dense, solution); } else { solution.from_dense(solution_dense); }
      work_estimate_ += solution_dense.size();
    }
    if (need_Lsol) {
      Lsol = solution;
      work_estimate_ += 2 * solution.i.size();
    }
    sum_L_ += static_cast<f_t>(solution.i.size()) / input_size;

    const f_t rhs_size = static_cast<f_t>(solution.i.size());
    estimate_solution_density(rhs_size, sum_U_, num_calls_U_, use_hypersparse);
    if (use_hypersparse) {
      u_solve(solution);
    } else {
      gpu_vector<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      if constexpr (Warp) { warp_to_dense(solution, solution_dense); } else { solution.to_dense(solution_dense); }
      work_estimate_ += solution_dense.size();
      u_solve<Warp>(solution_dense.data());
      if constexpr (Warp) { warp_from_dense(solution_dense, solution); } else { solution.from_dense(solution_dense); }
      work_estimate_ += solution_dense.size();
    }
    sum_U_ += static_cast<f_t>(solution.i.size()) / rhs_size;
    return 0;
  }

  template <bool Warp = false>
  __device__ i_t b_solve(const gpu_sparse_vector& rhs, gpu_sparse_vector& solution) const
  {
    gpu_sparse_vector Lsol(1, 0);
    return b_solve<Warp>(rhs, solution, Lsol, false);
  }

  template <bool Warp = false>
  __device__ i_t b_transpose_solve(const gpu_sparse_vector& rhs,
                                   gpu_sparse_vector& solution,
                                   gpu_sparse_vector& UTsol) const
  {
    bool use_hypersparse = false;
    const f_t input_size = static_cast<f_t>(rhs.i.size());
    estimate_solution_density(
      input_size, sum_U_transpose_, num_calls_U_transpose_, use_hypersparse);
    if (use_hypersparse) {
      solution = rhs;
      u_transpose_solve(solution);
    } else {
      gpu_vector<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      if constexpr (Warp) { warp_to_dense(rhs, solution_dense); } else { rhs.to_dense(solution_dense); }
      u_transpose_solve<Warp>(solution_dense.data());
      if constexpr (Warp) { warp_from_dense(solution_dense, solution); } else { solution.from_dense(solution_dense); }
    }
    UTsol = solution;
    sum_U_transpose_ += static_cast<f_t>(solution.i.size()) / input_size;

    const f_t rhs_size = static_cast<f_t>(solution.i.size());
    estimate_solution_density(
      rhs_size, sum_L_transpose_, num_calls_L_transpose_, use_hypersparse);
    if (use_hypersparse) {
      l_transpose_solve(solution);
    } else {
      gpu_vector<f_t>& solution_dense = dense_scratch_;  // reused (was a per-solve device malloc)
      if constexpr (Warp) { warp_to_dense(solution, solution_dense); } else { solution.to_dense(solution_dense); }
      l_transpose_solve<Warp>(solution_dense.data());
      if constexpr (Warp) { warp_from_dense(solution_dense, solution); } else { solution.from_dense(solution_dense); }
    }
    sum_L_transpose_ += static_cast<f_t>(solution.i.size()) / rhs_size;
    solution.inverse_permute_vector(row_permutation_.data());
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
    work_estimate_ += (m + nz) * log2((f_t)(m + nz));

    i_t S_start;
    i_t S_nz;
    grow_storage(nz + static_cast<i_t>(etilde.i.size()), S_start, S_nz);

    S_.append_column(nz, xi_workspace_.data() + m, x_workspace_.data());
    work_estimate_ += 5 * nz;

    etilde.sort(sort_bucket_, sort_i_, sort_x_, sort_perm_);  // reused scratch (no per-pivot malloc)
    work_estimate_ += etilde.i.size() * log2((f_t)etilde.i.size());
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
    work_estimate_ += 4 * etilde.i.size();

    const f_t mu = 1.0 + sparse_dot(S_.i.data() + S_.col_start[S_start],
                                    S_.x.data() + S_.col_start[S_start],
                                    S_.col_start[S_start + 1] - S_.col_start[S_start],
                                    S_.i.data() + S_.col_start[S_start + 1],
                                    S_.x.data() + S_.col_start[S_start + 1],
                                    S_.col_start[S_start + 2] - S_.col_start[S_start + 1]);
    work_estimate_ += 3 * min(nz, static_cast<i_t>(etilde.i.size()));

    if (fabs(mu) < 1E-8 || fabs(mu) > 1E+8) { return 1; }
    mu_values_.push_back(mu);
    // Clear the workspace.
    for (i_t k = 0; k < nz; ++k) {
      const i_t i          = xi_workspace_[m + k];
      xi_workspace_[i]     = 0;
      x_workspace_[i]      = 0.0;
      xi_workspace_[m + k] = 0;
    }
    work_estimate_ += 4 * nz;
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
// phase2::* helpers (phase2.cpp / solve.cpp). work_estimate is kept for fidelity
// but is unused by the result.
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
__device__ void update_objective(const i_t* basic_list,
                                 const gpu_sparse_vector& changed_basic_indices,
                                 const gpu_span<f_t>& objective,
                                 const f_t* delta_x,
                                 i_t entering_index,
                                 f_t& obj)
{
  const i_t changed_basic_nz = (i_t)changed_basic_indices.i.size();
  for (i_t k = 0; k < changed_basic_nz; ++k) {
    const i_t j = basic_list[changed_basic_indices.i[k]];
    obj += delta_x[j] * objective[j];
  }
  // Leaving index already included above
  obj += delta_x[entering_index] * objective[entering_index];
}

template <bool Warp = false>
__device__ void compute_reduced_cost_update(const gpu_lp& lp,
                                            const gpu_basis_update_mpf& ft,
                                            const i_t* basic_list,
                                            const i_t* nonbasic_list,
                                            const f_t* delta_y,
                                            i_t leaving_index,
                                            i_t direction,
                                            i_t* delta_z_mark,
                                            gpu_vector<i_t>& delta_z_indices,
                                            f_t* delta_z,
                                            f_t& work_estimate)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  if constexpr (Warp) {
    // One warp dispatch: parallel gather dots (one lane per nonbasic column, bit-exact) + ordered
    // append of the nonzero columns. delta_z_indices is empty here (cleared each iter) and reserved
    // to n, so the collective raw-writes [0,count) and we set the size back on return.
    i_t dz_count = 0;
    ft.warp_dispatch_dz(lp.A.col_start.data(), lp.A.i.data(), lp.A.x.data(), delta_y, basic_list,
                        nonbasic_list, delta_z, delta_z_indices.data(), delta_z_mark, &dz_count,
                        &work_estimate, m, n, leaving_index, direction);
    delta_z_indices.set_size(dz_count);
    return;
  }

  size_t nnzs_processed = 0;
  for (i_t k = 0; k < m; k++) {
    const i_t j = basic_list[k];
    delta_z[j]  = 0;
  }
  work_estimate += 2 * m;
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
    nnzs_processed += col_end - col_start;

    delta_z[j] = -dot;
    if (dot != 0.0) {
      delta_z_indices.push_back(j);
      delta_z_mark[j] = 1;
    }
  }
  work_estimate += 3 * num_nonbasic;
  work_estimate += 3 * nnzs_processed;
  work_estimate += 2 * delta_z_indices.size();
}

// Transpose delta_z: delta_z_N = -A_N' delta_y, row-wise on the CSR copy Arow,
// iterating only the nonzero rows of delta_y and only their nonbasic-column portion
// (Arow.row_start[i] .. nonbasic_end[i]). Mirrors the CPU compute_delta_z bit-for-bit.
__device__ void compute_delta_z(const gpu_csr_matrix& Arow,
                                const gpu_sparse_vector& delta_y,
                                i_t leaving_index,
                                i_t direction,
                                const i_t* nonbasic_end,
                                i_t* delta_z_mark,
                                gpu_vector<i_t>& delta_z_indices,
                                f_t* delta_z,
                                f_t& work_estimate)
{
  const i_t nz_delta_y = (i_t)delta_y.i.size();
  size_t nnz_processed = 0;
  for (i_t k = 0; k < nz_delta_y; k++) {
    const i_t i         = delta_y.i[k];
    const f_t delta_y_i = delta_y.x[k];
    if (fabs(delta_y_i) < 1e-12) { continue; }
    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = nonbasic_end[i] + 1;
    nnz_processed += row_end - row_start;
    for (i_t p = row_start; p < row_end; ++p) {
      const i_t j = Arow.j[p];
      delta_z[j] -= delta_y_i * Arow.x[p];
      if (!delta_z_mark[j]) {
        delta_z_mark[j] = 1;
        delta_z_indices.push_back(j);
      }
    }
  }
  work_estimate += 4 * nz_delta_y;
  work_estimate += 4 * nnz_processed;
  work_estimate += 2 * delta_z_indices.size();
  delta_z[leaving_index] = direction;
}

// Partition each row of Arow so nonbasic-variable coefficients occupy
// Arow.row_start[i] .. nonbasic_end[i]. Mirrors compute_initial_nonbasic_end.
__device__ void compute_initial_nonbasic_end(const i_t* basic_mark,
                                             gpu_csr_matrix& Arow,
                                             i_t* nonbasic_end)
{
  const i_t m = Arow.m;
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
                            gpu_csr_matrix& Arow,
                            f_t& work_estimate)
{
  row_mark.clear();
  const i_t col_start_enter = A.col_start[entering];
  const i_t col_end_enter   = A.col_start[entering + 1];
  for (i_t p = col_start_enter; p < col_end_enter; p++) { row_mark.push_back(A.i[p]); }
  work_estimate += 2 * (col_end_enter - col_start_enter);

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
    work_estimate += nb_end - row_start;
  }
  work_estimate += 2 * row_mark.size();

  row_mark.clear();
  const i_t col_start_leave = A.col_start[leaving];
  const i_t col_end_leave   = A.col_start[leaving + 1];
  for (i_t p = col_start_leave; p < col_end_leave; p++) { row_mark.push_back(A.i[p]); }
  work_estimate += 2 * (col_end_leave - col_start_leave);

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
    work_estimate += row_end - nb_end;
  }
  work_estimate += 2 * row_mark.size();
}

// Exact steepest-edge norm init: gamma_j = ||B^-T e_k||^2 for each basic var, via a
// singleton-row fast path (1/alpha^2) or a b_transpose_solve. Mirrors the CPU
// initialize_steepest_edge_norms (the simplified always uses the non-slack/exact path).
template <bool Warp = false>
__device__ i_t initialize_steepest_edge_norms(const gpu_lp& lp,
                                              const i_t* basic_list,
                                              gpu_basis_update_mpf& ft,
                                              f_t* delta_y_steepest_edge,
                                              f_t& work_estimate)
{
  const i_t m = lp.num_rows;
  gpu_vector<i_t> row_degree(m, 0);
  gpu_vector<i_t> mapping(m, -1);
  gpu_vector<f_t> coeff(m, 0.0);
  work_estimate += 3 * m;
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
    work_estimate += 5 * (col_end - col_start);
  }
  work_estimate += 3 * m;

  gpu_sparse_vector sparse_dy(m, 0);
  gpu_sparse_vector UTsol_scratch(m, 0);
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    f_t init    = -1.0;
    if (row_degree[mapping[k]] == 1) {
      const f_t alpha = coeff[k];
      init            = 1.0 / (alpha * alpha);
    } else {
      gpu_sparse_vector sparse_ei(m, 1);
      sparse_ei.x[0] = -1.0;
      sparse_ei.i[0] = k;
      sparse_dy.clear();
      ft.b_transpose_solve<Warp>(sparse_ei, sparse_dy, UTsol_scratch);
      f_t my_init = 0.0;
      for (i_t p = 0; p < (i_t)sparse_dy.x.size(); ++p) {
        my_init += sparse_dy.x[p] * sparse_dy.x[p];
      }
      work_estimate += 2 * sparse_dy.x.size();
      init = my_init;
    }
    delta_y_steepest_edge[j] = init;
  }
  work_estimate += 7 * m;
  return 0;
}

template <bool Warp = false>
__device__ void compute_primal_variables(const gpu_basis_update_mpf& ft,
                                         const gpu_span<f_t>& lp_rhs,
                                         const gpu_csc_view& A,
                                         const i_t* basic_list,
                                         const i_t* nonbasic_list,
                                         f_t tight_tol,
                                         f_t* x,
                                         gpu_vector<f_t>& xB_workspace,
                                         f_t& work_estimate)
{
  const i_t m = A.m;
  const i_t n = A.n;
  gpu_vector<f_t> rhs(m);
  for (i_t k = 0; k < m; ++k) { rhs[k] = lp_rhs[k]; }
  work_estimate += 2 * m;
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j         = nonbasic_list[k];
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    const f_t xj        = x[j];
    if (fabs(xj) < tight_tol * 10) continue;
    for (i_t p = col_start; p < col_end; ++p) {
      rhs[A.i[p]] -= xj * A.x[p];
    }
    work_estimate += 3 * (col_end - col_start);
  }
  work_estimate += 5 * (n - m);

  xB_workspace.resize(m);
  work_estimate += m;
  ft.b_solve<Warp>(rhs, xB_workspace);

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    x[j]        = xB_workspace[k];
  }
  work_estimate += 2 * m;
}

__device__ void clear_delta_z(
  i_t entering_index, i_t leaving_index, i_t* delta_z_mark, gpu_vector<i_t>& delta_z_indices, f_t* delta_z)
{
  const i_t nz = delta_z_indices.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j     = delta_z_indices[k];
    delta_z[j]      = 0.0;
    delta_z_mark[j] = 0;
  }
  if (entering_index != -1) { delta_z[entering_index] = 0.0; }
  delta_z[leaving_index] = 0.0;
  delta_z_indices.clear();
}

__device__ f_t compute_initial_primal_infeasibilities(const gpu_lp& lp,
                                                      const gpu_settings& settings,
                                                      const i_t* basic_list,
                                                      const f_t* x,
                                                      gpu_vector<f_t>& squared_infeasibilities,
                                                      gpu_vector<i_t>& infeasibility_indices,
                                                      f_t& primal_inf)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
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
                                                          f_t& max_val,
                                                          f_t& work_estimate)
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
  work_estimate += 3 * nz + 3 * max_count;
  basic_leaving = leaving_index >= 0 ? basic_mark[leaving_index] : -1;
  return leaving_index;
}

template <bool Warp = false>
__device__ i_t phase2_ratio_test(const gpu_lp& lp,
                                 const gpu_settings& settings,
                                 const gpu_basis_update_mpf& ft,
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

  if constexpr (Warp) {
    i_t entering = -1, kk = -1;
    f_t step = dev_inf();
    ft.warp_dispatch_rt(nonbasic_list, vstatus, z, delta_z, m, n, pivot_tol, dual_tol, zero_tol,
                        &entering, &kk, &step);
    if (entering != -1) { step_length = step; nonbasic_entering = kk; }
    return entering;
  }

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

template <bool Warp = false>
__device__ void compute_delta_y(const gpu_basis_update_mpf& ft,
                                i_t basic_leaving_index,
                                i_t direction,
                                gpu_sparse_vector& delta_y_sparse,
                                gpu_sparse_vector& UTsol_sparse)
{
  const i_t m = delta_y_sparse.n;
  gpu_sparse_vector& ei_sparse = ft.ei_scratch_;  // reused (was a per-pivot device malloc)
  ei_sparse.n = m;
  ei_sparse.i.resize(1);
  ei_sparse.x.resize(1);
  ei_sparse.i[0] = basic_leaving_index;
  ei_sparse.x[0] = -direction;
  ft.b_transpose_solve<Warp>(ei_sparse, delta_y_sparse, UTsol_sparse);

  if (direction != -1) { UTsol_sparse.negate(); }
}

__device__ i_t update_dual_variables(const gpu_sparse_vector& delta_y_sparse,
                                     const gpu_vector<i_t>& delta_z_indices,
                                     const f_t* delta_z,
                                     f_t step_length,
                                     i_t leaving_index,
                                     f_t* y,
                                     f_t* z,
                                     f_t& work_estimate)
{
  const i_t delta_y_nz = delta_y_sparse.i.size();
  for (i_t k = 0; k < delta_y_nz; ++k) {
    const i_t i = delta_y_sparse.i[k];
    y[i] += step_length * delta_y_sparse.x[k];
  }
  work_estimate += 3 * delta_y_nz;
  const i_t delta_z_nz = delta_z_indices.size();
  for (i_t k = 0; k < delta_z_nz; ++k) {
    const i_t j = delta_z_indices[k];
    z[j] += step_length * delta_z[j];
  }
  work_estimate += 3 * delta_z_nz;
  z[leaving_index] += step_length * delta_z[leaving_index];
  return 0;
}

template <bool Warp = false>
__device__ i_t compute_delta_x(const gpu_lp& lp,
                               const gpu_basis_update_mpf& ft,
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
                               f_t* delta_x,
                               f_t& work_estimate)
{
  f_t delta_x_leaving = direction == 1 ? lp.lower[leaving_index] - x[leaving_index]
                                       : lp.upper[leaving_index] - x[leaving_index];
  ft.b_solve<Warp>(rhs_sparse, scaled_delta_xB_sparse, utilde_sparse);
  scaled_delta_xB_sparse.negate();
  work_estimate += 2 * scaled_delta_xB_sparse.i.size();

  f_t scale = scaled_delta_xB_sparse.find_coefficient(basic_leaving_index);
  work_estimate += 2 * scaled_delta_xB_sparse.i.size();
  if (scale != scale) {
    gpu_vector<f_t> rhs;
    rhs_sparse.to_dense(rhs);
    work_estimate += 2 * rhs.size();
    const i_t m = lp.num_rows;
    gpu_vector<f_t> scaled_delta_xB(m);
    work_estimate += m;
    ft.b_solve<Warp>(rhs, scaled_delta_xB);
    if (scaled_delta_xB[basic_leaving_index] != 0.0 &&
        !isnan(scaled_delta_xB[basic_leaving_index])) {
      scaled_delta_xB_sparse.from_dense(scaled_delta_xB);
      scaled_delta_xB_sparse.negate();
      work_estimate += 2 * scaled_delta_xB_sparse.i.size() + scaled_delta_xB.size();
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
      return -1;
    }
  }
  const f_t primal_step_length = delta_x_leaving / scale;
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    delta_x[j]  = primal_step_length * scaled_delta_xB_sparse.x[k];
  }
  work_estimate += 4 * scaled_delta_xB_nz;
  delta_x[leaving_index]  = delta_x_leaving;
  delta_x[entering_index] = primal_step_length;
  return 0;
}

template <bool Warp = false>
__device__ i_t update_steepest_edge_norms(const gpu_settings& settings,
                                          const i_t* basic_list,
                                          const gpu_basis_update_mpf& ft,
                                          i_t direction,
                                          const gpu_sparse_vector& delta_y_sparse,
                                          f_t dy_norm_squared,
                                          const gpu_sparse_vector& scaled_delta_xB,
                                          i_t basic_leaving_index,
                                          i_t entering_index,
                                          f_t* v,
                                          gpu_sparse_vector& v_sparse,
                                          f_t* delta_y_steepest_edge,
                                          f_t& work_estimate)
{
  v_sparse.clear();
  ft.b_solve<Warp>(delta_y_sparse, v_sparse);
  if (direction == -1) {
    v_sparse.negate();
    work_estimate += 2 * v_sparse.i.size();
  }
  v_sparse.scatter(v);
  work_estimate += 2 * v_sparse.i.size();

  const f_t wr = -scaled_delta_xB.find_coefficient(basic_leaving_index);
  work_estimate += scaled_delta_xB.i.size();
  if (wr == 0) { return -1; }
  const f_t omegar             = dy_norm_squared / (wr * wr);
  const i_t scaled_delta_xB_nz = scaled_delta_xB.i.size();

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
  work_estimate += 5 * scaled_delta_xB_nz;

  const i_t v_nz = v_sparse.i.size();
  for (i_t k = 0; k < v_nz; ++k) {
    v[v_sparse.i[k]] = 0.0;
  }
  work_estimate += 2 * v_nz;
  return 0;
}

__device__ void update_primal_variables(const gpu_sparse_vector& scaled_delta_xB_sparse,
                                        const i_t* basic_list,
                                        const f_t* delta_x,
                                        i_t entering_index,
                                        f_t* x,
                                        f_t& work_estimate)
{
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    x[j] += delta_x[j];
  }
  work_estimate += 4 * scaled_delta_xB_nz;
  x[entering_index] += delta_x[entering_index];
}

__device__ void set_primal_variables_on_bounds(const gpu_lp& lp,
                                               const gpu_settings& settings,
                                               const f_t* z,
                                               variable_status_t* vstatus,
                                               f_t* x)
{
  const i_t n = lp.num_cols;
  const f_t inf = dev_inf();
  f_t tol       = 1e-10;
  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] == variable_status_t::BASIC) { continue; }
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
  }
}

// ---------------------------------------------------------------------------
// The device driver: a line-for-line copy of simplified_dual_phase2's
// pure-MPF-stacking path.
// ---------------------------------------------------------------------------
template <bool Warp = false>
__device__ dual::status_t gpu_simplified_dual_phase2_device(const gpu_lp& lp,
                                                            const gpu_settings& settings,
                                                            variable_status_t* vstatus,
                                                            gpu_basis_update_mpf& ft,
                                                            i_t* basic_list,
                                                            i_t* nonbasic_list,
                                                            gpu_solution& sol,
                                                            f_t* delta_y_steepest_edge,
                                                            bool dy_se_empty)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  // Device wall-clock reference for the time-limit guard.
  const unsigned long long start_ns = dev_globaltimer();

  f_t* x = sol.x.data();
  f_t* y = sol.y.data();
  f_t* z = sol.z.data();

  f_t work = 0.0;
  gpu_vector<f_t> xB_workspace(m);

  set_primal_variables_on_bounds(lp, settings, z, vstatus, x);
  compute_primal_variables<Warp>(
    ft, lp.rhs, lp.A, basic_list, nonbasic_list, settings.tight_tol, x, xB_workspace, work);

  if (dy_se_empty) {
    // Exact steepest-edge init (matches the CPU simplified's unconditional exact path).
    for (i_t k = 0; k < n; ++k) { delta_y_steepest_edge[k] = -1.0; }
    initialize_steepest_edge_norms<Warp>(lp, basic_list, ft, delta_y_steepest_edge, work);
  } else {
    for (i_t k = 0; k < m; ++k) {
      const i_t j = basic_list[k];
      if (delta_y_steepest_edge[j] <= 0.0) { delta_y_steepest_edge[j] = 1e-4; }
    }
  }

  gpu_vector<f_t> delta_y(m, 0.0);
  gpu_vector<f_t> delta_z(n, 0.0);
  gpu_vector<f_t> delta_x(n, 0.0);
  gpu_vector<f_t> v(m, 0.0);
  gpu_vector<i_t> basic_mark(n, -1);
  for (i_t k = 0; k < m; ++k) { basic_mark[basic_list[k]] = k; }
  gpu_vector<i_t> delta_z_mark(n, 0);
  gpu_vector<i_t> delta_z_indices;
  delta_z_indices.reserve(n);

  // Transpose delta_z state (Arow CSR copy of A + per-row nonbasic partition). Built
  // once here; nonbasic_end is maintained per pivot by update_Arow (no refactor rebuild
  // needed — the GPU is pure MPF stacking).
  gpu_csr_matrix Arow;
  Arow.from_csc(lp.A);
  gpu_vector<i_t> nonbasic_end(m);
  gpu_vector<i_t> row_mark;
  row_mark.reserve(m);
  compute_initial_nonbasic_end(basic_mark.data(), Arow, nonbasic_end.data());

  gpu_vector<f_t> squared_infeas;
  gpu_vector<i_t> infeas_idx;
  f_t primal_inf = 0.0;
  compute_initial_primal_infeasibilities(
    lp, settings, basic_list, x, squared_infeas, infeas_idx, primal_inf);

  gpu_sparse_vector delta_y_sparse(m, 0);
  gpu_sparse_vector UTsol_sparse(m, 0);
  gpu_sparse_vector utilde_sparse(m, 0);
  gpu_sparse_vector scaled_dxB_sparse(m, 0);
  gpu_sparse_vector rhs_sparse(m, 0);
  gpu_sparse_vector v_sparse(m, 0);

  const i_t iter_limit = min(500000, max(20000, 20 * (m + n)));
  // Incrementally-tracked objective for the cutoff, mirroring the CPU simplified (and
  // advanced) driver: init fresh once, then update_objective per pivot. Using the true
  // costs lp.objective (the GPU has no perturbation) is a valid dual bound.
  f_t obj  = compute_objective(lp, x);
  i_t iter = 0;
  while (iter < iter_limit) {
    if (g_gpu_alloc_failed) {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iter;
      return dual::status_t::NUMERICAL;
    }
    if (static_cast<f_t>(dev_globaltimer() - start_ns) * 1e-9 > settings.time_limit) {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iter;
      return dual::status_t::TIME_LIMIT;
    }

    // 1. Steepest-edge pricing.
    i_t direction        = 0;
    i_t basic_leaving    = -1;
    f_t max_val;
    const i_t leaving = steepest_edge_pricing_with_infeasibilities(
      lp, settings, x, delta_y_steepest_edge, basic_mark.data(), squared_infeas, infeas_idx,
      direction, basic_leaving, max_val, work);
    if (leaving == -1) {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iter;
      return dual::status_t::OPTIMAL;
    }

    // 2. BTran.
    delta_y_sparse.clear();
    UTsol_sparse.clear();
    compute_delta_y<Warp>(ft, basic_leaving, direction, delta_y_sparse, UTsol_sparse);
    const f_t dy_norm_sq = delta_y_sparse.norm2_squared();

    // Steepest-edge norm drift safeguard: dy_norm_sq is the true norm of the leaving
    // row. If the maintained norm used for pricing has drifted below
    // settings.steepest_edge_ratio of it, the leaving choice was made on a stale norm --
    // reset it and re-price (strict port of the CPU simplified, phase2.cpp:4037).
    if (delta_y_steepest_edge[leaving] < settings.steepest_edge_ratio * dy_norm_sq) {
      delta_y_steepest_edge[leaving] = dy_norm_sq;
      continue;
    }

    // 3. Reduced-cost SpMv: transpose (CSR) when delta_y is sparse (<=30% dense), else
    //    dense CSC. Mirrors the CPU driver.
    i_t delta_y_nz0      = 0;
    const i_t nz_delta_y = (i_t)delta_y_sparse.i.size();
    for (i_t k = 0; k < nz_delta_y; ++k) {
      if (fabs(delta_y_sparse.x[k]) > 1e-12) { delta_y_nz0++; }
    }
    const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
    const bool used_transpose_dz    = delta_y_nz_percentage <= 30.0;
    if (used_transpose_dz) {
      compute_delta_z(Arow, delta_y_sparse, leaving, direction, nonbasic_end.data(),
                      delta_z_mark.data(), delta_z_indices, delta_z.data(), work);
    } else {
      delta_y_sparse.to_dense(delta_y);
      compute_reduced_cost_update<Warp>(lp, ft, basic_list, nonbasic_list, delta_y.data(), leaving,
                                        direction, delta_z_mark.data(), delta_z_indices,
                                        delta_z.data(), work);
    }

    // 4. Simple ratio test.
    f_t step_length;
    i_t nonbasic_entering = -1;
    const i_t entering    = phase2_ratio_test<Warp>(
      lp, settings, ft, vstatus, nonbasic_list, z, delta_z.data(), step_length, nonbasic_entering);
    if (entering == -1) {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iter;
      return dual::status_t::DUAL_UNBOUNDED;
    }

    update_dual_variables(
      delta_y_sparse, delta_z_indices, delta_z.data(), step_length, leaving, y, z, work);

    // FTran.
    utilde_sparse.clear();
    scaled_dxB_sparse.clear();
    rhs_sparse.from_csc_column(lp.A, entering);
    const i_t dx_status =
      compute_delta_x<Warp>(lp, ft, entering, leaving, basic_leaving, direction, basic_list, rhs_sparse,
                      delta_z.data(), x, utilde_sparse, scaled_dxB_sparse, delta_x.data(), work);
    if (dx_status == -1) {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iter;
      return dual::status_t::NUMERICAL;
    }

    update_steepest_edge_norms<Warp>(settings, basic_list, ft, direction, delta_y_sparse, dy_norm_sq,
                               scaled_dxB_sparse, basic_leaving, entering, v.data(), v_sparse,
                               delta_y_steepest_edge, work);

    update_primal_variables(scaled_dxB_sparse, basic_list, delta_x.data(), entering, x, work);

    // Incrementally update the objective the SAME way the CPU driver does (phase2.cpp:4259):
    // obj += delta_x·objective over the changed basics + entering, true costs, called with the
    // OLD basic_list (before the swap below), WITHOUT the bound-flip contribution.
    update_objective(basic_list, scaled_dxB_sparse, lp.objective, delta_x.data(), entering, obj);

    // 5. Basis swap + MPF rank-1 stacking update.
    vstatus[entering] = variable_status_t::BASIC;
    vstatus[leaving]  = (lp.lower[leaving] == lp.upper[leaving])
                         ? variable_status_t::NONBASIC_FIXED
                         : static_cast<variable_status_t>(-direction);
    basic_list[basic_leaving]        = entering;
    nonbasic_list[nonbasic_entering] = leaving;
    basic_mark[leaving]              = -1;
    basic_mark[entering]             = basic_leaving;

    // Keep the CSR nonbasic partition in sync for the next iteration's transpose delta_z.
    update_Arow(leaving, entering, lp.A, row_mark, nonbasic_end.data(), Arow, work);

    const i_t _fu_status = ft.update(utilde_sparse, UTsol_sparse, basic_leaving);
    if (_fu_status == 1) {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iter;
      return dual::status_t::NUMERICAL;
    }

    compute_initial_primal_infeasibilities(
      lp, settings, basic_list, x, squared_infeas, infeas_idx, primal_inf);
    clear_delta_z(entering, leaving, delta_z_mark.data(), delta_z_indices, delta_z.data());

    // Increment only on a COMPLETED iteration, at the end of the body — matching the CPU
    // simplified driver's iter++ (phase2.cpp:4370). The SE-reset `continue` above jumps
    // back to `while` without counting, so `iter` counts pivots, not loop passes.
    iter++;

    // Objective cutoff (dual bound), checked AFTER the pivot/iter++ — mirrors the CPU
    // simplified driver's placement + comparand (phase2.cpp:4377): the incrementally
    // tracked objective `obj` (true costs, omits bound flips — a valid, conservative lower
    // bound). Gated on a finite cutoff (default +inf skips it).
    if (isfinite(settings.cut_off)) {
      const f_t cutoff_obj = obj;
      if (cutoff_obj >= settings.cut_off) {
        sol.objective      = compute_objective(lp, x);
        sol.user_objective = compute_user_objective(lp, sol.objective);
        sol.iterations     = iter;
        return dual::status_t::CUTOFF;
      }
    }
  }
  sol.objective      = compute_objective(lp, x);
  sol.user_objective = compute_user_objective(lp, sol.objective);
  sol.iterations     = iter_limit;
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
                                   gpu_vector<i_t>& lvlscratch, gpu_vector<i_t>& bigscratch,
                                   gpu_vector<i_t>* ls, gpu_vector<i_t>* co,
                                   gpu_vector<i_t>* lw, gpu_vector<unsigned>* bt)
{
  cfg.mode = g_warp_solve_mode;
  cfg.m    = ft.L0_.m;
  // op order: L, U, Lᵀ, Uᵀ (see warp_solve_op). Gather matrix + flags per op.
  // Indexed by warp_solve_op: {L, U, Lᵀ, Uᵀ}.
  const gpu_csc_view* mats[kNumWarpSolveOps] = {&ft.L0_transpose_, &ft.U0_transpose_, &ft.L0_, &ft.U0_};
  const bool df[kNumWarpSolveOps]  = {false, true, true, false};   // diagonal first?
  const bool dsc[kNumWarpSolveOps] = {false, true, false, false};  // inner accumulation descending?
  const bool asc[kNumWarpSolveOps] = {true, false, false, true};   // level pull processes j ascending?
  const bool inc[kNumWarpSolveOps] = {true, true, false, false};   // incremental subtract (L/U) vs sum?
  for (int o = 0; o < kNumWarpSolveOps; ++o) {
    cfg.cs[o]         = mats[o]->col_start.data();
    cfg.ridx[o]       = mats[o]->i.data();
    cfg.val[o]        = mats[o]->x.data();
    cfg.diag_first[o] = df[o];
    cfg.descending[o] = dsc[o];
    cfg.incremental[o] = inc[o];
    if (cfg.mode == 1) {     // big: sorted large-first order + bitmask
      cfg.nlev[o]  = warp_build_levels_big(*mats[o], df[o], asc[o], lvlscratch, ls[o], co[o], lw[o],
                                           bt[o], bigscratch);
      cfg.lword[o] = lw[o].data();
      cfg.bits[o]  = bt[o].data();
    } else {                 // plain wavefront: one lane per column
      cfg.nlev[o]  = warp_build_levels(*mats[o], df[o], asc[o], lvlscratch, ls[o], co[o]);
    }
    cfg.lstart[o]     = ls[o].data();
    cfg.corder[o]     = co[o].data();
  }
}

// One block == one problem. args points to an N-entry device array; block b solves
// args[blockIdx.x] entirely on its own device buffers. Launched <<<1,1>>> for the
// single-thread reference (Warp=false) and <<<1,32>>> for the warp-optimized path
// (Warp=true, lane 0 drives, lanes 1..31 run the collective wavefront solves).
// g_gpu_alloc_failed is reset by the HOST before launch (not here) so a late block
// can't clear a failure flag an earlier block already set.
template <bool Warp>
__global__ void
// Only the max-threads bound must change for the 32-lane warp path; minBlocks stays 8
// (no effect here: one block is ever launched, and 65536/(32*8) already exceeds the
// 255-register hardware cap, so the warp kernel keeps the full register budget).
__launch_bounds__(Warp ? 32 : 1, 8)
gpu_phase2_kernel(kernel_args* args)
{
  // Reject a launch geometry that doesn't match the compiled path (the whole point
  // of the compile-time Warp parameter: no runtime branch selects opt vs unopt).
  if constexpr (Warp) {
    if (blockDim.x != 32) { assert(false); return; }
  } else {
    if (blockDim.x != 1) { assert(false); return; }
  }

  // Shared request/config for the persistent helper warp (unused when Warp==false).
  __shared__ warp_solve_config cfg_shared;

  if constexpr (Warp) {
    if (threadIdx.x != 0) {  // lanes 1..31: run collective solves until lane 0 says DONE
      warp_helper_loop(&cfg_shared);
      return;
    }
  }

  kernel_args& a = args[blockIdx.x];

  const i_t m = a.m;
  const i_t n = a.n;

  gpu_lp lp;
  lp.num_rows     = m;
  lp.num_cols     = n;
  lp.A            = make_view(m, n, a.A_nz, a.A_col_start, a.A_i, a.A_x);
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

  gpu_basis_update_mpf ft;
  ft.num_updates_           = a.num_updates;
  ft.refactor_frequency_    = a.refactor_frequency;
  ft.L0_                    = make_view(m, m, a.L0_nz, a.L0_col_start, a.L0_i, a.L0_x);
  ft.U0_                    = make_view(m, m, a.U0_nz, a.U0_col_start, a.U0_i, a.U0_x);
  ft.L0_transpose_          = make_view(m, m, a.L0t_nz, a.L0t_col_start, a.L0t_i, a.L0t_x);
  ft.U0_transpose_          = make_view(m, m, a.U0t_nz, a.U0t_col_start, a.U0t_i, a.U0t_x);
  ft.row_permutation_       = gpu_span<i_t>(a.row_perm, m);
  ft.inverse_row_permutation_ = gpu_span<i_t>(a.inv_row_perm, m);
  ft.xi_workspace_          = gpu_span<i_t>(a.xi_workspace, 2 * m);
  ft.x_workspace_           = gpu_span<f_t>(a.x_workspace, m);
  ft.S_.init_empty(m);
  ft.hypersparse_threshold_ = 0.05;

  gpu_solution sol;
  sol.x = gpu_span<f_t>(a.x, n);
  sol.y = gpu_span<f_t>(a.y, m);
  sol.z = gpu_span<f_t>(a.z, n);

  // Warp path: precompute the fixed L0/U0 elimination-level schedules once and
  // wire ft to the shared config. These gpu_vectors stay alive (lane 0) for the
  // whole solve; their device buffers are readable by all lanes via the config.
  gpu_vector<i_t> lvlscratch;
  gpu_vector<i_t> bigscratch;
  gpu_vector<i_t> lvl_ls[4];
  gpu_vector<i_t> lvl_co[4];
  gpu_vector<i_t> lvl_lw[4];       // mode 1 (big): per-level base word
  gpu_vector<unsigned> lvl_bt[4];  // mode 1 (big): large-column bitmask
  if constexpr (Warp) {
    warp_build_config(ft, cfg_shared, lvlscratch, bigscratch, lvl_ls, lvl_co, lvl_lw, lvl_bt);
    ft.warp_cfg_ = &cfg_shared;
  }

  dual::status_t status = gpu_simplified_dual_phase2_device<Warp>(lp,
                                                            settings,
                                                            a.vstatus,
                                                            ft,
                                                            a.basic_list,
                                                            a.nonbasic_list,
                                                            sol,
                                                            a.dy_se,
                                                            a.dy_se_empty != 0);

  if constexpr (Warp) {  // release the helper lanes from warp_helper_loop
    cfg_shared.op = WARP_DONE;
    __syncwarp(0xffffffffu);
  }

  *a.out_status         = static_cast<i_t>(status);
  *a.out_objective      = sol.objective;
  *a.out_user_objective = sol.user_objective;
  *a.out_iterations     = sol.iterations;
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
static kernel_args marshal_problem(const lp_problem_t<i_t, f_t>& lp,
                                   const simplex_solver_settings_t<i_t, f_t>& settings,
                                   std::vector<variable_status_t>& vstatus,
                                   basis_update_mpf_t<i_t, f_t>& ft,
                                   std::vector<i_t>& basic_list,
                                   std::vector<i_t>& nonbasic_list,
                                   lp_solution_t<i_t, f_t>& sol,
                                   std::vector<f_t>& delta_y_steepest_edge)
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

  a.A_col_start = upload(A.col_start);
  a.A_nz        = A.nnz();
  a.A_i         = upload_n(A.i.data(), A.nnz());
  a.A_x         = upload_n(A.x.data(), A.nnz());
  a.rhs         = upload(lp.rhs);
  a.lower       = upload(lp.lower);
  a.upper       = upload(lp.upper);
  a.objective   = upload(lp.objective);
  a.obj_scale   = lp.obj_scale;
  a.obj_constant = lp.obj_constant;

  a.L0_col_start = upload(L0.col_start);
  a.L0_nz        = L0.nnz();
  a.L0_i         = upload_n(L0.i.data(), L0.nnz());
  a.L0_x         = upload_n(L0.x.data(), L0.nnz());
  a.U0_col_start = upload(U0.col_start);
  a.U0_nz        = U0.nnz();
  a.U0_i         = upload_n(U0.i.data(), U0.nnz());
  a.U0_x         = upload_n(U0.x.data(), U0.nnz());
  a.L0t_col_start = upload(L0t.col_start);
  a.L0t_nz        = L0t.nnz();
  a.L0t_i         = upload_n(L0t.i.data(), L0t.nnz());
  a.L0t_x         = upload_n(L0t.x.data(), L0t.nnz());
  a.U0t_col_start = upload(U0t.col_start);
  a.U0t_nz        = U0t.nnz();
  a.U0t_i         = upload_n(U0t.i.data(), U0t.nnz());
  a.U0t_x         = upload_n(U0t.x.data(), U0t.nnz());

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
  return a;
}

static void free_kernel_args(const kernel_args& a)
{
  for (void* p : {(void*)a.A_col_start, (void*)a.A_i, (void*)a.A_x, (void*)a.rhs, (void*)a.lower,
                  (void*)a.upper, (void*)a.objective, (void*)a.L0_col_start, (void*)a.L0_i,
                  (void*)a.L0_x, (void*)a.U0_col_start, (void*)a.U0_i, (void*)a.U0_x,
                  (void*)a.L0t_col_start, (void*)a.L0t_i, (void*)a.L0t_x, (void*)a.U0t_col_start,
                  (void*)a.U0t_i, (void*)a.U0t_x, (void*)a.row_perm, (void*)a.inv_row_perm,
                  (void*)a.xi_workspace, (void*)a.x_workspace, (void*)a.vstatus,
                  (void*)a.basic_list, (void*)a.nonbasic_list, (void*)a.x, (void*)a.y, (void*)a.z,
                  (void*)a.dy_se, (void*)a.out_status, (void*)a.out_objective,
                  (void*)a.out_user_objective, (void*)a.out_iterations}) {
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
  int solve_mode = 0;  // device warp dense-solve mode: 0 plain / 1 big
  if (warp_mode) { const int v = std::atoi(warp_env); solve_mode = (v >= 2) ? 1 : 0; }
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_warp_solve_mode, &solve_mode, sizeof(int)));
  const auto  _k0 = std::chrono::steady_clock::now();
  if (warp_mode) {
    gpu_phase2_kernel<true><<<1, 32>>>(d_args);
  } else {
    gpu_phase2_kernel<false><<<1, 1>>>(d_args);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  const double _kms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - _k0).count();
  std::printf("[GPUKERNEL] warp=%d mode=%d ms=%.3f\n", warp_mode ? 1 : 0, solve_mode, _kms);
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
  const double up_t0 = tic();
  std::vector<kernel_args> h_args(N);
  for (size_t b = 0; b < N; ++b) {
    const auto& it = items[b];
    h_args[b] = marshal_problem(*it.lp, *it.settings, *it.vstatus, *it.ft, *it.basic_list,
                                *it.nonbasic_list, *it.sol, *it.dy_se);
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
  int solve_mode = 0;  // 0 plain / 1 big
  if (warp_mode) { const int v = std::atoi(warp_env); solve_mode = (v >= 2) ? 1 : 0; }
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(g_warp_solve_mode, &solve_mode, sizeof(int)));
  cudaEvent_t ev_start, ev_stop;
  RAFT_CUDA_TRY(cudaEventCreate(&ev_start));
  RAFT_CUDA_TRY(cudaEventCreate(&ev_stop));
  RAFT_CUDA_TRY(cudaEventRecord(ev_start));
  if (warp_mode) {
    gpu_phase2_kernel<true><<<static_cast<unsigned>(N), 32>>>(d_args);
  } else {
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

  for (size_t b = 0; b < N; ++b) { free_kernel_args(h_args[b]); }
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

}  // namespace cuopt::linear_programming::dual_simplex
