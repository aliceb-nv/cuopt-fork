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

#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/sparse_vector.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <utilities/macros.cuh>

#include <cuda/std/limits>
#include <cuda/std/span>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {
namespace gpu_phase2_detail {

using i_t = int;
using f_t = double;

template <typename T>
using gpu_span = cuda::std::span<T>;

// Set if any device-side allocation fails; the driver bails with NUMERICAL.
__device__ int g_gpu_alloc_failed;

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
  __device__ gpu_vector(gpu_vector&& o) : data_(o.data_), size_(o.size_), cap_(o.cap_)
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

  __device__ void grow_to(size_t newcap)
  {
    if (newcap <= cap_) { return; }
    T* p = static_cast<T*>(malloc(newcap * sizeof(T)));
    if (p == nullptr) {
      printf("[gpu_phase2] device malloc FAILED for %llu bytes (cap %llu -> %llu)\n",
             (unsigned long long)(newcap * sizeof(T)),
             (unsigned long long)cap_,
             (unsigned long long)newcap);
      g_gpu_alloc_failed = 1;
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
    i_t nz = i.size();
    gpu_vector<i_t> i_perm(nz);
    for (i_t k = 0; k < nz; ++k) {
      i_perm[k] = p[i[k]];
    }
    i = static_cast<gpu_vector<i_t>&&>(i_perm);
  }

  __device__ void sort()
  {
    if (i.size() == 1) { return; }
    // If the number of nonzeros is large, use an O(n) bucket sort.
    if (i.size() > 0.3 * n) {
      gpu_vector<f_t> bucket(n, 0.0);
      const i_t nz = i.size();
      for (i_t k = 0; k < nz; ++k) {
        bucket[i[k]] = x[k];
      }
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
      // n log n sort: sort a permutation by index, then gather.
      const i_t nz = i.size();
      gpu_vector<i_t> i_sorted(nz);
      gpu_vector<f_t> x_sorted(nz);
      gpu_vector<i_t> perm(nz);
      for (i_t k = 0; k < nz; ++k) {
        perm[k] = k;
      }
      i_t* iunsorted = i.data();
      gpu_sort(perm.begin(), perm.end(), [iunsorted] __device__(i_t a, i_t b) {
        return iunsorted[a] < iunsorted[b];
      });
      for (i_t k = 0; k < nz; ++k) {
        i_sorted[k] = i[perm[k]];
        x_sorted[k] = x[perm[k]];
      }
      i = static_cast<gpu_vector<i_t>&&>(i_sorted);
      x = static_cast<gpu_vector<f_t>&&>(x_sorted);
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

__device__ i_t lower_triangular_solve(const gpu_csc_view& L, f_t* x, f_t& work_estimate)
{
  i_t n = L.n;
  for (i_t j = 0; j < n; ++j) {
    i_t col_start = L.col_start[j];
    i_t col_end   = L.col_start[j + 1];
    if (x[j] != 0.0) {
      x[j] /= L.x[col_start];
      const f_t x_j = x[j];
      for (i_t p = col_start + 1; p < col_end; ++p) {
        x[L.i[p]] -= L.x[p] * x_j;
      }
    }
  }
  work_estimate += 3 * n + 3 * L.col_start[n];
  return 0;
}

__device__ i_t lower_triangular_transpose_solve(const gpu_csc_view& L, f_t* x, f_t& work_estimate)
{
  const i_t n = L.n;
  for (i_t j = n - 1; j >= 0; --j) {
    const i_t col_start = L.col_start[j] + 1;
    const i_t col_end   = L.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      x[j] -= L.x[p] * x[L.i[p]];
    }
    x[j] /= L.x[L.col_start[j]];
  }
  work_estimate += 6 * n + 4 * L.col_start[n];
  return 0;
}

__device__ i_t upper_triangular_solve(const gpu_csc_view& U, f_t* x, f_t& work_estimate)
{
  const i_t n = U.n;
  for (i_t j = n - 1; j >= 0; --j) {
    const i_t col_start = U.col_start[j];
    const i_t col_end   = U.col_start[j + 1] - 1;
    if (x[j] != 0.0) {
      x[j] /= U.x[col_end];
      const f_t x_j = x[j];
      for (i_t p = col_start; p < col_end; ++p) {
        x[U.i[p]] -= U.x[p] * x_j;
      }
      work_estimate += 3 * (col_end - col_start) + 3;
    }
  }
  work_estimate += 3 * n;
  return 0;
}

__device__ i_t upper_triangular_transpose_solve(const gpu_csc_view& U, f_t* x, f_t& work_estimate)
{
  const i_t n = U.n;
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = U.col_start[j];
    const i_t col_end   = U.col_start[j + 1] - 1;
    for (i_t p = col_start; p < col_end; ++p) {
      x[j] -= U.x[p] * x[U.i[p]];
    }
    x[j] /= U.x[col_end];
  }
  work_estimate += 4 * n + 4 * U.col_start[n];
  return 0;
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
  // Statistics that drive the hypersparse-vs-dense path choice.
  mutable i_t num_calls_L_{0}, num_calls_U_{0}, num_calls_L_transpose_{0}, num_calls_U_transpose_{0};
  mutable f_t sum_L_{0.0}, sum_U_{0.0}, sum_L_transpose_{0.0}, sum_U_transpose_{0.0};
  f_t hypersparse_threshold_{0.05};
  mutable f_t work_estimate_{0.0};

  __device__ gpu_basis_update_mpf() {}
  gpu_basis_update_mpf(const gpu_basis_update_mpf&)            = delete;
  gpu_basis_update_mpf& operator=(const gpu_basis_update_mpf&) = delete;
  __device__ ~gpu_basis_update_mpf()                          = default;

  __device__ i_t num_updates() const { return num_updates_; }

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
  __device__ i_t u_transpose_solve(f_t* rhs) const
  {
    upper_triangular_transpose_solve(U0_, rhs, work_estimate_);
    return 0;
  }
  __device__ i_t u_transpose_solve(gpu_sparse_vector& rhs) const
  {
    i_t top = sparse_triangle_solve<true>(
      rhs, nullptr, xi_workspace_.data(), U0_transpose_, x_workspace_.data(), work_estimate_);
    solve_to_sparse_vector(top, rhs);
    return 0;
  }

  __device__ i_t l_transpose_solve(f_t* rhs) const
  {
    const f_t zero_tol = 1e-13;
    for (i_t k = num_updates_ - 1; k >= 0; --k) {
      const i_t u_col = 2 * k;
      const i_t v_col = 2 * k + 1;
      const f_t mu    = mu_values_[k];
      f_t dot         = dot_product(u_col, rhs);
      const f_t theta = dot / mu;
      if (fabs(theta) > zero_tol) { add_sparse_column(S_, v_col, -theta, rhs); }
    }
    work_estimate_ += 2 * num_updates_;
    lower_triangular_transpose_solve(L0_, rhs, work_estimate_);
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

    gpu_sparse_vector b(m, nz);
    work_estimate_ += nz;
    gather_into_sparse_vector(nz, b);
    i_t top = sparse_triangle_solve<false>(
      b, nullptr, xi_workspace_.data(), L0_transpose_, x_workspace_.data(), work_estimate_);
    solve_to_sparse_vector(top, rhs);
    return 0;
  }

  __device__ i_t u_solve(f_t* rhs) const
  {
    upper_triangular_solve(U0_, rhs, work_estimate_);
    return 0;
  }
  __device__ i_t u_solve(gpu_sparse_vector& rhs) const
  {
    i_t top = sparse_triangle_solve<false>(
      rhs, nullptr, xi_workspace_.data(), U0_, x_workspace_.data(), work_estimate_);
    solve_to_sparse_vector(top, rhs);
    return 0;
  }

  __device__ i_t l_solve(f_t* rhs) const
  {
    lower_triangular_solve(L0_, rhs, work_estimate_);
    const f_t zero_tol = 1e-16;  // Any higher and pilot_ja fails
    for (i_t k = 0; k < num_updates_; ++k) {
      const f_t mu    = mu_values_[k];
      const i_t u_col = 2 * k;
      const i_t v_col = 2 * k + 1;
      f_t dot         = dot_product(v_col, rhs);
      const f_t theta = dot / mu;
      if (fabs(theta) > zero_tol) { add_sparse_column(S_, u_col, -theta, rhs); }
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
  __device__ i_t b_solve(const gpu_vector<f_t>& rhs, gpu_vector<f_t>& solution) const
  {
    permute_vector(row_permutation_.data(), rhs.data(), solution.data(), L0_.m);
    work_estimate_ += 3 * rhs.size();
    l_solve(solution.data());
    u_solve(solution.data());
    return 0;
  }

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
      gpu_vector<f_t> solution_dense;
      solution.to_dense(solution_dense);
      work_estimate_ += solution_dense.size();
      l_solve(solution_dense.data());
      solution.from_dense(solution_dense);
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
      gpu_vector<f_t> solution_dense;
      solution.to_dense(solution_dense);
      work_estimate_ += solution_dense.size();
      u_solve(solution_dense.data());
      solution.from_dense(solution_dense);
      work_estimate_ += solution_dense.size();
    }
    sum_U_ += static_cast<f_t>(solution.i.size()) / rhs_size;
    return 0;
  }

  __device__ i_t b_solve(const gpu_sparse_vector& rhs, gpu_sparse_vector& solution) const
  {
    gpu_sparse_vector Lsol(1, 0);
    return b_solve(rhs, solution, Lsol, false);
  }

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
      gpu_vector<f_t> solution_dense;
      rhs.to_dense(solution_dense);
      u_transpose_solve(solution_dense.data());
      solution.from_dense(solution_dense);
    }
    UTsol = solution;
    sum_U_transpose_ += static_cast<f_t>(solution.i.size()) / input_size;

    const f_t rhs_size = static_cast<f_t>(solution.i.size());
    estimate_solution_density(
      rhs_size, sum_L_transpose_, num_calls_L_transpose_, use_hypersparse);
    if (use_hypersparse) {
      l_transpose_solve(solution);
    } else {
      gpu_vector<f_t> solution_dense;
      solution.to_dense(solution_dense);
      l_transpose_solve(solution_dense.data());
      solution.from_dense(solution_dense);
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

    etilde.sort();
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

__device__ void compute_reduced_cost_update(const gpu_lp& lp,
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
      ft.b_transpose_solve(sparse_ei, sparse_dy, UTsol_scratch);
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
  ft.b_solve(rhs, xB_workspace);

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

__device__ void initialize_steepest_edge_norms_from_slack_basis(const i_t* basic_list,
                                                                const i_t* nonbasic_list,
                                                                i_t m,
                                                                i_t n,
                                                                f_t* delta_y_steepest_edge)
{
  for (i_t k = 0; k < m; ++k) {
    const i_t j              = basic_list[k];
    delta_y_steepest_edge[j] = 1.0;
  }
  const i_t n_minus_m = n - m;
  for (i_t k = 0; k < n_minus_m; ++k) {
    const i_t j              = nonbasic_list[k];
    delta_y_steepest_edge[j] = 1e-4;
  }
}

__device__ i_t phase2_ratio_test(const gpu_lp& lp,
                                 const gpu_settings& settings,
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

__device__ void compute_delta_y(const gpu_basis_update_mpf& ft,
                                i_t basic_leaving_index,
                                i_t direction,
                                gpu_sparse_vector& delta_y_sparse,
                                gpu_sparse_vector& UTsol_sparse)
{
  const i_t m = delta_y_sparse.n;
  gpu_sparse_vector ei_sparse(m, 1);
  ei_sparse.i[0] = basic_leaving_index;
  ei_sparse.x[0] = -direction;
  ft.b_transpose_solve(ei_sparse, delta_y_sparse, UTsol_sparse);

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
  ft.b_solve(rhs_sparse, scaled_delta_xB_sparse, utilde_sparse);
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
    ft.b_solve(rhs, scaled_delta_xB);
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
  ft.b_solve(delta_y_sparse, v_sparse);
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
__device__ dual::status_t gpu_simplified_dual_phase2_device(const gpu_lp& lp,
                                                            const gpu_settings& settings,
                                                            variable_status_t* vstatus,
                                                            gpu_basis_update_mpf& ft,
                                                            i_t* basic_list,
                                                            i_t* nonbasic_list,
                                                            gpu_solution& sol,
                                                            f_t* delta_y_steepest_edge,
                                                            bool dy_se_empty,
                                                            i_t iter_limit_override,
                                                            bool gpu_log)
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
  compute_primal_variables(
    ft, lp.rhs, lp.A, basic_list, nonbasic_list, settings.tight_tol, x, xB_workspace, work);

  if (dy_se_empty) {
    // Exact steepest-edge init (matches the CPU simplified's unconditional exact path).
    for (i_t k = 0; k < n; ++k) { delta_y_steepest_edge[k] = -1.0; }
    initialize_steepest_edge_norms(lp, basic_list, ft, delta_y_steepest_edge, work);
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

  i_t iter_limit = min(500000, max(20000, 20 * (m + n)));
  if (iter_limit_override > 0) { iter_limit = iter_limit_override; }
  if (gpu_log) {
    printf("[GMP] start m=%d n=%d p_inf=%016llx iter_limit=%d\n",
           m,
           n,
           (unsigned long long)__double_as_longlong(primal_inf),
           iter_limit);
  }
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
    i_t direction     = 0;
    i_t basic_leaving = -1;
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
    compute_delta_y(ft, basic_leaving, direction, delta_y_sparse, UTsol_sparse);
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
      compute_reduced_cost_update(lp, basic_list, nonbasic_list, delta_y.data(), leaving, direction,
                                  delta_z_mark.data(), delta_z_indices, delta_z.data(), work);
    }

    // 4. Simple ratio test.
    f_t step_length;
    i_t nonbasic_entering = -1;
    const i_t entering    = phase2_ratio_test(
      lp, settings, vstatus, nonbasic_list, z, delta_z.data(), step_length, nonbasic_entering);
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
    if (compute_delta_x(lp, ft, entering, leaving, basic_leaving, direction, basic_list, rhs_sparse,
                        delta_z.data(), x, utilde_sparse, scaled_dxB_sparse, delta_x.data(),
                        work) == -1) {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iter;
      return dual::status_t::NUMERICAL;
    }

    update_steepest_edge_norms(settings, basic_list, ft, direction, delta_y_sparse, dy_norm_sq,
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

    if (ft.update(utilde_sparse, UTsol_sparse, basic_leaving) == 1) {
      sol.objective      = compute_objective(lp, x);
      sol.user_objective = compute_user_objective(lp, sol.objective);
      sol.iterations     = iter;
      return dual::status_t::NUMERICAL;
    }

    compute_initial_primal_infeasibilities(
      lp, settings, basic_list, x, squared_infeas, infeas_idx, primal_inf);
    clear_delta_z(entering, leaving, delta_z_mark.data(), delta_z_indices, delta_z.data());

    if (gpu_log) {
      printf("[GMP] iter=%d leave=%d dir=%d enter=%d step=%016llx obj=%016llx p_inf=%016llx nInf=%d "
             "nupd=%d\n",
             iter,
             leaving,
             direction,
             entering,
             (unsigned long long)__double_as_longlong(step_length),
             (unsigned long long)__double_as_longlong(compute_objective(lp, x)),
             (unsigned long long)__double_as_longlong(primal_inf),
             (int)infeas_idx.size(),
             ft.num_updates());
    }

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
        if (gpu_log) {
          printf("[GMP] cutoff iter=%d obj=%016llx cut_off=%016llx\n",
                 iter,
                 (unsigned long long)__double_as_longlong(cutoff_obj),
                 (unsigned long long)__double_as_longlong(settings.cut_off));
        }
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
  i_t iter_limit_override;  // >0 forces a fixed iteration cap (for bitwise CPU comparison)
  i_t gpu_log;              // emit the per-step [CMP] trace
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

__global__ void gpu_phase2_kernel(kernel_args a)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
  g_gpu_alloc_failed = 0;

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

  dual::status_t status = gpu_simplified_dual_phase2_device(lp,
                                                            settings,
                                                            a.vstatus,
                                                            ft,
                                                            a.basic_list,
                                                            a.nonbasic_list,
                                                            sol,
                                                            a.dy_se,
                                                            a.dy_se_empty != 0,
                                                            a.iter_limit_override,
                                                            a.gpu_log != 0);

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

  // Raise device heap (for in-kernel gpu_vector malloc) and stack (deep call
  // chain). These are per-context limits and the heap size can ONLY be set before
  // the first malloc-using kernel launch, so set both exactly once per process.
  static bool device_limits_set = false;
  if (!device_limits_set) {
    device_limits_set = true;
    size_t free_b = 0, total_b = 0;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_b, &total_b));
    size_t heap = static_cast<size_t>(0.7 * static_cast<double>(free_b));
    const size_t cap = static_cast<size_t>(8) << 30;  // 8 GiB cap
    if (heap > cap) { heap = cap; }
    if (heap < (static_cast<size_t>(256) << 20)) { heap = static_cast<size_t>(256) << 20; }
    cudaError_t hp = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap);
    if (hp != cudaSuccess) {
      cudaGetLastError();
      std::fprintf(stderr,
                   "[gpu_phase2] warning: could not set device heap size: %s\n",
                   cudaGetErrorString(hp));
    }
    // The deep (non-recursive) call chain wants more than the ~1 KB default stack.
    cudaError_t stk = cudaDeviceSetLimit(cudaLimitStackSize, static_cast<size_t>(32) << 10);
    if (stk != cudaSuccess) {
      cudaGetLastError();
      std::fprintf(stderr,
                   "[gpu_phase2] warning: could not raise device stack size: %s\n",
                   cudaGetErrorString(stk));
    }
  }

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
  // Optional fixed iteration cap (must match the CPU's CUOPT_SIMPLIFIED_ITER_LIMIT).
  {
    const char* e         = std::getenv("CUOPT_SIMPLIFIED_ITER_LIMIT");
    a.iter_limit_override = (e != nullptr) ? std::atoi(e) : 0;
  }
  a.gpu_log = settings.log.log ? 1 : 0;

  a.out_status         = alloc_zeroed<int>(1);
  a.out_objective      = alloc_zeroed<double>(1);
  a.out_user_objective = alloc_zeroed<double>(1);
  a.out_iterations     = alloc_zeroed<int>(1);

  gpu_phase2_kernel<<<1, 1>>>(a);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

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

  // Free device buffers.
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

  return static_cast<dual::status_t>(h_status);
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

}  // namespace cuopt::linear_programming::dual_simplex
