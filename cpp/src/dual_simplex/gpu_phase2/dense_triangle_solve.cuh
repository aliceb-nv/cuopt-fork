// dense_triangle_solve.cuh — pure device cores for the dense L/U triangular solves.
//
// Single source of truth shared by the production GPU dual-simplex kernel (gpu_phase2.cu) and the
// isolated microbenchmark (bench_dense_solve.cu), so the benchmark measures exactly the production
// arithmetic. Raw-pointer signatures (templated on index/value type); NO coupling to gpu_csc_view,
// warp_solve_config, or the profiling globals — those stay in gpu_phase2.cu as thin wrappers.
//
// Include at file scope (outside any namespace) in each translation unit; RDC is off, so a header
// is how the two .cu files share one definition without a cross-TU device call.
#pragma once

namespace dense_triangle_solve {

// ---- single-thread reference solves ----
// L/U are scatter (column) form; Lᵀ/Uᵀ are gather form. Arithmetic is unchanged, so results stay
// bit-for-bit identical.

template <typename i_t, typename f_t>
__device__ inline void lower_solve(const i_t* cs, const i_t* ri, const f_t* val, f_t* x, i_t n)
{
  for (i_t j = 0; j < n; ++j) {
    const i_t c0 = cs[j], c1 = cs[j + 1];
    if (x[j] != 0.0) {
      x[j] /= val[c0];
      const f_t xj = x[j];
      for (i_t p = c0 + 1; p < c1; ++p) { x[ri[p]] -= val[p] * xj; }
    }
  }
}

template <typename i_t, typename f_t>
__device__ inline void upper_solve(const i_t* cs, const i_t* ri, const f_t* val, f_t* x, i_t n)
{
  for (i_t j = n - 1; j >= 0; --j) {
    const i_t c0 = cs[j], ce = cs[j + 1] - 1;
    if (x[j] != 0.0) {
      x[j] /= val[ce];
      const f_t xj = x[j];
      for (i_t p = c0; p < ce; ++p) { x[ri[p]] -= val[p] * xj; }
    }
  }
}

template <typename i_t, typename f_t>
__device__ inline void lower_transpose_solve(const i_t* cs, const i_t* ri, const f_t* val, f_t* x,
                                             i_t n)
{
  for (i_t j = n - 1; j >= 0; --j) {
    f_t sum = 0.0;
    for (i_t p = cs[j] + 1; p < cs[j + 1]; ++p) { sum += val[p] * x[ri[p]]; }
    x[j] = (x[j] - sum) / val[cs[j]];
  }
}

template <typename i_t, typename f_t>
__device__ inline void upper_transpose_solve(const i_t* cs, const i_t* ri, const f_t* val, f_t* x,
                                             i_t n)
{
  for (i_t j = 0; j < n; ++j) {
    const i_t ce = cs[j + 1] - 1;
    f_t sum = 0.0;
    for (i_t p = cs[j]; p < ce; ++p) { sum += val[p] * x[ri[p]]; }
    x[j] = (x[j] - sum) / val[ce];
  }
}

// ---- warp (32-lane) wavefront gather solve core ----
// Processes the operator's precomputed elimination levels (lstart[nlev+1], corder[n]) one lane per
// column, __syncwarp between levels. diag_first / descending / incremental select the gather form:
//   L-solve  (over Lᵀ): diag_first=false descending=false incremental=true
//   U-solve  (over Uᵀ): diag_first=true  descending=true  incremental=true
//   Lᵀ-solve (over L):  diag_first=true  descending=false incremental=false
//   Uᵀ-solve (over U):  diag_first=false descending=false incremental=false
// Must be entered by all 32 lanes of the warp.
template <typename i_t, typename f_t>
__device__ inline void wavefront_gather_solve(const i_t* cs, const i_t* ri, const f_t* val, f_t* x,
                                              const i_t* lstart, const i_t* corder, int nlev,
                                              bool diag_first, bool descending, bool incremental)
{
  const int lane = threadIdx.x;
  for (int L = 0; L < nlev; ++L) {
    for (i_t t = lstart[L] + lane; t < lstart[L + 1]; t += 32) {
      const i_t j  = corder[t];
      const i_t c0 = cs[j], c1 = cs[j + 1];
      const i_t dp = diag_first ? c0 : (c1 - 1);
      const i_t lo = diag_first ? c0 + 1 : c0;
      const i_t hi = diag_first ? c1 : (c1 - 1);
      if (incremental) {
        f_t xi = x[j];
        if (!descending) { for (i_t p = lo; p < hi; ++p) { xi -= val[p] * x[ri[p]]; } }
        else             { for (i_t p = hi - 1; p >= lo; --p) { xi -= val[p] * x[ri[p]]; } }
        x[j] = xi / val[dp];
      } else {
        f_t sum = 0.0;
        if (!descending) { for (i_t p = lo; p < hi; ++p) { sum += val[p] * x[ri[p]]; } }
        else             { for (i_t p = hi - 1; p >= lo; --p) { sum += val[p] * x[ri[p]]; } }
        x[j] = (x[j] - sum) / val[dp];
      }
    }
    __syncwarp(0xffffffffu);
  }
}

// ---- big: wavefront + whole-warp cooperation on LARGE columns ----
// Same level schedule as wavefront_gather_solve, but columns are sorted large-first within each level and
// a per-32-column bitmask (bits[], lword[] base per level) marks the large ones. Per 32-column chunk the
// warp loads one word: word==0 -> fast path (one lane per column, identical arithmetic to the wavefront);
// word!=0 -> each set bit is a large column solved by the WHOLE warp (strided sum + shfl reduction), then
// the unset lanes solve their small column. Only large columns reorder the sum (not bit-exact there).
template <typename i_t, typename f_t>
__device__ inline void big_col_scalar(const i_t* cs, const i_t* ri, const f_t* val, f_t* x, i_t j,
                                      bool df, bool desc, bool inc)
{
  const i_t c0 = cs[j], c1 = cs[j + 1];
  const i_t dp = df ? c0 : (c1 - 1);
  const i_t lo = df ? c0 + 1 : c0;
  const i_t hi = df ? c1 : (c1 - 1);
  if (inc) {
    f_t xi = x[j];
    if (!desc) { for (i_t p = lo; p < hi; ++p) { xi -= val[p] * x[ri[p]]; } }
    else       { for (i_t p = hi - 1; p >= lo; --p) { xi -= val[p] * x[ri[p]]; } }
    x[j] = xi / val[dp];
  } else {
    f_t s = 0.0;
    if (!desc) { for (i_t p = lo; p < hi; ++p) { s += val[p] * x[ri[p]]; } }
    else       { for (i_t p = hi - 1; p >= lo; --p) { s += val[p] * x[ri[p]]; } }
    x[j] = (x[j] - s) / val[dp];
  }
}
template <typename i_t, typename f_t>
__device__ inline void big_col_warp(const i_t* cs, const i_t* ri, const f_t* val, f_t* x, i_t j,
                                    bool df, int lane)
{
  const i_t c0 = cs[j], c1 = cs[j + 1];
  const i_t dp = df ? c0 : (c1 - 1);
  const i_t lo = df ? c0 + 1 : c0;
  const i_t hi = df ? c1 : (c1 - 1);
  f_t s = 0.0;
  for (i_t p = lo + lane; p < hi; p += 32) { s += val[p] * x[ri[p]]; }
  for (int o = 16; o > 0; o >>= 1) { s += __shfl_down_sync(0xffffffffu, s, o); }
  if (lane == 0) { x[j] = (x[j] - s) / val[dp]; }
}
template <typename i_t, typename f_t>
__device__ inline void wavefront_big_solve(const i_t* cs, const i_t* ri, const f_t* val, f_t* x,
                                           const i_t* lstart, const i_t* corder, const i_t* lword,
                                           const unsigned* bits, int nlev, bool df, bool desc, bool inc)
{
  const int lane = threadIdx.x;
  for (int L = 0; L < nlev; ++L) {
    const i_t base = lstart[L], end = lstart[L + 1], wbase = lword[L];
    int c = 0;
    for (i_t t0 = base; t0 < end; t0 += 32, ++c) {
      const unsigned word = bits[wbase + c];
      if (word == 0u) {
        const i_t t = t0 + lane;
        if (t < end) { big_col_scalar(cs, ri, val, x, corder[t], df, desc, inc); }
      } else {
        unsigned mm = word;
        while (mm) {
          const int i = __ffs(mm) - 1;
          mm &= (mm - 1);
          big_col_warp(cs, ri, val, x, corder[t0 + i], df, lane);
        }
        const i_t t = t0 + lane;
        if (t < end && !((word >> lane) & 1u)) { big_col_scalar(cs, ri, val, x, corder[t], df, desc, inc); }
      }
    }
    __syncwarp(0xffffffffu);
  }
}

}  // namespace dense_triangle_solve
