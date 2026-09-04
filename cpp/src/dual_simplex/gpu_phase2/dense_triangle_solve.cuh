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

template <typename i_t, typename ri_t, typename f_t>
__device__ inline void lower_solve(const i_t* cs, const ri_t* ri, const f_t* val, f_t* x, i_t n)
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

template <typename i_t, typename ri_t, typename f_t>
__device__ inline void upper_solve(const i_t* cs, const ri_t* ri, const f_t* val, f_t* x, i_t n)
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

template <typename i_t, typename ri_t, typename f_t>
__device__ inline void lower_transpose_solve(const i_t* cs, const ri_t* ri, const f_t* val, f_t* x,
                                             i_t n)
{
  for (i_t j = n - 1; j >= 0; --j) {
    f_t sum = 0.0;
    for (i_t p = cs[j] + 1; p < cs[j + 1]; ++p) { sum += val[p] * x[ri[p]]; }
    x[j] = (x[j] - sum) / val[cs[j]];
  }
}

template <typename i_t, typename ri_t, typename f_t>
__device__ inline void upper_transpose_solve(const i_t* cs, const ri_t* ri, const f_t* val, f_t* x,
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
// column, __syncwarp between levels. Three flags select the gather form, and there are only FOUR
// legal combinations — one per operator:
//   L-solve  (over Lᵀ): diag_first=false descending=false incremental=true
//   U-solve  (over Uᵀ): diag_first=true  descending=true  incremental=true
//   Lᵀ-solve (over L):  diag_first=true  descending=false incremental=false
//   Uᵀ-solve (over U):  diag_first=false descending=false incremental=false
// Must be entered by all 32 lanes of the warp.
//
// The flags are TEMPLATE parameters, not runtime bools. As runtime bools ptxas had to re-evaluate
// them for every column of every level and keep BOTH inner-loop directions live, which also blocked
// specializing/unrolling either one. As template parameters each operator compiles to straight-line
// code with its own inner loop and no branch. The fission `Direct` path knows its operator as a
// literal at every call site, so the whole chain down to here is compile-time.
template <bool DiagFirst, bool Descending, bool Incremental, typename i_t, typename ri_t, typename f_t>
__device__ inline void wavefront_gather_solve_t(const i_t* cs, const ri_t* ri, const f_t* val, f_t* x,
                                                const i_t* lstart, const i_t* corder, int nlev)
{
  const int lane = threadIdx.x;
  for (int L = 0; L < nlev; ++L) {
    for (i_t t = lstart[L] + lane; t < lstart[L + 1]; t += 32) {
      const i_t j  = corder[t];
      const i_t c0 = cs[j], c1 = cs[j + 1];
      const i_t dp = DiagFirst ? c0 : (c1 - 1);
      const i_t lo = DiagFirst ? c0 + 1 : c0;
      const i_t hi = DiagFirst ? c1 : (c1 - 1);
      const f_t diagonal_reciprocal = f_t{1.0} / val[dp];
      if constexpr (Incremental) {
        f_t xi = x[j];
        if constexpr (!Descending) { for (i_t p = lo; p < hi; ++p) { xi -= val[p] * x[ri[p]]; } }
        else                       { for (i_t p = hi - 1; p >= lo; --p) { xi -= val[p] * x[ri[p]]; } }
        x[j] = xi * diagonal_reciprocal;
      } else {
        f_t sum = 0.0;
        if constexpr (!Descending) { for (i_t p = lo; p < hi; ++p) { sum += val[p] * x[ri[p]]; } }
        else                       { for (i_t p = hi - 1; p >= lo; --p) { sum += val[p] * x[ri[p]]; } }
        x[j] = (x[j] - sum) * diagonal_reciprocal;
      }
    }
    __syncwarp(0xffffffffu);
  }
}

// The four operators as named entry points. Flag triples MUST match the df/dsc/inc tables in
// gpu_phase2.cu's kOpDiagFirst/kOpDescending/kOpIncremental (which warp_build_config also uses).
#define DTS_WF_OP(name, DF, DESC, INC)                                                            \
  template <typename i_t, typename ri_t, typename f_t>                                                           \
  __device__ inline void name(const i_t* cs, const ri_t* ri, const f_t* val, f_t* x,               \
                              const i_t* lstart, const i_t* corder, int nlev)                     \
  {                                                                                               \
    wavefront_gather_solve_t<DF, DESC, INC>(cs, ri, val, x, lstart, corder, nlev);                 \
  }
DTS_WF_OP(wavefront_L_solve,  false, false, true)   // L-solve  over Lᵀ
DTS_WF_OP(wavefront_U_solve,  true,  true,  true)   // U-solve  over Uᵀ
DTS_WF_OP(wavefront_LT_solve, true,  false, false)  // Lᵀ-solve over L
DTS_WF_OP(wavefront_UT_solve, false, false, false)  // Uᵀ-solve over U
#undef DTS_WF_OP

// Compile-time operator dispatch. `Op` mirrors warp_solve_op {L,U,Lᵀ,Uᵀ} = {0,1,2,3} and is a
// literal at every fission call site, so this collapses to one direct call — no branch at all.
template <int Op, typename i_t, typename ri_t, typename f_t>
__device__ inline void wavefront_solve_op(const i_t* cs, const ri_t* ri, const f_t* val, f_t* x,
                                          const i_t* lstart, const i_t* corder, int nlev)
{
  static_assert(Op >= 0 && Op < 4, "Op must be one of {L,U,Lt,Ut} = {0,1,2,3}");
  if constexpr      (Op == 0) { wavefront_L_solve (cs, ri, val, x, lstart, corder, nlev); }
  else if constexpr (Op == 1) { wavefront_U_solve (cs, ri, val, x, lstart, corder, nlev); }
  else if constexpr (Op == 2) { wavefront_LT_solve(cs, ri, val, x, lstart, corder, nlev); }
  else                        { wavefront_UT_solve(cs, ri, val, x, lstart, corder, nlev); }
}

// There is deliberately NO runtime entry point of any kind here — neither runtime FLAGS nor a
// runtime OP. Both production callers resolve the operator at COMPILE time:
//   * fission  : the op is a literal at every call site -> warp_dispatch<Direct, Op> -> wavefront_solve_op<Op>
//   * monolith : ONE switch on cfg.op in warp_collective_solve -> mono_solve_arm<Op> -> wavefront_solve_op<Op>
// so nothing below the dispatch level ever branches. The microbenchmark, which sweeps ops at
// runtime, keeps its own switch in benchmarks/.../sptrsv_methods.cuh rather than putting a runtime
// path back into this header.

// NOTE: the whole-warp "big column" path (wavefront_big_solve + big_col_scalar/big_col_warp)
// was REMOVED 2026-08-07 — measured net-negative, see FISSION_CHALLENGES.md §19. The
// microbenchmark keeps its own self-contained `ours_wave_big` for research sweeps.

}  // namespace dense_triangle_solve
