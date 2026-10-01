# Product-form replay as dots + small triangular solve + axpys, with M[k][j] = col(2k+1).col(2j)
# kept per seat (one row appended per update). Falls back to the chunked replay when M is partial.
import sys
root=sys.argv[1]
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
g=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
f=root+'/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
patch(g,[
("""  i_t f_lean{0};   // fission fresh seat: x_B, objective and infeasibilities come from fission_xb""",
"""  i_t f_lean{0};   // fission fresh seat: x_B, objective and infeasibilities come from fission_xb
  i_t f_mrows{-1}; // rows of the fission replay matrix M held (== num_updates_ when usable)"""),
("""  i_t mpf_update_cap;""","""  i_t mpf_update_cap;
  f_t* fm{nullptr};  // fission replay matrix M of this seat, fm_kcap^2 (column-major)
  i_t fm_kcap{0};"""),
("""  std::optional<rmm::device_buffer> d_finv;""","""  std::optional<rmm::device_buffer> d_finv;
  std::optional<rmm::device_buffer> d_fm;  // fission replay matrices, one per seat"""),
("""  RAFT_CUDA_TRY(cudaMemcpyAsync(H->d_args,
                                H->h_args.data(),
                                N * sizeof(kernel_args),""","""  if (gpu_fission()) {
    const size_t kc = static_cast<size_t>(mpf_update_cap());
    if (!H->d_fm) { H->d_fm.emplace(N * kc * kc * sizeof(f_t), H->stream); }
    for (size_t b = 0; b < N; ++b) {
      H->h_args[b].fm      = static_cast<f_t*>(H->d_fm->data()) + b * kc * kc;
      H->h_args[b].fm_kcap = static_cast<i_t>(kc);
    }
  }
  RAFT_CUDA_TRY(cudaMemcpyAsync(H->d_args,
                                H->h_args.data(),
                                N * sizeof(kernel_args),"""),
("""          fission_update_block<<<Ni, kUpdThreads, 0, cs>>>(H->d_args, Ni);""",
"""          fission_update_block<<<Ni, kUpdThreads, static_cast<size_t>(H->rows) * sizeof(f_t), cs>>>(H->d_args, Ni);"""),
])
s=open(g).read()
a="""      fission_setup_kernel<<<Ni, 32, 0, cs>>>(H->d_args, Ni);"""
assert s.count(a)==1
s=s.replace(a,"""      RAFT_CUDA_TRY(cudaFuncSetAttribute(fission_update_block, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(static_cast<size_t>(H->rows) * sizeof(f_t))));
"""+a)
open(g,'w').write(s)
patch(f,[
# setup: M state
("""  if (a.slot->f_lean != 0) {  // fission_xb finishes this seat""","""  if (threadIdx.x == 0) { a.slot->f_mrows = a.slot->ft.num_updates_ == 0 ? 0 : -1; }
  __syncwarp(0xffffffffu);
  if (a.slot->f_lean != 0) {  // fission_xb finishes this seat"""),
# replay with M
("""__device__ __forceinline__ void fission_unstage(f_t* stage, size_t bytes)""","""// Product-form replay through M (see fission_update_block): all K dots against the input, the
// thetas by a triangular solve on warp 0, then all axpys (shared float atomics). Same tolerance
// rule: a theta with |theta| <= tol is dropped. scratch: 2K doubles of shared memory.
__device__ __forceinline__ void fission_tloop_m(const gpu_basis_update_t& ft, f_t* x, bool asc, f_t tol,
                                                const f_t* M, int kc, f_t* scratch)
{
  const i_t K = ft.num_updates_;
  if (K == 0) { return; }
  const i_t* cs = ft.S_.col_start.data();
  const i_t* ri = ft.S_.i.data();
  const f_t* vx = ft.S_.x.data();
  const f_t* mu = ft.mu_values_.data();
  f_t* d        = scratch;
  f_t* th       = scratch + K;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (i_t k = warp; k < K; k += nw) {
    const i_t c = asc ? 2 * k + 1 : 2 * k;
    f_t part    = 0.0;
    for (i_t p = cs[c] + lane; p < cs[c + 1]; p += 32) { part += vx[p] * x[ri[p]]; }
    part = warp_reduce_add(part);
    if (lane == 0) { d[k] = part; }
  }
  __syncthreads();
  if (warp == 0) {
    constexpr int kQ = 8;  // K <= 256 = 8 per lane
    if (asc) {
      for (i_t j = 0; j < K; ++j) {
        f_t t = d[j] / mu[j];
        if (fabs(t) <= tol) { t = 0.0; }
        if (lane == 0) { th[j] = t; }
        if (t != 0.0) {
          const f_t* col = M + static_cast<size_t>(j) * kc;
#pragma unroll
          for (int q = 0; q < kQ; ++q) {
            const i_t k = j + 1 + lane + 32 * q;
            if (k < K) { d[k] -= col[k] * t; }
          }
        }
        __syncwarp(0xffffffffu);
      }
    } else {
      for (i_t j = K - 1; j >= 0; --j) {
        f_t t = d[j] / mu[j];
        if (fabs(t) <= tol) { t = 0.0; }
        if (lane == 0) { th[j] = t; }
        if (t != 0.0) {
#pragma unroll
          for (int q = 0; q < kQ; ++q) {
            const i_t k = lane + 32 * q;
            if (k < j) { d[k] -= M[static_cast<size_t>(k) * kc + j] * t; }
          }
        }
        __syncwarp(0xffffffffu);
      }
    }
  }
  __syncthreads();
  for (i_t k = warp; k < K; k += nw) {
    const f_t t = th[k];
    if (t == 0.0) { continue; }
    const i_t c = asc ? 2 * k : 2 * k + 1;
    for (i_t p = cs[c] + lane; p < cs[c + 1]; p += 32) { atomicAdd(&x[ri[p]], -t * vx[p]); }
  }
  __syncthreads();
}

__device__ __forceinline__ void fission_unstage(f_t* stage, size_t bytes)"""),
("""  fission_tloop_chunked(st.ft, w, false, 1e-13, R, kFissionStage);
  FSTEP(S_B_TLOOP)""","""  if (st.f_mrows == st.ft.num_updates_) {
    fission_tloop_m(st.ft, w, false, 1e-13, a.fm, a.fm_kcap, R);
  } else {
    fission_tloop_chunked(st.ft, w, false, 1e-13, R, kFissionStage);
  }
  FSTEP(S_B_TLOOP)"""),
("""    fission_tloop_chunked(st.ft, z, true, 1e-16, R, kFissionStage);""","""    if (st.f_mrows == st.ft.num_updates_) {
      fission_tloop_m(st.ft, z, true, 1e-16, a.fm, a.fm_kcap, R);
    } else {
      fission_tloop_chunked(st.ft, z, true, 1e-16, R, kFissionStage);
    }"""),
# update: append row K of M
("""  FSTEP(S_U_REFCHECK)
  {
    // fission_infeas_update over the block, same list order.""","""  FSTEP(S_U_REFCHECK)
  {
    // Row K of M (the update just appended): M[K][j] = col(2K+1) . col(2j), j < K.
    const i_t K = ft.num_updates_ - 1;
    if (st.f_mrows == K && K >= 0) {
      extern __shared__ f_t ush[];
      const i_t* cs = ft.S_.col_start.data();
      const i_t* ri = ft.S_.i.data();
      const f_t* vx = ft.S_.x.data();
      for (i_t k = tid; k < a.m; k += kUpdThreads) { ush[k] = 0.0; }
      __syncthreads();
      for (i_t p = cs[2 * K + 1] + tid; p < cs[2 * K + 2]; p += kUpdThreads) { ush[ri[p]] = vx[p]; }
      __syncthreads();
      const int lane = tid & 31, warp = tid >> 5;
      for (i_t j = warp; j < K; j += kUpdThreads / 32) {
        f_t part = 0.0;
        for (i_t p = cs[2 * j] + lane; p < cs[2 * j + 1]; p += 32) { part += vx[p] * ush[ri[p]]; }
        part = warp_reduce_add(part);
        if (lane == 0) { a.fm[static_cast<size_t>(j) * a.fm_kcap + K] = part; }
      }
      __syncthreads();
      if (tid == 0) { st.f_mrows = K + 1; }
    } else if (tid == 0) {
      st.f_mrows = -1;
    }
  }
  {
    // fission_infeas_update over the block, same list order."""),
])
print('ok')
