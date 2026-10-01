# Replay thetas through R (theta = D^-1 (d - R d) asc, e - R^T e desc): no serial chain.
f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
s=open(f).read()
n0=s.index("// Product-form replay through M (see fission_update_block)")
n1=s.index("__device__ __forceinline__ void fission_unstage")
new='''// Product-form replay with the thetas in closed form. With M[k][j] = col(2k+1).col(2j) (j < k), the
// ascending replay solves (D + M) theta = d and the descending one (D + M^T) theta = d, D = diag(mu),
// d the dots of the input. R (strictly lower, kept per seat, see the FTRAN tail) gives
//   asc:  theta = D^-1 (d - R d)        desc: theta = e - R^T e,  e = D^-1 d.
// The monolith's drop of |theta| <= tol before later thetas use it is not reproduced; zero thetas skip
// their axpy. scratch: 2K doubles of shared memory.
__device__ __forceinline__ void fission_tloop_m(const gpu_basis_update_t& ft, f_t* x, bool asc, f_t tol,
                                                const f_t* R, int kc, f_t* scratch, size_t)
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
    if (lane == 0) { d[k] = asc ? part : part / mu[k]; }
  }
  __syncthreads();
  if (asc) {
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
  for (i_t k = warp; k < K; k += nw) {
    const f_t t = th[k];
    if (t == 0.0) { continue; }
    const i_t c = asc ? 2 * k : 2 * k + 1;
    for (i_t p = cs[c] + lane; p < cs[c + 1]; p += 32) { atomicAdd(&x[ri[p]], -t * vx[p]); }
  }
  __syncthreads();
}

'''
s=s[:n0]+new+s[n1:]
# FTRAN tail: R row instead of M row
i=s.index("  // Row K = num_updates_ of M for the coming update")
j=s.index("    } else if (threadIdx.x == 0) {\n      st.f_mrow_k = -1;", i)
newtail='''  // Row K = num_updates_ of R for the coming update. With w_j = M[K][j] / mu_j, M[K][j] = UTsol.col(2j):
  //   R[K][i] = w_i - sum_{i<j<K} w_j R[j][i]   (R[j][i] at i * kc + j). acc is zero; R region free.
  {
    const i_t K = st.ft.num_updates_;
    if (st.f_mrows == K && K < a.fm_kcap) {
      const auto& UT = st.UTsol_sparse;
      const i_t nu   = static_cast<i_t>(UT.i.size());
      f_t* w         = R;
      for (i_t p = threadIdx.x; p < nu; p += blockDim.x) { acc[UT.i[p]] = UT.x[p]; }
      __syncthreads();
      const i_t* cs = st.ft.S_.col_start.data();
      const i_t* ri = st.ft.S_.i.data();
      const f_t* vx = st.ft.S_.x.data();
      const f_t* mu = st.ft.mu_values_.data();
      const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
      for (i_t jq = warp; jq < K; jq += blockDim.x >> 5) {
        f_t part = 0.0;
        for (i_t p = cs[2 * jq] + lane; p < cs[2 * jq + 1]; p += 32) { part += vx[p] * acc[ri[p]]; }
        part = warp_reduce_add(part);
        if (lane == 0) { w[jq] = part / mu[jq]; }
      }
      __syncthreads();
      for (i_t p = threadIdx.x; p < nu; p += blockDim.x) { acc[UT.i[p]] = 0.0; }
      f_t* Rm = a.fm;
      const size_t kc = static_cast<size_t>(a.fm_kcap);
      for (i_t iq = warp; iq < K; iq += blockDim.x >> 5) {
        f_t part = 0.0;
        for (i_t jq = iq + 1 + lane; jq < K; jq += 32) { part += w[jq] * Rm[iq * kc + jq]; }
        part = warp_reduce_add(part);
        if (lane == 0) { Rm[iq * kc + K] = w[iq] - part; }
      }
      __syncthreads();
      if (threadIdx.x == 0) { st.f_mrow_k = K; }
'''
s=s[:i]+newtail+s[j:]
open(f,'w').write(s); print('ok')
