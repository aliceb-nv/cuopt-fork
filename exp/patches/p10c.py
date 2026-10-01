f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'; s=open(f).read()
i=s.index("  // Thetas, 32 at a time: warp 0 solves the 32x32 diagonal block of M (shared) by shuffles, then")
j=s.index("  const long long mc2 = clock64();")
new='''  // Thetas: chunks of M's columns staged in shared memory by the whole block (coalesced), then
  // warp 0 runs the sequential steps with d held in registers (k = lane + 32 q).
  f_t* imu = th + K;                 // 1 / mu
  f_t* Ms  = imu + K;                // staged columns, Ms[c * K + k]
  for (i_t k = threadIdx.x; k < K; k += blockDim.x) { imu[k] = 1.0 / mu[k]; }
  constexpr int kQ = 8;              // K <= 256
  const i_t cmax = max(i_t{1}, (static_cast<i_t>(stage_doubles) - 3 * K) / K);
  f_t r[kQ];
  if (warp == 0) {
#pragma unroll
    for (int q = 0; q < kQ; ++q) { r[q] = lane + 32 * q < K ? d[lane + 32 * q] : 0.0; }
  }
  for (i_t c0 = 0; c0 < K; c0 += cmax) {
    const i_t c1 = min(K, c0 + cmax);
    __syncthreads();
    const i_t nc = c1 - c0;
    for (i_t e = threadIdx.x; e < nc * K; e += blockDim.x) {
      if (asc) {
        const i_t c = e / K, k = e % K, jj = c0 + c;
        Ms[c * K + k] = k > jj ? M[static_cast<size_t>(jj) * kc + k] : 0.0;
      } else {
        const i_t k = e / nc, c = e % nc, jj = K - 1 - (c0 + c);
        Ms[c * K + k] = k < jj ? M[static_cast<size_t>(k) * kc + jj] : 0.0;
      }
    }
    __syncthreads();
    if (warp == 0) {
      for (i_t c = 0; c < nc; ++c) {
        const i_t jj = asc ? c0 + c : K - 1 - (c0 + c);
        const int qo = static_cast<int>(jj >> 5);
        f_t rj = 0.0;
#pragma unroll
        for (int q = 0; q < kQ; ++q) { rj = q == qo ? r[q] : rj; }
        f_t t = __shfl_sync(0xffffffffu, rj, static_cast<int>(jj & 31)) * imu[jj];
        if (fabs(t) <= tol) { t = 0.0; }
        if (lane == (jj & 31)) { th[jj] = t; }
        const f_t* col = Ms + c * K;
#pragma unroll
        for (int q = 0; q < kQ; ++q) {
          const i_t k = lane + 32 * q;
          if (k < K) { r[q] -= col[k] * t; }
        }
      }
    }
  }
  __syncthreads();
'''
s=s[:i]+new+s[j:]
a="""__device__ __forceinline__ void fission_tloop_m(const gpu_basis_update_t& ft, f_t* x, bool asc, f_t tol,
                                                const f_t* M, int kc, f_t* scratch)"""
assert s.count(a)==1
s=s.replace(a,"""__device__ __forceinline__ void fission_tloop_m(const gpu_basis_update_t& ft, f_t* x, bool asc, f_t tol,
                                                const f_t* M, int kc, f_t* scratch, size_t stage_doubles)""")
for v in ["w, false, 1e-13","z, true, 1e-16"]:
    a="fission_tloop_m(st.ft, %s, a.fm, a.fm_kcap, R);"%v
    assert s.count(a)==1
    s=s.replace(a,"fission_tloop_m(st.ft, %s, a.fm, a.fm_kcap, R, kFissionStage / sizeof(f_t));"%v)
open(f,'w').write(s); print('ok')
