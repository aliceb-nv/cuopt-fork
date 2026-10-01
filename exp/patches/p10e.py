f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
patch(g,[
("""  i_t f_mrows{-1}; // rows of the fission replay matrix M held (== num_updates_ when usable)""",
"""  i_t f_mrows{-1}; // rows of the fission replay matrix M held (== num_updates_ when usable)
  i_t f_mrow_k{-1}; // row of M written by the last FTRAN (from UTsol), pending the update"""),
("""          fission_update_block<<<Ni, kUpdThreads, static_cast<size_t>(H->rows) * sizeof(f_t), cs>>>(H->d_args, Ni);""",
"""          fission_update_block<<<Ni, kUpdThreads, 0, cs>>>(H->d_args, Ni);"""),
("""      RAFT_CUDA_TRY(cudaFuncSetAttribute(fission_update_block, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(static_cast<size_t>(H->rows) * sizeof(f_t))));
""",""),
])
s=open(f).read()
i=s.index("    // Row K of M (the update just appended): M[K][j] = col(2K+1) . col(2j), j < K.")
j=s.index("  {\n    // fission_infeas_update over the block, same list order.")
s=s[:i]+"""    // Row K of M was written by FTRAN from UTsol (the column 2K+1 just appended).
    const i_t K = ft.num_updates_ - 1;
    if (tid == 0) { st.f_mrows = (K >= 0 && st.f_mrows == K && st.f_mrow_k == K) ? K + 1 : -1; }
  }
"""+s[j:]
a="""  if (threadIdx.x == 0) { st.f_state = F_UPDATE; }"""
assert s.count(a)==1
s=s.replace(a,"""  // Row K = num_updates_ of M for the coming update: M[K][j] = UTsol . col(2j), j < K. acc is zero.
  {
    const i_t K = st.ft.num_updates_;
    if (st.f_mrows == K && K < a.fm_kcap) {
      const auto& UT = st.UTsol_sparse;
      const i_t nu   = static_cast<i_t>(UT.i.size());
      for (i_t p = threadIdx.x; p < nu; p += blockDim.x) { acc[UT.i[p]] = UT.x[p]; }
      __syncthreads();
      const i_t* cs = st.ft.S_.col_start.data();
      const i_t* ri = st.ft.S_.i.data();
      const f_t* vx = st.ft.S_.x.data();
      const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
      for (i_t jq = warp; jq < K; jq += blockDim.x >> 5) {
        f_t part = 0.0;
        for (i_t p = cs[2 * jq] + lane; p < cs[2 * jq + 1]; p += 32) { part += vx[p] * acc[ri[p]]; }
        part = warp_reduce_add(part);
        if (lane == 0) { a.fm[static_cast<size_t>(jq) * a.fm_kcap + K] = part; }
      }
      __syncthreads();
      for (i_t p = threadIdx.x; p < nu; p += blockDim.x) { acc[UT.i[p]] = 0.0; }
      if (threadIdx.x == 0) { st.f_mrow_k = K; }
    } else if (threadIdx.x == 0) {
      st.f_mrow_k = -1;
    }
  }
  if (threadIdx.x == 0) { st.f_state = F_UPDATE; }""")
open(f,'w').write(s); print('ok')
