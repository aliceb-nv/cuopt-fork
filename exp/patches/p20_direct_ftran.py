# CUOPT_EXP_FTRAN=direct: fission FTRAN through L0/U0 level-scheduled solves (no inverses in FTRAN).
f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
patch(g,[
("""  // Fission fresh seat: no reader of the device transposes or the level schedules.
  const bool lean = FissionSetup && !child_continue && a.refactor_resume == 0 && a.device_lu == 0;
  if constexpr (PersistentSeats) {
    if (!lean && !child_continue && a.transpose_on_device != 0) {""","""  // Fission fresh seat: x_B from the inverses. The transposes and level schedules are skipped unless
  // the direct FTRAN (CUOPT_EXP_FTRAN=direct) solves through them.
  const bool lean      = FissionSetup && !child_continue && a.refactor_resume == 0 && a.device_lu == 0;
  const bool skip_tri  = lean && c_ftran_direct == 0;
  if constexpr (PersistentSeats) {
    if (!skip_tri && !child_continue && a.transpose_on_device != 0) {"""),
("""  ft.path(reuse_levels ? P_LEVEL_REUSE : P_LEVEL_BUILD);
  if (!lean) {""","""  ft.path(reuse_levels ? P_LEVEL_REUSE : P_LEVEL_BUILD);
  if (!skip_tri) {"""),
("""__constant__ int c_warp_levels = 1;    // CUOPT_BNB_GPU_WARP_LEVELS""","""__constant__ int c_warp_levels = 1;    // CUOPT_BNB_GPU_WARP_LEVELS
__constant__ int c_ftran_direct = 0;   // CUOPT_EXP_FTRAN=direct: fission FTRAN through L0/U0"""),
("""      fission_presetup<<<Ni, kPresetupThreads, 0, cs>>>(H->d_args, Ni);""","""      {
        static const int direct = [] {
          const char* e = std::getenv("CUOPT_EXP_FTRAN");
          return (e != nullptr && std::string(e) == "direct") ? 1 : 0;
        }();
        RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(c_ftran_direct, &direct, sizeof(int), 0, cudaMemcpyHostToDevice, cs));
      }
      fission_presetup<<<Ni, kPresetupThreads, 0, cs>>>(H->d_args, Ni);"""),
])
s=open(f).read()
a="""__device__ __forceinline__ void fission_unstage(f_t* stage, size_t bytes)"""
assert s.count(a)==1
s=s.replace(a,"""// In-place level-scheduled triangular solve over the transposed factor (row j's entries in column j
// of `T`), schedule op `o` built by setup: level l's rows are co[ls[l-1] .. ls[l]); rows absent from
// the schedule have only a unit diagonal. One thread per row, a block barrier between levels.
__device__ __forceinline__ void fission_level_solve(const gpu_csc_view& T, const i_t* ls, const i_t* co,
                                                    int nlev, bool diag_first, f_t* x)
{
  const i_t* cs = T.col_start.data();
  const i_t* ri = T.i.data();
  const f_t* vx = T.x.data();
  for (int l = 1; l <= nlev; ++l) {
    for (i_t q = ls[l - 1] + threadIdx.x; q < ls[l]; q += blockDim.x) {
      const i_t j  = co[q] & ~kColNoop;
      const i_t c0 = cs[j], c1 = cs[j + 1];
      const i_t lo = diag_first ? c0 + 1 : c0, hi = diag_first ? c1 : c1 - 1;
      f_t v        = x[j];
      for (i_t p = lo; p < hi; ++p) { v -= vx[p] * x[ri[p]]; }
      x[j] = v / (diag_first ? vx[c0] : vx[c1 - 1]);
    }
    __syncthreads();
  }
}

"""+a)
i=s.index("fission_ftran_atomic_body(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)\n{")
def body_rep(a,b):
    global s
    j=s.index(a,i); s=s[:j]+b+s[j+len(a):]
body_rep("""    fission_scatter_atomic(scs, I.Lc_p, I.Lc_i, I.Lc_x, b.i.data(), b.x.data(), nb, pre, z);
    FSTEP(S_F_LSCATTER)""","""    const bool direct = c_ftran_direct != 0;
    if (direct) {
      const i_t* ip = st.ft.inverse_row_permutation_.data();
      for (int p = threadIdx.x; p < nb; p += blockDim.x) { z[ip[b.i[p]]] = b.x[p]; }
      __syncthreads();
      fission_level_solve(st.ft.L0_transpose_, st.lvl_ls[0].data(), st.lvl_co[0].data(), st.cfg_nlev[0], false, z);
    } else {
      fission_scatter_atomic(scs, I.Lc_p, I.Lc_i, I.Lc_x, b.i.data(), b.x.data(), nb, pre, z);
    }
    FSTEP(S_F_LSCATTER)""")
body_rep("""    const int nz = fission_compact_atomic(z, m, li, lx, kFissionCapA, &cnt);
    if (r == 0) {""","""    int nz = 0;
    if (direct) {
      // utilde from the dense L-space vector, which the U solve then overwrites in place.
      if (r == 0) {
        if (threadIdx.x == 0) { cnt = 0; }
        __syncthreads();
        auto& ut = st.utilde_sparse;
        for (int k = threadIdx.x; k < m; k += blockDim.x) {
          if (z[k] != 0.0) {
            const int q = atomicAdd(&cnt, 1);
            ut.i[q] = k, ut.x[q] = z[k];
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          ut.n = m;
          ut.i.set_size(cnt);
          ut.x.set_size(cnt);
        }
      }
    } else {
      nz = fission_compact_atomic(z, m, li, lx, kFissionCapA, &cnt);
    }
    if (!direct && r == 0) {""")
body_rep("""    fission_scatter_atomic(scs, I.Uc_p, I.Uc_i, I.Uc_x, li, lx, nz, pre, acc);
    FSTEP(S_F_USCATTER)""","""    if (direct) {
      fission_level_solve(st.ft.U0_transpose_, st.lvl_ls[1].data(), st.lvl_co[1].data(), st.cfg_nlev[1], true, acc);
    } else {
      fission_scatter_atomic(scs, I.Uc_p, I.Uc_i, I.Uc_x, li, lx, nz, pre, acc);
    }
    FSTEP(S_F_USCATTER)""")
open(f,'w').write(s); print('ok')
