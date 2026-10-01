# FTRAN (atomic): one block per (seat, right-hand side), grid 2N. The update phase accepts F_FTRAN.
f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
s=open(f).read()
n=s.index("__device__ __forceinline__ void fission_ftran_atomic_body(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)\n{")
e=s.index("\n}\n", n)+3
body=s[n:e]
def rep(a,b):
    global body
    assert body.count(a)==1,(body.count(a),a[:70]); body=body.replace(a,b)
rep("""  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];""","""  if (blockIdx.x >= 2 * n_problems) { return; }
  const i_t seat = blockIdx.x % n_problems;
  const int rr   = static_cast<int>(blockIdx.x / n_problems);  // 0: entering column, 1: delta_y
  kernel_args& a = args[seat];""")
rep("""  const auto& I = inv[blockIdx.x];""","""  const auto& I = inv[seat];""")
rep("""  for (int r = 0; r < 2; ++r) {""","""  for (int r = rr; r == rr; ++r) {""")
rep("""    if (st.f_mrows == K && K < a.fm_kcap) {""","""    if (rr == 1 && st.f_mrows == K && K < a.fm_kcap) {""")
rep("""    } else if (threadIdx.x == 0) {
      st.f_mrow_k = -1;""","""    } else if (rr == 1 && threadIdx.x == 0) {
      st.f_mrow_k = -1;""")
rep("""  if (threadIdx.x == 0) { st.f_state = F_UPDATE; }""","""  // The state stays F_FTRAN: the other right-hand side's block may not have started yet.""")
s=s[:n]+body+s[e:]
a="""    r.seat = blockIdx.x;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r.sm));
    if (blockIdx.x < n_problems && args[blockIdx.x].done == 0 && args[blockIdx.x].slot->f_state == F_FTRAN) {
      r.live = 1;
      r.nzx  = args[blockIdx.x].slot->ft.num_updates_;
      r.unz  = static_cast<int>(args[blockIdx.x].slot->rhs_sparse.i.size());
      r.enz  = static_cast<int>(args[blockIdx.x].slot->delta_y_sparse.i.size());"""
assert s.count(a)==1
s=s.replace(a,"""    const i_t sb = blockIdx.x % n_problems;
    r.seat = sb;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r.sm));
    if (blockIdx.x < 2 * n_problems && args[sb].done == 0 && args[sb].slot->f_state == F_FTRAN) {
      r.live = 1;
      r.nzx  = args[sb].slot->ft.num_updates_;
      r.unz  = static_cast<int>(args[sb].slot->rhs_sparse.i.size());
      r.enz  = static_cast<int>(args[sb].slot->delta_y_sparse.i.size());""")
# block update accepts F_FTRAN (atomic FTRAN leaves it) or F_UPDATE (deterministic FTRAN)
k=s.index("__device__ __forceinline__ void fission_update_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r)\n{")
j=s.index("  if (st.f_state != F_UPDATE) { return; }", k)
s=s[:j]+"  if (st.f_state != F_UPDATE && st.f_state != F_FTRAN) { return; }"+s[j+len("  if (st.f_state != F_UPDATE) { return; }"):]
open(f,'w').write(s)
t=open(g).read()
a="""          k_ftran<<<Ni, kFissionThreads, smem, cs>>>(H->d_args, finv, Ni);"""
assert t.count(a)==1
t=t.replace(a,"""          k_ftran<<<det ? Ni : 2 * Ni, kFissionThreads, smem, cs>>>(H->d_args, finv, Ni);""")
open(g,'w').write(t); print('ok')
