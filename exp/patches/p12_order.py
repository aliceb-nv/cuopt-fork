f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
patch(g,[
("""  f_t* fm{nullptr};  // fission replay matrix M of this seat, fm_kcap^2 (column-major)""",
"""  f_t* fm{nullptr};  // fission replay matrix M of this seat, fm_kcap^2 (column-major)
  i_t fperm{0};      // fission BTRAN/FTRAN launches: block (this index) runs seat fperm"""),
("""          k_btran<<<Ni, kFissionThreads, smem, cs>>>(H->d_args, finv, Ni);""",
"""          fission_order<<<1, kFissionThreads, 0, cs>>>(H->d_args, Ni, F_BTRAN);
          k_btran<<<Ni, kFissionThreads, smem, cs>>>(H->d_args, finv, Ni);"""),
("""          k_ftran<<<Ni, kFissionThreads, smem, cs>>>(H->d_args, finv, Ni);""",
"""          fission_order<<<1, kFissionThreads, 0, cs>>>(H->d_args, Ni, F_FTRAN);
          k_ftran<<<Ni, kFissionThreads, smem, cs>>>(H->d_args, finv, Ni);"""),
])
s=open(f).read()
# BTRAN atomic
a="""  fission_btran_atomic(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  kernel_args& a = args[blockIdx.x];
  if (blockIdx.x >= n_problems || a.done != 0) { return; }"""
b="""  fission_btran_atomic(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  if (blockIdx.x >= n_problems) { return; }
  const i_t seat = args[blockIdx.x].fperm;
  kernel_args& a = args[seat];
  if (a.done != 0) { return; }"""
assert s.count(a)==1; s=s.replace(a,b)
i=s.index(b); j=s.index("const auto& I = inv[blockIdx.x];", i)
s=s[:j]+"const auto& I = inv[seat];"+s[j+len("const auto& I = inv[blockIdx.x];"):]
# FTRAN atomic body + wrapper
a="""  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_FTRAN) { return; }"""
assert s.count(a)==1
b="""  if (blockIdx.x >= n_problems) { return; }
  const i_t seat = args[blockIdx.x].fperm;
  kernel_args& a = args[seat];
  if (a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_state != F_FTRAN) { return; }"""
s=s.replace(a,b)
i=s.index(b); j=s.index("const auto& I = inv[blockIdx.x];", i)
s=s[:j]+"const auto& I = inv[seat];"+s[j+len("const auto& I = inv[blockIdx.x];"):]
a="""    r.seat = blockIdx.x;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r.sm));
    if (blockIdx.x < n_problems && args[blockIdx.x].done == 0 && args[blockIdx.x].slot->f_state == F_FTRAN) {
      r.live = 1;
      r.nzx  = args[blockIdx.x].slot->ft.num_updates_;
      r.unz  = static_cast<int>(args[blockIdx.x].slot->rhs_sparse.i.size());
      r.enz  = static_cast<int>(args[blockIdx.x].slot->delta_y_sparse.i.size());"""
assert s.count(a)==1
s=s.replace(a,"""    const i_t sb = blockIdx.x < n_problems ? args[blockIdx.x].fperm : 0;
    r.seat = sb;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r.sm));
    if (blockIdx.x < n_problems && args[sb].done == 0 && args[sb].slot->f_state == F_FTRAN) {
      r.live = 1;
      r.nzx  = args[sb].slot->ft.num_updates_;
      r.unz  = static_cast<int>(args[sb].slot->rhs_sparse.i.size());
      r.enz  = static_cast<int>(args[sb].slot->delta_y_sparse.i.size());""")
a="""__device__ __forceinline__ void fission_unstage(f_t* stage, size_t bytes)"""
assert s.count(a)==1
s=s.replace(a,"""// Block order for the next BTRAN/FTRAN launch: seats in `state` by descending update count (their
// replay length), then the rest, so the longest seats start in the first wave.
constexpr int kOrderItems = 4;  // up to 4096 seats; beyond that the identity order is kept
__global__ void __launch_bounds__(kFissionThreads) fission_order(kernel_args* args, i_t n_problems, int state)
{
  using sort_t = cub::BlockRadixSort<int, kFissionThreads, kOrderItems, int>;
  __shared__ typename sort_t::TempStorage tmp;
  if (n_problems > kFissionThreads * kOrderItems) {
    for (i_t b = threadIdx.x; b < n_problems; b += blockDim.x) { args[b].fperm = b; }
    return;
  }
  int key[kOrderItems], val[kOrderItems];
#pragma unroll
  for (int q = 0; q < kOrderItems; ++q) {
    const i_t b = threadIdx.x * kOrderItems + q;
    key[q]      = -2;
    val[q]      = b;
    if (b < n_problems) {
      const kernel_args& a = args[b];
      key[q] = (a.done == 0 && a.slot->f_state == state) ? a.slot->ft.num_updates_ : -1;
    }
  }
  sort_t(tmp).SortDescending(key, val);
#pragma unroll
  for (int q = 0; q < kOrderItems; ++q) {
    const i_t b = threadIdx.x * kOrderItems + q;
    if (b < n_problems) { args[b].fperm = val[q]; }
  }
}

"""+a)
open(f,'w').write(s); print('ok')
