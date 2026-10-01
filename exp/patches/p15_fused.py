import re
f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
s=open(f).read()
def rep(a,b,cnt=1):
    global s
    assert s.count(a)==cnt,(s.count(a),a[:70]); s=s.replace(a,b)
# --- update body templated on block size
rep("""__device__ __forceinline__ void fission_primal_sync() { asm volatile("bar.sync 1, %0;" ::"n"(kUpdPrimal) : "memory"); }""",
"""template <int T>
__device__ __forceinline__ void fission_primal_sync() { asm volatile("bar.sync 1, %0;" ::"n"(T - 32) : "memory"); }""")
rep("""__device__ __forceinline__ void fission_update_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r);""",
"""template <int T>
__device__ __forceinline__ void fission_update_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r);""")
rep("""  fission_update_block_body(args, n_problems, r);""","""  fission_update_block_body<kUpdThreads>(args, n_problems, r);""")
b0=s.index("__device__ __forceinline__ void fission_update_block_body(kernel_args* args, i_t n_problems, fission_utrace_t& r)\n{")
b1=s.index("\n}\n", b0)+3
body=s[b0:b1]
body="template <int T>\n"+body
body=body.replace("typename fission_upd_scan_t::TempStorage","typename cub::BlockScan<int, T>::TempStorage")
body=body.replace("fission_upd_scan_t(scan_tmp)","cub::BlockScan<int, T>(scan_tmp)")
body=body.replace("fission_primal_sync()","fission_primal_sync<T>()")
body=body.replace("kUpdPrimal","(T - 32)").replace("kUpdThreads","T")
s=s[:b0]+body+s[b1:]
# --- price body
rep("""__global__ void __launch_bounds__(32) fission_price(kernel_args* args, i_t n_problems)
{
  kernel_args& a = args[blockIdx.x];""","""__device__ __forceinline__ void fission_price_body(kernel_args* args, i_t n_problems)
{
  kernel_args& a = args[blockIdx.x];""")
i=s.index("__device__ __forceinline__ void fission_price_body(kernel_args* args, i_t n_problems)")
j=s.index("\n}\n", i)+3
s=s[:j]+"""
__global__ void __launch_bounds__(32) fission_price(kernel_args* args, i_t n_problems)
{
  fission_price_body(args, n_problems);
}
"""+s[j:]
# --- yield body
rep("""__global__ void __launch_bounds__(32) fission_yield(kernel_args* args, i_t n_problems)
{
  kernel_args& a = args[blockIdx.x];""","""__device__ __forceinline__ void fission_yield_body(kernel_args* args, i_t n_problems)
{
  kernel_args& a = args[blockIdx.x];""")
i=s.index("__device__ __forceinline__ void fission_yield_body(kernel_args* args, i_t n_problems)")
j=s.index("\n}\n", i)+3
s=s[:j]+"""
__global__ void __launch_bounds__(32) fission_yield(kernel_args* args, i_t n_problems)
{
  fission_yield_body(args, n_problems);
}

// Fused rounds (CUOPT_EXP_FUSED): one block per seat runs `rounds` rounds of price, BTRAN, choose,
// FTRAN and update with block barriers only, so seats do not wait for each other between phases.
__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_fused(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems, int rounds, int yielding)
{
  __shared__ fission_utrace_t r;
  if (blockIdx.x >= n_problems) { return; }
  for (int k = 0; k < rounds; ++k) {
    if (args[blockIdx.x].done != 0 || args[blockIdx.x].slot->f_state == F_DONE) { return; }
    if (yielding && threadIdx.x < 32) { fission_yield_body(args, n_problems); }
    __syncthreads();
    if (threadIdx.x < 32) { fission_price_body(args, n_problems); }
    __syncthreads();
    fission_btran_atomic_body(args, inv, n_problems);
    __syncthreads();
    fission_choose_block_body(args, n_problems, r);
    __syncthreads();
    fission_ftran_atomic_body(args, inv, n_problems);
    __syncthreads();
    fission_update_block_body<kFissionThreads>(args, n_problems, r);
    __syncthreads();
  }
}
"""+s[j:]
# --- btran body
rep("""__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_btran_atomic(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{""","""__device__ __forceinline__ void fission_btran_atomic_body(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{""")
i=s.index("__device__ __forceinline__ void fission_btran_atomic_body(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)\n{")
j=s.index("\n}\n", i)+3
s=s[:j]+"""
__global__ void __launch_bounds__(kFissionThreads, 2)
  fission_btran_atomic(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  fission_btran_atomic_body(args, inv, n_problems);
}
"""+s[j:]
open(f,'w').write(s)
t=open(g).read()
a="""      int live0 = -1;
      bool yielding = false;
      for (;;) {
        for (int k = 0; k < gpu_fission_rounds(); ++k) {"""
assert t.count(a)==1
t=t.replace(a,"""      int live0 = -1;
      bool yielding = false;
      static const bool fused = std::getenv("CUOPT_EXP_FUSED") != nullptr && !det;
      const size_t smem_f     = std::max(smem, smem_c);
      if (fused) {
        RAFT_CUDA_TRY(cudaFuncSetAttribute(fission_fused, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(smem_f)));
      }
      for (;;) {
        if (fused) {
          fission_fused<<<Ni, kFissionThreads, smem_f, cs>>>(H->d_args, finv, Ni, gpu_fission_rounds(), yielding ? 1 : 0);
        }
        for (int k = 0; k < (fused ? 0 : gpu_fission_rounds()); ++k) {""")
open(g,'w').write(t); print('ok')
