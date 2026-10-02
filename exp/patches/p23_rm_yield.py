# Remove the fission straggler cut (CUOPT_BNB_GPU_FISSION_YIELD) entirely.
R='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/'
f=R+'cpp/src/dual_simplex/gpu_phase2/fission.cuh'; g=R+'cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
s=open(f).read()
i=s.index("// Straggler cut: every live seat at an iteration boundary")
j=s.index("// Fused rounds (CUOPT_EXP_FUSED)", i)
s=s[:i]+s[j:]
def rep(a,b):
    global s
    assert s.count(a)==1,(s.count(a),a[:70]); s=s.replace(a,b)
rep("  fission_fused(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems, int rounds, int yielding)",
    "  fission_fused(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems, int rounds)")
rep("""    if (yielding && threadIdx.x < 32) { fission_yield_body(args, n_problems); }
    __syncthreads();
""","")
i=s.index("inline double gpu_fission_yield()")
j=s.index("\n}\n", i)+3
s=s[:i]+s[j:].lstrip('\n')
open(f,'w').write(s)
s=open(g).read()
rep("""      int live0 = -1;
      bool yielding = false;
""","")
rep("fission_fused<<<Ni, kFissionThreads, smem_f, cs>>>(H->d_args, finv, Ni, gpu_fission_rounds(), yielding ? 1 : 0);",
    "fission_fused<<<Ni, kFissionThreads, smem_f, cs>>>(H->d_args, finv, Ni, gpu_fission_rounds());")
rep("""          if (yielding) { fission_yield<<<Ni, 32, 0, cs>>>(H->d_args, Ni); }
""","")
rep("""        if (live0 < 0) { live0 = live; }
        if (!yielding && live <= gpu_fission_yield() * live0) { yielding = true; }
""","")
open(g,'w').write(s)
a=R+'exp/acc.sh'; t=open(a).read(); t=t.replace(" CUOPT_BNB_GPU_FISSION_YIELD=0",""); open(a,'w').write(t)
print('ok')
