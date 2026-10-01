f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
s=open(f).read()
def rep(a,b):
    global s
    assert s.count(a)==1,(s.count(a),a[:70]); s=s.replace(a,b)
rep("""  f_t* dyd  = fsh;
  f_t* crat = fsh + m;
  i_t* cpos = reinterpret_cast<i_t*>(crat + nn);""","""  f_t* dyd  = fsh;
  f_t* crat = fsh + m;          // candidate ratio
  f_t* caz  = crat + nn;        // candidate |z_j|
  i_t* cpos = reinterpret_cast<i_t*>(caz + nn);  // candidate nonbasic position k
  i_t* cj   = cpos + nn;        // candidate column j""")
rep("""      cpos[ncand + pos] = k;
      crat[ncand + pos] = ratio;""","""      cpos[ncand + pos] = k;
      crat[ncand + pos] = ratio;
      cj[ncand + pos]   = j;
      caz[ncand + pos]  = fabs(z[j]);""")
rep("""    for (int c = 0; c < ncand; ++c) {
      const i_t k  = cpos[c];
      const i_t j  = a.nonbasic_list[k];
      const f_t rc = crat[c];
      if (rc < min_val) {
        min_val = rc, entering = j, step = rc, nonbasic_entering = k;
      } else if (rc < min_val + ztol && fabs(z[j]) > fabs(z[entering])) {
        min_val = rc, entering = j, step = rc, nonbasic_entering = k;
      }
    }""","""    // Shared memory only: |z| of the current choice is carried in a register.
    f_t az_e = 0.0;
    for (int c = 0; c < ncand; ++c) {
      const f_t rc = crat[c];
      const f_t az = caz[c];
      if (rc < min_val || (rc < min_val + ztol && az > az_e)) {
        min_val = rc, step = rc, az_e = az, entering = cj[c], nonbasic_entering = cpos[c];
      }
    }""")
open(f,'w').write(s)
t=open(g).read()
a="""      const size_t smem_c = static_cast<size_t>(H->rows) * sizeof(f_t) +
                            static_cast<size_t>(H->cols - H->rows) * (sizeof(f_t) + sizeof(i_t));"""
assert t.count(a)==1
t=t.replace(a,"""      const size_t smem_c = static_cast<size_t>(H->rows) * sizeof(f_t) +
                            static_cast<size_t>(H->cols - H->rows) * 2 * (sizeof(f_t) + sizeof(i_t));""")
open(g,'w').write(t); print('ok')
