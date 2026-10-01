f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
patch(f,[
("""  f_t* w   = fsh;
  f_t* R   = fsh + m;  // replay staging, then the lists
  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { w[k] = 0.0; }""","""  f_t* w   = st.ft.dense_scratch_.data();  // global, zero on entry; native float atomics
  f_t* R   = fsh;  // replay staging, then the lists
  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  FSTEP_BEGIN"""),
("""  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    const f_t v = w[k];
    if (v != 0.0) {
      const int q = atomicAdd(&cnt, 1);
      dy.i[q]     = k;
      dy.x[q]     = v;
    }
  }""","""  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    const f_t v = w[k];
    if (v != 0.0) {
      const int q = atomicAdd(&cnt, 1);
      dy.i[q]     = k;
      dy.x[q]     = v;
      w[k]        = 0.0;
    }
  }"""),
("""  f_t* z   = fsh;
  f_t* R   = fsh + m;  // replay staging, then the lists
  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  f_t* acc = z;      // the U step scatters back into z, zero after the compaction
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { z[k] = 0.0; }""","""  f_t* z   = st.ft.dense_scratch_.data();  // global, zero on entry; native float atomics
  f_t* R   = fsh;  // replay staging, then the lists
  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  f_t* acc = z;      // the U step scatters back into z, zero after the compaction
  FSTEP_BEGIN"""),
("""  if (threadIdx.x == 0) { a.slot->f_mrows = a.slot->ft.num_updates_ == 0 ? 0 : -1; }
  __syncwarp(0xffffffffu);""","""  if (threadIdx.x == 0) { a.slot->f_mrows = a.slot->ft.num_updates_ == 0 ? 0 : -1; }
  for (i_t k = threadIdx.x; k < a.m; k += 32) { a.slot->ft.dense_scratch_[k] = 0.0; }
  __syncwarp(0xffffffffu);"""),
])
patch(g,[("""            : static_cast<size_t>(H->rows) * sizeof(f_t) + kFissionStage;""","""            : kFissionStage;""")])
print('ok')
