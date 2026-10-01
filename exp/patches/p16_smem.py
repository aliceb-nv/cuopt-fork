f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
s=open(f).read()
old_lists="""  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);"""
new_lists="""  f_t* lx  = st.ft.sparse_scratch_.x.data();  // lists in per-seat global scratch (L1)
  i_t* li  = st.ft.sparse_scratch_.i.data();
  int* pre = st.ft.ei_scratch_.i.data();"""
assert s.count(old_lists)==2, s.count(old_lists)
s=s.replace(old_lists,new_lists)
a="""  f_t* crat = fsh + m;
  i_t* cpos = reinterpret_cast<i_t*>(crat + nn);"""
assert s.count(a)==1, s.count(a)
s=s.replace(a,"""  f_t* crat = st.ft.ei_scratch_.x.data();  // candidates in per-seat global scratch (L1)
  i_t* cpos = st.ft.ei_scratch_.i.data();""")
a="""constexpr size_t kFissionStage = 32768;  // CHOSEN: 2 blocks/SM at m = 5236 (8m + 32 KB + static)
static_assert(kFissionStage >= kFissionCapA * (sizeof(f_t) + 2 * sizeof(i_t)), "lists live in the stage"); """
assert s.count(a)==1
s=s.replace(a,"""constexpr size_t kFissionStage = 8192;  // CHOSEN: replay scratch (2K doubles) and chunked-replay stage""")
open(f,'w').write(s)
t=open(g).read()
a="""      const size_t smem_c = static_cast<size_t>(H->rows) * sizeof(f_t) +
                            static_cast<size_t>(H->cols - H->rows) * (sizeof(f_t) + sizeof(i_t));"""
assert t.count(a)==1
t=t.replace(a,"""      const size_t smem_c = static_cast<size_t>(H->rows) * sizeof(f_t);""")
open(g,'w').write(t); print('ok')
