# BTRAN/FTRAN (atomic): one m-vector + one 32 KB region (replay staging, then lists), 2 blocks/SM.
# The chunked replay falls back to global memory for an update larger than half the stage.
import sys
root=sys.argv[1]
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
f=root+'/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
patch(f,[
("""  __shared__ int rng[2][2];  // replay positions [s0, s1) held by each buffer""",
"""  __shared__ int rng[2][2];  // replay positions [s0, s1) held by each buffer
  __shared__ int big[2];     // 1: buffer holds one update replayed from global memory"""),
("""    if (t0 == 0) { rng[b][0] = s0, rng[b][1] = s0; }
    bar();
    for (i_t s1 = s0 + 1 + t0; s1 <= num && s1 <= s0 + nt; s1 += nt) {
      const i_t ka = asc ? s0 : num - s1;
      const i_t kb = asc ? s1 : num - s0;
      if (fission_chunk_bytes(cs, ka, kb) <= half) { atomicMax(&rng[b][1], s1); }
    }
    bar();
    if (t0 == 0 && rng[b][1] == s0 && s0 < num) { __trap(); }  // one update larger than half the stage
  };""","""    if (t0 == 0) { rng[b][0] = s0, rng[b][1] = s0, big[b] = 0; }
    bar();
    for (i_t s1 = s0 + 1 + t0; s1 <= num && s1 <= s0 + nt; s1 += nt) {
      const i_t ka = asc ? s0 : num - s1;
      const i_t kb = asc ? s1 : num - s0;
      if (fission_chunk_bytes(cs, ka, kb) <= half) { atomicMax(&rng[b][1], s1); }
    }
    bar();
    if (t0 == 0 && rng[b][1] == s0 && s0 < num) { rng[b][1] = s0 + 1, big[b] = 1; }
    bar();
  };"""),
("""    const i_t s0 = rng[b][0], s1 = rng[b][1];
    if (s0 >= s1) { return; }
    const i_t ka = asc ? s0 : num - s1, kb = asc ? s1 : num - s0;
    const i_t E0 = cs[2 * ka], E = cs[2 * kb] - E0, n = kb - ka;""","""    const i_t s0 = rng[b][0], s1 = rng[b][1];
    if (s0 >= s1 || big[b]) { return; }
    const i_t ka = asc ? s0 : num - s1, kb = asc ? s1 : num - s0;
    const i_t E0 = cs[2 * ka], E = cs[2 * kb] - E0, n = kb - ka;"""),
("""  auto replay = [&](int b) {
    const int lane = threadIdx.x;
    const i_t s0 = rng[b][0], s1 = rng[b][1];""","""  auto replay = [&](int b) {
    const int lane = threadIdx.x;
    const i_t s0 = rng[b][0], s1 = rng[b][1];
    if (big[b]) {
      const i_t k    = asc ? s0 : num - 1 - s0;
      const i_t dcol = asc ? 2 * k + 1 : 2 * k;
      const i_t acol = asc ? 2 * k : 2 * k + 1;
      f_t partial    = 0.0;
      for (i_t p = cs[dcol] + lane; p < cs[dcol + 1]; p += 32) { partial += gvx[p] * x[gri[p]]; }
      const f_t theta = warp_reduce_add(partial) / gmu[k];
      if (fabs(theta) > tol) {
        for (i_t p = cs[acol] + lane; p < cs[acol + 1]; p += 32) { x[gri[p]] -= theta * gvx[p]; }
      }
      __syncwarp(0xffffffffu);
      return;
    }"""),
# BTRAN
("""  f_t* w   = fsh;
  f_t* acc = fsh + m;
  f_t* lx  = fsh + 2 * m;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { w[k] = 0.0, acc[k] = 0.0; }""","""  f_t* w   = fsh;
  f_t* R   = fsh + m;  // replay staging, then the lists
  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { w[k] = 0.0; }"""),
("""  fission_unstage(acc, fission_tloop_chunked(st.ft, w, false, 1e-13, acc, static_cast<size_t>(m) * sizeof(f_t)));
  FSTEP(S_B_TLOOP)
  const int nw = fission_compact_atomic(w, m, li, lx, kFissionCapA, &cnt);
  FSTEP(S_B_COMPACT)
  fission_scatter_atomic(scs, I.Lr_p, I.Lr_i, I.Lr_x, li, lx, nw, pre, acc);
  FSTEP(S_B_LSCATTER)
  auto& dy = st.delta_y_sparse;
  if (threadIdx.x == 0) { cnt = 0; }
  __syncthreads();
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    const f_t v = acc[k];""","""  fission_tloop_chunked(st.ft, w, false, 1e-13, R, kFissionStage);
  FSTEP(S_B_TLOOP)
  const int nw = fission_compact_atomic(w, m, li, lx, kFissionCapA, &cnt);
  FSTEP(S_B_COMPACT)
  fission_scatter_atomic(scs, I.Lr_p, I.Lr_i, I.Lr_x, li, lx, nw, pre, w);
  FSTEP(S_B_LSCATTER)
  auto& dy = st.delta_y_sparse;
  if (threadIdx.x == 0) { cnt = 0; }
  __syncthreads();
  for (int k = threadIdx.x; k < m; k += blockDim.x) {
    const f_t v = w[k];"""),
# FTRAN
("""  f_t* z   = fsh;
  f_t* acc = fsh + m;
  f_t* lx  = fsh + 2 * m;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { z[k] = 0.0, acc[k] = 0.0; }""","""  f_t* z   = fsh;
  f_t* R   = fsh + m;  // replay staging, then the lists
  f_t* lx  = R;
  i_t* li  = reinterpret_cast<i_t*>(lx + kFissionCapA);
  int* pre = reinterpret_cast<int*>(li + kFissionCapA);
  f_t* acc = z;      // the U step scatters back into z, zero after the compaction
  FSTEP_BEGIN
  for (int k = threadIdx.x; k < m; k += blockDim.x) { z[k] = 0.0; }"""),
("""    fission_unstage(acc, fission_tloop_chunked(st.ft, z, true, 1e-16, acc, static_cast<size_t>(m) * sizeof(f_t)));""",
"""    fission_tloop_chunked(st.ft, z, true, 1e-16, R, kFissionStage);"""),
("""        dx.i.set_size(cnt);
        dx.x.set_size(cnt);
      }
    } else {
      auto& v = st.ft.dense_scratch2_;
      for (int k = threadIdx.x; k < m; k += blockDim.x) { v[k] = acc[k]; }""","""        dx.i.set_size(cnt);
        dx.x.set_size(cnt);
      }
    } else {
      auto& v = st.ft.dense_scratch2_;
      for (int k = threadIdx.x; k < m; k += blockDim.x) { v[k] = acc[k], acc[k] = 0.0; }"""),
("""constexpr int kFissionCapA = 1536;""","""constexpr int kFissionCapA = 1536;
constexpr size_t kFissionStage = 32768;  // CHOSEN: 2 blocks/SM at m = 5236 (8m + 32 KB + static)
static_assert(kFissionStage >= kFissionCapA * (sizeof(f_t) + 2 * sizeof(i_t)), "lists live in the stage"); """),
])
g=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
patch(g,[("""            : 2 * static_cast<size_t>(H->rows) * sizeof(f_t) + kFissionCapA * (sizeof(f_t) + 2 * sizeof(i_t));""",
"""            : static_cast<size_t>(H->rows) * sizeof(f_t) + kFissionStage;""")])
print('ok')
