# Fission: explicit inverses built on the device in every relaunch (default); CUOPT_EXP_INV=library
# keeps the harness's start-up library.
f='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
g='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
t='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/benchmarks/linear_programming/cuopt/test_netlib.cu'
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
s=open(f).read()
anchor="inline bool gpu_fission()\n{"
assert s.count(anchor)==1
s=s.replace(anchor,r'''// ---- Explicit inverses built on the device every relaunch (see benchmarks/.../inv_bench.cu).
// L0^{-1} P and U0^{-1} of every seat, column (CSC) and row (CSR) forms, from the staged factors.
// Capacities: the per-seat maxima measured on lrsa-pool-120s (--inv-bench); exceeding one fails loudly.
constexpr long long kInvCapL = 19926, kInvCapU = 329296;
struct finv_build_t {
  int N{0}, m{0}, warps{0}, grid{0};
  long long cap[2]{kInvCapL, kInvCapU};
  int *cur[2], *cstart[2], *clen[2], *si[2], *cp[2], *ci[2], *rp[2], *rj[2];
  double *sx[2], *cv[2], *rv[2];
  int *tasks, *ntasks, *next, *err;
  double* ws_y;
  unsigned long long* ws_b;
  int* ws_l;
};

__device__ __forceinline__ void finv_factor(const kernel_args& a, int f, const i_t*& cp, const i_t*& ri, const f_t*& cx)
{
  cp = f == 0 ? a.L0_col_start : a.U0_col_start;
  ri = f == 0 ? a.L0_i : a.U0_i;
  cx = f == 0 ? a.L0_x : a.U0_x;
}

// Thread per (seat, factor, column): trivial columns written directly, the rest listed.
__global__ void finv_trivial(const kernel_args* args, finv_build_t b)
{
  const long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const int m = b.m;
  if (t >= 2LL * b.N * m) { return; }
  const int k = static_cast<int>(t % m), f = static_cast<int>((t / m) % 2), s = static_cast<int>(t / (2LL * m));
  const kernel_args& a = args[s];
  if (a.done != 0) { return; }
  const i_t *cp, *ri;
  const f_t* cx;
  finv_factor(a, f, cp, ri, cx);
  if (cp[k + 1] - cp[k] != 1) {
    b.tasks[atomicAdd(b.ntasks, 1)] = (s * 2 + f) * m + k;
    return;
  }
  const int pos = atomicAdd(&b.cur[f][s], 1);
  if (pos >= b.cap[f]) {
    atomicExch(b.err, 1);
    return;
  }
  const size_t base     = static_cast<size_t>(s) * b.cap[f];
  b.si[f][base + pos]   = k;
  b.sx[f][base + pos]   = 1.0 / cx[cp[k]];
  const size_t c        = static_cast<size_t>(s) * m + (f == 0 ? a.row_perm[k] : k);
  b.cstart[f][c]        = pos;
  b.clen[f][c]          = 1;
}

// Warp per non-trivial column: T y = e_k with rows finalized in index order (ascending for L0,
// descending for U0), discovered rows in a bitset; per-warp workspace left zero after each task.
__global__ void __launch_bounds__(256) finv_columns(const kernel_args* args, finv_build_t b)
{
  const int lane = threadIdx.x & 31;
  const int gw   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int m = b.m, W64 = (m + 63) / 64;
  double* y = b.ws_y + static_cast<size_t>(gw) * m;
  unsigned long long* bits = b.ws_b + static_cast<size_t>(gw) * W64;
  int* lst = b.ws_l + static_cast<size_t>(gw) * m;
  const int nt = *b.ntasks;
  for (;;) {
    int t = 0;
    if (lane == 0) { t = atomicAdd(b.next, 1); }
    t = __shfl_sync(0xffffffffu, t, 0);
    if (t >= nt) { return; }
    const int code = b.tasks[t];
    const int k = code % m, f = (code / m) % 2, s = code / (2 * m);
    const bool asc = f == 0;
    const kernel_args& a = args[s];
    const i_t *cp, *ri;
    const f_t* cx;
    finv_factor(a, f, cp, ri, cx);
    if (lane == 0) {
      y[k] = 1.0;
      bits[k >> 6] |= 1ull << (k & 63);
    }
    __syncwarp();
    int cnt = 0, cur = k;
    for (;;) {
      int r = -1;
      const int w0 = cur >> 6;
      for (int base = 0; r < 0; base += 32) {
        const int w = asc ? w0 + base + lane : w0 - base - lane;
        unsigned long long word = 0;
        if (w >= 0 && w < W64) {
          word = bits[w];
          if (w == w0) {
            const int bb = cur & 63;
            word &= asc ? (~0ull << bb) : (bb == 63 ? ~0ull : ((2ull << bb) - 1));
          }
        }
        const unsigned hit = __ballot_sync(0xffffffffu, word != 0);
        if (hit != 0) {
          const int src = __ffs(hit) - 1;
          const unsigned long long ww = __shfl_sync(0xffffffffu, word, src);
          const int wi = asc ? w0 + base + src : w0 - base - src;
          r = wi * 64 + (asc ? __ffsll(static_cast<long long>(ww)) - 1 : 63 - __clzll(static_cast<long long>(ww)));
        } else if ((asc && w0 + base + 31 >= W64 - 1) || (!asc && w0 - base - 31 <= 0)) {
          break;
        }
      }
      if (r < 0) { break; }
      const double yr = y[r] / (asc ? cx[cp[r]] : cx[cp[r + 1] - 1]);
      __syncwarp();
      if (lane == 0) {
        y[r] = yr;
        lst[cnt] = r;
        bits[r >> 6] &= ~(1ull << (r & 63));
      }
      ++cnt;
      __syncwarp();
      const int p0 = asc ? cp[r] + 1 : cp[r];
      const int p1 = asc ? cp[r + 1] : cp[r + 1] - 1;
      for (int p = p0 + lane; p < p1; p += 32) {
        const int i = ri[p];
        y[i] -= cx[p] * yr;
        atomicOr(&bits[i >> 6], 1ull << (i & 63));
      }
      __syncwarp();
      cur = r;
    }
    int nz = 0;
    for (int q = lane; q < cnt; q += 32) { nz += y[lst[q]] != 0.0 ? 1 : 0; }
    for (int o = 16; o > 0; o >>= 1) { nz += __shfl_xor_sync(0xffffffffu, nz, o); }
    int pos = 0;
    if (lane == 0) { pos = atomicAdd(&b.cur[f][s], nz); }
    pos = __shfl_sync(0xffffffffu, pos, 0);
    const bool fit = pos + nz <= b.cap[f];
    if (!fit && lane == 0) { atomicExch(b.err, 1); }
    const size_t base = static_cast<size_t>(s) * b.cap[f];
    int w = 0;
    for (int q0 = 0; q0 < cnt; q0 += 32) {
      const int q = q0 + lane;
      const int row = q < cnt ? lst[q] : 0;
      const double v = q < cnt ? y[row] : 0.0;
      const bool keep = q < cnt && v != 0.0;
      const unsigned bal = __ballot_sync(0xffffffffu, keep);
      if (keep && fit) {
        const int o = pos + w + __popc(bal & ((1u << lane) - 1u));
        b.si[f][base + o] = row;
        b.sx[f][base + o] = v;
      }
      w += __popc(bal);
      if (q < cnt) { y[row] = 0.0; }
    }
    __syncwarp();
    if (lane == 0) {
      const size_t c = static_cast<size_t>(s) * m + (f == 0 ? a.row_perm[k] : k);
      b.cstart[f][c] = pos;
      b.clen[f][c]   = nz;
    }
  }
}

// Block per (seat, factor): CSC column pointers by scan, entries copied in.
constexpr int kFinvPack = 1024;
__global__ void __launch_bounds__(kFinvPack) finv_pack(const kernel_args* args, finv_build_t b)
{
  using scan_t = cub::BlockScan<int, kFinvPack, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename scan_t::TempStorage tmp;
  const int s = blockIdx.x / 2, f = blockIdx.x % 2, m = b.m;
  if (args[s].done != 0) { return; }
  const int per = (m + kFinvPack - 1) / kFinvPack;
  const int c0 = threadIdx.x * per, c1 = min(m, c0 + per);
  const int* clen = b.clen[f] + static_cast<size_t>(s) * m;
  const int* cst  = b.cstart[f] + static_cast<size_t>(s) * m;
  int* cp         = b.cp[f] + static_cast<size_t>(s) * (m + 1);
  int sum = 0;
  for (int c = c0; c < c1; ++c) { sum += clen[c]; }
  int off = 0, tot = 0;
  scan_t(tmp).ExclusiveSum(sum, off, tot);
  for (int c = c0; c < c1; ++c) {
    cp[c] = off;
    off += clen[c];
  }
  if (threadIdx.x == 0) { cp[m] = tot; }
  __syncthreads();
  const size_t base = static_cast<size_t>(s) * b.cap[f];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int c = warp; c < m; c += kFinvPack / 32) {
    const int src = cst[c], dst = cp[c], len = clen[c];
    for (int q = lane; q < len; q += 32) {
      b.ci[f][base + dst + q] = b.si[f][base + src + q];
      b.cv[f][base + dst + q] = b.sx[f][base + src + q];
    }
  }
}

// Block per (seat, factor): CSR by count, scan and a scatter with shared atomic row cursors (order
// within a row varies run to run).
constexpr int kFinvTr = 256;
__global__ void __launch_bounds__(kFinvTr) finv_transpose(const kernel_args* args, finv_build_t b)
{
  extern __shared__ int rc[];
  using scan_t = cub::BlockScan<int, kFinvTr, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename scan_t::TempStorage tmp;
  const int s = blockIdx.x / 2, f = blockIdx.x % 2, m = b.m;
  if (args[s].done != 0) { return; }
  const size_t base = static_cast<size_t>(s) * b.cap[f];
  const int* cp     = b.cp[f] + static_cast<size_t>(s) * (m + 1);
  const int* ci     = b.ci[f] + base;
  const double* cv  = b.cv[f] + base;
  int* rp           = b.rp[f] + static_cast<size_t>(s) * (m + 1);
  for (int r = threadIdx.x; r < m; r += kFinvTr) { rc[r] = 0; }
  __syncthreads();
  const int nnz = cp[m];
  for (int q = threadIdx.x; q < nnz; q += kFinvTr) { atomicAdd(&rc[ci[q]], 1); }
  __syncthreads();
  const int per = (m + kFinvTr - 1) / kFinvTr;
  const int r0 = threadIdx.x * per, r1 = min(m, r0 + per);
  int sum = 0;
  for (int r = r0; r < r1; ++r) { sum += rc[r]; }
  int off = 0, tot = 0;
  scan_t(tmp).ExclusiveSum(sum, off, tot);
  __syncthreads();
  for (int r = r0; r < r1; ++r) {
    const int c = rc[r];
    rp[r] = off, rc[r] = off;
    off += c;
  }
  if (threadIdx.x == 0) { rp[m] = tot; }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int c = warp; c < m; c += kFinvTr / 32) {
    for (int q = cp[c] + lane; q < cp[c + 1]; q += 32) {
      const int o = atomicAdd(&rc[ci[q]], 1);
      b.rj[f][base + o] = c;
      b.rv[f][base + o] = cv[q];
    }
  }
}

// CUOPT_EXP_INV=library: the harness's start-up inverse library (not per-generation work).
inline bool gpu_fission_inv_library()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_EXP_INV");
    return e != nullptr && std::string(e) == "library";
  }();
  return on;
}

'''+anchor)
open(f,'w').write(s)
patch(g,[
("""  std::optional<rmm::device_buffer> d_fm;  // fission replay matrices, one per seat""",
"""  std::optional<rmm::device_buffer> d_fm;  // fission replay matrices, one per seat
  std::optional<rmm::device_buffer> d_finv_mem;  // device-built explicit inverses (finv_build_t)
  gpu_phase2_detail::finv_build_t finv_b{};
  int* h_finv_err{nullptr};"""),
("""  if (gpu_fission() && command != gpu_batch_seat_command::CHILD_CONTINUE) {
    if (item.fission_inv == nullptr) {""","""  if (gpu_fission() && gpu_fission_inv_library() && command != gpu_batch_seat_command::CHILD_CONTINUE) {
    if (item.fission_inv == nullptr) {"""),
("""      if (H->h_finv.size() != N) {
        error = "fissioned pivot loop: seats without explicit inverses";
        return false;
      }
      if (!H->d_finv) { H->d_finv.emplace(N * sizeof(gpu_fission_inv), H->stream); }""","""      const bool lib = gpu_fission_inv_library();
      if (lib && H->h_finv.size() != N) {
        error = "fissioned pivot loop: seats without explicit inverses";
        return false;
      }
      if (!H->d_finv) { H->d_finv.emplace(N * sizeof(gpu_fission_inv), H->stream); }
      if (!lib && !H->d_finv_mem) {
        // Device-built inverses: one carved buffer, fixed per-seat capacities; seat pointers set once.
        auto& b = H->finv_b;
        b.N = static_cast<int>(N), b.m = static_cast<int>(H->rows);
        int dev = 0, sms = 0, bps = 0;
        RAFT_CUDA_TRY(cudaGetDevice(&dev));
        RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, finv_columns, 256, 0));
        b.grid = sms * bps, b.warps = b.grid * 8;
        const size_t m = b.m, Nn = N, W64 = (m + 63) / 64;
        size_t bytes = 0;
        auto need = [&](size_t nb) { bytes += (nb + 255) / 256 * 256; };
        for (int f = 0; f < 2; ++f) {
          const size_t ne = Nn * static_cast<size_t>(b.cap[f]);
          need(Nn * 4); need(Nn * m * 4); need(Nn * m * 4); need(ne * 4); need(ne * 8);
          need(Nn * (m + 1) * 4); need(ne * 4); need(ne * 8); need(Nn * (m + 1) * 4); need(ne * 4); need(ne * 8);
        }
        need(2 * Nn * m * 4); need(4 * 4);
        need(static_cast<size_t>(b.warps) * m * 8); need(static_cast<size_t>(b.warps) * W64 * 8);
        need(static_cast<size_t>(b.warps) * m * 4);
        H->d_finv_mem.emplace(bytes, H->stream);
        auto* p = static_cast<unsigned char*>(H->d_finv_mem->data());
        auto take = [&](size_t nb) { auto* q = p; p += (nb + 255) / 256 * 256; return q; };
        for (int f = 0; f < 2; ++f) {
          const size_t ne = Nn * static_cast<size_t>(b.cap[f]);
          b.cur[f] = reinterpret_cast<int*>(take(Nn * 4));
          b.cstart[f] = reinterpret_cast<int*>(take(Nn * m * 4));
          b.clen[f] = reinterpret_cast<int*>(take(Nn * m * 4));
          b.si[f] = reinterpret_cast<int*>(take(ne * 4));
          b.sx[f] = reinterpret_cast<double*>(take(ne * 8));
          b.cp[f] = reinterpret_cast<int*>(take(Nn * (m + 1) * 4));
          b.ci[f] = reinterpret_cast<int*>(take(ne * 4));
          b.cv[f] = reinterpret_cast<double*>(take(ne * 8));
          b.rp[f] = reinterpret_cast<int*>(take(Nn * (m + 1) * 4));
          b.rj[f] = reinterpret_cast<int*>(take(ne * 4));
          b.rv[f] = reinterpret_cast<double*>(take(ne * 8));
        }
        b.tasks  = reinterpret_cast<int*>(take(2 * Nn * m * 4));
        b.ntasks = reinterpret_cast<int*>(take(16));
        b.next = b.ntasks + 1, b.err = b.ntasks + 2;
        b.ws_y = reinterpret_cast<double*>(take(static_cast<size_t>(b.warps) * m * 8));
        b.ws_b = reinterpret_cast<unsigned long long*>(take(static_cast<size_t>(b.warps) * W64 * 8));
        b.ws_l = reinterpret_cast<int*>(take(static_cast<size_t>(b.warps) * m * 4));
        RAFT_CUDA_TRY(cudaMemsetAsync(b.ws_y, 0, static_cast<size_t>(b.warps) * m * 8, cs));
        RAFT_CUDA_TRY(cudaMemsetAsync(b.ws_b, 0, static_cast<size_t>(b.warps) * W64 * 8, cs));
        H->h_finv.assign(N, gpu_fission_inv{});
        for (size_t s = 0; s < N; ++s) {
          auto& v = H->h_finv[s];
          const size_t pb = s * (m + 1), e0 = s * b.cap[0], e1 = s * b.cap[1];
          v.Lc_p = b.cp[0] + pb, v.Lc_i = b.ci[0] + e0, v.Lc_x = b.cv[0] + e0;
          v.Lr_p = b.rp[0] + pb, v.Lr_i = b.rj[0] + e0, v.Lr_x = b.rv[0] + e0;
          v.Uc_p = b.cp[1] + pb, v.Uc_i = b.ci[1] + e1, v.Uc_x = b.cv[1] + e1;
          v.Ur_p = b.rp[1] + pb, v.Ur_i = b.rj[1] + e1, v.Ur_x = b.rv[1] + e1;
        }
        RAFT_CUDA_TRY(cudaFuncSetAttribute(finv_transpose, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(m * sizeof(int))));
        RAFT_CUDA_TRY(cudaMallocHost(&H->h_finv_err, sizeof(int)));
      }"""),
("""      fission_presetup<<<Ni, kPresetupThreads, 0, cs>>>(H->d_args, Ni);""","""      if (!lib) {
        // Every seat's inverses from the factors just staged: per-generation work.
        auto& b = H->finv_b;
        for (int f = 0; f < 2; ++f) { RAFT_CUDA_TRY(cudaMemsetAsync(b.cur[f], 0, N * sizeof(int), cs)); }
        RAFT_CUDA_TRY(cudaMemsetAsync(b.ntasks, 0, 3 * sizeof(int), cs));
        const long long T = 2LL * b.N * b.m;
        finv_trivial<<<static_cast<unsigned>((T + 255) / 256), 256, 0, cs>>>(H->d_args, b);
        finv_columns<<<b.grid, 256, 0, cs>>>(H->d_args, b);
        finv_pack<<<2 * Ni, kFinvPack, 0, cs>>>(H->d_args, b);
        finv_transpose<<<2 * Ni, kFinvTr, b.m * sizeof(int), cs>>>(H->d_args, b);
        RAFT_CUDA_TRY(cudaMemcpyAsync(H->h_finv_err, b.err, sizeof(int), cudaMemcpyDeviceToHost, cs));
      }
      fission_presetup<<<Ni, kPresetupThreads, 0, cs>>>(H->d_args, Ni);"""),
])
# fail loudly after the first live-count sync
s=open(g).read()
a="""        RAFT_CUDA_TRY(cudaStreamSynchronize(cs));
        static const bool trace = std::getenv("CUOPT_BNB_GPU_FISSION_TRACE") != nullptr;"""
assert s.count(a)==1
s=s.replace(a,"""        RAFT_CUDA_TRY(cudaStreamSynchronize(cs));
        if (!lib && *H->h_finv_err != 0) {
          throw std::runtime_error("device explicit inverses: a seat exceeded its per-seat capacity");
        }
        static const bool trace = std::getenv("CUOPT_BNB_GPU_FISSION_TRACE") != nullptr;""")
open(g,'w').write(s)
# harness: build the library only when asked
patch(t,[("""  if (const char* e = std::getenv("CUOPT_BNB_GPU_FISSION"); e != nullptr && std::string(e) != "0") {
    namespace ei = cuopt::linear_programming::dual_simplex::explicit_inverse;""","""  const char* inv_mode = std::getenv("CUOPT_EXP_INV");
  if (const char* e = std::getenv("CUOPT_BNB_GPU_FISSION");
      e != nullptr && std::string(e) != "0" && inv_mode != nullptr && std::string(inv_mode) == "library") {
    namespace ei = cuopt::linear_programming::dual_simplex::explicit_inverse;""")])
print('ok')
