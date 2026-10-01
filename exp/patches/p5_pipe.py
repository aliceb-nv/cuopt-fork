# flush(): host copies and the host-to-device copy pipelined in chunks of pieces.
import sys
root=sys.argv[1]
p=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'; s=open(p).read()
i=s.index("""    {
      // Contiguous runs of pieces of about equal bytes per thread.""")
j=s.index("""    auto* base = static_cast<unsigned char*>(d->data());""")
k=s.index("""    if (total > dev_cap) {""")
assert i<k<j
pre=s[i:k]
s=s[:i]+s[k:]
a="""    auto* base = static_cast<unsigned char*>(d->data());
    auto** hs  = reinterpret_cast<void**>(hdata() + desc);
    auto** hd  = hs + P;
    auto* hn   = reinterpret_cast<size_t*>(hd + P);
    for (size_t k = 0; k < P; ++k) {
      hs[k] = base + src[k];
      hd[k] = dst[k];
      hn[k] = bytes[k];
    }
    RAFT_CUDA_TRY(cudaMemcpyAsync(base, hdata(), total, cudaMemcpyHostToDevice, s.value()));"""
b="""    auto* base = static_cast<unsigned char*>(d->data());
    {
      // Chunks of whole pieces: chunk c is copied to the device while chunk c+1 is packed.
      constexpr size_t kChunk = size_t{64} << 20;  // CHOSEN
      unsigned char* hb = hdata();
      const auto run = [&](size_t k0, size_t k1) {
        for (size_t k = k0; k < k1; ++k) { std::memcpy(hb + src[k], from[k], bytes[k]); }
      };
      size_t c0 = 0;
      while (c0 < P) {
        size_t c1 = c0;
        while (c1 < P && src[c1] + bytes[c1] - src[c0] <= kChunk) { ++c1; }
        if (c1 == c0) { c1 = c0 + 1; }
        const size_t b0 = src[c0], b1 = src[c1 - 1] + bytes[c1 - 1], cb = b1 - b0;
        const int T     = cb < (size_t{4} << 20) ? 1 : kCopyThreads;
        std::vector<std::thread> ths;
        size_t k0 = c0;
        for (int t = 0; t < T && k0 < c1; ++t) {
          const size_t goal = b0 + cb * static_cast<size_t>(t + 1) / static_cast<size_t>(T);
          size_t k1         = k0;
          while (k1 < c1 && (t == T - 1 || src[k1] < goal)) { ++k1; }
          if (t + 1 < T) { ths.emplace_back(run, k0, k1); } else { run(k0, k1); }
          k0 = k1;
        }
        for (auto& th : ths) { th.join(); }
        RAFT_CUDA_TRY(cudaMemcpyAsync(base + b0, hb + b0, cb, cudaMemcpyHostToDevice, s.value()));
        c0 = c1;
      }
    }
    auto** hs  = reinterpret_cast<void**>(hdata() + desc);
    auto** hd  = hs + P;
    auto* hn   = reinterpret_cast<size_t*>(hd + P);
    for (size_t k = 0; k < P; ++k) {
      hs[k] = base + src[k];
      hd[k] = dst[k];
      hn[k] = bytes[k];
    }
    RAFT_CUDA_TRY(cudaMemcpyAsync(base + desc, hdata() + desc, total - desc, cudaMemcpyHostToDevice, s.value()));"""
assert s.count(a)==1; s=s.replace(a,b)
open(p,'w').write(s); print('ok')
