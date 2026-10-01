# Staging: packed_upload_t::add records the piece; flush() does all host copies with several threads.
import sys
root=sys.argv[1]
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
g=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
patch(g,[
("""  std::vector<size_t> src;
  std::vector<void*> dst;
  std::vector<size_t> bytes;
  std::uint64_t flushes{0}, flushed_bytes{0}, flushed_pieces{0}, grows{0};
""","""  std::vector<size_t> src;
  std::vector<void*> dst;
  std::vector<size_t> bytes;
  std::vector<const void*> from;  // host copies are deferred to flush(); sources must stay unchanged
  std::uint64_t flushes{0}, flushed_bytes{0}, flushed_pieces{0}, grows{0};
  static constexpr int kCopyThreads = 16;  // CHOSEN
"""),
("""    auto grown = pinned_make<unsigned char>(nullptr, cuda::stream_ref{s.value()}, c);
    if (used != 0) { std::memcpy(grown.data(), h->data(), used); }
    h.emplace(std::move(grown));""","""    auto grown = pinned_make<unsigned char>(nullptr, cuda::stream_ref{s.value()}, c);
    h.emplace(std::move(grown));  // nothing to keep: the pieces are copied in flush()"""),
("""    reserve(align(used) + n, s);
    used = align(used);
    std::memcpy(hdata() + used, from, n);
    src.push_back(used);""","""    used = align(used);
    this->from.push_back(from);
    src.push_back(used);"""),
("""    const size_t desc  = align(used);
    const size_t total = desc + P * (2 * sizeof(void*) + sizeof(size_t));
    reserve(total, s);""","""    const size_t desc  = align(used);
    const size_t total = desc + P * (2 * sizeof(void*) + sizeof(size_t));
    reserve(total, s);
    {
      // Contiguous runs of pieces of about equal bytes per thread.
      unsigned char* hb = hdata();
      const auto run = [&](size_t k0, size_t k1) {
        for (size_t k = k0; k < k1; ++k) { std::memcpy(hb + src[k], from[k], bytes[k]); }
      };
      const int T = used < (size_t{8} << 20) ? 1 : kCopyThreads;
      std::vector<std::thread> ths;
      size_t k0 = 0;
      for (int t = 0; t < T && k0 < P; ++t) {
        const size_t goal = used * static_cast<size_t>(t + 1) / static_cast<size_t>(T);
        size_t k1         = k0;
        while (k1 < P && (t == T - 1 || src[k1] < goal)) { ++k1; }
        if (t + 1 < T) { ths.emplace_back(run, k0, k1); } else { run(k0, k1); }
        k0 = k1;
      }
      for (auto& th : ths) { th.join(); }
    }"""),
("""    src.clear();
    dst.clear();
    bytes.clear();
    used = 0;
  }
};""","""    src.clear();
    dst.clear();
    bytes.clear();
    from.clear();
    used = 0;
  }
};"""),
])
s=open(g).read()
if '#include <thread>' not in s:
    s=s.replace('#include <mutex>\n','#include <mutex>\n#include <thread>\n',1); open(g,'w').write(s)
print('ok')
