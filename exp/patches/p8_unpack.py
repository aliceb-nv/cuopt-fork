# compact_unpack: per-seat unpack of the full LP states split over threads (seats are disjoint).
import sys
root=sys.argv[1]
p=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'; s=open(p).read()
a="""  H->seat_kind.resize(N);
  for (size_t b = 0; b < N; ++b) {
    const auto& e  = hd[b];"""
b="""  H->seat_kind.resize(N);
  const auto unpack = [&](size_t b0, size_t b1) {
  for (size_t b = b0; b < b1; ++b) {
    const auto& e  = hd[b];"""
assert s.count(a)==1; s=s.replace(a,b)
a="""      std::copy_n(reinterpret_cast<const variable_status_t*>(li + n), n, H->h_vstatus.data() + b * n);
    }
  }
}"""
b="""      std::copy_n(reinterpret_cast<const variable_status_t*>(li + n), n, H->h_vstatus.data() + b * n);
    }
  }
  };
  constexpr int kUnpackThreads = 8;  // CHOSEN
  if (full < 64) {
    unpack(0, N);
    return;
  }
  std::vector<std::thread> ths;
  for (int t = 1; t < kUnpackThreads; ++t) { ths.emplace_back(unpack, N * t / kUnpackThreads, N * (t + 1) / kUnpackThreads); }
  unpack(0, N / kUnpackThreads);
  for (auto& th : ths) { th.join(); }
}"""
assert s.count(a)==1; s=s.replace(a,b)
open(p,'w').write(s); print('ok')
