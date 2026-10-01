# flush(): a persistent pool of copy threads in place of per-chunk std::thread creation.
import sys
root=sys.argv[1]
p=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'; s=open(p).read()
a="""struct packed_upload_t {"""
b="""// Workers that run fn(t) for t in [1, T) while the caller runs fn(0); run() returns when all are done.
struct copy_pool_t {
  explicit copy_pool_t(int T) : T_(T)
  {
    for (int t = 1; t < T; ++t) {
      th_.emplace_back([this, t] {
        int seen = 0;
        for (;;) {
          std::unique_lock<std::mutex> lk(mu_);
          cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
          if (stop_) { return; }
          seen = gen_;
          lk.unlock();
          (*job_)(t);
          lk.lock();
          if (--pending_ == 0) { done_.notify_one(); }
        }
      });
    }
  }
  ~copy_pool_t()
  {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) { t.join(); }
  }
  template <class Fn>
  void run(Fn&& fn)
  {
    std::function<void(int)> f(fn);
    {
      std::lock_guard<std::mutex> lk(mu_);
      job_     = &f;
      pending_ = T_ - 1;
      ++gen_;
    }
    cv_.notify_all();
    f(0);
    std::unique_lock<std::mutex> lk(mu_);
    done_.wait(lk, [&] { return pending_ == 0; });
  }
  int T_;
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  std::function<void(int)>* job_{nullptr};
  int gen_{0}, pending_{0};
  bool stop_{false};
};

struct packed_upload_t {"""
assert s.count(a)==1; s=s.replace(a,b)
a="""        const int T     = cb < (size_t{4} << 20) ? 1 : kCopyThreads;
        std::vector<std::thread> ths;
        size_t k0 = c0;
        for (int t = 0; t < T && k0 < c1; ++t) {
          const size_t goal = b0 + cb * static_cast<size_t>(t + 1) / static_cast<size_t>(T);
          size_t k1         = k0;
          while (k1 < c1 && (t == T - 1 || src[k1] < goal)) { ++k1; }
          if (t + 1 < T) { ths.emplace_back(run, k0, k1); } else { run(k0, k1); }
          k0 = k1;
        }
        for (auto& th : ths) { th.join(); }"""
b="""        if (cb < (size_t{4} << 20)) {
          run(c0, c1);
        } else {
          static copy_pool_t pool(kCopyThreads);
          size_t kb[kCopyThreads + 1];
          kb[0] = c0;
          for (int t = 0; t < kCopyThreads; ++t) {
            const size_t goal = b0 + cb * static_cast<size_t>(t + 1) / kCopyThreads;
            size_t k1         = kb[t];
            while (k1 < c1 && (t == kCopyThreads - 1 || src[k1] < goal)) { ++k1; }
            kb[t + 1] = k1;
          }
          pool.run([&](int t) { run(kb[t], kb[t + 1]); });
        }"""
assert s.count(a)==1; s=s.replace(a,b)
if '#include <condition_variable>' not in s: s=s.replace('#include <thread>\n','#include <thread>\n#include <condition_variable>\n#include <functional>\n',1)
open(p,'w').write(s); print('ok')
