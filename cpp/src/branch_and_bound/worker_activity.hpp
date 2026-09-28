/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
// Pool telemetry (CUOPT_BNB_GPU_LOG): what a B&B worker thread was doing over a past interval, from a
// per-thread ring of activity transitions. Used to attribute the time a finished pool waited for its
// owning worker.
#pragma once

#include <math_optimization/tic_toc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace cuopt::mathematical_optimization::mip {

enum class worker_activity_t : std::uint8_t { OTHER, CPU_NODE_LP, STRONG_BRANCH, POOL_STEP, COUNT };

inline bool worker_activity_on()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LOG");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}

struct worker_activity_log_t {
  static constexpr int cap = 4096;
  std::array<double, cap> t{};
  std::array<std::uint8_t, cap> a{};
  std::uint64_t n{0};  // transitions written
  worker_activity_t cur{worker_activity_t::OTHER};

  void push(worker_activity_t next)
  {
    t[n % cap] = tic();
    a[n % cap] = static_cast<std::uint8_t>(next);
    ++n;
    cur = next;
  }

  // Seconds per activity inside [t0, t1]; time before the oldest kept transition counts as OTHER.
  void breakdown(double t0, double t1, double* out) const
  {
    const std::uint64_t first = n > cap ? n - cap : 0;
    worker_activity_t act = cur;
    double end            = t1;
    for (std::uint64_t k = n; k > first; --k) {
      const double tk = t[(k - 1) % cap];
      if (tk <= t0) { break; }
      if (tk < end) { out[static_cast<int>(act)] += end - tk; }
      end = std::min(end, tk);
      act = k - 1 > first ? static_cast<worker_activity_t>(a[(k - 2) % cap]) : worker_activity_t::OTHER;
    }
    if (end > t0) { out[static_cast<int>(act)] += end - t0; }
  }
};

inline worker_activity_log_t& worker_activity_log()
{
  static thread_local worker_activity_log_t log;
  return log;
}

// Marks the enclosed code as `a` on this thread's log; restores the enclosing activity on exit.
class worker_activity_scope {
 public:
  explicit worker_activity_scope(worker_activity_t a)
  {
    if (!worker_activity_on()) { return; }
    auto& log = worker_activity_log();
    prev_     = log.cur;
    on_       = true;
    log.push(a);
  }
  ~worker_activity_scope()
  {
    if (on_) { worker_activity_log().push(prev_); }
  }
  worker_activity_scope(const worker_activity_scope&)            = delete;
  worker_activity_scope& operator=(const worker_activity_scope&) = delete;

 private:
  worker_activity_t prev_{worker_activity_t::OTHER};
  bool on_{false};
};

}  // namespace cuopt::mathematical_optimization::mip
