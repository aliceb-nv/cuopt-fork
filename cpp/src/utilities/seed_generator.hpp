/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <random>

namespace cuopt {

class seed_generator {
  static int64_t seed_;

 public:
  template <typename seed_t>
  static void set_seed(seed_t seed)
  {
#ifdef BENCHMARK
    seed_ = std::random_device{}();
#else
    seed_ = static_cast<int64_t>(seed);
#endif
  }

  template <typename arg0, typename arg1, typename... args>
  static void set_seed(arg0 seed0, arg1 seed1, args... seeds)
  {
    set_seed(seed1 + ((seed0 + seed1) * (seed0 + seed1 + 1) / 2), seeds...);
  }

  static int64_t get_seed() { return seed_++; }

  seed_generator(seed_generator const&) = delete;
  void operator=(seed_generator const&) = delete;
};

}  // namespace cuopt
