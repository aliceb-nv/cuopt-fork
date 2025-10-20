/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <raft/random/rng_device.cuh>
#include <utilities/cuda_helpers.cuh>

namespace cuopt {

// TODO: should be thread local?
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

 public:
  seed_generator(seed_generator const&) = delete;
  void operator=(seed_generator const&) = delete;
};

}  // namespace cuopt
