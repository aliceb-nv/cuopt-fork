/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
// Pinned host memory backed by 2 MB transparent huge pages and registered with cudaHostRegister.
//
// Registration calls serialize process-wide: with 127 pools each registering its own buffers the
// solver queued ~890 registrations for ~7 s at start-up (2026-09-28, mik, 8x H100). The process-wide
// arena registers one region (or one per NUMA node): equal slices, one per GPU pool handle (reused
// through a free list), and a shared overflow region for buffers sized after the handle exists.
#pragma once

#include <sched.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::dual_simplex::pinned_host {

struct mapping_t {
  void* base{nullptr};    // mmap result
  std::size_t len{0};     // mmap length
  void* p{nullptr};       // 2 MB-aligned, registered
  std::size_t bytes{0};   // registered length (multiple of 2 MB)
};

// One region of at least `bytes`, rounded up to 2 MB, zero-filled and registered (portable).
// node >= 0: pages bound to that NUMA node (mbind MPOL_BIND) and filled by the calling thread.
mapping_t map(std::size_t bytes, int node = -1);
void unmap(const mapping_t& m);

// Single region: `slices` slices of `slice_bytes` followed by `overflow_bytes`.
// Returns false if an arena already exists (the first reservation wins).
bool reserve(std::size_t slice_bytes, std::size_t slices, std::size_t overflow_bytes);
// One region per NUMA node, created in parallel by threads pinned to that node:
// per entry (node, slices), `slices` slices of `slice_bytes` plus slices * `overflow_per_slice`.
bool reserve_nodes(std::size_t slice_bytes,
                   const std::vector<std::pair<int, std::size_t>>& node_slices,
                   std::size_t overflow_per_slice);
bool reserved();
// A free slice (on `node`'s region when per-node, else any): its id (>= 0), start and size;
// -1 when none is free or nothing is reserved.
int acquire_slice(void** p, std::size_t* bytes, int node = -1);
void release_slice(int id);
// Bump allocation in an overflow region (256-byte aligned): `node`'s region when per-node, else
// the region of the NUMA node the calling thread runs on; nullptr when it is full.
void* overflow_take(std::size_t bytes, int node = -1);
// True for any address inside the process arena.
bool contains(const void* p);
// Summary lines (empty when nothing was reserved).
std::string report();

// Topology helpers.
int device_node(int device);                   // host NUMA node closest to a CUDA device, -1 if unknown
bool node_cpus(int node, cpu_set_t* set);      // CPUs of a NUMA node (sysfs cpulist)
int current_node();                            // NUMA node of the CPU the thread runs on

}  // namespace cuopt::linear_programming::dual_simplex::pinned_host
