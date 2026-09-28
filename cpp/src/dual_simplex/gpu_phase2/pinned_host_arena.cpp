/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include "pinned_host_arena.hpp"

#include <raft/core/nvtx.hpp>

#include <cuda_runtime_api.h>
#include <linux/mempolicy.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace cuopt::linear_programming::dual_simplex::pinned_host {

namespace {
constexpr std::size_t huge_page = std::size_t{2} << 20;
std::size_t align256(std::size_t b) { return (b + 255) / 256 * 256; }

struct region_t {
  int node{-1};
  mapping_t map;
  std::size_t slice_bytes{0}, slices{0}, first_id{0};
  std::vector<int> free_slices;
  char* overflow{nullptr};
  std::size_t overflow_bytes{0}, overflow_used{0};
  std::size_t acquires{0}, in_use{0}, peak_in_use{0}, no_slice{0}, overflow_takes{0},
    overflow_misses{0};
  double map_s{0.0};
};

struct arena_t {
  std::mutex mu;
  std::atomic<bool> ready{false};
  bool per_node{false};
  std::vector<std::unique_ptr<region_t>> regions;
  std::vector<region_t*> region_of_slice;
  double reserve_s{0.0};
};
arena_t g_arena;

void carve(region_t& r, std::size_t slice_bytes, std::size_t slices, std::size_t first_id)
{
  r.slice_bytes = slice_bytes;
  r.slices      = slices;
  r.first_id    = first_id;
  for (std::size_t k = slices; k > 0; --k) { r.free_slices.push_back(static_cast<int>(first_id + k - 1)); }
  r.overflow       = static_cast<char*>(r.map.p) + slice_bytes * slices;
  r.overflow_bytes = r.map.bytes - slice_bytes * slices;
}

region_t* region_for(int node)
{
  if (g_arena.regions.size() == 1 || !g_arena.per_node) { return g_arena.regions.front().get(); }
  if (node < 0) { node = current_node(); }
  for (auto& r : g_arena.regions) {
    if (r->node == node) { return r.get(); }
  }
  return nullptr;
}
}  // namespace

mapping_t map(std::size_t bytes, int node)
{
  mapping_t m;
  m.bytes = (std::max<std::size_t>(bytes, 1) + huge_page - 1) / huge_page * huge_page;
  m.len   = m.bytes + huge_page;   // room to align the start to 2 MB
  m.base  = mmap(nullptr, m.len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m.base == MAP_FAILED) { throw std::runtime_error("pinned_host::map: mmap failed"); }
  m.p = reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(m.base) + huge_page - 1) /
                                huge_page * huge_page);
  madvise(m.p, m.bytes, MADV_HUGEPAGE);
  if (node >= 0) {
    unsigned long mask[16] = {};
    mask[node / 64] |= 1UL << (node % 64);
    if (syscall(SYS_mbind, m.p, m.bytes, MPOL_BIND, mask, sizeof(mask) * 8, 0) != 0) {
      std::fprintf(stderr, "[PINNED-ARENA] mbind to node %d failed: %s\n", node, std::strerror(errno));
    }
  }
  {
    raft::common::nvtx::range fault_scope("GpuDS::hp_map_fault");
    std::memset(m.p, 0, m.bytes);   // fault the pages in as 2 MB pages before registering
  }
  raft::common::nvtx::range register_scope("GpuDS::hp_map_register");
  if (cudaHostRegister(m.p, m.bytes, cudaHostRegisterPortable) != cudaSuccess) {
    munmap(m.base, m.len);
    throw std::runtime_error("pinned_host::map: cudaHostRegister failed");
  }
  return m;
}

void unmap(const mapping_t& m)
{
  if (m.base == nullptr) { return; }
  cudaHostUnregister(m.p);
  munmap(m.base, m.len);
}

bool reserve(std::size_t slice_bytes, std::size_t slices, std::size_t overflow_bytes)
{
  std::lock_guard<std::mutex> lk(g_arena.mu);
  if (g_arena.ready.load()) { return false; }
  raft::common::nvtx::range scope("GpuDS::pinned_arena_reserve");
  const std::size_t s = align256(slice_bytes);
  auto r              = std::make_unique<region_t>();
  const auto t0       = std::chrono::steady_clock::now();
  r->map              = map(s * slices + align256(overflow_bytes));
  r->map_s = g_arena.reserve_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  carve(*r, s, slices, 0);
  g_arena.region_of_slice.assign(slices, r.get());
  g_arena.regions.push_back(std::move(r));
  g_arena.ready.store(true);
  return true;
}

bool reserve_nodes(std::size_t slice_bytes,
                   const std::vector<std::pair<int, std::size_t>>& node_slices,
                   std::size_t overflow_per_slice)
{
  std::lock_guard<std::mutex> lk(g_arena.mu);
  if (g_arena.ready.load()) { return false; }
  raft::common::nvtx::range scope("GpuDS::pinned_arena_reserve");
  const std::size_t s = align256(slice_bytes);
  const auto t0       = std::chrono::steady_clock::now();
  std::size_t next_id = 0;
  for (const auto& [node, slices] : node_slices) {
    auto r      = std::make_unique<region_t>();
    r->node     = node;
    r->first_id = next_id;
    r->slices   = slices;
    next_id += slices;
    g_arena.regions.push_back(std::move(r));
  }
  int device = 0;
  cudaGetDevice(&device);
  std::vector<std::thread> fill;
  std::vector<std::exception_ptr> err(g_arena.regions.size());
  for (std::size_t i = 0; i < g_arena.regions.size(); ++i) {
    fill.emplace_back([&, i] {
      region_t& r = *g_arena.regions[i];
      try {
        cudaSetDevice(device);
        cpu_set_t set;
        if (node_cpus(r.node, &set)) { pthread_setaffinity_np(pthread_self(), sizeof(set), &set); }
        const auto t = std::chrono::steady_clock::now();
        r.map        = map(s * r.slices + align256(overflow_per_slice * r.slices), r.node);
        r.map_s      = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
      } catch (...) {
        err[i] = std::current_exception();
      }
    });
  }
  for (auto& t : fill) { t.join(); }
  for (auto& e : err) {
    if (e) { std::rethrow_exception(e); }
  }
  g_arena.region_of_slice.assign(next_id, nullptr);
  for (auto& r : g_arena.regions) {
    const std::size_t n = r->slices;
    r->slices           = 0;
    carve(*r, s, n, r->first_id);
    for (std::size_t k = 0; k < n; ++k) { g_arena.region_of_slice[r->first_id + k] = r.get(); }
  }
  g_arena.per_node  = true;
  g_arena.reserve_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  g_arena.ready.store(true);
  return true;
}

bool reserved() { return g_arena.ready.load(); }

int acquire_slice(void** p, std::size_t* bytes, int node)
{
  if (!g_arena.ready.load()) { return -1; }
  std::lock_guard<std::mutex> lk(g_arena.mu);
  region_t* r = region_for(node);
  if (r == nullptr) { return -1; }
  if (r->free_slices.empty()) {
    ++r->no_slice;
    return -1;
  }
  const int id = r->free_slices.back();
  r->free_slices.pop_back();
  ++r->acquires;
  r->peak_in_use = std::max(r->peak_in_use, ++r->in_use);
  *p     = static_cast<char*>(r->map.p) + (static_cast<std::size_t>(id) - r->first_id) * r->slice_bytes;
  *bytes = r->slice_bytes;
  return id;
}

void release_slice(int id)
{
  if (id < 0) { return; }
  std::lock_guard<std::mutex> lk(g_arena.mu);
  region_t* r = g_arena.region_of_slice[id];
  r->free_slices.push_back(id);
  --r->in_use;
}

void* overflow_take(std::size_t bytes, int node)
{
  if (!g_arena.ready.load()) { return nullptr; }
  std::lock_guard<std::mutex> lk(g_arena.mu);
  region_t* r = region_for(node);
  if (r == nullptr) { return nullptr; }
  const std::size_t off = align256(r->overflow_used);
  if (off + bytes > r->overflow_bytes) {
    ++r->overflow_misses;
    return nullptr;
  }
  r->overflow_used = off + bytes;
  ++r->overflow_takes;
  return r->overflow + off;
}

bool contains(const void* p)
{
  if (!g_arena.ready.load()) { return false; }
  const char* c = static_cast<const char*>(p);
  for (const auto& r : g_arena.regions) {
    const char* a = static_cast<const char*>(r->map.p);
    if (c >= a && c < a + r->map.bytes) { return true; }
  }
  return false;
}

std::string report()
{
  if (!g_arena.ready.load()) { return {}; }
  std::lock_guard<std::mutex> lk(g_arena.mu);
  std::string out;
  char buf[400];
  std::snprintf(buf, sizeof(buf), "[PINNED-ARENA] regions=%zu per_node=%d reserve_s=%.3f\n",
                g_arena.regions.size(), static_cast<int>(g_arena.per_node), g_arena.reserve_s);
  out += buf;
  for (const auto& r : g_arena.regions) {
    std::snprintf(buf,
                  sizeof(buf),
                  "[PINNED-ARENA] node=%d region_MB=%.1f created_in=%.3fs slices=%zu slice_MB=%.2f acquires=%zu "
                  "peak_in_use=%zu no_free_slice=%zu | overflow used_MB=%.2f of %.2f takes=%zu misses=%zu\n",
                  r->node, r->map.bytes / 1e6, r->map_s, r->slices, r->slice_bytes / 1e6, r->acquires,
                  r->peak_in_use, r->no_slice, r->overflow_used / 1e6, r->overflow_bytes / 1e6,
                  r->overflow_takes, r->overflow_misses);
    out += buf;
  }
  return out;
}

int device_node(int device)
{
  int v = -1;
  if (cudaDeviceGetAttribute(&v, cudaDevAttrHostNumaId, device) != cudaSuccess) { return -1; }
  return v;
}

bool node_cpus(int node, cpu_set_t* set)
{
  if (node < 0) { return false; }
  std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
  std::string list;
  if (!(f >> list)) { return false; }
  CPU_ZERO(set);
  std::size_t a = 0;
  while (a < list.size()) {
    std::size_t e       = list.find(',', a);
    const std::string r = list.substr(a, e == std::string::npos ? std::string::npos : e - a);
    const std::size_t d = r.find('-');
    const int lo        = std::stoi(r.substr(0, d));
    const int hi        = d == std::string::npos ? lo : std::stoi(r.substr(d + 1));
    for (int c = lo; c <= hi; ++c) { CPU_SET(c, set); }
    a = e == std::string::npos ? list.size() : e + 1;
  }
  return true;
}

int current_node()
{
  unsigned cpu = 0, node = 0;
  if (syscall(SYS_getcpu, &cpu, &node, nullptr) != 0) { return -1; }
  return static_cast<int>(node);
}

}  // namespace cuopt::linear_programming::dual_simplex::pinned_host
