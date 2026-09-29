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
  int node{-1};     // lookup key: NUMA node, or device (by_device)
  int numa{-1};     // NUMA node the pages are bound to
  int device{-1};   // >= 0: registered in this device's context only
  mapping_t map;
  std::size_t slice_bytes{0}, slices{0}, first_id{0};
  std::vector<int> free_slices;
  char* overflow{nullptr};
  std::size_t overflow_bytes{0}, overflow_used{0};
  std::size_t acquires{0}, in_use{0}, peak_in_use{0}, no_slice{0}, overflow_takes{0},
    overflow_misses{0};
  double map_s{0.0};
  std::atomic<bool> live{false};   // mapped, registered and carved
};

struct arena_t {
  std::mutex mu;
  std::atomic<bool> ready{false};
  bool per_node{false};
  bool by_device{false};
  std::vector<std::unique_ptr<region_t>> regions;
  std::vector<region_t*> region_of_slice;
  double reserve_s{0.0}, sync_s{0.0};
  std::atomic<bool> published{false};   // every planned region is live
  std::thread background;                // maps the regions past the synchronous ones
  ~arena_t()
  {
    if (background.joinable()) { background.join(); }
  }
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

// Live regions serving `node` (all of them when not per node); caller holds g_arena.mu.
template <typename F>
region_t* find_region(int node, F&& ok)
{
  if (g_arena.per_node && node < 0) {
    int dev = -1;
    node    = g_arena.by_device ? (cudaGetDevice(&dev) == cudaSuccess ? dev : -1) : current_node();
  }
  for (auto& r : g_arena.regions) {
    if (!r->live.load(std::memory_order_acquire)) { continue; }
    if (g_arena.per_node && r->node != node) { continue; }
    if (ok(*r)) { return r.get(); }
  }
  return nullptr;
}
}  // namespace

mapping_t map(std::size_t bytes, int node, int device)
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
  const auto f0 = std::chrono::steady_clock::now();
  {
    raft::common::nvtx::range fault_scope("GpuDS::hp_map_fault");
    std::memset(m.p, 0, m.bytes);   // fault the pages in as 2 MB pages before registering
  }
  const auto f1 = std::chrono::steady_clock::now();
  m.fault_s     = std::chrono::duration<double>(f1 - f0).count();
  raft::common::nvtx::range register_scope("GpuDS::hp_map_register");
  if (device >= 0) { cudaSetDevice(device); }
  if (cudaHostRegister(m.p, m.bytes, device >= 0 ? cudaHostRegisterDefault : cudaHostRegisterPortable) !=
      cudaSuccess) {
    munmap(m.base, m.len);
    throw std::runtime_error("pinned_host::map: cudaHostRegister failed");
  }
  m.register_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - f1).count();
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
  r->live.store(true, std::memory_order_release);
  g_arena.published.store(true);
  g_arena.region_of_slice.assign(slices, r.get());
  g_arena.regions.push_back(std::move(r));
  g_arena.ready.store(true);
  return true;
}

namespace {
struct plan_t {
  int key, numa, device;
  std::size_t slices;
  std::size_t sync_regions;   // mapped before reserve returns (the rest in the background)
};
bool reserve_plan(std::size_t slice_bytes,
                  const std::vector<plan_t>& node_slices,
                  std::size_t overflow_per_slice,
                  std::size_t chunk,
                  bool by_device)
{
  {
    std::lock_guard<std::mutex> lk(g_arena.mu);
    if (g_arena.ready.load()) { return false; }
    // Plan every region up front (the vector never grows afterwards, so lookups need no care about
    // reallocation); each node's regions are then mapped in order and published one by one.
    std::size_t next_id = 0;
    for (const auto& [node, numa, dev, slices, sync_regions] : node_slices) {
      const std::size_t per = chunk > 0 ? chunk : slices;
      for (std::size_t done = 0; done < slices; done += per) {
        auto r      = std::make_unique<region_t>();
        r->node     = node;
        r->numa     = numa;
        r->device   = dev;
        r->first_id = next_id;
        r->slices   = std::min(per, slices - done);
        next_id += r->slices;
        g_arena.regions.push_back(std::move(r));
      }
    }
    g_arena.region_of_slice.assign(next_id, nullptr);
    for (auto& r : g_arena.regions) {
      for (std::size_t k = 0; k < r->slices; ++k) { g_arena.region_of_slice[r->first_id + k] = r.get(); }
    }
    g_arena.per_node  = true;
    g_arena.by_device = by_device;
    g_arena.ready.store(true);
  }
  raft::common::nvtx::range scope("GpuDS::pinned_arena_reserve");
  const std::size_t s = align256(slice_bytes);
  const auto t0       = std::chrono::steady_clock::now();
  int device = 0;
  cudaGetDevice(&device);
  // CUOPT_BNB_GPU_ARENA_THREADS: threads per node mapping regions concurrently. CHOSEN default 16:
  // unloaded, 16 threads register ~39 GB/s vs 9 GB/s for one (pinbench, 2026-09-28).
  static const int per_node_threads = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_ARENA_THREADS");
    return e != nullptr ? std::max(1, std::atoi(e)) : 16;
  }();
  // Each key's regions in plan order, split into the synchronous head and the background tail.
  using lists_t = std::vector<std::vector<std::size_t>>;
  lists_t head(node_slices.size()), tail(node_slices.size());
  for (std::size_t k = 0; k < g_arena.regions.size(); ++k) {
    for (std::size_t i = 0; i < node_slices.size(); ++i) {
      if (g_arena.regions[k]->node != node_slices[i].key) { continue; }
      const std::size_t sync = node_slices[i].sync_regions;
      (sync == 0 || head[i].size() < sync ? head[i] : tail[i]).push_back(k);
    }
  }
  // Maps every region of `lists` with per_node_threads threads per key, pinned to its NUMA node.
  const auto run = [s, overflow_per_slice, device, node_slices](const lists_t& lists) {
    std::vector<std::atomic<std::size_t>> next(lists.size());
    for (auto& x : next) { x.store(0); }
    std::vector<std::thread> fill;
    std::vector<std::exception_ptr> err(lists.size() * per_node_threads);
    for (std::size_t i = 0; i < lists.size() * per_node_threads; ++i) {
      const std::size_t ni = i / per_node_threads;
      if (lists[ni].empty()) { continue; }
      const int node = node_slices[ni].numa;
      const int dev  = node_slices[ni].device;
      fill.emplace_back([&, i, ni, node, dev] {
        try {
          cudaSetDevice(dev >= 0 ? dev : device);
          cpu_set_t set;
          if (node_cpus(node, &set)) { pthread_setaffinity_np(pthread_self(), sizeof(set), &set); }
          for (std::size_t q = next[ni].fetch_add(1); q < lists[ni].size(); q = next[ni].fetch_add(1)) {
            region_t& r         = *g_arena.regions[lists[ni][q]];
            const auto t        = std::chrono::steady_clock::now();
            const std::size_t n = r.slices;
            mapping_t m         = map(s * n + align256(overflow_per_slice * n), node, dev);
            std::lock_guard<std::mutex> lk(g_arena.mu);
            r.map    = m;
            r.map_s  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
            r.slices = 0;
            carve(r, s, n, r.first_id);
            r.live.store(true, std::memory_order_release);
          }
        } catch (...) {
          err[i] = std::current_exception();
        }
      });
    }
    for (auto& t : fill) { t.join(); }
    for (auto& e : err) {
      if (e) { std::rethrow_exception(e); }
    }
  };
  run(head);
  g_arena.sync_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bool any_tail = false;
  for (const auto& l : tail) { any_tail |= !l.empty(); }
  if (!any_tail) {
    g_arena.published.store(true);
    g_arena.reserve_s = g_arena.sync_s;
    return true;
  }
  g_arena.background = std::thread([run, tail, t0] {
    try {
      run(tail);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[PINNED-ARENA] background reservation failed: %s\n", e.what());
    }
    g_arena.reserve_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    g_arena.published.store(true);
  });
  return true;
}
}  // namespace

bool reserve_nodes(std::size_t slice_bytes,
                   const std::vector<std::pair<int, std::size_t>>& node_slices,
                   std::size_t overflow_per_slice,
                   std::size_t chunk,
                   const std::vector<std::size_t>& sync_regions)
{
  std::vector<plan_t> plan;
  for (std::size_t i = 0; i < node_slices.size(); ++i) {
    const auto& [node, slices] = node_slices[i];
    plan.push_back({node, node, -1, slices, i < sync_regions.size() ? sync_regions[i] : 0});
  }
  return reserve_plan(slice_bytes, plan, overflow_per_slice, chunk, false);
}

bool reserve_devices(std::size_t slice_bytes,
                     const std::vector<std::pair<int, std::size_t>>& device_slices,
                     std::size_t overflow_per_slice,
                     std::size_t chunk,
                     const std::vector<std::size_t>& sync_regions)
{
  std::vector<plan_t> plan;
  for (std::size_t i = 0; i < device_slices.size(); ++i) {
    const auto& [dev, slices] = device_slices[i];
    plan.push_back({dev, device_node(dev), dev, slices, i < sync_regions.size() ? sync_regions[i] : 0});
  }
  return reserve_plan(slice_bytes, plan, overflow_per_slice, chunk, true);
}

void wait_reserved()
{
  if (g_arena.background.joinable()) { g_arena.background.join(); }
}

int key_for_device(int device) { return g_arena.by_device ? device : device_node(device); }

bool slice_available(int node)
{
  if (!g_arena.ready.load()) { return false; }
  std::lock_guard<std::mutex> lk(g_arena.mu);
  return find_region(node, [](region_t& r) { return !r.free_slices.empty(); }) != nullptr;
}

bool reserved() { return g_arena.ready.load(); }

namespace {
struct claim_t {
  int id{-1}, node{-1};
  void* p{nullptr};
  std::size_t bytes{0};
};
thread_local claim_t tl_claim;
int acquire_free(void** p, std::size_t* bytes, int node);
}  // namespace

bool claim_for_thread(int node)
{
  if (tl_claim.id >= 0) { return true; }
  // No pool starts while registrations are running (they hold the process mmap_lock).
  if (!g_arena.published.load()) { return false; }
  void* p            = nullptr;
  std::size_t bytes  = 0;
  const int id       = acquire_free(&p, &bytes, node);
  if (id < 0) { return false; }
  tl_claim = {id, node, p, bytes};
  return true;
}

void drop_thread_claim()
{
  if (tl_claim.id >= 0) { release_slice(tl_claim.id); }
  tl_claim = {};
}

int acquire_slice(void** p, std::size_t* bytes, int node)
{
  if (tl_claim.id >= 0 && (node < 0 || tl_claim.node == node)) {
    *p           = tl_claim.p;
    *bytes       = tl_claim.bytes;
    const int id = tl_claim.id;
    tl_claim     = {};
    return id;
  }
  return acquire_free(p, bytes, node);
}

namespace {
int acquire_free(void** p, std::size_t* bytes, int node)
{
  if (!g_arena.ready.load()) { return -1; }
  std::lock_guard<std::mutex> lk(g_arena.mu);
  region_t* r = find_region(node, [](region_t& x) { return !x.free_slices.empty(); });
  if (r == nullptr) {
    if (region_t* any = find_region(node, [](region_t&) { return true; })) { ++any->no_slice; }
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

}  // namespace

int acquire_slice_wait(void** p, std::size_t* bytes, int node)
{
  for (;;) {
    const int id = acquire_slice(p, bytes, node);
    if (id >= 0 || !g_arena.ready.load() || g_arena.published.load()) { return id; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
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
  region_t* r = find_region(
    node, [bytes](region_t& x) { return align256(x.overflow_used) + bytes <= x.overflow_bytes; });
  if (r == nullptr) {
    if (region_t* any = find_region(node, [](region_t&) { return true; })) { ++any->overflow_misses; }
    return nullptr;
  }
  const std::size_t off = align256(r->overflow_used);
  r->overflow_used = off + bytes;
  ++r->overflow_takes;
  return r->overflow + off;
}

bool contains(const void* p)
{
  if (!g_arena.ready.load()) { return false; }
  const char* c = static_cast<const char*>(p);
  std::lock_guard<std::mutex> lk(g_arena.mu);
  for (const auto& r : g_arena.regions) {
    if (!r->live.load(std::memory_order_acquire)) { continue; }
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
  std::snprintf(buf, sizeof(buf), "[PINNED-ARENA] regions=%zu per_node=%d by_device=%d sync_s=%.3f reserve_s=%.3f\n",
                g_arena.regions.size(), static_cast<int>(g_arena.per_node),
                static_cast<int>(g_arena.by_device), g_arena.sync_s, g_arena.reserve_s);
  out += buf;
  for (const auto& r : g_arena.regions) {
    std::snprintf(buf,
                  sizeof(buf),
                  "[PINNED-ARENA] key=%d region_MB=%.1f created_in=%.3fs (fault %.3f register %.3f) slices=%zu slice_MB=%.2f acquires=%zu "
                  "peak_in_use=%zu no_free_slice=%zu | overflow used_MB=%.2f of %.2f takes=%zu misses=%zu\n",
                  r->node, r->map.bytes / 1e6, r->map_s, r->map.fault_s, r->map.register_s, r->slices, r->slice_bytes / 1e6, r->acquires,
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
