/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/bad_node_dump.hpp>
#include <branch_and_bound/gpu_node_pool.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase2.hpp>

#include <math_optimization/tic_toc.hpp>

#include <raft/core/error.hpp>
#include <raft/core/nvtx.hpp>

#include <array>
#include <atomic>
#include <cassert>
#include <omp.h>

#include <cstring>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <unistd.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <string>

namespace cuopt::mathematical_optimization::mip {

namespace {

// Test hook. On a single-GPU development machine there is no spare device, so the pool can never
// engage and none of the branch-and-bound wiring can be exercised. Setting this to a device index
// forces the pool onto that device even when it is the only one. NOT a production path: it puts the
// dual simplex on the same device as the heuristics, which is exactly what the design forbids.
int forced_device()
{
  const char* e = std::getenv("CUOPT_BNB_GPU_DEVICE");
  return (e != nullptr) ? std::atoi(e) : -1;
}

}  // namespace

// `get_basis_from_vstatus` INDEXES basic_list but PUSH_BACKS into nonbasic_list and
// superbasic_list -- it does not clear them. Callers that reuse their vectors (the worker's
// basic_list/nonbasic_list live for the whole solve, and a pool seat's live for the whole run) must
// therefore reset them first, or the lists GROW by (n-m) on every node and every consumer that
// iterates the first n-m entries silently reads the FIRST node's basis forever after.
template <typename i_t>
static void basis_from_vstatus_reset(i_t m,
                                     i_t n,
                                     const std::vector<simplex::variable_status_t>& vstatus,
                                     std::vector<i_t>& basic_list,
                                     std::vector<i_t>& nonbasic_list,
                                     std::vector<i_t>& superbasic_list)
{
  basic_list.assign(static_cast<size_t>(m), 0);
  nonbasic_list.clear();
  nonbasic_list.reserve(static_cast<size_t>(n - m));
  superbasic_list.clear();
  simplex::get_basis_from_vstatus<i_t>(m, vstatus, basic_list, nonbasic_list, superbasic_list);
}

template <typename f_t>
static bool gpu_resume_dual_feasible(std::span<const simplex::variable_status_t> vstatus,
                                     std::span<const f_t> z,
                                     f_t dual_tol)
{
  if (z.size() < vstatus.size()) { return false; }
  f_t infeasibility = 0.0;
  for (size_t j = 0; j < vstatus.size(); ++j) {
    if (!std::isfinite(z[j])) { return false; }
    if (vstatus[j] == simplex::variable_status_t::NONBASIC_LOWER && z[j] < -dual_tol) {
      infeasibility += -z[j];
    } else if (vstatus[j] == simplex::variable_status_t::NONBASIC_UPPER && z[j] > dual_tol) {
      infeasibility += z[j];
    } else if (vstatus[j] == simplex::variable_status_t::NONBASIC_FREE) {
      infeasibility += std::abs(z[j]);
    }
    if (infeasibility > dual_tol) { return false; }
  }
  return true;
}


// Devices available to the dual simplex. Not a template: it depends on nothing but the driver, the
// `num_gpus` solver parameter and the CUOPT_BNB_GPU_DEVICE override.
//
// `num_gpus` is the same parameter that sizes the concurrent root LP (PDLP on device 0, barrier on
// device 1 when num_gpus > 1) and that cuopt_cli uses to give devices [0, num_gpus) an RMM pool.
// Honouring it here keeps every dual-simplex allocation on a pooled device; ignoring it put the
// pools on devices that only had the default cudaMalloc resource.
std::vector<int> gpu_usable_devices(int num_gpus)
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) { return {}; }

  const int forced = forced_device();
  if (forced >= 0) {
    if (forced >= count) { return {}; }
    return {forced};
  }
  // Device 0 runs PDLP, feasibility jump and probing. Only the extra requested devices are ours, so
  // num_gpus <= 1 (the default) or a single-GPU machine yields nothing and the pool stays disabled.
  const int limit = num_gpus < 0 ? count : std::min(num_gpus, count);
  std::vector<int> devices;
  for (int d = 1; d < limit; ++d) { devices.push_back(d); }
  return devices;
}


// #4 -- STATIC POOL->DEVICE MAP, computed ONCE before any worker starts.
//
// Workers do not all start at the same moment, so "claim the least-used GPU when you arrive" would
// hand out devices in thread-arrival order -- which is not reproducible and would make the device a
// worker is on vary run to run. Instead the assignment is computed up front from two numbers that
// are both known at that point: the usable devices and the BFS worker count. Each worker then
// claims its pre-assigned slot on first use, so the outcome is identical whatever order they arrive
// in.
//
// Round-robin by worker id: with 11 workers on 2 GPUs this gives 6/5 and is independent of thread
// arrival order.
std::vector<int> gpu_pool_device_map(int num_workers, int num_gpus)
{
  const auto devices = gpu_usable_devices(num_gpus);
  std::vector<int> map(static_cast<size_t>(std::max(num_workers, 0)), -1);
  if (devices.empty() || num_workers <= 0) { return map; }
  const int nd = static_cast<int>(devices.size());
  // CUOPT_BNB_GPU_POOLS_PER_GPU=K: only the first K * devices workers own a pool; the rest are CPU-only.
  static const int per_gpu = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_POOLS_PER_GPU");
    return e != nullptr ? std::max(0, std::atoi(e)) : 0;
  }();
  const int with_pool = per_gpu > 0 ? std::min(num_workers, per_gpu * nd) : num_workers;
  for (int w = 0; w < with_pool; ++w) {
    map[static_cast<size_t>(w)] = devices[static_cast<size_t>(w % nd)];
  }
  return map;
}

// How many pools land on each usable device, for the [CONFIG] line. Reporting it is the only way to
// see that the split is what was intended without instrumenting a run.
std::vector<std::pair<int, int>> gpu_pool_device_load(const std::vector<int>& map)
{
  std::vector<std::pair<int, int>> load;
  for (int d : map) {
    if (d < 0) { continue; }
    auto it = std::find_if(load.begin(), load.end(), [d](const auto& e) { return e.first == d; });
    if (it == load.end()) { load.emplace_back(d, 1); } else { ++it->second; }
  }
  std::sort(load.begin(), load.end());
  return load;
}

bool gpu_serial_enabled()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_SERIAL");
    const bool enabled = e != nullptr && std::string(e) != "0";
    // Announce it once. A run that silently did NOT use the GPU would otherwise look like a clean
    // pass, which is the failure mode this whole test path exists to avoid.
    if (enabled) {
      const auto devs = gpu_usable_devices();
      std::fprintf(stderr,
                   "[CONFIG] branch-and-bound node LPs -> GPU (serial substitution, batch of 1); "
                   "usable devices=%zu%s\n",
                   devs.size(),
                   devs.empty() ? " -- NONE, falling back to the CPU driver" : "");
    }
    return enabled;
  }();
  return on;
}

// TEST-ONLY diagnostic (CUOPT_BNB_GPU_VERIFY=1), meaningful only together with the serial
// substitution above. Every node is solved TWICE -- once on the device, once on the CPU with the
// FULL driver the branch-and-bound would otherwise have used -- and any node where the two reach a
// different FATHOM decision is reported. This is what turns "the tree came out different" into "node
// X disagrees", which is the only tractable way to debug a search-level divergence.
// 0 = off. 1 = the GATE only: re-solve each node with the CPU FULL driver (what production uses)
// and compare the prune decision. 2 = also the two DIAGNOSTIC legs -- the CPU SIMPLIFIED driver
// (separates "the port is wrong" from "the minimal algorithm is too weak") and a re-solve from the
// basis the device returned. Level 2 costs three extra CPU solves per node, so it roughly triples
// the run; use it when investigating, not when sweeping.
int gpu_serial_verify_level()
{
  static const int level = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_VERIFY");
    return (e != nullptr) ? std::atoi(e) : 0;
  }();
  return level;
}
bool gpu_serial_verify() { return gpu_serial_verify_level() > 0; }

// Directory for wrong-decision node dumps, or nullptr when disabled. Off by default: these are
// multi-megabyte binary images and only wanted while a defect is being chased.
const char* dump_bad_nodes_dir()
{
  static const char* dir = std::getenv("CUOPT_BNB_DUMP_BAD_NODES");
  return (dir != nullptr && dir[0] != '\0') ? dir : nullptr;
}


// FNV-1a over a double array's raw bits. Used only to localise WHERE a vector changes between two
// points in this function; a checksum answers "did it change", and printing the vector does not.
static unsigned long long dbg_hash_vs(const std::vector<simplex::variable_status_t>& v)
{
  unsigned long long h = 1469598103934665603ull;
  for (size_t k = 0; k < v.size(); ++k) {
    h ^= static_cast<unsigned long long>(static_cast<int>(v[k]));
    h *= 1099511628211ull;
  }
  return h;
}

template <typename f_t>
static unsigned long long dbg_hash(const std::vector<f_t>& v)
{
  unsigned long long h = 1469598103934665603ull;
  for (size_t k = 0; k < v.size(); ++k) {
    unsigned long long bits;
    std::memcpy(&bits, &v[k], sizeof(bits));
    h ^= bits;
    h *= 1099511628211ull;
  }
  return h;
}

// Shared by the serial verify leg and the pool's CPU recheck gate. Documented in the header.
template <typename i_t, typename f_t>
bool gpu_decision_differs(simplex::dual_status_t gpu_status,
                          simplex::dual_status_t cpu_status,
                          f_t gpu_objective,
                          f_t cpu_objective,
                          f_t cut_off,
                          bool& cpu_decided,
                          bool& objective_differs)
{
  // Both statuses that make branch-and-bound PRUNE count, not just infeasibility: a spurious CUTOFF
  // fathoms exactly as destructively as a spurious DUAL_UNBOUNDED.
  auto prunes = [](simplex::dual_status_t st) {
    return st == simplex::dual_status_t::DUAL_UNBOUNDED || st == simplex::dual_status_t::CUTOFF;
  };
  // The CPU leg must actually have REACHED a verdict. ITERATION_LIMIT / TIME_LIMIT / NUMERICAL mean
  // it ran out of budget and returned obj=nan, so counting the difference is a false positive.
  // Measured: every flagged mismatch on csched008, seymour and wachplan after the basis-list fix was
  // this case.
  cpu_decided = cpu_status == simplex::dual_status_t::OPTIMAL ||
                cpu_status == simplex::dual_status_t::DUAL_UNBOUNDED ||
                cpu_status == simplex::dual_status_t::CUTOFF;

  const bool both_optimal = gpu_status == simplex::dual_status_t::OPTIMAL &&
                            cpu_status == simplex::dual_status_t::OPTIMAL;
  const f_t obj_tol = f_t{1e-6} * std::max(f_t{1.0}, std::fabs(cpu_objective));
  objective_differs = both_optimal && std::fabs(gpu_objective - cpu_objective) > obj_tol;

  // CUTOFF is an EARLY EXIT, not a verdict about the subtree: the driver stops the moment its
  // tracked objective reaches cut_off. So CUTOFF on one side and OPTIMAL on the other is a DEFECT
  // only when the side that solved to optimality would NOT itself have been cut off -- i.e. the
  // other side pruned a node still capable of improving the incumbent. When both values sit past
  // the cutoff, branch-and-bound fathoms either way and the two agree.
  auto cutoff_vs_optimal_disagrees = [&](simplex::dual_status_t cut_side,
                                         simplex::dual_status_t opt_side,
                                         f_t opt_objective) {
    if (cut_side != simplex::dual_status_t::CUTOFF ||
        opt_side != simplex::dual_status_t::OPTIMAL) {
      return false;   // not this pair; the generic prune test applies
    }
    if (!std::isfinite(cut_off)) { return true; }   // an infinite cutoff can never fire
    return opt_objective < cut_off;                 // still promising -> the prune lost a subtree
  };

  const bool gpu_prunes = prunes(gpu_status);
  const bool cpu_prunes = prunes(cpu_status);
  if (gpu_prunes != cpu_prunes && (gpu_status == simplex::dual_status_t::CUTOFF ||
                                   cpu_status == simplex::dual_status_t::CUTOFF)) {
    return cutoff_vs_optimal_disagrees(gpu_status, cpu_status, cpu_objective) ||
           cutoff_vs_optimal_disagrees(cpu_status, gpu_status, gpu_objective);
  }
  return gpu_prunes != cpu_prunes;
}

template bool gpu_decision_differs<int, double>(
  simplex::dual_status_t, simplex::dual_status_t, double, double, double, bool&, bool&);

template <typename i_t, typename f_t>
bool gpu_solve_node_serial(const simplex::lp_problem_t<i_t, f_t>& lp,
                           const csr_matrix_t<i_t, f_t>& /*Arow*/,
                           const std::vector<simplex::variable_type_t>& /*var_types*/,
                           const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                           const std::vector<f_t>& lower,
                           const std::vector<f_t>& upper,
                           std::vector<simplex::variable_status_t>& vstatus,
                           std::vector<i_t>& basic_list,
                           std::vector<i_t>& nonbasic_list,
                           simplex::lp_solution_t<i_t, f_t>& sol,
                           std::vector<f_t>& dy_se,
                           i_t& iters,
                           simplex::dual_status_t& status)
{
  static const std::vector<int> devices = gpu_usable_devices();
  if (devices.empty()) { return false; }

  // DIAGNOSTIC (CUOPT_BNB_GPU_ONE_THREAD=1): let only the FIRST worker thread that gets here use
  // the device; every other thread takes the CPU driver. Several workers otherwise interleave their
  // trace output, and lines from different threads cannot be paired -- which would make a device
  // line and an unrelated CPU line look like a divergence. Not a production path: it serialises the
  // offload.
  static const bool one_thread = std::getenv("CUOPT_BNB_GPU_ONE_THREAD") != nullptr;
  if (one_thread) {
    static std::atomic<bool> claimed{false};
    thread_local bool decided  = false;
    thread_local bool is_owner = false;
    if (!decided) {
      bool expected = false;
      is_owner      = claimed.compare_exchange_strong(expected, true);
      decided       = true;
    }
    if (!is_owner) { return false; }
  }

  const i_t m = lp.num_rows;

  // Every B&B LP starts from the root steepest-edge vector. An empty vector is the generic dual
  // simplex cold-start signal and must not be used by this path (basis repair is rejected below).
  assert(dy_se.size() == static_cast<size_t>(lp.num_cols));

  // Snapshot the warm start BEFORE anything touches it: `refactor_basis` may repair `vstatus`, and
  // the batch overwrites it with the FINAL basis. The cross-check needs the inputs as they arrived.
  const bool verify = gpu_serial_verify();
  std::vector<simplex::variable_status_t> vstatus_ref;
  std::vector<f_t> dy_se_ref;
  if (verify) {
    vstatus_ref = vstatus;
    dy_se_ref   = dy_se;
    if (dump_bad_nodes_dir() != nullptr) {
      std::fprintf(stderr, "[DYSEWHERE] tid=%d entry %016llx ptr=%p\n", omp_get_thread_num(), dbg_hash(dy_se), (const void*)&dy_se);
    }
  }

  // The batch requires a FRESH factorization of THIS node's basis: pure MPF stacking on the device,
  // no on-device refactorization. `ft` is function-local so nothing stale can leak into it.
  simplex::basis_update_mpf_t<i_t, f_t> ft(m, settings.refactor_frequency);
  std::vector<i_t> superbasic_list;
  basis_from_vstatus_reset<i_t>(m, lp.num_cols, vstatus, basic_list, nonbasic_list, superbasic_list);

  const f_t t0 = cuopt::mathematical_optimization::tic();
  // Factorize the current basis. Periodic and terminal-certificate resumes are independently priced
  // and rejected if the returned basis is not dual feasible. A rejected-update resume is exempt:
  // its negative-step/cutoff path has its own clamp+tolerance guard and preserves the established
  // recovery trajectory.
  auto refactorize = [&](simplex::lp_solution_t<i_t, f_t>& s,
                         bool initialize_basis,
                         bool certify_resume) {
    // FILL ONLY -- see the batch path. On a resume the device has already returned the lists in the
    // order it maintains them; rebuilding them from `vstatus` re-sorts both and changes both the LU
    // and the order the ratio test scans the nonbasics in.
    if (initialize_basis) {
      basis_from_vstatus_reset<i_t>(
        m, lp.num_cols, vstatus, basic_list, nonbasic_list, superbasic_list);
    }
    bool repaired = false;
    if (ft.refactor_basis(
          lp.A, settings, lower, upper, t0, basic_list, nonbasic_list, vstatus, &repaired) != 0 ||
        repaired) {
      return false;   // singular / repaired basis: let the CPU driver handle this node
    }
    // The device consumes the duals as INPUTS; the simplified driver it ports has no dual prologue.
    // On fill, price them into the solution. On certified resumes, fresh pricing is an independent
    // check: reject a dual-infeasible basis, but preserve the maintained duals and their trajectory.
    f_t work = 0.0;
    if (initialize_basis) {
      simplex::phase2::compute_dual_solution_from_basis(
        lp, ft, basic_list, nonbasic_list, s.y, s.z, work);
    } else if (certify_resume) {
      std::vector<f_t> fresh_y(m, 0.0), fresh_z(lp.num_cols, 0.0);
      simplex::phase2::compute_dual_solution_from_basis(
        lp, ft, basic_list, nonbasic_list, fresh_y, fresh_z, work);
      if (!gpu_resume_dual_feasible<f_t>(vstatus, fresh_z, settings.dual_tol)) { return false; }
    }
    return true;
  };

  // The driver expects a CLEAN solution object -- it starts the primal point from the basis and the
  // bounds, and accumulates the iteration count. `sol` is the worker's `leaf_solution`, reused for
  // every node it ever solves, so handing it over directly would start this node from the PREVIOUS
  // node's primal point. Solve into a clean one and publish only the outputs.
  simplex::lp_solution_t<i_t, f_t> node_sol(m, lp.num_cols);
  if (!refactorize(node_sol, /*initialize_basis*/ true, /*certify_resume*/ false)) {
    return false;
  }
  if (verify && dump_bad_nodes_dir() != nullptr) {
    std::fprintf(stderr, "[DYSEWHERE] tid=%d after-price %016llx\n", omp_get_thread_num(), dbg_hash(dy_se));
  }

  // DIAGNOSTIC (CUOPT_BNB_DY_PROBE=<column index>): recompute delta_y = B^-T e_r on the HOST, from
  // the very factorization that is about to be uploaded, for the basic position of that column.
  // The device reports its own delta_y nnz in the [GMIN] trace, so this says whether the device's
  // BTran agrees with the CPU's on identical input -- which is the whole question behind a false
  // infeasibility certificate: an empty ratio test is only valid if delta_y is right.
  {
    static const int probe_col = [] {
      const char* e = std::getenv("CUOPT_BNB_DY_PROBE");
      return (e != nullptr) ? std::atoi(e) : -1;
    }();
    if (probe_col >= 0) {
      for (i_t k = 0; k < m; ++k) {
        if (basic_list[k] != probe_col) { continue; }
        sparse_vector_t<i_t, f_t> ei(m, 1), dy(m, 0), utsol(m, 0);
        ei.i[0] = k;
        ei.x[0] = -1;   // -direction for direction = +1, as compute_delta_y builds it
        ft.b_transpose_solve(ei, dy, utsol);
        std::fprintf(stderr,
                     "[DYPROBE] col=%d basic_pos=%d host_dy_nz=%d\n",
                     probe_col, (int)k, (int)dy.i.size());
        break;
      }
    }
  }

  // The basis EXACTLY as uploaded. `refactor_basis` may repair vstatus, so the entry snapshot is
  // not what the device was given -- comparing against that snapshot would make the CPU start from
  // a different basis and manufacture a divergence that is not there.
  std::vector<simplex::variable_status_t> vstatus_uploaded;
  if (verify) { vstatus_uploaded = vstatus; }
  // x / y / z are INPUTS to the device (`a.x/.y/.z = upload(sol.x/.y/.z)`), so a replay that
  // rebuilds them instead of reproducing them is not replaying the same solve. Snapshot them here,
  // at the state the first launch uploads.
  std::vector<f_t> x_uploaded, y_uploaded, z_uploaded;
  std::vector<f_t> dy_se_at_launch, x_at_launch, y_at_launch, z_at_launch;
  // The LU factor is the last input a replay REBUILDS instead of reproducing: everything else is an
  // array the dump copies verbatim. Snapshot what the first launch actually uploads, so a replay
  // that disagrees can be tested against the factorization rather than assumed identical to it.
  std::vector<f_t> l0x_uploaded, u0x_uploaded;
  std::vector<i_t> l0cs_uploaded, u0cs_uploaded, rowperm_uploaded;
  if (verify) {
    x_uploaded       = node_sol.x;
    y_uploaded       = node_sol.y;
    z_uploaded       = node_sol.z;
    l0x_uploaded     = ft.L0().x;
    u0x_uploaded     = ft.U0().x;
    l0cs_uploaded    = ft.L0().col_start;
    u0cs_uploaded    = ft.U0().col_start;
    rowperm_uploaded = ft.row_permutation();
  }

  gds::gpu_batch_item<i_t, f_t> it{};
  it.lp            = &lp;
  it.lower         = &lower;      // per-node bounds; the shared lp carries A/rhs/objective
  it.upper         = &upper;
  it.settings      = &settings;
  it.vstatus       = &vstatus;
  it.ft            = &ft;
  it.basic_list    = &basic_list;
  it.nonbasic_list = &nonbasic_list;
  it.sol           = &node_sol;
  // Why a REFACTOR_NEEDED leg stopped: 0 = periodic schedule, 1 = rejected rank-1 update.
  int refactor_reason = 0;
  it.refactor_reason  = &refactor_reason;
  it.dy_se         = &dy_se;

  // Size contract for everything the marshal uploads. The kernel indexes these by column/row index
  // with no bounds check, so a short vector is not a wrong answer -- it is an out-of-bounds device
  // read, which surfaces far away as `cudaErrorIllegalAddress` on the next synchronising call and
  // says nothing about which array was wrong. Check here, where the answer is one line.
  {
    const i_t n_cols = lp.num_cols;
    const char* bad  = nullptr;
    if (static_cast<i_t>(vstatus.size()) != n_cols) { bad = "vstatus"; }
    else if (static_cast<i_t>(basic_list.size()) != m) { bad = "basic_list"; }
    else if (static_cast<i_t>(nonbasic_list.size()) != n_cols - m) { bad = "nonbasic_list"; }
    else if (static_cast<i_t>(dy_se.size()) != n_cols) { bad = "dy_se"; }
    else if (static_cast<i_t>(node_sol.x.size()) != n_cols) { bad = "sol.x"; }
    else if (static_cast<i_t>(node_sol.y.size()) != m) { bad = "sol.y"; }
    else if (static_cast<i_t>(node_sol.z.size()) != n_cols) { bad = "sol.z"; }
    if (bad != nullptr) {
      std::fprintf(stderr,
                   "[GPUSIZE] refusing launch: %s has the wrong length "
                   "(m=%d n=%d vstatus=%zu basic=%zu nonbasic=%zu dy_se=%zu x=%zu y=%zu z=%zu)\n",
                   bad, (int)m, (int)n_cols, vstatus.size(), basic_list.size(),
                   nonbasic_list.size(), dy_se.size(), node_sol.x.size(), node_sol.y.size(),
                   node_sol.z.size());
      return false;   // fall back to the CPU driver rather than corrupt device memory
    }
  }

  std::vector<gds::gpu_batch_item<i_t, f_t>> items{it};
  std::vector<simplex::dual_status_t> out;
  gds::gpu_batch_timing timing;
  // EXPLICIT STREAM, one per worker thread. The synchronous batch entry runs on the LEGACY default
  // stream, and cuOpt captures CUDA graphs elsewhere in the solver -- once a capture is live, any
  // legacy-stream work fails with cudaErrorStreamCaptureImplicit ("would make the legacy stream
  // depend on a capturing blocking stream") and aborts the process. It is intermittent, because it
  // needs a capture to be in flight on another thread at the same moment.
  //
  // thread_local, not static: several branch-and-bound workers call this concurrently and each must
  // own its stream, exactly as the pool gives every slice its own.
  static thread_local rmm::cuda_stream worker_stream;

  // REFACTOR-AND-RESUME. The device stops with REFACTOR_NEEDED when its rank-1 update is unsafe or
  // its preallocated update store is full -- the same condition the CPU full driver answers by
  // refactorizing IN PLACE and continuing. The device cannot: it has no LU of its own. So the host
  // does that rebuild and calls it straight back.
  //
  // Without this the node was abandoned to a full CPU re-solve at the 256-update cap, which on
  // csched008 is most long nodes -- so the GPU never finished the very solves it needs to be
  // measured on, and no GPU-vs-CPU comparison of a long node was possible.
  //
  // The resume is exact, not a restart: the device hands back the basis it reached (vstatus), its
  // primal point, its objective and its steepest-edge norms, and every one of those is fed
  // straight back in. Only the factorization is rebuilt.
  //
  // The node is bounded by its CUMULATIVE iteration count (see the check in the loop), which is the
  // same bound the CPU plunging worker gets. A leg count is deliberately NOT the bound: a
  // legitimate long node needs many legs -- one per `refactor_frequency` pivots -- and capping legs
  // gives up on nodes the CPU would simply keep solving. The only leg-based guard is on legs that
  // take ZERO pivots, which is a genuine livelock and cannot be caught by an iteration count.
  // DIAGNOSTIC (CUOPT_BNB_GPU_LOCK=1): serialise every worker's device work behind one mutex. Each
  // worker owns its stream and its buffers, so if a defect disappears under this lock it is
  // cross-worker interference, and if it survives the lock it is not. Nothing else distinguishes
  // the two, because a wrong objective looks the same either way.
  static const bool serialize_gpu = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LOCK");
    return e != nullptr && std::string(e) != "0";
  }();
  static std::mutex gpu_lock;
  std::unique_lock<std::mutex> gpu_guard(gpu_lock, std::defer_lock);
  if (serialize_gpu) { gpu_guard.lock(); }

  i_t total_iters           = 0;
  int resumes               = 0;
  int zero_pivot_legs       = 0;
  // Set when the loop stops on an iteration bound rather than on a device answer. Needed as a
  // separate flag because `status` is overwritten from out[0] after the loop, which would turn
  // this into a leaked non-terminal REFACTOR_NEEDED.
  bool stopped_on_iter_limit = false;
  while (true) {
    // The steepest-edge vector AS UPLOADED by the first launch. Everything else the dump records is
    // snapshotted at function entry, which is only the same thing if nothing touches it in between.
    if (verify && resumes == 0 && dump_bad_nodes_dir() != nullptr) {
      std::fprintf(stderr, "[DYSEWHERE] tid=%d at-launch %016llx ptr=%p\n", omp_get_thread_num(),
                   dbg_hash(*items[0].dy_se), (const void*)items[0].dy_se);
    }
    if (verify && resumes == 0) {
      dy_se_at_launch = *items[0].dy_se;
      x_at_launch     = node_sol.x;
      y_at_launch     = node_sol.y;
      z_at_launch     = node_sol.z;
    }
    auto* handle = gds::gpu_simplified_dual_phase2_batch_launch<i_t, f_t>(
      items, timing, worker_stream.view(), /*allow_refactor_exit*/ true);
    if (handle == nullptr) { return false; }   // rejected: timing.message says needed-vs-free
    gds::gpu_simplified_dual_phase2_batch_finish<i_t, f_t>(handle, out, timing);
    if (timing.rejected || out.empty()) { return false; }

    // ASSIGN, not accumulate: the device carries `iter` across a resume now, so what comes back
    // is already this node's running total. Accumulating would count every leg twice.
    total_iters = node_sol.iterations;
    // Per-leg trace while capturing bad nodes: a replay that disagrees with the live solve has to
    // be pinned to a LEG before it can be pinned to a cause, and the node-level numbers alone
    // cannot say whether the two paths parted at the first launch or after a resume.
    if (dump_bad_nodes_dir() != nullptr) {
      // The BASIS hash is the point. A node that pivots without moving the objective is either
      // CYCLING -- the hash repeats, so the solve revisits a basis it already had -- or STALLING on
      // degeneracy, where the hash keeps changing while the objective does not. Those have different
      // causes and the objective alone cannot tell them apart.
      const unsigned long long vs_h = dbg_hash_vs(vstatus);
      static thread_local std::vector<unsigned long long> seen_bases;
      if (resumes == 0) { seen_bases.clear(); }
      int repeat_of = -1;
      for (size_t k = 0; k < seen_bases.size(); ++k) {
        if (seen_bases[k] == vs_h) { repeat_of = static_cast<int>(k); break; }
      }
      seen_bases.push_back(vs_h);
      std::fprintf(stderr,
                   "[BADNODE-LEG] %d %s it=%d cum=%d obj=%.17g vs=%016llx x=%016llx dyse=%016llx\n",
                   resumes, simplex::dual_status_to_string(out[0]).c_str(), node_sol.iterations,
                   (int)total_iters, node_sol.objective, vs_h, dbg_hash(node_sol.x),
                   dbg_hash(*items[0].dy_se));
      if (repeat_of >= 0) {
        std::fprintf(stderr, "[BADNODE-CYCLE] leg %d repeats the basis of leg %d\n",
                     resumes, repeat_of);
      }
    }
    if (out[0] != simplex::dual_status_t::REFACTOR_NEEDED) { break; }

    // SAME BOUND AS THE CPU PLUNGING WORKER: the node's cumulative iteration count, not a count of
    // resume legs. `settings.iteration_limit` is exactly what solve_node_lp was handed, so the two
    // workers read the same number -- today that is INT_MAX for a main-tree plunge node, so this
    // check is inert and the node is bounded only by the time limit, like the CPU's.
    //
    // It is deliberately not a LEG count: the CPU refactorizes in place an unbounded number of
    // times, so capping legs would give up on a node the CPU would keep solving -- measured
    // elsewhere at ~68 legs for greenbeb and ~218 for dfl001 at the default refactor frequency.
    //
    // Returning ITERATION_LIMIT (rather than false) is what makes this the same behaviour: the node
    // goes back on the queue exactly as a CPU ITERATION_LIMIT node does, instead of being re-solved
    // from scratch on the CPU.
    // DIAGNOSTIC (CUOPT_BNB_GPU_ITER_CAP=<n>, 0 = off): a second, much tighter cap, so a node that
    // STALLS can be made to terminate and therefore reach the verification block below. Without it a
    // stalling node never completes, never emits a [GPUVERIFY] line, and so can never be compared
    // against the CPU drivers -- which is exactly the node we most need to compare.
    static const i_t diag_iter_cap = [] {
      const char* e = std::getenv("CUOPT_BNB_GPU_ITER_CAP");
      return (e != nullptr) ? static_cast<i_t>(std::atoi(e)) : i_t{0};
    }();
    const bool hit_cap = (diag_iter_cap > 0 && total_iters >= diag_iter_cap);

    if (hit_cap || total_iters >= settings.iteration_limit) {
      stopped_on_iter_limit = true;
      // BREAK, not return: the verification block after this loop is the whole point of the
      // diagnostic cap. Returning here would skip it.
      break;
    }

    // Livelock guard, and the ONLY thing a leg count is good for: a leg that takes zero pivots has
    // made no progress, so `total_iters` would never reach the cap above. One is normal; several in
    // a row is a stuck node.
    if (node_sol.iterations == 0) {
      if (++zero_pivot_legs >= 8) { return false; }
    } else {
      zero_pivot_legs = 0;
    }
    ++resumes;
    // Rebuild the factorization of the basis the device stopped on, and re-derive the duals from
    // it. node_sol.x / .objective and dy_se already hold the device's state and are passed back
    // unchanged, so the next launch continues rather than restarts.
    // A/B KNOB ONLY (CUOPT_BNB_GPU_RESUME_PRICE=1), default OFF. The CPU never re-prices at a
    // refactorization, so neither do we: the device now CONTINUES the solve instead of restarting
    // it, which is what makes the maintained duals valid across a resume. Kept because it was the
    // measurement that localised the defect.
    //
    // WHY, and why this differs from the CPU. The CPU refactorizes IN PLACE and keeps iterating, so
    // carrying its incrementally maintained y/z across a refactorization is continuous and correct.
    // A GPU resume is not that: the device exits, and the next launch runs setup() again, which
    // recomputes the primal infeasibilities, the objective and the working set from scratch. The
    // device has NO dual prologue -- setup() never computes y/z, it consumes whatever the host
    // uploads -- so feeding maintained duals into that restart mixes two representations: a basis
    // freshly factorized on the host against reduced costs accumulated against the old one.
    //
    // Measured on csched008: carrying them, one node pivoted 686 times over 69,972 iterations
    // without terminating, already sitting at its optimum (obj=173), the basis never repeating --
    // so not cycling, just never satisfying the optimality test. Re-pricing: the same instance goes
    // from 0 solves / 2 nodes to 8 / 20. germanrr 28 -> 32, mcsched 1086 -> 1202, zero mismatches.
    //
    // It is NOT drift. The same probe shows germanrr, which converges, carrying the same
    // 5e-06..1.4e-04 dual error as csched008 does.
    //
    // Cost is one BTRAN plus a pass over the nonbasics per resume, against a kernel launch.
    if (!refactorize(node_sol,
                     /*initialize_basis*/ false,
                     /*certify_resume*/ refactor_reason != 1)) {
      return false;
    }

    // Rebuild the primal point the way the CPU's in-place refactor does, and tell the device not to
    // redo it with its cold-start formula. Without this the two part at the first refactor.
    //
    // This calls the CPU function ITSELF, not a GPU copy of it -- the solver's own resume path calls
    // the same one, so there is no second formula to keep in step.
    //
    // It reads the bounds from `lp`, which is correct HERE because this entry is handed
    // `worker->leaf_problem`, whose lower/upper are this node's. A POOL that shares one
    // lp_problem_t across seats and carries per-seat bounds cannot call it this way -- it will need
    // a bounds-explicit overload.
    std::vector<f_t> xB_workspace(m, 0.0);
    f_t work = 0.0;
    simplex::phase2::compute_primal_solution_from_basis(
      lp, ft, basic_list, nonbasic_list, vstatus, node_sol.x, xB_workspace, work);
    items[0].refactor_resume = true;
  }

  status = stopped_on_iter_limit ? simplex::dual_status_t::ITERATION_LIMIT : out[0];
  iters  = total_iters;
  sol.x              = node_sol.x;
  sol.objective      = node_sol.objective;
  sol.user_objective = node_sol.user_objective;
  // total_iters, not the last leg's count: a node interrupted by REFACTOR_NEEDED is ONE solve, and
  // the out-param `iters` already reports it that way. Publishing the last leg here made the verify
  // line and the bad-node dumps understate a resumed solve (142 rather than 142 of 175).
  sol.iterations     = total_iters;
  // `basic_list` / `nonbasic_list` already hold the FINAL basis: the batch copies them back next to
  // `vstatus`, in the order the device maintains them (in-place swap, i.e. pivot history).
  //
  // Do NOT re-derive them from `vstatus` here. get_basis_from_vstatus scans j ascending, so it
  // returns the same basis SET in SORTED order -- a different permutation. That matters twice: the
  // caller's next refactorization factorizes A(:, basic_list), and the ratio test scans
  // nonbasic_list IN ORDER with an order-dependent near-tie rule (see warp_rt_body). Re-sorting
  // here silently changes both on every node the device solves.

  if (verify) {
    // The two DIAGNOSTIC legs below cost three extra CPU solves per node. They answered their
    // questions during bring-up, so they are opt-in now (CUOPT_BNB_GPU_VERIFY=2) and a plain
    // sweep runs only the gate.
    const bool deep = gpu_serial_verify_level() >= 2;

    // Fourth leg, run FIRST because it is the one that separates "the device computed the wrong
    // answer" from "the device computed the right answer and labelled it wrong": take the basis the
    // device RETURNED and re-solve from it on the CPU. If that comes back OPTIMAL in zero
    // iterations, the device had the optimal basis in hand when it reported infeasible.
    std::vector<simplex::variable_status_t> vstatus_out = vstatus;
    std::vector<f_t> dy_se_out                          = dy_se_ref;
    simplex::basis_update_mpf_t<i_t, f_t> ft_out(m, settings.refactor_frequency);
    std::vector<i_t> basic_out(m), nonbasic_out, superbasic_out;
    simplex::lp_solution_t<i_t, f_t> sol_out(m, lp.num_cols);
    simplex::dual_status_t status_out = simplex::dual_status_t::UNSET;
    if (deep) {
      simplex::get_basis_from_vstatus<i_t>(m, vstatus_out, basic_out, nonbasic_out, superbasic_out);
      const f_t tout = cuopt::mathematical_optimization::tic();
      if (ft_out.refactor_basis(
            lp.A, settings, lower, upper, tout, basic_out, nonbasic_out, vstatus_out) == 0) {
        f_t work_out = 0.0;
        simplex::phase2::compute_dual_solution_from_basis(
          lp, ft_out, basic_out, nonbasic_out, sol_out.y, sol_out.z, work_out);
        status_out = simplex::simplified_dual_phase2<i_t, f_t>(
          lp, settings, vstatus_out, ft_out, basic_out, nonbasic_out, sol_out, dy_se_out);
      }
    }

    // Third leg: the CPU SIMPLIFIED driver -- the exact algorithm the device ports. Running it on
    // the same inputs separates "the GPU port is wrong" from "the minimal algorithm is too weak for
    // this node", which need completely different fixes.
    std::vector<simplex::variable_status_t> vstatus_min = vstatus_uploaded;
    std::vector<f_t> dy_se_min                          = dy_se_ref;
    simplex::basis_update_mpf_t<i_t, f_t> ft_min(m, settings.refactor_frequency);
    std::vector<i_t> basic_min(m), nonbasic_min, superbasic_min;
    simplex::lp_solution_t<i_t, f_t> sol_min(m, lp.num_cols);
    simplex::dual_status_t status_min = simplex::dual_status_t::UNSET;
    if (deep) {
      simplex::get_basis_from_vstatus<i_t>(
        m, vstatus_min, basic_min, nonbasic_min, superbasic_min);
      const f_t tmin = cuopt::mathematical_optimization::tic();
      if (ft_min.refactor_basis(
            lp.A, settings, lower, upper, tmin, basic_min, nonbasic_min, vstatus_min) == 0) {
        f_t work_min = 0.0;
        simplex::phase2::compute_dual_solution_from_basis(
          lp, ft_min, basic_min, nonbasic_min, sol_min.y, sol_min.z, work_min);
        status_min = simplex::simplified_dual_phase2<i_t, f_t>(
          lp, settings, vstatus_min, ft_min, basic_min, nonbasic_min, sol_min, dy_se_min);
      }
    }

    simplex::basis_update_mpf_t<i_t, f_t> ft_ref(m, settings.refactor_frequency);
    // `get_basis_from_vstatus` INDEXES basic_list (it only push_backs the other two), so the caller
    // owns its sizing.
    std::vector<i_t> basic_ref(m), nonbasic_ref;
    simplex::lp_solution_t<i_t, f_t> sol_ref(m, lp.num_cols);
    i_t iter_ref = 0;
    // `dual_phase2_with_advanced_basis` takes the steepest-edge vector by NON-CONST reference and
    // updates it in place. Passing `dy_se_ref` handed it the very snapshot this function records as
    // the node's input, so a captured dump held a vector the CPU reference solve had rewritten --
    // and a replay from it started somewhere the device never was. Give the reference solve its own
    // copy; the snapshot has to stay the input.
    std::vector<f_t> dy_se_ref_scratch = dy_se_ref;
    const simplex::dual_status_t status_ref =
      simplex::dual_phase2_with_advanced_basis<i_t, f_t>(2,
                                                         0,
                                                         true,  // factorize vstatus_ref ourselves
                                                         cuopt::mathematical_optimization::tic(),
                                                         lp,
                                                         settings,
                                                         vstatus_ref,
                                                         ft_ref,
                                                         basic_ref,
                                                         nonbasic_ref,
                                                         sol_ref,
                                                         iter_ref,
                                                         dy_se_ref_scratch);

    // ONE implementation of the decision test, shared with the pool's recheck gate -- see
    // gpu_decision_differs. A second copy would drift from this one, and the two would then disagree
    // about what counts as a divergence.
    bool cpu_decided = false, obj_differs = false;
    const bool decision_differs = gpu_decision_differs<i_t, f_t>(
      status, status_ref, sol.objective, sol_ref.objective, settings.cut_off, cpu_decided,
      obj_differs);
    const f_t obj_gap = obj_differs ? std::fabs(sol.objective - sol_ref.objective) : f_t{0.0};
    const f_t obj_tol = f_t{0.0};

    // Capture the node whenever the decision differs. These failures are rare and the tree is
    // non-deterministic, so a node that mis-decides once is usually never revisited -- without a
    // dump the only honest report is "it did not reproduce". Written under
    // CUOPT_BNB_DUMP_BAD_NODES=<dir>; replay with `test_netlib --replay-node <file>`.
    // Also capture a node stopped by the DIAGNOSTIC cap. That node is a STALL -- the device pivots
    // without moving the objective -- and it is the one we most need as a replayable artifact,
    // precisely because it never terminates on its own and so never reaches the normal dump test.
    const bool capped = (status == simplex::dual_status_t::ITERATION_LIMIT);
    if ((capped || (cpu_decided && (decision_differs || obj_gap > obj_tol))) &&
        dump_bad_nodes_dir() != nullptr) {
      static std::atomic<int> seq{0};
      char path[1024];
      std::snprintf(path, sizeof(path), "%s/badnode_m%d_n%d_%04d.bin",
                    dump_bad_nodes_dir(), (int)m, (int)lp.num_cols, seq.fetch_add(1));
      // vstatus_uploaded, not the entry snapshot: refactor_basis may have repaired it, and what the
      // drivers consumed is what has to be reproduced. Same for `lower`/`upper`, which are already
      // bound-strengthened.
      const bool ok = bad_node::write<i_t, f_t>(
        path, lp, lower, upper, vstatus_uploaded, dy_se_at_launch,
        x_at_launch, y_at_launch, z_at_launch, settings,
        static_cast<int>(status), static_cast<int>(status_ref),
        sol.objective, sol_ref.objective, sol.iterations, iter_ref);
      std::fprintf(stderr, "[BADNODE] %s %s\n", ok ? "wrote" : "FAILED to write", path);
      // SELF-CHECK. A dump is only worth having if replaying it reproduces the failure, and a dump
      // that silently omits an input reproduces nothing -- which reads exactly like "the defect did
      // not reproduce" and sends the investigation the wrong way. So read the file straight back
      // and re-solve it here, in the same process, through the same refactor-and-resume loop. If
      // this does not reproduce the status just observed, the dump is lossy and the replay tool
      // cannot be trusted; say so at the moment of capture rather than hours later.
      if (ok) {
        bad_node::header_t rh{};
        simplex::lp_problem_t<i_t, f_t> rlp(lp.handle_ptr, 1, 1, 1);
        std::vector<f_t> rlower, rupper, rdy_se, rx, ry, rz;
        std::vector<simplex::variable_status_t> rvstatus;
        std::string why;
        if (!bad_node::read<i_t, f_t>(path, rh, rlp, rlower, rupper, rvstatus, rdy_se, rx, ry, rz, why)) {
          std::fprintf(stderr, "[BADNODE-SELFCHECK] cannot read back: %s\n", why.c_str());
        } else {
          simplex::simplex_solver_settings_t<i_t, f_t> rs = settings;
          bad_node::apply_settings<i_t, f_t>(rh, rs);
          // Split "the FILE lost something" from "the recorded entry state is not what the live
          // solve actually started from". Compare the reconstructed arrays against the live ones
          // before re-solving: if they are identical the file is faithful, and the divergence has
          // to come from somewhere the dump does not reach.
          {
            auto cmp = [](const char* what, const auto& a, const auto& b) {
              if (a.size() != b.size()) {
                std::fprintf(stderr, "[BADNODE-SELFCHECK] %s SIZE %zu vs %zu\n", what, a.size(),
                             b.size());
                return;
              }
              size_t diff = 0;
              for (size_t k = 0; k < a.size(); ++k) { diff += (a[k] != b[k]); }
              if (diff != 0) {
                std::fprintf(stderr, "[BADNODE-SELFCHECK] %s DIFFERS in %zu entries\n", what, diff);
              }
            };
            cmp("A.col_start", rlp.A.col_start, lp.A.col_start);
            cmp("A.i", rlp.A.i, lp.A.i);
            cmp("A.x", rlp.A.x, lp.A.x);
            cmp("rhs", rlp.rhs, lp.rhs);
            cmp("objective", rlp.objective, lp.objective);
            cmp("lower", rlower, lower);
            cmp("upper", rupper, upper);
            cmp("vstatus", rvstatus, vstatus_uploaded);
            cmp("dy_se", rdy_se, dy_se_ref);
            if (rlp.num_rows != lp.num_rows || rlp.num_cols != lp.num_cols ||
                rlp.obj_scale != lp.obj_scale || rlp.obj_constant != lp.obj_constant) {
              std::fprintf(stderr, "[BADNODE-SELFCHECK] scalar lp fields DIFFER\n");
            }
          }
          const i_t rm = rlp.num_rows;
          std::vector<i_t> rbasic(rm), rnonbasic, rsuper;
          simplex::basis_update_mpf_t<i_t, f_t> rft(rm, rs.refactor_frequency);
          simplex::lp_solution_t<i_t, f_t> rsol(rm, rlp.num_cols);
          const f_t rt0 = cuopt::mathematical_optimization::tic();
          auto rprice   = [&]() {
            basis_from_vstatus_reset<i_t>(rm, rlp.num_cols, rvstatus, rbasic, rnonbasic, rsuper);
            if (rft.refactor_basis(rlp.A, rs, rlower, rupper, rt0, rbasic, rnonbasic, rvstatus) !=
                0) {
              return false;
            }
            f_t w = 0.0;
            simplex::phase2::compute_dual_solution_from_basis(
              rlp, rft, rbasic, rnonbasic, rsol.y, rsol.z, w);
            return true;
          };
          // BISECT. The reconstructed arrays are element-wise identical to the live ones, so if the
          // replay still answers differently the cause is not in the arrays. Run ONE more leg-0
          // launch that reuses the LIVE `lp`, `lower`, `upper` and `settings` objects and differs
          // from the live call only in taking fresh basis/solution containers. Whichever of the two
          // replays matches the live result names the half the cause is in.
          for (int brep = 0; brep < 3; ++brep) {
            std::vector<simplex::variable_status_t> bvs = vstatus_uploaded;
            std::vector<f_t> bse                        = dy_se_at_launch;
            std::vector<i_t> bbasic(m), bnonbasic, bsuper;
            simplex::basis_update_mpf_t<i_t, f_t> bft(m, settings.refactor_frequency);
            simplex::lp_solution_t<i_t, f_t> bsol(m, lp.num_cols);
            basis_from_vstatus_reset<i_t>(m, lp.num_cols, bvs, bbasic, bnonbasic, bsuper);
            const f_t bt0 = cuopt::mathematical_optimization::tic();
            if (bft.refactor_basis(lp.A, settings, lower, upper, bt0, bbasic, bnonbasic, bvs) == 0) {
              f_t bw = 0.0;
              simplex::phase2::compute_dual_solution_from_basis(
                lp, bft, bbasic, bnonbasic, bsol.y, bsol.z, bw);
              if (brep == 0) {
                auto cmpv = [](const char* what, const std::vector<f_t>& a,
                               const std::vector<f_t>& b) {
                  if (a.size() != b.size()) {
                    std::fprintf(stderr, "[BADNODE-SOLCMP] %s SIZE %zu vs %zu\n", what, a.size(),
                                 b.size());
                    return;
                  }
                  size_t d = 0;
                  for (size_t k = 0; k < a.size(); ++k) { d += (a[k] != b[k]); }
                  std::fprintf(stderr, "[BADNODE-SOLCMP] %s differs in %zu of %zu\n", what, d,
                               a.size());
                };
                cmpv("entry-vs-launch dy_se", dy_se_ref, dy_se_at_launch);
                cmpv("entry-vs-launch x", x_uploaded, x_at_launch);
                cmpv("entry-vs-launch y", y_uploaded, y_at_launch);
                cmpv("entry-vs-launch z", z_uploaded, z_at_launch);
                cmpv("sol.x", bsol.x, x_uploaded);
                cmpv("sol.y", bsol.y, y_uploaded);
                cmpv("sol.z", bsol.z, z_uploaded);
                cmpv("L0.x", bft.L0().x, l0x_uploaded);
                cmpv("U0.x", bft.U0().x, u0x_uploaded);
                auto cmpi = [](const char* what, const std::vector<i_t>& a,
                               const std::vector<i_t>& b) {
                  if (a.size() != b.size()) {
                    std::fprintf(stderr, "[BADNODE-SOLCMP] %s SIZE %zu vs %zu\n", what, a.size(),
                                 b.size());
                    return;
                  }
                  size_t d = 0;
                  for (size_t k = 0; k < a.size(); ++k) { d += (a[k] != b[k]); }
                  std::fprintf(stderr, "[BADNODE-SOLCMP] %s differs in %zu of %zu\n", what, d,
                               a.size());
                };
                cmpi("L0.col_start", bft.L0().col_start, l0cs_uploaded);
                cmpi("U0.col_start", bft.U0().col_start, u0cs_uploaded);
                cmpi("row_perm", bft.row_permutation(), rowperm_uploaded);
              }
              // Replay the uploaded values, not a fresh reconstruction of them.
              bsol.x = x_at_launch;
              bsol.y = y_at_launch;
              bsol.z = z_at_launch;
              bse    = dy_se_at_launch;
              gds::gpu_batch_item<i_t, f_t> bit{};
              bit.lp            = &lp;
              bit.lower         = &lower;
              bit.upper         = &upper;
              bit.settings      = &settings;
              bit.vstatus       = &bvs;
              bit.ft            = &bft;
              bit.basic_list    = &bbasic;
              bit.nonbasic_list = &bnonbasic;
              bit.sol           = &bsol;
              bit.dy_se         = &bse;
              std::vector<gds::gpu_batch_item<i_t, f_t>> bbatch{bit};
              std::vector<simplex::dual_status_t> bout;
              gds::gpu_batch_timing btiming;
              auto* bh = gds::gpu_simplified_dual_phase2_batch_launch<i_t, f_t>(
                bbatch, btiming, worker_stream.view(), /*allow_refactor_exit*/ true);
              if (bh != nullptr) {
                gds::gpu_simplified_dual_phase2_batch_finish<i_t, f_t>(bh, bout, btiming);
                if (!bout.empty()) {
                  std::fprintf(stderr, "[BADNODE-LIVEOBJ] rep=%d leg0 %s it=%d obj=%.17g\n", brep,
                               simplex::dual_status_to_string(bout[0]).c_str(), bsol.iterations,
                               bsol.objective);
                }
              }
            }
          }

          simplex::dual_status_t rstatus = simplex::dual_status_t::UNSET;
          i_t rtotal                     = 0;
          if (!rprice()) {
            std::fprintf(stderr, "[BADNODE-SELFCHECK] reconstruction failed to factorize\n");
          } else {
            for (int rres = 0; rres <= 64; ++rres) {   // diagnostic replay only; bound is arbitrary
              gds::gpu_batch_item<i_t, f_t> rit{};
              rit.lp            = &rlp;
              rit.lower         = &rlower;
              rit.upper         = &rupper;
              rit.settings      = &rs;
              rit.vstatus       = &rvstatus;
              rit.ft            = &rft;
              rit.basic_list    = &rbasic;
              rit.nonbasic_list = &rnonbasic;
              rit.sol           = &rsol;
              rit.dy_se         = &rdy_se;
              std::vector<gds::gpu_batch_item<i_t, f_t>> rbatch{rit};
              std::vector<simplex::dual_status_t> rout;
              gds::gpu_batch_timing rtiming;
              auto* rh2 = gds::gpu_simplified_dual_phase2_batch_launch<i_t, f_t>(
                rbatch, rtiming, worker_stream.view(), /*allow_refactor_exit*/ true);
              if (rh2 == nullptr) { break; }
              gds::gpu_simplified_dual_phase2_batch_finish<i_t, f_t>(rh2, rout, rtiming);
              if (rtiming.rejected || rout.empty()) { break; }
              rtotal += rsol.iterations;
              rstatus = rout[0];
              std::fprintf(stderr, "[BADNODE-RLEG] %d %s it=%d cum=%d obj=%.17g\n", rres,
                           simplex::dual_status_to_string(rstatus).c_str(), rsol.iterations,
                           (int)rtotal, rsol.objective);
              if (rstatus != simplex::dual_status_t::REFACTOR_NEEDED) { break; }
              if (!rprice()) { break; }
            }
          }
          const bool same = (rstatus == status);
          std::fprintf(stderr,
                       "[BADNODE-SELFCHECK] %s live=%s obj=%.17g it=%d | replay=%s obj=%.17g it=%d\n",
                       same ? "REPRODUCES" : "DOES NOT REPRODUCE -- the dump is missing an input",
                       simplex::dual_status_to_string(status).c_str(), sol.objective, iters,
                       simplex::dual_status_to_string(rstatus).c_str(), rsol.objective, rtotal);
        }
      }
    }

    if (cpu_decided && (decision_differs || obj_gap > obj_tol)) {
      std::fprintf(stderr,
                   "[GPUVERIFY] MISMATCH m=%d n=%d se=%zu | gpu=%s obj=%.17g it=%d | cpumin=%s obj=%.17g "
                   "it=%d | cpufull=%s obj=%.17g it=%d | fromgpubasis=%s obj=%.17g it=%d | objgap=%.3e "
                   "cut_off=%.17g\n",
                   m,
                   lp.num_cols,
                   dy_se_ref.size(),
                   simplex::dual_status_to_string(status).c_str(),
                   sol.objective,
                   sol.iterations,
                   simplex::dual_status_to_string(status_min).c_str(),
                   sol_min.objective,
                   sol_min.iterations,
                   simplex::dual_status_to_string(status_ref).c_str(),
                   sol_ref.objective,
                   iter_ref,
                   simplex::dual_status_to_string(status_out).c_str(),
                   sol_out.objective,
                   sol_out.iterations,
                   obj_gap,
                   // The threshold the CUTOFF decision was taken against. Without it a
                   // gpu=CUTOFF/cpu=OPTIMAL line cannot be classified at all -- which is exactly why
                   // the three nexp-150-20-8-5 cases stayed unresolved.
                   settings.cut_off);
    } else {
      // Emit a line per AGREEING node too. Without it the log gives a mismatch count with no
      // denominator, and "12 mismatches" out of 20 nodes and out of 2000 nodes are very different
      // results.
      // Carry the objectives and the cutoff on the AGREEING line as well. A gate that got quieter
      // must not also get blinder: a CUTOFF that now resolves to agreement still leaves its
      // evidence in the log, so the decision can be re-checked after the fact.
      std::fprintf(stderr,
                   "[GPUVERIFY] OK %s it=%d obj=%.17g cpu=%s obj=%.17g cut_off=%.17g\n",
                   simplex::dual_status_to_string(status).c_str(),
                   sol.iterations,
                   sol.objective,
                   simplex::dual_status_to_string(status_ref).c_str(),
                   sol_ref.objective,
                   settings.cut_off);
    }
  }
  return true;
}

template bool gpu_solve_node_serial<int, double>(
  const simplex::lp_problem_t<int, double>&, const csr_matrix_t<int, double>&,
  const std::vector<simplex::variable_type_t>&,
  const simplex::simplex_solver_settings_t<int, double>&,
  const std::vector<double>&, const std::vector<double>&,
  std::vector<simplex::variable_status_t>&, std::vector<int>&, std::vector<int>&,
  simplex::lp_solution_t<int, double>&, std::vector<double>&, int&,
  simplex::dual_status_t&);

// ===========================================================================================
// WHOLE-BATCH POOL
// ===========================================================================================

// Correctness gate, default OFF (CUOPT_BNB_GPU_POOL_RECHECK=1 enables). Every answer the pool produces
// is re-solved by the CPU full driver and compared; on disagreement the CPU answer is used and the
// node is dumped. Roughly doubles the LP work.
thread_local gpu_pool_lap_t* tl_stage_laps = nullptr;
static inline void stage_lap(int k)
{
  if (tl_stage_laps != nullptr) { tl_stage_laps->lap(k); }
}

bool gpu_pool_device_strengthen()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_DEVICE_STRENGTHEN");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

bool gpu_pool_inplace()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_INPLACE");
    return e == nullptr || std::string(e) != "0";   // default on
  }();
  return on;
}

bool gpu_pool_defer_refill()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_DEFER_REFILL");
    return e == nullptr || std::string(e) != "0";   // default on
  }();
  return on;
}

static bool gpu_pool_env_on(const char* name)
{
  const char* e = std::getenv(name);
  return e == nullptr || std::string(e) != "0";
}
bool gpu_pool_path_bounds()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_PATH_BOUNDS");
  return on;
}
bool gpu_pool_slim_d2h()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_SLIM_D2H") && !gpu_pool_recheck() &&
                         dump_bad_nodes_dir() == nullptr;
  return on;
}
int gpu_pool_lean_begin()
{
  static const int k = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LEAN_BEGIN");
    return e != nullptr ? std::max(0, std::atoi(e)) : 16;
  }();
  return k;
}
int gpu_pool_pipeline()
{
  static const int k = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PIPELINE");
    return e != nullptr ? std::max(1, std::atoi(e)) : 1;
  }();
  return k;
}
bool gpu_pool_pipeline_split()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PIPELINE_SPLIT");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}
bool gpu_pool_batch_stats()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_BATCH_STATS");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

int gpu_pool_dive()
{
  static const int r = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_DIVE");
    return e != nullptr ? std::max(1, std::atoi(e)) : 3;
  }();
  return r;
}

bool gpu_pool_lazy_install()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LAZY_INSTALL");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}
bool gpu_pool_retain()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_RETAIN");
  return on;
}
double gpu_pool_park_min()
{
  static const double f = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PARK_MIN");
    return e != nullptr ? std::clamp(std::atof(e), 0.0, 1.0) : 0.5;
  }();
  return f;
}
bool gpu_pool_park_short()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PARK_SHORT");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}
bool gpu_pool_batch_fetch()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_BATCH_FETCH");
  return on;
}
bool gpu_pool_device_scores()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_DEVICE_SCORES");
  return on;
}
bool gpu_pool_packed_fresh()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_PACKED_FRESH");
  return on;
}
bool gpu_pool_bounds_view()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_BOUNDS_VIEW");
  return on;
}
bool gpu_pool_light_settings()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_LIGHT_SETTINGS");
  return on;
}
bool gpu_pool_batch_heap()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_BATCH_HEAP");
  return on;
}
bool gpu_pool_device_objective()
{
  static const bool on = gpu_pool_env_on("CUOPT_BNB_GPU_DEVICE_OBJ");
  return on;
}

bool gpu_pool_recheck()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_POOL_RECHECK");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}

// How many times a seat may be refilled with the child the PLUNGE would continue with, before the
// node goes back to the queue. 0 = off (every node reconstructed from the root).
int gpu_pool_plunge_depth()
{
  static const int d = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_PLUNGE");
    return (e != nullptr) ? std::atoi(e) : 8;
  }();
  return d;
}

// Overlap CPU work with an in-flight round (CUOPT_BNB_GPU_ASYNC, default on). Off = the worker
// blocks on the device, which is the synchronous behaviour the correctness sweeps were built on.
// Validate an OPTIMAL against freshly priced reduced costs before accepting it
// (CUOPT_BNB_GPU_VALIDATE_OPTIMAL, default on).
bool gpu_pool_overlap()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_ASYNC");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// Per-node iteration budget for a GPU-seated node LP. settings_ carries INT_MAX, which is right for
// the CPU plunge path but wrong for a seat: a node that stalls would hold its seat forever, and the
// ITERATION_LIMIT status the consume path uses to hand hard nodes back to the CPU would never fire.
// 5000 is ~20x the p99 of the measured per-node distribution (min 0 / p50 10 / p90 126 / p99 234 /
// max 256 over 3,156 real node solves), so it bounds pathology without touching normal nodes.
// CUOPT_BNB_GPU_ITER_LIMIT=<n> overrides; 0 restores the unbounded behaviour.
int gpu_batch_iter_limit()
{
  static const int lim = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_ITER_LIMIT");
    if (e == nullptr) { return 5000; }
    const int v = std::atoi(e);
    return (v > 0) ? v : std::numeric_limits<int>::max();
  }();
  return lim;
}

// CUOPT_BNB_GPU_DUMP_BASES=<dir>: write the fresh seats' pre-LU bases (with A) for offline LU
// benchmarks, up to CUOPT_BNB_GPU_DUMP_BASES_MAX bases per process (default 20000).
static const char* gpu_dump_bases_dir()
{
  static const char* d = std::getenv("CUOPT_BNB_GPU_DUMP_BASES");
  return d;
}
static long long gpu_dump_bases_max()
{
  static const long long v = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_DUMP_BASES_MAX");
    return e != nullptr ? std::atoll(e) : 20000LL;
  }();
  return v;
}
static std::atomic<long long> g_dumped_bases{0};
static std::atomic<int> g_dump_files{0};

double gpu_pool_log_warmup_s()
{
  static const double s = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LOG_WARMUP");
    return e != nullptr ? std::max(0.0, std::atof(e)) : 0.0;
  }();
  return s;
}

bool gpu_pool_logging()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_LOG");
    return e != nullptr && std::string(e) != "0";
  }();
  return on;
}

// Refactor frequency of the GPU dual simplex seats (CPU node solves keep their own setting). 200:
// lrsa120 (2026-09-29, 180 s) 8-16 % more iterations/s than 100 (43 % fewer host LUs); gen-ip054,
// gen-ip002, mas74, mik-250-20-75-4 within run-to-run spread.
int gpu_pool_refac_freq() { return 200; }

// #3 -- how full this worker's queue must be before the pool engages, as a MULTIPLE of the seat
// count. The pool takes `seats` nodes OUT of the queue; firing when the queue barely holds that
// many empties it, and the worker then has nothing to overlap with and nothing to steal next cycle.
// Waiting for seats*N leaves the queue with (N-1)*seats after a fill.
//
// N=5 by default, per the design note: with a 1K working queue and a 200-seat pool you want ~5K
// before engaging. CUOPT_BNB_GPU_HEAP_FACTOR=<n> overrides; 0 disables the gate entirely (the old
// behaviour -- engage whenever there is any work at all).
//
// There is deliberately no root/depth escape: CPU search grows the queue until this threshold is
// reached. A worker that never accumulates enough GPU-eligible nodes simply remains CPU-only.
int gpu_pool_heap_factor()
{
  static const int f = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_HEAP_FACTOR");
    if (e == nullptr) { return 1; }
    const int v = std::atoi(e);
    return (v > 0) ? v : 0;
  }();
  return f;
}

bool gpu_pool_should_engage(size_t queue_size, int seats)
{
  if (seats <= 0) { return false; }
  const int factor = gpu_pool_heap_factor();
  if (factor <= 0) { return queue_size > 0; }   // gate disabled
  // A lean begin seats only gpu_pool_lean_begin() nodes; the refills fill the rest, so that many is
  // enough to start. CUOPT_BNB_GPU_ENGAGE_FULL=1: require a full batch.
  static const bool full = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_ENGAGE_FULL");
    return e != nullptr && std::string(e) == "1";
  }();
  const int need = !full && gpu_pool_lean_begin() > 0 ? std::min(seats, gpu_pool_lean_begin()) : seats;
  return queue_size >= static_cast<size_t>(need) * static_cast<size_t>(factor);
}

// #2 -- drive the pool from the WORKER LOOP. The loop asks "is the GPU done?" by atomically loading
// the mapped pinned completion word published by the last resident block; if it is not, the worker
// RETURNS to its loop and carries on with its own work instead of waiting. It blocks on the device
// only when it has nothing else left to do, at the bottom of the loop -- spinning there instead
// would burn a core and starve the other workers' threads.
//
// DEFAULT ON. CUOPT_BNB_GPU_ASYNC=0 restores the old drive-to-completion behaviour for A/B.
bool gpu_pool_async()
{
  static const bool on = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_ASYNC");
    return e == nullptr || std::string(e) != "0";   // DEFAULT ON
  }();
  return on;
}

int gpu_batch_seats()
{
  static const int seats = [] {
    const char* e = std::getenv("CUOPT_BNB_GPU_BATCH");
    if (e == nullptr) { return 0; }
    if (std::string(e) == "auto") { return -1; }
    const int v = std::atoi(e);
    return (v > 0) ? v : 0;
  }();
  return seats;
}

int gpu_batch_auto_seats(int device, int num_rows, int num_cols, std::size_t num_nonzeros)
{
  if (device < 0 || num_rows <= 0 || num_cols <= 0) { return 0; }

  int previous_device = 0;
  if (cudaGetDevice(&previous_device) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
    return 0;
  }

  cudaDeviceProp properties{};
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  const int blocks_per_sm = gds::gpu_max_blocks_per_sm();
  const bool valid = blocks_per_sm > 0 &&
                     cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
                     cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
  cudaSetDevice(previous_device);
  if (!valid) { return 0; }

  // The queue absorbs the long tail after the resident waves diverge in pivot count. Four waves
  // (~9.5K seats per B200: 4 x 148 SMs x 16 blocks) replaced the original two (4224 on H100),
  // which left the device underfed at 111 BFS workers.
  constexpr int oversubscription = 4;
  const int64_t occupancy_limit = oversubscription *
                                  static_cast<int64_t>(properties.multiProcessorCount) *
                                  static_cast<int64_t>(blocks_per_sm);

  // This is deliberately conservative and only chooses the initial target. The batch allocator's
  // exact preflight remains authoritative and halves the target if the complete allocation does
  // not fit.
  const size_t per_seat_bytes =
    static_cast<size_t>(num_rows + num_cols) * 64 + num_nonzeros * sizeof(double);
  if (per_seat_bytes == 0) { return 0; }
  const int64_t memory_limit = static_cast<int64_t>((free_bytes / 2) / per_seat_bytes);
  return static_cast<int>(std::max<int64_t>(0, std::min(occupancy_limit, memory_limit)));
}

namespace {

// Per-seat host state for one node in flight. Everything the batch item points at must outlive the
// launch, so these live in one vector for the whole call and are never reallocated.
// batch_seat_state_t now lives in gpu_node_pool.hpp: the POOL owns one per seat so the
// factorization and its MPF stack survive across rounds and across plunge children.

}  // namespace

// ---------------------------------------------------------------------------
// A batch RUN, resumable across host loop iterations.
//
// The solve of one pool-full of nodes is a sequence of rounds: launch the active seats, read their
// statuses, host-refactorize the ones that came back REFACTOR_NEEDED, relaunch those. This object
// holds the state between rounds so the caller can LAUNCH, go and do CPU work, and come back when
// the device is ready -- which is what the asynchronous pool needs. The synchronous entry point
// below is the same thing with the waiting inlined.
//
// `nodes` is BORROWED and must outlive the run; the pool owns it.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
struct gpu_batch_run_t {
  const simplex::lp_problem_t<i_t, f_t>& lp;
  std::vector<gpu_batch_node_t<i_t, f_t>>& nodes;
  rmm::cuda_stream_view stream;
  gds::gpu_ro_cache<i_t, f_t>* ro_cache{nullptr};

  i_t m{0}, n{0}, N{0};
  f_t t0{0.0};
  // BORROWED from the pool. One-shot rounds keep their host factor here; the persistent pool keeps the live LU+MPF
  // representation on the device and uses this host factor only for fresh/refactor commands.
  std::vector<batch_seat_state_t<i_t, f_t>>& st;

  std::vector<gds::gpu_batch_item<i_t, f_t>> items;
  std::vector<i_t> seat_of_item;
  std::vector<simplex::dual_status_t> out;
  gds::gpu_batch_timing timing;
  gds::gpu_batch_handle<i_t, f_t>* handle{nullptr};
  i_t rounds{0};
  std::size_t terminal_noop_seats{0};
  i_t periodic_refactors{0}, rejected_refactors{0}, certificate_refactors{0};
  double upload_s{0.0}, kernel_s{0.0}, copyback_s{0.0}, host_factor_s{0.0};
  double last_harvest_d2h_s{0.0}, last_harvest_scatter_s{0.0}, last_harvest_unpack_s{0.0};
  double last_harvest_d2h_enqueue_s{0.0}, last_harvest_d2h_device_s{0.0};
  double last_prekernel_s{0.0}, last_prekernel_lu_s{0.0};
  double last_prekernel_h2d_s{0.0}, last_prekernel_scatter_s{0.0}, last_prekernel_deltas_s{0.0};
  double last_done_after_pre_s{0.0};
  bool persistent_seats{false};
  // Split of the initial launch (telemetry only; see gpu_persistent_batch_begin).
  bool result_views{false};
  bool bounds_mirror{true};  // see gpu_batch_run_set_bounds_mirror
  const std::vector<simplex::variable_type_t>* var_types{nullptr};  // set => device strengthening
  // Host LU telemetry of this run (CUOPT_BNB_GPU_LOG) and the optional base dump.
  struct lu_stats_t {
    uint64_t fresh{0}, resume{0}, ok{0}, failed{0}, repaired{0}, retried{0};
    std::array<uint64_t, 8> nucleus_hist{};  // 0, 1-8, 9-16, 17-32, 33-64, 65-128, 129-256, >256
    long long sum_nucleus{0}, sum_nucleus_nz{0}, sum_B_nz{0}, sum_L_nz{0}, sum_U_nz{0};
    int max_nucleus{0};
    double host_s{0.0};
  } lu_stats;
  int lu_attempts{0};
  bool lu_repaired{false};
  std::vector<i_t> lu_pre_basis;
  std::FILE* dump_{nullptr};
  gds::gpu_batch_timing setup_timing;
  double setup_wall_s{0.0}, setup_host_lu_s{0.0};
  gpu_batch_run_t(const simplex::lp_problem_t<i_t, f_t>& lp_,
                  std::vector<gpu_batch_node_t<i_t, f_t>>& nodes_,
                  rmm::cuda_stream_view s,
                  gds::gpu_ro_cache<i_t, f_t>* ro,
                  std::vector<batch_seat_state_t<i_t, f_t>>& seat_state,
                  bool persistent = false,
                  bool views = false)
    : lp(lp_), nodes(nodes_), stream(s), ro_cache(ro),
      m(lp_.num_rows), n(lp_.num_cols), N(static_cast<i_t>(nodes_.size())),
      t0(cuopt::mathematical_optimization::tic()), st(seat_state), persistent_seats(persistent),
      result_views(views)
  {
    // Grow to cover this round's seats; existing entries KEEP their factorization.
    while (static_cast<i_t>(st.size()) < N) {
      st.emplace_back(m, nodes[st.size()].settings.refactor_frequency);
    }
    // A seat whose node is NOT a plunge continuation starts clean: fresh counters, and prepare()
    // will refactorize it. A continuing seat keeps everything, which is the point.
    for (i_t s2 = 0; s2 < N; ++s2) {
      if (nodes[s2].continue_basis) { continue; }
      auto& w                     = st[s2];
      w.total_iters               = 0;
      w.leg_iters                 = 0;
      w.zero_pivot_legs           = 0;
      w.refactor_resume           = false;
      w.terminal_certificate_retry = false;
      w.refactor_reason           = 0;
      w.fail                      = gpu_batch_node_t<i_t, f_t>::drop_t::NONE;
      if (persistent_seats) { w.active = false; }
    }
  }

  gds::gpu_batch_item<i_t, f_t> make_item(i_t s,
                                          gds::gpu_batch_seat_command command) const
  {
    auto& node = nodes[s];
    auto& w    = st[s];
    gds::gpu_batch_item<i_t, f_t> it{};
    it.lp              = &lp;
    it.lower           = &node.lower;
    it.upper           = &node.upper;
    it.root_lower      = command == gds::gpu_batch_seat_command::FRESH_NODE ? node.root_lower : nullptr;
    it.root_upper      = command == gds::gpu_batch_seat_command::FRESH_NODE ? node.root_upper : nullptr;
    it.packed_vstatus  = it.root_lower != nullptr ? node.packed_vstatus : nullptr;
    it.root_dy_se      = it.root_lower != nullptr ? node.root_dy_se : nullptr;
    it.settings        = &node.settings;
    it.vstatus         = &node.vstatus;
    it.ft              = &w.ft;
    it.basic_list      = &w.basic_list;
    it.nonbasic_list   = &w.nonbasic_list;
    it.sol             = &node.sol;
    it.dy_se           = &node.dy_se;
    it.refactor_resume = command == gds::gpu_batch_seat_command::REFACTOR_RESUME;
    it.terminal_certificate_retry = w.terminal_certificate_retry;
    it.refactor_reason            = &w.refactor_reason;
    it.continue_basis             = command == gds::gpu_batch_seat_command::CHILD_CONTINUE;
    it.command                    = command;
    return it;
  }

  // Factorize and price seat `s` from its CURRENT basis. A resume proceeds only when fresh pricing
  // confirms that the returned basis/status pair is dual feasible.
  bool refactorize(i_t s, bool initialize_basis)
  {
    const bool record = gpu_pool_logging() || gpu_dump_bases_dir() != nullptr;
    if (!record) { return refactorize_impl(s, initialize_basis); }
    lu_attempts     = 0;
    lu_repaired     = false;
    const double t0 = cuopt::mathematical_optimization::tic();
    const bool ok   = refactorize_impl(s, initialize_basis);
    record_lu(s, ok, initialize_basis, cuopt::mathematical_optimization::toc(t0));
    return ok;
  }

  void record_lu(i_t s, bool ok, bool fresh, double secs)
  {
    const auto& fb = simplex::factorize_basis_last_stats();
    auto& L        = lu_stats;
    ++(fresh ? L.fresh : L.resume);
    if (!ok) { ++L.failed; }
    if (lu_repaired) { ++L.repaired; }
    if (lu_attempts > 1) { ++L.retried; }
    const int nu = fb.nucleus;
    const int b  = nu == 0 ? 0 : nu <= 8 ? 1 : nu <= 16 ? 2 : nu <= 32 ? 3 : nu <= 64 ? 4
                 : nu <= 128 ? 5 : nu <= 256 ? 6 : 7;
    ++L.nucleus_hist[b];
    L.max_nucleus = std::max(L.max_nucleus, nu);
    L.sum_nucleus += nu;
    L.sum_nucleus_nz += fb.nucleus_nz;
    L.sum_B_nz += fb.B_nz;
    if (ok) {
      ++L.ok;
      L.sum_L_nz += st[s].ft.L0().nnz();
      L.sum_U_nz += st[s].ft.U0().nnz();
    }
    L.host_s += secs;
    if (fresh && gpu_dump_bases_dir() != nullptr && !lu_pre_basis.empty() &&
        g_dumped_bases.fetch_add(1) < gpu_dump_bases_max()) {
      if (dump_ == nullptr) {
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/bases_%d_%05d.bin", gpu_dump_bases_dir(),
                      (int)getpid(), g_dump_files.fetch_add(1));
        dump_ = std::fopen(path, "wb");
        if (dump_ != nullptr) {
          // Header: magic, m, n, nnz, then A in CSC (int32 col_start / row indices, f64 values).
          const int32_t hdr[3] = {(int32_t)m, (int32_t)n, (int32_t)lp.A.col_start[n]};
          std::fwrite("CUBASES1", 1, 8, dump_);
          std::fwrite(hdr, sizeof(int32_t), 3, dump_);
          std::fwrite(lp.A.col_start.data(), sizeof(i_t), n + 1, dump_);
          std::fwrite(lp.A.i.data(), sizeof(i_t), hdr[2], dump_);
          std::fwrite(lp.A.x.data(), sizeof(f_t), hdr[2], dump_);
        }
      }
      if (dump_ != nullptr) {
        // Record: ok, repaired, attempts, nucleus, L nz, U nz, host seconds, pre-LU basic list.
        const int32_t rec[6] = {(int32_t)ok, (int32_t)lu_repaired, (int32_t)lu_attempts, (int32_t)nu,
                                ok ? (int32_t)st[s].ft.L0().nnz() : -1,
                                ok ? (int32_t)st[s].ft.U0().nnz() : -1};
        std::fwrite(rec, sizeof(int32_t), 6, dump_);
        std::fwrite(&secs, sizeof(double), 1, dump_);
        std::fwrite(lu_pre_basis.data(), sizeof(i_t), m, dump_);
      }
    }
  }

  bool refactorize_impl(i_t s, bool initialize_basis)
  {
    raft::common::nvtx::range scope("GPUPool::host_lu");
    auto& w    = st[s];
    auto& node = nodes[s];
    const auto result = result_views && !initialize_basis
                          ? gds::gpu_simplified_dual_phase2_batch_result<i_t, f_t>(handle, s)
                          : gds::gpu_batch_result_view<i_t, f_t>{};
    std::span<simplex::variable_status_t> vstatus = result_views && !initialize_basis
                                                      ? result.vstatus
                                                      : std::span(node.vstatus);
    std::span<i_t> basic_list = result_views && !initialize_basis
                                  ? result.basic_list
                                  : std::span(w.basic_list);
    std::span<i_t> nonbasic_list = result_views && !initialize_basis
                                     ? result.nonbasic_list
                                     : std::span(w.nonbasic_list);
    // A genuine rank deficiency is reported against the phase it happened in; the clock and halt
    // codes are phase-independent.
    const auto deficient_drop = initialize_basis ? gpu_batch_node_t<i_t, f_t>::drop_t::FACTORIZE_FILL
                                      : gpu_batch_node_t<i_t, f_t>::drop_t::FACTORIZE_RESUME;
    // FILL ONLY. On a RESUME the device has already handed back `basic_list` / `nonbasic_list` as it
    // maintains them -- by in-place swap, so their order is the pivot history. Rebuilding them from
    // `vstatus` here would re-sort both lists on every leg, which changes the LU that
    // `refactor_basis` produces (it factorizes A(:, basic_list)) and changes the order the
    // order-dependent ratio test scans the nonbasics in. Both make the resume diverge from an
    // uninterrupted solve, and from the CPU's in-place refactorization.
    if (initialize_basis) {
      basis_from_vstatus_reset<i_t>(
        m, n, node.vstatus, w.basic_list, w.nonbasic_list, w.superbasic_list);
      if (gpu_dump_bases_dir() != nullptr) { lu_pre_basis = w.basic_list; }
    }
    // Record WHICH return code stopped us. refactor_basis has three distinct failure modes and they
    // mean completely different things -- out of time, halted, or genuinely rank-deficient.
    auto classify = [&](i_t code) {
      w.fail = (code == TIME_LIMIT_RETURN)        ? gpu_batch_node_t<i_t, f_t>::drop_t::DEADLINE
               : (code == CONCURRENT_HALT_RETURN) ? gpu_batch_node_t<i_t, f_t>::drop_t::CONCURRENT_HALT
                                                  : deficient_drop;
      return false;
    };
    // RETRY EXACTLY AS THE CPU DRIVER DOES (phase2.cpp:3631-3660): a rank-deficient basis is
    // repaired by substituting slacks, and if it is still deficient the CPU loosens the
    // partial-pivoting threshold and tries again, up to 10 times, before giving up as NUMERICAL.
    const f_t refactor_start =
      persistent_seats ? cuopt::mathematical_optimization::tic() : t0;
    bool repaired = false;
    i_t rc = w.ft.refactor_basis(lp.A, node.settings, node.lower, node.upper, refactor_start,
                                 basic_list, nonbasic_list, vstatus, &repaired);
    ++lu_attempts;
    lu_repaired = lu_repaired || repaired;
    if (rc == CONCURRENT_HALT_RETURN || rc == TIME_LIMIT_RETURN) { return classify(rc); }
    // A repair during RESUME changes the basis underneath maintained device state, so that node
    // must leave the GPU and restart on the CPU. A fresh repair retains the incoming steepest-edge
    // vector, matching the CPU driver.
    if (repaired && !initialize_basis) { return classify(rc); }
    if (rc > 0) {
      i_t count = 0;
      while (true) {
        repaired = false;
        rc = w.ft.refactor_basis(lp.A, node.settings, node.lower, node.upper, refactor_start,
                                 basic_list, nonbasic_list, vstatus, &repaired);
    ++lu_attempts;
    lu_repaired = lu_repaired || repaired;
        if (rc == CONCURRENT_HALT_RETURN || rc == TIME_LIMIT_RETURN) { return classify(rc); }
        if (repaired && !initialize_basis) { return classify(rc); }
        if (rc <= 0) { break; }
        node.settings.threshold_partial_pivoting_tol = 1.0;
        if (++count > 10) { return classify(rc); }
      }
    }
    if (rc < 0) { return classify(rc); }
    f_t work = 0.0;
    if (initialize_basis) {
      simplex::phase2::compute_dual_solution_from_basis(
        lp, w.ft, w.basic_list, w.nonbasic_list, node.sol.y, node.sol.z, work);
    } else if (persistent_seats || w.refactor_reason != 1) {
      std::vector<f_t> fresh_y(m, 0.0), fresh_z(n, 0.0);
      simplex::phase2::compute_dual_solution_from_basis(
        lp, w.ft, basic_list, nonbasic_list, fresh_y, fresh_z, work);
      if (!gpu_resume_dual_feasible<f_t>(vstatus, fresh_z, node.settings.dual_tol)) {
        w.fail = gpu_batch_node_t<i_t, f_t>::drop_t::DUAL_INFEASIBLE_RESUME;
        return false;
      }
    }
    return true;
  }

  void prepare()
  {
  // ---- fill ------------------------------------------------------------------------------
  // A seat that fails preparation is simply never activated; its `solved` stays false and the
  // caller takes it on the CPU. One bad seat must not cost the whole batch.
  for (i_t s = 0; s < N; ++s) {
    auto& node = nodes[s];
    // Every B&B node carries a size-n root or parent steepest-edge vector, including after repair.
    assert(node.dy_se.size() == static_cast<size_t>(n));
    if (node.parked) {
      node.solved  = true;
      node.status  = simplex::dual_status_t::NUMERICAL;
      st[s].active = false;
      continue;
    }
    node.sol.resize(m, n);
    node.status = simplex::dual_status_t::UNSET;
    node.solved = false;
    node.drop   = gpu_batch_node_t<i_t, f_t>::drop_t::NONE;
    // PLUNGE CONTINUATION: reuse the parent's factorization exactly as the CPU does. `recompute_basis`
    // there is `node_status != HAS_CHILDREN`, so a dived child is handed basis_factors as they stand
    // -- LU plus the accumulated MPF updates -- and the driver refactorizes only when its own
    // predicate fires, in place, mid-solve. Refactorizing here would pay an LU the CPU never pays and
    // would reset the update count, disabling the refactor schedule down the dive.
    if (node.continue_basis) {
      st[s].active = true;
      continue;
    }
    const double factor_t0 = cuopt::mathematical_optimization::tic();
    const bool factor_ok   = refactorize(s, /*initialize_basis*/ true);
    host_factor_s += cuopt::mathematical_optimization::toc(factor_t0);
    if (!factor_ok) {
      node.drop = st[s].fail;
      if (std::getenv("CUOPT_BNB_GPU_DROP_TRACE") != nullptr) {
        i_t nb = 0, nfree = 0;
        for (i_t j = 0; j < n; ++j) {
          nb += (node.vstatus[j] == simplex::variable_status_t::BASIC);
          nfree += (node.vstatus[j] == simplex::variable_status_t::NONBASIC_FREE);
        }
        std::fprintf(stderr,
                     "[DROPTRACE] factorize_fill m=%d n=%d basic=%d free=%d vstatus_sz=%zu "
                     "basic_sz=%zu nonbasic_sz=%zu\n",
                     (int)m, (int)n, (int)nb, (int)nfree, node.vstatus.size(),
                     st[s].basic_list.size(), st[s].nonbasic_list.size());
      }
      continue;
    }

    // Size contract. The kernel indexes these with no bounds check, so a short vector is an
    // out-of-bounds DEVICE read that surfaces far away as cudaErrorIllegalAddress on an unrelated
    // synchronising call. Check here, where the answer is one line -- and here it matters more than
    // on the serial path, because one malformed seat would corrupt the whole batch.
    const char* bad = nullptr;
    if (static_cast<i_t>(node.vstatus.size()) != n) { bad = "vstatus"; }
    else if (static_cast<i_t>(node.lower.size()) != n) { bad = "lower"; }
    else if (static_cast<i_t>(node.upper.size()) != n) { bad = "upper"; }
    else if (static_cast<i_t>(st[s].basic_list.size()) != m) { bad = "basic_list"; }
    else if (static_cast<i_t>(st[s].nonbasic_list.size()) != n - m) { bad = "nonbasic_list"; }
    else if (static_cast<i_t>(node.dy_se.size()) != n) { bad = "dy_se"; }
    if (bad != nullptr) {
      std::fprintf(stderr,
                   "[GPUBATCH] seat %d not launched: %s has the wrong length (m=%d n=%d)\n",
                   (int)s, bad, (int)m, (int)n);
      node.drop = gpu_batch_node_t<i_t, f_t>::drop_t::SIZE_CONTRACT;
      continue;
    }
    // SNAPSHOT for a replayable dump. Taken here, AFTER refactorize() -- `refactor_basis` may have
    // repaired the basis, and what the device consumes is what a replay has to reproduce. `vstatus`
    // itself is overwritten with the FINAL basis by the solve, so it cannot serve.
    node.vstatus_uploaded = node.vstatus;
    st[s].active          = true;
  }
  }

  // Relaunch the SAME fixed seat array every round. Active seats run normally; a seat that already
  // reached a terminal status carries that status in the item and its block exits immediately.
  // This deliberately avoids compaction, dynamic refill, or an asynchronous scheduling policy:
  // mark terminal seats done, host-refactor the requests, and rerun the whole batch.
  bool launch_round()
  {
    items.clear();
    seat_of_item.clear();
    bool any_active = false;
    for (i_t s = 0; s < N; ++s) {
      auto& node = nodes[s];
      gds::gpu_batch_item<i_t, f_t> it{};
      if (st[s].active) {
        any_active         = true;
        it.lp              = &lp;
        it.lower           = &node.lower;   // per-node bounds; shared lp carries A/rhs/objective
        it.upper           = &node.upper;
        it.settings        = &node.settings;
        it.continue_basis  = node.continue_basis;
        it.var_types       = var_types;
        // Snapshot what the device is about to consume, before the launch overwrites it in place.
        if (node.dy_se_at_launch.empty()) {
          node.dy_se_at_launch = node.dy_se;
          node.x_at_launch     = node.sol.x;
          node.y_at_launch     = node.sol.y;
          node.z_at_launch     = node.sol.z;
        }
        it.vstatus         = &node.vstatus;
        it.ft              = &st[s].ft;
        it.basic_list      = &st[s].basic_list;
        it.nonbasic_list   = &st[s].nonbasic_list;
        it.sol             = &node.sol;
        it.dy_se           = &node.dy_se;
        it.refactor_resume = st[s].refactor_resume;
        it.terminal_certificate_retry = st[s].terminal_certificate_retry;
        it.refactor_reason = &st[s].refactor_reason;
        // Before the FIRST launch only: `dy_se` is in/out and the solve rewrites it, so a snapshot
        // taken later is not what was uploaded.
        if (!st[s].refactor_resume) { node.dy_se_at_launch = node.dy_se; }
      } else {
        // Preparation/drop failures are also no-op seats, but remain unsolved so the caller still
        // sends them to the CPU. NUMERICAL here is only the no-op marker returned by this round.
        it.terminal_status = node.solved ? node.status : simplex::dual_status_t::NUMERICAL;
      }
      items.push_back(it);
      seat_of_item.push_back(s);
    }
    if (!any_active) { return false; }

    i_t active_count = 0;
    for (const auto& w : st) { active_count += w.active ? 1 : 0; }

    handle = gds::gpu_simplified_dual_phase2_batch_launch<i_t, f_t>(
      items, timing, stream, /*allow_refactor_exit*/ true, ro_cache, persistent_seats,
      /*factor_slots*/ result_views);
    if (handle == nullptr) {
      // Rejected before launch (VRAM pre-flight); timing.message says needed-vs-free. Every seat
      // still active goes to the CPU rather than being silently lost.
      std::fprintf(stderr, "[GPUBATCH] launch rejected: %s\n", timing.message.c_str());
      for (i_t s = 0; s < N; ++s) {
        if (st[s].active) { nodes[s].drop = gpu_batch_node_t<i_t, f_t>::drop_t::LAUNCH_FAILED; }
      }
      return false;
    }
    if (result_views) { gds::gpu_simplified_dual_phase2_batch_use_views<i_t, f_t>(handle); }
    ++rounds;
    terminal_noop_seats += static_cast<std::size_t>(N - active_count);
    return true;
  }

  // Non-blocking: has the launched round finished on the device?
  bool ready() const
  {
    return handle == nullptr ||
           gds::gpu_simplified_dual_phase2_batch_ready<i_t, f_t>(handle);
  }

  // The persistent pool harvests exactly one complete N-seat launch and keeps the device handle/allocations alive.
  // No refactorization or B&B disposition is hidden here: the caller must classify every seat,
  // stage one replacement/resume command for each, and only then issue one full relaunch.
  bool harvest_persistent()
  {
    if (!persistent_seats || handle == nullptr) { return false; }
    gds::gpu_simplified_dual_phase2_batch_harvest<i_t, f_t>(
      handle, out, timing, result_views && gpu_pool_slim_d2h());
    upload_s += timing.upload_s;
    kernel_s += timing.kernel_s;
    copyback_s += timing.copyback_s;
    last_harvest_d2h_s     = timing.copyback_sync_s;
    last_harvest_d2h_enqueue_s = timing.copyback_enqueue_s;
    last_harvest_d2h_device_s  = timing.copyback_device_s;
    last_prekernel_s       = timing.prekernel_s;
    last_prekernel_lu_s      = timing.prekernel_lu_s;
    last_prekernel_h2d_s     = timing.prekernel_h2d_s;
    last_prekernel_scatter_s = timing.prekernel_scatter_s;
    last_prekernel_deltas_s  = timing.prekernel_deltas_s;
    last_done_after_pre_s    = timing.done_after_pre_s;
    last_harvest_scatter_s = std::max(0.0, timing.copyback_s - timing.copyback_sync_s);
    if (timing.rejected || out.size() != static_cast<size_t>(N)) { return false; }
    const double unpack_t0 = cuopt::mathematical_optimization::tic();
    for (i_t s = 0; s < N; ++s) {
      auto& w    = st[s];
      auto& node = nodes[s];
      const auto result = result_views
                            ? gds::gpu_simplified_dual_phase2_batch_result<i_t, f_t>(handle, s)
                            : gds::gpu_batch_result_view<i_t, f_t>{};
      const i_t iterations = result_views ? result.iterations : node.sol.iterations;
      w.leg_iters   = iterations - w.total_iters;
      w.total_iters = iterations;
      // DEBUG: seats whose solve runs past every 20,000 iterations.
      if (iterations / 20000 != (iterations - w.leg_iters) / 20000 && iterations >= 20000) {
        std::fprintf(stderr, "[SEAT-LONG] seat=%d iterations=%d leg=%d status=%d refactor_reason=%d\n",
                     (int)s, (int)iterations, (int)w.leg_iters, (int)out[s], (int)w.refactor_reason);
      }
      if (result_views) { w.refactor_reason = result.refactor_reason; }
      // A never-staged (empty) seat has no host bounds to mirror into.
      if (!result.lower.empty() && !node.lower.empty() &&
          (bounds_mirror || out[s] == simplex::dual_status_t::REFACTOR_NEEDED)) {
        if (node.lower.size() != result.lower.size() || node.upper.size() != result.upper.size()) {
          throw std::logic_error("GPU harvest: seat bounds size does not match the result");
        }
        std::copy(result.lower.begin(), result.lower.end(), node.lower.begin());
        std::copy(result.upper.begin(), result.upper.end(), node.upper.begin());
      }
      node.status = out[s];
      node.iters  = w.total_iters;
      node.solved = out[s] != simplex::dual_status_t::REFACTOR_NEEDED;
      if (!result_views) {
        node.basic_list    = w.basic_list;
        node.nonbasic_list = w.nonbasic_list;
      }
    }
    last_harvest_unpack_s = cuopt::mathematical_optimization::toc(unpack_t0);
    return true;
  }

  bool stage_persistent_fresh(i_t s, std::string& error)
  {
    auto& node = nodes[s];
    auto& w    = st[s];
    // Fresh basis repair retains the incoming vector, matching the CPU driver.
    assert(node.packed_vstatus != nullptr || node.dy_se.size() == static_cast<size_t>(n));
    node.sol.resize(m, n);
    node.solved = false;
    node.drop   = gpu_batch_node_t<i_t, f_t>::drop_t::NONE;
    w.total_iters                = 0;
    w.leg_iters                  = 0;
    w.zero_pivot_legs            = 0;
    w.refactor_resume            = false;
    w.terminal_certificate_retry = false;
    w.refactor_reason            = 0;
    w.fail                       = gpu_batch_node_t<i_t, f_t>::drop_t::NONE;
    const bool device_lu = gds::gpu_simplified_dual_phase2_batch_device_lu<i_t, f_t>(handle);
    stage_lap(SUB_FRESH_PREP);
    if (device_lu && node.packed_vstatus != nullptr) {
      node.vstatus.resize(n);
      node.dy_se.resize(n);
      w.basic_list.resize(m);
      w.nonbasic_list.resize(n - m);
    } else if (device_lu) {
      basis_from_vstatus_reset<i_t>(
        m, n, node.vstatus, w.basic_list, w.nonbasic_list, w.superbasic_list);
    }
    stage_lap(SUB_FRESH_BASIS);
    const double factor_t0 = cuopt::mathematical_optimization::tic();
    const bool ok          = device_lu || refactorize(s, /*initialize_basis=*/true);
    host_factor_s += cuopt::mathematical_optimization::toc(factor_t0);
    if (!ok) {
      node.drop = w.fail;
      error     = "fresh-node factorization failed";
      return false;
    }
    if (!device_lu &&
        !gds::gpu_simplified_dual_phase2_batch_factor_fits<i_t, f_t>(handle, w.ft, false)) {
      node.drop = gpu_batch_node_t<i_t, f_t>::drop_t::FACTOR_CAP_FRESH;
      error     = "fresh-node LU exceeds the seat's factor slot";
      return false;
    }
    if (gpu_pool_recheck() || dump_bad_nodes_dir() != nullptr) {
      node.vstatus_uploaded = node.vstatus;
      node.dy_se_at_launch  = node.dy_se;
      node.x_at_launch      = node.sol.x;
      node.y_at_launch      = node.sol.y;
      node.z_at_launch      = node.sol.z;
    }
    auto item      = make_item(s, gds::gpu_batch_seat_command::FRESH_NODE);
    item.device_lu = device_lu;
    stage_lap(SUB_FRESH_ITEM);
    static const std::vector<gds::gpu_bound_delta<i_t, f_t>> none;
    return gds::gpu_simplified_dual_phase2_batch_stage<i_t, f_t>(
      handle,
      static_cast<size_t>(s),
      item,
      gds::gpu_batch_seat_command::FRESH_NODE,
      node.root_lower != nullptr ? node.path_deltas : none,
      error);
  }

  bool stage_persistent_refactor(i_t s, std::string& error)
  {
    auto& node = nodes[s];
    auto& w    = st[s];
    assert(result_views || node.dy_se.size() == static_cast<size_t>(n));
    if (w.total_iters >= node.settings.iteration_limit) {
      node.status = simplex::dual_status_t::ITERATION_LIMIT;
      node.iters  = w.total_iters;
      node.solved = true;
      error = "GPU per-node iteration limit reached";
      return false;
    }
    if (w.leg_iters <= 0 && ++w.zero_pivot_legs >= 8) {
      node.drop = gpu_batch_node_t<i_t, f_t>::drop_t::LIVELOCK;
      error     = "GPU refactor-resume livelock";
      return false;
    }
    if (w.leg_iters > 0) { w.zero_pivot_legs = 0; }
    if (w.refactor_reason == 3) {  // fission yield: the seat's device state is intact, no host LU
      return gds::gpu_simplified_dual_phase2_batch_stage_in_place<i_t, f_t>(handle, static_cast<size_t>(s), error);
    }
    w.terminal_certificate_retry = (w.refactor_reason == 2);
    if (w.refactor_reason == 1) { ++rejected_refactors; }
    else if (w.refactor_reason == 2) { ++certificate_refactors; }
    else { ++periodic_refactors; }
    const double factor_t0 = cuopt::mathematical_optimization::tic();
    const bool ok          = refactorize(s, /*initialize_basis=*/false);
    host_factor_s += cuopt::mathematical_optimization::toc(factor_t0);
    if (!ok) {
      node.drop = w.fail;
      error     = "refactor-resume factorization/certificate failed";
      return false;
    }
    if (!gds::gpu_simplified_dual_phase2_batch_factor_fits<i_t, f_t>(handle, w.ft, true)) {
      node.drop = gpu_batch_node_t<i_t, f_t>::drop_t::FACTOR_CAP_RESUME;
      error     = "refactor-resume LU exceeds the seat's factor slot";
      return false;
    }
    f_t work = 0.0;
    const auto result = result_views
                          ? gds::gpu_simplified_dual_phase2_batch_result<i_t, f_t>(handle, s)
                          : gds::gpu_batch_result_view<i_t, f_t>{};
    simplex::phase2::compute_primal_solution_from_basis(lp,
                                                        node.lower,
                                                        node.upper,
                                                        w.ft,
                                                        result_views ? std::span<const i_t>(result.basic_list) : std::span<const i_t>(w.basic_list),
                                                        result_views ? std::span<const i_t>(result.nonbasic_list) : std::span<const i_t>(w.nonbasic_list),
                                                        result_views ? std::span<const simplex::variable_status_t>(result.vstatus) : std::span<const simplex::variable_status_t>(node.vstatus),
                                                        result_views ? result.x : std::span(node.sol.x),
                                                        w.xB_workspace,
                                                        work);
    w.refactor_resume = true;
    const auto item = make_item(s, gds::gpu_batch_seat_command::REFACTOR_RESUME);
    static const std::vector<gds::gpu_bound_delta<i_t, f_t>> none;
    return gds::gpu_simplified_dual_phase2_batch_stage<i_t, f_t>(handle,
                                                                 static_cast<size_t>(s),
                                                                 item,
                                                                 gds::gpu_batch_seat_command::REFACTOR_RESUME,
                                                                 none,
                                                                 error);
  }

  bool stage_persistent_child(i_t s,
                      const std::vector<gds::gpu_bound_delta<i_t, f_t>>& deltas,
                      std::string& error)
  {
    auto& node = nodes[s];
    auto& w    = st[s];
    assert(result_views || node.dy_se.size() == static_cast<size_t>(n));
    node.sol.resize(m, n);  // host output destination only; device x/y/z remain the parent's
    node.solved = false;
    node.drop   = gpu_batch_node_t<i_t, f_t>::drop_t::NONE;
    w.total_iters                = 0;
    w.leg_iters                  = 0;
    w.zero_pivot_legs            = 0;
    w.refactor_resume            = false;
    w.terminal_certificate_retry = false;
    w.refactor_reason            = 0;
    if (gpu_pool_recheck() || dump_bad_nodes_dir() != nullptr) {
      node.vstatus_uploaded = node.vstatus;
      node.dy_se_at_launch  = node.dy_se;
      node.x_at_launch.clear();
      node.y_at_launch.clear();
      node.z_at_launch.clear();
    }
    const auto item = make_item(s, gds::gpu_batch_seat_command::CHILD_CONTINUE);
    return gds::gpu_simplified_dual_phase2_batch_stage<i_t, f_t>(
      handle, static_cast<size_t>(s), item, gds::gpu_batch_seat_command::CHILD_CONTINUE, deltas,
      error);
  }

  // FRESH staging from a node prepared ahead (gpu_batch_factor_fresh): no host LU here. The source
  // node and seat state are only read and must outlive the relaunch.
  bool stage_persistent_fresh_from(i_t s,
                                   const gpu_batch_node_t<i_t, f_t>& cn,
                                   const batch_seat_state_t<i_t, f_t>& cw,
                                   std::string& error)
  {
    auto& node = nodes[s];
    auto& w    = st[s];
    node.solved                  = false;
    node.status                  = simplex::dual_status_t::UNSET;
    node.drop                    = gpu_batch_node_t<i_t, f_t>::drop_t::NONE;
    w.total_iters                = 0;
    w.leg_iters                  = 0;
    w.zero_pivot_legs            = 0;
    w.refactor_resume            = false;
    w.terminal_certificate_retry = false;
    w.refactor_reason            = 0;
    w.fail                       = gpu_batch_node_t<i_t, f_t>::drop_t::NONE;
    if (!gds::gpu_simplified_dual_phase2_batch_factor_fits<i_t, f_t>(handle, cw.ft, false)) {
      error = "prepared LU exceeds the seat's factor slot";
      return false;
    }
    gds::gpu_batch_item<i_t, f_t> it{};
    it.lp              = &lp;
    it.lower           = &cn.lower;
    it.upper           = &cn.upper;
    it.settings        = &cn.settings;
    it.vstatus         = const_cast<std::vector<simplex::variable_status_t>*>(&cn.vstatus);
    it.ft              = const_cast<simplex::basis_update_mpf_t<i_t, f_t>*>(&cw.ft);
    it.basic_list      = const_cast<std::vector<i_t>*>(&cw.basic_list);
    it.nonbasic_list   = const_cast<std::vector<i_t>*>(&cw.nonbasic_list);
    it.sol             = const_cast<simplex::lp_solution_t<i_t, f_t>*>(&cn.sol);
    it.dy_se           = const_cast<std::vector<f_t>*>(&cn.dy_se);
    it.refactor_reason = &w.refactor_reason;
    it.command         = gds::gpu_batch_seat_command::FRESH_NODE;
    // Root bounds + path deltas when the node carries them (as stage_persistent_fresh does).
    it.root_lower = cn.root_lower;
    it.root_upper = cn.root_upper;
    static const std::vector<gds::gpu_bound_delta<i_t, f_t>> none;
    return gds::gpu_simplified_dual_phase2_batch_stage<i_t, f_t>(
      handle, static_cast<size_t>(s), it, gds::gpu_batch_seat_command::FRESH_NODE,
      cn.root_lower != nullptr ? cn.path_deltas : none, error);
  }

  bool relaunch_persistent(std::string& error)
  {
    if (!persistent_seats || handle == nullptr) { return false; }
    const bool ok = gds::gpu_simplified_dual_phase2_batch_relaunch<i_t, f_t>(
      handle, timing, error);
    if (ok) { ++rounds; }
    return ok;
  }

  void destroy_persistent()
  {
    if (handle != nullptr) {
      gds::gpu_simplified_dual_phase2_batch_destroy<i_t, f_t>(handle);
      handle = nullptr;
    }
  }

  // Read back the finished round, service every REFACTOR_NEEDED seat, and launch the next round.
  // Returns true while a round is in flight.
  // HARVEST ONLY. Finishes the launch that is in flight, classifies every seat, and host-refactors
  // the ones that came back REFACTOR_NEEDED. It does NOT relaunch -- that is `relaunch()`, so the
  // caller sees one kernel launch per call and can go and do other work between them.
  //
  // Splitting these two was the point: while they were fused, a batch's REFACTOR_NEEDED legs were
  // invisible to the branch-and-bound worker loop, so "is the GPU done?" could not be answered with
  // a single test and the worker could not return to its own work between legs.
  bool harvest()
  {
    if (handle == nullptr) { return false; }
    gds::gpu_simplified_dual_phase2_batch_finish<i_t, f_t>(handle, out, timing);
    handle = nullptr;
    upload_s += timing.upload_s;
    kernel_s += timing.kernel_s;
    copyback_s += timing.copyback_s;
    if (timing.rejected || out.size() != items.size()) {
      std::fprintf(stderr, "[GPUBATCH] batch did not return a status per seat (%zu of %zu)\n",
                   out.size(), items.size());
      for (i_t s = 0; s < N; ++s) {
        if (st[s].active) { nodes[s].drop = gpu_batch_node_t<i_t, f_t>::drop_t::LAUNCH_FAILED; }
      }
      return false;
    }
    for (size_t k = 0; k < items.size(); ++k) {
      const i_t s = seat_of_item[k];
      auto& w     = st[s];
      auto& node  = nodes[s];

      // This item was already terminal before this fixed-seat relaunch. Its block was a no-op and
      // its host state must remain untouched.
      if (!w.active) { continue; }

      // ASSIGN, not accumulate: the device carries `iter` across a resume, so what comes back is
      // already this seat's running total.
      w.total_iters = node.sol.iterations;

      if (out[k] != simplex::dual_status_t::REFACTOR_NEEDED) {
        node.status = out[k];
        node.iters  = w.total_iters;
        node.solved = true;
        w.active    = false;
        continue;
      }

      // Same bound as the CPU plunging worker: the node's CUMULATIVE iteration count, not a count
      // of resume legs. A legitimate long node needs one leg per refactor_frequency pivots.
      if (w.total_iters >= node.settings.iteration_limit) {
        node.status = simplex::dual_status_t::ITERATION_LIMIT;
        node.iters  = w.total_iters;
        node.solved = true;
        w.active    = false;
        continue;
      }
      // Livelock guard, and the only thing a leg count is good for: a leg that takes zero pivots
      // makes no progress, so the iteration bound above would never be reached.
      if (node.sol.iterations == 0 && ++w.zero_pivot_legs >= 8) {
        node.drop = gpu_batch_node_t<i_t, f_t>::drop_t::LIVELOCK;
        w.active  = false;       // solved stays false -> the caller re-solves this node on the CPU
        continue;
      }
      if (node.sol.iterations != 0) { w.zero_pivot_legs = 0; }

      // Rebuild the factorization of the basis the device stopped on. No re-pricing: the device
      // CONTINUES the solve, and a freshly priced z can violate the dual-feasibility signs the
      // ratio test established against the maintained one.
      //
      // Snapshot the basis FIRST: refactor_basis repairs in place, so after a failure `node.vstatus`
      // is no longer the basis that failed and a dump taken then would not reproduce it.
      std::vector<simplex::variable_status_t> vs_before_refactor = node.vstatus;
      w.terminal_certificate_retry = (w.refactor_reason == 2);
      if (w.refactor_reason == 1) { ++rejected_refactors; }
      else if (w.refactor_reason == 2) { ++certificate_refactors; }
      else { ++periodic_refactors; }
      const double factor_t0 = cuopt::mathematical_optimization::tic();
      const bool factor_ok   = refactorize(s, /*initialize_basis*/ false);
      host_factor_s += cuopt::mathematical_optimization::toc(factor_t0);
      if (!factor_ok) {
        node.drop = w.fail;
        w.active  = false;       // solved stays false -> CPU
        // Capture the basis the DEVICE handed back, so it can be examined offline. The CPU driver
        // never fails this call on these instances, so every such basis is one only the device
        // reaches.
        if (dump_bad_nodes_dir() != nullptr) {
          static std::atomic<int> rseq{0};
          char path[1024];
          std::snprintf(path, sizeof(path), "%s/refacfail_m%d_n%d_%04d.bin",
                        dump_bad_nodes_dir(), (int)m, (int)n, rseq.fetch_add(1));
          const bool ok = bad_node::write<i_t, f_t>(
            path, lp, node.lower, node.upper, vs_before_refactor, node.dy_se_at_launch,
            node.x_at_launch, node.y_at_launch, node.z_at_launch, node.settings, static_cast<int>(out[k]), -1, node.sol.objective, 0.0,
            node.sol.iterations, 0);
          std::fprintf(stderr, "[REFACFAIL-GPU] leg_iters=%d cum=%d %s %s\n",
                       (int)node.sol.iterations, (int)w.total_iters,
                       ok ? "wrote" : "FAILED to write", path);
        }
        continue;
      }

      // Rebuild the primal point the way the CPU's in-place refactor does, and tell the device not
      // to redo it with its cold-start formula. THIS is the call that needs the bounds-explicit
      // overload: `lp` is shared across seats and its own bounds are not this node's.
      f_t work = 0.0;
      simplex::phase2::compute_primal_solution_from_basis(lp,
                                                          node.lower,
                                                          node.upper,
                                                          w.ft,
                                                          w.basic_list,
                                                          w.nonbasic_list,
                                                          node.vstatus,
                                                          node.sol.x,
                                                          w.xB_workspace,
                                                          work);
      w.refactor_resume = true;
    }
    // Any seat still active needs another launch; the caller decides when to issue it.
    for (i_t s = 0; s < N; ++s) {
      if (st[s].active) { return true; }
    }
    return false;
  }

  // Issue the next launch for whatever seats are still active. Separated from harvest() so one
  // worker-loop pass equals one kernel launch.
  bool relaunch() { return launch_round(); }

  // Broadcast the newest incumbent to every seat that is about to run. Called before EVERY launch,
  // whatever the seats' statuses were: the host has been back in the worker loop since the last
  // one, so an incumbent may have arrived from this worker or from any other. cut_off is unpacked
  // into a per-block copy at launch and never re-read, so a seat runs its whole leg on whatever
  // value was current when it started -- this is the only moment it can be updated.
  void set_cut_off(f_t cut_off)
  {
    for (i_t s = 0; s < N; ++s) { nodes[s].settings.cut_off = cut_off; }
  }

  // Backwards-compatible composition for the synchronous callers (the batch entry point used by
  // test_netlib), which have no worker loop to return to.
  bool advance()
  {
    if (!harvest()) { return false; }
    return relaunch();
  }

  void publish()
  {
  // ---- publish ---------------------------------------------------------------------------
  for (i_t s = 0; s < N; ++s) {
    if (!nodes[s].solved) { continue; }
    // total_iters, not the last leg's count: a node interrupted by REFACTOR_NEEDED is ONE solve.
    nodes[s].sol.iterations = nodes[s].iters;
    // The device returns `basic_list` / `nonbasic_list` for its FINAL basis, in the order it
    // maintains them (in-place swap, i.e. pivot history). Re-deriving them from `vstatus` here would
    // hand the caller a SORTED permutation instead -- a different order for the same basis set,
    // which the order-dependent ratio test would then scan differently on the node's next solve.
    nodes[s].basic_list    = st[s].basic_list;
    nodes[s].nonbasic_list = st[s].nonbasic_list;
  }
  if (dump_ != nullptr) {
    std::fclose(dump_);
    dump_ = nullptr;
  }
  if (gpu_pool_logging() && (lu_stats.fresh + lu_stats.resume) != 0) {
    const auto& L = lu_stats;
    const double n_all = static_cast<double>(L.fresh + L.resume);
    const double n_ok  = static_cast<double>(std::max<uint64_t>(L.ok, 1));
    std::string hist;
    for (size_t b = 0; b < L.nucleus_hist.size(); ++b) {
      hist += (b ? "," : "") + std::to_string(L.nucleus_hist[b]);
    }
    std::fprintf(stderr,
                 "[GPUBATCH-LU] m=%d fresh=%llu resume=%llu ok=%llu failed=%llu repaired=%llu "
                 "retried=%llu nucleus_hist=%s max_nucleus=%d avg_nucleus=%.1f avg_nucleus_nz=%.1f "
                 "avg_B_nz=%.1f avg_L_nz=%.1f avg_U_nz=%.1f host_s=%.4f\n",
                 (int)m, (unsigned long long)L.fresh, (unsigned long long)L.resume,
                 (unsigned long long)L.ok, (unsigned long long)L.failed,
                 (unsigned long long)L.repaired, (unsigned long long)L.retried, hist.c_str(),
                 L.max_nucleus, L.sum_nucleus / n_all, L.sum_nucleus_nz / n_all, L.sum_B_nz / n_all,
                 L.sum_L_nz / n_ok, L.sum_U_nz / n_ok, L.host_s);
  }
  if (std::getenv("CUOPT_BNB_GPU_BATCH_TIMING") != nullptr) {
    std::fprintf(stderr,
                 "[GPUBATCH-TIMING] seats=%d rounds=%d terminal_noops=%zu periodic_refactors=%d "
                 "rejected_refactors=%d certificate_refactors=%d upload=%.6f kernel=%.6f "
                 "copyback=%.6f host_factor=%.6f wall=%.6f\n",
                 (int)N, (int)rounds, terminal_noop_seats, (int)periodic_refactors,
                 (int)rejected_refactors, (int)certificate_refactors, upload_s, kernel_s,
                 copyback_s, host_factor_s, cuopt::mathematical_optimization::toc(t0));
  }
  }
};

// ---- asynchronous interface ---------------------------------------------------------------
template <typename i_t, typename f_t>
gpu_batch_run<i_t, f_t>* gpu_batch_begin(const simplex::lp_problem_t<i_t, f_t>& lp,
                                         std::vector<gpu_batch_node_t<i_t, f_t>>& nodes,
                                         rmm::cuda_stream_view stream,
                                         gds::gpu_ro_cache<i_t, f_t>* ro_cache,
                                         std::vector<batch_seat_state_t<i_t, f_t>>* seat_state)
{
  static const std::vector<int> devices = gpu_usable_devices();
  if (devices.empty() || nodes.empty()) { return nullptr; }
  auto* run = new gpu_batch_run_t<i_t, f_t>(lp, nodes, stream, ro_cache, *seat_state);
  run->prepare();
  if (!run->launch_round()) {   // nothing seatable at all
    run->publish();
    delete run;
    return nullptr;
  }
  return run;
}

template <typename i_t, typename f_t>
gpu_batch_run<i_t, f_t>* gpu_persistent_batch_begin(
  const simplex::lp_problem_t<i_t, f_t>& lp,
  std::vector<gpu_batch_node_t<i_t, f_t>>& nodes,
  rmm::cuda_stream_view stream,
  gds::gpu_ro_cache<i_t, f_t>* ro_cache,
  std::vector<batch_seat_state_t<i_t, f_t>>* seat_state,
  bool result_views,
  const std::vector<simplex::variable_type_t>* var_types)
{
  raft::common::nvtx::range scope("GPUPool::batch_begin");
  static const std::vector<int> devices = gpu_usable_devices();
  if (devices.empty() || nodes.empty() || seat_state == nullptr) { return nullptr; }
  const bool telemetry = gpu_pool_logging();
  gds::gpu_batch_set_setup_telemetry(telemetry);
  const double setup_t0 = telemetry ? cuopt::mathematical_optimization::tic() : 0.0;
  auto* run = new gpu_batch_run_t<i_t, f_t>(lp, nodes, stream, ro_cache, *seat_state, true, result_views);
  run->var_types = var_types;
  run->prepare();
  // The persistent pool never launches terminal/no-op seats. The caller replaces per-node preparation failures and
  // retries; nullptr here also represents pool-wide launch failures that require a drain.
  for (i_t s = 0; s < run->N; ++s) {
    if (!run->st[s].active && !nodes[s].parked) {
      run->publish();
      delete run;
      return nullptr;
    }
  }
  if (!run->launch_round()) {
    run->publish();
    delete run;
    return nullptr;
  }
  if (telemetry) {
    run->setup_timing    = run->timing;
    run->setup_host_lu_s = run->host_factor_s;   // prepare(): every seat's host LU
    run->setup_wall_s    = cuopt::mathematical_optimization::toc(setup_t0);
  }
  return run;
}

void gpu_pool_lap_t::start()
{
  if (sub != nullptr) { t = cuopt::mathematical_optimization::tic(); }
}

void gpu_pool_lap_t::lap(int k)
{
  if (sub == nullptr) { return; }
  const double now = cuopt::mathematical_optimization::tic();
  sub[k] += now - t;
  t = now;
}

template <typename i_t, typename f_t>
bool gpu_persistent_batch_harvest(gpu_batch_run<i_t, f_t>* run)
{
  return run != nullptr && run->harvest_persistent();
}

template <typename i_t, typename f_t>
gds::gpu_batch_result_view<i_t, f_t> gpu_persistent_batch_result(
  gpu_batch_run<i_t, f_t>* run, i_t seat)
{
  return gds::gpu_simplified_dual_phase2_batch_result<i_t, f_t>(run->handle,
                                                                  static_cast<size_t>(seat));
}

template <typename i_t, typename f_t>
void gpu_persistent_batch_fetch_seat(gpu_batch_run<i_t, f_t>* run, i_t seat)
{
  gds::gpu_simplified_dual_phase2_batch_fetch_seat<i_t, f_t>(run->handle, static_cast<size_t>(seat));
}

template <typename i_t, typename f_t>
void gpu_persistent_batch_fetch_seats(gpu_batch_run<i_t, f_t>* run, const std::vector<int>& seats)
{
  gds::gpu_simplified_dual_phase2_batch_fetch_seats<i_t, f_t>(run->handle, seats);
}

template <typename i_t, typename f_t>
void gpu_persistent_batch_read_seat(gpu_batch_run<i_t, f_t>* run,
                                    i_t seat,
                                    std::vector<simplex::variable_status_t>& vstatus,
                                    std::vector<f_t>& lower,
                                    std::vector<f_t>& upper,
                                    std::vector<f_t>& dy_se)
{
  gds::gpu_simplified_dual_phase2_batch_read_seat<i_t, f_t>(
    run->handle, static_cast<size_t>(seat), vstatus, lower, upper, dy_se);
}

template <typename i_t, typename f_t>
bool gpu_persistent_batch_park(gpu_batch_run<i_t, f_t>* run, i_t seat)
{
  run->nodes[seat].status = simplex::dual_status_t::NUMERICAL;
  return gds::gpu_simplified_dual_phase2_batch_park<i_t, f_t>(run->handle,
                                                                static_cast<size_t>(seat));
}

template <typename i_t, typename f_t>
void gpu_persistent_batch_unstage(gpu_batch_run<i_t, f_t>* run, i_t lo, i_t hi)
{
  gds::gpu_simplified_dual_phase2_batch_unstage<i_t, f_t>(
    run->handle, static_cast<size_t>(lo), static_cast<size_t>(hi));
}

template <typename i_t, typename f_t>
bool gpu_persistent_batch_launched(gpu_batch_run<i_t, f_t>* run)
{
  return run != nullptr && gds::gpu_simplified_dual_phase2_batch_launched<i_t, f_t>(run->handle);
}

template <typename i_t, typename f_t>
bool gpu_persistent_batch_stage_fresh(gpu_batch_run<i_t, f_t>* run, i_t seat, std::string& error)
{
  return run != nullptr && run->stage_persistent_fresh(seat, error);
}

template <typename i_t, typename f_t>
bool gpu_persistent_batch_stage_refactor(gpu_batch_run<i_t, f_t>* run, i_t seat, std::string& error)
{
  return run != nullptr && run->stage_persistent_refactor(seat, error);
}

template <typename i_t, typename f_t>
bool gpu_persistent_batch_stage_child(
  gpu_batch_run<i_t, f_t>* run,
  i_t seat,
  const std::vector<gds::gpu_bound_delta<i_t, f_t>>& deltas,
  std::string& error)
{
  return run != nullptr && run->stage_persistent_child(seat, deltas, error);
}

template <typename i_t, typename f_t>
bool gpu_persistent_batch_stage_fresh_from(gpu_batch_run<i_t, f_t>* run,
                                           i_t seat,
                                           const gpu_batch_node_t<i_t, f_t>& node,
                                           const batch_seat_state_t<i_t, f_t>& prepared,
                                           std::string& error)
{
  return run != nullptr && run->stage_persistent_fresh_from(seat, node, prepared, error);
}

template <typename i_t, typename f_t>
std::vector<char> gpu_batch_factor_fresh(const simplex::lp_problem_t<i_t, f_t>& lp,
                                         std::vector<gpu_batch_node_t<i_t, f_t>>& nodes,
                                         std::vector<batch_seat_state_t<i_t, f_t>>& seats,
                                         int threads)
{
  const i_t N = static_cast<i_t>(nodes.size());
  while (static_cast<i_t>(seats.size()) < N) {
    seats.emplace_back(lp.num_rows, nodes[seats.size()].settings.refactor_frequency);
  }
  for (auto& node : nodes) { node.sol.resize(lp.num_rows, lp.num_cols); }
  threads = std::max(1, std::min<int>(threads, N));
  // One run per thread (refactorize keeps per-run scratch), constructed before any thread starts.
  std::vector<std::unique_ptr<gpu_batch_run_t<i_t, f_t>>> runs;
  for (int t = 0; t < threads; ++t) {
    runs.push_back(std::make_unique<gpu_batch_run_t<i_t, f_t>>(
      lp, nodes, rmm::cuda_stream_view{}, nullptr, seats, true, true));
  }
  std::vector<char> ok(N, 0);
  std::atomic<i_t> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (i_t s = next++; s < N; s = next++) { ok[s] = runs[t]->refactorize(s, true) ? 1 : 0; }
    });
  }
  for (auto& th : pool) { th.join(); }
  return ok;
}

template <typename i_t, typename f_t>
bool gpu_persistent_batch_relaunch(gpu_batch_run<i_t, f_t>* run, std::string& error)
{
  return run != nullptr && run->relaunch_persistent(error);
}

template <typename i_t, typename f_t>
bool gpu_batch_run_ready(gpu_batch_run<i_t, f_t>* run)
{
  return run == nullptr || run->ready();
}

template <typename i_t, typename f_t>
bool gpu_batch_run_path_bounds(gpu_batch_run<i_t, f_t>* run)
{
  return run != nullptr && run->result_views && run->var_types != nullptr &&
         gds::gpu_simplified_dual_phase2_batch_lean_fresh<i_t, f_t>(run->handle);
}

template <typename i_t, typename f_t>
void gpu_batch_run_set_bounds_mirror(gpu_batch_run<i_t, f_t>* run, bool mirror)
{
  if (run != nullptr) { run->bounds_mirror = mirror; }
}

template <typename i_t, typename f_t>
void gpu_batch_run_set_dive(gpu_batch_run<i_t, f_t>* run, int rounds, const f_t* root_x)
{
  if (run == nullptr || !run->result_views) { return; }
  gds::gpu_simplified_dual_phase2_batch_set_dive<i_t, f_t>(run->handle, rounds, root_x);
}

template <typename i_t, typename f_t>
gds::gpu_dive_view<i_t, f_t> gpu_persistent_batch_dive(gpu_batch_run<i_t, f_t>* run, i_t seat)
{
  return gds::gpu_simplified_dual_phase2_batch_dive<i_t, f_t>(run->handle, static_cast<size_t>(seat));
}

template <typename i_t, typename f_t>
void gpu_batch_run_set_pseudocosts(gpu_batch_run<i_t, f_t>* run,
                                   const f_t* sum_down,
                                   const f_t* sum_up,
                                   const i_t* num_down,
                                   const i_t* num_up,
                                   f_t avg_down,
                                   f_t avg_up,
                                   i_t reliable_threshold,
                                   f_t integer_tol)
{
  if (run == nullptr || !run->result_views) { return; }
  gds::gpu_simplified_dual_phase2_batch_set_pseudocosts<i_t, f_t>(
    run->handle, sum_down, sum_up, num_down, num_up, avg_down, avg_up, reliable_threshold,
    integer_tol);
}

template <typename i_t, typename f_t>
gpu_batch_run_metrics gpu_batch_run_get_metrics(const gpu_batch_run<i_t, f_t>* run)
{
  gpu_batch_run_metrics result;
  if (run == nullptr) { return result; }
  result.seats                   = static_cast<int>(run->N);
  result.rounds                  = static_cast<int>(run->rounds);
  result.terminal_noop_seats     = run->terminal_noop_seats;
  result.periodic_refactors      = static_cast<int>(run->periodic_refactors);
  result.rejected_refactors      = static_cast<int>(run->rejected_refactors);
  result.certificate_refactors   = static_cast<int>(run->certificate_refactors);
  result.upload_s                = run->upload_s;
  result.kernel_s                = run->kernel_s;
  result.copyback_s              = run->copyback_s;
  result.host_factor_s           = run->host_factor_s;
  result.last_harvest_d2h_s      = run->last_harvest_d2h_s;
  result.last_harvest_d2h_enqueue_s = run->last_harvest_d2h_enqueue_s;
  result.last_harvest_d2h_device_s  = run->last_harvest_d2h_device_s;
  result.last_harvest_scatter_s  = run->last_harvest_scatter_s;
  result.last_harvest_unpack_s   = run->last_harvest_unpack_s;
  result.last_prekernel_s        = run->last_prekernel_s;
  result.last_prekernel_lu_s      = run->last_prekernel_lu_s;
  result.last_prekernel_h2d_s     = run->last_prekernel_h2d_s;
  result.last_prekernel_scatter_s = run->last_prekernel_scatter_s;
  result.last_prekernel_deltas_s  = run->last_prekernel_deltas_s;
  result.last_done_after_pre_s    = run->last_done_after_pre_s;
  result.wall_s = cuopt::mathematical_optimization::toc(run->t0);
  result.setup_wall_s            = run->setup_wall_s;
  result.setup_host_lu_s         = run->setup_host_lu_s;
  result.setup_alloc_s           = run->setup_timing.setup_alloc_s;
  result.setup_alloc_n           = run->setup_timing.setup_alloc_n;
  result.setup_h2d_s             = run->setup_timing.setup_h2d_s;
  result.setup_h2d_n             = run->setup_timing.setup_h2d_n;
  result.setup_h2d_bytes         = run->setup_timing.setup_h2d_bytes;
  result.setup_memset_s          = run->setup_timing.setup_memset_s;
  result.setup_memset_n          = run->setup_timing.setup_memset_n;
  return result;
}

// Harvest the launch in flight (results + host refactor). True => seats remain, so the caller
// should process them and then call gpu_batch_run_relaunch.
template <typename i_t, typename f_t>
bool gpu_batch_run_harvest(gpu_batch_run<i_t, f_t>* run)
{
  return run != nullptr && run->harvest();
}

// Issue the next launch. True => a launch is now in flight.
template <typename i_t, typename f_t>
bool gpu_batch_run_relaunch(gpu_batch_run<i_t, f_t>* run)
{
  return run != nullptr && run->relaunch();
}

template <typename i_t, typename f_t>
void gpu_batch_run_set_cut_off(gpu_batch_run<i_t, f_t>* run, f_t cut_off)
{
  if (run != nullptr) { run->set_cut_off(cut_off); }
}

template <typename i_t, typename f_t>
bool gpu_batch_run_advance(gpu_batch_run<i_t, f_t>* run)
{
  return run != nullptr && run->advance();
}

template <typename i_t, typename f_t>
void gpu_batch_run_end(gpu_batch_run<i_t, f_t>* run)
{
  if (run == nullptr) { return; }
  if (run->persistent_seats) { run->destroy_persistent(); }
  run->publish();
  delete run;
}

// ---- synchronous entry point (unchanged behaviour) ----------------------------------------
template <typename i_t, typename f_t>
void gpu_solve_nodes_batch(const simplex::lp_problem_t<i_t, f_t>& lp,
                           std::vector<gpu_batch_node_t<i_t, f_t>>& nodes,
                           rmm::cuda_stream_view stream,
                           gds::gpu_ro_cache<i_t, f_t>* ro_cache)
{
  // No pool here, so the seat state is local and the factorization is not carried anywhere -- this
  // entry solves one batch and returns.
  std::vector<batch_seat_state_t<i_t, f_t>> local_seat_state;
  auto* run = gpu_batch_begin<i_t, f_t>(lp, nodes, stream, ro_cache, &local_seat_state);
  if (run == nullptr) { return; }
  while (gpu_batch_run_advance<i_t, f_t>(run)) {}
  gpu_batch_run_end<i_t, f_t>(run);
}

template void gpu_solve_nodes_batch<int, double>(const simplex::lp_problem_t<int, double>&,
                                                 std::vector<gpu_batch_node_t<int, double>>&,
                                                 rmm::cuda_stream_view,
                                                 gds::gpu_ro_cache<int, double>*);
template gpu_batch_run<int, double>* gpu_batch_begin<int, double>(
  const simplex::lp_problem_t<int, double>&, std::vector<gpu_batch_node_t<int, double>>&,
  rmm::cuda_stream_view, gds::gpu_ro_cache<int, double>*,
  std::vector<batch_seat_state_t<int, double>>*);
template gpu_batch_run<int, double>* gpu_persistent_batch_begin<int, double>(
  const simplex::lp_problem_t<int, double>&, std::vector<gpu_batch_node_t<int, double>>&,
  rmm::cuda_stream_view, gds::gpu_ro_cache<int, double>*,
  std::vector<batch_seat_state_t<int, double>>*, bool, const std::vector<simplex::variable_type_t>*);
template bool gpu_persistent_batch_harvest<int, double>(gpu_batch_run<int, double>*);
template gds::gpu_batch_result_view<int, double> gpu_persistent_batch_result<int, double>(
  gpu_batch_run<int, double>*, int);
template void gpu_persistent_batch_fetch_seat<int, double>(gpu_batch_run<int, double>*, int);
template bool gpu_persistent_batch_stage_fresh_from<int, double>(
  gpu_batch_run<int, double>*, int, const gpu_batch_node_t<int, double>&,
  const batch_seat_state_t<int, double>&, std::string&);
template std::vector<char> gpu_batch_factor_fresh<int, double>(
  const simplex::lp_problem_t<int, double>&, std::vector<gpu_batch_node_t<int, double>>&,
  std::vector<batch_seat_state_t<int, double>>&, int);
template void gpu_persistent_batch_read_seat<int, double>(gpu_batch_run<int, double>*,
                                                         int,
                                                         std::vector<simplex::variable_status_t>&,
                                                         std::vector<double>&,
                                                         std::vector<double>&,
                                                         std::vector<double>&);
template void gpu_batch_run_set_dive<int, double>(gpu_batch_run<int, double>*, int, const double*);
template gds::gpu_dive_view<int, double> gpu_persistent_batch_dive<int, double>(
  gpu_batch_run<int, double>*, int);
template bool gpu_persistent_batch_park<int, double>(gpu_batch_run<int, double>*, int);
template void gpu_persistent_batch_unstage<int, double>(gpu_batch_run<int, double>*, int, int);
template bool gpu_persistent_batch_launched<int, double>(gpu_batch_run<int, double>*);
template void gpu_persistent_batch_fetch_seats<int, double>(gpu_batch_run<int, double>*,
                                                            const std::vector<int>&);
template bool gpu_persistent_batch_stage_fresh<int, double>(
  gpu_batch_run<int, double>*, int, std::string&);
template bool gpu_persistent_batch_stage_refactor<int, double>(
  gpu_batch_run<int, double>*, int, std::string&);
template bool gpu_persistent_batch_stage_child<int, double>(
  gpu_batch_run<int, double>*, int,
  const std::vector<gds::gpu_bound_delta<int, double>>&, std::string&);
template bool gpu_persistent_batch_relaunch<int, double>(gpu_batch_run<int, double>*, std::string&);
template bool gpu_batch_run_ready<int, double>(gpu_batch_run<int, double>*);
template bool gpu_batch_run_path_bounds<int, double>(gpu_batch_run<int, double>*);
template void gpu_batch_run_set_bounds_mirror<int, double>(gpu_batch_run<int, double>*, bool);
template void gpu_batch_run_set_pseudocosts<int, double>(
  gpu_batch_run<int, double>*, const double*, const double*, const int*, const int*, double, double,
  int, double);
template gpu_batch_run_metrics gpu_batch_run_get_metrics<int, double>(
  const gpu_batch_run<int, double>*);
template bool gpu_batch_run_advance<int, double>(gpu_batch_run<int, double>*);
template bool gpu_batch_run_harvest<int, double>(gpu_batch_run<int, double>*);
template bool gpu_batch_run_relaunch<int, double>(gpu_batch_run<int, double>*);
template void gpu_batch_run_set_cut_off<int, double>(gpu_batch_run<int, double>*, double);
template void gpu_batch_run_end<int, double>(gpu_batch_run<int, double>*);


}  // namespace cuopt::mathematical_optimization::mip
