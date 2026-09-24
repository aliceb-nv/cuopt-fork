/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// gpu_simplified_dual_phase2: a faithful, single-thread GPU port of the CPU
// dual-simplex driver `simplified_dual_phase2` (dual_simplex/phase2.cpp).
//
// It has the EXACT same signature and produces the EXACT same results as the
// CPU function. The wrapper is just a prologue (marshal inputs CPU->GPU) and
// epilogue (marshal outputs GPU->CPU) around a single <<<1,1>>> kernel that
// replays the CPU algorithm line-for-line on the device.
//
// Scope: PURE MPF STACKING only (no on-device LU refactorization). It matches
// the CPU default behavior (CUOPT_SIMPLIFIED_REFACTOR unset / 0).

#include <rmm/cuda_stream_view.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <math_optimization/types.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// The shared simplex primitives (lp_problem_t, variable_status_t,
// basis_update_mpf_t, dual::status_t, ...) live in mathematical_optimization::simplex;
// make them visible unqualified here.
using namespace cuopt::mathematical_optimization::simplex;

// GPU port of simplified_dual_phase2. `delta_y_steepest_edge` must contain one maintained norm per
// column; unlike the generic CPU driver, this B&B-oriented path does not initialize an empty
// vector. Only instantiated for <int, double>.
template <typename i_t, typename f_t>
dual::status_t gpu_simplified_dual_phase2(const lp_problem_t<i_t, f_t>& lp,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          std::vector<variable_status_t>& vstatus,
                                          basis_update_mpf_t<i_t, f_t>& ft,
                                          std::vector<i_t>& basic_list,
                                          std::vector<i_t>& nonbasic_list,
                                          lp_solution_t<i_t, f_t>& sol,
                                          std::vector<f_t>& delta_y_steepest_edge);

// ---------------------------------------------------------------------------
// Multi-problem batch: solve N independent problems in ONE launch (one block
// per problem). Used by the throughput benchmark. Each item must already be set
// up exactly as a single solve is — a FRESH factorized `ft` (num_updates == 0),
// `basic_list`/`nonbasic_list` from `vstatus`, and duals in `sol.y`/`sol.z` —
// i.e. the same steps run_phase2_from_basis performs before the driver call.
// The pointers are borrowed (not owned) and must outlive the call.
// ---------------------------------------------------------------------------
enum class gpu_batch_seat_command : int {
  FRESH_NODE      = 0,
  REFACTOR_RESUME = 1,
  CHILD_CONTINUE  = 2
};

// One changed child bound. `mask & 1` updates lower and `mask & 2` updates upper.  Keeping the
// update sparse is important for a plunge: the parent bounds are already resident in the seat and
// branching normally changes one column (bound strengthening may add a few more).
template <typename i_t, typename f_t>
struct gpu_bound_delta {
  i_t column{};
  f_t lower{};
  f_t upper{};
  unsigned char mask{};
};

template <typename i_t, typename f_t>
struct gpu_batch_item {
  const lp_problem_t<i_t, f_t>* lp;
  // Per-node variable bounds. Null => use lp->lower / lp->upper.
  //
  // This exists so a batch of branch-and-bound nodes can SHARE one lp_problem_t. Every node of a
  // tree has the same A, rhs and objective and differs only in its bounds, but the read-only dedup
  // keys on `lp->A.col_start.data()` -- so giving each node its own lp_problem_t copy would upload
  // the matrix once per node instead of once per batch.
  const std::vector<f_t>* lower;
  const std::vector<f_t>* upper;
  const simplex_solver_settings_t<i_t, f_t>* settings;
  std::vector<variable_status_t>* vstatus;  // in: warm-start basis; out: FINAL basis
  basis_update_mpf_t<i_t, f_t>* ft;
  std::vector<i_t>* basic_list;
  std::vector<i_t>* nonbasic_list;
  lp_solution_t<i_t, f_t>* sol;  // in: y/z (duals); out: x, objective, user_objective, iterations
                                 // y/z are copied back for callers with a host resume loop
  std::vector<f_t>* dy_se;       // carried steepest-edge norms; size must equal lp->num_cols
  // Set on a REFACTOR_NEEDED resume. The host has already rebuilt the primal point from the fresh
  // factorization exactly as the CPU driver's in-place refactor does
  // (gpu_recompute_primal_after_refactor), so the device must NOT recompute it: its own cold-start
  // reconstruction uses a different formula -- it sets the nonbasic values from the sign of z and
  // skips columns with |x_j| < 10*tight_tol -- and the difference moves the trajectory.
  bool refactor_resume = false;

  // Set only on the relaunch requested by refactor_reason=2. The host rebuilt the same terminal
  // basis specifically to retry the optimality certificate; this distinguishes that one-shot
  // proof retry from periodic and rejected-update refactor resumes.
  bool terminal_certificate_retry = false;

  // OUT, optional (null => not reported). Why the solve came back REFACTOR_NEEDED:
  //   0 = periodic schedule -- num_updates passed refactor_frequency. The pending rank-1 update was
  //       SKIPPED, exactly as the CPU driver skips it on the same trigger. The maintained y/z are
  //       consistent; carry them.
  //   1 = the rank-1 update was REJECTED as unsafe (|mu| = |1 + v'u| outside [1e-8, 1e8]) AFTER the
  //       pivot's basis swap had already been applied. The host must refactor that returned basis,
  //       then continue with the maintained y/z. Fresh pricing is not a rejection certificate for
  //       this recovery path: it can temporarily violate the signs before the next leg restores
  //       feasibility.
  //   2 = a terminal optimality certificate was too weak with stacked updates. Refactor the same
  //       terminal basis and retry once; maintained y/z may be carried because the stack was not
  //       rejected and the certificate itself always prices a fresh y from the new LU.
  // Only meaningful when the returned status is REFACTOR_NEEDED. Reasons 0 and 1 mirror the CPU
  // drivers at the same places; reason 2 is the GPU certificate safeguard's one-shot retry.
  int* refactor_reason = nullptr;
  // Set when this seat CONTINUES a plunge on its parent's factorization, so the launch-time
  // "factorization must be fresh" check knows a warm MPF stack is intended, not a bug.
  bool continue_basis = false;

  // Fixed-seat relaunch marker. A synchronous host-refactor loop keeps the same item array for
  // every round; once a seat has a terminal answer, its block returns immediately on later
  // launches. No mutable pointers above are read for such an item, and the host preserves this
  // status directly instead of copying solver state back again.
  dual::status_t terminal_status = dual::status_t::UNSET;

  // Persistent-pool command. One-shot launches leave the default and therefore retain fresh-slot
  // behaviour. REFACTOR_RESUME replaces the base LU while clearing the MPF stack, and
  // CHILD_CONTINUE keeps the live LU+MPF representation and applies only the sparse bound delta
  // before repricing the child.
  gpu_batch_seat_command command = gpu_batch_seat_command::FRESH_NODE;
};

// Timing + pre-flight result for a batch launch. If `rejected` is true the batch
// did NOT run (VRAM pre-flight failed); `message` explains needed-vs-free.
struct gpu_batch_timing {
  double upload_s          = 0.0;  // host: marshal + H2D of all N problems
  double kernel_s          = 0.0;  // device compute (CUDA events, kernel only)
  double copyback_s        = 0.0;  // host: D2H of all N results
  // Split of a launch's marshal (all N problems + the shared read-only block), filled only while
  // gpu_batch_set_setup_telemetry(true): host wall time spent inside each class of CUDA call.
  double setup_alloc_s       = 0.0;  // device_buffer allocations
  std::size_t setup_alloc_n  = 0;
  double setup_h2d_s         = 0.0;  // host-to-device copies
  std::size_t setup_h2d_n    = 0;
  std::size_t setup_h2d_bytes = 0;
  double setup_memset_s      = 0.0;  // device memsets
  std::size_t setup_memset_n = 0;
  std::size_t vram_needed_bytes = 0;
  std::size_t vram_free_bytes   = 0;
  bool rejected            = false;
  std::string message;
};

// Solve every item in one <<<N,1>>> launch. The kernel does no in-kernel malloc: every buffer it
// touches is preallocated by the host, so there is no device heap to size.
// Fails loud: if the batch will not fit it sets timing.rejected + timing.message and
// returns without launching (never a silent OOM). On success fills out_status[i] and
// each item->sol. Only instantiated for <int, double>.
// ---------------------------------------------------------------------------
// Asynchronous form of the batch solve. `_launch` marshals, uploads and launches without
// synchronising and hands back an opaque in-flight batch; `_ready` polls it; `_finish` waits, copies
// every result back and destroys the handle. Branch-and-bound uses these so the worker that summoned
// the GPU keeps plunging on the CPU while its nodes run.
//
// `_launch` returns nullptr if the batch was rejected -- timing.rejected / timing.message say why.
//
// CONCURRENT BATCHES ARE FINE. This used to say "one batch in flight per device" because the launch
// path wrote device globals (the work-stealing counter, the capacity flag, the solve-class config).
// All three are gone -- the counter is a per-launch kernel parameter and the other two were deleted
// -- so batches on different streams no longer interfere.
// ---------------------------------------------------------------------------
namespace gpu_phase2_detail {
template <typename i_t, typename f_t>
struct gpu_batch_handle_t;
template <typename i_t, typename f_t>
struct gpu_ro_cache_t;
}
template <typename i_t, typename f_t>
using gpu_batch_handle = gpu_phase2_detail::gpu_batch_handle_t<i_t, f_t>;

// ---------------------------------------------------------------------------
// A read-only upload that outlives a single launch: A, rhs, objective and the Arow master, which
// every node of one branch-and-bound tree shares. Create ONE per pool, pass it to every launch that
// pool makes, destroy it with the pool. Without it the immutable problem block is re-uploaded on
// every launch, which in-seat plunging turns into many launches per cycle.
//
// It holds ONLY immutable data. Per-seat mutable state -- bounds, LU, permutations, and the Arow
// WORKING copy that the nonbasic scan permutes in place -- must stay per launch; sharing any of it
// across co-resident seats is a data race.
//
// The cache is bound to the stream it was created on; use the same stream for its launches.
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
using gpu_ro_cache = gpu_phase2_detail::gpu_ro_cache_t<i_t, f_t>;

template <typename i_t, typename f_t>
// The cache OWNS its stream: it is shared by every pool on a device, so it cannot borrow one pool's
// stream without tying its lifetime to that pool's.
gpu_ro_cache<i_t, f_t>* gpu_ro_cache_create();
// DIAGNOSTIC: cache borrows the given stream instead of owning one. Reproduces the pre-fix layout.
template <typename i_t, typename f_t>
gpu_ro_cache<i_t, f_t>* gpu_ro_cache_create_borrowed(rmm::cuda_stream_view borrowed);

template <typename i_t, typename f_t>
void gpu_ro_cache_destroy(gpu_ro_cache<i_t, f_t>* cache);

// `allow_refactor_exit` selects a SECOND kernel instantiation in which the periodic-refactorization
// and unsafe-update exits are compiled IN, so the batch can return REFACTOR_NEEDED per problem.
//
// It is off by default and must stay off for any caller without a host resume loop: a
// REFACTOR_NEEDED returned to a caller that cannot answer it strands that problem with no result.
// Measured with the exits wrongly live in the benchmark path: 180 of 500 problems came back
// REFACTOR_NEEDED and 345 diverged from the sequential reference.
//
// Branch-and-bound sets it to true. The cost is one more kernel to compile and the branches the
// `false` variant does not carry (the batch path measured +0.7% from having them compiled out).
template <typename i_t, typename f_t>
gpu_batch_handle<i_t, f_t>* gpu_simplified_dual_phase2_batch_launch(
  const std::vector<gpu_batch_item<i_t, f_t>>& items,
  gpu_batch_timing& timing,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default,
  bool allow_refactor_exit     = false,
  gpu_ro_cache<i_t, f_t>* ro_cache = nullptr,
  bool persistent_seats = false);

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_ready(gpu_batch_handle<i_t, f_t>* handle);

// Host telemetry switch for the gpu_batch_timing::setup_* split (process-wide, default off). With
// it off the marshal path takes no timestamps.
void gpu_batch_set_setup_telemetry(bool on);

// Per-thread CUDA call accounting for host telemetry. gpu_call_stats_begin() resets and enables it
// on the calling thread; gpu_call_stats_end() disables it and returns the host wall time spent in
// allocations, host-to-device copies and memsets made by this file (e.g. persistent staging) since.
struct gpu_call_stats {
  double alloc_s{0.0}, h2d_s{0.0}, memset_s{0.0};
  std::size_t alloc_n{0}, h2d_n{0}, h2d_bytes{0}, memset_n{0};
};
void gpu_call_stats_begin();
gpu_call_stats gpu_call_stats_end();

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_finish(gpu_batch_handle<i_t, f_t>* handle,
                                             std::vector<dual::status_t>& out_status,
                                             gpu_batch_timing& timing);

// Persistent-seat form used by the B&B pool. Harvest copies one completed full-pool launch but
// keeps every device allocation alive. The host then stages exactly one command for every seat and
// relaunches the complete array once. Staging only enqueues stream-ordered H2D copies, allowing
// fresh replacement seats to be copied before CPU refactor/branch work is performed.
template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_harvest(gpu_batch_handle<i_t, f_t>* handle,
                                              std::vector<dual::status_t>& out_status,
                                              gpu_batch_timing& timing);

// Nondeterministic B&B keeps the harvested arrays as the authoritative host state. A view remains
// valid until the next harvest or until the handle is destroyed; writes through it update the
// staging array used by the next bulk upload. The default batch API still scatters into items.
template <typename i_t, typename f_t>
struct gpu_batch_result_view {
  std::span<f_t> x, y, z, dy_se;
  std::span<variable_status_t> vstatus;
  std::span<i_t> basic_list, nonbasic_list;
  dual::status_t status{dual::status_t::UNSET};
  f_t objective{}, user_objective{};
  i_t iterations{}, refactor_reason{};
};

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_use_views(gpu_batch_handle<i_t, f_t>* handle);

template <typename i_t, typename f_t>
gpu_batch_result_view<i_t, f_t> gpu_simplified_dual_phase2_batch_result(
  gpu_batch_handle<i_t, f_t>* handle, std::size_t seat);

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_stage(
  gpu_batch_handle<i_t, f_t>* handle,
  std::size_t seat,
  const gpu_batch_item<i_t, f_t>& item,
  gpu_batch_seat_command command,
  const std::vector<gpu_bound_delta<i_t, f_t>>& bound_deltas,
  std::string& error);

template <typename i_t, typename f_t>
bool gpu_simplified_dual_phase2_batch_relaunch(gpu_batch_handle<i_t, f_t>* handle,
                                               gpu_batch_timing& timing,
                                               std::string& error);

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch_destroy(gpu_batch_handle<i_t, f_t>* handle);

template <typename i_t, typename f_t>
void gpu_simplified_dual_phase2_batch(const std::vector<gpu_batch_item<i_t, f_t>>& items,
                                      std::vector<dual::status_t>& out_status,
                                      gpu_batch_timing& timing,
                                      bool allow_refactor_exit = false);

// Blocks per SM the batch kernel actually achieves at its 32-thread block size, from
// cudaOccupancyMaxActiveBlocksPerMultiprocessor on the current device. Callers sizing a node pool
// need this: register pressure decides it, and 65536/(32*regs) is only a ceiling that ignores the
// shared-memory carveout and the hardware blocks/SM limit. Returns 0 if the query fails.
int gpu_max_blocks_per_sm();

// On a REFACTOR_NEEDED resume the host must rebuild the primal point before calling back, and it
// must use the CPU driver's own formula: call `simplex::phase2::compute_primal_solution_from_basis`
// (declared in dual_simplex/phase2.hpp) directly, then set `gpu_batch_item::refactor_resume` so the
// device skips its cold-start reconstruction. There is deliberately NO GPU-side copy of that
// routine -- a second formula would have to be kept in step with the first, and the two differ:
// the cold start sets nonbasic values from the sign of z and skips columns with |x_j| < 10*tight_tol.

}  // namespace cuopt::linear_programming::dual_simplex
