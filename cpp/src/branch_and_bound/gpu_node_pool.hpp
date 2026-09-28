/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/mip_node.hpp>
#include <branch_and_bound/node_queue.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/gpu_phase2/gpu_phase2.cuh>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>

#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace cuopt::mathematical_optimization::mip {

namespace gds = cuopt::linear_programming::dual_simplex;

// ---------------------------------------------------------------------------
// GPU node pool: offload branch-and-bound node LPs to the batched GPU dual simplex.
//
// The live pool is `gpu_batch_pool_t` further down, one per plunging worker, driven by
// `gpu_pool_cycle` in branch_and_bound.cpp.
//
// MULTI-GPU ONLY. Device 0 keeps PDLP / feasibility-jump / probing, so the pool engages only when
// the machine has more than one device and claims 1..K-1. `CUOPT_BNB_GPU_DEVICE` overrides the
// device for testing on a single-GPU machine; it is a test hook, not a production path.
// ---------------------------------------------------------------------------

// Devices available to the dual simplex: devices 1..num_gpus-1 of the `num_gpus` solver parameter
// (-1 = every visible device), clamped to the visible count. Device 0 keeps the heuristics. These
// are exactly the devices the entry point gave an RMM pool to (cuopt_cli sets one per device in
// [0, num_gpus)). Empty for num_gpus <= 1 unless the test override names a device.
std::vector<int> gpu_usable_devices(int num_gpus = -1);

// ---------------------------------------------------------------------------
// Serial substitution (TEST PATH, CUOPT_BNB_GPU_SERIAL=1).
//
// Solves ONE node on the GPU as a batch of one, synchronously, in place of the CPU driver. No pool,
// no slices, no overlap -- deliberately: it isolates "does the GPU LP give the same answer" from
// "does the parallel pool perturb the search". With settings.deterministic and a node_limit, two
// runs walk the same tree while LP results agree, so any divergence is attributable to one node and
// is reproducible.
//
// Returns false if the node could not be prepared (bound strengthening proved it infeasible, or the
// basis would not factorize); the caller then falls back to the CPU driver.
//
// THE CALLER MUST FORCE recompute_basis = true AFTERWARDS. The device pivots its own copy of the
// factorization, so the worker's host-side basis_factors no longer matches the returned basis. A
// continuing plunge that reused it would hand the next node a factorization for the wrong basis.
template <typename i_t, typename f_t>
bool gpu_solve_node_serial(const simplex::lp_problem_t<i_t, f_t>& lp,
                           const csr_matrix_t<i_t, f_t>& Arow,
                           const std::vector<simplex::variable_type_t>& var_types,
                           const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                           const std::vector<f_t>& lower,
                           const std::vector<f_t>& upper,
                           std::vector<simplex::variable_status_t>& vstatus,
                           std::vector<i_t>& basic_list,
                           std::vector<i_t>& nonbasic_list,
                           simplex::lp_solution_t<i_t, f_t>& sol,
                           std::vector<f_t>& dy_se,
                           i_t& iters,
                           simplex::dual_status_t& status);

// True when the serial test path is enabled.
bool gpu_serial_enabled();

// ---------------------------------------------------------------------------
// WHOLE-BATCH POOL (CUOPT_BNB_GPU_BATCH=<seats>, 0 = off).
//
// One pool per plunging worker. The worker fills every seat from its own queue, launches ALL of them
// in ONE kernel on ONE stream, and harvests the pool as a unit. Blocks steal seats from a per-launch
// counter, so the grid is min(resident_blocks, seats) rather than one block per seat.
//
// Versus the serial substitution above, the only new machinery is the PER-SEAT resume state machine:
// the device returns REFACTOR_NEEDED per problem, and seats hit it at different times, so each
// relaunch carries only the seats that are still running. A seat that finishes drops out.
//
// TAILING IS ACCEPTED. A launch ends when its slowest seat ends. That is bounded here because every
// long seat stops for a host refactorization anyway, and the production shape runs many pools
// concurrently, so the device stays busy across workers even while one pool is in its final wave.
// ---------------------------------------------------------------------------

// One node's input and output. The caller owns these; the batch borrows them for the call.
template <typename i_t, typename f_t>
struct gpu_batch_node_t {
  // PLUNGE CONTINUATION. Set when this seat is being refilled with the child the search would have
  // dived into next. The seat then KEEPS the parent's factorization -- LU plus the accumulated MPF
  // rank-1 updates and their count -- and its basic_list / nonbasic_list in pivot-history order,
  // instead of being refactorized from vstatus. That is what the CPU does: `recompute_basis` is
  // `node_status != HAS_CHILDREN`, so the factorization is reused while diving and rebuilt only on
  // leaving the dive. Carrying the stack without carrying its count would be worse than not
  // carrying it, because the refactor schedule fires on the CUMULATIVE count.
  bool continue_basis{false};
  // Lean begin: no node at the first launch; marshalled from a template seat and parked.
  bool parked{false};

  // ---- in ----
  // This node's bounds, already reconstructed and strengthened. Every seat shares the caller's ONE
  // lp_problem_t for A / rhs / objective and differs only here.
  //
  // Held BY VALUE, not by pointer, for two reasons. The batch borrows them for the whole call, so a
  // pointer into caller scratch that the caller then reuses for the next node would be read after it
  // changed. And the consumer re-derives the node's bounds and compares against these: the incumbent
  // can improve mid-cycle, and reduced-cost fixing then rewrites `original_lp_.lower/upper` globally
  // (branch_and_bound.cpp:3438), so a later seat's true LP is no longer the one the device was
  // given. Keeping the fill-time bounds makes that check exact.
  std::vector<f_t> lower, upper;
  // Non-null: bounds built on the device from these and path_deltas.
  const std::vector<f_t>* root_lower{nullptr};
  const std::vector<f_t>* root_upper{nullptr};
  std::vector<gds::gpu_bound_delta<i_t, f_t>> path_deltas;
  // Non-null: vstatus, basis lists and dy_se built on the device.
  const std::vector<std::uint8_t>* packed_vstatus{nullptr};
  const std::vector<f_t>* root_dy_se{nullptr};
  // Per node, because the cutoff is derived from the incumbent as it stood when the seat was filled.
  simplex::simplex_solver_settings_t<i_t, f_t> settings;

  // ---- in / out ----
  std::vector<simplex::variable_status_t> vstatus;   // in: warm start; out: FINAL basis
  std::vector<f_t> dy_se;                            // carried steepest-edge norms (may be empty)

  // ---- snapshots, for a replayable dump of a node whose answer is later disputed ----
  //
  // Neither of the two above can serve. Both are in/out and the solve overwrites them, so capturing
  // after the fact records the FINAL state, not what the device consumed.
  //
  // `vstatus_uploaded` is taken AFTER the first `refactor_basis`, which may have REPAIRED the basis.
  // Dumping the entry snapshot instead makes a replay start from a basis the device never had, which
  // manufactures a divergence that was not there.
  std::vector<simplex::variable_status_t> vstatus_uploaded;
  std::vector<f_t> dy_se_at_launch;   // taken immediately before the FIRST launch, not at fill time
  // Same snapshot discipline for the SOLVE STATE. Without these a dump cannot reproduce a seat that
  // was mid-solve: a REFACTOR_NEEDED resume carries x/y/z across the leg boundary, so a replay that
  // recomputes them from the basis starts somewhere else and follows a different trajectory. That
  // is exactly what made the one wrong-objective node we caught undiagnosable.
  std::vector<f_t> x_at_launch, y_at_launch, z_at_launch;

  // ---- out ----
  simplex::lp_solution_t<i_t, f_t> sol;
  // The FINAL basis, re-derived from the final `vstatus`. Returned rather than left to the consumer
  // because `get_basis_from_vstatus` appends without clearing -- the defect that made every node
  // after the first read the FIRST node's basis.
  std::vector<i_t> basic_list, nonbasic_list;
  i_t iters{0};
  simplex::dual_status_t status{simplex::dual_status_t::UNSET};
  // False when this seat never produced a device answer. The caller must solve that node on the CPU.
  // It is NOT a solver answer.
  bool solved{false};

  // WHY `solved` is false. Every path that drops a seat must name itself: a silent drop sends the
  // node to the CPU and still reads as a successful GPU run, which is how half a workload can
  // disappear without any line in the log.
  enum class drop_t : int {
    NONE = 0,          // solved
    FACTORIZE_FILL,    // refactor_basis was rank-deficient on the initial factorization
    SIZE_CONTRACT,     // a marshalled array had the wrong length; refused rather than read OOB
    FACTORIZE_RESUME,  // refactor_basis was rank-deficient rebuilding after a REFACTOR_NEEDED
    LIVELOCK,          // 8 consecutive resume legs took zero pivots
    LAUNCH_FAILED,     // the launch was rejected, or returned fewer statuses than seats
    DUAL_INFEASIBLE_RESUME,  // fresh pricing after a host refactor violated dual feasibility
    // The solve ran out of wall clock inside refactor_basis (TIME_LIMIT_RETURN). NOT a failure: the
    // CPU driver maps the same return to dual_status_t::TIME_LIMIT. Kept as its own reason because
    // collapsing it into FACTORIZE_* once cost an afternoon -- every seat still in flight when the
    // run hits its deadline reports it at once, which reads exactly like a numerical cliff.
    DEADLINE,
    CONCURRENT_HALT,   // CONCURRENT_HALT_RETURN: another solver signalled a halt
    FACTOR_CAP_FRESH,  // the fresh LU does not fit the seat's factor slot (node goes to the CPU)
    FACTOR_CAP_RESUME  // the refactor-resume LU does not fit the seat's factor slot
  };
  drop_t drop{drop_t::NONE};

  // lp_solution_t has no default constructor, so the dimensions are required here.
  gpu_batch_node_t(i_t m, i_t n) : sol(m, n) {}
};

// Seat configuration from CUOPT_BNB_GPU_BATCH. 0 disables the batch path, -1 selects automatic
// sizing, and a positive value is an explicit seat count. Automatic sizing is used only by the
// nondeterministic coordinator. The established deterministic coordinator treats a positive value
// as seats per worker; the nondeterministic coordinator treats it as the total target per GPU and
// quotient/remainder splits that target across the workers assigned to the device.
int gpu_batch_seats();
// Production seat target for one device: twice the resident capacity of the actual kernel, capped
// at half of currently free VRAM using a coarse per-seat footprint. A failed/zero query disables
// that device without affecting the others.
int gpu_batch_auto_seats(int device, int num_rows, int num_cols, std::size_t num_nonzeros);
// Per-node simplex iteration budget for a seated node LP. Exceeding it returns ITERATION_LIMIT;
// The persistent pool marks that B&B node CPU-only and returns it to the owning worker's queue.
// CUOPT_BNB_GPU_ITER_LIMIT=<n>, default 5000.
int gpu_batch_iter_limit();
// Host-side B&B GPU-DS telemetry (CUOPT_BNB_GPU_LOG=1). Off by default and never adds device work.
bool gpu_pool_logging();
// #4 -- pool index -> CUDA device, computed once before any worker starts. Entry is -1 when no
// device is usable. `num_gpus` is the solver parameter (see gpu_usable_devices). See the rationale
// in gpu_node_pool.cu.
std::vector<int> gpu_pool_device_map(int num_workers, int num_gpus);
// #3 -- engage the pool only once the worker's queue holds seats * factor nodes, so a fill does not
// empty it. CUOPT_BNB_GPU_HEAP_FACTOR, default 1 (was 5; sweep 2026-09-27); 0 disables the gate.
int gpu_pool_heap_factor();
// CUOPT_BNB_REFAC_FREQ: diagnostic override of refactor_frequency (0 = off).
int gpu_pool_refac_freq();
// #2 -- CUOPT_BNB_GPU_ASYNC: worker loop drives the pool between rounds. Default off.
bool gpu_pool_async();
bool gpu_pool_should_engage(size_t queue_size, int seats);
std::vector<std::pair<int, int>> gpu_pool_device_load(const std::vector<int>& map);

// Overlap CPU node solves with an in-flight batch round (CUOPT_BNB_GPU_ASYNC, default on).
bool gpu_pool_overlap();

// Max in-seat PLUNGE steps per seat (CUOPT_BNB_GPU_PLUNGE, default 8). 0 disables it.
// "Plunge", not "dive": diving_worker_t is a different mechanism (a primal heuristic on a detached
// throwaway tree) that this pool never touches.
int gpu_pool_plunge_depth();

// Directory for replayable dumps of nodes whose GPU and CPU answers disagree, or nullptr when
// disabled (CUOPT_BNB_DUMP_BAD_NODES=<dir>). Off by default: the images are multi-megabyte.
const char* dump_bad_nodes_dir();

// Correctness gate (CUOPT_BNB_GPU_POOL_RECHECK), default OFF -- see the definition.
bool gpu_pool_recheck();

// CUOPT_BNB_GPU_LOG_WARMUP (seconds): telemetry counts only steps after it.
double gpu_pool_log_warmup_s();

// Env switches, default on ("0" disables).
bool gpu_pool_device_strengthen();

bool gpu_pool_inplace();
bool gpu_pool_defer_refill();
bool gpu_pool_light_settings();
bool gpu_pool_batch_heap();
bool gpu_pool_device_objective();
bool gpu_pool_path_bounds();
bool gpu_pool_bounds_view();
bool gpu_pool_packed_fresh();
bool gpu_pool_device_scores();
// CUOPT_BNB_GPU_SLIM_D2H: y / z / dy_se / basis lists fetched per seat only when needed.
bool gpu_pool_slim_d2h();
bool gpu_pool_batch_fetch();
int gpu_pool_lean_begin();   // CUOPT_BNB_GPU_LEAN_BEGIN=k: first launch with k real seats (default 16, 0 = off)
int gpu_pool_pipeline();         // CUOPT_BNB_GPU_PIPELINE=K: K pools per worker (default 1)
bool gpu_pool_pipeline_split();  // CUOPT_BNB_GPU_PIPELINE_SPLIT: the K pools share the worker's seats
bool gpu_pool_lazy_install();
bool gpu_pool_batch_stats();  // CUOPT_BNB_GPU_BATCH_STATS (0 disables): pool step flushes shared counters once per generation
int gpu_pool_dive();  // CUOPT_BNB_GPU_DIVE: solve rounds per launch (device dives, default 3), 1 = off  // CUOPT_BNB_GPU_LAZY_INSTALL: device-scored seats skip the bound copy
bool gpu_pool_retain();      // CUOPT_BNB_GPU_RETAIN (0 disables): a drained pool keeps its handle (seats parked)
bool gpu_pool_park_short();  // CUOPT_BNB_GPU_PARK_SHORT: park seats the heap cannot refill
double gpu_pool_park_min();  // CUOPT_BNB_GPU_PARK_MIN: retire a short pool below this seated fraction (0.5)  // CUOPT_BNB_GPU_BATCH_FETCH: one fetch per generation, before the seat loop

// Do a GPU answer and a CPU answer for the SAME node reach a different branch-and-bound DECISION?
//
// This is deliberately not a status comparison. Degenerate LPs have alternate optima, so the two
// drivers may legitimately land on different vertices, and iteration counts differ by construction.
// What may never differ is "is this subtree prunable" and, when both solve, the optimal value the
// bound and the branching are read from.
//
// The subtle case is CUTOFF beside OPTIMAL. CUTOFF is an EARLY EXIT, not a verdict about the
// subtree: the driver stops the moment its tracked objective passes `settings.cut_off`. So it is a
// defect only when the side that solved to optimality would NOT itself have been cut off -- i.e. a
// subtree still capable of improving the incumbent was pruned. Comparing status NAMES instead
// reports every benign early exit as a divergence.
//
// `cpu_decided` is an out-parameter: ITERATION_LIMIT / TIME_LIMIT / NUMERICAL mean the CPU ran out
// of budget and returned obj=nan, so there is no CPU decision to compare against and the caller must
// not count the difference.
template <typename i_t, typename f_t>
bool gpu_decision_differs(simplex::dual_status_t gpu_status,
                          simplex::dual_status_t cpu_status,
                          f_t gpu_objective,
                          f_t cpu_objective,
                          f_t cut_off,
                          bool& cpu_decided,
                          bool& objective_differs);

// One plunging worker's pool. The worker STEALS n nodes out of its own queue, solves them in one
// launch, and then processes all n before returning to the loop -- so the pool owns each node for
// exactly one cycle and is empty at every barrier.
// Forward declaration so the pool can park a live run between worker-loop iterations. The full
// declaration and the begin/ready/advance/end API are further down this header.
template <typename i_t, typename f_t>
struct gpu_batch_run_t;

// One seat's host-side context for the current round. Hoisted out of gpu_pool_cycle so it can live
// on the pool across worker-loop iterations when the cycle runs asynchronously (#2).
template <typename i_t, typename f_t>
struct gpu_seat_ctx_t {
  mip_node_t<i_t, f_t>* node{nullptr};
  std::vector<f_t> parent_lower, parent_upper;   // empty => reconstructed from the root
  int plunge_depth{0};
  bool seated{false};   // a batch entry exists for this node this round
};

// Per-seat HOST state, owned by the POOL so it lives for the whole run rather than for one launch.
//
// The host copy carries the current ordered basis lists and the last host-refactorized base LU.
// In the persistent pool the accumulated MPF stack and its routing statistics live only in the DEVICE
// slot; CHILD_CONTINUE intentionally does not reconstruct this host `ft`. REFACTOR_RESUME replaces
// the device base LU from here and clears that stack, while FRESH_NODE overwrites all logical state.
template <typename i_t, typename f_t>
struct batch_seat_state_t {
  simplex::basis_update_mpf_t<i_t, f_t> ft;
  std::vector<i_t> basic_list, nonbasic_list, superbasic_list;
  std::vector<f_t> xB_workspace;
  i_t total_iters{0};
  i_t leg_iters{0};
  int zero_pivot_legs{0};
  bool active{false};          // still to be launched (or relaunched)
  bool refactor_resume{false}; // the host rebuilt x for this seat; the device must not redo it
  bool terminal_certificate_retry{false}; // reason-2 relaunch: retry OPTIMAL proof only
  // Why this seat's last leg came back REFACTOR_NEEDED: 0 = periodic schedule, 1 = the rank-1 update
  // was REJECTED as unsafe, 2 = terminal certificate retry. Written by the batch copy-back through
  // gpu_batch_item::refactor_reason.
  // In the persistent pool every reason is independently re-priced after the host refactor as a feasibility
  // certificate; the maintained y/z are carried only when that certificate passes.
  int refactor_reason{0};
  // Why refactorize() last returned false. Set by refactorize, consumed at the drop site.
  typename gpu_batch_node_t<i_t, f_t>::drop_t fail{gpu_batch_node_t<i_t, f_t>::drop_t::NONE};

  batch_seat_state_t(i_t m, i_t rf) : ft(m, rf), basic_list(m), xB_workspace(m, 0.0) {}
};

// Sub-phases of the step split's harvest / process / refill (CUOPT_BNB_GPU_LOG=1).
enum gpu_pool_sub_t : int {
  SUB_HARVEST_D2H,       // D2H enqueue + stream sync
  SUB_HARVEST_SCATTER,   // statuses (and full results without views) into the batch items
  SUB_HARVEST_UNPACK,    // per-seat loop of harvest_persistent (device bounds into the nodes)
  SUB_PROCESS_SNAPSHOT,  // resident bound copies + result view
  SUB_PROCESS_INSTALL,   // terminal/iteration records + nondeterministic_gpu_pool_install_result
  SUB_PROCESS_TREE,      // node counters + update_tree_impl
  SUB_PROCESS_CHILD,     // plunge-child preparation (bounds, symmetry, vstatus, deltas)
  SUB_PROCESS_PUSH,      // heap pushes
  SUB_PROCESS_FLUSH,     // flush_deferred (tree lock)
  SUB_REFILL_POP,        // pop_gpu_eligible + cut-off fathoming
  SUB_REFILL_VSTATUS,    // decompress_vstatus + edge norms + ensure_orbital_fixing
  SUB_REFILL_BOUNDS,     // set_lp_variable_bounds
  SUB_REFILL_SYMMETRY,   // orbital-fixing reset + symmetry reductions
  SUB_REFILL_BUILD,      // batch node construction, copies, seat install
  SUB_REFILL_LB,         // nondeterministic_gpu_pool_update_lower_bound
  SUB_REFILL_CUTOFF,     // fathoming popped nodes above the incumbent
  SUB_REFILL_FAIL,       // a pop that found no GPU-eligible node (the refill fails)
  SUB_REFILL_FLUSH,      // deferred tree updates applied after the refill (DEFER_REFILL)
  // Parts of process_tree (update_tree_impl), process_child and refill_build.
  SUB_TREE_FRAC,         // fractional variables + objective
  SUB_TREE_PSEUDO,       // pseudocost update
  SUB_TREE_SELECT,       // branching variable selection
  SUB_TREE_ESTIMATE,     // objective estimate
  SUB_TREE_BRANCH,       // child node creation (packed vstatus)
  SUB_CHILD_SETUP,       // child batch node, settings, bound copies, branched bound
  SUB_CHILD_SYMMETRY,    // leaf bounds + symmetry reductions
  SUB_CHILD_VSTATUS,     // decompress + same-basis check
  SUB_CHILD_DELTA,       // bound deltas, seat install
  SUB_BUILD_ALLOC,       // batch node construction
  SUB_BUILD_SETTINGS,    // per-seat solver settings
  SUB_BUILD_COPY,        // bounds / vstatus / edge norms into the batch node
  SUB_STAGE_FRESH,       // stage: fresh seats (basis lists, slabs, device-LU target)
  SUB_FRESH_PREP,        // stage fresh: seat reset + solution resize
  SUB_FRESH_BASIS,       // stage fresh: basis lists from vstatus
  SUB_FRESH_ITEM,        // stage fresh: batch item
  SUB_BSTAGE_ARGS,       // stage fresh, batch_stage: argument setup
  SUB_BSTAGE_BOUNDS,     // stage fresh, batch_stage: bounds packed
  SUB_BSTAGE_SLABS,      // stage fresh, batch_stage: slab copies
  SUB_BSTAGE_SLICES,     // stage fresh, batch_stage: slices packed
  SUB_STAGE_REFACTOR,    // stage: refactor-resume seats (host LU + factor staging)
  SUB_STAGE_CHILD,       // stage: child seats (bound deltas)
  SUB_RELAUNCH_PACK,     // relaunch: packed upload flush
  SUB_RELAUNCH_LU,       // relaunch: dense LU enqueue
  SUB_RELAUNCH_ARGS,     // relaunch: delta / argument uploads, memsets
  SUB_RELAUNCH_LAUNCH,   // relaunch: occupancy query + kernel launch
  SUB_NOTICE_PREKERNEL,  // part of notice: device work queued before the solve kernel
  SUB_NOTICE_LU,         // part of SUB_NOTICE_PREKERNEL: the dense LU
  SUB_NOTICE_H2D,        // part of SUB_NOTICE_PREKERNEL: packed upload copy
  SUB_NOTICE_SCATTER,    // part of SUB_NOTICE_PREKERNEL: batched scatter of the packed pieces
  SUB_NOTICE_DELTAS,     // part of SUB_NOTICE_PREKERNEL: child bound-delta copy
  SUB_FLUSH_WAIT,        // part of SUB_PROCESS_FLUSH: waiting for the tree lock
  SUB_FLUSH_HOLD,        // part of SUB_PROCESS_FLUSH: applying the updates under the lock
  SUB_D2H_ENQUEUE,       // part of SUB_HARVEST_D2H: enqueueing the result copies
  SUB_D2H_DEVICE,        // part of SUB_HARVEST_D2H: device span of the result copies
  SUB_TREE_SELECT_SB,    // branching selection of nodes with unreliable candidates (strong branching)
  SUB_COUNT
};
inline constexpr const char* gpu_pool_sub_names[SUB_COUNT] = {
  "harvest_d2h", "harvest_scatter", "harvest_unpack", "process_snapshot", "process_install",
  "process_tree", "process_child", "process_push", "process_flush", "refill_pop",
  "refill_vstatus", "refill_bounds", "refill_symmetry", "refill_build", "refill_lb",
  "refill_cutoff", "refill_fail", "refill_flush", "tree_frac", "tree_pseudo", "tree_select", "tree_estimate",
  "tree_branch", "child_setup", "child_symmetry", "child_vstatus", "child_delta", "build_alloc",
  "build_settings", "build_copy", "stage_fresh", "fresh_prep", "fresh_basis", "fresh_item", "bstage_args", "bstage_bounds", "bstage_slabs", "bstage_slices",
  "stage_refactor", "stage_child", "relaunch_pack",
  "relaunch_lu", "relaunch_args", "relaunch_launch", "notice_prekernel", "notice_lu", "notice_h2d", "notice_scatter", "notice_deltas", "flush_wait",
  "flush_hold", "d2h_enqueue", "d2h_device", "tree_select_sb"};

// Telemetry lap timer: lap(k) adds the time since the previous lap to sub[k]; no-op if sub is null.
struct gpu_pool_lap_t {
  double* sub{nullptr};
  double t{0.0};
  void start();
  void lap(int k);
};
// Set by the pool step around staging (null otherwise); the stage functions lap into it.
extern thread_local gpu_pool_lap_t* tl_stage_laps;

template <typename i_t, typename f_t>
struct gpu_batch_pool_t {
  // #4 -- the device this pool lives on, from gpu_pool_device_map(). MUST be set on the calling
  // thread (cudaSetDevice) before the stream is created and before every launch: the CUDA current
  // device is per-THREAD state, and rmm::cuda_stream binds to whatever device is current when it is
  // constructed. A pool whose stream belongs to device A but whose launches run on device B is not
  // a performance problem, it is an illegal-access crash.
  int device{-1};

  rmm::cuda_stream stream;   // one per worker: pools must not serialise

  // A / rhs / objective / Arow master. NON-OWNING: one cache per GPU, shared by every pool on that
  // device. All the pools on a device are solving nodes of the SAME MIP, so the matrix is identical
  // for all of them -- a per-pool cache kept 7 identical copies of A resident on a 7-worker run.
  // Owned by gpu_ro_caches_ in branch_and_bound_t, created before any worker starts, so no locking
  // is needed: it is populated single-threaded and read-only thereafter.
  gds::gpu_ro_cache<i_t, f_t>* ro_cache{nullptr};

  // The nodes this pool OWNS for the current cycle, in the order they were taken. They are OUT of
  // the worker's queue, so nothing else can see them -- which is safe only because the cycle is
  // synchronous and empties `pending` before returning to the loop. See the invariant in
  // deterministic_sync_callback().
  // A vector, not a deque: every use is clear/empty/size/iterate/index, and `dequeue_gpu_nodes`
  // already returns a std::vector -- so this makes the per-cycle hand-off a move instead of an
  // element-by-element copy. Nothing anywhere push_front/pop_fronts it.
  std::vector<mip_node_t<i_t, f_t>*> pending;

  // ---- ASYNC CYCLE STATE (#2) ---------------------------------------------------------------
  // With CUOPT_BNB_GPU_ASYNC the cycle no longer runs to completion inside one call: the worker
  // launches a round, RETURNS to its loop to do other work, and comes back to harvest. Everything
  // the round needs between those calls lives here instead of on the stack.
  //
  // While `busy` is true this pool HOLDS nodes that are in no queue. deterministic_bfs_worker_t's
  // has_work_impl() therefore reports the pool, which is what keeps the worker out of the barrier
  // until the pool drains -- see the invariant in deterministic_sync_callback().
  std::atomic<bool> busy{false};
  std::atomic<int> seated{0};              // telemetry: seats holding a node (per generation)
  const char* drain_reason{"other"};       // telemetry: why the next finish drains

  // Nodes owned by a busy pool are absent from every CPU heap. Publish their inherited B&B bound
  // here so the opportunistic global-bound scan cannot run past hidden GPU work.
  std::atomic<f_t> lower_bound{std::numeric_limits<f_t>::infinity()};

  // EFFECTIVE seat count for THIS pool. Starts at the requested CUOPT_BNB_GPU_BATCH and is halved
  // once if the device refuses the batch for want of VRAM. `disabled` is terminal: this worker has
  // been told twice that it cannot have a pool, so it stops asking and runs purely on the CPU.
  // Per-pool rather than global: with several pools per GPU, one worker failing to fit must not
  // shrink or disable everyone else's.
  int seats{0};
  bool halved{false};
  bool disabled{false};

  // Start only after this worker has completed a representative window of node LPs on the CPU.
  // A mean/median hides a small but expensive long tail; nearest-rank p90 catches it while keeping
  // the update O(1). The O(64) nth_element runs only when considering a pool launch.
  static constexpr std::size_t iteration_window = 64;
  std::array<i_t, iteration_window> recent_iterations{};
  std::size_t recent_iteration_count{0};
  std::size_t recent_iteration_cursor{0};

  void record_completed_iterations(i_t iterations)
  {
    recent_iterations[recent_iteration_cursor] = std::max<i_t>(iterations, 0);
    recent_iteration_cursor = (recent_iteration_cursor + 1) % iteration_window;
    recent_iteration_count  = std::min(recent_iteration_count + 1, iteration_window);
  }

  bool iteration_gate_allows(i_t iteration_limit, i_t& p90) const
  {
    if (recent_iteration_count < iteration_window) { return false; }
    auto sample = recent_iterations;
    constexpr std::size_t rank = (9 * iteration_window + 9) / 10 - 1;
    std::nth_element(sample.begin(), sample.begin() + rank, sample.end());
    p90 = sample[rank];
    const int64_t threshold = (9LL * static_cast<int64_t>(iteration_limit)) / 10LL;
    return static_cast<int64_t>(p90) < threshold;
  }

  // Host-only telemetry. These counters are dormant unless CUOPT_BNB_GPU_LOG=1, except the
  // iteration history above, which is part of the launch policy rather than instrumentation.
  double available_since_s{0.0};
  double launch_started_at{0.0};
  double activation_wait_s{0.0}, activation_wait_max_s{0.0};
  double cpu_overlap_s{0.0}, host_service_s{0.0}, host_blocked_s{0.0};
  double upload_s{0.0}, kernel_s{0.0}, copyback_s{0.0}, host_factor_s{0.0};
  double completion_notice_s{0.0}, completion_notice_max_s{0.0};
  // Finished GPU work waiting for the owning worker (steady window): per launch, from the last queued
  // GPU op to step entry, split by the worker's activity (worker_activity_t) meanwhile.
  double wait_s{0.0}, wait_max_s{0.0};
  std::array<double, 4> wait_act_s{};
  uint64_t waits{0}, waits_early{0};
  double accounted_upload_s{0.0}, accounted_kernel_s{0.0};
  uint64_t starts{0}, launches{0}, drains{0}, activation_samples{0};
  uint64_t cpu_overlap_nodes{0}, cpu_only_nodes{0};
  uint64_t terminal_nodes{0}, optimal_nodes{0}, cutoff_nodes{0}, dual_unbounded_nodes{0};
  uint64_t numerical_nodes{0}, iteration_limit_nodes{0}, other_terminal_nodes{0};
  uint64_t terminal_pivots{0}, unhealthy_nodes{0}, unhealthy_pivots{0};
  uint64_t fill_errors{0}, harvest_errors{0}, relaunch_errors{0}, device_errors{0};
  uint64_t refill_shortfalls{0}, preparation_failures{0};
  uint64_t allocation_halvings{0}, allocation_disables{0};
  // Setup split (one sample per pool start): `fill` = host seat preparation in the start, `begin` =
  // gpu_persistent_batch_begin = host LU of every seat + marshal (alloc / H2D / memset) + launch.
  uint64_t setups{0}, setup_alloc_n{0}, setup_h2d_n{0}, setup_h2d_bytes{0}, setup_memset_n{0};
  double setup_fill_s{0.0}, setup_begin_s{0.0}, setup_host_lu_s{0.0};
  double setup_alloc_s{0.0}, setup_h2d_s{0.0}, setup_memset_s{0.0};
  double last_fill_s{0.0};
  double fill_split_s[4]{};  // telemetry: guard, pop, node construction, prepare_fresh
  // Steady-state service-step split, summed over the steps that harvested a launch: `guard` /
  // `restore` = saving / restoring the worker's CPU LP continuation around the step, `harvest` =
  // results D2H, `process` = per-seat result install + tree update + child placement, `refill` =
  // new nodes into free seats, `stage` = per-seat settings + host LU + H2D + memsets (+ growth
  // allocations), `relaunch` = argument upload + kernel launch.
  uint64_t steps{0}, step_h2d_n{0}, step_h2d_bytes{0}, step_memset_n{0}, step_alloc_n{0};
  double step_total_s{0.0}, step_guard_s{0.0}, step_restore_s{0.0}, step_harvest_s{0.0};
  double step_process_s{0.0}, step_refill_s{0.0}, step_stage_s{0.0}, step_stage_settings_s{0.0};
  double step_stage_lu_s{0.0}, step_stage_h2d_s{0.0}, step_stage_memset_s{0.0};
  double step_stage_alloc_s{0.0}, step_relaunch_s{0.0};
  // Kernel time and completion-notice latency of the same steps (the `time` line covers whole runs).
  double step_kernel_s{0.0}, step_notice_s{0.0};
  std::array<double, SUB_COUNT> step_sub_s{};
  bool warmup_done{false};
  void reset_step_split()
  {
    steps = step_h2d_n = step_h2d_bytes = step_memset_n = step_alloc_n = 0;
    step_total_s = step_guard_s = step_restore_s = step_harvest_s = 0.0;
    step_process_s = step_refill_s = step_stage_s = step_stage_settings_s = 0.0;
    step_stage_lu_s = step_stage_h2d_s = step_stage_memset_s = 0.0;
    step_stage_alloc_s = step_relaunch_s = step_drain_s = 0.0;
    step_kernel_s = step_notice_s = 0.0;
    step_sub_s.fill(0.0);
    wait_s = wait_max_s = 0.0;
    wait_act_s.fill(0.0);
    waits = waits_early = 0;
  }
  // Pool retirement (nondeterministic_gpu_pool_finish: destroy the run + return nodes): all of it,
  // and the part that happened inside a harvesting step (`drain` in the step split).
  double finish_s{0.0}, step_drain_s{0.0};

  // One entry per seat, sized once. Reset for a fresh node, PRESERVED for a plunge child.
  std::vector<batch_seat_state_t<i_t, f_t>> seat_state;

  std::vector<mip_node_t<i_t, f_t>*> prefetched;
  size_t prefetch_next{0};
  std::vector<mip_node_t<i_t, f_t>*> pending_push;
  std::vector<f_t> pc_sum_down, pc_sum_up;
  std::vector<i_t> pc_num_down, pc_num_up;
  simplex::simplex_solver_settings_t<i_t, f_t> gen_settings;

  // The LIVE run, parked here between worker-loop iterations. This is what lets the loop ask "is the
  // GPU done?" and, when it is not, go do something else -- rather than waiting inside the step.
  // Non-null means a batch is on the device right now. `busy` is broader: it also covers the gap
  // between a run finishing and its results being processed.
  gpu_batch_run_t<i_t, f_t>* run{nullptr};
  bool restart{false};           // CUOPT_BNB_GPU_RETAIN: next step refills a retained handle
  uint64_t restarts{0};          // telemetry: starts that reused a retained handle
  std::vector<gpu_batch_node_t<i_t, f_t>> batch;
  std::vector<gpu_seat_ctx_t<i_t, f_t>> ctx;

  // Results, by node. An entry is present only for a node the device actually answered; a node that
  // failed preparation is absent and takes its normal CPU path.
  std::unordered_map<const mip_node_t<i_t, f_t>*, gpu_batch_node_t<i_t, f_t>> done;
};

// Solve every node in `nodes` as ONE batch on `stream`, resuming each seat independently until it
// reaches a terminal status. Blocks until all seats are done.
//
// `lp` supplies A / Arow / rhs / objective and is shared by every seat; its own lower/upper are
// IGNORED in favour of each node's. The caller must force a host basis rebuild afterwards for the
// same reason the serial path must: the device pivoted its own copy of the factorization.
template <typename i_t, typename f_t>
void gpu_solve_nodes_batch(const simplex::lp_problem_t<i_t, f_t>& lp,
                           std::vector<gpu_batch_node_t<i_t, f_t>>& nodes,
                           rmm::cuda_stream_view stream,
                           gds::gpu_ro_cache<i_t, f_t>* ro_cache = nullptr);

// ---------------------------------------------------------------------------
// ASYNCHRONOUS form of the same solve.
//
// Solving a pool-full of nodes is a sequence of rounds -- launch the active seats, read their
// statuses, host-refactorize the ones that returned REFACTOR_NEEDED, relaunch those. A run holds the
// state between rounds, so a worker can launch, go back to solving CPU nodes, and return when the
// device is ready:
//
//     run = gpu_batch_begin(...)                 // launches round 1, does not block
//     ... worker does CPU work ...
//     while (!gpu_batch_run_ready(run)) { ... }  // non-blocking poll
//     if (!gpu_batch_run_advance(run)) { gpu_batch_run_end(run); }   // advance launches the next round
//
// `nodes` is BORROWED for the lifetime of the run and must not be resized or destroyed; the pool
// owns it. `gpu_batch_run_end` publishes the results into `nodes` and frees the run -- it must be
// called exactly once, including on the early-out path where `begin` returns nullptr (it does not).
// ---------------------------------------------------------------------------
template <typename i_t, typename f_t>
struct gpu_batch_run_t;
template <typename i_t, typename f_t>
using gpu_batch_run = gpu_batch_run_t<i_t, f_t>;

struct gpu_batch_run_metrics {
  int seats{0};
  int rounds{0};
  std::size_t terminal_noop_seats{0};
  int periodic_refactors{0}, rejected_refactors{0}, certificate_refactors{0};
  double upload_s{0.0}, kernel_s{0.0}, copyback_s{0.0}, host_factor_s{0.0}, wall_s{0.0};
  // Split of the latest harvest (D2H + sync, scatter, per-seat unpack).
  double last_harvest_d2h_s{0.0}, last_harvest_scatter_s{0.0}, last_harvest_unpack_s{0.0};
  double last_harvest_d2h_enqueue_s{0.0}, last_harvest_d2h_device_s{0.0};
  double last_prekernel_s{0.0};   // device time of the relaunch's uploads + dense LU
  double last_prekernel_lu_s{0.0};
  double last_prekernel_h2d_s{0.0}, last_prekernel_scatter_s{0.0}, last_prekernel_deltas_s{0.0};
  double last_done_after_pre_s{0.0};  // device: relaunch start to the last queued GPU op
  // Initial launch (gpu_persistent_batch_begin) split; zero unless CUOPT_BNB_GPU_LOG=1.
  double setup_wall_s{0.0}, setup_host_lu_s{0.0};
  double setup_alloc_s{0.0}, setup_h2d_s{0.0}, setup_memset_s{0.0};
  std::size_t setup_alloc_n{0}, setup_h2d_n{0}, setup_h2d_bytes{0}, setup_memset_n{0};
};

template <typename i_t, typename f_t>
// `seat_state` is the POOL's persistent per-seat state. Passing the pool's vector is what lets a
// plunge child continue on its parent's factorization; a caller with no pool passes a local one.
gpu_batch_run<i_t, f_t>* gpu_batch_begin(const simplex::lp_problem_t<i_t, f_t>& lp,
                                         std::vector<gpu_batch_node_t<i_t, f_t>>& nodes,
                                         rmm::cuda_stream_view stream,
                                         gds::gpu_ro_cache<i_t, f_t>* ro_cache,
                                         std::vector<batch_seat_state_t<i_t, f_t>>* seat_state);

template <typename i_t, typename f_t>
gpu_batch_run<i_t, f_t>* gpu_persistent_batch_begin(
  const simplex::lp_problem_t<i_t, f_t>& lp,
  std::vector<gpu_batch_node_t<i_t, f_t>>& nodes,
  rmm::cuda_stream_view stream,
  gds::gpu_ro_cache<i_t, f_t>* ro_cache,
  std::vector<batch_seat_state_t<i_t, f_t>>* seat_state,
  bool result_views = false,
  const std::vector<simplex::variable_type_t>* var_types = nullptr);

template <typename i_t, typename f_t>
bool gpu_persistent_batch_harvest(gpu_batch_run<i_t, f_t>* run);

template <typename i_t, typename f_t>
gds::gpu_batch_result_view<i_t, f_t> gpu_persistent_batch_result(
  gpu_batch_run<i_t, f_t>* run, i_t seat);

// Completes a slim-harvested seat's view (y, z, dy_se, basis lists).
template <typename i_t, typename f_t>
void gpu_persistent_batch_fetch_seat(gpu_batch_run<i_t, f_t>* run, i_t seat);
// One fetch (one stream sync) for all listed seats.
template <typename i_t, typename f_t>
void gpu_persistent_batch_fetch_seats(gpu_batch_run<i_t, f_t>* run, const std::vector<int>& seats);
template <typename i_t, typename f_t>
bool gpu_persistent_batch_park(gpu_batch_run<i_t, f_t>* run, i_t seat);
template <typename i_t, typename f_t>
void gpu_persistent_batch_unstage(gpu_batch_run<i_t, f_t>* run, i_t lo, i_t hi);
template <typename i_t, typename f_t>
bool gpu_persistent_batch_launched(gpu_batch_run<i_t, f_t>* run);

template <typename i_t, typename f_t>
bool gpu_persistent_batch_stage_fresh(gpu_batch_run<i_t, f_t>* run, i_t seat, std::string& error);

template <typename i_t, typename f_t>
bool gpu_persistent_batch_stage_refactor(gpu_batch_run<i_t, f_t>* run, i_t seat, std::string& error);

template <typename i_t, typename f_t>
bool gpu_persistent_batch_stage_child(
  gpu_batch_run<i_t, f_t>* run,
  i_t seat,
  const std::vector<gds::gpu_bound_delta<i_t, f_t>>& deltas,
  std::string& error);

template <typename i_t, typename f_t>
bool gpu_persistent_batch_relaunch(gpu_batch_run<i_t, f_t>* run, std::string& error);

template <typename i_t, typename f_t>
bool gpu_batch_run_ready(gpu_batch_run<i_t, f_t>* run);

template <typename i_t, typename f_t>
bool gpu_batch_run_path_bounds(gpu_batch_run<i_t, f_t>* run);
template <typename i_t, typename f_t>
void gpu_batch_run_set_bounds_mirror(gpu_batch_run<i_t, f_t>* run, bool mirror);

template <typename i_t, typename f_t>
void gpu_batch_run_set_dive(gpu_batch_run<i_t, f_t>* run, int rounds, const f_t* root_x);
template <typename i_t, typename f_t>
gds::gpu_dive_view<i_t, f_t> gpu_persistent_batch_dive(gpu_batch_run<i_t, f_t>* run, i_t seat);

template <typename i_t, typename f_t>
void gpu_batch_run_set_pseudocosts(gpu_batch_run<i_t, f_t>* run,
                                   const f_t* sum_down,
                                   const f_t* sum_up,
                                   const i_t* num_down,
                                   const i_t* num_up,
                                   f_t avg_down,
                                   f_t avg_up,
                                   i_t reliable_threshold,
                                   f_t integer_tol);

template <typename i_t, typename f_t>
gpu_batch_run_metrics gpu_batch_run_get_metrics(const gpu_batch_run<i_t, f_t>* run);

template <typename i_t, typename f_t>
bool gpu_batch_run_advance(gpu_batch_run<i_t, f_t>* run);
// The two halves of advance, for a caller that owns a loop: harvest the landed launch (results +
// host refactor), then relaunch once the host has processed and refilled the seats. One relaunch
// per worker-loop pass == one kernel launch per pass.
template <typename i_t, typename f_t>
bool gpu_batch_run_harvest(gpu_batch_run<i_t, f_t>* run);
template <typename i_t, typename f_t>
bool gpu_batch_run_relaunch(gpu_batch_run<i_t, f_t>* run);
// Broadcast the newest incumbent to every seat. MUST be called before every launch, whatever the
// seats' statuses: the device reads cut_off once per launch and never again.
template <typename i_t, typename f_t>
void gpu_batch_run_set_cut_off(gpu_batch_run<i_t, f_t>* run, f_t cut_off);

template <typename i_t, typename f_t>
void gpu_batch_run_end(gpu_batch_run<i_t, f_t>* run);

}  // namespace cuopt::mathematical_optimization::mip
