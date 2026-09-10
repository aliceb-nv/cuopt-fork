/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "internal.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::fj_cpu_deleter_t::operator()(fj_cpu_climber_t<i_t, f_t>* ptr) const
{
  delete ptr;
}

template <typename i_t, typename f_t>
std::shared_ptr<fj_cpu_shared_incumbent_t<i_t, f_t>> make_fj_cpu_shared_incumbent()
{
  return std::make_shared<fj_cpu_shared_incumbent_t<i_t, f_t>>();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::create_worker(
  const lp_problem_t<i_t, f_t>& problem,
  const std::vector<simplex::variable_type_t>& variable_types,
  i_t n_structural,
  const std::vector<f_t>& seed_assignment,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  std::string log_prefix,
  int64_t seed,
  int lane)
{
  auto new_climber = init_fj_cpu_from_host_lp(
    problem, variable_types, n_structural, seed_assignment, settings, preemption_flag, seed);
  fj_cpu.reset(new_climber.release());
  fj_cpu->log_prefix           = std::move(log_prefix);
  fj_cpu->improvement_callback = improvement_callback;
  fj_cpu->shared_incumbent     = shared_incumbent;
  fj_cpu->halted               = false;
  preemption_flag              = false;
  is_initialized               = true;
  if (lane >= 0) { apply_lane_diversification<i_t, f_t>(*fj_cpu, lane, fj_cpu->settings.seed); }
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::run_async(f_t time_limit, double work_unit_limit)
{
  if (!is_initialized) return;

  auto& fj_ptr = fj_cpu;
#pragma omp task shared(fj_cpu, is_initialized, fj_ptr) firstprivate(time_limit, work_unit_limit) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) default(none) depend(out : fj_ptr)
  {
    if (is_initialized) { cpufj_solve(fj_cpu.get(), time_limit, work_unit_limit); }
  }
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::run_sync(f_t time_limit, double work_unit_limit)
{
  if (!is_initialized) return;
  cpufj_solve(fj_cpu.get(), time_limit, work_unit_limit);
  is_initialized = false;
  fj_cpu.reset();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::stop()
{
  if (!is_initialized) return;

  preemption_flag = true;

  auto& fj_ptr = fj_cpu;
#pragma omp taskwait depend(in : fj_ptr)
  is_initialized = false;
  fj_cpu.reset();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::send_stop_signal()
{
  preemption_flag = true;
}

template <typename i_t, typename f_t>
void apply_lane_diversification(fj_cpu_climber_t<i_t, f_t>& c, int lane, int64_t base_seed)
{
  const bool extreme_hub = c.problem->max_var_degree > 512 &&
                           c.problem->max_var_degree > 64.0 * c.problem->avg_var_degree;
  const bool mixed_integer = c.n_integer_vars > 0 && c.n_integer_vars < c.problem->n_variables;
  // A low-rank, all-general-integer equality system needs coordinated residual transfers: changing
  // one dense column almost inevitably opens another equality. Keep this structural signal narrow
  // so ordinary integer and assignment models retain their existing lane personas.
  const bool low_rank_integer_equalities = c.n_binary_vars == 0 &&
    c.n_integer_vars == c.problem->n_variables && c.problem->equality_fraction > 0.99 &&
    int64_t{32} * c.problem->n_constraints < c.problem->n_variables;

  // Setup personas. LP work is lane-local and therefore does not delay the portfolio launch.
  c.use_lp_seed = lane == 4 || lane == 10 || lane == 14 || lane == 11;
  c.lp_seed_feasibility_objective = lane == 4 || lane == 10 || lane == 11;
  // Lane 6 has no sole crossing or best-objective ownership and retains its distinct SAPS descent.
  // Do not spend its window on a second deep LP pump; lane 4 remains the dedicated deep-LP persona.
  c.use_deep_lp_pump = lane == 4;
  c.use_integer_bit_encoding = lane != 0 && lane != 7;
  // Lane 11 already solves a feasibility-objective LP and has no sole-crosser or best-incumbent
  // contribution. Do not follow it with the separate 0.8-second dive; that duplicate LP work
  // contends with every other climber during the highest-value first-incumbent window.
  c.use_lp_feasibility_dive = lane == 4 || lane == 13;
  c.use_lp_polish = lane == 9 || lane == 14 || lane == 11 || lane == 3 || lane == 8 || lane == 13 ||
                    lane == 5 || lane == 10 || lane == 12;
  c.use_precedence_seed = lane % 8 == 0;
  c.use_affine_equality_seed = lane == 2 || lane == 4 || lane == 11;
  c.use_bound_prop = lane % 2 == 0 && !c.low_latency;
  c.use_weight_donation = lane % 8 == 5 || lane % 8 == 6;
  c.degree_balance_mtm = lane == 5 || lane == 6 || (lane == 9 && extreme_hub);
  // Lanes 9 and 12 carry no seed, exchange or repair persona of their own (just LP polish and
  // restart tuning), so they are the cheapest incremental place to extend batching beyond the
  // original three policy classes: the self-disabling probe still bounds the cost wherever it does
  // not pay off, and this adds only two more lanes rather than the whole portfolio at once. Lanes
  // carrying their own coordinated repair or exchange operator (3, 5, 11, 13 compound repair; 1, 7,
  // 15 cardinality/factor-scope exchange) tried and measured worse, so stay off batching here.
  c.use_move_batching = c.n_colors > 0 &&
    (lane % 8 == 0 || lane % 8 == 2 || lane % 8 == 6 || lane == 9 || lane == 12);
  // Dense all-integer equality masters have no working scalar lane in the baseline. Repurpose only
  // lane 13 on that matrix-certified class: obtain another independently cost-perturbed LP vertex
  // instead of running the shallow dive, whose repeated root solve supplies no distinct basin.
  // Other models retain lane 13 exactly, including its sole-crosser role on general mixed MIPs.
  if (lane == 13 && low_rank_integer_equalities) {
    c.use_lp_seed                    = true;
    c.lp_seed_feasibility_objective = true;
    c.use_deep_lp_pump              = false;
    c.use_lp_feasibility_dive       = false;
  }

  // Create specialized lanes for difficult instances with more aggressive restart strategies.
  // (Lane 9's restart_window/degrade_ratio are set once, below, by the "also a sole crosser"
  // block -- an earlier, narrower lane-9 override here used to write the same two fields first
  // and was unconditionally overwritten by that later block before ever being read. Removed.)
  // Lane 5 gets ultra-aggressive restarts for symmetry breaking
  if (lane == 5) {
    c.infeasible_restart_window = 80;   // Very aggressive restart
    c.infeasible_restart_degrade_ratio = 1.03;
    // Enable more exploration
    c.settings.parameters.weight_smoothing_probability = 0.005;
  }
  
  // Add specialized "escape" lane that aggressively explores the search space
  if (lane == 12) {
    c.infeasible_restart_window = 60;   // Ultra-aggressive restart
    c.infeasible_restart_degrade_ratio = 1.02;
    c.settings.parameters.weight_smoothing_probability = 0.01;
    // Increase tabu tenure variation to encourage different search paths
    c.settings.parameters.tabu_tenure_min = 1;
    c.settings.parameters.tabu_tenure_max = 30;
  }
  
  // Enhance lane 15 (structure-aware lane) for problematic structured instances
  if (lane == 15) {
    // Slightly more persistent search on structured problems
    c.infeasible_restart_window = 180;  // Increase from typical 250-500
    c.infeasible_restart_degrade_ratio = 1.04;  // Slightly more tolerant
  }
  
  // Enhance lane 1 (already a sole crosser) for better performance on marginal instances
  if (lane == 1) {
    // Slightly more exploration while maintaining effectiveness
    c.settings.parameters.weight_smoothing_probability = 0.001;  // Increased from 0.0
    c.infeasible_restart_window = 150;  // Increased from 130
  }
  
  // Enhance lane 9 (also a sole crosser) for better performance on extreme hub instances
  if (lane == 9) {
    // Slightly adjust for better balance between aggression and persistence
    c.infeasible_restart_window = 110;  // Increased from 100
    c.infeasible_restart_degrade_ratio = 1.04;  // Reduced from 1.05 for more tolerance
  }

  bool cumulative = false;
  {
    phase_timer_t timer(c.t_seed);
    switch (lane % 8) {
      case 1: apply_structural_completion_seed<i_t, f_t>(c); break;
      case 2: apply_aggressive_constraint_seed<i_t, f_t>(c); break;
      case 3: apply_greedy_covering_seed<i_t, f_t>(c); break;
      // Not where the LP pump runs: it rewrites the assignment wholesale and would discard this.
      case 4:
        if (!c.use_lp_seed) {
          apply_ambiguous_lock_seed<i_t, f_t>(c);
          apply_greedy_covering_seed<i_t, f_t>(c);
        }
        break;
      case 5: apply_bipartite_matching_seed<i_t, f_t>(c); break;
      case 6: break;  // preserve the shared exact-cardinality anchor
      case 0:
        if (lane == 8) {
          apply_exact_k_seed<i_t, f_t>(c);
          apply_greedy_covering_seed<i_t, f_t>(c);
        }
        break;
      default: break;
    }
    if (lane == 10) {
      apply_greedy_covering_seed<i_t, f_t>(c);
      cumulative = apply_cumulative_chain_seed<i_t, f_t>(
        c, (uint32_t)(base_seed + 2246822519u));
    }
    if (lane == 12 || lane == 15) apply_structural_completion_seed<i_t, f_t>(c);
    // Keep this as a single, structurally gated portfolio persona. Other lanes retain the
    // low-degree exact-k anchor, which is preferable when one-hot member order is not ordinal.
    if (lane == 15) {
      apply_exact_one_repair_seed<i_t, f_t>(c);
      apply_ordinal_midpoint_seed<i_t, f_t>(c);
    }
    // Lane 11 targets hard instances with LP seeding and additional structure recognition
    if (lane == 11) {
      apply_structural_completion_seed<i_t, f_t>(c);
      apply_greedy_covering_seed<i_t, f_t>(c);
    }
    if (lane == 2) {
      cumulative = apply_cumulative_chain_seed<i_t, f_t>(
        c, (uint32_t)(base_seed + 3141592653u));
    }
  }

  std::mt19937 rng(base_seed + 7919u * lane);
  c.mtm_viol_samples = std::uniform_int_distribution<i_t>(10, 80)(rng);
  c.mtm_sat_samples  = std::uniform_int_distribution<i_t>(5, 50)(rng);
  c.nnz_samples      = std::uniform_int_distribution<i_t>(1000, 20000)(rng);
  c.perturb_interval = std::uniform_int_distribution<i_t>(10, 2000)(rng);
  c.objective_corner_jump_gate = c.perturb_interval * 4;

  static constexpr double smoothing[8] = {0.0003, 0.0, 0.001, 0.003, 0.0001, 0.0006, 0.002, 0.0003};
  static constexpr int tabu_min[8] = {3, 1, 5, 3, 2, 6, 4, 3};
  static constexpr int tabu_max[8] = {13, 7, 21, 13, 10, 25, 17, 13};
  static constexpr i_t restart[8] = {250, 130, 400, 250, 130, 500, 350, 250};
  static constexpr f_t degrade[8] = {1.12, 1.04, 1.20, 1.12, 1.06, 1.25, 1.15, 1.12};
  const int policy = lane % 8;
  c.settings.parameters.weight_smoothing_probability = smoothing[policy];
  c.settings.parameters.tabu_tenure_min = tabu_min[policy];
  c.settings.parameters.tabu_tenure_max = tabu_max[policy];
  c.infeasible_restart_window = restart[policy];
  c.infeasible_restart_degrade_ratio = degrade[policy];
  c.use_cardinality_exchange = lane == 1 || lane == 7 || lane == 15 || lane == 11;
  c.use_compound_repair = lane == 3 || lane == 5 || lane == 11;

  static constexpr i_t kick_interval[8] = {40, 80, 150, 25, 60, 120, 200, 90};
  c.infeasible_kick_interval = lane >= 8 ? kick_interval[policy]
                                         : ((policy == 1 || policy == 5) ? kick_interval[policy] / 2 : 0);
  c.infeasible_kick_vars = 3 + lane % 3;
  c.use_directed_infeasible_kick = lane == 7 || (lane == 9 && extreme_hub);
  if (lane == 2) { c.use_directed_infeasible_kick = true; c.infeasible_kick_interval = 35; }
  if (lane == 8) { c.use_directed_infeasible_kick = true; c.infeasible_kick_interval = 25; }
  if (lane == 12) { c.use_directed_infeasible_kick = true; c.infeasible_kick_interval = 20; }
  if (lane == 11) { c.use_directed_infeasible_kick = true; c.infeasible_kick_interval = 40; }
  if (lane == 0) {
    c.use_directed_infeasible_kick = true;
    c.infeasible_kick_interval = 15;
    c.infeasible_kick_vars = 5;
  }

  // Plain descents retain distinct breadths; lane 6 keeps its mixed-integer LP pump.
  // Lane 4 is instead an independent LP-feasibility dive and retains its normal sampled breadth.
  if (lane == 6 || lane == 7) {
    if (lane != 6) c.use_lp_seed = c.use_deep_lp_pump = false;
    c.use_precedence_seed = false;
    c.use_weight_donation = c.degree_balance_mtm = c.use_directed_infeasible_kick = false;
    c.infeasible_kick_interval = 0;
    if (lane != 6) c.use_affine_equality_seed = false;
  }
  if (lane == 6 && mixed_integer && c.use_deep_lp_pump) {
    c.mtm_viol_samples = 384; c.mtm_sat_samples = 96; c.nnz_samples = 200000;
  }
  if (lane == 7 || lane == 8) {
    c.mtm_viol_samples = 192; c.mtm_sat_samples = 64; c.nnz_samples = 100000;
  }
  if (lane == 0) {
    c.mtm_viol_samples = std::uniform_int_distribution<i_t>(40, 100)(rng);
    c.mtm_sat_samples = std::uniform_int_distribution<i_t>(20, 60)(rng);
    c.nnz_samples = std::uniform_int_distribution<i_t>(10000, 30000)(rng);
  }
  if (lane == 12) {
    c.mtm_viol_samples = std::uniform_int_distribution<i_t>(50, 120)(rng);
    c.mtm_sat_samples = std::uniform_int_distribution<i_t>(25, 70)(rng);
  }
  if (lane == 11) {
    c.mtm_viol_samples = std::uniform_int_distribution<i_t>(30, 100)(rng);
    c.mtm_sat_samples = std::uniform_int_distribution<i_t>(15, 50)(rng);
  }
  if (lane == 9 && extreme_hub) {
    c.mtm_viol_samples = 8; c.mtm_sat_samples = 3; c.nnz_samples = 2000;
  }

  // Two dynamic-weight trajectories; lane 10 yields to the cumulative or LP seed when available.
  if (lane == 6) {
    c.use_multiplicative_weights = true;
    c.saps_multiplier = (f_t)1.3;
    c.settings.parameters.weight_smoothing_probability = 0.01;
    c.settings.seed += 104729;
  }
  if (lane == 10) {
    c.use_affine_equality_seed = cumulative;
    c.use_bound_prop = false;
    c.use_lp_seed = !cumulative;
    c.lp_seed_feasibility_objective = !cumulative;
    c.settings.seed += 224737;
  }

  // Lane 5 is the extreme-hub deep LP/continuous-repair attempt. Lane 15 has no sole crossing or
  // objective ownership and already carries the structural exact-one/factor-scope seed, so preserve
  // that seed and start its scalar/compound search immediately instead of overwriting it with a
  // second deep LP trajectory.
  if (extreme_hub && lane == 5) {
    c.use_lp_seed = c.use_deep_lp_pump = true;
    c.lp_seed_feasibility_objective = false;
  }

  // Seek feasibility without objective interference; restore lane-specific pressure after crossing.
  // The floor only takes effect once a lane is already feasible (h_objective_weight starts at 0 and
  // is raised to seed_objective_weight only at the incumbent gate), so raising it cannot delay or
  // lose a first incumbent -- it only strengthens the objective descent that the primal integral
  // scores directly once across. The previous floors {1,4,32,1} left half the lanes (lane%4 in
  // {0,3}) pulling almost nothing on the objective, so many instances crossed and then held a poor
  // gap for the rest of the window. Spread the post-crossing pressure across the whole portfolio so
  // several lanes drive the objective hard while a couple stay near feasibility-only for diversity.
  const f_t obj_weight_floor[4] = {2, 8, 32, 1};
  i_t continuous_objective_vars = 0;
  for (i_t var : c.problem->h_objective_vars)
    continuous_objective_vars += !is_integer_var<i_t, f_t>(c, var);
  const int64_t objective_var_count = static_cast<int64_t>(c.problem->h_objective_vars.size());
  const bool continuous_objective_model =
    objective_var_count > 0 &&
    int64_t{10} * continuous_objective_vars >= int64_t{9} * objective_var_count &&
    int64_t{10} * objective_var_count >= int64_t{c.problem->n_variables};
  c.h_objective_weight = !continuous_objective_model ? f_t{0}
                         : lane == 3                 ? f_t{4}
                         : lane == 9                 ? f_t{8}
                         : lane == 15                ? f_t{16}
                                                     : f_t{0};
  // Lane 1 is the portfolio's strongest cardinality-exchange incumbent owner. Once it has crossed,
  // give those equality-preserving swaps the same objective pressure as the existing objective
  // specialists; the other modulo-1 lanes retain the moderate trajectory for diversity.
  // The LP-free structural lane and lanes 4/5 own difficult incumbents, so give them stronger
  // post-cross pressure than their default floors. Lane 3 retains its low-pressure
  // trajectory: stronger pull consistently moved its sole-crosser incumbent to a worse objective.
  // These weights remain inactive before the first incumbent.
  c.seed_objective_weight = lane == 1 ? f_t{32}
                            : lane == 4 ? f_t{8}
                            : lane == 5 ? f_t{16}
                            : lane == 15 ? f_t{8}
                                         : obj_weight_floor[lane % 4];
}

template <typename i_t, typename f_t>
void complete_climber_portfolio(
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> first_climber,
  const std::vector<int64_t>& lane_seed,
  std::vector<std::atomic<bool>>& preemption_flags,
  std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>>& climbers,
  int64_t base_seed,
  bool low_latency)
{
  const int n_climbers = static_cast<int>(climbers.size());
  cuopt_assert(n_climbers > 0, "a CPUFJ portfolio needs at least one climber");
  cuopt_assert(preemption_flags.size() == climbers.size(), "preemption flag count mismatch");
  cuopt_assert(lane_seed.size() == climbers.size(), "lane seed count mismatch");

  // Lane 0 is a genuine dependency: its completed host state is the template for every other lane.
  {
    climbers[0] = std::move(first_climber);
    cuopt_assert(climbers[0] != nullptr, "missing first CPUFJ climber");
    climbers[0]->low_latency = low_latency;
    // Runs before the clones are taken, so every lane starts from the repaired anchor.
    apply_exact_k_seed<i_t, f_t>(*climbers[0]);
    repair_difficult_anchor<i_t, f_t>(*climbers[0]);
    apply_lane_diversification<i_t, f_t>(*climbers[0], 0, base_seed);
  }

  // The remaining lanes depend only on lane 0's finished, read-only template, and the O(nnz) clone
  // and seed passes are otherwise paid serially on one thread while the other pinned CPUs idle.
#ifdef _OPENMP
#pragma omp parallel for num_threads(std::max(1, n_climbers - 1)) schedule(static)
#endif
  for (int k = 1; k < n_climbers; ++k) {
    fj_settings_t settings;
    settings.seed = (int)lane_seed[k];
    climbers[k]   = init_fj_cpu_clone(*climbers[0], preemption_flags[k], settings);
    climbers[k]->low_latency = low_latency;
    apply_lane_diversification<i_t, f_t>(*climbers[k], k, base_seed);
  }

  auto shared = std::make_shared<fj_cpu_shared_incumbent_t<i_t, f_t>>();
  for (int k = 0; k < n_climbers; ++k)
    climbers[k]->shared_incumbent = shared;
}


}  // namespace cuopt::mathematical_optimization::mip
