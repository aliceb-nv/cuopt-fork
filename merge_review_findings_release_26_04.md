# Merge Review Findings vs `release/26.04`

Scope:
- Current merge state reviewed statically against `release/26.04`
- Excluding `cpp/src/branch_and_bound/pseudo_costs.cpp`
- Notes are incremental and may grow as the review continues

## Resolved High Confidence Findings

1. `cpp/src/mip_heuristics/solver_context.cuh`
   - Restored the `release/26.04` scaling ownership model for MIP.
   - Removed the extra scaling constructor parameter from `mip_solver_context_t`; callers now match the context definition again.

2. `cpp/src/mip_heuristics/solution_callbacks.cuh`
   - Removed the incorrect `pdlp_initial_scaling_strategy_t` dependency from MIP callback plumbing.
   - `solution_publication_t` and `solution_injection_t` no longer try to own or apply scaling; they now operate on the release-side MIP flow and dispatch both `GET_SOLUTION` and `GET_SOLUTION_EXT`.

3. `cpp/src/mip_heuristics/solver.cu`
   - Removed `context.scaling` uses from incumbent publication and injection paths.
   - Removed the stale `bb_callback_adapter_t::settings_` reference member, which was left uninitialized by the merge.

4. `cpp/src/mip_heuristics/diversity/population.cu`
   - Removed `context.scaling` from the callback/publication and injection calls so the file matches the release-side scaling model.

5. `cpp/src/mip_heuristics/solve.cu`
   - Deleted the stale local `invoke_solution_callbacks(...)` helper instead of extending it.
   - Rewired the early incumbent publication paths to the determinism-side callback dispatch (`GET_SOLUTION_EXT` compatible, with origin and work timestamp metadata).
   - Removed the stray `scaling.scale_problem()` / `scale_primal(...)` block from `run_mip()`, which had no scaling object in scope.
   - Restored the `try` / `catch` structure in `run_mip()` after the merge splice dropped the opening `try`.
   - Updated the early-heuristic gates to the bitset model by allowing them only when `determinism_mode == CUOPT_DETERMINISM_NONE`.

6. `cpp/src/mip_heuristics/problem/problem.cuh`, `cpp/src/mip_heuristics/problem/problem.cu`, `cpp/src/mip_heuristics/problem/presolve_data.cuh`
   - Repaired the half-merged `post_process_assignment(...)` overloads.
   - The handle-override wrappers now forward the override stream correctly, and the stream-based implementation no longer references the nonexistent `handle_override` variable.

7. `cpp/src/mip_heuristics/diversity/diversity_manager.cu`
   - Restored the missing `tolerance_divisor` local used to derive PDLP relative tolerances in the non-deterministic root LP path.

8. `cpp/src/mip_heuristics/feasibility_jump/feasibility_jump.cuh`, `cpp/src/mip_heuristics/feasibility_jump/early_gpufj.cu`
   - Fixed the early GPU FJ merge splice where `early_gpufj_t` reached into the now-private `fj_t::improvement_callback`.
   - Added a proper setter and updated the caller to use it.

9. `cpp/src/mip_heuristics/solve.cu`
   - Removed merge-leftover unused locals (`running_mip`, `hyper_params`) that were tripping `-Werror`.

## Lower Confidence Risks

1. `cpp/src/mip_heuristics/diversity/population.cu`
   - In deterministic B&B mode, `run_solution_callbacks()` updates `best_feasible_objective` immediately after queueing a heuristic solution to B&B, before B&B validates or repairs it.
   - If the queued solution is later rejected after crushing/validation, later heuristic candidates can be suppressed against an incumbent objective that never actually became valid.
