# Setup: block-wide bounds reset + strengthening in a pre-kernel; body skips them (and the
# device transposes / level schedules, which no fission kernel reads) under FissionSetup.
import sys
root=sys.argv[1]
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
g=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
patch(g,[
("""  bool child_continue = false;
  if constexpr (PersistentSeats) {
    child_continue = a.command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE);
    if (a.bounds_from_root != 0) {""","""  bool child_continue = false;
  if constexpr (PersistentSeats) {
    child_continue = a.command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE);
    if (!FissionSetup && a.bounds_from_root != 0) {  // fission: fission_presetup did it"""),
("""  if constexpr (PersistentSeats) {
    if (child_continue) {
    // The persistent seat""","""  if constexpr (PersistentSeats) {
    if (!FissionSetup && child_continue) {
    // The persistent seat"""),
("""  if (a.strengthen != 0 && (child_continue || a.refactor_resume == 0)) {
    __syncwarp(0xffffffffu);
    warp_strengthen_bounds(a, child_continue);""","""  if (!FissionSetup && a.strengthen != 0 && (child_continue || a.refactor_resume == 0)) {
    __syncwarp(0xffffffffu);
    warp_strengthen_bounds(a, child_continue);"""),
])
f=root+'/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
patch(f,[
("""__global__ void __launch_bounds__(32) fission_setup_kernel(kernel_args* args, i_t n_problems)""",
"""// warp_strengthen_bounds over a whole block (same passes and arithmetic, per-column owners).
__device__ void block_strengthen_bounds(kernel_args& a, bool from_deltas)
{
  const i_t m = a.m, n = a.n;
  const i_t tid = static_cast<i_t>(threadIdx.x), nt = static_cast<i_t>(blockDim.x);
  const f_t inf  = cuda::std::numeric_limits<f_t>::infinity();
  const f_t ptol = a.primal_tol;
  f_t* dmin          = reinterpret_cast<f_t*>(a.bs_scratch);
  f_t* dmax          = dmin + m;
  unsigned char* rc  = reinterpret_cast<unsigned char*>(dmax + m);
  unsigned char* rcn = rc + m;
  unsigned char* vc  = rcn + m;
  f_t* lower         = a.lower;
  f_t* upper         = a.upper;
  for (i_t i = tid; i < m; i += nt) { rc[i] = from_deltas ? 0 : 1, rcn[i] = 0; }
  for (i_t j = tid; j < n; j += nt) { vc[j] = 0; }
  __syncthreads();
  if (from_deltas) {
    for (i_t q = 0; q < a.bound_delta_count; ++q) {
      const i_t j = a.bound_deltas[q].column;
      for (i_t p = a.A_col_start[j] + tid; p < a.A_col_start[j + 1]; p += nt) { rc[a.A_i[p]] = 1; }
    }
    __syncthreads();
  }
  for (int iter = 0; iter < 10; ++iter) {
    int infeasible = 0;
    for (i_t i = tid; i < m; i += nt) {
      if (!rc[i]) { continue; }
      f_t min_a = 0.0, max_a = 0.0;
      for (i_t p = a.Arow_row_start[i]; p < a.Arow_row_start[i + 1]; ++p) {
        const i_t j    = a.Arow_j[p];
        const f_t a_ij = a.Arow_x[p];
        vc[j]          = 1;
        if (a_ij > 0) {
          min_a += a_ij * lower[j];
          max_a += a_ij * upper[j];
        } else if (a_ij < 0) {
          min_a += a_ij * upper[j];
          max_a += a_ij * lower[j];
        }
        if (upper[j] == inf && a_ij > 0) { max_a = inf; }
        if (lower[j] == -inf && a_ij < 0) { max_a = inf; }
        if (lower[j] == -inf && a_ij > 0) { min_a = -inf; }
        if (upper[j] == inf && a_ij < 0) { min_a = -inf; }
      }
      const f_t b = a.rhs[i];
      if (min_a > b + ptol || max_a < b - ptol) { infeasible = 1; }
      dmin[i] = b - min_a;
      dmax[i] = b - max_a;
    }
    if (__syncthreads_or(infeasible)) { return; }
    int changed = 0;
    for (i_t k = tid; k < n; k += nt) {
      if (!vc[k]) { continue; }
      const f_t old_lb = lower[k];
      const f_t old_ub = upper[k];
      f_t new_lb       = old_lb;
      f_t new_ub       = old_ub;
      for (i_t p = a.A_col_start[k]; p < a.A_col_start[k + 1]; ++p) {
        const i_t i = a.A_i[p];
        if (!rc[i]) { continue; }
        const f_t a_ik = a.A_x[p];
        const f_t dmn  = dmin[i] + ((a_ik < 0) ? a_ik * old_ub : a_ik * old_lb);
        const f_t dmx  = dmax[i] + ((a_ik > 0) ? a_ik * old_ub : a_ik * old_lb);
        const f_t lb_c = (a_ik < 0.) ? dmn / a_ik : dmx / a_ik;
        const f_t ub_c = (a_ik < 0.) ? dmx / a_ik : dmn / a_ik;
        new_lb         = (new_lb < lb_c) ? lb_c : new_lb;
        new_ub         = (ub_c < new_ub) ? ub_c : new_ub;
      }
      if (a.var_int[k]) {
        new_lb = ceil(new_lb - a.integer_tol);
        new_ub = floor(new_ub + a.integer_tol);
      }
      const bool updated = fabs(new_lb - old_lb) > 1e3 * ptol || fabs(new_ub - old_ub) > 1e3 * ptol;
      new_lb = (new_lb < old_lb) ? old_lb : new_lb;
      new_ub = (old_ub < new_ub) ? old_ub : new_ub;
      if (new_lb > new_ub + ptol) {
        infeasible = 1;
        continue;
      }
      if (new_lb != old_lb || new_ub != old_ub) {
        for (i_t p = a.A_col_start[k]; p < a.A_col_start[k + 1]; ++p) { rcn[a.A_i[p]] = 1; }
      }
      lower[k] = (new_lb < new_ub) ? new_lb : new_ub;
      upper[k] = (new_lb < new_ub) ? new_ub : new_lb;
      changed |= updated ? 1 : 0;
    }
    if (__syncthreads_or(infeasible)) { return; }
    if (!__syncthreads_or(changed)) { return; }
    for (i_t i = tid; i < m; i += nt) { rc[i] = rcn[i], rcn[i] = 0; }
    for (i_t j = tid; j < n; j += nt) { vc[j] = 0; }
    __syncthreads();
  }
}

// Fission setup, part 1 (block): the kernel body's bounds reset, child deltas and strengthening.
constexpr int kPresetupThreads = 512;
__global__ void __launch_bounds__(kPresetupThreads) fission_presetup(kernel_args* args, i_t n_problems)
{
  const i_t idx = static_cast<i_t>(blockIdx.x);
  if (idx >= n_problems) { return; }
  kernel_args& a = args[idx];
  if (a.done != 0) { return; }
  if (a.device_lu != 0 && a.lu_info[0] != 0) { return; }
  const bool child_continue = a.command == static_cast<i_t>(gpu_batch_seat_command::CHILD_CONTINUE);
  if (a.bounds_from_root != 0) {
    for (i_t j = threadIdx.x; j < a.n; j += blockDim.x) { a.lower[j] = a.root_lower[j], a.upper[j] = a.root_upper[j]; }
    __syncthreads();
    for (i_t q = threadIdx.x; q < a.bound_delta_count; q += blockDim.x) {
      const auto& d = a.bound_deltas[q];
      a.lower[d.column] = d.lower;
      a.upper[d.column] = d.upper;
    }
    __syncthreads();
  }
  if (child_continue) {
    if (threadIdx.x == 0) {
      for (i_t q = 0; q < a.bound_delta_count; ++q) {
        const auto& d = a.bound_deltas[q];
        if ((d.mask & 1u) != 0u) { a.lower[d.column] = d.lower; }
        if ((d.mask & 2u) != 0u) { a.upper[d.column] = d.upper; }
      }
    }
    __syncthreads();
  }
  if (a.strengthen != 0 && (child_continue || a.refactor_resume == 0)) {
    block_strengthen_bounds(a, child_continue);
  }
}

__global__ void __launch_bounds__(32) fission_setup_kernel(kernel_args* args, i_t n_problems)"""),
])
patch(g,[("""      fission_setup_kernel<<<Ni, 32, 0, cs>>>(H->d_args, Ni);""",
"""      fission_presetup<<<Ni, kPresetupThreads, 0, cs>>>(H->d_args, Ni);
      fission_setup_kernel<<<Ni, 32, 0, cs>>>(H->d_args, Ni);""")])
print('ok')
