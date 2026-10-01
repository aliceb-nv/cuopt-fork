# Fission fresh seats: no device transposes / level schedules; x_B from the explicit inverses
# (fission_xb), then objective and infeasibility list.
import sys
root=sys.argv[1]
def patch(p, pairs):
    s=open(p).read()
    for a,b in pairs:
        assert s.count(a)==1,(p,s.count(a),a[:80]); s=s.replace(a,b)
    open(p,'w').write(s)
g=root+'/cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu'
patch(g,[
("""  i_t f_state{0};  // fissioned pivot loop (fission.cuh): next phase""",
"""  i_t f_state{0};  // fissioned pivot loop (fission.cuh): next phase
  i_t f_lean{0};   // fission fresh seat: x_B, objective and infeasibilities come from fission_xb"""),
("""    if (!refactor_resume) {
      set_primal_variables_on_bounds(lp, settings, z, vstatus, x);

      // Both of these solve through the factors, so they follow the kernel's geometry — the
      // monolith has always run them on the warp via Direct.
      compute_primal_variables(ft,""","""    if (!refactor_resume) {
      set_primal_variables_on_bounds(lp, settings, z, vstatus, x);
    }
    if (!refactor_resume && f_lean == 0) {
      // Both of these solve through the factors, so they follow the kernel's geometry — the
      // monolith has always run them on the warp via Direct.
      compute_primal_variables(ft,"""),
("""    // WARP_PINF: scan the m basics one lane per k and compact the infeasible indices in order.
    // squared_infeas is read only through that freshly written index list.
    compute_initial_primal_infeasibilities(
      lp, settings, ft, basic_list, x, squared_infeas, infeas_idx, primal_inf);
""","""    // WARP_PINF: scan the m basics one lane per k and compact the infeasible indices in order.
    // squared_infeas is read only through that freshly written index list.
    if (f_lean == 0) {
      compute_initial_primal_infeasibilities(
        lp, settings, ft, basic_list, x, squared_infeas, infeas_idx, primal_inf);
    }
"""),
("""    obj  = refactor_resume ? resume_obj : compute_objective(lp, x);""",
"""    obj  = refactor_resume ? resume_obj : (f_lean != 0 ? f_t{0.0} : compute_objective(lp, x));"""),
("""  if constexpr (PersistentSeats) {
    if (!child_continue && a.transpose_on_device != 0) {""","""  // Fission fresh seat: no reader of the device transposes or the level schedules.
  const bool lean = FissionSetup && !child_continue && a.refactor_resume == 0 && a.device_lu == 0;
  if constexpr (PersistentSeats) {
    if (!lean && !child_continue && a.transpose_on_device != 0) {"""),
("""  ft.path(reuse_levels ? P_LEVEL_REUSE : P_LEVEL_BUILD);
  if (!reuse_levels && c_warp_levels != 0) {""","""  ft.path(reuse_levels ? P_LEVEL_REUSE : P_LEVEL_BUILD);
  if (!lean) {
  if (!reuse_levels && c_warp_levels != 0) {"""),
("""    ft.warp_cfg_ = &cfg_shared;
  });
  if (threadIdx.x == 0) {
    const long long pc3 = clock64();""","""    ft.warp_cfg_ = &cfg_shared;
  });
  }
  lane0_serial([&] { st.f_lean = lean ? 1 : 0; });
  if (threadIdx.x == 0) {
    const long long pc3 = clock64();"""),
("""      fission_setup_kernel<<<Ni, 32, 0, cs>>>(H->d_args, Ni);""","""      fission_setup_kernel<<<Ni, 32, 0, cs>>>(H->d_args, Ni);
      {
        const size_t smem_x = 2 * static_cast<size_t>(H->rows) * sizeof(f_t);
        RAFT_CUDA_TRY(cudaFuncSetAttribute(fission_xb, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(smem_x)));
        fission_xb<<<Ni, kXbThreads, smem_x, cs>>>(H->d_args, finv, Ni);
      }"""),
])
f=root+'/cpp/src/dual_simplex/gpu_phase2/fission.cuh'
patch(f,[
("""  const long long c6 = clock64();
  if (tr && threadIdx.x == 0) {
    r.live = 1;
    for (int q = 0; q < 6; ++q) { r.sub[q] = static_cast<int>(a.slot->pre[q]); }
  }""","""  const long long c6 = clock64();
  if (tr && threadIdx.x == 0) {
    r.live = 1;
    for (int q = 0; q < 6; ++q) { r.sub[q] = static_cast<int>(a.slot->pre[q]); }
  }
  if (a.slot->f_lean != 0) {  // fission_xb finishes this seat
    emit();
    return;
  }"""),
("""// Incremental infeasibility update after a pivot""","""// Fresh fission seat: x_B = U0^{-1} L0^{-1} (b - A_N x_N) through the explicit inverses (row-wise,
// one owner per output, |x_j| < 10 tight_tol skipped as in compute_primal_variables), then the
// objective c'x and the infeasibility list. Shared: two m-vectors.
constexpr int kXbThreads = 512;
__global__ void __launch_bounds__(kXbThreads) fission_xb(kernel_args* args, const gpu_fission_inv* inv, i_t n_problems)
{
  extern __shared__ f_t fsh[];
  if (blockIdx.x >= n_problems) { return; }
  kernel_args& a = args[blockIdx.x];
  if (a.done != 0) { return; }
  auto& st = *a.slot;
  if (st.f_lean == 0) { return; }
  __shared__ typename cub::BlockReduce<f_t, kXbThreads>::TempStorage otmp;
  const i_t m = a.m, n = a.n;
  const int tid = threadIdx.x;
  f_t* r = fsh;
  f_t* z = fsh + m;
  f_t* x = a.x;
  const f_t tt = a.tight_tol * 10;
  for (i_t i = tid; i < m; i += kXbThreads) {
    f_t v = a.rhs[i];
    for (i_t p = a.Arow_row_start[i]; p < a.Arow_row_start[i + 1]; ++p) {
      const i_t j = a.Arow_j[p];
      if (a.vstatus[j] == variable_status_t::BASIC) { continue; }
      const f_t xj = x[j];
      if (fabs(xj) < tt) { continue; }
      v -= a.Arow_x[p] * xj;
    }
    r[i] = v;
  }
  __syncthreads();
  const auto& I = inv[blockIdx.x];
  for (i_t i = tid; i < m; i += kXbThreads) {
    f_t v = 0.0;
    for (i_t p = I.Lr_p[i]; p < I.Lr_p[i + 1]; ++p) { v += I.Lr_x[p] * r[I.Lr_i[p]]; }
    z[i] = v;
  }
  __syncthreads();
  for (i_t k = tid; k < m; k += kXbThreads) {
    f_t v = 0.0;
    for (i_t p = I.Ur_p[k]; p < I.Ur_p[k + 1]; ++p) { v += I.Ur_x[p] * z[I.Ur_i[p]]; }
    x[a.basic_list[k]] = v;
  }
  __syncthreads();
  f_t part = 0.0;
  for (i_t j = tid; j < n; j += kXbThreads) { part += a.objective[j] * x[j]; }
  const f_t obj = cub::BlockReduce<f_t, kXbThreads>(otmp).Sum(part);
  if (tid == 0) { st.obj = obj; }
  if (tid < 32) {
    gpu_lp lp;
    gpu_settings settings;
    gpu_solution sol;
    fission_views(a, lp, settings, sol);
    f_t* sq = st.squared_infeas.data();
    for (i_t j = tid; j < n; j += 32) { sq[j] = 0.0; }
    __syncwarp(0xffffffffu);
    compute_initial_primal_infeasibilities(lp, settings, st.ft, a.basic_list, x, st.squared_infeas,
                                           st.infeas_idx, st.primal_inf);
  }
}

// Incremental infeasibility update after a pivot"""),
])
print('ok')
