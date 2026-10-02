# Remove the host-built inverse library from the replay path: inverses only come from the device build.
import re
R='/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/'
def edit(p, fn):
    s=open(R+p).read(); s2=fn(s); assert s2!=s, p; open(R+p,'w').write(s2)
def cut(s, a, b_end, keep_end=False):
    i=s.index(a); j=s.index(b_end, i)
    return s[:i]+(b_end if keep_end else '')+s[j+len(b_end):] if not keep_end else s[:i]+s[j:]
# test_netlib: drop the library block (and the EXP inverse diagnostics inside it)
def tn(s):
    a="  // CUOPT_BNB_GPU_FISSION: explicit inverses of every stored node, resident on the device for the\n"
    b="  std::vector<ds::variable_type_t> var_types(h.n_var_types);"
    i=s.index(a); j=s.index(b,i); s=s[:i]+s[j:]
    old="              run, s, stored[seat_node[s]], prepared[seat_node[s]], err, &finv[seat_node[s]])) {"
    assert s.count(old)==1
    return s.replace(old,"              run, s, stored[seat_node[s]], prepared[seat_node[s]], err)) {")
edit('benchmarks/linear_programming/cuopt/test_netlib.cu', tn)
def pool(s):
    s=s.replace("""                                   std::string& error,
                                   const gds::gpu_fission_inv* finv = nullptr)
  {
    auto& node = nodes[s];""","""                                   std::string& error)
  {
    auto& node = nodes[s];""")
    s=s.replace("    it.fission_inv     = finv;\n","")
    s=s.replace("""                                           std::string& error,
                                           const gds::gpu_fission_inv* finv)
{
  return run != nullptr && run->stage_persistent_fresh_from(seat, node, prepared, error, finv);""","""                                           std::string& error)
{
  return run != nullptr && run->stage_persistent_fresh_from(seat, node, prepared, error);""")
    return s.replace("const batch_seat_state_t<int, double>&, std::string&, const gds::gpu_fission_inv*);","const batch_seat_state_t<int, double>&, std::string&);")
edit('cpp/src/branch_and_bound/gpu_node_pool.cu', pool)
edit('cpp/src/branch_and_bound/gpu_node_pool.hpp', lambda s: s.replace("""                                           std::string& error,
                                           const gds::gpu_fission_inv* finv = nullptr);""","""                                           std::string& error);"""))
edit('cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cuh', lambda s: s.replace("""
  // Fissioned path: this seat's inverses of the base factors (required for FRESH / REFACTOR_RESUME
  // staging when CUOPT_BNB_GPU_FISSION is on; a child continuation keeps the seat's previous ones).
  const gpu_fission_inv* fission_inv{nullptr};
""",""))
def g(s):
    i=s.index("  if (gpu_fission() && gpu_fission_inv_library() && command != gpu_batch_seat_command::CHILD_CONTINUE) {")
    j=s.index("    H->h_finv[seat] = *item.fission_inv;\n  }\n", i)+len("    H->h_finv[seat] = *item.fission_inv;\n  }\n")
    s=s[:i]+s[j:]
    s=s.replace("""      const bool lib = gpu_fission_inv_library();
      if (lib && H->h_finv.size() != N) {
        error = "fissioned pivot loop: seats without explicit inverses";
        return false;
      }
""","")
    s=s.replace("      if (!lib && !H->d_finv_mem) {","      if (!H->d_finv_mem) {")
    s=s.replace("""      if (!lib) {
        // Every seat's inverses from the factors just staged: per-generation work.""","""      {
        // Every seat's inverses from the factors just staged: per-generation work.""")
    return s.replace("        if (!lib && *H->h_finv_err != 0) {","        if (*H->h_finv_err != 0) {")
edit('cpp/src/dual_simplex/gpu_phase2/gpu_phase2.cu', g)
def fc(s):
    i=s.index("// CUOPT_EXP_INV=library: the harness's start-up inverse library (not per-generation work).")
    j=s.index("\n}\n", i)+3
    return s[:i]+s[j:]
edit('cpp/src/dual_simplex/gpu_phase2/fission.cuh', fc)
print('ok')
