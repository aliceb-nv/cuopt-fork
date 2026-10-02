# ncu_src.py <report> <kernel> [ntop]: kernel summary, stall samples by fission.cuh function and top source lines.
import csv, collections, re, subprocess, sys

rep, kern = sys.argv[1], sys.argv[2]
ntop = int(sys.argv[3]) if len(sys.argv) > 3 else 20
FISSION = '/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp/cpp/src/dual_simplex/gpu_phase2/fission.cuh'


def run(*a):
    return subprocess.run(['ncu', '--import', rep, *a], capture_output=True, text=True).stdout


# Function owning each fission.cuh line: the last __device__/__global__ definition above it.
owner = {}
cur = '?'
for i, l in enumerate(open(FISSION), 1):
    m = re.search(r'(?:__device__|__global__)[^(]*?\b(\w+)\s*\(', l)
    if m and not l.rstrip().endswith(';'):
        cur = m.group(1)
    owner[i] = cur

raw = list(csv.reader(run('--page', 'raw', '--csv', '--kernel-name', kern).splitlines()))
h = raw[0]
d = dict(zip(h, raw[2]))
f = lambda k: float(d[k].replace(',', '')) if k in d and d[k] not in ('', 'n/a') else float('nan')
print(f"{kern}: {f('gpu__time_duration.sum'):.1f} us | DRAM {f('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% "
      f"L2 {f('lts__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% L1 {f('l1tex__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% "
      f"SM {f('sm__throughput.avg.pct_of_peak_sustained_elapsed'):.1f}% | warps active {f('sm__warps_active.avg.pct_of_peak_sustained_active'):.1f}% "
      f"IPC/SM {f('sm__inst_executed.avg.per_cycle_elapsed'):.3f} | L2 hit {f('lts__t_sector_hit_rate.pct'):.1f}%")
st = {k.replace('smsp__average_warps_issue_stalled_', '').replace('_per_issue_active.ratio', ''): f(k)
      for k in h if k.startswith('smsp__average_warps_issue_stalled_') and k.endswith('_per_issue_active.ratio')}
print('stalls (warps per issue):', ', '.join(f'{k} {v:.1f}' for k, v in sorted(st.items(), key=lambda x: -x[1])[:6]))

rows = list(csv.reader(run('--page', 'source', '--kernel-name', kern, '--print-source', 'cuda,sass', '--csv').splitlines()))
fp, hdr, lines = None, None, []
for r in rows:
    if len(r) == 2 and r[0] in ('File Path', 'File Name'):
        fp = r[1]
        continue
    if r and r[0] == 'Line No':
        hdr = r
        continue
    if hdr and r and r[0].isdigit() and len(r) > 5 and r[5].isdigit():
        reasons = {hdr[i].replace('stall_', ''): int(r[i] or 0) for i in range(len(hdr))
                   if hdr[i].startswith('stall_') and '(Not Issued)' not in hdr[i] and (r[i] or '0').isdigit()}
        lines.append((int(r[5]), fp, int(r[0]), r[1].strip(), reasons))
tot = sum(x[0] for x in lines) or 1
g = collections.Counter()
for s, p, l, _, _ in lines:
    g[owner.get(l, '?') if p == FISSION else p.split('/')[-1]] += s
print(f'stall samples {tot}; by function / file:')
for k, v in g.most_common(12):
    print(f'  {100 * v / tot:5.1f}%  {k}')
print('top lines:')
for s, p, l, src, rs in sorted(lines, key=lambda x: -x[0])[:ntop]:
    top = ', '.join(f'{k} {100 * c / tot:.1f}' for k, c in sorted(rs.items(), key=lambda x: -x[1])[:3] if c)
    print(f'  {100 * s / tot:5.1f}%  {p.split("/")[-1]}:{l} [{owner.get(l, "") if p == FISSION else ""}] {src[:90]}  ({top})')
