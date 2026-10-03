#!/usr/bin/env python3
"""ab180.py <log>...: per-run rates from exp/real180.sh logs (elapsed-time prefixed lines).

GPU pivots / nodes: [GPUPOOL-GPU] totals (iterations of nodes that finished on the GPU). Rates use each
GPU's pool window: its first [GPUPOOL-SETUP] line to the 'Explored' line. Non-pool iterations: the
'Explored' total minus all GPU pivots (CPU nodes, root LP, sub-MIPs), over the whole solve time."""
import re
import statistics as stt
import sys

rows = []
for f in sys.argv[1:]:
    gpu, start = {}, {}
    end = total_it = nodes = solve_t = None
    bound = None
    hdr = ''
    for l in open(f):
        m = re.match(r'\s*([\d.]+) (.*)', l)
        if not m:
            continue
        t, s = float(m[1]), m[2]
        if 'device:' in s or 'git hash' in s:
            hdr += s.strip() + ' | '
        g = re.search(r'\[GPUPOOL-SETUP\] worker=\d+ gpu=(\d+)', s)
        if g:
            start.setdefault(int(g[1]), t)
        g = re.search(r'\[GPUPOOL-GPU\] gpu=(\d+) workers=\d+ seats=\d+ starts=\d+ launches=(\d+) nodes=(\d+) pivots=(\d+)', s)
        if g:
            gpu[int(g[1])] = (int(g[2]), int(g[3]), int(g[4]))
        g = re.search(r'Explored (\d+) nodes \((\d+) simplex iterations\) in ([\d.]+)s', s)
        if g:
            end, nodes, total_it, solve_t = t, int(g[1]), int(g[2]), float(g[3])
        g = re.match(r'\s*\d+\s+\d+\s+\S+\s+([-+.\deE]+)\s', s)
        if g:
            bound = g[1]
    if end is None or not gpu:
        print(f, 'incomplete')
        continue
    piv = {k: v[2] / (end - start[k]) for k, v in gpu.items()}
    nod = {k: v[1] / (end - start[k]) for k, v in gpu.items()}
    gp = sum(v[2] for v in gpu.values())
    rows.append((f, piv, nod, gp, total_it, nodes, solve_t, bound, hdr, min(start.values())))

for f, piv, nod, gp, it, nodes, st, bound, hdr, t0 in rows:
    print(f)
    print('  ', hdr[:160])
    print('   pool start %.1fs, solve %.1fs, explored %d nodes, %d iterations, final bound %s' % (t0, st, nodes, it, bound))
    print('   per-GPU pivots/s (M): ' + ' '.join('g%d=%.4f' % (k, piv[k] / 1e6) for k in sorted(piv)) +
          '  mean %.4f  sum %.3f' % (stt.mean(piv.values()) / 1e6, sum(piv.values()) / 1e6))
    print('   per-GPU nodes/s (k):  ' + ' '.join('g%d=%.2f' % (k, nod[k] / 1e3) for k in sorted(nod)) +
          '  sum %.2f' % (sum(nod.values()) / 1e3))
    print('   GPU pivots %d, non-pool iterations %d (%.4f M/s over the solve)' % (gp, it - gp, (it - gp) / st / 1e6))
