# Fission per-round work, session 2 (2026-10-02, A100 80GB PCIe, device 1)

Node g492-ha0-0004 (8x A100 80GB PCIe). Build `80-real`. Every timing: `exp/timed.sh` (1 clone, no dive,
30 s window after 5 s warmup); every gate: `exp/acc.sh` against `exp/logs/acc-ref-a100-fresh.log`
(existing kernel, exact starting weights, this node).

## Baseline and gate noise

- Unchanged fission: 0.162-0.163M it/s over 5 runs (generation 0.472-0.475 s, host restage 92-94 ms).
- Unchanged fission vs the reference, 4 runs: same-status 1030, 1029, 1037, 1034 (max rel 1.6e-14 ..
  2.0e-14). The `>= 1032` gate is inside the run-to-run spread of the unchanged code (atomics), so each
  change gets 2-3 gate runs; every change kept stays within the baseline's spread and max rel <= 6.2e-14.

## Kept changes (commit, it/s after)

| commit | change | it/s |
|---|---|---|
| 9dc42a11 | dy solve: v only at the rows the SE update reads (row gather over scaled_dxB); 512-thread B/C/D blocks (32-register cap spilled); batched scatter | 0.180 |
| 5e9df8a5 | gather: long rows queued in shared memory | 0.182-0.183 |
| b784ec7e | replay: R d split over threads, axpys through the balanced scatter | 0.185-0.186 |
| 7607eeb2 | fresh seats compute y, z on the device from the inverses; staging skips y, z | 0.189 |
| d705e64c | update: one fused dx/primal pass, positions cached in shared memory | 0.190-0.191 |
| b47ea259 | inverse build: transpose thread/warp hybrid (11.8 -> 7.5 ms/gen) | 0.191-0.192 |
| de657fa0 | replay: R d over nonzero d only, axpys over nonzero thetas only | 0.194-0.195 |
| fb08d3bd | FTRAN tail reuses BTRAN's replay dots, sums over nonzero w | 0.195 |
| d4717680, c211ab66 | choose: dz before the compaction strips (long columns queued), one packed scan | 0.198-0.200 |
| 1cc50726 | dy row gather: rows up to 32 entries per thread | 0.199-0.200 |
| 118257c8 | next pricing done in the update's infeasibility pass | 0.203 |
| 9005be82 | restaging: host-LU fresh seats stage bounds as root + path deltas | 0.205-0.206 |

Every change keeps the old path behind an env switch (see the `[MODE]` line in each log).

## Tried and reverted (no gain or worse)

- Warp-per-segment scatter alone (heavy-tailed segments: per-seat time up); row gather of all of v.
- Batch-of-4 scatter at 1024 threads (spills); batch size 1/2/4 at 512 threads: no difference.
- Fusing the SE update into the primal pass: MPF warp became the critical path; gate 1030/1027.
- Tiled R d over (32-row chunk, k): R is DRAM-resident (K x 256 doubles per seat), no gain.
- Dynamic warp scheduling of the replay dots: no gain.
- Padding to 1 seat/SM (test): FTRAN only +13% -> 3 seats/SM would not pay for FTRAN.

## Final state (HEAD 6f6f060b)

- Separate kernels (default): 0.205-0.206M it/s over 5 runs (+26% over 0.162-0.163M); generation
  0.374 s, host restage 0.072-0.073 s.
- Fused kernel (`CUOPT_EXP_FUSED=1`, still opt-in): 0.216-0.218M it/s over 5 runs (+33%).
  With 2 clones: fused 0.287M vs separate 0.284M aggregate, so most of fused's 1-clone gain is the
  tail (seats no longer wait for each other between phases), which a second clone also hides.
- Gate, 3 runs of the default: same-status 1033/1034/1035, max rel <= 4.5e-14.

## Blockers and structural findings

- Memory: each clone allocates 12.8 GiB for the per-generation inverse buffers (per-seat capacity
  at the measured maximum, kInvCapU = 329296, three forms, 1056 seats). 4 and 8 clones fail with
  out_of_memory on the 80 GB A100, so fission cannot run the 8 pools of a real run here. Actual
  per-generation inverse nnz is ~49M entries vs ~369M reserved; sizing by actual counts (prefix
  sum of per-seat counts after the column solve) would cut it ~7.5x. Not changed.
- 1-clone ceiling: kernel times stay flat or grow through a generation while live seats drop from
  ~940 to ~160 (replay cost grows with K), so the second half of each generation runs a few slow
  high-K seats on a mostly idle GPU. 2 clones reach 0.284M aggregate. Reaching 0.3M on 1 clone needs
  a structural change, e.g. refilling finished seats mid-generation; that changes pool scheduling
  and was not attempted.
- Also tried after the table above: update kernel at 4 blocks/SM (no change: latency-bound, not
  occupancy-bound); L0 staged without its diagonal (the diagonal is +-1, not always 1, and the
  host compaction costs what the copy saves); unordered infeasibility list (no gain).

## Where a generation goes now (A100, 1 clone)

GPU ~0.375 s + host ~0.072 s. Per seat and iteration (cycles): FTRAN dy solve (L 31k, replay 82k,
gather 48k) is the largest block; update MPF warp 64k vs primal side ~85k; choose dz 44k; BTRAN replay
42k. FTRAN kernel time grows through a generation (replay cost grows with K) while live seats drop,
so the late rounds are single-seat latency at K ~ 190.
