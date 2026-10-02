#!/bin/bash
# timed.sh <tag> <nruns> <device> [extra env...]: 1 clone, no dive, 30 s window after 5 s warmup.
W=/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp; S=/home/scratch.yboucher_gpu_1/nvidia_cuopt/bench_results/2026-10-01/snapshot/lrsa-pool-120s.bin
tag=$1; n=${2:-1}; dev=${3:-1}; shift 3 2>/dev/null
mkdir -p $W/exp/logs
for r in $(seq 1 $n); do
  env CUDA_MODULE_LOADING=LAZY CUOPT_BNB_GPU_BATCH=auto CUOPT_BNB_GPU_LOG=1 CUOPT_BNB_GPU_FISSION=1 "$@" \
    LD_LIBRARY_PATH=$W/cpp/build timeout 1200 $NUMA $W/cpp/build/test_netlib --replay-pool $S --device $dev --clones 1 --seconds 35 --warmup 5 --no-dive > $W/exp/logs/$tag-$r.log 2>&1
  echo "$tag-$r rc=$? $(grep -oE 'name="[^"]*"' $W/exp/logs/$tag-$r.log) $(grep -E '^window' $W/exp/logs/$tag-$r.log | grep -oE '[0-9.]+M it/s') $(grep -E '^generation' $W/exp/logs/$tag-$r.log | grep -oE 'p50=[0-9.]+') host $(grep -E '^host' $W/exp/logs/$tag-$r.log | grep -oE 'p50=[0-9.]+') rounds $(grep -oE 'per generation: min=[0-9]+ p10=[0-9]+ p50=[0-9]+' $W/exp/logs/$tag-$r.log | grep -oE 'p50=[0-9]+')"
done
