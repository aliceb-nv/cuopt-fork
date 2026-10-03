#!/bin/bash
# real180.sh <tag> <fission 0|1> [time_limit]: lrsa120, cuopt_cli, all GPUs (pools on 1..N-1), 4 pools per GPU.
W=/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp
I=/home/scratch.yboucher_gpu_1/nvidia_cuopt/bench_results/2026-09-28/instances/lrsa120.mps.gz
tag=$1; F=$2; T=${3:-180}; mkdir -p $W/exp/logs/real180
env CUDA_MODULE_LOADING=LAZY CUOPT_BNB_GPU_BATCH=auto CUOPT_BNB_GPU_LOG=1 CUOPT_BNB_GPU_LOG_WARMUP=5 \
  CUOPT_BNB_GPU_POOLS_PER_GPU=4 CUOPT_BNB_GPU_FISSION=$F $EXTRA_ENV LD_LIBRARY_PATH=$W/cpp/build timeout $((T + 300)) \
  $W/cpp/build/cuopt_cli $I --mip-determinism-mode 0 --num-gpus -1 --num-cpu-threads 128 --time-limit $T 2>&1 |
  python3 -u -c "import sys,time
t0=time.time()
for l in sys.stdin: sys.stdout.write('%8.3f %s' % (time.time()-t0, l))" > $W/exp/logs/real180/$tag.log
echo "$tag rc=${PIPESTATUS[0]}"
