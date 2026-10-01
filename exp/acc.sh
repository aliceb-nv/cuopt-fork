#!/bin/bash
# acc.sh <tag> [device]: 1 clone, yield 0, 3 generations; compare with the existing-kernel reference.
W=/home/scratch.yboucher_gpu_1/nvidia_cuopt-fission-exp; S=/home/scratch.yboucher_gpu_1/nvidia_cuopt/bench_results/2026-10-01/snapshot/lrsa-pool-120s.bin
tag=$1; dev=${2:-1}; mkdir -p $W/exp/logs
env CUDA_MODULE_LOADING=LAZY CUOPT_BNB_GPU_BATCH=auto CUOPT_BNB_GPU_FISSION=1 CUOPT_BNB_GPU_FISSION_YIELD=0 LD_LIBRARY_PATH=$W/cpp/build timeout 1800 \
  $W/cpp/build/test_netlib --replay-pool $S --device $dev --clones 1 --generations 3 --no-dive > $W/exp/logs/acc-$tag.log 2>&1
echo "acc-$tag rc=$?"
python3 - $W/exp/logs/acc-$tag.log <<'PY'
import re,sys
def load(f):
    d={}
    for l in open(f):
        m=re.match(r'seat\s+(\d+) node\s+(\d+) (\S+)\s+obj=(\S+) it=(\d+)',l)
        if m: d[int(m[1])]=(int(m[2]),m[3],float(m[4]),int(m[5]))
    return d
a=load('/home/scratch.yboucher_gpu_1/nvidia_cuopt/bench_results/2026-10-01/fission/det-mono.log'); b=load(sys.argv[1]); ks=sorted(set(a)&set(b))
o=sorted(abs(a[s][2]-b[s][2])/(1+abs(a[s][2])) for s in ks if a[s][1]==b[s][1]=='OPTIMAL')
same=sum(1 for s in ks if a[s][1]==b[s][1]); mx=o[-1] if o else 1
ok= len(ks)==1056 and same>=1032 and mx<=1e-12
print('seats %d same-status %d same-iters %d OPTIMAL n=%d max-rel %.2e -> %s'%(len(ks),same,sum(1 for s in ks if a[s][3]==b[s][3]),len(o),mx,'PASS' if ok else 'FAIL'))
PY
