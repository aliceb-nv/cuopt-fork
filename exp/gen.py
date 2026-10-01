# gen.py <sqlite>: timeline of 1-clone generations (device 1): GPU ops and idle, by category.
import sqlite3, collections, sys, numpy as np
c=sqlite3.connect(sys.argv[1]); S=dict(c.execute("select id,value from StringIds"))
ops=[(s,e,S[n]) for s,e,n in c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=1")]
ops+=[(s,e,{1:'H2D',2:'D2H',8:'D2D'}.get(k,'cpy%d'%k)) for s,e,k in c.execute("select start,end,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=1")]
ops+=[(s,e,'memset') for s,e in c.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMSET where deviceId=1")]
ops.sort(); w0=ops[0][0]+5e9
first=[s for s,e,n in ops if n in('fission_presetup','fission_setup_kernel') and s>w0]
gs=sorted(set(first)); starts=[]
for t in gs:
    if not starts or t-starts[-1]>50e6: starts.append(t)
cat=lambda n: n.replace('fission_','') if n.startswith('fission') else ('copy '+n if n in('H2D','D2H','D2D','memset') else 'other kernels')
rows=[]
for g0,g1 in zip(starts,starts[1:]):
    d=collections.defaultdict(float); busy=0; last=g0
    for s,e,n in ops:
        if s<g0 or s>=g1: continue
        d[cat(n)]+=(e-s)/1e6
        busy+=max(0,e-max(s,last))/1e6; last=max(last,e)
    d['GPU idle']=(g1-g0)/1e6-busy; d['TOTAL']=(g1-g0)/1e6; rows.append(d)
keys=sorted(set(k for r in rows for k in r),key=lambda k:-np.median([r.get(k,0) for r in rows]))
print('generations',len(rows),'(ms per generation)'); print('  %-22s %8s %8s %8s %8s'%('','min','median','p90','max'))
for k in keys:
    v=np.array([r.get(k,0) for r in rows]); print('  %-22s %8.1f %8.1f %8.1f %8.1f'%(k,v.min(),np.median(v),np.percentile(v,90),v.max()))
