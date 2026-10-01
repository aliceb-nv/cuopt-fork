# hostgap.py <sqlite>: CPU samples of the replay thread between the last fission round kernel and
# the next presetup, plus the GPU ops in that window.
import sqlite3, collections, bisect, sys
c=sqlite3.connect(sys.argv[1]); S=dict(c.execute("select id,value from StringIds"))
k=[(s,e,S[n]) for s,e,n in c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
w0=k[0][0]+5e9
pres=[s for s,e,n in k if n=='fission_presetup' and s>w0]
gaps=[]
for g0,g1 in zip(pres,pres[1:]):
    le=max(e for s,e,n in k if g0<=s<g1 and n in('fission_update_block','fission_count','fission_yield','fission_price'))
    gaps.append((le,g1))
print('gaps',len(gaps),'mean ms %.1f'%(sum(b-a for a,b in gaps)/len(gaps)/1e6))
L=[a for a,b in gaps]
def ingap(t):
    i=bisect.bisect_right(L,t)-1
    return i>=0 and t<gaps[i][1]
cc=collections.defaultdict(list)
for i,sym,d in c.execute("select id,symbol,stackDepth from SAMPLING_CALLCHAINS"): cc[i].append((d,S.get(sym,'?')))
ev=[(i,t,tid) for i,t,tid in c.execute("select id,start,globalTid from COMPOSITE_EVENTS") if ingap(t)]
thr=collections.Counter(tid for i,t,tid in ev); main=thr.most_common(1)[0][0]
leaf=collections.Counter(); incl=collections.Counter(); n=0
for i,t,tid in ev:
    if tid!=main: continue
    n+=1; st=sorted(cc[i])
    if st: leaf[st[0][1][:100]]+=1
    for f in set(x[1][:100] for x in st): incl[f]+=1
print('main-thread samples',n,'of',len(ev))
print('--- leaf'); [print('%5.1f%% %s'%(100*v/n,f)) for f,v in leaf.most_common(12)]
print('--- inclusive (named)'); [print('%5.1f%% %s'%(100*v/n,f)) for f,v in incl.most_common(60) if not f.startswith('0x')][:0]
for f,v in incl.most_common(80):
    if not f.startswith('0x') and v/n>0.02: print('%5.1f%% %s'%(100*v/n,f))
ops=collections.defaultdict(float)
for s,e,nm in k:
    if ingap(s): ops[nm]+=(e-s)/1e6/len(gaps)
print('GPU kernels in gap (ms/gen):',{a:round(b,2) for a,b in sorted(ops.items(),key=lambda t:-t[1])[:8]})
