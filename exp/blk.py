# blk.py <trace.bin>: per-launch spans/tails of a 1-clone per-block trace; live block durations by nzx.
import numpy as np, sys
dt=np.dtype([('t0','u8'),('t1','u8'),('args','u8'),('seat','i4'),('sm','i4'),('live','i4'),('nzx','i4'),('unz','i4'),('enz','i4'),('cm','i4'),('cp','i4'),('sub','i4',10)])
r=np.fromfile(sys.argv[1],dtype=dt); r=r[r['t0']>0]; x=r[np.argsort(r['t0'])]
L=[];mx=0;st=0
for i in range(len(x)):
    if i>0 and x['t0'][i]>mx: L.append(x[st:i]); st=i
    mx=max(mx,x['t1'][i])
L.append(x[st:]); L=[l for l in L if len(l)==1056 and l['live'].sum()>0][1:-1]
def P(v,lab): v=np.asarray(v,float); print('  %-46s %8.0f %8.0f %8.0f %8.0f'%(lab,v.min(),np.median(v),np.percentile(v,90),v.max()))
print('launches',len(L)); print('  %-46s %8s %8s %8s %8s'%('','min','median','p90','max'))
P([(l['t1'].max()-l['t0'].min())/1e3 for l in L],'launch span (us)')
P([l['live'].sum() for l in L],'live blocks')
P([(l['t0'].max()-l['t0'].min())/1e3 for l in L],'last block start offset (us)')
P([(l['t1'].max()-np.percentile(l['t1'],90))/1e3 for l in L],'tail: 90% done -> last end (us)')
P([(l['t1'].max()-np.percentile(l['t1'][l['live']==1],50))/1e3 for l in L],'tail: 50% live done -> last end (us)')
lv=np.concatenate([l[l['live']==1] for l in L]); d=(lv['t1']-lv['t0'])/1e3
P(d,'live block duration (us)')
P(lv['nzx'],'live: nzx field')
for lo,hi in [(0,25),(25,50),(50,100),(100,150),(150,201)]:
    s=(lv['nzx']>=lo)&(lv['nzx']<hi)
    if s.sum(): print('    nzx [%3d,%3d): %5.1f%%  duration us p50 %6.0f p90 %6.0f'%(lo,hi,100*s.mean(),np.median(d[s]),np.percentile(d[s],90)))
last=np.array([l[np.argmax(l['t1'])] for l in L])
P((last['t1']-last['t0'])/1e3,'last-ending block duration (us)'); P(last['nzx'],'last-ending block nzx')
P((last['t0']-np.array([l['t0'].min() for l in L]))/1e3,'last-ending block start offset (us)')
