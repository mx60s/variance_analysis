"""Known continuous response replay on all 30 curated HD trajectories.
Eight Poisson repeats per response; actual alignment, speed selection, fold
assignment, support, smoothing, and native-grid variance estimator.
"""
import os,sys,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
B=Path(__file__).resolve().parents[2];R=Path(os.environ.get('EV_REPO',str(B)));sys.path[:0]=[str(B),str(B/'scripts/finalization')]
from src.aligned_analysis import coalesce_tracking_timestamps,average_in_bins,align_tracking,labels,grouped_maps
from variance_estimator import score,components
OUT=B/'outputs/grid_primary';OUT.mkdir(exist_ok=True)
REPEATS=8;NAMES=['hd_broad_0','hd_broad_120','hd_narrow_240','place_center','place_offcenter','mixed_gain']
def rates_at(x,y,theta):
 x,y,theta=np.broadcast_arrays(x,y,theta)
 h=[np.exp(k*(np.cos(theta-a)-1)) for k,a in [(2,0),(2,2*np.pi/3),(8,4*np.pi/3)]]
 p=[np.exp(-((x-a)**2+(y-b)**2)/(2*.15**2)) for a,b in [(.5,.5),(.3,.7)]]
 return np.stack([1+19*h[0],1+19*h[1],1+19*h[2],1+19*p[0],1+19*p[1],1+19*h[0]*(.2+.8*p[0])],axis=-1)
for ii,folder in enumerate(sorted((B/'results/review_20261116/revalidation/grid_10').glob('dusz_*'))):
 dest=OUT/(folder.name+'.npz')
 if dest.exists():continue
 t0=time.time();name=folder.name.split('_',2)[2];z=np.load(R/'results/review_20261116/revalidation/curated_inputs'/f'{name}.npz');pos,_=coalesce_tracking_timestamps(z['positions']);spikes=z['spikes'];tau=.05;edges=np.arange(spikes[0,0],spikes[-1,0]+tau,tau)
 dt=np.diff(pos[:,0]);v=np.linalg.norm(np.gradient(pos[:,1:3],pos[:,0],axis=0),axis=1);bad=np.r_[False,dt>1.5*np.median(dt)]|np.r_[dt>1.5*np.median(dt),False];v[bad]=np.nan;g=np.isfinite(v);den=gaussian_filter1d(g.astype(float),3,mode='nearest');v=np.divide(gaussian_filter1d(np.where(g,v,0),3,mode='nearest'),den,out=np.full_like(v,np.nan),where=den>0);v[bad]=np.nan
 vel=average_in_bins(pos[:,0],v,edges);p,ti,valid=align_tracking(pos,edges);valid=valid&(vel>=2)
 bounds=[(np.nanmin(pos[:,j]),np.nanmax(pos[:,j])) for j in [1,2]]
 xx=(p[:,1]-bounds[0][0])/(bounds[0][1]-bounds[0][0]);yy=(p[:,2]-bounds[1][0])/(bounds[1][1]-bounds[1][0]);expected=rates_at(xx,yy,p[:,3]*np.pi/180);expected=np.nan_to_num(expected)
 rng=np.random.default_rng(72100+ii);observed=np.concatenate([rng.poisson(expected*tau)/tau for _ in range(REPEATS)],axis=1);allrates=np.c_[expected,observed];arrays={}
 for n in [10,20,30]:
  xe,ye=[np.linspace(a,b+1e-6,n+1) for a,b in bounds];x,y=labels(p[:,1],xe),labels(p[:,2],ye);h=labels(p[:,3],np.linspace(0,360+1e-6,91));row=np.where((x>=0)&(y>=0),y*n+x,-1)
  a,b,o=grouped_maps(allrates,row,h,n*n,90,valid,floor=5)
  d=score(a,b,o,spatial=True,sigma=(n/20,1),zoom=(1,1))
  # Deterministic fold replay targets account for the same time samples in each fold.
  arrays[f'g{n}_noiseless']=d['ev'][:len(NAMES)];arrays[f'g{n}_estimate']=d['ev'][len(NAMES):].reshape(REPEATS,len(NAMES),2)
  arrays[f'g{n}_raw_noiseless']=d['raw'][:len(NAMES)];arrays[f'g{n}_raw_estimate']=d['raw'][len(NAMES):].reshape(REPEATS,len(NAMES),2)
  arrays[f'g{n}_total_noiseless']=d['total'][:len(NAMES)];arrays[f'g{n}_total_estimate']=d['total'][len(NAMES):].reshape(REPEATS,len(NAMES))
  arrays[f'g{n}_support']=d['support'][0]
  saved=np.load(B/'results/review_20261116/revalidation'/({10:'grid_10',20:'curated_native_coarse',30:'grid_30'}[n])/folder.name/'xyhd.npz');assert d['support'][0]==saved['support'][0]
  xc,yc=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n);th=(np.arange(90)+.5)*2*np.pi/90;full=rates_at(xc.ravel()[:,None],yc.ravel()[:,None],th[None,:]+np.zeros((n*n,1)))
  arrays[f'g{n}_full_grid']=np.array([components(full[:,:,j],full[:,:,j])['normalized'] for j in range(len(NAMES))])
 np.savez_compressed(dest,**arrays);print(name,round(time.time()-t0,1),flush=True)
rows=[]
zz=[np.load(p) for p in sorted(OUT.glob('dusz_*.npz'))];assert len(zz)==30
for n in [10,20,30]:
 for j,name in enumerate(NAMES):
  est=np.stack([z[f'g{n}_estimate'][:,j] for z in zz]);truth=np.stack([z[f'g{n}_noiseless'][j] for z in zz]);target=np.stack([z[f'g{n}_full_grid'][j] for z in zz]);raw=np.stack([z[f'g{n}_raw_estimate'][:,j] for z in zz]);rt=np.stack([z[f'g{n}_raw_noiseless'][j] for z in zz]);total=np.stack([z[f'g{n}_total_noiseless'][j] for z in zz]);good=np.isfinite(est).all(-1)
  rows.append(dict(grid=n,response=name,n=int(good.size),n_finite=int(good.sum()),median=np.median(est[good],axis=0).tolist(),noiseless_median=np.median(truth,axis=0).tolist(),full_grid_median=np.median(target,axis=0).tolist(),median_abs_ratio_error=np.nanmedian(np.abs(est-truth[:,None,:]),axis=(0,1)).tolist(),mean_raw_error_over_known_total=np.mean((raw-rt[:,None,:])/total[:,None,None],axis=(0,1)).tolist(),hd_greater=float(np.mean(est[good,1]>est[good,0]))))
(OUT/'recovery_summary.json').write_text(json.dumps(dict(seed=72100,repeat_count=REPEATS,sessions=30,angular_bins=90,rate_range_hz=[1,20],spatial_field_sigma_fraction=.15,note='Synthetic continuous functions replayed on actual trajectories, 50-ms Poisson sampling. Full-grid reference is unsmoothed bin-center signal; noiseless replay is the same folded, smoothed and supported pipeline without Poisson noise. This validation does not establish the biological origin of residual spatial modulation.',rows=rows),indent=2));print(json.dumps(rows,indent=2))
