from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.rodent_data import load_pfeiffer_data
from src.aligned_analysis import coalesce_tracking_timestamps
OUT=Path(__file__).resolve().parents[2]/'outputs/ca1_selection';OUT.mkdir(exist_ok=True)
for rat in range(1,5):
 source=ROOT/f'data/rodents/pfeiffer/Rat{rat}/Open1';pos,spikes,ids=load_pfeiffer_data(source);pos,_=coalesce_tracking_timestamps(pos)
 # Each valid tracking interval contributes its measured dwell time and spikes;
 # position is linearly interpolated for each spike within that interval.
 t=pos[:,0];dt=np.diff(t);delta=np.diff(pos[:,1:3],axis=0);speed=np.linalg.norm(delta,axis=1)/dt
 good=(dt>0)&(dt<=1.5*np.median(dt))&(speed>5)&np.isfinite(delta).all(1)
 edges=np.arange(0,202.00001,2.);n=len(edges)-1
 midpoint=(pos[:-1,1:3]+pos[1:,1:3])/2
 occupancy=np.histogram2d(midpoint[good,1],midpoint[good,0],bins=(edges,edges),weights=dt[good])[0]
 ix=np.searchsorted(t,spikes[:,0],side='right')-1;valid=(ix>=0)&(ix<len(dt));ixc=np.clip(ix,0,len(dt)-1);valid &= good[ixc]
 alpha=(spikes[:,0]-t[ixc])/dt[ixc];xy=pos[ixc,1:3]+alpha[:,None]*delta[ixc]
 smooth_occ=gaussian_filter(occupancy,2,mode='constant');peaks=[];alt=[]
 for cell in ids:
  m=valid&(spikes[:,1]==cell);counts=np.histogram2d(xy[m,1],xy[m,0],bins=(edges,edges))[0]
  numerator=gaussian_filter(counts,2,mode='constant');rate=np.divide(numerator,smooth_occ,out=np.zeros_like(numerator),where=smooth_occ>0)
  # Only locations visited during movement are eligible for a field peak.
  peaks.append(float(np.max(rate[occupancy>0])))
  raw=np.divide(counts,occupancy,out=np.zeros_like(counts),where=occupancy>0);den=gaussian_filter((occupancy>0).astype(float),2,mode='constant');other=np.divide(gaussian_filter(raw,2,mode='constant'),den,out=np.zeros_like(raw),where=den>0);alt.append(float(np.max(other[occupancy>0])))
 peaks=np.array(peaks);alt=np.array(alt);mask=peaks>1
 result=dict(rat=rat,ids=ids.tolist(),peak_hz=peaks.tolist(),selected=mask.tolist(),alternative_rate_then_smooth_peak_hz=alt.tolist(),alternative_selected=(alt>1).tolist(),n_total=len(ids),n_selected=int(mask.sum()),n_alternative_selected=int((alt>1).sum()),n_disagree=int(np.sum(mask!=(alt>1))),bin_cm=2,smoothing_sigma_cm=4,speed_threshold_cm_s=5,method='Gaussian-smoothed spike counts divided by Gaussian-smoothed dwell time; peaks restricted to sampled bins; actual tracking intervals and linear spike-position interpolation; gap exclusion 1.5 median dt',source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.mat')})
 (OUT/f'rat{rat}.json').write_text(json.dumps(result,indent=2)+'\n');print(rat,len(ids),mask.sum(),'alternative',sum(alt>1),'disagree',result['n_disagree'],flush=True)
