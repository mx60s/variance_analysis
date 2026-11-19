"""Rebuild identified biological examples from aligned source data; save plot inputs."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from scipy.ndimage import gaussian_filter1d
from regenerate_biology import *
from src.explained_variance import smooth_location_hd

def main(out):
 out.mkdir(parents=True,exist_ok=True)
 for kind,name,j in [('pfeiffer','4',63),('pfeiffer','2',133),('dusz','A3702-191126',13),('dusz','A3710-200609',57)]:
  target=out/f'{kind}_{name}_{j}.npz'
  if target.exists():continue
  if kind=='pfeiffer':
   source=ROOT/f'data/rodents/pfeiffer/Rat{name}/Open1';pos,spikes,ids=load_pfeiffer_data(source);vs=compute_velocity_series(pos);tau=.25;nh=60;bounds=[(0,201),(0,201)]
  else:
   source=ROOT/'results/review_20261116/revalidation/curated_inputs'/f'{name}.npz';z=np.load(source);spikes,pos,ids=z['spikes'],z['positions'],z['neuron_ids'];tau=.05;nh=90
   # Resolve the historical selected-array index to an actual unit ID before matching the curated release.
   _,_,cell,_=load_dusz_data(ROOT/'data/rodents/dusz',1,name);oldids=np.flatnonzero((cell['hd']==1)&(cell['ex']==1));uid=oldids[j]
   matches=np.flatnonzero(ids==uid)
   if len(matches)!=1:raise ValueError((name,uid,ids))
   jcur=int(matches[0]);pos,_=coalesce_tracking_timestamps(pos);dt=np.diff(pos[:,0]);v=np.linalg.norm(np.gradient(pos[:,1:3],pos[:,0],axis=0),axis=1);bad=np.r_[False,dt>1.5*np.median(dt)]|np.r_[dt>1.5*np.median(dt),False];v[bad]=np.nan;good=np.isfinite(v);den=gaussian_filter1d(good.astype(float),3,mode='nearest');num=gaussian_filter1d(np.where(good,v,0),3,mode='nearest');v=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);v[bad]=np.nan;vs=np.c_[pos[:,0],v];bounds=[(float(np.nanmin(pos[:,k])),float(np.nanmax(pos[:,k]))) for k in [1,2]]
  jcur=j if kind=='pfeiffer' else jcur;uid=int(ids[jcur]);pos,_=coalesce_tracking_timestamps(pos);edges=np.arange(spikes[0,0],spikes[-1,0]+tau,tau);rates=build_spike_rates(spikes,edges,tau,[uid]);aligned,ti,valid=align_tracking(pos,edges);vel=average_in_bins(vs[:,0],vs[:,1],edges)
  xe,ye=[np.linspace(a,b+1e-6,21) for a,b in bounds];he=np.linspace(0,360+1e-6,nh+1);x,y,h=labels(aligned[:,1],xe),labels(aligned[:,2],ye),labels(aligned[:,3],he);row=np.where((x>=0)&(y>=0),y*20+x,-1)
  a,b,o=grouped_maps(rates,row,h,400,nh,valid&(vel>=(5 if kind=='pfeiffer' else 2)),floor=5)
  raw=(a[:,:,0]+b[:,:,0])/2;smooth=smooth_location_hd(raw,(1,2) if kind=='pfeiffer' else (1,1));m=smooth.reshape(20,20,nh).transpose(1,0,2);raw=raw.reshape(20,20,nh).transpose(1,0,2);occ=o.reshape(20,20,nh).transpose(1,0,2)
  np.savez_compressed(target,map=m,raw=raw,occupancy=occ,x_edges=xe,y_edges=ye,hd_edges=he,unit_id=uid)
  target.with_suffix('.json').write_text(json.dumps({'kind':kind,'session':name,'historical_selection_index':j,'curated_selection_index':jcur,'unit_id':uid,'tau':tau,'source':str(source),'map':'equal-weight mean of split-half maps; true XY smoothing, circular HD; no upsampling','sd':'population SD across finite smoothed HD bins; coverage is raw supported heading count'},indent=2));print(target,uid,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
