"""Paired XY-decoder audit: unsupported-state masking versus time-bin assignment.

All variants use the SAME legacy data, folds, counts, aligned test targets and
speed-eligible bins. This isolates two implementation changes, not every change
between the historical manuscript and the curated-release analysis.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from regenerate_biology import sessions,curves
from src.rodent_data import load_dusz_data,build_spike_rates
from src.aligned_analysis import align_tracking,coalesce_tracking_timestamps,average_in_bins,labels
from src.decoding import make_cv_masks

def predict_pair(rate,counts,tau):
 r=rate*tau;valid=np.isfinite(r).any(1)
 logs=np.where(np.isfinite(r),np.log(np.nan_to_num(r)+1e-10),0)
 penalty=np.nansum(r,axis=1)
 old=[];fixed=[]
 for start in range(0,len(counts),256):
  ll=counts[start:start+256]@logs.T-penalty
  old.append(np.argmax(ll,axis=1));ll[:,~valid]=-np.inf
  fixed.append(np.argmax(ll,axis=1))
 return np.concatenate(old),np.concatenate(fixed),valid

def run(output):
 out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[]
 with threadpool_limits(limits=2):
  for kind,d,name in sessions():
   if kind!='dusz':continue
   dest=out/(name+'.npz');summary=out/(name+'.json')
   if summary.exists():rows.append(json.loads(summary.read_text()));continue
   start=time.monotonic();spikes,pos,cell,velocity=load_dusz_data(ROOT/'data/rodents/dusz',d,name)
   ids=np.flatnonzero((cell['hd']==1)&(cell['ex']==1));pos,duplicates=coalesce_tracking_timestamps(pos)
   tau=.05;edges=np.arange(spikes[0,0],spikes[-1,0]+tau,tau);rates=build_spike_rates(spikes,edges,tau,ids)
   aligned,idx,valid=align_tracking(pos,edges);vel=average_in_bins(velocity[:,0],velocity[:,1],edges);good=valid&np.isfinite(vel)&(vel>=2)
   xe,ye=[np.linspace(np.nanmin(pos[:,j]),np.nanmax(pos[:,j])+1e-6,33) for j in [1,2]]
   x,y=labels(aligned[:,1],xe),labels(aligned[:,2],ye);lab=np.where((x>=0)&(y>=0),y*32+x,-1)
   sample_bins=np.searchsorted(edges,pos[:,0],side='right')-1;sx,sy=labels(pos[:,1],xe),labels(pos[:,2],ye);slab=sy*32+sx
   sv=(sample_bins>=0)&(sample_bins<len(rates))&(sx>=0)&(sy>=0)
   sb=sample_bins[sv];sl=slab[sv]
   centers=np.stack(np.meshgrid((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2),-1).reshape(-1,2)
   data={k:[] for k in ['legacy_unmasked','legacy_masked','aligned_masked','legacy_occupancy','aligned_occupancy']};invalid=[];differences=[];multi=[];foldrows=[]
   for fold,train in enumerate(make_cv_masks(len(rates),4)):
    keep=train[sb]&(vel[sb]>=2)&np.isfinite(vel[sb]);bins=sb[keep];states=sl[keep]
    occ=np.bincount(states,minlength=1024);sums=np.zeros((1024,len(ids)));np.add.at(sums,states,rates[bins])
    legacy=np.divide(sums,occ[:,None],out=np.full_like(sums,np.nan),where=occ[:,None]>0);legacy[occ<10]=np.nan
    aligned_maps,aligned_occ=curves(rates,lab,good&train,1024);aligned_maps[aligned_occ<10]=np.nan
    test=np.flatnonzero(~train);counts=gaussian_filter1d(rates[test]*tau,2,axis=0,mode='nearest');eligible=good[test]&(lab[test]>=0);ti=test[eligible];counts=counts[eligible]
    raw,fixed,support=predict_pair(legacy,counts,tau);_,new,new_support=predict_pair(aligned_maps,counts,tau)
    pred={'legacy_unmasked':raw,'legacy_masked':fixed,'aligned_masked':new,'legacy_occupancy':np.full(len(ti),occ.argmax()),'aligned_occupancy':np.full(len(ti),aligned_occ.argmax())}
    for k,p in pred.items():data[k].append(np.linalg.norm(centers[p]-aligned[ti,1:3],axis=1))
    invalid.extend((~support[raw]).tolist());differences.extend((raw!=fixed).tolist())
    multiplicity=np.bincount(np.unique(bins*1024+states)//1024,minlength=len(rates));assigned=multiplicity[train&(multiplicity>0)]
    multi.extend((assigned>1).tolist())
    foldrows.append({'fold':fold,'legacy_unsupported_states':int((~support).sum()),'aligned_unsupported_states':int((~new_support).sum()),'test_bins':len(ti)})
   arrays={k:np.concatenate(v) for k,v in data.items()};np.savez_compressed(dest,**arrays)
   row={'session':name,'dataset':d,'test_bins':len(arrays['legacy_masked']),'median_errors':{k:float(np.median(v)) for k,v in arrays.items()},'fraction_legacy_predictions_unsupported':float(np.mean(invalid)),'fraction_predictions_changed_by_mask':float(np.mean(differences)),'fraction_training_bins_assigned_multiple_xy_states':float(np.mean(multi)),'duplicate_timestamps_merged':duplicates,'folds':foldrows,'seconds':time.monotonic()-start}
   summary.write_text(json.dumps(row,indent=2));rows.append(row);print(name,row['median_errors'],row['fraction_legacy_predictions_unsupported'],flush=True)
 pooled={k:np.concatenate([np.load(out/(r['session']+'.npz'))[k] for r in rows]) for k in rows[0]['median_errors']}
 result={'sessions':rows,'pooled_median_errors':{k:float(np.median(v)) for k,v in pooled.items()},'units':'legacy coordinate units','comparison':'same legacy data, four held-out folds, common aligned targets/test bins; legacy sample-weighted maps, then unsupported-state mask, then one-state-per-bin maps/occupancy','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (out/'summary.json').write_text(json.dumps(result,indent=2));print('COMPLETE',result['pooled_median_errors'],flush=True)

if __name__=='__main__':run(sys.argv[1])
