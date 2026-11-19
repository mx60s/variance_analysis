"""Revalidate HD/MD scores and matched-bin decoders with gap-aware kinematics."""
import os,sys,json,time,hashlib
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
BASE=Path(__file__).resolve().parents[2];REPO=BASE
sys.path[:0]=[str(REPO),str(REPO/'scripts'),str(BASE/'scripts/finalization')]
import numpy as np
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits
from src.movement_direction import estimate_movement
from src.rodent_data import build_spike_rates,get_md_from_xy
from src.aligned_analysis import coalesce_tracking_timestamps,align_tracking,grouped_maps,labels,poisson_argmax,average_in_bins
from src.decoding import make_cv_masks,gaussian_filter_hd
from regenerate_biology import sessions,curves
from variance_estimator import score,summarize
OUT=BASE/'outputs/md_revision';INPUT=Path(os.environ.get('MD_CURATED_INPUTS',str(REPO/'results/review_20261116/revalidation/curated_inputs')))
def circerr(a,b):return np.abs((a-b+180)%360-180)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(rates,hd,md,good,tau):
    labs={k:labels(v,np.linspace(0,360+1e-6,361)) for k,v in [('hd',hd),('md',md)]}
    good=good&(labs['hd']>=0)&(labs['md']>=0);arrays={}
    for fold,train in enumerate(make_cv_masks(len(rates),4)):
        test=np.flatnonzero(~train);counts=gaussian_filter1d(rates[test]*tau,.1/tau,axis=0,mode='nearest');eligible=good[test];ti=test[eligible];sc=counts[eligible]
        for axis,lab in labs.items():
            m,occ=curves(rates,lab,train&good,360);m[occ<10]=np.nan;m=gaussian_filter_hd(m,3)
            pred=poisson_argmax(m,sc,tau,offset=1e-12)
            centers=np.arange(360)+.5;truth=hd[ti] if axis=='hd' else md[ti]
            err=circerr(centers[np.maximum(pred,0)],truth);err[pred<0]=np.nan
            base=circerr(centers[np.argmax(occ)],truth)
            arrays[f'{axis}_fold{fold}_errors']=err;arrays[f'{axis}_fold{fold}_baseline']=base;arrays[f'{axis}_fold{fold}_timebins']=ti
    return arrays

def run(dataset,session):
    out=OUT/'sessions'/f'dusz_{dataset}_{session}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    start=time.monotonic();z=np.load(INPUT/(session+'.npz'))
    pos,_=coalesce_tracking_timestamps(z['positions']);spikes=z['spikes'];ids=z['neuron_ids'];tau=.05
    edges=np.arange(spikes[0,0],spikes[-1,0]+tau,tau);rates=build_spike_rates(spikes,edges,tau,ids)
    aligned,ti,valid=align_tracking(pos,edges);hd=aligned[:,3];valid &= np.isfinite(hd)
    legacy=(get_md_from_xy(pos,20)+90)%360;legacy=legacy[ti]
    report={'session':session,'dataset':dataset,'n_neurons':len(ids),'n_spike_bins':len(rates),'variants':{},'tracking':{}}
    for window in [.1,.2,.4]:
        kin=estimate_movement(pos,window);md=kin['direction'][ti];speed=kin['speed'][ti]
        ok=valid&kin['valid'][ti];md[~ok]=np.nan
        report['tracking'][str(window)]={k:int(v) if isinstance(v,(int,np.integer)) else float(v) for k,v in kin.items() if k not in ['velocity','speed','direction','valid']}
        for threshold in [1,2,5,10]:
            name=f'w{round(window*1000)}_s{threshold}';good=ok&(speed>=threshold)
            for bins in ([36,90] if window==.2 and threshold==5 else [36]):
                suffix='' if bins==36 else '_b90';p=out/(name+suffix+'.npz')
                hm,mm=[labels(v,np.linspace(0,360+1e-6,bins+1)) for v in [hd,md]]
                a,b,o=grouped_maps(rates,hm,mm,bins,bins,good,floor=10)
                if not p.exists():
                    res=score(a,b,o,spatial=False,sigma=(bins/90,bins/90),zoom=(1,1));np.savez_compressed(p,**res,unit_ids=ids,occupancy=o)
                else:res=dict(np.load(p))
                ss=summarize(res);ss.update(n_eligible=int(good.sum()),seconds_eligible=float(good.sum()*tau),supported_joint_bins=int(np.sum(o>0)),median_md_change=float(np.nanmedian(circerr(md[good],legacy[good]))))
                report['variants'][name+suffix]=ss
            # Decode 200ms thresholds, plus window controls at the primary speed.
            if window==.2 or threshold==5:
                dp=out/(name+'_decoding.npz')
                if not dp.exists():np.savez_compressed(dp,**decode(rates,hd,md,good,tau))
            if window==.2 and threshold==5:
                # Matched old/new MD: identical times, masks, folds, and counts.
                hm,mm=[labels(v,np.linspace(0,360+1e-6,37)) for v in [hd,legacy]]
                a,b,o=grouped_maps(rates,hm,mm,36,36,good&np.isfinite(legacy),floor=10)
                old=score(a,b,o,spatial=False,sigma=(.4,.4),zoom=(1,1));np.savez_compressed(out/'legacy_matched.npz',**old,unit_ids=ids,occupancy=o)
                report['variants']['legacy_matched']=summarize(old)
                if not (out/'legacy_matched_decoding.npz').exists():np.savez_compressed(out/'legacy_matched_decoding.npz',**decode(rates,hd,legacy,good,tau))
                hist,he=np.histogram((hd[good]-md[good]+180)%360-180,bins=np.arange(-180,181,10))
                # Independence baseline preserves each session's angular marginals.
                hcounts=np.bincount(labels(hd[good],np.linspace(0,360+1e-6,361)),minlength=360);mcounts=np.bincount(labels(md[good],np.linspace(0,360+1e-6,361)),minlength=360)
                diff=circerr(np.arange(360)[:,None],np.arange(360)[None,:]);ind=float(np.sum((hcounts[:,None]*mcounts[None,:])*(diff<=30))/good.sum()**2)
                np.savez_compressed(out/'behavior.npz',histogram=hist,edges=he,n_eligible=good.sum(),within30=np.sum(circerr(hd[good],md[good])<=30),independence_fraction=ind)
                # Bounded paired trajectory sample for inspection and reproducible QA.
                sample=np.flatnonzero(ok)[::10]
                np.savez_compressed(out/'trajectory_sample.npz',t=aligned[sample,0],x=aligned[sample,1],y=aligned[sample,2],hd=hd[sample],md=md[sample],legacy_md=legacy[sample],speed=speed[sample])
    report['seconds']=time.monotonic()-start;report['source_sha256']=sha(INPUT/(session+'.npz'))
    report['code_sha256']={str(p.name):sha(p) for p in [Path(__file__),REPO/'src/movement_direction.py',BASE/'scripts/finalization/variance_estimator.py']}
    (out/'complete.json').write_text(json.dumps(report,indent=2));print('DONE',session,round(report['seconds'],1),report['variants']['w200_s5'],flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        for kind,d,r in sessions():
            if kind=='dusz':run(d,r)
    print('ALL 30 SESSIONS COMPLETE',flush=True)
