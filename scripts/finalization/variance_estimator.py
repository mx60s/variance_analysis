"""Cross-fold conditional response variance on fixed common support.

On a complete grid this is exactly the 69c8870 estimator divided by grid size.
Missing support defines a uniform distribution on the retained joint bins; each
conditional mean is weighted by that distribution. No product-grid ANOVA claim
is made for incomplete support. Normalized ratios are not unbiased estimators.
"""
import numpy as np

def components(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=2: raise ValueError('Matching 2-D maps required')
    valid=np.isfinite(a)&np.isfinite(b); count=int(valid.sum())
    if count<2:return dict(raw=np.full(2,np.nan),total=np.nan,normalized=np.full(2,np.nan),reliability=np.nan,support=count)
    a=np.where(valid,a,0.);b=np.where(valid,b,0.)
    ca=np.where(valid,a-a.sum()/count,0); cb=np.where(valid,b-b.sum()/count,0)
    total=float(np.sum(ca*cb)/count)
    energies=np.array([np.sum(ca*ca),np.sum(cb*cb)])/count
    raw=[]
    for axis in [0,1]:
        n=valid.sum(axis=axis,keepdims=True)
        ma=np.divide(a.sum(axis=axis,keepdims=True),n,out=np.zeros_like(n,dtype=float),where=n>0)
        mb=np.divide(b.sum(axis=axis,keepdims=True),n,out=np.zeros_like(n,dtype=float),where=n>0)
        raw.append(np.sum(np.where(valid,(a-ma)*(b-mb),0))/count)
    raw=np.array(raw)
    floor=1e-12*max(float(np.max(energies)),1e-30)
    norm=raw/total if total>floor else np.full(2,np.nan)
    reliability=total/np.sqrt(np.prod(energies)) if np.min(energies)>floor else np.nan
    return dict(raw=raw,total=total,normalized=norm,reliability=reliability,support=count)

def score(f0,f1,occ,spatial=True,sigma=(1,1),zoom=None,cutoff=True):
    import sys,time
    from pathlib import Path
    # Existing, audited smoothing and interpolation, held fixed for metric comparison.
    from src.explained_variance import smooth_location_hd,nan_aware_gaussian_filter,masked_zoom
    if f0.ndim==4:
        f0=f0.reshape(-1,*f0.shape[2:]);f1=f1.reshape(-1,*f1.shape[2:]);occ=occ.reshape(f0.shape[:2])
    if zoom is None:zoom=tuple(max(f0.shape[:2])/x for x in f0.shape[:2])
    oz=masked_zoom(occ,zoom); positive=oz[oz>0]
    minimum=float(np.percentile(positive/2,10)) if cutoff and len(positive) else 0
    mask=oz/2>minimum
    results=[]
    for n in range(f0.shape[-1]):
        if n%128==0: blocks=[np.array(f[:,:,n:n+128],copy=True) for f in [f0,f1]]
        maps=[smooth_location_hd(f[:,:,n%128],sigma) if spatial else nan_aware_gaussian_filter(f[:,:,n%128],sigma,(0,1)) for f in blocks]
        maps=[np.where(mask,masked_zoom(f,zoom),np.nan) for f in maps]
        results.append(components(*maps))
    d={k:np.array([r[k] for r in results]) for k in results[0]}
    # Compatibility for figure readers; the metric identity is explicit in every archive.
    d['ev']=d['normalized'];d['reliability']=np.repeat(d['reliability'][:,None],2,axis=1)
    d.update(min_occ=minimum,zoom=zoom,metric='cross_fold_conditional_response_variance',normalizer='cross_fold_total_signal_variance')
    return d

def summarize(d):
    ev=d['ev'];v=np.isfinite(ev).all(1)
    return dict(n=len(ev),n_finite=int(v.sum()),median=np.median(ev[v],0).tolist(),axis0_greater=float(np.mean(ev[v,0]>ev[v,1])) if v.any() else None,n_nonpositive_total=int(np.sum(d['total']<=0)),n_low_reliability=int(np.sum(d['reliability'][:,0]<.1)),n_negative_numerator=int(np.sum((d['raw']<0).any(1))),median_raw=np.nanmedian(d['raw'],0).tolist(),median_total=float(np.nanmedian(d['total'])))
