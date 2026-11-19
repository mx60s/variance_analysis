from pathlib import Path
import json,sys
import numpy as np
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/md_revision';dirs=sorted((OUT/'sessions').glob('*'));assert len(dirs)==30 and all((p/'complete.json').exists() for p in dirs)
rows={}
names=[f'w{w}_s{s}' for w in [100,200,400] for s in [1,2,5,10]]+['w200_s5_b90','legacy_matched']
for name in names:
 zs=[np.load(p/(name+'.npz')) for p in dirs];ev=np.concatenate([z['ev'] for z in zs]);raw=np.concatenate([z['raw'] for z in zs]);total=np.concatenate([z['total'] for z in zs]);good=np.isfinite(ev).all(1)
 r={'n':len(ev),'n_finite':int(good.sum()),'median':np.median(ev[good],0).tolist(),'hd_greater':float(np.mean(ev[good,0]>ev[good,1])),'n_nonpositive_total':int(np.sum(total<=0)),'n_negative_numerator':int(np.sum((raw<0).any(1))),'n_sessions':len(dirs)}
 configs=[json.loads((p/'complete.json').read_text())['variants'][name] for p in dirs]
 if name!='legacy_matched':r.update(n_eligible=sum(x['n_eligible'] for x in configs),session_supported_bins=[x['supported_joint_bins'] for x in configs],session_median_direction_change=[x['median_md_change'] for x in configs])
 paths=[p/(name+'_decoding.npz') for p in dirs]
 if all(p.exists() for p in paths):
  decode={};zs=[np.load(p) for p in paths]
  # Both neural decoders and each baseline summarized on identical decodable times.
  for axis in ['hd','md']:
   errors=[];bases=[];sessions=[];n_eligible=0
   for z in zs:
    es=[];bs=[]
    for fold in range(4):
     np.testing.assert_array_equal(z[f'hd_fold{fold}_timebins'],z[f'md_fold{fold}_timebins'])
     mask=np.isfinite(z[f'hd_fold{fold}_errors'])&np.isfinite(z[f'md_fold{fold}_errors']);n_eligible+=len(mask)
     es.extend(z[f'{axis}_fold{fold}_errors'][mask]);bs.extend(z[f'{axis}_fold{fold}_baseline'][mask])
    errors.extend(es);bases.extend(bs);sessions.append([float(np.median(es)),float(np.median(bs))])
   decode[axis]={'median':float(np.median(errors)),'baseline':float(np.median(bases)),'session_medians':sessions,'n_decoded':len(errors),'n_eligible':n_eligible,'sessions_better':int(np.sum(np.array(sessions)[:,0]<np.array(sessions)[:,1]))}
  r['decoding']=decode
 rows[name]=r
beh=[np.load(p/'behavior.npz') for p in dirs];n=sum(int(z['n_eligible']) for z in beh)
behavior={'n_eligible':n,'within30_fraction':sum(int(z['within30']) for z in beh)/n,'independence_fraction':sum(int(z['n_eligible'])*float(z['independence_fraction']) for z in beh)/n,'angular_bin_edges':beh[0]['edges'].tolist(),'angular_histogram_counts':np.sum([z['histogram'] for z in beh],axis=0).tolist()}
summary={'primary':'w200_s5','variants':rows,'behavior':behavior,'policy':'Primary: centered local linear XY fit over 200 ms, speed >=5 cm/s, 36x36 joint grid; selected before inspecting results. All decoder comparisons use matched eligible and decodable time bins.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
for k,v in rows.items():
 print(k,v['n_finite'],v['median'],v['hd_greater'],{a:(d['median'],d['baseline']) for a,d in v.get('decoding',{}).items()})
print('Behavior',behavior['within30_fraction'],behavior['independence_fraction'])
