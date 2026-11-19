"""Reproduce trajectory-replay validation from saved numerical trials."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=True,parents=True)
z=[np.load(p) for p in sorted((ROOT/'outputs/grid_primary').glob('dusz_*.npz'))];assert len(z)==30
coverage=json.loads((ROOT/'outputs/hd_support/audit.json').read_text())['summary'];coverage=[next(r for r in coverage if r['spatial_bins']==n and r['heading_bins']==90) for n in [10,20,30]]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,2,figsize=(10,7.5),layout='constrained');grids=[10,20,30]
ax[0,0].plot(grids,[100*r['support_fraction'] for r in coverage],'o-',c='#446e91');ax[0,0].set(title='a  Retained joint coverage',ylabel='Possible joint bins retained (%)',xlabel='Spatial bins per axis',xticks=grids,ylim=(0,25))
statistics={}
for a,inds,title in [(ax[0,1],[0,1,2],'b  Known HD-only responses'),(ax[1,0],[3,4],'c  Known place-only responses'),(ax[1,1],[5],'d  Known spatial gain × HD')]:
 est=[];noiseless=[];full=[]
 for n in grids:
  arr=np.concatenate([q[f'g{n}_estimate'][:,inds,:].reshape(-1,2) for q in z]);truth=np.concatenate([q[f'g{n}_noiseless'][inds,:] for q in z]);ref=np.concatenate([q[f'g{n}_full_grid'][inds,:] for q in z])
  est.append(np.median(arr,axis=0));noiseless.append(np.median(truth,axis=0));full.append(np.median(ref,axis=0))
 est,noiseless,full=np.array(est),np.array(noiseless),np.array(full)
 for j,col,name in [(0,'#e69f00','Location'),(1,'#0072b2','HD')]:
  a.plot(grids,est[:,j],'o-',c=col,label=name+' (Poisson)');a.plot(grids,noiseless[:,j],':',c=col,lw=2,label=name+' (noiseless)')
  a.plot(grids,full[:,j],'--',c=col,alpha=.65,label=name+' (full grid)')
 a.set(title=title,xlabel='Spatial bins per axis',ylabel='Median normalized variance',xticks=grids,ylim=(-.04,1.04))
 statistics[title]={'estimated':est.tolist(),'noiseless':noiseless.tolist(),'complete_grid':full.tolist()}
ax[0,1].legend(fontsize=7,ncol=2,loc='upper right');ax[1,1].legend(fontsize=7,ncol=2,loc='upper right')
for a in ax.flat:a.axvline(10,c='.7',lw=.7,zorder=-1)
for ext in ['pdf','png']:fig.savefig(OUT/('trajectory_recovery.'+ext),dpi=180,bbox_inches='tight')
(OUT/'trajectory_recovery_statistics.json').write_text(json.dumps(statistics,indent=2));plt.close(fig)
