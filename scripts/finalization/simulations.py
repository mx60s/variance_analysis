"""Seeded variance-estimator validation and within-bin sampling demonstration."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variance_estimator import components
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(f,name):
 f.savefig(OUT/(name+'.pdf'),bbox_inches='tight');f.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(f)
rng=np.random.default_rng(413)
x,y=np.meshgrid((np.arange(15)+.5)/15,(np.arange(15)+.5)/15,indexing='ij')
fig,axs=plt.subplots(2,3,figsize=(10,5.3),layout='constrained');validation={}
for j,deg in enumerate([0,25,45]):
 u=(x-.4)*np.cos(np.deg2rad(deg))-(y-.4)*np.sin(np.deg2rad(deg))
 f=np.exp(np.log(5)+(np.log(35)-np.log(5))*np.exp(-u*u/(2*.1**2)))
 truth=components(f,f); samples=[]
 for k in range(600):
  a,b=[rng.poisson(f*4)/4 for _ in range(2)]
  samples.append(components(a,b)['raw']/truth['total'])
 samples=np.array(samples)
 axs[0,j].imshow(f.T,origin='lower',extent=[0,1,0,1],cmap='viridis');axs[0,j].set(title=f'{deg}° ridge',xlabel='x',ylabel='y')
 for axis,color in enumerate(['#0072b2','#e69f00']):
  axs[1,j].hist(samples[:,axis],bins=30,alpha=.55,color=color,label=['x','y'][axis]);axs[1,j].axvline(truth['normalized'][axis],color=color,ls='--')
 axs[1,j].set(xlabel='Cross-fold variance / true total variance',ylabel='Datasets');axs[1,j].legend()
 validation[str(deg)]={'target':truth['normalized'].tolist(),'mean':samples.mean(0).tolist(),'sd':samples.std(0).tolist()}
save(fig,'estimator_validation')
# Sample-count sensitivity, raw numerator scaled by known total (no ratio bias).
f=5+25*np.exp(-((x-.4)/.12)**2/2);total=components(f,f)['total'];ns=[1,2,4,8,16,32,64]
samples=[]
for n in ns:
 samples.append(np.array([components(rng.poisson(f*n)/n,rng.poisson(f*n)/n)['raw']/total for k in range(400)]))
means=np.array([s.mean(0) for s in samples]);sd=np.array([s.std(0) for s in samples])
fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained')
for i,c in enumerate(['#0072b2','#e69f00']):
 ax.plot(ns,means[:,i],'o-',color=c,label=['Tuned x','Invariant y'][i]);ax.fill_between(ns,means[:,i]-sd[:,i],means[:,i]+sd[:,i],alpha=.18,color=c)
ax.axhline(0,color='.4',ls='--');ax.axhline(1,color='.4',ls='--');ax.set(xscale='log',xlabel='Observations per joint bin per fold',ylabel='Cross-fold variance / true total variance');ax.legend();save(fig,'noise_floor_samples')
# Isolate systematic within-bin sampling: complete occupancy, same observations
# per bin, and a known continuous rate depending ONLY on x.
def rate(x):return 5+25*np.exp(-.5*((x-.4)/.12)**2)
ngrid=20;centers=(np.arange(ngrid)+.5)/ngrid
cx=np.repeat(centers[:,None],ngrid,1)
offsets=np.random.default_rng(971).uniform(-.35,.35,(ngrid,ngrid))/ngrid
q=(np.arange(100)+.5)/100-.5
binned=rate(cx[:,:,None]+offsets[:,:,None]+q*.2/ngrid).mean(2)
discrete=rate(cx);smooth=gaussian_filter(binned,1,mode='nearest')
cont_var=float(np.var(rate((np.arange(10000)+.5)/10000)))
targets=np.array([components(a,a)['raw'][1]/cont_var for a in [discrete,binned,smooth]])
bias=[]
for k in range(600):
 a,b=[rng.poisson(binned*20)/20 for _ in range(2)]
 ad,bd=[rng.poisson(discrete*20)/20 for _ in range(2)]
 bias.append([components(ad,bd)['raw'][1]/cont_var,components(a,b)['raw'][1]/cont_var,components(gaussian_filter(a,1,mode='nearest'),gaussian_filter(b,1,mode='nearest'))['raw'][1]/cont_var])
bias=np.array(bias)
fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
axs[0].plot(np.linspace(0,1,500),rate(np.linspace(0,1,500)));axs[0].set(xlabel='Continuous x',ylabel='Expected rate',title='a  True response depends only on x')
im=axs[1].imshow((binned-discrete).T,origin='lower',cmap='coolwarm',vmin=-np.max(abs(binned-discrete)),vmax=np.max(abs(binned-discrete)),extent=[0,1,0,1]);axs[1].set(xlabel='x bin',ylabel='y bin',title='b  Within-bin sampling effect');fig.colorbar(im,ax=axs[1],shrink=.7,label='Rate difference')
for j,(label,c) in enumerate(zip(['Discrete control','Continuous, binned','Continuous, smoothed'],['#0072b2','#d55e00','#009e73'])):
 axs[2].hist(bias[:,j],bins=35,alpha=.45,color=c,label=label);axs[2].axvline(targets[j],color=c,ls='--')
axs[2].axvline(0,color='black',lw=.8);axs[2].set(xlabel='Estimated y variance / true total variance',ylabel='Datasets',title='c  Smoothing mitigates this artifact');axs[2].legend(fontsize=7)
save(fig,'binning_bias')
report={'validation':validation,'sample_counts':ns,'sampling_means':means.tolist(),'sampling_sd':sd.tolist(),'binning':{'targets':targets.tolist(),'means':bias.mean(0).tolist(),'sd':bias.std(0).tolist(),'repetitions':600,'bins':20,'observations_per_fold_bin':20,'smoothing_sigma':1,'target':'numerator divided by known continuous signal variance; true continuous y effect is zero'}}
(OUT/'variance_simulations.json').write_text(json.dumps(report,indent=2))
# Schematic describes operation rather than reusing correlation matrices.
fig,axs=plt.subplots(1,3,figsize=(10,2.8),layout='constrained')
f=rate(x)*(1+.6*np.sin(2*np.pi*y));center=f-f.mean(0,keepdims=True)
for ax,a,title in zip(axs[:2],[f,center],['a  Fold map: response f(x,y)','b  Subtract mean over x at each y']):
 im=ax.imshow(a.T,origin='lower',cmap='viridis' if ax is axs[0] else 'coolwarm');ax.set(title=title,xlabel='x bin',ylabel='y bin');fig.colorbar(im,ax=ax,shrink=.7)
axs[2].axis('off');axs[2].text(.03,.8,'c  Multiply centered maps across folds',fontsize=11);axs[2].text(.04,.55,r'$\widehat V_x=\langle(f_0-\bar f_{0,x})(f_1-\bar f_{1,x})\rangle$',fontsize=12);axs[2].text(.04,.27,'Average over supported joint bins.\nNormalize both axes by the same\ncross-fold total signal variance.',fontsize=10)
save(fig,'method_schematic')
print(json.dumps(report['binning'],indent=2))
