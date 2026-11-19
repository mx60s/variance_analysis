"""Readable, seeded schematic of cross-fold conditional-curve correlations."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
rng=np.random.default_rng(42);grid=np.linspace(0,1,25);a,b=np.meshgrid(grid,grid,indexing='ij');truth=np.exp(-((b-.25-.4*a)**2)/.02)+.25*np.exp(-((b-.8)**2)/.008)
folds=[truth+rng.normal(0,.035,truth.shape) for _ in range(2)];centered=[x-x.mean(1,keepdims=True) for x in folds];normalized=[x/np.linalg.norm(x,axis=1,keepdims=True) for x in centered];C=normalized[0]@normalized[1].T;diag=np.trace(C)/len(C);off=C[~np.eye(len(C),dtype=bool)].mean()
plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(10,5.2),layout='constrained')
colors=['#0072b2','#d55e00','#009e73'];indices=[3,12,21]
for col,f in enumerate(folds):
 im=axs[0,col].imshow(f,origin='lower',extent=[0,1,0,1],cmap='jet',vmin=0,vmax=1.3,aspect='auto');axs[0,col].set(title=f'Joint tuning: fold {col}',xlabel='B',ylabel='A');fig.colorbar(im,ax=axs[0,col],label='Response',shrink=.8)
 for i,c in zip(indices,colors):axs[0,col].axhline(grid[i],c=c,lw=1.4,ls='--');axs[1,col].plot(grid,f[i],c=c,label=f'A = {grid[i]:.2f}')
 axs[1,col].set(title=f'Conditional curves: fold {col}',xlabel='B',ylabel='Response');axs[1,col].legend(fontsize=9,frameon=False)
im=axs[0,2].imshow(C,origin='lower',cmap='coolwarm',vmin=-1,vmax=1,aspect='auto');axs[0,2].plot([0,24],[0,24],c='black',ls='--',lw=1);axs[0,2].set(title='Cross-fold correlations',xlabel='j: A bin in fold 1',ylabel='i: A bin in fold 0');fig.colorbar(im,ax=axs[0,2],label='Pearson r',shrink=.8)
axs[1,2].axis('off');axs[1,2].text(.02,.94,r'$C_{ij}=\mathrm{corr}(f_i^{(0)},f_j^{(1)})$'+'\n\n'+r'Reliability: $\langle C_{ii}\rangle$'+f' = {diag:.2f}\n\n'+r'$\mathrm{EV}(A)=1-\langle C_{ij}\rangle_{i\ne j}$'+f'\n                 = {1-off:.2f}\n\nDifferent A values: off-diagonal\nSame A value: diagonal',transform=axs[1,2].transAxes,va='top',fontsize=11)
fig.savefig(out/'method_schematic.pdf',bbox_inches='tight');fig.savefig(out/'method_schematic.png',dpi=170,bbox_inches='tight');np.savez_compressed(out/'method_schematic_inputs.npz',fold0=folds[0],fold1=folds[1],correlations=C)
