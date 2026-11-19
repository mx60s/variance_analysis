"""Reproduce MD primary and sensitivity figures from packaged session results."""
import json
import numpy as np
import matplotlib.pyplot as plt

def build(root,out,scatter):
 s=json.loads((root/'outputs/md_revision/summary.json').read_text());v=s['variants'];p=v[s['primary']]
 ev=np.concatenate([np.load(f)['ev'] for f in sorted((root/'outputs/md_revision/sessions').glob('dusz_*/w200_s5.npz'))])
 fig,axs=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
 scatter(axs[0],ev[:,::-1],'a  HD versus MD','MD','HD')
 c=s['behavior'];e=np.array(c['angular_bin_edges']);h=np.array(c['angular_histogram_counts'])
 axs[1].bar((e[:-1]+e[1:])/2,h/h.sum()*100,width=10,color='#446e91')
 axs[1].set(xlabel='HD − MD (degrees)',ylabel='Time bins (%)',title='b  Behavioral coupling')
 for x in [-30,30]:axs[1].axvline(x,c='.4',ls='--')
 for i,axis in enumerate(['hd','md']):
  d=p['decoding'][axis];x=np.array([0,1])+i*2.6
  axs[2].bar(x,[d['median'],d['baseline']],color=['#0072b2','#e69f00'],width=.75)
 axs[2].set(xticks=[0,1,2.6,3.6],xticklabels=['HD\nNeural','HD\nOcc-based','MD\nNeural','MD\nOcc-based'],ylabel='Median error (degrees)',title='c  Matched decoding')
 axs[2].tick_params(axis='x',labelsize=8)
 for ext in ['pdf','png']:fig.savefig(out/('md_summary.'+ext),dpi=180)
 plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained');speeds=[1,2,5,10]
 for w,col in zip([100,200,400],['#0072b2','#d55e00','#009e73']):
  rows=[v[f'w{w}_s{x}'] for x in speeds]
  ax[0,0].plot(speeds,[r['median'][0] for r in rows],'o-',color=col,label=f'{w} ms: HD')
  ax[0,0].plot(speeds,[r['median'][1] for r in rows],'s--',color=col,label=f'{w} ms: MD')
  ax[0,1].plot(speeds,[100*r['hd_greater'] for r in rows],'o-',color=col,label=f'{w} ms')
 ax[0,0].set(title='a  Conditional response variance',ylabel='Median normalized score',xlabel='Minimum speed (cm/s)');ax[0,0].legend(fontsize=8,ncol=2)
 ax[0,1].set(title='b  HD preference',ylabel='Defined cells with HD > MD (%)',xlabel='Minimum speed (cm/s)',ylim=(97,100.3));ax[0,1].legend(fontsize=8)
 for axis,col in [('hd','#0072b2'),('md','#d55e00')]:
  ax[1,0].plot(speeds,[v[f'w200_s{x}']['decoding'][axis]['median'] for x in speeds],'o-',color=col,label=axis.upper())
 ax[1,0].set(title='c  Speed selection (200 ms)',ylabel='Median decoding error (degrees)',xlabel='Minimum speed (cm/s)');ax[1,0].legend()
 ax[1,1].plot([100,200,400],[v[f'w{x}_s5']['decoding']['md']['median'] for x in [100,200,400]],'o-',color='#d55e00',label='MD')
 ax[1,1].set(title='d  Temporal scale (≥5 cm/s)',ylabel='Median decoding error (degrees)',xlabel='Movement-estimation window (ms)')
 for a in ax.flat:a.spines[['right','top']].set_visible(False)
 for ext in ['pdf','png']:fig.savefig(out/('md_sensitivity.'+ext),dpi=180)
 plt.close(fig)
 return p
if __name__=='__main__':
 from pathlib import Path
 from figures import scatter
 root=Path(__file__).resolve().parents[2];build(root,root/'outputs/finalization/figures',scatter)
