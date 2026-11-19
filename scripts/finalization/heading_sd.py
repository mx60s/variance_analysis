"""Mean and population SD across headings, with explicit biological coverage."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(base):
 out=base/'finalization';out.mkdir(exist_ok=True)
 pred=base/'paper_recipe/analysis/post_pred';z=np.load(pred/'gallery_maps.npz');sel=json.loads((pred/'gallery_selection.json').read_text());m=z['maps'][8]
 bio=out/'biology_examples/dusz_A3702-191126_13.npz';z=np.load(bio);b=z['map'];occ=z['occupancy'];coverage=(occ>0).sum(2)
 fig,axs=plt.subplots(2,4,figsize=(14,7.2),layout='constrained')
 for row,(a,title,unit,cov) in enumerate([(m,f'Predictive coder post-prediction unit {sel[8]["unit_id"]}','Activation',np.full(m.shape[:2],90)),(b,f'HD cell A3702-191126, unit {int(z["unit_id"])}','Hz',coverage)]):
  mean=np.nanmean(a,2);sd=np.nanstd(a,2,ddof=0)
  # Biological SD is descriptive over supported smoothed headings; mask locations with <2 raw headings.
  if row:mean[cov<2]=np.nan;sd[cov<2]=np.nan
  plots=[(mean,'Mean: full-map scale',float(np.nanmin(a)),float(np.nanmax(a)),unit),(mean,'Mean: its own scale',float(np.nanmin(mean)),float(np.nanmax(mean)),unit),(sd,'SD across headings',0,float(np.nanmax(sd)),unit),(cov,'Sampled heading bins',0,90,'Bins / 90')]
  for col,(v,t,lo,hi,label) in enumerate(plots):
   ax=axs[row,col];im=ax.imshow(v.T,origin='lower',cmap='jet',vmin=lo,vmax=hi);ax.set(title=t,xlabel='X bin',ylabel='Z bin' if row==0 else 'Y bin');fig.colorbar(im,ax=ax,shrink=.7,label=label)
  axs[row,0].text(0,1.23,title,transform=axs[row,0].transAxes,fontsize=12,weight='bold')
 fig.savefig(out/'heading_sd.png',dpi=170);fig.savefig(out/'heading_sd.pdf');plt.close(fig)
 np.savez_compressed(out/'heading_sd_inputs.npz',model_map=m,biological_map=b,biological_occupancy=occ)
 (out/'heading_sd_notes.txt').write_text('SD is descriptive population SD across headings (ddof=0), not standard error. Model: all 90 headings equally weighted, raw signed activations, first randomly sampled exclusive place-classified gallery unit. Biology: equal weighting of finite smoothed headings, raw coverage shown; locations with fewer than two observed headings omitted. Missing directions are not treated as zero. Biological SD is coverage-dependent and is not a quantitative cross-population comparison. Mean and SD have the original response units. Full-map mean preserves original color limits; adjacent mean uses its own explicitly labeled limits.\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);main(p.parse_args().outputs)
