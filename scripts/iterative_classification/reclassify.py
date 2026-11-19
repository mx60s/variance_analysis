"""Recompute iterative place fields from cached clean activation maps.

Usage: python scripts/iterative_classification/reclassify.py --activations clean.npy
       --baseline-chart old/neuron_chart.npy --output new [--rectify]
The baseline chart supplies active, border and HD scores; its single-threshold
field statistics must reproduce exactly before the place-field update proceeds.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ann.score_utils import _compute_single_heatmap_fields_info_new
from src.ann.classify_neurons import classify_neurons

def reclassify(acts,baseline,rectify=False,require_heading_stability=False):
 if acts.shape[:2]!=(289,90) or acts.shape[2]!=len(baseline):raise ValueError('Expected 17x17 locations by 90 headings, matching baseline units')
 chart=baseline.copy()
 for i in range(len(chart)):
  if not int(np.asarray(chart[i,0]).ravel()[0]):continue
  a=np.array(acts[:,:,i],copy=True)
  if rectify:np.maximum(a,0,out=a)
  args=(a,16,16,0,0,15,144)
  old=_compute_single_heatmap_fields_info_new(*args,field_method='adapted')
  for j,v in enumerate(old,1):np.testing.assert_allclose(np.asarray(v,dtype=float),np.asarray(baseline[i,j],dtype=float),equal_nan=True)
  new=_compute_single_heatmap_fields_info_new(*args,field_method='tanni_iterative' if require_heading_stability else 'tanni_iterative_no_stability')
  for j,v in enumerate(new,1):chart[i,j]=v
 return chart,classify_neurons(chart)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--activations',type=Path,required=True);p.add_argument('--baseline-chart',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rectify',action='store_true');p.add_argument('--require-heading-stability',action='store_true');a=p.parse_args()
 acts=np.load(a.activations,mmap_mode='r');base=np.load(a.baseline_chart,allow_pickle=True)
 chart,groups=reclassify(acts,base,a.rectify,a.require_heading_stability);a.output.mkdir(parents=True,exist_ok=False)
 np.save(a.output/'neuron_chart.npy',chart);(a.output/'classifications.json').write_text(json.dumps(groups,indent=2))
 print({k:len(v) for k,v in groups.items()})
