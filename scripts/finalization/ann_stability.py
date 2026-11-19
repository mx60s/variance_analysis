"""Compare Luo primary classification with archived iterative alternatives."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def build():
 rows=[]
 for key in json.loads((ROOT/'provenance/luo_classification/summary.json').read_text()):
  m,l=key.split('/');p=ROOT/('outputs/paper_recipe/analysis' if m=='predictive' else 'outputs/finalization/cnns/'+m)/l
  ev=np.load(p/'ev.npz')['ev'];values=[]
  for base in [p, ROOT/'provenance/luo_classification/iterative_no_stability'/m/l, ROOT/'provenance/no_stability/stability_filtered'/m/l]:
   ids=json.loads((base/'classifications.json').read_text())['exclusive_place_cells_indices'];a=ev[ids];a=a[np.isfinite(a).all(1)];med=np.median(a,axis=0)
   values += [str(len(ids)),f'{med[0]:.2f}',f'{med[1]:.2f}']
  label={'predictive':'Predictive','resnet50':'ResNet','resnet50_untrained':'Untrained ResNet','vgg16':'VGG'}[m]+' '+l.replace('_',r'\_')
  rows.append(label+' & '+' & '.join(values)+' '+chr(92)*2)
 text=r'''\begin{table}[htbp]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrrr}
\hline
 & \multicolumn{3}{c}{Luo (primary)} & \multicolumn{3}{c}{Iterative fields} & \multicolumn{3}{c}{Iterative + stability} \\
Layer & $n$ & Location & HD & $n$ & Location & HD & $n$ & Location & HD \\\hline
'''+ '\n'.join(rows)+r'''
\hline
\end{tabular}
\caption{\textbf{Conditional response dependence under alternative ANN field criteria.} Counts are selected exclusive place-classified units (neither border nor HD). Location and HD columns give median normalized conditional variances over defined ratios. The primary detector uses Luo's single spatial-mean threshold and 10--144-bin fields. Iterative detection uses 15--144-bin fields and a relative peak criterion; the stability variant additionally requires within-field correlation $r\geq0.3$ between heading-group maps. Per-unit EV and all non-place criteria are held fixed.}
\label{tab:ann_stability}
\end{table}
'''
 out=ROOT/'outputs/finalization/figures/ann_stability_table.tex';out.write_text(text);return out
if __name__=='__main__':print(build())
