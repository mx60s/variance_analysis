"""Regenerate supplementary ANN stability comparison from classified unit IDs."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def build():
 summary=json.loads((ROOT/'provenance/no_stability/summary.json').read_text());rows=[]
 for key in summary:
  m,l=key.split('/');p=ROOT/('outputs/paper_recipe/analysis' if m=='predictive' else 'outputs/finalization/cnns/'+m)/l
  ev=np.load(p/'ev.npz')['ev'];values=[]
  for base,kind in [(p,'no_stability'),(ROOT/'provenance/no_stability/stability_filtered'/m/l,'stability_filtered')]:
   ids=json.loads((base/'classifications.json').read_text())['exclusive_place_cells_indices'];a=ev[ids];a=a[np.isfinite(a).all(1)];med=np.median(a,axis=0)
   assert len(ids)==summary[key][kind]['classified'];np.testing.assert_allclose(med,summary[key][kind]['median'])
   values += [f'{len(a)}/{len(ids)}',f'{med[0]:.2f}',f'{med[1]:.2f}']
  label={'predictive':'Predictive','resnet50':'ResNet','resnet50_untrained':'Untrained ResNet','vgg16':'VGG'}[m]+' '+l.replace('_',r'\_')
  rows.append(label+' & '+' & '.join(values)+r' \\')
 text=r'''\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr}
\hline
 & \multicolumn{3}{c}{Field criteria (primary)} & \multicolumn{3}{c}{With heading-group stability} \\
Layer & Defined/selected & Location & HD & Defined/selected & Location & HD \\\hline
'''+ '\n'.join(rows)+r'''
\hline
\end{tabular}
\caption{\textbf{Heading-group stability does not eliminate heading dependence in ANN place-classified units.} Both analyses use the same iterative field detector, size and peak thresholds, sampled units, and EV estimates. The supplementary selection additionally requires within-field correlation $r\geq0.3$ between maps averaged over two heading groups. Counts give defined EV ratios over all selected exclusive place-classified units (neither border nor HD); location and HD columns give median normalized conditional variances over defined ratios.}
\label{tab:ann_stability}
\end{table}
'''
 out=ROOT/'outputs/finalization/figures/ann_stability_table.tex';out.write_text(text);return out
if __name__=='__main__':print(build())
