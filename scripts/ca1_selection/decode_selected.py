from pathlib import Path
import sys,os,json
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'scripts'),str(R)]
import regenerate_biology as b
OUT=Path(__file__).resolve().parents[2]/'outputs/ca1_selection'
for rat in range(1,5):
 dest=OUT/f'pfeiffer_0_{rat}';dest.mkdir(exist_ok=True)
 if (dest/'decoding.json').exists():continue
 s=json.loads((OUT/f'rat{rat}.json').read_text());pos,spikes,ids=b.load_pfeiffer_data(R/f'data/rodents/pfeiffer/Rat{rat}/Open1');assert np.array_equal(ids,s['ids']);ids=ids[np.asarray(s['selected'])]
 velocity=b.compute_velocity_series(pos);pos,_=b.coalesce_tracking_timestamps(pos);tau=.25;edges=np.arange(spikes[0,0],spikes[-1,0]+tau,tau);rates=b.build_spike_rates(spikes,edges,tau,ids);vel=b.average_in_bins(velocity[:,0],velocity[:,1],edges);aligned,_,valid=b.align_tracking(pos,edges)
 xe=ye=np.linspace(0,201+1e-6,101)
 b.decoding('pfeiffer',rates,aligned,None,valid,vel,tau,xe,ye,dest)
 (dest/'selection.json').write_text(json.dumps({'selected_cell_ids':ids.tolist(),'n':len(ids),'selection':'full-session peak spatial rate >1 Hz; no HD criterion','tuning_validation':'four held-out contiguous folds; original decoder settings retained'},indent=2)+'\n')
 print('RAT COMPLETE',rat,flush=True)
