"""Known-trajectory test; synthetic noise levels are illustrative, not calibrated."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.integrate import cumulative_trapezoid
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.movement_direction import estimate_movement
from src.rodent_data import get_md_from_xy
rng=np.random.default_rng(1309);t=np.arange(0,120,.01)
heading=.6*np.sin(.2*t)+.3*np.sin(.8*t)
rows=[]
for speed in [2,5,10,20]:
 vx=speed*np.sin(heading);vy=speed*np.cos(heading)
 x=cumulative_trapezoid(vx,t,initial=0);y=cumulative_trapezoid(vy,t,initial=0)
 truth=np.degrees(heading)%360
 for noise in [.05,.2,.5]:
  pos=np.c_[t,x+rng.normal(0,noise,len(t)),y+rng.normal(0,noise,len(t))]
  legacy=(get_md_from_xy(pos,20)+90)%360
  for window in [.1,.2,.4]:
   r=estimate_movement(pos,window);v=r['valid'];err=lambda a:np.abs((a[v]-truth[v]+180)%360-180)
   rows.append(dict(speed=speed,position_noise_sd_cm=noise,window_s=window,n=int(v.sum()),legacy_median_error=float(np.median(err(legacy))),new_median_error=float(np.median(err(r['direction']))),new_p95_error=float(np.percentile(err(r['direction']),95))))
assert all(r['new_median_error']<r['legacy_median_error'] for r in rows if r['window_s']==.2)
out=Path('outputs/md_revision');(out/'synthetic_validation.json').write_text(json.dumps({'note':'Smooth synthetic trajectories with known heading; independent Gaussian XY noise; not a measurement of actual tracking error.','conditions':rows},indent=2))
print('PASS: 200-ms estimate improves median error in all 12 synthetic speed/noise conditions')
for r in rows:
 if r['position_noise_sd_cm']==.2 and r['window_s']==.2:print(r)
