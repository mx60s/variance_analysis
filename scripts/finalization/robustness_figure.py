from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
if (ROOT/'sensitivity.json').exists():DATA=ROOT/'sensitivity.json'
else:DATA=ROOT/'outputs/variance_version/sensitivity.json'
s=json.loads(DATA.read_text());OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(8,6),layout='constrained')
for col,(kind,mid,label,axis) in enumerate([('pfeiffer','ca1_native','CA1',0),('dusz','hd_native','HD cells',1)]):
 rows=[s['grid_10_'+kind],s[mid],s['grid_30_'+kind]];n=[10,20,30]
 pref=[100*(r['fraction_axis0_greater'] if axis==0 else 1-r['fraction_axis0_greater']) for r in rows]
 axs[0,col].plot(n,pref,'o-',c='#0072b2');axs[0,col].axhline(50,c='.6',ls='--');axs[0,col].set(ylim=(45,105),xticks=n,ylabel='Cells favoring characteristic variable (%)',title=label,xlabel='Spatial bins per axis')
 for a,c,l in [(0,'#e69f00','Location'),(1,'#0072b2','HD')]:axs[1,col].plot(n,[r['median'][a] for r in rows],'o-',color=c,label=l)
 axs[1,col].set(ylim=(0,1),xticks=n,xlabel='Spatial bins per axis',ylabel='Median normalized variance');axs[1,col].legend()
for ext in ['pdf','png']:fig.savefig(OUT/('binning_sensitivity.'+ext),bbox_inches='tight',dpi=150)
plt.close(fig)
