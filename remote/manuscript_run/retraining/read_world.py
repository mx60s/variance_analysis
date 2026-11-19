from pathlib import Path
import zlib,io,json
import nbtlib,numpy as np
root=Path('work/upstream/masters-thesis/environments/small-world')
level=nbtlib.load(root/'level.dat');print('version',level['Data'].get('Version'))
cache={}
def block(x,y,z):
 cx,cz=x//16,z//16;key=(cx,cz)
 if key not in cache:
  p=root/'region'/f'r.{cx//32}.{cz//32}.mca';b=p.read_bytes();i=4*((cx%32)+(cz%32)*32);off=int.from_bytes(b[i:i+3],'big')*4096;n=int.from_bytes(b[off:off+4],'big');typ=b[off+4];raw=b[off+5:off+4+n]
  data=nbtlib.File.parse(io.BytesIO(zlib.decompress(raw)));cache[key]=data['Level']
 for s in cache[key]['Sections']:
  if int(s['Y'])==y//16:
   return int(s['Blocks'][(y%16)*256+(z%16)*16+x%16])%256
 return 0
walk=np.zeros((17,17),bool)
for x in range(120,137):
 for z in range(197,214):walk[x-120,z-197]=block(x,3,z)!=0 and block(x,4,z)==0 and block(x,5,z)==0
print('walkable',walk.sum());print('\n'.join(''.join('.' if walk[x,z] else '#' for x in range(17)) for z in range(17)))
np.save('work/retraining/walkable.npy',walk)
print('blocktypes',sorted(set(block(x,y,z) for x in range(120,137) for z in range(197,214) for y in [3,4,5])))
