from pathlib import Path
import json
import numpy as np
here=Path('work/paper_recipe'); paths=here/'panorama_paths';paths.mkdir(exist_ok=True)
episodes=[]
for i,(x,z) in enumerate(( (x,z) for x in np.arange(120.5,137,1) for z in np.arange(197.5,214,1))):
 poses=np.array([(x,z,yaw,pitch) for yaw in range(0,360,4) for pitch in [0,-3,-1,1,3]],float)
 id=f'loc_{x:+.1f}_{z:+.1f}'
 np.save(paths/(id+'.npy'),poses)
 episodes.append(dict(id=id,location=[x,z]))
(paths/'manifest.json').write_text(json.dumps(dict(episodes=episodes,kind='panoramas',pitch_degrees=[0,-3,-1,1,3],yaw_step_degrees=4),indent=2))
root=Path('/Users/mvonebers/variance_analysis')
s=(root/'scripts/retraining/collect_malmo.py').read_text().replace('for index, (x, z, yaw) in enumerate(poses):','for index, (x, z, yaw, pitch) in enumerate(poses):').replace('host.sendCommand(f"setYaw {yaw:.9f}")','host.sendCommand(f"setYaw {yaw:.9f}")\n                host.sendCommand(f"setPitch {pitch:.9f}")').replace('and abs(frame.pitch) < 0.002','and abs(frame.pitch-pitch) < 0.002').replace('rendered.append([frame.xPos, frame.zPos, frame.yaw])','rendered.append([frame.xPos, frame.zPos, frame.yaw, frame.pitch])').replace('camera_pitch=0.0','camera_pitch=[0,-3,-1,1,3]').replace('simulation_time=np.arange(len(poses)) * 0.1,','yaw=np.array(rendered)[:,2],\n                pitch=np.array(rendered)[:,3],').replace('Verified smooth heading-aligned trajectories','Verified static panorama grid')
(root/'scripts/retraining/collect_panoramas.py').write_text(s)
s=Path('work/retraining/modal_collect_parallel.py').read_text().replace('HERE.parent / "upstream','HERE.parent / "upstream').replace('HERE / "paths_v2"','HERE / "panorama_paths"').replace('collect_malmo.py','collect_panoramas.py').replace('variance-minecraft-collect-parallel','variance-matched-panoramas').replace('timeout=3600','timeout=7200').replace('/persist/smooth-v1','/persist/panoramas-centers-v1')
a=s.index('    base = json.loads',s.index('def aggregate'))
b=s.index('    base["shards"]',a)
s=s[:a]+'''    base = None
    manifests = []
    for start in range(0,289,37):
        p = root / 'shards' / str(start) / 'manifest.json'
        m = json.loads(p.read_text())
        if not m['complete'] or len(m['episodes']) != min(37,289-start):
            raise ValueError('Incomplete shard')
        if base is None: base = dict(m,episodes=[])
        if m['world_files_sha256'] != base['world_files_sha256']: raise ValueError('World mismatch')
        for ep in m['episodes']:
            ep['file'] = 'shards/'+str(start)+'/'+ep['id']+'.npz'
            path = root / ep['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != ep['sha256']: raise ValueError('Hash mismatch')
            import numpy as np
            with np.load(path) as f:
                if f['rgb'].shape != (450,64,64,3): raise ValueError('Wrong shape')
                delta=f['state']-f['requested_state']
                delta[:,2]=(delta[:,2]+180)%360-180
                if abs(delta).max()>.002: raise ValueError('Wrong pose')
            base['episodes'].append(ep)
        manifests.append(dict(file=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if len(set(e['id'] for e in base['episodes'])) != 289: raise ValueError('Duplicates')
'''+s[b:]
s=s.replace('[(s, 60) for s in range(20, 500, 60)]','[(s, min(37,289-s)) for s in range(0,289,37)]')
(here/'modal_panoramas.py').write_text(s)
