from pathlib import Path
import modal
ROOT=Path('/Users/mvonebers/variance_analysis');HERE=Path(__file__).resolve().parent
volume=modal.Volume.from_name('variance-minecraft-render',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.14.1','numpy==2.5.4','scipy==1.18.1','matplotlib==3.11.2','threadpoolctl==3.7.0','plotly==5.24.1','opencv-python-headless','tqdm','numba==0.68.0','Pillow==13.0.0')
 .env({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLBACKEND':'Agg'})
 .add_local_dir(ROOT/'src','/project/src',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'scripts','/project/scripts',ignore=['**/__pycache__/**'])
 .add_local_dir(HERE/'provenance','/project/results/review_20261116/predictive_coder',ignore=['*.npz','*.npy','*.png','*.pdf','**/before/**']))
app=modal.App('variance-paper-recipe-analysis')
@app.function(image=image,gpu=['L4','A10'],cpu=4,memory=16384,timeout=3600,max_containers=1,retries=0,volumes={'/persist':volume})
def extract():
 import sys
 sys.path.insert(0,'/project')
 from scripts.retraining.analyze_recipe import extract
 return extract('/persist/paper-recipe-v1/inference.pt','/persist/panoramas-centers-v1','/persist/paper-recipe-analysis-v1',volume.commit)
@app.function(image=image,cpu=4,memory=16384,timeout=7200,max_containers=4,retries=0,volumes={'/persist':volume})
def analyze(layer):
 import sys
 sys.path.insert(0,'/project')
 from scripts.retraining.analyze_recipe import analyze_layer
 return analyze_layer('/persist/paper-recipe-analysis-v1',layer,volume.commit)
@app.function(image=image,cpu=1,memory=2048,timeout=300,volumes={'/persist':volume})
def package():
 import zipfile,json
 root=Path('/persist/paper-recipe-analysis-v1')
 with zipfile.ZipFile('/persist/paper-recipe-analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
  z.write(root/'extraction.json','extraction.json')
  for layer in ['conv1','down1','pre_pred','post_pred']:
   for name in ['ev.npz','ev_native.npz','gallery_maps.npz','unit_ids.npy','locations.npy','gallery_selection.json','analysis_summary.json','classification.png','classifications.json','summary.json','neuron_chart.npy']:
    z.write(root/layer/name,layer+'/'+name)
 import tarfile,hashlib
 pano=Path('/persist/panoramas-centers-v1')
 manifest=json.loads((pano/'manifest.json').read_text())
 archive=Path('/persist/panoramas-centers-v1.tar')
 with tarfile.open(archive,'w') as tar:
  tar.add(pano/'manifest.json',arcname='panoramas-centers-v1/manifest.json')
  for ep in manifest['episodes']:tar.add(pano/ep['file'],arcname='panoramas-centers-v1/'+ep['file'])
 with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 (pano/'archive.json').write_text(json.dumps(dict(sha256=digest,bytes=archive.stat().st_size,images=130050,locations=289)))
 volume.commit()
 return '/persist/paper-recipe-analysis.zip'
@app.function(image=image,cpu=2,memory=4096,timeout=300,max_containers=1,retries=0)
def preflight():
 import sys,numpy as np,torch
 sys.path.insert(0,'/project');sys.path.insert(0,'/project/scripts')
 from scripts.retraining.recipe_model import make_model
 from scripts.retraining.analyze_recipe import outputs
 from regenerate_ev import score
 from reclassify import classify
 torch.set_num_threads(2)
 model=make_model().eval()
 with torch.no_grad():a=outputs(model,torch.zeros(1,3,64,64))
 folds=np.random.default_rng(1).random((9,90,3)).astype('float32')
 result=score(folds,folds,np.ones((9,90))*4,zoom=(1,1),cutoff=False)
 assert result['ev'].shape==(3,2)
 target=Path('/tmp/classification-probe');target.mkdir()
 np.savez(target/'input.npz',activations=folds,locations=np.array([(x,y) for x in range(3) for y in range(3)]))
 classify(target/'input.npz','predictive_coder','pre_pred',target)
 return {'layers':{k:list(v.shape) for k,v in a.items()},'ev_shape':list(result['ev'].shape),'imports':'passed'}

@app.local_entrypoint()
def main(preflight_only: bool=False):
 import json
 if preflight_only:
  print(preflight.remote());return
 (HERE/'extraction_result.json').write_text(json.dumps(extract.remote(),indent=2))
 results=list(analyze.map(['conv1','down1','pre_pred','post_pred']))
 (HERE/'analysis_result.json').write_text(json.dumps(results,indent=2))
 print(package.remote())
