from pathlib import Path
import modal
ROOT=Path('/Users/mvonebers/variance_analysis');HERE=Path(__file__).resolve().parent
app=modal.App('variance-recipe-controls')
volume=modal.Volume.from_name('variance-minecraft-render',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.14.1','numpy==2.5.4','Pillow==13.0.0')
 .add_local_dir(ROOT/'src','/project/src',ignore=['**/__pycache__/**'])
 .add_local_dir(ROOT/'scripts/retraining','/project/scripts/retraining',ignore=['**/__pycache__/**']))
@app.function(image=image,gpu=['L4','A10'],cpu=4,memory=8192,timeout=600,max_containers=1,retries=0,volumes={'/persist':volume})
def controls():
 import sys
 sys.path.insert(0,'/project')
 from scripts.retraining.evaluate_recipe_controls import run
 result=run('/persist/smooth-v1','/persist/paper-recipe-v1/inference.pt','/persist/paper-recipe-v1/controls.json');volume.commit();return result
@app.local_entrypoint()
def main():
 import json
 (HERE/'control_result.json').write_text(json.dumps(controls.remote(),indent=2))
