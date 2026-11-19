from pathlib import Path
root=Path('/Users/mvonebers/variance_analysis')
s=(root/'scripts/retraining/train.py').read_text()
s=s.replace('from scripts.retraining.trajectories import validate','from scripts.retraining.trajectories import validate\nfrom scripts.retraining.recipe_model import make_model')
s=s.replace('steps=200,','steps=12600,').replace('max_seconds=600,','max_seconds=5400,')
s=s.replace('data, mean, std, identity = load_data(data_root)','''data, measured_mean, measured_std, identity = load_data(data_root)
    mean = np.array([121.6697,149.3242,154.9510])/255
    std = np.array([40.7521,47.7267,103.2739])/255
    chunks = [(k,i) for k,e in enumerate(data['train']) for i in range(0,len(e)-9,10)]
    batches_per_epoch = (len(chunks)+63)//64
    if steps != 200*batches_per_epoch:
        raise ValueError('Recipe requires exactly 200 epochs')''')
s=s.replace('model = PredictiveCoder(3, 3, [2, 2], 10, causal_normalization=True).to(device)','model = make_model().to(device)')
s=s.replace('optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-6)','''optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.1,epochs=200,steps_per_epoch=batches_per_epoch)
    order = None''')
s=s.replace('optimizer.load_state_dict(ck["optimizer"])','''optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck['scheduler'])
        order = ck['epoch_order']
        if ck['steps_total'] != steps: raise ValueError('Schedule changed')''')
s=s.replace('squared = 0.0','squared = 0.0\n        normalized_squared = 0.0')
s=s.replace('squared += float(((pred - target) * std_t).square().sum())','squared += float(((pred - target) * std_t).square().sum())\n                normalized_squared += float((pred-target).square().sum())')
s=s.replace('"rgb_mse": squared / n,','"rgb_mse": squared / n,\n            "normalized_mse": normalized_squared / n,')
s=s.replace('initial = evaluate("validation", 128)','initial = evaluate("validation")')
s=s.replace('batch_size = 16','batch_size = 64')
s=s.replace('"optimizer": optimizer.state_dict(),','''"optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch_order": order,
            "steps_total": steps,
            "recipe": "authors SGD/OneCycle 200 epochs; causal normalization",
            "recipe_model_sha256": hashlib.sha256(Path(__import__('inspect').getfile(make_model)).read_bytes()).hexdigest(),''')
s=s.replace('"layers": [2, 2],','"layers": [2, 2, 2, 2],\n                "num_heads": 2,')
start=s.index('        windows = []\n',s.index('    while step < steps'))
end=s.index('        x = tensor(windows)',start)
s=s[:start]+'''        batch_index = step % batches_per_epoch
        if batch_index == 0:
            order = rng.permutation(len(chunks)).tolist()
        windows = [data['train'][chunks[j][0]][chunks[j][1]:chunks[j][1]+10]
                   for j in order[batch_index*64:(batch_index+1)*64]]
'''+s[end:]
s=s.replace('prediction = model(x[:, :-1])','prediction = model(x)[:, :-1]')
a=s.index('        gn = torch.nn.utils.clip_grad_norm_('); b=s.index('        step += 1',a)
s=s[:a]+'''        gn = torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None))
        if not torch.isfinite(gn): raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        scheduler.step()
'''+s[b:]
s=s.replace('if step % 100 == 0 or step == steps:', 'if step % batches_per_epoch == 0 or step == steps:')
s=s.replace('val = evaluate("validation", 128)','val = evaluate("validation")')
s=s.replace('val["rgb_mse"] < best','val["normalized_mse"] < best').replace('min(best, val["rgb_mse"])','min(best, val["normalized_mse"])')
s=s.replace('"step": step,\n                "training_normalized_mse"','"step": step,\n                "epoch": step // batches_per_epoch,\n                "learning_rate": optimizer.param_groups[0]["lr"],\n                "momentum": optimizer.param_groups[0]["momentum"],\n                "training_normalized_mse"')
s=s.replace('    save()\n    if not', '    save()\n    if step != steps: raise RuntimeError("Training budget ended; resume required before final evaluation")\n    if not')
s=s.replace('"model_code_sha256": best_ck["model_code_sha256"],','"model_code_sha256": best_ck["model_code_sha256"],\n            "recipe_model_sha256": best_ck["recipe_model_sha256"],')
s=s.replace('"objective": "normalized next-frame pixel MSE; AdamW lr=0.0003; batch=16; uniformly sampled episodes and 10-frame windows",','''"objective": "normalized next-frame pixel MSE; SGD momentum .9; OneCycle max_lr .1; batch64; 200 shuffled epochs; nonoverlapping 10-frame chunks",
        "epochs": step/batches_per_epoch,
        "training_chunks": len(chunks),
        "measured_training_mean": measured_mean.tolist(),
        "measured_training_std": measured_std.tolist(),
        "validation_selection": "full validation RGB-channel-normalized MSE, corrected averaging",
        "recipe": {"num_heads":2,"epochs":200,"batch_size":64,"max_lr":.1,"weight_decay":5e-6,"nesterov":False,"gradient_clipping":False,"rgb_statistics":"authors fixed values","dataset":"fresh smooth-v1"},''')
(root/'scripts/retraining/train_recipe.py').write_text(s)
(root/'scripts/retraining/recipe_model.py').write_text('''"""Authors' two-head configuration with per-frame causal normalization."""
from src.ann.predictive_coder import PredictiveCoder

def make_model(architecture=None):
    config = dict(architecture or dict(layers=[2,2,2,2],seq_len=10,causal_normalization=True,num_heads=2))
    heads = config.pop('num_heads')
    if heads != 2 or not config['causal_normalization']:
        raise ValueError('Expected causal two-head recipe')
    model = PredictiveCoder(3,3,**config)
    # Projection dimensions do not change; only the partition into heads does.
    for attention in [model.attention_1,model.attention_2]:
        attention.num_heads = heads
        attention.d_head = 128//heads
    return model
''')
p=Path('work/retraining/modal_train.py');s=p.read_text().replace('variance-causal-retraining','variance-paper-recipe').replace('timeout=3900','timeout=6000').replace('steps: int = 200','steps: int = 12600').replace('seconds: int = 600','seconds: int = 5400').replace('final: bool = False','final: bool = True').replace('causal-v1','paper-recipe-v1').replace('from scripts.retraining.train import run','from scripts.retraining.train_recipe import run');Path('work/paper_recipe/modal_train.py').write_text(s)
