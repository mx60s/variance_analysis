"""Small mathematical audit; no training and no changes to original analysis.

Run with Python and NumPy. AST-load the repository's exact
get_splits/_corr_matrix/quantify_variance_corr functions, removing only the njit
decorator so this standalone audit does not require SciPy or Numba. Preprocessing
functions requiring SciPy are NOT run. Output is not regenerated paper data.
Undefined numeric results are serialized as JSON null.
"""
import ast
import json
import warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", message="Mean of empty slice")
tree = ast.parse((ROOT / "src/explained_variance.py").read_text())
keep = {"get_splits", "_corr_matrix", "quantify_variance_corr"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
for n in nodes:
    n.decorator_list = []
ns = {"np": np}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "repository_functions", "exec"), ns)
score = ns["quantify_variance_corr"]

def evaluate(a, b=None):
    if b is None:
        b = a.copy()
    out = {}
    for axis in [0, 1]:
        off, diag = score(a, b, np.ones(a.shape), axis=axis)
        out[str(axis)] = {"raw_EV": float(1-off), "reliability": float(diag)}
    return out

rng = np.random.default_rng(71623)
x = np.arange(1, 13, dtype=float)[:, None]
y = np.arange(1, 33, dtype=float)[None, :]
results = {"note": __doc__, "exact_counterexamples": {}}
maps = {"additive": x+y, "multiplicative": x*y,
        "pure_x": np.broadcast_to(x, (12, 32)).copy(),
        "pure_noise": None,
        "crossover_interaction": 20+(x-6.5)*np.sin(y)}
for name, a in maps.items():
    if name == "pure_noise":
        results["exact_counterexamples"][name] = evaluate(rng.normal(size=(12,32)), rng.normal(size=(12,32)))
    else:
        results["exact_counterexamples"][name] = evaluate(a)
results["pure_x_perturbation"] = {
    str(eps): evaluate(maps["pure_x"]+eps*rng.normal(size=(12,32)),
                       maps["pure_x"]+eps*rng.normal(size=(12,32)))
    for eps in [1e-4, 0.1]
}

# Invariant true slices, heteroskedastic independent observation noise.
# Population-across-bin formula is exact in the large-bin limit.
q = np.array([1.]*6+[.01]*6)
diag = q.mean()
off = ((np.sqrt(q).sum())**2-q.sum())/(len(q)*(len(q)-1))
g = rng.normal(size=(1, 4000))
sd = np.sqrt(1/q-1)[:,None]
a = g+sd*rng.normal(size=(12,4000))
b = g+sd*rng.normal(size=(12,4000))
off_emp, diag_emp = score(a,b,np.ones(a.shape),axis=0)
results["heterogeneous_reliability"] = {
    "q": q.tolist(), "theory_diag": float(diag), "theory_off": float(off),
    "theory_raw_EV": float(1-off), "theory_normalized_EV": float(1-off/diag),
    "empirical_diag": float(diag_emp), "empirical_off": float(off_emp),
    "empirical_normalized_EV": float(1-off_emp/diag_emp)}

# Odd occupancy and internal-NaN assumptions in get_splits.
v = np.array([1.,2.,3.,4.,5.]).reshape(1,1,5,1)
orig = v.copy()
splits = ns["get_splits"](v,seed=42)
results["get_splits"] = {"input_mutated": bool(not np.array_equal(v,orig)),
    "five_visits_reported_fold_occupancies": [int(splits[3][0,0]),int(splits[4][0,0])],
    "actual_fold_counts": [2,3]}
v = np.array([1.,np.nan,3.,4.]).reshape(1,1,4,1)
splits = ns["get_splits"](v,seed=42)
results["get_splits"]["internal_nan_reported_total_mean"] = float(splits[0][0,0,0])
results["get_splits"]["internal_nan_correct_total_mean"] = float(np.nanmean(v))

# True marginal of the GLM toy model under the manuscript's Gaussian design.
# E_y A exp(-(cx-sy)^2/(2 sigma^2))
# = A sigma/sqrt(sigma^2+s^2) exp(-c^2 x^2/(2(sigma^2+s^2))).
theta=np.pi/4; sigma=1.2; A=5.; c=np.cos(theta); s=np.sin(theta)
xx=np.array([0.,2.])
mx=A*sigma/np.sqrt(sigma**2+s**2)*np.exp(-c*c*xx**2/(2*(sigma**2+s**2)))
results["glm_true_marginal_theta45"] = {"x":xx.tolist(),"marginal":mx.tolist()}

# Inventory stored summaries and check possible complement / axis transformations.
cache=[]
for p in sorted((ROOT/'results/ann').glob('*/ev_all_neurons_pitch*.npz')):
    with np.load(p) as d:
        if not {'ev_v1','ev_v2'} <= set(d.files): continue
        a,b=np.asarray(d['ev_v1']),np.asarray(d['ev_v2'])
        good=np.isfinite(a)&np.isfinite(b)
        row={"path":str(p.relative_to(ROOT)),"keys":list(d.files),
            "n_total":int(a.size),"n_finite_pairs":int(good.sum()),
            "median_v1":float(np.median(a[good])),"median_v2":float(np.median(b[good])),
            "fraction_v1_gt_v2":float(np.mean(a[good]>b[good]))}
        if 'is_place' in d.files:
            mask=good & d['is_place'].astype(bool)
            row['n_place']=int(mask.sum())
            row['place_median_v1']=float(np.median(a[mask]))
            row['place_median_v2']=float(np.median(b[mask]))
        if p.name.endswith('_corr.npz'):
            other=p.with_name('ev_all_neurons_pitch.npz')
            if other.exists():
                with np.load(other) as o:
                    oa,ob=np.asarray(o['ev_v1']),np.asarray(o['ev_v2'])
                    ogood=np.isfinite(oa)&np.isfinite(ob)
                    if oa.shape!=a.shape and int(ogood.sum())==a.size:
                        oa,ob=oa[ogood],ob[ogood]
                        row['comparison_note']='Aligned original arrays by finite-pair mask; no unit IDs stored.'
                    if oa.shape==a.shape:
                        row['comparison_to_pitch_cache_max_abs_diff']={
                            "same":float(np.nanmax(np.abs(a-oa))),
                            "complement":float(np.nanmax(np.abs(a-(1-oa)))),
                            "complement_swapped":float(np.nanmax(np.abs(a-(1-ob)))),
                            "second_axis_complement_swapped":float(np.nanmax(np.abs(b-(1-oa))))}
        cache.append(row)
results['cached_summaries']=cache
def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

results = json_safe(results)
out=ROOT/'research_review/source_notes/method_diagnostics.json'
out.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in results.items() if k not in ['note','cached_summaries']},indent=2,allow_nan=False))
print(f'Wrote {out}; inspected {len(cache)} result archives')
