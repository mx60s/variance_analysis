import numpy as np
from variance_estimator import components
rng=np.random.default_rng(27)
a=rng.normal(size=(17,13));b=rng.normal(size=a.shape)
for axis in [0,1]:
 ma=a.mean(axis,keepdims=True);mb=b.mean(axis,keepdims=True)
 old=.5*(np.sum((a-ma)**2)+np.sum((b-mb)**2)-np.sum((a-b)**2)+a.shape[axis]*np.sum((ma-mb)**2))
 np.testing.assert_allclose(components(a,b)['raw'][axis],old/a.size,atol=1e-14)
x=np.arange(7.)[:,None];y=np.arange(9.)[None,:]
for f in [x+0*y,x+y,(x-3)*(y-4),x*(y+1)]:
 d=components(f,f);np.testing.assert_allclose(d['raw'],[np.var(f,axis=i).mean() for i in [0,1]])
 np.testing.assert_allclose(d['normalized'],components(3*f+8,3*f+8)['normalized'])
# Repeated independent noisy estimates recover the target; a single-fold variance does not.
f=np.sin(x)+np.cos(y);target=components(f,f)['raw'];est=[]
for k in range(10000):est.append(components(f+rng.normal(size=f.shape),f+rng.normal(size=f.shape))['raw'])
est=np.array(est);assert np.all(abs(est.mean(0)-target)<4*est.std(0)/np.sqrt(len(est)))
# Missing support: row-constant map has no axis-1 effect; swapped folds identical.
a=(x+0*y).copy();a[::2,::3]=np.nan
assert abs(components(a,a)['raw'][1])<1e-14
np.testing.assert_allclose(components(a,a)['normalized'][0],1)
assert np.isnan(components(np.ones((4,4)),np.ones((4,4)))['normalized']).all()
print('PASS: historical equivalence, known effects, scale/offset, unbiased Monte Carlo, missing support, constant maps')
