import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ann.tanni_fields import secondary_filter
from src.ann.score_utils import _compute_single_heatmap_fields_info_new
class FieldTests(unittest.TestCase):
 def test_omission_does_not_require_defined_stability(self):
  field=np.ones((4,4));kw=dict(max_area_bins=16,stability_ratemaps=(field,field),stability_kwargs=dict(min_included_value=.1,min_bins=9))
  self.assertFalse(secondary_filter(field,min_stability=.3,**kw))
  self.assertTrue(secondary_filter(field,min_stability=None,**kw))
  kw['max_area_bins']=15
  self.assertFalse(secondary_filter(field,min_stability=None,**kw))
 def test_marginal_fields_do_not_require_heading_consistency(self):
  x,y=np.meshgrid(np.arange(17),np.arange(17),indexing='ij')
  a=np.exp(-((x-4)**2+(y-8)**2)/10).reshape(-1)
  b=np.exp(-((x-12)**2+(y-8)**2)/10).reshape(-1)
  acts=np.stack([a]*45+[b]*45,axis=1)
  args=(16,16,0,0,15,144)
  got=_compute_single_heatmap_fields_info_new(acts,*args,field_method='tanni_iterative_no_stability')
  shuffled=_compute_single_heatmap_fields_info_new(acts[:,np.random.default_rng(33).permutation(90)],*args,field_method='tanni_iterative_no_stability')
  self.assertGreater(int(got[0][0]),0)
  np.testing.assert_array_equal(got[0],shuffled[0])
  np.testing.assert_array_equal(sorted(got[1]),sorted(shuffled[1]))
if __name__=='__main__':unittest.main()
