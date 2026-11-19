import unittest
import numpy as np
from src.ann.score_utils import _compute_single_heatmap_fields_info_old, _compute_single_heatmap_fields_info_new

class LuoFields(unittest.TestCase):
    def test_diagonal_components_use_eight_neighbors(self):
        h=np.zeros((17,17));h[2:3,2:7]=1;h[3:4,7:12]=1
        fields=_compute_single_heatmap_fields_info_old(h,10,144)
        self.assertEqual(fields[0][0],1)
        self.assertEqual(fields[1][0],10)

    def test_single_threshold_does_not_search_for_subfields(self):
        h=np.zeros((17,17));h[1:16,1:16]=1;h[5:12,5:12]=2
        acts=np.repeat(h.reshape(-1,1),90,axis=1)
        luo=_compute_single_heatmap_fields_info_new(acts,16,16,0,0,15,144,field_method='luo')
        iterative=_compute_single_heatmap_fields_info_new(acts,16,16,0,0,15,144,field_method='tanni_iterative_no_stability')
        self.assertEqual(luo[0][0],0)
        self.assertGreater(iterative[0][0],0)
