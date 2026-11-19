import unittest
import numpy as np
from src.ann.fig import plot_neuron_2d_3d

class NeuronPlotTests(unittest.TestCase):
    def test_last_hd_bin_included_and_xy_not_stretched(self):
        x=np.zeros((1,3,3,90));x[:,:,:,-1]=23
        fig=plot_neuron_2d_3d(x,0,n_slices=4)
        slices=[t for t in fig.data if t.type=='heatmap'][1:]
        self.assertEqual(len(slices),4)
        np.testing.assert_array_equal(slices[-1].z,np.ones((3,3)))
        self.assertEqual(fig.layout.yaxis.scaleanchor,'x')
        self.assertTrue(any('360' in a.text for a in fig.layout.annotations))

if __name__=='__main__':unittest.main()
