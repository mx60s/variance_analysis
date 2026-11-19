import unittest
import torch
from scripts.retraining.recipe_model import make_model
from scripts.retraining.analyze_recipe import outputs

class RecipeModelTests(unittest.TestCase):
    def test_two_head_causality_and_static_extraction(self):
        torch.set_num_threads(2)
        torch.manual_seed(918)
        m=make_model()
        self.assertEqual(m.attention_1.num_heads,2)
        x=torch.randn(2,10,3,64,64)
        xx=x.clone();xx[:,5:]+=3
        for training in [True,False]:
            m.train(training)
            with torch.no_grad():
                a=m(x);b=m(xx)
            torch.testing.assert_close(a[:,:5],b[:,:5],rtol=1e-5,atol=1e-5)
        m.eval()
        static=x[:,:1].expand(-1,10,-1,-1,-1).contiguous()
        with torch.no_grad():
            actual=outputs(m,static[:,0])
            for l in actual:
                expected=m.get_encoder_latents(static,l)[:,-1] if l in ['conv1','down1'] else m.get_latents(static,which=0 if l=='pre_pred' else -1)[:,-1]
                torch.testing.assert_close(actual[l],expected,rtol=2e-4,atol=2e-5)
        restored=make_model(dict(layers=[2,2,2,2],seq_len=10,causal_normalization=True,num_heads=2)).eval()
        restored.load_state_dict(m.state_dict(),strict=True)
        with torch.no_grad():torch.testing.assert_close(restored(x),m(x))

if __name__=='__main__':unittest.main()
