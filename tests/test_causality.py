import unittest
import torch
from src.ann.predictive_coder import PredictiveCoder


class Causality(unittest.TestCase):
    def test_new_training_architecture_does_not_use_future_frames(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        m = PredictiveCoder(3, 3, [2, 2], seq_len=4, causal_normalization=True)
        x = torch.randn(1, 4, 3, 64, 64)
        y = x.clone()
        y[:, 2:] += 3
        for train in [True, False]:
            m.train(train)
            with torch.no_grad():
                a = m(x)
                b = m(y)
            torch.testing.assert_close(a[:, :2], b[:, :2], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
