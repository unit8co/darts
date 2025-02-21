import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import torch

from darts.utils.losses import MAELoss, MapeLoss, SmapeLoss


class TestLosses:
    x = torch.tensor([1.1, 2.2, 0.6345, -1.436])
    y = torch.tensor([1.5, 0.5])

    def helper_test_loss(self, exp_loss_val, exp_w_grad, loss_fn):
        W = torch.tensor([[0.1, -0.2, 0.3, -0.4], [-0.8, 0.7, -0.6, 0.5]])
        W.requires_grad = True
        y_hat = W @ self.x
        lval = loss_fn(y_hat, self.y)
        lval.backward()

        assert torch.allclose(lval, exp_loss_val, atol=1e-3)
        assert torch.allclose(W.grad, exp_w_grad, atol=1e-3)

    def test_smape_loss(self):
        exp_val = torch.tensor(0.7753)
        exp_grad = torch.tensor([
            [-0.2843, -0.5685, -0.1640, 0.3711],
            [-0.5859, -1.1718, -0.3380, 0.7649],
        ])
        self.helper_test_loss(exp_val, exp_grad, SmapeLoss())

    def test_mape_loss(self):
        exp_val = torch.tensor(1.2937)
        exp_grad = torch.tensor([
            [-0.3667, -0.7333, -0.2115, 0.4787],
            [-1.1000, -2.2000, -0.6345, 1.4360],
        ])
        self.helper_test_loss(exp_val, exp_grad, MapeLoss())

    def test_mae_loss(self):
        exp_val = torch.tensor(1.0020)
        exp_grad = torch.tensor([
            [-0.5500, -1.1000, -0.3173, 0.7180],
            [-0.5500, -1.1000, -0.3173, 0.7180],
        ])
        self.helper_test_loss(exp_val, exp_grad, MAELoss())
