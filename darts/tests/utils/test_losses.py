from darts.logging import get_logger

logger = get_logger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Loss tests will be skipped.")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    from darts.tests.base_test_class import DartsBaseTestClass
    from darts.utils.losses import MAELoss, MapeLoss, SmapeLoss

    class LossesTestCase(DartsBaseTestClass):
        x = torch.tensor([1.1, 2.2, 0.6345, -1.436])
        y = torch.tensor([1.5, 0.5])

        def helper_test_loss(self, exp_loss_val, exp_w_grad, loss_fn):
            W = torch.tensor([[0.1, -0.2, 0.3, -0.4], [-0.8, 0.7, -0.6, 0.5]])
            W.requires_grad = True
            y_hat = W @ self.x
            lval = loss_fn(y_hat, self.y)
            lval.backward()

            self.assertTrue(torch.allclose(lval, exp_loss_val, atol=1e-3))
            self.assertTrue(torch.allclose(W.grad, exp_w_grad, atol=1e-3))

        def test_smape_loss(self):
            exp_val = torch.tensor(0.7753)
            exp_grad = torch.tensor(
                [
                    [-0.2843, -0.5685, -0.1640, 0.3711],
                    [-0.5859, -1.1718, -0.3380, 0.7649],
                ]
            )
            self.helper_test_loss(exp_val, exp_grad, SmapeLoss())

        def test_mape_loss(self):
            exp_val = torch.tensor(1.2937)
            exp_grad = torch.tensor(
                [
                    [-0.3667, -0.7333, -0.2115, 0.4787],
                    [-1.1000, -2.2000, -0.6345, 1.4360],
                ]
            )
            self.helper_test_loss(exp_val, exp_grad, MapeLoss())

        def test_mae_loss(self):
            exp_val = torch.tensor(1.0020)
            exp_grad = torch.tensor(
                [
                    [-0.5500, -1.1000, -0.3173, 0.7180],
                    [-0.5500, -1.1000, -0.3173, 0.7180],
                ]
            )
            self.helper_test_loss(exp_val, exp_grad, MAELoss())
