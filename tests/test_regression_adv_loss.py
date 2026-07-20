"""
Tests for the regression-based adversarial generator loss option in GANLoss.
"""
import numpy as np
import torch
from audiotools import AudioSignal

from dac.nn.loss import GANLoss
from dac.nn.regression_adv_loss import regression_generator_loss


class _AlternatingDiscriminator(torch.nn.Module):
    """Stand-in for ``dac.model.Discriminator``.

    ``GANLoss.forward`` calls the discriminator once on the fake batch and once on the real
    batch. This module returns preset logits, alternating per call so the first call yields the
    fake logit and the second the real logit. Each call returns a single
    sub-discriminator with one final-output tensor (no intermediate features), matching the
    ``list[list[Tensor]]`` contract used by ``GANLoss``.
    """

    def __init__(self, fake_logit, real_logit):
        super().__init__()
        self.register_buffer("fake_logit", fake_logit)
        self.register_buffer("real_logit", real_logit)
        self._call = 0

    def forward(self, x):
        self._call += 1
        logit = self.fake_logit if self._call % 2 == 1 else self.real_logit
        return [[logit]]


def test_regression_generator_loss_matches_mse_to_real():
    # The regression loss is MSE(D(real), D(fake)) summed over sub-discriminators:
    # the fixed-target "1" is replaced by the empirical D(real).
    fake_logit = torch.tensor([0.1, 0.4, 0.9])
    real_logit = torch.tensor([0.8, 0.2, 0.6])

    d_fake = [[fake_logit]]
    d_real = [[real_logit]]

    expected = torch.mean((real_logit - fake_logit) ** 2)
    assert torch.allclose(regression_generator_loss(d_fake, d_real), expected)


def test_generator_loss_regression_branch_is_wired():
    # Exercises the real GANLoss path (the call site in dac.nn.loss) with the
    # "regression" option enabled, proving the wiring edit invokes the new module.
    fake_logit = torch.tensor([0.1, 0.4, 0.9])
    real_logit = torch.tensor([0.8, 0.2, 0.6])
    disc = _AlternatingDiscriminator(fake_logit, real_logit)

    fake = AudioSignal(np.zeros(8, dtype=np.float32), 44_100)
    real = AudioSignal(np.ones(8, dtype=np.float32), 44_100)

    gan = GANLoss(disc, generator_loss_type="regression")
    loss_g, loss_feature = gan.generator_loss(fake, real)

    expected = torch.mean((real_logit - fake_logit) ** 2)
    assert torch.allclose(loss_g, expected)
    # No intermediate features in the stand-in, so the feature-matching term is 0.
    assert loss_feature == 0


def test_generator_loss_hinge_default_is_unchanged():
    # Default construction preserves the original least-squares behavior: regress D(fake) -> 1.
    fake_logit = torch.tensor([0.1, 0.4, 0.9])
    real_logit = torch.tensor([0.8, 0.2, 0.6])
    disc = _AlternatingDiscriminator(fake_logit, real_logit)

    fake = AudioSignal(np.zeros(8, dtype=np.float32), 44_100)
    real = AudioSignal(np.ones(8, dtype=np.float32), 44_100)

    gan = GANLoss(disc)
    loss_g, _ = gan.generator_loss(fake, real)

    expected = torch.mean((1 - fake_logit) ** 2)
    assert torch.allclose(loss_g, expected)
