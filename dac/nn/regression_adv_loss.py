"""Regression-based adversarial generator loss.

Adapted from MCGAN, which reformulates the GAN generator's adversarial loss as
a regression task: instead of pushing the discriminator's output on generated
samples toward a fixed target (the usual least-squares GAN target of 1), the
generator minimizes the mean squared error between the discriminator's output on
real data and on generated data. Regressing toward the empirical real target
provides stronger, data-driven supervision that dampens the oscillation of the
standard adversarial generator loss.

Reference
---------

Baoren Xiao, Hao Ni, Weixin Yang.
"MCGAN: Enhancing GAN Training with Regression-Based Generator Loss."
arXiv:2405.17191, 2024. https://arxiv.org/abs/2405.17191

Adaptation note
---------------

The paper estimates the expected discriminator output on fake data by Monte-Carlo
sampling several generated examples per update. The DAC training loop produces a
single generated batch per step, which is itself a Monte-Carlo draw from the
fake distribution; the per-batch element-wise MSE below is the single-sample
Monte-Carlo estimate of the regression loss, and accumulating it across SGD
steps recovers the multi-sample estimate. The paper's separate evaluation
protocol is therefore intentionally out of scope here.
"""

import torch


def regression_generator_loss(d_fake, d_real):
    """MCGAN regression generator loss.

    For each sub-discriminator, minimizes the mean squared error between the
    discriminator's final output on real samples and on generated samples, summed
    across sub-discriminators. This is the regression-toward-real analogue of the
    fixed-target least-squares generator loss (``mean((1 - D(fake)) ** 2)``):
    the constant target ``1`` is replaced by the empirical ``D(real)``.

    Parameters
    ----------
    d_fake : list[list[torch.Tensor]]
        Discriminator outputs on generated data, as returned by
        ``GANLoss.forward`` (one list of feature maps plus a final logit per
        sub-discriminator; the final logit is the last element of each inner
        list).
    d_real : list[list[torch.Tensor]]
        Discriminator outputs on real data, same structure as ``d_fake``.

    Returns
    -------
    torch.Tensor
        Scalar regression loss to use as the generator's adversarial term.
    """
    loss = 0
    for x_fake, x_real in zip(d_fake, d_real):
        loss += torch.mean((x_real[-1] - x_fake[-1]) ** 2)
    return loss
