"""
Distributional matching objectives for vector quantization.

Adapted from "Distributional Matching for Vector Quantization: A Unified
Theoretical and Empirical Framework" (https://arxiv.org/abs/2607.15933v1).

The standard VQ-VQGAN commitment / codebook losses are pointwise MSEs made
trainable through the straight-through estimator (STE): one detaches the
codes (the *commitment* loss, trains the encoder) and the other detaches the
features (the *codebook* loss, trains the codebook). That detach-asymmetry is
the gradient mismatch the paper traces to an underlying *distributional*
mismatch between feature vectors (``z_e``) and code vectors (``z_q``).

The objectives in this module replace that pair with a single distribution-
level loss that aligns the marginal distributions of ``z_e`` and ``z_q``.
Because the loss is distribution-level rather than pointwise, gradients flow
to *both* the encoder (through ``z_e``) and the codebook (through
``z_q = embedding(indices)``, which is differentiable w.r.t. the codebook
weights) with no STE detach -- the "STE-bypassing" objective of the paper.

Two instantiations are provided, matching the paper:
  * ``wasserstein2_gaussian_loss`` -- closed-form squared 2-Wasserstein
    distance under a diagonal-Gaussian approximation of each distribution
    (parameter-free, ``O(B * D * T)``). This is the paper's primary objective.
  * ``mmd_rbf_loss`` -- nonparametric maximum mean discrepancy with an RBF
    kernel (parameter-free, ``O(B * T^2 * D)``), shown in the paper to yield
    comparable performance.

Both operate per batch element over the ``T`` time-steps of ``(B, D, T)``
latent tensors and return a per-element loss of shape ``(B,)``, matching the
convention of :func:`dac.nn.quantize.VectorQuantize.forward`.
"""

import torch


def _as_set(x: torch.Tensor) -> torch.Tensor:
    """``(B, D, T)`` -> ``(B, T, D)``: a set of ``T`` D-dim vectors per batch element."""
    return x.transpose(1, 2)


def wasserstein2_gaussian_loss(
    z_e: torch.Tensor, z_q: torch.Tensor
) -> torch.Tensor:
    """Closed-form squared 2-Wasserstein distance between diagonal-Gaussian
    approximations of the ``z_e`` (feature) and ``z_q`` (code) distributions.

    For diagonal-covariance Gaussians ``N(mu_e, diag(sig_e^2))`` and
    ``N(mu_q, diag(sig_q^2))`` the squared 2-Wasserstein distance admits the
    closed form ``W2^2 = ||mu_e - mu_q||^2 + ||sig_e - sig_q||_F^2``.

    Parameters
    ----------
    z_e, z_q : Tensor[B x D x T]
        Feature and code vectors.

    Returns
    -------
    Tensor[B]
        Per-batch-element squared 2-Wasserstein distance.
    """
    mu_e = z_e.mean(dim=-1)  # (B, D)
    mu_q = z_q.mean(dim=-1)
    # Population moments: frame each batch element's T positions as samples.
    std_e = z_e.std(dim=-1, unbiased=False)
    std_q = z_q.std(dim=-1, unbiased=False)
    w2 = ((mu_e - mu_q) ** 2 + (std_e - std_q) ** 2).sum(dim=-1)  # (B,)
    return w2


def mmd_rbf_loss(
    z_e: torch.Tensor,
    z_q: torch.Tensor,
    bandwidth: float = None,
) -> torch.Tensor:
    """Biased MMD^2 (V-statistic) with an RBF kernel between the ``z_e`` and
    ``z_q`` point sets.

    Per batch element over the ``T`` positions, computes
    ``MMD^2 = E[k(e, e)] + E[k(q, q)] - 2 E[k(e, q)]`` with
    ``k(x, y) = exp(-||x - y||^2 / (2 sigma^2))``. If ``bandwidth`` is ``None``
    a per-batch median-heuristic ``sigma^2`` is estimated from the cross
    distances.

    Parameters
    ----------
    z_e, z_q : Tensor[B x D x T]
        Feature and code vectors.
    bandwidth : float, optional
        Fixed ``sigma^2``; if ``None`` the median heuristic is used.

    Returns
    -------
    Tensor[B]
        Per-batch-element MMD^2 (lies in ``[0, 2)`` for RBF kernels).
    """
    e = _as_set(z_e)  # (B, T, D)
    q = _as_set(z_q)

    d_ee = torch.cdist(e, e).pow(2)  # (B, T, T)
    d_qq = torch.cdist(q, q).pow(2)
    d_eq = torch.cdist(e, q).pow(2)

    if bandwidth is None:
        # Median heuristic on the cross-distances, per batch element.
        sigma2 = d_eq.flatten(1).median(dim=1).values.clamp_min(1e-6)
        sigma2 = sigma2.view(-1, 1, 1)
    else:
        sigma2 = bandwidth

    k_ee = torch.exp(-d_ee / (2.0 * sigma2))
    k_qq = torch.exp(-d_qq / (2.0 * sigma2))
    k_eq = torch.exp(-d_eq / (2.0 * sigma2))

    mmd = (
        k_ee.mean(dim=(1, 2))
        + k_qq.mean(dim=(1, 2))
        - 2.0 * k_eq.mean(dim=(1, 2))
    )
    return mmd  # (B,)


def distributional_matching_loss(
    z_e: torch.Tensor,
    z_q: torch.Tensor,
    kind: str = "wasserstein",
    **kwargs,
) -> torch.Tensor:
    """Dispatch to a distributional matching objective.

    Parameters
    ----------
    z_e, z_q : Tensor[B x D x T]
        Feature and code vectors.
    kind : {"wasserstein", "mmd"}
        Which instantiation of the distributional matching framework to use.
    **kwargs
        Forwarded to the chosen objective (e.g. ``bandwidth`` for ``mmd``).

    Returns
    -------
    Tensor[B]
        Per-batch-element distributional matching loss.
    """
    if kind == "wasserstein":
        return wasserstein2_gaussian_loss(z_e, z_q)
    if kind == "mmd":
        return mmd_rbf_loss(z_e, z_q, **kwargs)
    raise ValueError(f"unknown distributional matching kind: {kind!r}")
