"""
Tests for the distributional-matching VQ objective wiring.

These exercise the call-site edit in ``dac.nn.quantize`` (a NON-new module):
that enabling ``distributional=True`` collapses the dual STE-asymmetric
commitment/codebook losses into the single STE-bypassing distributional
objective, while ``distributional=False`` preserves the original behavior.
"""
import torch

from dac.nn.quantize import ResidualVectorQuantize
from dac.nn.quantize import VectorQuantize
from dac.nn.vq_distributional import wasserstein2_gaussian_loss


def _make_vq(distributional=False, dist_kind="wasserstein"):
    torch.manual_seed(0)
    return VectorQuantize(
        input_dim=8,
        codebook_size=16,
        codebook_dim=8,
        distributional=distributional,
        dist_kind=dist_kind,
    )


def test_default_keeps_dual_ste_losses():
    """distributional=False preserves the original commitment + codebook pair."""
    vq = _make_vq(distributional=False)
    z = torch.randn(4, 8, 20)
    _, commitment_loss, codebook_loss, _, _ = vq(z)

    assert commitment_loss.shape == (4,)
    assert codebook_loss.shape == (4,)
    # Both STE-asymmetric terms are active in the default path.
    assert torch.all(commitment_loss > 0)
    assert torch.all(codebook_loss > 0)


def test_distributional_collapses_to_single_objective():
    """distributional=True zeroes the commitment slot and keeps a single term."""
    vq = _make_vq(distributional=True, dist_kind="wasserstein")
    z = torch.randn(4, 8, 20)
    _, commitment_loss, codebook_loss, _, _ = vq(z)

    assert torch.all(commitment_loss == 0)
    assert torch.all(codebook_loss > 0)


def test_distributional_mmd_kind_runs():
    vq = _make_vq(distributional=True, dist_kind="mmd")
    z = torch.randn(2, 8, 16)
    _, commitment_loss, codebook_loss, _, _ = vq(z)

    assert torch.all(commitment_loss == 0)
    assert torch.isfinite(codebook_loss).all()
    assert torch.all(codebook_loss >= 0)


def test_distributional_bypasses_ste_to_codebook_and_encoder():
    """The single objective trains the codebook AND encoder with no STE detach."""
    vq = _make_vq(distributional=True)
    z = torch.randn(4, 8, 20)
    _, _, codebook_loss, _, _ = vq(z)

    codebook_loss.mean().backward()

    # Gradient reaches the codebook via the embedding lookup of z_q ...
    assert vq.codebook.weight.grad is not None
    assert torch.any(vq.codebook.weight.grad != 0)
    # ... and the encoder via z_e (the in_proj weight-norm params).
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for p in vq.in_proj.parameters())


def test_residual_vector_quantize_distributional():
    """The flag threads through RVQ and collapses the summed losses."""
    torch.manual_seed(0)
    rvq = ResidualVectorQuantize(
        input_dim=8,
        n_codebooks=3,
        codebook_size=16,
        codebook_dim=8,
        distributional=True,
        dist_kind="wasserstein",
    )
    z = torch.randn(4, 8, 20)
    rvq.eval()  # deterministic n_quantizers (no dropout randomness)
    _, _, _, commitment_loss, codebook_loss = rvq(z)

    assert commitment_loss == 0.0
    assert codebook_loss > 0.0


def test_wasserstein_loss_is_zero_for_identical_distributions():
    """Sanity: aligning the feature and code distributions drives the loss to 0."""
    torch.manual_seed(1)
    z_e = torch.randn(4, 8, 64)
    z_q = z_e.clone()
    assert torch.allclose(
        wasserstein2_gaussian_loss(z_e, z_q),
        torch.zeros(4),
        atol=1e-6,
    )
