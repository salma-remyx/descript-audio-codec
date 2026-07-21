"""
Tests for the Time-Invariant Representation Extraction (TIRE) wiring in
DAC.encode. Adapted from TiCodec (arXiv:2607.05250v1).
"""
import torch

from dac.model.dac import DAC
from dac.nn.tire import TimeInvariantFactorizer


def _tiny_model(use_tire):
    # A small CPU-friendly DAC so tests need no pretrained weights.
    return DAC(
        encoder_dim=8,
        encoder_rates=[2, 2, 2],
        decoder_dim=32,
        decoder_rates=[2, 2, 2],
        n_codebooks=2,
        codebook_size=64,
        codebook_dim=4,
        use_tire=use_tire,
    )


def test_factorize_isolates_global_component():
    torch.manual_seed(0)
    z = torch.randn(2, 16, 50)
    tire = TimeInvariantFactorizer(dim=16)

    z_inv, z_residual = tire.factorize(z)

    # The global invariant is broadcastable over time, residual is [B x D x T].
    assert z_inv.shape == (2, 16, 1)
    assert z_residual.shape == z.shape

    # Pooling proxy removes the per-channel temporal mean from the residual.
    assert z_residual.mean(dim=-1).abs().max().item() < 1e-5

    # combine is an exact inverse of factorize (before quantization).
    assert torch.allclose(tire.combine(z_inv, z_residual), z, atol=1e-6)

    # invariant_ratio is a scalar probe in [0, 1].
    ratio = tire.invariant_ratio(z).item()
    assert 0.0 <= ratio <= 1.0


def test_pooling_factorizer_adds_no_parameters():
    # Default (pooling) TIRE has no parameters, so pretrained DAC weights
    # still load with strict=True.
    tire = TimeInvariantFactorizer(dim=64)
    assert list(tire.parameters()) == []
    learned = TimeInvariantFactorizer(dim=64, learned=True)
    assert len(list(learned.parameters())) > 0


def test_tire_flag_routed_through_encode():
    # Same weights, TIRE off vs on. Because the factorization routes a
    # different signal through the RVQ, the returned latent differs -- but
    # the [B x D x T] contract and a full decode are preserved.
    torch.manual_seed(0)
    base = _tiny_model(use_tire=False).eval()
    torch.manual_seed(0)
    with_tire = _tiny_model(use_tire=True).eval()
    with_tire.load_state_dict(base.state_dict())  # no extra params to mismatch

    x = torch.randn(1, 1, 8192)
    with torch.no_grad():
        z_plain, *_ = base.encode(x)
        z_tire, *_ = with_tire.encode(x)

    assert z_plain.shape == z_tire.shape
    assert not torch.allclose(z_plain, z_tire)

    # End-to-end forward still produces audio of the input length.
    out = with_tire(x)
    assert out["audio"].shape[-1] == x.shape[-1]
