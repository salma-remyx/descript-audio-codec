"""
Tests for the opt-in finite scalar quantization (FSQ) quantizer.

These exercise the wiring in the existing ``dac.nn.quantize`` module
(``ResidualVectorQuantize`` built with ``quantizer_type="fsq"``) rather than
self-testing ``dac.nn.fsq`` in isolation, so they prove the integration.
"""
import numpy as np
import pytest
import torch

from dac.nn.quantize import FiniteScalarQuantize
from dac.nn.quantize import ResidualVectorQuantize


def test_fsq_residual_quantizer_matches_rvq_contract():
    """FSQ plugged into RVQ honors the RVQ.forward I/O contract."""
    rvq = ResidualVectorQuantize(
        input_dim=64,
        n_codebooks=4,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.0,
        quantizer_type="fsq",
    )
    z = torch.randn(2, 64, 80)
    z_q, codes, latents, commitment_loss, codebook_loss = rvq(z)

    assert z_q.shape == z.shape
    # codes: [B, N, T], all FSQ code indices valid non-negative integers.
    assert codes.shape == (2, 4, 80)
    assert codes.min() >= 0
    # latents stack the per-quantizer projected codes: [B, N * d, T].
    assert latents.shape[0] == 2 and latents.shape[2] == 80
    # FSQ is codebook-free, so both auxiliary losses are exactly zero.
    assert torch.all(commitment_loss == 0)
    assert torch.all(codebook_loss == 0)


def test_fsq_codes_round_trip_through_from_codes():
    """Reconstructing from the emitted codes reproduces the quantized latent."""
    torch.manual_seed(0)
    rvq = ResidualVectorQuantize(
        input_dim=64,
        n_codebooks=4,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_type="fsq",
    )
    rvq.eval()
    z = torch.randn(1, 64, 50)
    with torch.no_grad():
        z_q, codes, _, _, _ = rvq(z)
        z_q_from_codes, _, _ = rvq.from_codes(codes)

    assert z_q_from_codes.shape == z_q.shape
    assert torch.allclose(z_q, z_q_from_codes, atol=1e-5)


def test_fsq_default_quantizer_is_rvq():
    """Without the opt-in flag, RVQ still builds learned VectorQuantize codebooks."""
    from dac.nn.quantize import VectorQuantize

    rvq = ResidualVectorQuantize(input_dim=64, n_codebooks=2, codebook_size=64)
    assert isinstance(rvq.quantizers[0], VectorQuantize)
    assert not isinstance(rvq.quantizers[0], FiniteScalarQuantize)
    # Sanity: the default path still runs end-to-end.
    z = torch.randn(1, 64, 16)
    z_q, codes, _, _, _ = rvq(z)
    assert z_q.shape == z.shape
    assert codes.shape == (1, 2, 16)


def test_fsq_is_reachable_from_dac_encode():
    """FSQ is reachable through the top-level DAC codec entry point."""
    audiotools = pytest.importorskip("audiotools")
    from dac.model.dac import DAC

    model = DAC(
        encoder_dim=8,
        encoder_rates=[2, 4, 8],
        decoder_dim=32,
        decoder_rates=[8, 4, 2],
        n_codebooks=2,
        codebook_size=64,
        codebook_dim=4,
        quantizer_type="fsq",
    )
    model.eval()
    # hop_length = prod(encoder_rates) = 64
    audio = torch.randn(1, 1, 64 * 8)
    with torch.no_grad():
        z, codes, latents, commitment_loss, codebook_loss = model.encode(audio)
    assert codes.shape[0] == 1 and codes.shape[1] == 2
    assert torch.all(commitment_loss == 0)
    assert torch.all(codebook_loss == 0)


if __name__ == "__main__":
    # Allow running without pytest for a quick smoke check.
    test_fsq_residual_quantizer_matches_rvq_contract()
    test_fsq_codes_round_trip_through_from_codes()
    test_fsq_default_quantizer_is_rvq()
    print("ok")
    _ = np  # keep numpy import meaningful for the standalone path
    _ = torch
