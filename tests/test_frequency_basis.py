"""
Tests for Gabor Latent Refactorization (GLRF) and its integration into DAC.

These exercise the wiring through the existing public ``DAC`` interface
(``dac.model.dac``) plus the filterbank properties of the new module.
"""
import pytest
import torch

from dac.model.dac import DAC
from dac.model.dac import conv_receptive_field
from dac.nn.frequency_basis import GaborLatentRefactorization


def _small_glrf(**overrides):
    kwargs = dict(
        receptive_field=8.0,
        n_filters=4,
        filter_len=15,
        n_fit_signals=64,
        fit_length=64,
    )
    kwargs.update(overrides)
    return GaborLatentRefactorization(**kwargs)


def _harmonic_latents(batch=2, channels=3, length=64):
    """Smooth harmonic channel signals, like structured encoder latents."""
    t = torch.arange(length, dtype=torch.float32)
    f0 = torch.rand(batch, channels, 1) * 0.1 + 0.02
    z = torch.zeros(batch, channels, length)
    for h in range(1, 5):
        phase = 2.0 * torch.pi * torch.rand(batch, channels, 1)
        z = z + (1.0 / h) * torch.sin(2.0 * torch.pi * h * f0 * t + phase)
    return z


def test_filterbank_is_fixed_and_frequency_localized():
    glrf = _small_glrf()
    # Parameter-free: nothing trainable, nothing in state_dict.
    assert len(list(glrf.parameters())) == 0
    assert len(glrf.state_dict()) == 0

    # A pure latent sinusoid at a filter center peaks in that filter's band.
    length = 64
    t = torch.arange(length, dtype=torch.float32)
    centers = torch.linspace(0.0, 0.5, glrf.n_filters)
    for k in (0, glrf.n_filters // 2, glrf.n_filters - 1):
        z = torch.cos(2.0 * torch.pi * centers[k] * t).view(1, 1, length)
        g = glrf(z)  # [1, 1, 2F, T]
        energy = g[0, 0].pow(2).sum(-1)  # [2F]
        band = energy[: glrf.n_filters] + energy[glrf.n_filters :]
        assert band.argmax().item() == k


def test_forward_expands_to_gabor_features():
    glrf = _small_glrf()
    z = torch.randn(2, 5, 32)
    g = glrf(z)
    assert g.shape == (2, 5, 2 * glrf.n_filters, 32)
    # Deterministic, fixed transform.
    assert torch.allclose(g, glrf(z), atol=1e-6)


def test_inverse_recovers_harmonic_latents():
    glrf = _small_glrf()
    z = _harmonic_latents()
    z_hat = glrf.inverse(glrf(z))
    assert z_hat.shape == z.shape
    # Paper reports latent cosine similarity >= 0.97 after GLRF on DAC.
    cos = torch.nn.functional.cosine_similarity(z.flatten(), z_hat.flatten(), dim=0)
    assert cos > 0.95, cos


def test_inverse_requires_forward_stats():
    glrf = _small_glrf()
    g = torch.randn(1, 2, 2 * glrf.n_filters, 16)
    with pytest.raises(RuntimeError, match="forward"):
        glrf.inverse(g)


def test_receptive_field_helper_counts_encoder_convs():
    model = DAC(encoder_rates=[2, 4], decoder_rates=[4, 2])
    rf = conv_receptive_field(model.encoder)
    # First conv (k=7) + per-block dilated stacks + strided convs + final k=3.
    assert rf > model.hop_length
    assert isinstance(rf, int)


def test_dac_glrf_round_trip_preserves_contract():
    # Exercises the decode-path wiring via the existing DAC public interface.
    torch.manual_seed(0)
    model = DAC(encoder_rates=[2, 4], decoder_rates=[4, 2], use_glrf=True)
    model = model.to("cpu").eval()
    assert model.glrf is not None

    audio = torch.randn(1, 1, 2048)
    with torch.no_grad():
        out = model(audio)
        # Post-hoc GLRF features decode back through the same interface.
        g = model.glrf(out["z"])
        y = model.decode(g)
        y_flat = model.decode(g.flatten(1, 2))

    assert out["audio"].shape == audio.shape
    assert out["codes"].dim() == 3
    assert y.shape == audio.shape
    assert torch.allclose(y, y_flat, atol=1e-5)


def test_glrf_adds_no_state_dict_keys():
    # GLRF is retraining-free: enabling it must not add weights, so a
    # pretrained checkpoint still loads under strict loading.
    without_glrf = DAC(use_glrf=False)
    with_glrf = DAC(use_glrf=True)
    base_keys = set(without_glrf.state_dict().keys())
    glrf_keys = set(with_glrf.state_dict().keys())
    assert not any("glrf" in k for k in glrf_keys)
    assert base_keys == glrf_keys


def test_glrf_disabled_by_default():
    model = DAC()
    assert model.glrf is None
