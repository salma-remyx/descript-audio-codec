"""
Tests for Gabor Latent Refactorization (GLRF) and its integration into DAC.

These exercise the wiring through the existing public ``DAC`` interface
(``dac.model.dac``) plus the basis properties of the new module.
"""
import torch

from dac.model.dac import DAC
from dac.nn.frequency_basis import GaborLatentRefactorization


def test_basis_is_orthonormal_and_invertible():
    glrf = GaborLatentRefactorization()
    for length in (1, 8, 16, 32):
        basis = glrf._basis(length, torch.device("cpu"), torch.float32)
        # Square, orthonormal basis: G^T G == I.
        eye = torch.eye(length)
        gtg = basis.transpose(-1, -2) @ basis
        assert torch.allclose(gtg, eye, atol=1e-5), length
        # Full rank -> forward then inverse recovers the input exactly.
        z = torch.randn(3, 64, length)
        recovered = glrf.inverse(glrf(z))
        assert torch.allclose(recovered, z, atol=1e-4), length


def test_forward_preserves_shape_and_changes_representation():
    glrf = GaborLatentRefactorization()
    z = torch.randn(2, 64, 16)
    out = glrf(z)
    assert out.shape == z.shape
    # A non-trivial basis change: the refactored representation differs.
    assert not torch.allclose(out, z, atol=1e-4)


def test_dac_glrf_forward_round_trips_and_preserves_shapes():
    # Exercises the encode-path wiring via the existing DAC public interface.
    torch.manual_seed(0)
    model = DAC(use_glrf=True).to("cpu").eval()
    assert model.glrf is not None

    audio = torch.randn(1, 1, 4096)
    with torch.no_grad():
        out = model(audio)

    # Reconstruction shape matches the input; latents/codes keep their contract.
    assert out["audio"].shape == audio.shape
    assert out["z"].shape[0] == 1
    assert out["codes"].dim() == 3


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
