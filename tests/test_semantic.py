"""
Tests for the X-Codec-style semantic codebook branch (``use_semantic``).

Adapted from "Codec Does Matter" (arXiv:2408.17175): an opt-in coarse-rate
semantic codebook whose embedding is fused into the acoustic latent and whose
discrete codes augment the acoustic RVQ codes.
"""
import math

import torch

from dac.model.dac import DAC


def _small_model(**kwargs):
    # Keep the encoder/decoder architecture valid (default encoder_dim) but
    # shrink the codebooks so the test runs quickly on CPU.
    return DAC(n_codebooks=2, codebook_dim=4, **kwargs)


def test_default_model_has_no_semantic_branch():
    # Regression guard: the default codec is unchanged.
    model = _small_model()
    assert model.use_semantic is False
    assert model.semantic_quantizer is None

    x = torch.randn(1, 1, 8192)
    out = model(x, model.sample_rate)
    assert "semantic_codes" not in out


def test_encode_preserves_five_tuple_contract():
    # compress (dac.model.base) unpacks encode positionally -> 5-tuple required.
    model = _small_model(use_semantic=True, semantic_downsample=8)
    x = torch.randn(1, 1, 8192)
    z, codes, latents, commitment_loss, codebook_loss = model.encode(
        model.preprocess(x, model.sample_rate)
    )
    assert z.shape[0] == 1
    assert codes.shape[1] == 2
    assert commitment_loss.dim() == 0
    assert codebook_loss.dim() == 0


def test_forward_exposes_coarse_semantic_codes():
    torch.manual_seed(0)
    model = _small_model(use_semantic=True, semantic_downsample=8)
    assert model.semantic_quantizer is not None

    x = torch.randn(1, 1, 8192)
    out = model(x, model.sample_rate)

    # Standard contract preserved.
    for key in (
        "audio",
        "z",
        "codes",
        "latents",
        "vq/commitment_loss",
        "vq/codebook_loss",
    ):
        assert key in out
    assert out["audio"].shape[-1] == 8192

    # Semantic codes are exposed at the coarse frame rate.
    semantic = out["semantic_codes"]
    t_acoustic = out["codes"].shape[-1]
    assert semantic.dim() == 2
    assert semantic.shape[0] == 1
    assert semantic.shape[-1] == math.ceil(t_acoustic / 8)
    assert semantic.dtype == torch.long
    cb_size = model.semantic_quantizer.quantizer.codebook_size
    assert semantic.min() >= 0
    assert semantic.max() < cb_size


def test_semantic_fusion_changes_latent():
    # The restored semantic embedding is non-trivial and fuses into z.
    torch.manual_seed(0)
    model = _small_model(use_semantic=True, semantic_downsample=4)
    x = torch.randn(1, 1, 8192)
    z = model.encoder(model.preprocess(x, model.sample_rate))
    embed, codes, losses = model._semantic_branch(z)

    assert embed.shape == z.shape
    assert not torch.allclose(embed, torch.zeros_like(embed))
    assert codes.shape[-1] == math.ceil(z.shape[-1] / 4)
    assert "vq/commitment_loss" in losses
    assert "vq/codebook_loss" in losses


def test_from_codes_roundtrips_embedding():
    torch.manual_seed(0)
    model = _small_model(use_semantic=True, semantic_downsample=8)
    x = torch.randn(1, 1, 8192)
    z = model.encoder(model.preprocess(x, model.sample_rate))
    embed, codes, _ = model.semantic_quantizer(z)

    restored = model.semantic_quantizer.from_codes(codes, z.shape[-1])
    assert restored.shape == embed.shape


def test_augmented_token_stream():
    # The paper's headline: a coarse semantic code stream augments the
    # fine-rate acoustic codes.
    torch.manual_seed(0)
    model = _small_model(use_semantic=True, semantic_downsample=8)
    x = torch.randn(1, 1, 8192)
    out = model(x, model.sample_rate)

    acoustic = out["codes"]  # [B, N, T_ac]
    semantic = out["semantic_codes"]  # [B, T_sem]
    assert acoustic.dim() == 3
    assert semantic.dim() == 2
    # Semantic stream is strictly coarser than the acoustic one.
    assert semantic.shape[-1] < acoustic.shape[-1]
