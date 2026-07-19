"""
Integration tests for the L3AC-inspired single-codebook quantizer.

These tests import the EXISTING codec (``dac.model.dac.DAC``) and exercise the
new ``SingleVectorQuantize`` through the real encode -> quantize -> decode and
compress-style from_codes paths, proving the single-quantizer collapse keeps
the audio<->codes contract intact.
"""
import torch

from dac.model.dac import DAC
from dac.nn.quantize import ResidualVectorQuantize
from dac.nn.single_quantize import SingleVectorQuantize


def _tiny_model():
    # Small DAC so the round-trip runs fast on CPU. ``n_codebooks=9`` matches
    # the default RVQ stack so the swap is apples-to-apples.
    return DAC(
        encoder_dim=16,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=64,
        n_codebooks=9,
        codebook_size=256,
        codebook_dim=8,
    )


def test_single_quantizer_collapses_codebook_axis():
    """The L3AC result: codes go from N=9 (RVQ) to N=1 (single quantizer)
    while the latent width is preserved across the swap."""
    rvq = ResidualVectorQuantize(input_dim=128, n_codebooks=9, codebook_dim=8)
    sq = SingleVectorQuantize(input_dim=128, n_codebooks=9, codebook_dim=8)

    z = torch.randn(2, 128, 80)
    _, rvq_codes, rvq_latents, _, _ = rvq(z)
    _, sq_codes, sq_latents, _, _ = sq(z)

    assert rvq_codes.shape[1] == 9, "baseline RVQ should use 9 codebooks"
    assert sq_codes.shape[1] == 1, "single quantizer must collapse to N=1"
    # Capacity preservation: projected latent width is unchanged.
    assert rvq_latents.shape[1] == sq_latents.shape[1]


def test_dac_runs_end_to_end_with_single_quantizer():
    """Swapping ``model.quantizer`` for the single quantizer keeps the full
    DAC forward pass working and emits single-codebook codes."""
    model = _tiny_model().eval()
    model.quantizer = SingleVectorQuantize(
        input_dim=model.latent_dim,
        n_codebooks=model.n_codebooks,
        codebook_size=model.codebook_size,
        codebook_dim=model.codebook_dim,
    )

    hop = int(model.hop_length)
    audio = torch.randn(2, 1, hop * 4)

    with torch.no_grad():
        out = model(audio)

    assert out["codes"].shape[0] == 2
    assert out["codes"].shape[1] == 1, "decoded codes must carry a single codebook"
    assert out["audio"].shape == audio.shape, "round-trip must preserve audio shape"
    assert torch.isfinite(out["vq/commitment_loss"]).all()
    assert torch.isfinite(out["vq/codebook_loss"]).all()


def test_from_codes_round_trips_through_real_decoder():
    """The contract that ``CodecMixin.decompress`` relies on: ``from_codes``
    reproduces the quantized latent, and the decoder turns it back into audio
    of the right shape -- with a single codebook."""
    model = _tiny_model().eval()
    model.quantizer = SingleVectorQuantize(
        input_dim=model.latent_dim,
        n_codebooks=model.n_codebooks,
        codebook_size=model.codebook_size,
        codebook_dim=model.codebook_dim,
    )

    hop = int(model.hop_length)
    audio = torch.randn(1, 1, hop * 2)

    with torch.no_grad():
        _, codes, _, _, _ = model.encode(audio)
        # from_codes is what decompress calls; it must reproduce the quantized
        # latent (eval mode, same indices) up to float reassociation between
        # the straight-through forward path and the direct lookup.
        z_q_forward = model.quantizer(model.encoder(audio))[0]
        z_q_from_codes = model.quantizer.from_codes(codes)[0]
        decoded = model.decode(z_q_from_codes)

    torch.testing.assert_close(z_q_from_codes, z_q_forward, rtol=1e-5, atol=1e-6)
    assert decoded.shape == (1, 1, hop * 2)


def test_from_codes_is_shape_agnostic_for_legacy_codes():
    """``from_codes`` decodes the first codebook axis, so a legacy RVQ file
    (N=9) can still be read by a single-quantizer codec without error."""
    model = _tiny_model().eval()
    model.quantizer = SingleVectorQuantize(
        input_dim=model.latent_dim,
        n_codebooks=model.n_codebooks,
        codebook_size=model.codebook_size,
        codebook_dim=model.codebook_dim,
    )

    hop = int(model.hop_length)
    audio = torch.randn(1, 1, hop)
    with torch.no_grad():
        _, codes, _, _, _ = model.encode(audio)

    # Pretend this is a 9-codebook file by stacking the single codebook.
    legacy_codes = codes.repeat(1, 9, 1)
    z_q, z_p, returned_codes = model.quantizer.from_codes(legacy_codes)

    assert z_q.shape[0] == 1
    assert z_p.shape[0] == 1
    assert returned_codes.shape[1] == 9, "codes are returned unchanged"
