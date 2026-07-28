"""
Tests for CLI.
"""
import subprocess
from pathlib import Path

import argbind
import numpy as np
import pytest
import torch
from audiotools import AudioSignal

from dac.__main__ import run
from dac.model.dac import DAC
from dac.nn.dist_match import DistributionalMatchLoss
from dac.nn.quantize import ResidualVectorQuantize, VectorQuantize


def setup_module(module):
    data_dir = Path(__file__).parent / "assets"
    data_dir.mkdir(exist_ok=True, parents=True)
    input_dir = data_dir / "input"
    input_dir.mkdir(exist_ok=True, parents=True)

    for i in range(5):
        signal = AudioSignal(np.random.randn(1000), 44_100)
        signal.write(input_dir / f"sample_{i}.wav")
    return input_dir


def teardown_module(module):
    repo_root = Path(__file__).parent.parent
    subprocess.check_output(["rm", "-rf", f"{repo_root}/tests/assets"])


@pytest.mark.parametrize("model_type", ["44khz", "24khz", "16khz"])
def test_reconstruction(model_type):
    # Test encoding
    input_dir = Path(__file__).parent / "assets" / "input"
    output_dir = input_dir.parent / model_type / "encoded_output"
    args = {
        "input": str(input_dir),
        "output": str(output_dir),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_type": model_type,
    }
    with argbind.scope(args):
        run("encode")

    # Test decoding
    input_dir = output_dir
    output_dir = input_dir.parent / model_type / "decoded_output"
    args = {
        "input": str(input_dir),
        "output": str(output_dir),
        "model_type": model_type,
    }
    with argbind.scope(args):
        run("decode")


def test_compression():
    # Test encoding
    input_dir = Path(__file__).parent / "assets" / "input"
    output_dir = input_dir.parent / "encoded_output_quantizers"
    args = {
        "input": str(input_dir),
        "output": str(output_dir),
        "n_quantizers": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    with argbind.scope(args):
        run("encode")

    # Open .dac file
    dac_file = output_dir / "sample_0.dac"
    artifacts = np.load(dac_file, allow_pickle=True)[()]
    codes = artifacts["codes"]

    # Ensure that the number of quantizers is correct
    assert codes.shape[1] == 3

    # Ensure that dtype of compression is uint16
    assert codes.dtype == np.uint16


def test_dist_match_round_trip():
    # Entry-point surface: the compress/decompress round-trip runs with the
    # distributional matching loss wired in (eval mode -> loss is skipped).
    model = DAC(encoder_dim=16, n_codebooks=2, codebook_size=64, codebook_dim=8)
    model.eval()

    signal = AudioSignal(torch.randn(1, 1, 8192), 44100)
    x = model.compress(signal)
    y = model.decompress(x)

    assert y.audio_data.shape[-1] == signal.audio_data.shape[-1]


def test_dac_forward_surfaces_dist_match_loss():
    # DAC.forward surfaces the loss alongside the existing VQ losses, exactly
    # as scripts/train.py consumes it.
    model = DAC(encoder_dim=16, n_codebooks=2, codebook_size=64, codebook_dim=8)
    model.train()
    audio = torch.randn(1, 1, 8192)

    out = model(audio, sample_rate=model.sample_rate)

    assert "vq/dist_match_loss" in out
    loss = out["vq/dist_match_loss"]
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    # It must be differentiable into the total training loss.
    loss.backward()


def test_rvq_forward_returns_dist_match_loss():
    # Training mode: the RVQ must surface the distributional matching loss.
    rvq = ResidualVectorQuantize(
        input_dim=64, n_codebooks=3, codebook_size=256, codebook_dim=8
    )
    rvq.train()
    z = torch.randn(4, 64, 80)

    out = rvq(z)
    assert len(out) == 6
    z_q, codes, latents, commitment_loss, codebook_loss, dist_match_loss = out

    assert dist_match_loss.dim() == 0
    assert torch.isfinite(dist_match_loss)
    assert dist_match_loss.item() >= 0.0
    # Same value is cached on the quantizer for DAC.forward to read.
    assert rvq.dist_match_loss is dist_match_loss


def test_dist_match_gradients_reach_codebook_only():
    # L_match gradients bypass the STE and update the codebook, while the
    # features are stop-gradiented so the encoder keeps training on the
    # reconstruction/commitment signal alone.
    vq = VectorQuantize(input_dim=64, codebook_size=256, codebook_dim=8)
    vq.train()
    z = torch.randn(4, 64, 80)

    _, _, _, dist_match_loss, _, _ = vq(z)
    dist_match_loss.backward()

    assert vq.codebook.weight.grad is not None
    assert vq.codebook.weight.grad.abs().sum().item() > 0.0
    # The encoder side (in_proj / out_proj) must receive no gradient from
    # L_match: features are detached inside the loss.
    encoder_params = list(vq.in_proj.parameters()) + list(vq.out_proj.parameters())
    assert all(p.grad is None for p in encoder_params)


def test_dist_match_zero_when_distributions_aligned():
    loss_fn = DistributionalMatchLoss(kind="wasserstein")
    codes = torch.randn(256, 8)

    # Identical feature/code distributions -> ~0 mismatch.
    val_aligned = loss_fn(codes.clone(), codes)
    # Shifted feature mean -> large mismatch.
    val_shifted = loss_fn(codes + 5.0, codes)

    assert val_aligned.item() < 1e-4
    assert val_shifted.item() > 1.0
    assert val_shifted.item() > val_aligned.item()


def test_loss_detaches_features_directly():
    # Module-level guarantee: even when the caller passes features that
    # require grad, L_match back-propagates to the codebook only.
    loss_fn = DistributionalMatchLoss(kind="wasserstein")
    feats = torch.randn(64, 8, requires_grad=True)
    codes = torch.randn(64, 8, requires_grad=True)

    loss_fn(feats, codes).backward()

    assert feats.grad is None
    assert codes.grad is not None
    assert codes.grad.abs().sum().item() > 0.0


def test_mmd_kind_runs_and_zero_when_aligned():
    vq = VectorQuantize(
        input_dim=32, codebook_size=128, codebook_dim=8, dist_match_kind="mmd"
    )
    vq.train()
    _, _, _, dist_match_loss, _, _ = vq(torch.randn(4, 32, 40))
    assert torch.isfinite(dist_match_loss)
    assert dist_match_loss.item() >= 0.0

    # MMD between identical distributions is zero.
    mmd = DistributionalMatchLoss(kind="mmd")
    codes = torch.randn(128, 8)
    assert mmd(codes.clone(), codes).item() < 1e-5


def test_eval_mode_skips_dist_match():
    # At inference the loss is zero (no compute / graph overhead).
    vq = VectorQuantize(input_dim=64, codebook_size=256, codebook_dim=8)
    vq.eval()
    _, _, _, dist_match_loss, _, _ = vq(torch.randn(2, 64, 20))
    assert dist_match_loss.item() == 0.0


# CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_cli.py -s
