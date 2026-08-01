"""Post-training INT8 weight quantization of the codec's conv tokenizer.

Adapted from *VibeVoice-ASR-BitNet* (arXiv:2607.21075), which applies INT8
post-training quantization to the VAE acoustic tokenizer to run real-time
inference on edge CPUs. Descript Audio Codec is itself a conv tokenizer
(``encoder`` -> RVQ -> ``decoder``); the discrete-codebook RVQ is already a
compact lookup table and is left in float, so only the float conv weights of
the encoder and decoder are quantized -- the slice of the recipe that ports
cleanly to PyTorch.

Intentionally *not* ported (they need infrastructure a pure-Python codec does
not host):

* the BitNet-style ternary weights of the autoregressive-LM half -- DAC has
  no autoregressive language model;
* the progressive quantization-aware training -- it needs the training loop;
* the custom ggml / SIMD fused kernels that actually deliver the wall-clock
  speedup -- C++ on x86/ARM, not transferable to PyTorch.

Instead this module ships the target-native substitute: a backend-agnostic
INT8 (or N-bit) symmetric weight quantization of the conv stack plus a small
measurement harness -- projected memory footprint, reconstruction-quality
delta (SI-SNR) vs. the float model, and CPU real-time factor -- so the team
can read off the speed / quality / memory trade-off the paper reports for
*their* codec. Weights are written back as float parameters, so the quantized
model runs on any backend (no fbgemm/qnnpack, no FX tracing, no int8
activations); DAC's scripted ``Snake1d`` activation makes those eager-static
paths brittle.
"""
import time
from typing import Dict

import torch
from torch import nn
from torch.nn.utils import remove_weight_norm


def _conv_modules(root: nn.Module):
    """Yield ``(module, name)`` for every Conv1d / ConvTranspose1d in ``root``."""
    for name, module in root.named_modules():
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            yield module, name


def quantize_conv_weights(
    module: nn.Module,
    bits: int = 8,
    per_channel: bool = True,
) -> Dict[str, float]:
    """Quantize the conv weights of ``module`` to ``bits``-bit ints, in place.

    Symmetric per-output-channel (or per-tensor) integer quantization. The
    rounded weights are written back as float parameters, so the module keeps
    running on any backend while carrying exactly the quantization noise the
    paper's INT8 tokenizer does. Any ``weight_norm`` wrappers are first baked
    into a plain ``.weight`` (``remove_weight_norm``).

    Returns ``{"n_conv", "rel_l2_weight_error"}`` -- the latter is the mean
    relative L2 distance between the float and quantized conv weights.
    """
    assert bits >= 2, "bits must be >= 2"
    qmax = 2 ** (bits - 1) - 1
    n_conv = 0
    err_num = 0.0
    err_den = 0.0
    for conv, _name in _conv_modules(module):
        try:
            remove_weight_norm(conv)
        except (ValueError, AttributeError):
            # Not weight-normalized -- its .weight is already a plain tensor.
            pass
        if conv.weight is None:
            continue
        n_conv += 1
        weight = conv.weight.detach()
        if per_channel:
            out_dim = 1 if isinstance(conv, nn.ConvTranspose1d) else 0
            reduce_dims = tuple(d for d in range(weight.dim()) if d != out_dim)
            scale = weight.abs().amax(dim=reduce_dims, keepdim=True) / qmax
        else:
            scale = weight.abs().amax() / qmax
        # All-zero slice -> leave it at zero instead of dividing by a zero scale.
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quantized = torch.round(weight / scale).clamp(-qmax, qmax) * scale
        err_num += (weight - quantized).norm().item()
        err_den += weight.norm().item()
        with torch.no_grad():
            conv.weight.copy_(quantized)
    return {"n_conv": n_conv, "rel_l2_weight_error": err_num / (err_den + 1e-12)}


def projected_footprint(module: nn.Module) -> Dict[str, float]:
    """Projected storage of the conv-tokenizer weights: float32 vs packed int8.

    The int8 figure is the *packed* footprint (1 byte/weight) the paper's SIMD
    kernels would hold; the in-memory model here keeps float weights carrying
    the quantization noise, so this is the storage you would realise once the
    quantized weights are packed -- i.e. the memory axis the paper reports.
    """
    n_params = 0
    for conv, _name in _conv_modules(module):
        n_params += conv.weight.numel()
    float32_bytes = n_params * 4
    int8_bytes = n_params * 1
    return {
        "n_params": n_params,
        "float32_bytes": float32_bytes,
        "int8_bytes": int8_bytes,
        "ratio": int8_bytes / float32_bytes if float32_bytes else 0.0,
    }


def si_snr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Scale-invariant SNR (dB) between two waveforms of matching length."""
    x = x.reshape(-1).to(torch.float64)
    y = y.reshape(-1).to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    alpha = (x @ y) / (y @ y + 1e-12)
    target = alpha * y
    noise = x - target
    return float(
        10.0 * torch.log10((target.pow(2).sum() + 1e-12) / (noise.pow(2).sum() + 1e-12))
    )


@torch.no_grad()
def benchmark_rtf(
    model: nn.Module,
    audio_data: torch.Tensor,
    sample_rate: int,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> Dict[str, float]:
    """Measure real-time factor = decode_time / audio_duration on the model's device."""
    was_training = model.training
    model.eval()
    try:
        duration = audio_data.shape[-1] / float(sample_rate)
        for _ in range(n_warmup):
            model(audio_data, sample_rate)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(audio_data, sample_rate)
        elapsed = (time.perf_counter() - start) / n_runs
        return {
            "seconds_per_run": elapsed,
            "audio_seconds": duration,
            "rtf": elapsed / duration if duration else float("inf"),
        }
    finally:
        model.train(was_training)


@torch.no_grad()
def assess_quantization(
    model: nn.Module,
    audio_data: torch.Tensor,
    sample_rate: int,
    bits: int = 8,
    n_runs: int = 3,
) -> Dict[str, float]:
    """One-shot INT8-PTQ viability check -- the paper's headline trade-off.

    Decodes ``audio_data`` with the float model, INT8-quantizes the conv
    tokenizer in place, decodes again, and reports reconstruction quality
    (SI-SNR dB), projected memory ratio, and CPU RTF for float vs. int8.
    Mutates ``model`` (its conv weights are quantized); clone first if you
    still need the float weights afterwards.
    """
    model.eval()
    reference = model(audio_data, sample_rate)["audio"]
    rtf_float = benchmark_rtf(model, audio_data, sample_rate, n_runs=n_runs)["rtf"]
    stats = quantize_conv_weights(model, bits=bits)
    quantized = model(audio_data, sample_rate)["audio"]
    rtf_int8 = benchmark_rtf(model, audio_data, sample_rate, n_runs=n_runs)["rtf"]
    footprint = projected_footprint(model)
    return {
        "bits": bits,
        "n_conv": stats["n_conv"],
        "rel_l2_weight_error": stats["rel_l2_weight_error"],
        "quality_si_snr_db": si_snr(reference, quantized),
        "footprint_ratio": footprint["ratio"],
        "float32_bytes": footprint["float32_bytes"],
        "int8_bytes": footprint["int8_bytes"],
        "rtf_float": rtf_float,
        "rtf_int8": rtf_int8,
    }
