"""
Tests for INT8 edge-deployment weight quantization of the conv tokenizer.
"""
import torch
import torch.nn as nn

from dac.model.dac import DAC
from dac.nn.edge_quantize import assess_quantization
from dac.nn.edge_quantize import projected_footprint
from dac.nn.edge_quantize import quantize_conv_weights
from dac.nn.edge_quantize import si_snr


def _tiny_dac():
    # A small codec so the conv-tokenizer path runs fast on CPU in CI.
    return DAC(
        encoder_dim=4,
        encoder_rates=[2, 2],
        decoder_dim=16,
        decoder_rates=[2, 2],
        n_codebooks=2,
        codebook_size=16,
        codebook_dim=4,
        sample_rate=16000,
    )


def _conv_modules(model):
    return [
        m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d))
    ]


def test_quantize_conv_weights_quantizes_and_bakes_weight_norm():
    torch.manual_seed(0)
    model = _tiny_dac().eval()

    stats = quantize_conv_weights(model.encoder, bits=8)
    assert stats["n_conv"] > 0
    assert 0.0 < stats["rel_l2_weight_error"] < 1.0

    convs = _conv_modules(model.encoder)
    assert convs
    for conv in convs:
        assert torch.isfinite(conv.weight).all()
        # weight_norm is baked off into a plain .weight Parameter
        assert not hasattr(conv, "weight_g")


def test_quantize_conv_stack_wires_through_dac_forward():
    torch.manual_seed(1)
    model = _tiny_dac().eval()
    x = torch.randn(1, 1, 4096)
    model(x)  # populate effective .weight on every weight_norm conv

    before = {id(m): m.weight.detach().clone() for m in _conv_modules(model)}

    returned = model.quantize_conv_stack(bits=8)
    assert returned is model  # the call-site hook returns self

    changed = 0
    for conv in _conv_modules(model):
        assert torch.isfinite(conv.weight).all()
        if not torch.equal(before[id(conv)], conv.weight.detach()):
            changed += 1
    assert changed > 0  # the conv weights were actually quantized

    out = model(x)["audio"]  # the quantized codec still decodes end-to-end
    assert out.shape == (1, 1, 4096)


def test_si_snr_is_self_consistent():
    torch.manual_seed(2)
    x = torch.randn(2048)
    assert si_snr(x, x) > 60.0
    assert si_snr(x, torch.randn(2048)) < si_snr(x, x)


def test_projected_footprint_reports_int8_quarter():
    model = _tiny_dac()
    foot = projected_footprint(model)
    assert foot["n_params"] > 0
    assert foot["ratio"] == 0.25
    assert foot["int8_bytes"] * 4 == foot["float32_bytes"]


def test_assess_quantization_reports_tradeoff():
    torch.manual_seed(3)
    model = _tiny_dac().eval()
    x = torch.randn(1, 1, 4096)

    report = assess_quantization(model, x, sample_rate=16000, n_runs=1)

    assert report["bits"] == 8
    assert report["n_conv"] > 0
    assert report["footprint_ratio"] == 0.25
    assert 0.0 <= report["rel_l2_weight_error"] < 1.0
    assert torch.isfinite(torch.tensor(report["quality_si_snr_db"]))
    assert report["rtf_float"] >= 0.0
    assert report["rtf_int8"] >= 0.0
