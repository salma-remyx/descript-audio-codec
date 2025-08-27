#!/usr/bin/env python3
"""
Script to encode and decode an audio file using a DAC model with bfloat16 autocast.
Usage: python encode_decode_audio.py <model_path> <audio_file>
"""

import argbind
import torch
from pathlib import Path
from audiotools import AudioSignal
from audiotools.data import transforms
from dac.model import DAC


@argbind.bind(without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode_decode_audio(
    model_path: Path = Path(""),
    audio_file: str = "",
    device: str = "",
):
    """Encode and decode audio using DAC model with bfloat16 autocast.
    
    Parameters
    ----------
    model_path : str
        Path to the DAC model checkpoint
    audio_file : str
        Path to the input audio file
    device : str, optional
        Device to use, by default auto-selects cuda if available, otherwise cpu
    """
    # Validate inputs
    audio_path = Path(audio_file)

    device = device if device != "" else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model package
    model = DAC.load(model_path / "best/dac/package.pth")
    model.to(device)
    model.eval()
    
    # Load audio
    signal = AudioSignal(str(audio_path))
    
    # Resample to model's sample rate if needed
    if signal.sample_rate != model.sample_rate:
        signal.resample(model.sample_rate)
    
    signal.to(device)
    
    # Apply training-like postprocess via _instantiate/_transform: VolumeNorm -> RescaleAudio
    _vol_norm = transforms.VolumeNorm(db=("const", -16.0))
    _rescale = transforms.RescaleAudio(val=1.0)
    vn_params = _vol_norm._instantiate(None)
    signal = _vol_norm._transform(signal, **vn_params)
    rs_params = _rescale._instantiate(None)
    signal = _rescale._transform(signal, **rs_params)
    
    # Encode and decode with bfloat16 autocast
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # Encode
        z = model.encode(signal.audio_data)
        if isinstance(z, tuple):
            z = z[0]
        # Decode
        y = model.decode(z)
    
    # Cast from bfloat16 to float32 for audio file writing
    output_signal = AudioSignal(y.detach().cpu().float(), model.sample_rate)
    
    # Save the decoded audio
    output_signal.write(str(audio_path.parent / f"{audio_path.stem}_decoded{audio_path.suffix}"))


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode_decode_audio() 