#!/usr/bin/env python3
"""
Script to encode and decode an audio file using a DAC model with bfloat16 autocast.
Usage: python encode_decode_audio.py <model_path> <audio_file>
"""

import argbind
import torch
from pathlib import Path
from audiotools import AudioSignal
import dac


@argbind.bind(without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode_decode_audio(
    model_path: str = "",
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
    
    # Load model
    model = dac.DAC.load(model_path)
    model.to(device)
    model.eval()
    
    # Load audio
    signal = AudioSignal(str(audio_path))
    
    # Resample to model's sample rate if needed
    if signal.sample_rate != model.sample_rate:
        signal.resample(model.sample_rate)
    
    signal.to(device)
    
    # Encode and decode with bfloat16 autocast
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # Encode
        z = model.encode(signal.audio_data)
        
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