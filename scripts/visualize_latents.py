import torch
import matplotlib.pyplot as plt
import numpy as np
import argbind
from pathlib import Path
from audiotools import AudioSignal
from dac.model import DAC
from sklearn.decomposition import PCA

@argbind.bind(without_prefix=True)
def visualize(
    audio_file: Path = Path(""),
    model_path: Path = Path(""),
):
    """Visualize mel spectrogram and first 64 PCA components of DAC latents for an audio file.
    
    Args:
        audio_file (Path): Path to input audio file
        model_path (Path): Path to model checkpoint file
    """
    # Load model
    model = DAC.load(model_path / "best/dac/weights.pth")
    model.eval()
    
    # Load and process audio
    signal = AudioSignal(audio_file)
    signal.to(model.device)
    
    # Get DAC latents
    with torch.no_grad():
        latents = model.encode(signal.audio_data)
    
    # Compute mel spectrogram
    mel_spec = signal.mel_spectrogram(n_mels=64, window_length=4096, hop_length=2048).squeeze().cpu().numpy()
    
    # Perform PCA on latents
    latents = latents.squeeze().cpu().numpy()  # [n_frames, n_dims]
    pca = PCA(n_components=64)
    latents_pca = pca.fit_transform(latents.T).T  # [64, n_frames]
    
    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Plot mel spectrogram
    ax1.imshow(
        np.log(mel_spec + 1e-8), 
        aspect='auto', 
        origin='lower',
        extent=[0, signal.signal_duration, 0, mel_spec.shape[0]]
    )
    
    # Plot PCA components
    ax2.imshow(
        latents_pca,
        aspect='auto',
        origin='lower',
        extent=[0, signal.signal_duration, 0, latents_pca.shape[0]]
    )
    ax2.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(audio_file.with_suffix('.svg'), format='svg')
    plt.close()

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        visualize() 