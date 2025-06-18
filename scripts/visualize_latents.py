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
    n_components: int = 64,
    pca: bool = False,
):
    """Visualize mel spectrogram and first n_components PCA components of DAC latents for an audio file.
    
    Args:
        audio_file (Path): Path to input audio file
        model_path (Path): Path to model checkpoint file
        n_components (int): Number of PCA components to visualize
        pca (bool): Whether to perform PCA on latents
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
    mel_spec = signal.mel_spectrogram(n_mels=n_components, window_length=4096, hop_length=2048).squeeze().cpu().numpy()
    
    # Perform PCA on latents
    latents = latents.squeeze().cpu().numpy()  # [n_frames, n_dims]
    if pca:
        pca = PCA(n_components=n_components)
        latents = pca.fit_transform(latents.T).T  # [n_components, n_frames]

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
        latents,
        aspect='auto',
        origin='lower',
        extent=[0, signal.signal_duration, 0, latents.shape[0]]
    )
    ax2.set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(audio_file.parent / f"latents_{model_path.name}{'_pca' if pca else ''}.svg", format='svg')
    plt.close()

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        visualize() 