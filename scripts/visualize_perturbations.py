import torch
import matplotlib.pyplot as plt
import numpy as np
import argbind
from pathlib import Path
from audiotools import AudioSignal
from dac.model import DAC
from scipy.fftpack import dct
from tqdm import tqdm

def compute_mcd(mel_spec1, mel_spec2):
    """
    Compute Mel-Cepstral Distortion (MCD) between two mel-spectrograms.
    
    Args:
        mel_spec1, mel_spec2: mel-spectrograms [n_mels, n_frames]
    
    Returns:
        MCD value in dB
    """
    min_length = min(mel_spec1.shape[-1], mel_spec2.shape[-1])
    return (10 / np.log(10)) * np.mean(np.sqrt(2 * np.sum((
        dct(np.log(mel_spec1[..., :min_length] + 1e-8), axis=0, norm='ortho') -
        dct(np.log(mel_spec2[..., :min_length] + 1e-8), axis=0, norm='ortho'))**2, axis=0)))


@argbind.bind(without_prefix=True)
def analyze_latent_perturbations(
    audio_file: Path = Path(""),
    model_path: Path = Path(""),
    hop_length: int = 2048,
    n_mels: int = 64,
    window_before: int = 10,
    window_after: int = 10,
    max_batch_size: int = 16,
    sample_perturbations: str = "",
):
    """Analyze how perturbations to DAC latents affect mel-spectrogram reconstruction.
    
    Args:
        audio_file (Path): Path to input audio file
        model_path (Path): Path to model checkpoint file
        hop_length (int): Hop length for mel spectrogram
        n_mels (int): Number of mel bins
        window_before (int): Number of frames before perturbation to consider
        window_after (int): Number of frames after perturbation to consider
        max_batch_size (int): Maximum batch size for perturbation calculations
        sample_perturbations (str): Comma-separated list of perturbation magnitudes to save audio samples for
    """
    # Device handling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hardcoded parameters
    output_dir = audio_file.parent  # Same path as audio file
    
    # Parse sample perturbations
    sample_magnitudes = [float(x.strip()) for x in sample_perturbations.split(',')] if sample_perturbations.strip() else []
    
    # Load model and explicitly move to specified device
    model = DAC.load(model_path / "best/dac/weights.pth")
    model.to(device)  # Explicitly move model to GPU
    model.eval()
    
    # Load and process audio
    signal = AudioSignal(audio_file)
    signal.to(device)  # This uses the model's device
    signal.audio_data = model.preprocess(signal.audio_data, signal.sample_rate)
    
    # Compute original mel spectrogram
    mel_orig = signal.mel_spectrogram(
        n_mels=n_mels, window_length=hop_length, hop_length=hop_length).squeeze().cpu().numpy()

    # Get original latents and reconstruction
    with torch.no_grad():
        latents_orig = model.encode(signal.audio_data)
        mel_recons = AudioSignal(model.decode(latents_orig), sample_rate=signal.sample_rate).mel_spectrogram(
            n_mels=n_mels, window_length=hop_length, hop_length=hop_length).squeeze().cpu().numpy()
    
    # Compute baseline MCD
    baseline_mcd = compute_mcd(mel_orig, mel_recons)
    
    # Compute covariance matrix
    cov_matrix = np.cov(latents_orig.squeeze(0).cpu().numpy())

    # Compute square root of covariance matrix for generating correlated noise
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals.real, 0)  # Avoid negative eigenvalues
    cov_sqrt = torch.tensor(eigenvecs @ np.diag(np.sqrt(eigenvals)), device=device, dtype=torch.float32)
    
    # Plot covariance matrix
    plt.figure()
    cov_normalized = cov_matrix / np.trace(cov_matrix) * cov_matrix.shape[0]
    plt.imshow(cov_normalized, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Covariance / Trace * Dim')
    plt.title(f'Latent Covariance Matrix\nCondition Number: {np.max(eigenvals) / (np.min(eigenvals) + 1e-10):.1e}')
    plt.tight_layout()
    plt.savefig(output_dir / f"covariance_{model_path.name}.svg")
    plt.close()

    perturbation_magnitudes = np.logspace(-2, 0, 100)  # 100 logarithmically spaced from 1e-3 to 1e0

    # Process perturbations in batches to respect max_batch_size
    smoothness_errors = []
    
    for start_idx in tqdm(range(0, len(perturbation_magnitudes), max_batch_size), desc="Smoothness"):
        end_idx = min(start_idx + max_batch_size, len(perturbation_magnitudes))
        batch_magnitudes = perturbation_magnitudes[start_idx:end_idx]
        
        # Create batch of perturbations
        latents_batch = latents_orig.repeat(len(batch_magnitudes), 1, 1)  # [batch_size, channels, time]

        # Scale by magnitudes and add to latents
        latents_batch += torch.einsum(
            'ij,bjk->bik', cov_sqrt, torch.randn(len(batch_magnitudes), latents_batch.shape[1], latents_batch.shape[2], device=device)) * torch.tensor(
                batch_magnitudes, device=device).view(-1, 1, 1)
        
        # Decode batch
        with torch.no_grad():
            audio_batch = model.decode(latents_batch)
        
        # Compute mel spectrograms and MCDs for this batch
        for i in range(len(batch_magnitudes)):
            mel_pert = AudioSignal(
                audio_batch[i:i+1], sample_rate=signal.sample_rate).mel_spectrogram(
                    n_mels=n_mels, window_length=hop_length, hop_length=hop_length).squeeze().cpu().numpy()
            smoothness_errors.append(compute_mcd(mel_recons, mel_pert))
    
    # Plot smoothness analysis
    plt.figure()
    plt.semilogx(perturbation_magnitudes, smoothness_errors, 'b-', label='Perturbations')
    plt.xlabel('Perturbation')
    plt.ylabel('MCD [dB]')
    plt.xlim(min(perturbation_magnitudes), max(perturbation_magnitudes))
    plt.grid(True, alpha=0.3)
    plt.title(f'Robustness\nBaseline MCD: {baseline_mcd:.0f}dB')
    plt.tight_layout()
    plt.savefig(output_dir / f"smoothness_{model_path.name}.svg")
    plt.close()
    
    # Process sample magnitudes in batches
    for start_idx in tqdm(range(0, len(sample_magnitudes), max_batch_size), desc="Audio Sampling"):
        end_idx = min(start_idx + max_batch_size, len(sample_magnitudes))
        batch_magnitudes = sample_magnitudes[start_idx:end_idx]
        
        # Create batch of perturbations
        latents_batch = latents_orig.repeat(len(batch_magnitudes), 1, 1)  # [batch_size, channels, time]

        # Scale by magnitudes and add to latents
        latents_batch += torch.einsum(
            'ij,bjk->bik', cov_sqrt, torch.randn(len(batch_magnitudes), latents_batch.shape[1], latents_batch.shape[2], device=device)) * torch.tensor(
                batch_magnitudes, device=device).view(-1, 1, 1)
        
        # Decode batch
        with torch.no_grad():
            audio_batch = model.decode(latents_batch)
        
        # Save individual audio files from batch
        for i in range(len(batch_magnitudes)):
            perturbed_audio = AudioSignal(audio_batch[i:i+1].cpu(), sample_rate=signal.sample_rate)
            perturbed_audio.write(output_dir / f"{audio_file.stem}_{model_path.name}_{i}.wav")
    
    # Collect MCD measurements for each relative frame distance
    relative_distances = list(range(-window_before, window_after + 1))
    mcds = np.zeros(len(relative_distances))
    
    # Get all positions to perturb
    positions = list(range(window_before, latents_orig.shape[-1] - window_after))
    
    # Process positions in batches to respect max_batch_size
    for start_idx in tqdm(range(0, len(positions), max_batch_size), desc="Locality"):
        end_idx = min(start_idx + max_batch_size, len(positions))
        batch_positions = positions[start_idx:end_idx]
        
        # Create batch of perturbed latents
        latents_batch = latents_orig.repeat(len(batch_positions), 1, 1)  # [batch_size, channels, time]

        # Use advanced indexing to add noise at different positions for each batch element
        latents_batch[torch.arange(len(batch_positions), device=device), :, torch.tensor(batch_positions, device=device)] += torch.einsum(
            'ij,bj->bi', cov_sqrt, torch.randn(len(batch_positions), latents_batch.shape[1], device=device))
        
        # Decode batch
        with torch.no_grad():
            audio_batch = model.decode(latents_batch)
        
        # Convert to mel spectrograms and compute MCDs
        for i, pos in enumerate(batch_positions):
            mel_pert = AudioSignal(
                audio_batch[i:i+1], sample_rate=signal.sample_rate).mel_spectrogram(
                    n_mels=n_mels, window_length=hop_length, hop_length=hop_length).squeeze().cpu().numpy()
            
            # Extract MCD for frames around the perturbation
            for j, rel_dist in enumerate(relative_distances):
                mel_frame_idx = np.round((pos + rel_dist) * np.prod(model.encoder_rates) / hop_length).astype(int)
                
                mcds[j] += compute_mcd(mel_recons[:, mel_frame_idx:mel_frame_idx+1],
                                       mel_pert[:, mel_frame_idx:mel_frame_idx+1])
    
    # Calculate mean MCD for each relative distance
    mcds /= len(positions)
    
    relative_times = [dist * hop_length / signal.sample_rate for dist in relative_distances]

    # Plot temporal locality
    plt.figure()
    plt.plot(relative_times, mcds, 'bo-', label='Perturbations')
    plt.xlabel('Time from Perturbation [s]')
    plt.ylabel('MCD [dB]')
    plt.xlim(min(relative_times), max(relative_times))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"locality_{model_path.name}.svg")
    plt.close()


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        analyze_latent_perturbations() 