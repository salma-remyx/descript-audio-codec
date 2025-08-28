import torch
import matplotlib.pyplot as plt
import numpy as np
import argbind
from pathlib import Path
from audiotools import AudioSignal
from audiotools.data import transforms
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


def _mean_ci(values):
    arr = np.array(values, dtype=float)
    mean = float('nan')
    ci95 = float('nan')
    if arr.size > 0:
        mean = float(arr.mean())
        if arr.size > 1:
            ci95 = 1.96 * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return mean, ci95

@argbind.bind(without_prefix=True)
def analyze_latent_perturbations(
    dataset_path: Path = Path(""),
    model_path: Path = Path(""),
    hop_length: int = 2048,
    n_mels: int = 64,
    window_before: int = 10,
    window_after: int = 10,
    max_batch_size: int = 16,
    sample_perturbations: str = "",
):
    """Analyze how perturbations to DAC latents affect mel-spectrogram reconstruction over a dataset.
    
    Args:
        dataset_path (Path): Root directory. Recursively finds all .wav files beneath this path.
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
    
    # Output directory
    output_dir = Path(dataset_path) if str(dataset_path) != "" else Path(".")
    
    # Parse sample perturbations
    sample_magnitudes = [float(x.strip()) for x in sample_perturbations.split(',') if x.strip()] if sample_perturbations.strip() else []
    
    # Load model package and explicitly move to specified device
    model = DAC.load(model_path / "best/dac/package.pth")
    model.to(device)  # Explicitly move model to GPU
    model.eval()
    
    
    # Prepare transforms (match training postprocess)
    _vol_norm = transforms.VolumeNorm(db=("const", -16.0))
    _rescale = transforms.RescaleAudio(val=1.0)

    # Collect all wav files
    wav_files = sorted(Path(dataset_path).rglob("*.wav")) if str(dataset_path) != "" else []

    # Aggregation containers
    baseline_mcd_list = []
    cond_number_list = []
    cov_accum = None

    perturbation_magnitudes = np.logspace(-2, 0, 100)
    smoothness_sum = np.zeros_like(perturbation_magnitudes, dtype=float)
    smoothness_count = 0

    relative_distances = list(range(-window_before, window_after + 1))
    locality_sum = np.zeros(len(relative_distances), dtype=float)
    locality_count = 0

    for wav_path in tqdm(wav_files, desc="Files"):
        # Load and process audio
        signal = AudioSignal(str(wav_path))
        if signal.sample_rate != model.sample_rate:
            signal.resample(model.sample_rate)
        signal.to(device)
        
        # Apply training postprocess via _instantiate/_transform: VolumeNorm -> RescaleAudio
        vn_params = _vol_norm._instantiate(None)
        signal = _vol_norm._transform(signal, **vn_params)
        rs_params = _rescale._instantiate(None)
        signal = _rescale._transform(signal, **rs_params)
        # Ensure model padding behavior still applies before encoding
        signal.audio_data = model.preprocess(signal.audio_data, signal.sample_rate)
        
        # Compute original mel spectrogram
        mel_orig = signal.mel_spectrogram(
            n_mels=n_mels, window_length=hop_length, hop_length=hop_length).squeeze().cpu().numpy()
        
        # Get original latents and reconstruction
        with torch.no_grad():
            latents_orig = model.encode(signal.audio_data)
            if isinstance(latents_orig, tuple):
                latents_orig = latents_orig[0]
            mel_recons = AudioSignal(model.decode(latents_orig), sample_rate=signal.sample_rate).mel_spectrogram(
                n_mels=n_mels, window_length=hop_length, hop_length=hop_length).squeeze().cpu().numpy()
        
        # Baseline MCD for this file
        baseline_mcd = compute_mcd(mel_orig, mel_recons)
        baseline_mcd_list.append(float(baseline_mcd))
        
        # Covariance and condition number for this file
        cov_matrix = np.cov(latents_orig.squeeze(0).cpu().numpy())
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals.real, 0)
        cond_number = float(np.max(eigenvals) / (np.min(eigenvals) + 1e-10))
        cond_number_list.append(cond_number)
        cov_sqrt = torch.tensor(eigenvecs @ np.diag(np.sqrt(eigenvals)), device=device, dtype=torch.float32)
        
        cov_normalized = cov_matrix / np.trace(cov_matrix) * cov_matrix.shape[0]
        if cov_accum is None:
            cov_accum = np.zeros_like(cov_normalized)
        cov_accum += cov_normalized
        
        # Smoothness (robustness) curve for this file
        per_file_errors = []
        for start_idx in range(0, len(perturbation_magnitudes), max_batch_size):
            end_idx = min(start_idx + max_batch_size, len(perturbation_magnitudes))
            batch_magnitudes = perturbation_magnitudes[start_idx:end_idx]
            
            latents_batch = latents_orig.repeat(len(batch_magnitudes), 1, 1)
            latents_batch += torch.einsum(
                'ij,bjk->bik', cov_sqrt, torch.randn(len(batch_magnitudes), latents_batch.shape[1], latents_batch.shape[2], device=device)) * torch.tensor(
                    batch_magnitudes, device=device).view(-1, 1, 1)
            with torch.no_grad():
                audio_batch = model.decode(latents_batch)
            for i in range(len(batch_magnitudes)):
                mel_pert = AudioSignal(
                    audio_batch[i:i+1], sample_rate=signal.sample_rate).mel_spectrogram(
                        n_mels=n_mels, window_length=hop_length, hop_length=hop_length).squeeze().cpu().numpy()
                per_file_errors.append(compute_mcd(mel_recons, mel_pert))
        smoothness_sum += np.array(per_file_errors, dtype=float)
        smoothness_count += 1
        
        # Optional audio sampling per file
        for start_idx in range(0, len(sample_magnitudes), max_batch_size):
            end_idx = min(start_idx + max_batch_size, len(sample_magnitudes))
            batch_magnitudes = sample_magnitudes[start_idx:end_idx]
            if len(batch_magnitudes) == 0:
                break
            latents_batch = latents_orig.repeat(len(batch_magnitudes), 1, 1)
            latents_batch += torch.einsum(
                'ij,bjk->bik', cov_sqrt, torch.randn(len(batch_magnitudes), latents_batch.shape[1], latents_batch.shape[2], device=device)) * torch.tensor(
                    batch_magnitudes, device=device).view(-1, 1, 1)
            with torch.no_grad():
                audio_batch = model.decode(latents_batch)
            for i in range(len(batch_magnitudes)):
                perturbed_audio = AudioSignal(audio_batch[i:i+1].cpu(), sample_rate=signal.sample_rate)
                perturbed_audio.write(output_dir / f"{Path(wav_path).stem}_{model_path.name}_{start_idx + i}.wav")
        
        # Locality curve for this file (average over positions)
        try:
            encoder_strides = model.encoder_strides
        except:
            encoder_strides = model.encoder_rates
        positions = list(range(window_before, latents_orig.shape[-1] - window_after))
        per_file_mcds = np.zeros(len(relative_distances), dtype=float)
        for start_idx in range(0, len(positions), max_batch_size):
            end_idx = min(start_idx + max_batch_size, len(positions))
            batch_positions = positions[start_idx:end_idx]
            latents_batch = latents_orig.repeat(len(batch_positions), 1, 1)
            latents_batch[torch.arange(len(batch_positions), device=device), :, torch.tensor(batch_positions, device=device)] += torch.einsum(
                'ij,bj->bi', cov_sqrt, torch.randn(len(batch_positions), latents_batch.shape[1], device=device))
            with torch.no_grad():
                audio_batch = model.decode(latents_batch)
            for i, pos in enumerate(batch_positions):
                mel_pert = AudioSignal(
                    audio_batch[i:i+1], sample_rate=signal.sample_rate).mel_spectrogram(
                        n_mels=n_mels, window_length=hop_length, hop_length=hop_length).squeeze().cpu().numpy()
                for j, rel_dist in enumerate(relative_distances):
                    mel_frame_idx = int(np.round((pos + rel_dist) * np.prod(encoder_strides) / hop_length))
                    per_file_mcds[j] += compute_mcd(mel_recons[:, mel_frame_idx:mel_frame_idx+1],
                                                    mel_pert[:, mel_frame_idx:mel_frame_idx+1])
        if len(positions) > 0:
            per_file_mcds /= len(positions)
            locality_sum += per_file_mcds
            locality_count += 1

    # Aggregate statistics
    cov_mean = cov_accum / max(len(wav_files), 1)
    cond_mean, cond_ci = _mean_ci(cond_number_list)
    baseline_mean, baseline_ci = _mean_ci(baseline_mcd_list)

    smoothness_mean = smoothness_sum / max(smoothness_count, 1)

    # Use sample rate from the model (all signals resampled to this)
    relative_times = [dist * hop_length / model.sample_rate for dist in relative_distances]
    locality_mean = locality_sum / max(locality_count, 1)

    # Plot mean covariance with condition number CI
    plt.figure()
    plt.imshow(cov_mean, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Covariance / Trace * Dim')
    plt.title(f'Latent Covariance Matrix\nCondition Number: {cond_mean:.1e} ± {cond_ci:.1e}')
    plt.tight_layout()
    plt.savefig(output_dir / f"covariance_{model_path.name}_aggregate.svg")
    plt.close()

    # Plot smoothness mean curve with baseline MCD CI
    plt.figure()
    plt.semilogx(perturbation_magnitudes, smoothness_mean, 'b-', label='Perturbations')
    plt.xlabel('Perturbation')
    plt.ylabel('MCD [dB]')
    plt.xlim(min(perturbation_magnitudes), max(perturbation_magnitudes))
    plt.grid(True, alpha=0.3)
    plt.title(f'Robustness\nBaseline MCD: {baseline_mean:.2f} ± {baseline_ci:.2f} dB')
    plt.tight_layout()
    plt.savefig(output_dir / f"smoothness_{model_path.name}_aggregate.svg")
    plt.close()

    # Plot temporal locality (mean across files)
    plt.figure()
    plt.plot(relative_times, locality_mean, 'bo-', label='Perturbations')
    plt.xlabel('Time from Perturbation [s]')
    plt.ylabel('MCD [dB]')
    plt.xlim(min(relative_times), max(relative_times))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"locality_{model_path.name}_aggregate.svg")
    plt.close()


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        analyze_latent_perturbations() 