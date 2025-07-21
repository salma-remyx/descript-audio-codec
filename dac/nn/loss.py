import typing
from typing import List
import signal
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torchaudio
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn
from transformers import WavLMModel, AutoFeatureExtractor
from silero_vad import load_silero_vad, get_speech_timestamps


class L1Loss(nn.L1Loss):
    """L1 Loss between AudioSignals. Defaults
    to comparing ``audio_data``, but any
    attribute of an AudioSignal can be used.

    Parameters
    ----------
    attribute : str, optional
        Attribute of signal to compare, defaults to ``audio_data``.
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: AudioSignal, y: AudioSignal):
        """
        Parameters
        ----------
        x : AudioSignal
            Estimate AudioSignal
        y : AudioSignal
            Reference AudioSignal

        Returns
        -------
        torch.Tensor
            L1 loss between AudioSignal attributes.
        """
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(
        self,
        scaling: int = True,
        reduction: str = "mean",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal):
        eps = 1e-8
        # nb, nc, nt
        if isinstance(x, AudioSignal):
            references = x.audio_data
            estimates = y.audio_data
        else:
            references = x
            estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature


class L2LatentsLoss(nn.Module):
    """Compute L2 penalty on latents to encourage Gaussian distribution.
    
    Parameters
    ----------
    weight : float, optional
        Weight of this loss, defaults to 1.0
    """
    def forward(self, latents: torch.Tensor):
        """Compute L2 penalty on latents.
        
        Parameters
        ----------
        latents : Tensor[B x D x T]
            Latent representations from the encoder
            
        Returns
        -------
        Tensor[1]
            L2 penalty loss on latents
        """
        # Compute mean squared value across batch and time dimensions
        return torch.mean(torch.pow(latents, 2))


class WavLMLoss(nn.Module):
    """WavLM loss that returns both cosine and MSE losses between embeddings.
    
    This loss computes the similarity between predicted WavLM embeddings and
    embeddings extracted from the target audio using a pre-trained WavLM model.
    
    Features:
    - Uses Voice Activity Detection (VAD) to mask non-speech regions,
      computing the loss only on detected speech segments.
    - This is particularly useful for noisy audio where WavLM embeddings on 
      non-speech regions (silence, noise) are not meaningful.
    - The loss is computed on the embeddings of the target audio, not the predicted audio.
    
    Parameters
    ----------
    device : torch.device
        Device to run the model on
    sample_rate : int, optional
        Input audio sample rate, by default 44100
    """
    def __init__(self, device, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = 16000
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=self.sample_rate
        ).to(device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        self.wavlm_model.eval()  # Freeze the model
        
        # Load Silero VAD
        self.vad_model = load_silero_vad(onnx=False).to(device)
        self.vad_model.eval()
        
    def forward(self, pred_embeddings: torch.Tensor, target_signal: AudioSignal) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute WavLM losses between predicted and target audio.
        
        The loss is weighted by the proportion of speech in each embedding frame.
        VAD is used to detect speech regions at the audio sample level, then
        area-interpolated to create frame weights (0-1) representing how much of
        each embedding frame contains speech. This provides smooth transitions and
        accurate weighting for frames that are partially speech/silence.
        The loss is computed on the embeddings of the target audio, not the predicted audio.
        
        Args:
            pred_embeddings: Predicted WavLM embeddings from the model [B x 1024 x T]
            target_signal: Target audio signal
            
        Returns:
            Tuple of (cosine_loss, mse_loss)
        """print(f"  Input shapes: pred_embeddings={pred_embeddings.shape}, audio={target_signal.audio_data.shape}")
        
        # First get target embeddings with no_grad and autocast
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Resample audio and move to CPU immediately
            resampled_audio = self.resampler(target_signal.audio_data.squeeze(1))
            resampled_audio_np = resampled_audio.cpu().numpy()

            # Create VAD mask for each batch item
            vad_masks = []
            for i in range(resampled_audio.shape[0]):
                # Get speech timestamps for this audio
                audio_16k = resampled_audio[i].squeeze()
                
                try:
                    # Add timeout protection
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    exception_queue = queue.Queue()
                    
                    def vad_worker():
                        try:
                            timestamps = get_speech_timestamps(
                                audio_16k,
                                self.vad_model,
                                sampling_rate=self.sample_rate,
                                return_seconds=False  # Return sample indices
                            )
                            result_queue.put(timestamps)
                        except Exception as e:
                            exception_queue.put(e)
                    
                    thread = threading.Thread(target=vad_worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=5.0)  # 5 second timeout
                    
                    if thread.is_alive():
                        speech_timestamps = []
                    elif not exception_queue.empty():
                        e = exception_queue.get()
                        speech_timestamps = []
                    else:
                        speech_timestamps = result_queue.get()
                        print(f"      Found {len(speech_timestamps)} speech segments")
                except Exception as e:
                    speech_timestamps = []  # Fallback to empty
                
                # Create binary mask: 1 for speech, 0 for non-speech
                mask = torch.zeros(audio_16k.shape[0], device=resampled_audio.device)
                for timestamp in speech_timestamps:
                    mask[timestamp['start']:timestamp['end']] = 1.0
                
                vad_masks.append(mask)

            del resampled_audio  # Clean up GPU memory

            # WavLM has ~20ms frame shift at 16kHz (320 samples)
            # Need to downsample mask from audio samples to embedding frames
            # Use 'area' mode to get the average speech proportion in each frame
            # This creates a weighted mask (0-1) representing how much of each
            # embedding frame contains speech vs non-speech
            vad_mask = F.interpolate(
                torch.stack(vad_masks).unsqueeze(1).float(),
                size=pred_embeddings.shape[-1],
                mode="area"  # Average pooling - gives proportion of speech per frame
            )
            
            # Process target audio through WavLM
            inputs = self.feature_extractor(
                resampled_audio_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).to(pred_embeddings.device)
            
            # Get target embeddings using mixed precision
            target_embeddings = self.wavlm_model(**inputs).last_hidden_state.transpose(1, 2)
            del inputs  # Clean up GPU memory
            
            # Resample target embeddings if lengths don't match
            if target_embeddings.shape[-1] != pred_embeddings.shape[-1]:
                target_embeddings = F.interpolate(
                    target_embeddings,
                    size=pred_embeddings.shape[-1],
                    mode="linear",
                    align_corners=False
                )

        # Compute weighted mean over frames
        # speech_weight is the total weight (sum of speech proportions)
        speech_weight = vad_mask.sum()
        
        if speech_weight > 0:
            cosine_loss = 1 - (vad_mask.squeeze(1) * F.cosine_similarity(pred_embeddings, target_embeddings, dim=1)).sum() / speech_weight
        else:
            # No speech detected, return zero loss
            cosine_loss = torch.tensor(0.0, device=pred_embeddings.device)

        if speech_weight > 0:
            mse_loss = (vad_mask * (pred_embeddings - target_embeddings) ** 2).sum() / (speech_weight * pred_embeddings.shape[1])
        else:
            mse_loss = torch.tensor(0.0, device=pred_embeddings.device)
        
        # Clean up
        del target_embeddings
        del vad_mask
        torch.cuda.empty_cache()
        
        return cosine_loss, mse_loss
