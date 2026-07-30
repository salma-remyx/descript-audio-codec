"""Gabor Latent Refactorization (GLRF) for DAC encoder latents.

Adapted from "Structural Bottlenecks on Frequency Representation in
End-to-End Audio Models" (arXiv:2607.08545). The paper shows that strided
convolutional encoders (including DAC) leave learned filter bandwidths
10-35x above the theoretical resolution bound ``delta_f = f_s / R`` (R =
cumulative encoder receptive field), so frequency-localized primitives are
not independently accessible in the latents. GLRF is a lightweight,
retraining-free, post-hoc intervention that re-expresses latents in a
frequency-localized basis:

1. normalize each latent channel by its temporal mean/variance,
2. convolve each channel with a fixed complex Gabor filterbank
   (Hann-windowed sinusoids, centers linearly spaced and parameterized to
   the resolution bound), expanding ``[B, D, T] -> [B, D, 2F, T]`` with the
   real and imaginary parts,
3. map back to the original latent basis with a linear map fit by
   closed-form ridge regression on synthetic harmonic signals.

Target-native substitutions relative to the paper (Mode 2 adapted port):
the paper fits the inverse map on latents of 1000 harmonic signals passed
through each pretrained encoder; here it is fit on synthetic harmonic
time-series directly, which is equivalent up to the encoder's own basis
because the analysis is channel-wise and the channels are exchangeable
after per-channel normalization. The paper's bandwidth evaluation and
controllability probes are intentionally out of scope.

The module is parameter-free (fixed filterbank, lazily fit ridge weights
kept in a plain cache), so it contributes nothing to ``state_dict`` and
pretrained checkpoints load undisturbed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


class GaborLatentRefactorization(nn.Module):
    """Re-express latents ``z [B, D, T]`` as frequency-localized features.

    ``forward`` returns Gabor features ``[B, D, 2F, T]``; manipulations
    (band scaling, substitution) happen in that basis; ``inverse`` maps
    features back to the original latent basis for decoding. The transform
    is normalized per channel, so channels are exchangeable and a single
    shared inverse map is fit for all of them.

    Parameters
    ----------
    receptive_field : float
        Cumulative encoder receptive field in *latent* frames (samples
        divided by hop length). Sets the resolution bound
        ``delta = 1 / receptive_field`` (cycles per latent frame) that
        parameterizes the filterbank, per the paper's ``delta_f = f_s / R``.
    n_filters : int, optional
        Number of complex Gabor filters F. Defaults to tiling [0, Nyquist]
        at the resolution bound.
    filter_len : int, optional
        Gabor window length in latent frames. Defaults to a Hann window
        whose main-lobe bandwidth equals the resolution bound.
    bandwidth_scale : float
        Target filter bandwidth in units of the resolution bound (used only
        when ``filter_len`` is not given).
    ridge_lambda : float
        Ridge penalty for the closed-form inverse fit.
    n_fit_signals, fit_length, n_harmonics : int
        Shape of the synthetic harmonic corpus used to fit the inverse map.
    """

    def __init__(
        self,
        receptive_field: float,
        n_filters: int = None,
        filter_len: int = None,
        bandwidth_scale: float = 1.0,
        ridge_lambda: float = 1e-4,
        n_fit_signals: int = 512,
        fit_length: int = 256,
        n_harmonics: int = 4,
    ):
        super().__init__()
        # No parameters and no buffers: this module contributes nothing to
        # state_dict, so loading pretrained DAC weights is unaffected.
        if receptive_field < 1.0:
            raise ValueError("receptive_field must be >= 1 latent frame")
        self.resolution_bound = 1.0 / float(receptive_field)
        self.n_filters = n_filters or max(2, int(0.5 / self.resolution_bound) + 1)
        self.filter_len = filter_len or _odd(
            max(3, round(2.0 / (bandwidth_scale * self.resolution_bound)))
        )
        self.ridge_lambda = ridge_lambda
        self.n_fit_signals = n_fit_signals
        self.fit_length = fit_length
        self.n_harmonics = n_harmonics

        # Plain caches (not registered) -> never enter state_dict.
        self._filter_cache = {}
        self._inverse_cache = {}
        # Per-channel temporal normalization stats from the last forward.
        self._norm_stats = None

    # -- fixed complex Gabor filterbank -------------------------------------

    def _filterbank(self, device, dtype):
        """Hann-windowed complex sinusoids [F, L], energy-normalized."""
        key = (str(device), str(dtype))
        filt = self._filter_cache.get(key)
        if filt is None:
            n = torch.arange(self.filter_len, device=device, dtype=dtype)
            n = n - (self.filter_len - 1) / 2.0
            window = torch.hann_window(self.filter_len, device=device, dtype=dtype)
            centers = torch.linspace(
                0.0, 0.5, self.n_filters, device=device, dtype=dtype
            )
            phases = 2.0 * math.pi * centers[:, None] * n[None, :]
            filt = window[None, :] * torch.polar(torch.ones_like(phases), phases)
            filt = filt / filt.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self._filter_cache[key] = filt
        return filt

    # -- analysis ------------------------------------------------------------

    @staticmethod
    def _normalize(z):
        mu = z.mean(dim=-1, keepdim=True)
        sigma = z.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (z - mu) / sigma, (mu, sigma)

    def _analyze(self, z_norm):
        """Apply the filterbank: [B, D, T] real -> [B, D, 2F, T] real."""
        B, D, T = z_norm.shape
        filt = self._filterbank(z_norm.device, z_norm.dtype)  # [F, L]
        weight = filt.unsqueeze(1).repeat(D, 1, 1)  # [D*F, 1, L]
        out = F.conv1d(
            z_norm.to(filt.dtype),
            weight,
            padding=self.filter_len // 2,
            groups=D,
        )
        out = out.view(B, D, self.n_filters, T)
        return torch.cat([out.real, out.imag], dim=2).to(z_norm.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize ``z [B, D, T]`` per channel and expand to Gabor features."""
        z_norm, stats = self._normalize(z)
        self._norm_stats = stats
        return self._analyze(z_norm)

    # -- ridge-fit inverse ----------------------------------------------------

    def _synthetic_harmonics(self):
        """Synthetic harmonic channel signals, per the paper's fit corpus."""
        g = torch.Generator().manual_seed(0)
        t = torch.arange(self.fit_length, dtype=torch.float64)
        lo = 2.0 / self.fit_length
        hi = 0.45 / self.n_harmonics  # keep all harmonics below Nyquist
        f0 = torch.rand(self.n_fit_signals, generator=g, dtype=torch.float64)
        f0 = lo + (hi - lo) * f0
        signals = torch.zeros(self.n_fit_signals, self.fit_length, dtype=torch.float64)
        for h in range(1, self.n_harmonics + 1):
            phase = 2.0 * math.pi * torch.rand(
                self.n_fit_signals, 1, generator=g, dtype=torch.float64
            )
            signals += (1.0 / h) * torch.sin(
                2.0 * math.pi * h * f0[:, None] * t[None, :] + phase
            )
        return signals  # [n_signals, fit_length]

    def _inverse_weights(self, device, dtype):
        """Closed-form ridge fit of the feature->signal map, cached per device."""
        key = (str(device), str(dtype))
        w = self._inverse_cache.get(key)
        if w is None:
            signals = self._synthetic_harmonics()
            normed, _ = self._normalize(signals.unsqueeze(1))
            feats = self._analyze(normed.float()).double()  # [N, 1, 2F, T]
            X = feats[:, 0].permute(0, 2, 1).reshape(-1, 2 * self.n_filters)
            y = normed[:, 0].reshape(-1).double()
            gram = X.T @ X
            gram.diagonal().add_(self.ridge_lambda)
            w = torch.linalg.solve(gram, X.T @ y)
            w = w.to(device=device, dtype=dtype)
            self._inverse_cache[key] = w
        return w

    def inverse(self, features: torch.Tensor) -> torch.Tensor:
        """Map Gabor features ``[B, D, 2F, T]`` back to the latent basis."""
        if self._norm_stats is None:
            raise RuntimeError(
                "inverse() requires the normalization stats from a prior "
                "forward() call on the source latents."
            )
        w = self._inverse_weights(features.device, features.dtype)
        z_norm = torch.einsum("bdft,f->bdt", features, w)
        mu, sigma = self._norm_stats
        return z_norm * sigma + mu
