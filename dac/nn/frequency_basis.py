import math

import torch
import torch.nn as nn


class GaborLatentRefactorization(nn.Module):
    """Re-express encoder latents in a frequency-localized (Gabor) basis.

    Post-hoc, retraining-free basis change applied to the latent tensor
    ``z [B, D, T]`` along the time axis. It projects each channel's
    time-series onto a fixed, orthonormal basis of Gaussian-windowed sinusoids
    (Gabor atoms), so the refactored latents expose frequency-localized
    coordinates -- the ``j``-th time slot of the output corresponds to a
    well-defined frequency band -- rather than the entangled basis the learned
    strided encoder happens to land in. This makes attributes that depend on
    frequency-localized primitives (e.g. pitch, timbre) explicit and steerable
    in the compressed codes.

    The basis is parameter-free (no learned weights), full-rank and orthonormal
    (``G^T G = I``), so the transform is exactly inverted by :meth:`inverse`.
    The codec applies the inverse before decoding, so the pretrained decoder
    and codebook keep operating in the basis they were trained on and
    reconstruction fidelity is preserved in the round-trip sense.

    Adapted from "Structural Bottlenecks on Frequency Representation in
    End-to-End Audio Models" (arXiv:2607.08545), which introduces Gabor Latent
    Refactorization (GLRF) as a lightweight, retraining-free intervention. The
    paper's bandwidth-measurement evaluation (filter bandwidths 10-35x ->
    1.5-3x of the theoretical resolution bound) is intentionally out of scope
    here; this module ships the core mechanism -- re-expressing latents in a
    frequency-localized basis -- as an optional encode-path transform.

    Parameters
    ----------
    sigma_scale : float
        Width of the Gaussian window as a fraction of the time length. Larger
        values widen the time window (narrower frequency localization).
    """

    def __init__(self, sigma_scale: float = 0.25):
        super().__init__()
        # No parameters and no buffers: this module contributes nothing to
        # state_dict, so loading pretrained DAC weights is unaffected.
        self.sigma_scale = sigma_scale
        # Plain dict (not a ParameterDict/ModuleDict) -> not registered, so it
        # never enters state_dict. Keyed by (length, device, dtype).
        self._basis_cache = {}

    @staticmethod
    def _atom_matrix(length, sigma_scale, device, dtype):
        """Build ``length`` Gaussian-windowed cosine atoms as columns [T, T]."""
        t = torch.arange(length, device=device, dtype=dtype)
        t = t - (length - 1) / 2.0
        sigma = max(length * sigma_scale, 1.0)
        window = torch.exp(-(t**2) / (2.0 * sigma**2))
        # Distinct frequencies f_k in [0, 0.5) cycles/sample.
        k = torch.arange(length, device=device, dtype=dtype)
        freqs = k / (2.0 * length)
        phases = 2.0 * math.pi * freqs[None, :] * t[:, None]
        return window[:, None] * torch.cos(phases)

    def _basis(self, length, device, dtype):
        key = (length, str(device), str(dtype))
        basis = self._basis_cache.get(key)
        if basis is None:
            atoms = self._atom_matrix(length, self.sigma_scale, device, dtype)
            # Orthonormalize the atoms into a square, full-rank basis so that
            # G^T G = I and the transform is exactly invertible.
            basis, _ = torch.linalg.qr(atoms)
            self._basis_cache[key] = basis
        return basis

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Re-express ``z [B, D, T]`` in the frequency-localized basis (``z @ G``)."""
        basis = self._basis(z.shape[-1], z.device, z.dtype)
        return torch.matmul(z, basis)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Map ``z [B, D, T]`` back to the original basis (``z @ G^T``)."""
        basis = self._basis(z.shape[-1], z.device, z.dtype)
        return torch.matmul(z, basis.transpose(-1, -2))
