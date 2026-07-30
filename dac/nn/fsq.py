import math

import torch
import torch.nn as nn

from dac.nn.layers import WNConv1d

"""
Finite Scalar Quantization (FSQ), adapted from the ``AuEmoCodec`` component of

    AuEmoChat: Authentic Emotion Understanding and Rendering for Conversational
    Speech Synthesis (https://arxiv.org/abs/2607.15755v1)

AuEmoCodec learns a discrete token space from audio via finite scalar
quantization. This module ports that quantization mechanism into the Descript
Audio Codec as a codebook-free alternative to the learned RVQ codebooks.

FSQ projects the continuous latent into a low-dimensional space, bounds it with
tanh, and rounds each dimension onto a fixed grid of ``L_d`` integer levels.
The "codebook" is the implicit Cartesian product of the per-dimension level
sets (``prod(levels)`` entries), so:

  * there is no learned embedding table, and therefore no codebook collapse,
  * no commitment / codebook auxiliary loss is required (both are returned as
    zeros), and
  * gradients still reach the encoder through the straight-through estimator.

Everything outside the quantization mechanism itself (AuEmoChat's emotion
labels, the AuEmoToMe token-merging, the autoregressive text-speech model and
Authentic Emotion Flow Matching, and the NCSSD-EmCap benchmark) is intentionally
out of scope: a conversational-speech pipeline has no call site in a pure-conv
audio codec. Only the FSQ quantizer is ported.
"""


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient (identity on the backward pass)."""
    return (x.round() - x).detach() + x


def _default_levels(codebook_size: int, max_dim: int = 8) -> list:
    """Pick per-dimension levels (each in [2, 8]) using the fewest dimensions
    such that ``prod(levels) >= codebook_size``. Returns a list of ints."""
    codebook_size = max(2, int(codebook_size))
    for d in range(1, max_dim + 1):
        level = max(2, min(8, math.ceil(codebook_size ** (1.0 / d))))
        if level**d >= codebook_size:
            return [level] * d
    return [8] * max_dim


class FiniteScalarQuantize(nn.Module):
    """Codebook-free quantizer matching the ``VectorQuantize`` I/O contract so
    it can be dropped into ``ResidualVectorQuantize`` in place of RVQ.

    Parameters mirror ``VectorQuantize.__init__`` for drop-in construction:
        input_dim     : dimensionality of the incoming latent ``z``.
        codebook_size : target number of discrete codes; the actual codebook
                        size is ``prod(levels)`` (>= ``codebook_size``).
        codebook_dim  : treated as an upper bound on the number of FSQ
                        dimensions; the actual dimensionality is
                        ``len(levels)``.
        levels        : optional explicit per-dimension level list, overriding
                        the derivation from ``codebook_size`` / ``codebook_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        levels: list = None,
    ):
        super().__init__()
        if levels is None:
            levels = _default_levels(codebook_size, max_dim=max(2, int(codebook_dim)))
        levels = [int(level) for level in levels]
        assert len(levels) >= 1, "FSQ requires at least one quantized dimension"
        assert all(level >= 2 for level in levels), "each FSQ level must be >= 2"

        self._levels_list = levels
        _levels = torch.tensor(levels, dtype=torch.float32)
        # Mixed-radix basis: basis[d] = prod(levels[:d]) -> [1, l0, l0*l1, ...]
        _basis = torch.cumprod(_levels, dim=0) // _levels
        self.register_buffer("_levels", _levels, persistent=False)
        self.register_buffer("_basis", _basis, persistent=False)

        # The actual FSQ dimensionality may differ from the requested hint.
        self.codebook_dim = len(levels)
        self.codebook_size = int(torch.prod(_levels).item())

        self.in_proj = WNConv1d(input_dim, self.codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(self.codebook_dim, input_dim, kernel_size=1)

    @property
    def _half_width(self) -> torch.Tensor:
        # half-width of each dimension's level grid, shape [d] for broadcasting.
        return (self._levels - 1) / 2.0

    def _quantize(self, z_e: torch.Tensor):
        """Quantize projected latents ``z_e`` [B x d x T] in [-1, 1] space.

        Returns the continuous quantized codes (with straight-through gradients)
        and the integer code indices."""
        half = self._half_width.view(1, -1, 1)
        bounded = torch.tanh(z_e)  # [-1, 1]
        scaled = bounded * half + half  # map to [0, L - 1]
        numerators = _round_ste(scaled)  # straight-through integer codes
        codes = numerators / half - 1.0  # back to continuous [-1, 1]
        indices = (numerators * self._basis.view(1, -1, 1)).sum(dim=1)  # [B, T]
        return codes, indices

    def _indices_to_numerators(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode code indices [B x T] to per-dimension integer codes [B x d x T]."""
        numerators = (indices.unsqueeze(-1) // self._basis) % self._levels  # [B, T, d]
        return numerators.permute(0, 2, 1).contiguous()  # [B, d, T]

    def forward(self, z: torch.Tensor):
        """Quantize ``z`` using a fixed scalar grid.

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]   quantized continuous representation
        Tensor[B]           commitment loss (always zero for FSQ)
        Tensor[B]           codebook loss (always zero for FSQ)
        Tensor[B x T]       discrete code indices
        Tensor[B x d x T]   projected latents before quantization
        """
        z_e = self.in_proj(z)  # [B, d, T]
        codes, indices = self._quantize(z_e)  # codes in [-1, 1]
        z_q = self.out_proj(codes)  # [B, D, T]

        # FSQ has no learned codebook, so there is nothing to commit to or to
        # update -- both auxiliary losses are exactly zero by construction.
        zeros = z_e.new_zeros(z_e.shape[0])
        return z_q, zeros, zeros, indices, z_e

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """embed_id [B x T] -> code vectors [B x T x d]."""
        return self.decode_code(embed_id).permute(0, 2, 1)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """embed_id [B x T] -> continuous code vectors [B x d x T]."""
        numerators = self._indices_to_numerators(embed_id)
        half = self._half_width.view(1, -1, 1)
        return numerators / half - 1.0

    def decode_latents(self, latents: torch.Tensor):
        """latents [B x d x T] -> (continuous codes [B x d x T], indices [B x T])."""
        return self._quantize(latents)
