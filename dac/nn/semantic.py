"""Coarse-rate semantic codebook branch for an acoustic codec.

Adapted from X-Codec ("Codec Does Matter: Exploring the Semantic
Shortcoming of Codec for Audio Language Model", arXiv:2408.17175).

X-Codec observes that acoustic codecs (EnCodec, DAC) are optimized for
reconstruction and therefore discard the *semantic* content a downstream
audio language model needs. It restores that content with a separate
semantic codebook, derived from frozen HuBERT features, whose discrete
codes are prepended to the acoustic RVQ codes and whose restored
embedding is fused back into the codec latent before decoding.

This is an adapted port (Mode 2): the semantic *mechanism* -- a separate
low-rate semantic codebook whose embedding is fused into the acoustic
latent before quantization and whose discrete codes augment the acoustic
codes -- is kept intact, while the auxiliary frozen-HuBERT feature
extractor (an external model DAC does not host) is replaced by a
parameter-light, target-native proxy: a 1x1 projection of the codec's own
encoder latent, average-pooled to a coarse frame rate that emulates the
slow, low-rate semantic stream HuBERT provides.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dac.nn.layers import WNConv1d
from dac.nn.quantize import VectorQuantize


class SemanticQuantizer(nn.Module):
    """Coarse-rate semantic codebook fused into an acoustic codec latent.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the acoustic latent ``z`` (``DAC.latent_dim``).
    downsample : int
        Coarse-rate factor for the semantic stream. The acoustic latent is
        average-pooled by this factor, emulating the low frame rate of a
        frozen semantic extractor such as HuBERT.
    codebook_size : int
        Number of entries in the semantic codebook.
    codebook_dim : int
        Per-entry dimensionality used for the factorized nearest-neighbor
        lookup (see ``dac.nn.quantize.VectorQuantize``).
    """

    def __init__(
        self,
        input_dim: int,
        downsample: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 32,
    ):
        super().__init__()
        self.downsample = downsample
        # Target-native stand-in for frozen HuBERT features: a learnable
        # projection of the codec's own latent, downsampled to a coarse
        # frame rate.
        self.semantic_proj = WNConv1d(input_dim, input_dim, kernel_size=1)
        self.quantizer = VectorQuantize(
            input_dim=input_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )
        # Restore + fusion projection back into the acoustic latent space,
        # mirroring X-Codec's semantic-restore module that adds into z.
        self.restore_proj = WNConv1d(input_dim, input_dim, kernel_size=1)

    def _coarse(self, z: torch.Tensor) -> torch.Tensor:
        """Project ``z`` to the coarse semantic frame rate."""
        t = z.shape[-1]
        t_sem = max(1, math.ceil(t / self.downsample))
        z_sem = self.semantic_proj(z)
        return F.adaptive_avg_pool1d(z_sem, t_sem)

    def forward(self, z: torch.Tensor):
        """Compute the semantic branch for an acoustic latent.

        Parameters
        ----------
        z : Tensor[B x D x T]
            Acoustic latent (output of ``DAC.encoder``).

        Returns
        -------
        Tensor[B x D x T]
            Restored semantic embedding at the acoustic frame rate, ready to
            fuse into ``z`` before the acoustic RVQ.
        Tensor[B x T_s]
            Discrete semantic codes at the coarse frame rate
            (``T_s = ceil(T / downsample)``), to prepend to the acoustic
            codes for a downstream audio language model.
        dict
            Mean-reduced VQ losses for the semantic codebook, keyed
            ``vq/commitment_loss`` and ``vq/codebook_loss``.
        """
        z_sem = self._coarse(z)
        z_q_sem, commit_loss, codebook_loss, codes, _ = self.quantizer(z_sem)

        # Upsample the restored embedding back to the acoustic frame rate.
        z_q_sem = F.interpolate(z_q_sem, size=z.shape[-1], mode="nearest")
        embed = self.restore_proj(z_q_sem)

        return (
            embed,
            codes,
            {
                "vq/commitment_loss": commit_loss.mean(),
                "vq/codebook_loss": codebook_loss.mean(),
            },
        )

    def from_codes(self, codes: torch.Tensor, target_len: int) -> torch.Tensor:
        """Restore the semantic embedding from discrete semantic codes."""
        z_q_sem = self.quantizer.out_proj(self.quantizer.decode_code(codes))
        z_q_sem = F.interpolate(z_q_sem, size=target_len, mode="nearest")
        return self.restore_proj(z_q_sem)
