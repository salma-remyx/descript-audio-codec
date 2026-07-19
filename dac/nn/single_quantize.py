from typing import Union

import torch
import torch.nn as nn

from dac.nn.quantize import VectorQuantize


class SingleVectorQuantize(nn.Module):
    """Single-codebook quantizer that is interface-compatible with
    :class:`dac.nn.quantize.ResidualVectorQuantize`.

    Adapted from the core contribution of *L3AC: Towards a Lightweight and
    Lossless Audio Codec* (Ma et al., 2025, https://arxiv.org/abs/2504.04949).
    L3AC observes that the multi-codebook Residual Vector Quantization (RVQ)
    stack used by codecs such as DAC and EnCodec can be collapsed to a SINGLE
    quantizer, recovering the lost representational capacity with a wider
    codebook dimension. This module delivers that insight as a drop-in: it
    accepts the same constructor arguments and exposes the same
    ``forward`` / ``from_codes`` / ``from_latents`` contract as
    ``ResidualVectorQuantize``, but emits ``codes`` with ``N = 1`` codebook
    instead of ``N = 9``.

    The audio<->codes contract is preserved exactly: ``from_codes`` decodes
    the single codebook axis, so a model whose ``.quantizer`` is swapped for
    this class keeps round-tripping (encode -> codes -> from_codes -> decode)
    without any change to the encoder, decoder, or ``.dac`` file format.

    Implementation notes (Mode 2 adapted port):

    - IN (paper core, full fidelity): single-quantizer collapse (N:9 -> 1) and
      the wider-codebook capacity compensation. The single codebook reuses the
      repo's existing :class:`VectorQuantize` rather than reintroducing it.
    - SUBSTITUTED (target-native): the paper's exact single-codebook sizing is
      replaced by a capacity-preservation heuristic -- the one codebook's
      dimension is set to the sum of the residual codebook dimensions it
      replaces (``codebook_dim * n_codebooks``), so the projected latent width
      is unchanged by the swap.
    - OUT (training-run scale, downstream PR): L3AC's lightweight backbone
      retraining and its finite-scalar "lossless" quantization variant. Those
      require re-training the codec and are not reachable from this call site.
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        # ``n_codebooks`` / ``quantizer_dropout`` are accepted only so this can
        # be constructed where ``ResidualVectorQuantize`` was. L3AC uses one
        # codebook, so both are collapsed away.
        del quantizer_dropout

        if isinstance(codebook_dim, list):
            # RVQ passes a per-codebook list; the single codebook takes the sum
            # so the projected latent width is preserved across the swap.
            single_dim = int(sum(codebook_dim))
        else:
            single_dim = int(codebook_dim) * int(n_codebooks)

        self.n_codebooks = 1
        self.codebook_size = codebook_size
        self.codebook_dim = single_dim
        self.quantizer = VectorQuantize(input_dim, codebook_size, single_dim)

    def forward(self, z, n_quantizers: int = None):
        """Quantize ``z`` with a single codebook.

        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            Accepted for signature compatibility with
            ``ResidualVectorQuantize``. A single-quantizer codec always uses
            one codebook, so this argument is ignored.

        Returns
        -------
        z_q : Tensor[B x D x T]
            Quantized continuous representation of input
        codes : Tensor[B x 1 x T]
            Codebook indices (single codebook -- L3AC's N:9 -> 1 collapse)
        latents : Tensor[B x C x T]
            Projected latents before quantization (``C`` is the widened
            single-codebook dimension)
        commitment_loss : Tensor[1]
        codebook_loss : Tensor[1]
        """
        del n_quantizers

        z_q, commitment_loss, codebook_loss, indices, z_e = self.quantizer(z)
        codes = indices.unsqueeze(1)  # [B, 1, T] -- matches the RVQ contract
        latents = z_e
        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Reconstruct the continuous representation from ``codes``.

        Shape-agnostic like ``ResidualVectorQuantize.from_codes``: any codebook
        axis beyond the first is ignored, since L3AC quantizes with a single
        codebook. ``codes`` of shape ``[B, 1, T]`` (this codec) and ``[B, k, T]``
        (a legacy RVQ file, taking its first codebook) both decode.

        Parameters
        ----------
        codes : Tensor[B x N x T]

        Returns
        -------
        z_q : Tensor[B x D x T]
        z_p : Tensor[B x C x T]
        codes : Tensor[B x N x T]
        """
        indices = codes[:, 0, :]
        z_p = self.quantizer.decode_code(indices)
        z_q = self.quantizer.out_proj(z_p)
        return z_q, z_p, codes

    def from_latents(self, latents: torch.Tensor):
        """Reconstruct the continuous representation from projected latents.

        Parameters
        ----------
        latents : Tensor[B x C x T]

        Returns
        -------
        z_q : Tensor[B x D x T]
        z_p : Tensor[B x C x T]
        codes : Tensor[B x 1 x T]
        """
        z_p, indices = self.quantizer.decode_latents(latents)
        z_q = self.quantizer.out_proj(z_p)
        codes = indices.unsqueeze(1)
        return z_q, z_p, codes


if __name__ == "__main__":
    # Mirror the RVQ self-check in quantize.py so the contract is visible.
    sq = SingleVectorQuantize(input_dim=512, n_codebooks=9, codebook_dim=8)
    x = torch.randn(16, 512, 80)
    z_q, codes, latents, _, _ = sq(x)
    print("codes:", tuple(codes.shape), "latents:", tuple(latents.shape))
