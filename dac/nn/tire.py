"""Time-Invariant Representation Extraction (TIRE).

Adapted from TiCodec, "Streaming Neural Speech Codecs through
Time-Invariant Representations" (arXiv:2607.05250v1).

TiCodec factors a global, time-invariant representation (broadly
speaker / environment information) out of the encoder latent, so the
residual vector quantizer only has to model the time-varying content at
the frame level. This module implements that factorization as an
*addition* to DAC's existing ``[B x D x T]`` quantizer contract:

    z_inv, z_residual = tire.factorize(z)   # z_inv: [B x D x 1], global
    ...                                      # quantize z_residual only
    z = tire.combine(z_inv, z_q_residual)   # re-add the global vector
                                              # before decoding

Adaptation note (Mode 2). The paper learns a dedicated TIRE extractor
with a global information-bottleneck objective and a cross-file segment
sampling strategy. Here the *learned extractor* is substituted by a
parameter-free temporal-pooling proxy by default: a global per-channel
mean over the time axis, which approximates the same "time-invariant
summary" signal without a separate estimator or its training procedure.
The factorization itself (subtract a global vector, quantize the
residual, add it back) is kept at full fidelity. An optional small
bottleneck projector is provided for the trainable variant; both share
the ``factorize`` / ``combine`` contract, so swapping in a learned
estimator later is a drop-in change.

The paper's global-information-bottleneck training loss, its cross-file
sampling scheme, the Dual-TIRE multi-level architecture, and the
streaming evaluation suite are intentionally out of scope here and
belong in a follow-up.
"""

from typing import Tuple
from typing import Union

import torch
from torch import nn


class TimeInvariantFactorizer(nn.Module):
    """Factor a global time-invariant vector out of a ``[B x D x T]`` latent.

    Parameters
    ----------
    dim : int
        Channel dimension ``D`` of the latent.
    learned : bool, optional
        If True, route the pooled global vector through a small bottleneck
        projector (trainable TIRE). If False (default), use the
        parameter-free pooling proxy, which adds no parameters and so does
        not affect pretrained weight loading.
    bottleneck_ratio : float, optional
        Width of the learned bottleneck as a fraction of ``dim``.
    """

    def __init__(
        self,
        dim: int,
        learned: bool = False,
        bottleneck_ratio: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.learned = learned
        if learned:
            bottleneck = max(1, int(dim * bottleneck_ratio))
            self.projector: Union[nn.Module, None] = nn.Sequential(
                nn.Linear(dim, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, dim),
            )
        else:
            self.projector = None

    def global_vector(self, z: torch.Tensor) -> torch.Tensor:
        """Summarize ``z`` into a single global, time-invariant vector.

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x 1]
        """
        pooled = z.mean(dim=-1)  # [B x D]
        if self.projector is not None:
            pooled = self.projector(pooled)
        return pooled.unsqueeze(-1)  # [B x D x 1]

    def factorize(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split ``z`` into a global invariant vector and a time-varying residual.

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        z_inv : Tensor[B x D x 1]
            Global, time-invariant component (broadcastable over T).
        z_residual : Tensor[B x D x T]
            Time-varying component fed to the quantizer.
        """
        z_inv = self.global_vector(z)
        return z_inv, z - z_inv

    def combine(
        self, z_inv: torch.Tensor, z_q_residual: torch.Tensor
    ) -> torch.Tensor:
        """Re-add the global invariant to the quantized residual.

        Parameters
        ----------
        z_inv : Tensor[B x D x 1]
        z_q_residual : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
        """
        return z_q_residual + z_inv

    def invariant_ratio(self, z: torch.Tensor) -> torch.Tensor:
        """Fraction of latent energy captured by the global invariant.

        A parameter-free probe (no retraining required): how much of the
        encoder latent sits in the time-invariant component versus the
        time-varying residual the RVQ must quantize. Returns a scalar in
        ``[0, 1]``; higher means more of the latent is time-invariant and
        therefore candidate for factoring out.

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[]
        """
        z_inv = self.global_vector(z)
        num = z_inv.pow(2).sum()
        den = z.pow(2).sum().clamp_min(torch.finfo(z.dtype).eps)
        return num / den
