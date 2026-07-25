import torch
import torch.nn as nn


class DistributionalMatchLoss(nn.Module):
    """Distributional matching objective for vector quantization.

    Aligns the distribution of encoder feature vectors with the distribution
    of codebook vectors. Unlike the per-vector commitment / codebook losses
    (whose codebook gradient is routed through the straight-through estimator),
    this objective compares the two distributions *directly*: gradients flow to
    **both** the encoder features and the codebook. That distributional signal
    is what mitigates the STE gradient mismatch and the codebook collapse that
    arise when the feature and code distributions drift apart.

    Adapted from "Distributional Matching for Vector Quantization: A Unified
    Theoretical and Empirical Framework" (arXiv:2607.15933), which instantiates
    the framework with a Wasserstein-based objective that has a closed form
    under a diagonal-Gaussian approximation, and shows a non-parametric MMD
    alternative reaches comparable performance.

    The paper's evaluation is on visual-tokenization benchmarks (a different
    modality with its own training harness); here the loss term itself is
    implemented faithfully and plugged into the existing audio RVQ, while the
    paper's separate benchmark suite is intentionally left out (downstream).

    Parameters
    ----------
    kind : str
        ``"wasserstein"`` (default, the paper's headline objective),
        ``"mmd"`` for the non-parametric alternative, or ``"none"`` to disable
        (returns 0, e.g. to opt out at inference time).
    eps : float
        Numerical stabilizer for per-dimension standard deviations and the MMD
        kernel bandwidth.
    mmd_max_features : int
        Upper bound on the number of feature vectors sampled for the MMD
        estimator (bounds the O(M * N) kernel cost). Ignored for the
        Wasserstein kind, which is O(D).
    """

    def __init__(
        self,
        kind: str = "wasserstein",
        eps: float = 1e-6,
        mmd_max_features: int = 1024,
    ):
        super().__init__()
        kind = "none" if kind is None else kind
        if kind not in {"wasserstein", "mmd", "none"}:
            raise ValueError(f"Unknown dist_match kind: {kind!r}")
        self.kind = kind
        self.eps = eps
        self.mmd_max_features = mmd_max_features

    def forward(self, features: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Compute the scalar distributional matching loss.

        Parameters
        ----------
        features : Tensor
            Encoder feature vectors to be quantized (e.g. the projected
            latents ``z_e`` of a single VQ stage). Any shape with a trailing
            feature dimension is flattened to ``(*, D)``.
        codebook : Tensor
            The codebook vectors of the same stage, shape ``(N, D)`` (or any
            shape with a trailing ``D``).

        Returns
        -------
        Tensor
            Scalar loss. Both inputs keep gradients, so backprop updates the
            encoder (through ``features``) and the codebook (through
            ``codebook``).
        """
        if self.kind == "none":
            return features.new_zeros(())

        feats = self._as_vectors(features)
        codes = self._as_vectors(codebook)

        if self.kind == "wasserstein":
            return self._wasserstein(feats, codes)
        return self._mmd(feats, codes)

    @staticmethod
    def _as_vectors(x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        return x.float()

    def _wasserstein(self, feats: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        # Squared 2-Wasserstein distance under a diagonal-Gaussian approximation:
        #   W2^2(P, Q) = ||mu_P - mu_Q||^2 + Tr(Sigma_P + Sigma_Q
        #                                 - 2 (Sigma_Q^{1/2} Sigma_P Sigma_Q^{1/2})^{1/2})
        # For diagonal covariances the Bures trace term collapses to
        # ||sigma_P - sigma_Q||^2 (per-dimension standard deviations), giving an
        # O(D), parameter-free, fully differentiable objective. Averaged over
        # dimensions so its scale is comparable to the existing per-element VQ
        # losses (the paper's W2 distance is recovered by scaling the weight by
        # the feature dimension).
        f_mean = feats.mean(dim=0)
        c_mean = codes.mean(dim=0)
        # unbiased=False so a single code vector yields std 0 (then clamped),
        # and so the feature/code estimators match.
        f_std = feats.std(dim=0, unbiased=False).clamp_min(self.eps)
        c_std = codes.std(dim=0, unbiased=False).clamp_min(self.eps)
        mean_term = (f_mean - c_mean).pow(2).mean()
        std_term = (f_std - c_std).pow(2).mean()
        return mean_term + std_term

    def _mmd(self, feats: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        # Unbiased MMD^2 estimator with a Gaussian RBF kernel. Bandwidth uses a
        # data-driven median heuristic (computed under no_grad, so it is not a
        # learned parameter) over the feature/code cross distances.
        if feats.shape[0] > self.mmd_max_features:
            idx = torch.randperm(feats.shape[0], device=feats.device)[
                : self.mmd_max_features
            ]
            feats = feats[idx]
        cross = self._pairwise_sqdist(feats, codes)
        with torch.no_grad():
            bandwidth = cross.median().clamp_min(self.eps)
        k_ff = self._rbf(self._pairwise_sqdist(feats, feats), bandwidth)
        k_cc = self._rbf(self._pairwise_sqdist(codes, codes), bandwidth)
        k_fc = self._rbf(cross, bandwidth)
        return (k_ff.mean() + k_cc.mean() - 2.0 * k_fc.mean()).clamp_min(0.0)

    @staticmethod
    def _pairwise_sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (M, D), b: (N, D) -> (M, N) squared Euclidean distances.
        return a.pow(2).sum(-1, keepdim=True) + b.pow(2).sum(-1) - 2.0 * a @ b.t()

    @staticmethod
    def _rbf(sqdist: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
        return torch.exp(-sqdist.clamp_min(0.0) / bandwidth)
