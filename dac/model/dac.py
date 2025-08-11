import math
from typing import List

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from .base import CodecMixin
from dac.nn.layers import RMSNorm
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False, use_rmsnorm: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            RMSNorm(dim) if use_rmsnorm else nn.Identity(),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=4 if causal else 7, dilation=dilation, causal=causal),
            RMSNorm(dim) if use_rmsnorm else nn.Identity(),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1, causal=causal),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 16, stride: int = 1, causal: bool = False, use_rmsnorm: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, causal=causal, use_rmsnorm=use_rmsnorm),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                causal=causal,
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        multipliers: list = [2, 4, 8, 8],
        d_latent: int = 64,
        causal: bool = False,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        kernel_size = 4 if causal else 7
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=kernel_size, causal=causal)]

        # Create EncoderBlocks that increase channels by multipliers as they downsample by strides
        current_dim = d_model
        for stride, multiplier in zip(strides, multipliers):
            output_dim = current_dim * multiplier
            self.block += [EncoderBlock(current_dim, output_dim, stride=stride, causal=causal, use_rmsnorm=use_rmsnorm)]
            current_dim = output_dim

        # Create last convolution
        self.block += [
            RMSNorm(current_dim) if use_rmsnorm else nn.Identity(),
            Snake1d(current_dim),
            WNConv1d(current_dim, d_latent, kernel_size=kernel_size, causal=causal),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = current_dim

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, causal: bool = False, use_rmsnorm: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                causal=causal,
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal, use_rmsnorm=use_rmsnorm),
            ResidualUnit(output_dim, dilation=1, causal=causal, use_rmsnorm=use_rmsnorm),
            ResidualUnit(output_dim, dilation=1, causal=causal, use_rmsnorm=use_rmsnorm),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        strides,
        multipliers,
        d_out: int = 1,
        causal: bool = False,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        kernel_size = 4 if causal else 7

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=kernel_size, causal=causal)]

        # Add upsampling + MRF blocks
        input_dim = channels
        for stride, multiplier in zip(strides, multipliers):
            output_dim = input_dim // multiplier
            layers += [DecoderBlock(input_dim, output_dim, stride, causal=causal, use_rmsnorm=use_rmsnorm)]
            input_dim = output_dim

        # Add final conv layer
        layers += [
            RMSNorm(output_dim) if use_rmsnorm else nn.Identity(),
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=kernel_size, causal=causal),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class WavLMDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        strides: List[int] = [],
        multipliers: List[int] = [],
        d_out: int = 1024,
        causal: bool = False,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        kernel_size = 4 if causal else 7

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=kernel_size, causal=causal)]

        # Add upsampling + decoder blocks
        input_dim = channels
        for stride, multiplier in zip(strides, multipliers):
            output_dim = input_dim // multiplier
            layers += [DecoderBlock(input_dim, output_dim, stride, causal=causal, use_rmsnorm=use_rmsnorm)]
            input_dim = output_dim

        # Add final conv layers
        layers += [
            RMSNorm(input_dim) if use_rmsnorm else nn.Identity(),
            Snake1d(input_dim),
            WNConv1d(input_dim, d_out, kernel_size=kernel_size, causal=causal),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_strides: List[int] = [2, 4, 8, 8],
        encoder_multipliers: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_strides: List[int] = [8, 8, 4, 2],
        decoder_multipliers: List[int] = [8, 8, 4, 2],
        wavlm_decoder_strides: List[int] = [],
        wavlm_decoder_multipliers: List[int] = [],
        latent_noise_max: float = 0.05,  # Maximum standard deviation for noise injection
        sample_rate: int = 44100,
        causal: bool = False,
        use_rmsnorm: bool = True,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_strides = encoder_strides
        self.encoder_multipliers = encoder_multipliers
        self.decoder_dim = decoder_dim
        self.decoder_strides = decoder_strides
        self.decoder_multipliers = decoder_multipliers
        self.sample_rate = sample_rate
        self.latent_noise_max = latent_noise_max

        if latent_dim is None:
            latent_dim = encoder_dim * np.prod(encoder_multipliers)

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_strides)
        self.encoder = Encoder(encoder_dim, encoder_strides, encoder_multipliers, latent_dim, causal=causal, use_rmsnorm=use_rmsnorm)

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_strides,
            decoder_multipliers,
            causal=causal,
            use_rmsnorm=use_rmsnorm,
        )
        self.wavlm_decoder = WavLMDecoder(
            latent_dim,
            decoder_dim,
            wavlm_decoder_strides,
            wavlm_decoder_multipliers,
            causal=causal,
            use_rmsnorm=use_rmsnorm,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
    ):
        """Encode given audio data

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode

        Returns
        -------
        Tensor[B x D x T]
            Encoded latent representation
        """
        return self.encoder(audio_data)

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        training: bool = False,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of input audio, by default None
        training : bool, optional
            Whether in training mode, by default False
            If True, injects random Gaussian noise to the latents with std
            uniformly sampled from [0, latent_noise_max]

        Returns
        -------
        dict
            Dictionary containing:
            "z" : Tensor[B x D x T]
                Encoded latent representation (with noise if training=True)
            "z_clean" : Tensor[B x D x T]
                Clean encoded latent representation (without noise)
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Reconstructed audio
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z_clean = self.encode(audio_data)
        
        if training:
            # Add Gaussian noise with random std between 0 and latent_noise_max
            z = z_clean + torch.randn_like(z_clean) * torch.rand(z_clean.shape[0], 1, 1, device=z_clean.device) * self.latent_noise_max
        else:
            z = z_clean
            
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "z_clean": z_clean,
            "wavlm": self.wavlm_decoder(z),
        }


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = DAC().to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    model.decompress(model.compress(x, verbose=True), verbose=True)
