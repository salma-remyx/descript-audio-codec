"""
Audio transforms for DAC, reusable by other repositories.
"""

import torch
import typing
from audiotools import AudioSignal
from audiotools.data.transforms import BaseTransform


class PowerNorm(BaseTransform):
    """
    Normalize audio signal based on power level in dB.
    
    This transform calculates the power of the signal in time domain,
    converts it to dB, and normalizes the signal to a desired dB level
    by multiplying by a scalar.
    
    Parameters
    ----------
    db : float or tuple, optional
        Target power level in dB, by default ("const", -16.0)
        Can be a float or a tuple/list like ("const", -16.0) for config compatibility
    """
    
    # Hardcoded small constant to avoid log of zero
    EPS = 1e-10
    
    def __init__(
        self,
        db: typing.Union[float, typing.Tuple[str, float]] = ("const", -16.0),
        name: typing.Optional[str] = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)
        
        # Handle both direct float and tuple format (for compatibility with config)
        if isinstance(db, (list, tuple)):
            # Format: ["const", -16.0] or ("const", -16.0)
            self.target_db = db[1] if len(db) > 1 else db[0]
        else:
            self.target_db = float(db)

    def _transform(self, signal: AudioSignal, **kwargs):
        """
        Apply power normalization to the audio signal.
        
        Parameters
        ----------
        signal : AudioSignal
            Input audio signal to normalize
        
        Returns
        -------
        AudioSignal
            Normalized audio signal
        """
        signal.audio_data *= 10.0 ** ((torch.full(
            (signal.audio_data.shape[0], 1, 1), 
            float(self.target_db), 
            dtype=signal.audio_data.dtype, 
            device=signal.audio_data.device
        ) - 10.0 * torch.log10(torch.mean(signal.audio_data.pow(2), dim=-1, keepdim=True) + self.EPS)) / 20.0)
        
        return signal


# For backward compatibility and ease of use
def power_norm(signal: AudioSignal, target_db: float = -16.0):
    """
    Functional interface for PowerNorm transform.
    
    Parameters
    ----------
    signal : AudioSignal
        Input audio signal to normalize
    target_db : float
        Target power level in dB
    
    Returns
    -------
    AudioSignal
        Normalized audio signal
    """
    transform = PowerNorm(target_db)
    return transform._transform(signal)
