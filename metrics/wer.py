"""Word Error Rate (WER) metric using Whisper for transcription."""

import whisper
import jiwer
import torch


class WER:
    """
    Word Error Rate (WER) metric class.
    
    Uses OpenAI's Whisper model to transcribe audio and jiwer to calculate WER.
    WER measures the difference between transcriptions of original and encoded audio.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the WER metric with Whisper turbo model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model = whisper.load_model("turbo", device=device)
    
    def __call__(self, path_original, path_encoded):
        """
        Calculate WER between transcriptions of original and encoded-decoded audio.
        
        Args:
            path_original: Path to original audio file
            path_encoded: Path to encoded-decoded audio file
        
        Returns:
            float: Word Error Rate (0.0 = perfect, 1.0 = completely wrong)
        """
        with torch.no_grad():
            return jiwer.wer(
                self.model.transcribe(
                    path_original,
                    temperature=0.0,
                    best_of=1,
                )["text"],
                self.model.transcribe(
                    path_encoded,
                    temperature=0.0,
                    best_of=1,
                )["text"])