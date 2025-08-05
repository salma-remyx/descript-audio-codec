"""Speaker Identification Metric (SIM) - Simple version with delayed import."""

import torch
import torchaudio


class SIM:
    """
    Speaker Identification Metric (SIM) class.
    
    Delays SpeechBrain import until first use to avoid initialization conflicts.
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the SIM metric."""
        self.device = device
        self._model = None
    
    def _get_model(self):
        """Load model on first use."""
        if self._model is None:
            # Set thread safety for multiprocessing environments
            import os
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
            
            # Import here to avoid conflicts during initialization
            import speechbrain.inference
            
            # Temporarily limit threads during model loading
            import torch
            original_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            
            try:
                self._model = speechbrain.inference.EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": self.device},
                )
            finally:
                torch.set_num_threads(original_threads)
                
        return self._model
    
    def __call__(self, audio_original, audio_encoded, sample_rate=16000):
        """
        Calculate SIM score between original and encoded-decoded audio.
        
        Args:
            audio_original: Original audio signal (numpy array)
            audio_encoded: Encoded-decoded audio signal (numpy array)
            sample_rate: Sample rate of the audio (default: 16000)
        
        Returns:
            float: Cosine similarity score between -1 and 1 (higher is better)
        """
        try:
            # Force garbage collection before processing
            import gc
            gc.collect()
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_original = resampler(torch.FloatTensor(audio_original).unsqueeze(0)).squeeze(0).numpy()
                audio_encoded = resampler(torch.FloatTensor(audio_encoded).unsqueeze(0)).squeeze(0).numpy()
            
            # Get model (loads on first use)
            model = self._get_model()
            
            # Thread-safe computation with cleanup
            with torch.no_grad():
                # Temporarily limit threads during inference
                original_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                
                try:
                    # Ensure tensors are on CPU
                    tensor1 = torch.FloatTensor(audio_original).unsqueeze(0).cpu()
                    tensor2 = torch.FloatTensor(audio_encoded).unsqueeze(0).cpu()
                    
                    # Process one at a time and clean up
                    emb1 = model.encode_batch(tensor1).squeeze(1).cpu()
                    del tensor1
                    
                    emb2 = model.encode_batch(tensor2).squeeze(1).cpu()
                    del tensor2
                    
                    similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
                    
                    # Clean up embeddings
                    del emb1, emb2
                    
                    return similarity
                    
                finally:
                    torch.set_num_threads(original_threads)
                
        except Exception as e:
            print(f"[SIM] Error in calculation: {e}")
            return 0.0
        finally:
            # Always clean up
            gc.collect()