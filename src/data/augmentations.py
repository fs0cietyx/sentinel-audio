import torch
import torchaudio.functional as F

def apply_reverberation(audio: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """
    Applies reverberation to an audio signal using a Room Impulse Response (RIR)
    via FFT-based convolution to simulate physical spaces.
    
    Args:
        audio (torch.Tensor): Audio signal of shape (..., time).
        rir (torch.Tensor): Room Impulse Response signal of shape (..., time).
        
    Returns:
        torch.Tensor: Reverberated audio signal, same shape as input audio.
    """
    # Normalize RIR power
    rir = rir / torch.norm(rir, p=2)
    
    # Apply FFT convolution to simulate reverberation
    reverberated = F.fftconvolve(audio, rir)
    
    # Trim the output to the original audio length
    return reverberated[..., :audio.shape[-1]]

def apply_clipping(audio: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """
    Simulate microphone peaking/clipping caused by loud impulsive noises
    (e.g., gunshots, artillery, industrial noise).
    
    Args:
        audio (torch.Tensor): Audio signal.
        threshold (float): Maximum absolute amplitude. Values above this are clipped.
        
    Returns:
        torch.Tensor: Clipped audio signal.
    """
    return torch.clamp(audio, min=-threshold, max=threshold)

class ComposeAugmentations:
    """
    Utility class to chain multiple augmentations together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio
