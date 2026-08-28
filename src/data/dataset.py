import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
import math

class ANCDataset(Dataset):
    """
    PyTorch Dataset for Active Noise Cancellation (ANC).
    Dynamically mixes clean speech with noise at random SNRs on-the-fly.
    """
    def __init__(self, clean_audio_paths, noise_audio_paths, target_sr=16000, 
                 snr_range=(-10.0, 20.0), transform=None, duration=3):
        """
        Args:
            clean_audio_paths (list of str): Paths to clean speech audio files.
            noise_audio_paths (list of str): Paths to noise audio files.
            target_sr (int): Target sampling rate.
            snr_range (tuple): Range of Signal-to-Noise Ratios in dB (min, max).
            transform (callable, optional): Transform/augmentation to apply to the mixed audio.
            duration (int): Fixed duration in seconds for batching.
        """
        self.clean_audio_paths = clean_audio_paths
        self.noise_audio_paths = noise_audio_paths
        self.target_sr = target_sr
        self.snr_range = snr_range
        self.transform = transform
        self.fixed_len = target_sr * duration

    def __len__(self):
        return len(self.clean_audio_paths)

    def _calculate_power(self, audio):
        """Calculate the average power of the audio signal."""
        return torch.mean(audio ** 2)

    def mix_audio(self, clean, noise, snr_db):
        """
        Mix clean speech and noise at a specific SNR (Signal-to-Noise Ratio).
        """
        clean_power = self._calculate_power(clean)
        noise_power = self._calculate_power(noise)
        
        # Handle silence to prevent division by zero
        if noise_power == 0 or clean_power == 0:
            return clean + noise, clean
        
        # Calculate target noise power based on desired SNR
        target_noise_power = clean_power / (10 ** (snr_db / 10.0))
        
        # Scale noise to reach the target power
        scale_factor = torch.sqrt(target_noise_power / noise_power)
        scaled_noise = noise * scale_factor
        
        # Mix clean speech and scaled noise
        mixed = clean + scaled_noise
            
        return mixed, clean

    def __getitem__(self, idx):
        clean_path = self.clean_audio_paths[idx]
        # Randomly select a noise file for each clean speech
        noise_path = random.choice(self.noise_audio_paths)

        import soundfile as sf
        
        # Load audio files bypassing torchaudio entirely to avoid ffmpeg errors
        clean_data, sr_clean = sf.read(clean_path)
        if clean_data.ndim > 1:
            clean_data = clean_data.T
        else:
            clean_data = clean_data.reshape(1, -1)
            
        noise_data, sr_noise = sf.read(noise_path)
        if noise_data.ndim > 1:
            noise_data = noise_data.T
        else:
            noise_data = noise_data.reshape(1, -1)
            
        clean_audio = torch.tensor(clean_data, dtype=torch.float32)
        noise_audio = torch.tensor(noise_data, dtype=torch.float32)

        # Resample if sampling rates do not match the target
        if sr_clean != self.target_sr:
            clean_audio = torchaudio.functional.resample(clean_audio, sr_clean, self.target_sr)
        if sr_noise != self.target_sr:
            noise_audio = torchaudio.functional.resample(noise_audio, sr_noise, self.target_sr)

        # Force fixed length for clean audio so batching works
        clean_len = clean_audio.shape[1]
        if clean_len > self.fixed_len:
            start = random.randint(0, clean_len - self.fixed_len)
            clean_audio = clean_audio[:, start:start + self.fixed_len]
        elif clean_len < self.fixed_len:
            pad = self.fixed_len - clean_len
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad))

        # Force fixed length for noise audio
        noise_len = noise_audio.shape[1]
        if noise_len > self.fixed_len:
            start = random.randint(0, noise_len - self.fixed_len)
            noise_audio = noise_audio[:, start:start + self.fixed_len]
        elif noise_len < self.fixed_len:
            repeats = math.ceil(self.fixed_len / noise_len)
            noise_audio = noise_audio.repeat(1, repeats)[:, :self.fixed_len]

        # Select random SNR from the specified range
        snr = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Generate mixed audio
        mixed_audio, clean_target = self.mix_audio(clean_audio, noise_audio, snr)

        # Apply data augmentations (e.g., clipping, reverberation)
        if self.transform:
            mixed_audio = self.transform(mixed_audio)

        # Returns (Augmented Noisy Audio, Clean Target Speech)
        return mixed_audio, clean_target

def create_dataloader(clean_paths, noise_paths, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, **kwargs):
    """
    Creates a PyTorch DataLoader for the ANCDataset.
    """
    dataset = ANCDataset(clean_paths, noise_paths, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
