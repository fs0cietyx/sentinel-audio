import os
import glob
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pystoi import stoi
from pesq import pesq

from src.models.anc_network import DCCRNPlaceholder
from src.training.losses import MultiLoss
from src.data.dataset import create_dataloader
from src.data.augmentations import ComposeAugmentations, apply_clipping

def calculate_metrics(pred_wav, clean_wav, sr=16000):
    """Calculates real STOI and PESQ metrics."""
    # Convert to numpy arrays for the metric libraries
    pred_np = pred_wav.detach().cpu().numpy().flatten()
    clean_np = clean_wav.detach().cpu().numpy().flatten()
    
    try:
        stoi_score = stoi(clean_np, pred_np, sr, extended=False)
    except Exception:
        stoi_score = 0.0
        
    try:
        pesq_score = pesq(sr, clean_np, pred_np, 'wb')
    except Exception:
        pesq_score = 1.0 
        
    return stoi_score, pesq_score

def wav_to_spec(wav, n_fft=512, hop_length=256):
    # Ensure wav is 2D: (Batch, Time)
    if wav.dim() == 3:
        wav = wav.squeeze(1)
    # Compute STFT with Hann window to prevent spectral leakage
    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    # Convert complex tensor to (..., 2) where last dim is real/imag
    stft_real = torch.view_as_real(stft)
    # Rearrange to (Batch, 2, Freq, Time)
    spec = stft_real.permute(0, 3, 1, 2)
    return spec

def spec_to_wav(spec, n_fft=512, hop_length=256):
    # spec is (Batch, 2, Freq, Time)
    # permute back to (Batch, Freq, Time, 2)
    spec = spec.permute(0, 2, 3, 1)
    # view_as_complex converts back to complex tensor
    stft = torch.view_as_complex(spec.contiguous())
    # Inverse STFT with Hann window
    window = torch.hann_window(n_fft, device=stft.device)
    wav = torch.istft(stft, n_fft=n_fft, hop_length=hop_length, window=window)
    return wav.unsqueeze(1)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DCCRNPlaceholder().to(device)
    
    # ---------------------------------------------------------
    # FPGA PREPARATION: Quantization-Aware Training (QAT) Setup
    # ---------------------------------------------------------
    # We must prepare the model to simulate INT8 precision during training
    # so the STOI/PESQ metrics do not crash when compiled for the PYNQ board.
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(model, inplace=True)
    print("Model configured for Quantization-Aware Training (INT8 Simulation).")
    # ---------------------------------------------------------

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Grab both .wav and .flac (LibriSpeech uses .flac)
    clean_speech_files = glob.glob("dataset/clean_speech/**/*.wav", recursive=True) + glob.glob("dataset/clean_speech/**/*.flac", recursive=True)
    military_noise_files = glob.glob("dataset/noise/impulsive/**/*.wav", recursive=True)
    
    if not clean_speech_files or not military_noise_files:
        print(f"Found {len(clean_speech_files)} clean and {len(military_noise_files)} noise files.")
        print("Please ensure the datasets are extracted correctly.")
        return

    print(f"Initializing Dataset with {len(clean_speech_files)} clean speech files and {len(military_noise_files)} military noise files...")
    augmentations = ComposeAugmentations([
        apply_clipping
    ])

    dataloader = create_dataloader(
        clean_paths=clean_speech_files,
        noise_paths=military_noise_files,
        target_sr=16000,
        snr_range=(-5.0, 15.0),
        transform=augmentations,
        batch_size=32,
        num_workers=2,
        pin_memory=True
    )
    
    epochs = 30
    best_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    
    criterion = MultiLoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for noisy_wav, clean_wav in progress_bar:
            noisy_wav, clean_wav = noisy_wav.to(device), clean_wav.to(device)
            
            # Compute STFT on the fly
            noisy_spec = wav_to_spec(noisy_wav).to(device)
            clean_spec = wav_to_spec(clean_wav).to(device)
            
            optimizer.zero_grad()
            
            pred_spec = model(noisy_spec)
            
            # Inverse STFT to get predicted waveform
            pred_wav = spec_to_wav(pred_spec)
            
            # Ensure lengths match in time domain
            min_len = min(pred_wav.shape[-1], clean_wav.shape[-1])
            pred_wav = pred_wav[..., :min_len]
            clean_wav = clean_wav[..., :min_len]
            
            loss, loss_dict = criterion(pred_wav, clean_wav, pred_spec, clean_spec)
            
            loss.backward()
            
            # Add gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'si_snr': loss_dict['si_snr']})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        model.eval()
        with torch.no_grad():
            stoi_val, pesq_val = calculate_metrics(pred_wav[0], clean_wav[0])
            print(f"Validation Metrics - STOI: {stoi_val:.2f}, PESQ: {pesq_val:.2f}")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/best_anc_model.pth')
            print("Saved best model.")

if __name__ == "__main__":
    train()
