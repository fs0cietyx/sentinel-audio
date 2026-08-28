import numpy as np
import time
import argparse
import onnxruntime as ort
import torch
from src.inference.hybrid_filter import LMSFilter 

def wav_to_spec(wav, n_fft=512, hop_length=256):
    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    stft_real = torch.view_as_real(stft)
    spec = stft_real.permute(0, 3, 1, 2)
    return spec

def spec_to_wav(spec, n_fft=512, hop_length=256):
    spec = spec.permute(0, 2, 3, 1)
    stft = torch.view_as_complex(spec.contiguous())
    window = torch.hann_window(n_fft, device=stft.device)
    wav = torch.istft(stft, n_fft=n_fft, hop_length=hop_length, window=window)
    return wav

class AudioInterfaceMock:
    """Mocks the hardware audio interface for dual-microphone input and headphone output."""
    def __init__(self, sample_rate=16000, chunk_size=512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def read_chunk(self):
        t = np.linspace(0, self.chunk_size / self.sample_rate, self.chunk_size, endpoint=False)
        target = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        noise = 0.3 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(self.chunk_size)
        
        primary_mic = target + noise
        reference_mic = noise + 0.05 * np.random.randn(self.chunk_size)
        
        return primary_mic, reference_mic

    def write_chunk(self, audio_data):
        pass

class RealtimePipeline:
    def __init__(self, model_path, use_lms=True):
        self.audio_interface = AudioInterfaceMock(sample_rate=16000, chunk_size=512)
        
        print(f"Loading ONNX model from {model_path}...")
        try:
            self.ort_session = ort.InferenceSession(model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            print(f"Warning: Could not load ONNX model ({e}). Using mock inference.")
            self.ort_session = None

        self.use_lms = use_lms
        if self.use_lms:
            self.lms_filter = LMSFilter(filter_length=64, step_size=0.005)
            print("LMS Hybrid Filter initialized.")

    def run(self, duration_sec=10):
        print(f"Starting real-time inference pipeline for {duration_sec} seconds...")
        
        num_chunks = int(duration_sec * self.audio_interface.sample_rate / self.audio_interface.chunk_size)
        
        start_time = time.time()
        for i in range(num_chunks):
            primary_mic, reference_mic = self.audio_interface.read_chunk()
            
            if self.ort_session:
                # 1. Convert waveform to spectrogram tensor
                wav_tensor = torch.tensor(primary_mic, dtype=torch.float32).unsqueeze(0)
                spec = wav_to_spec(wav_tensor) # (1, 2, 257, 3)
                
                # 2. Run ONNX Inference
                input_tensor = spec.numpy().astype(np.float32)
                nn_output_spec = self.ort_session.run(None, {self.input_name: input_tensor})[0]
                
                # 3. Convert spectrogram back to waveform
                out_spec_tensor = torch.tensor(nn_output_spec)
                nn_output_wav = spec_to_wav(out_spec_tensor).squeeze().numpy()
                
                # Ensure lengths match
                min_len = min(len(primary_mic), len(nn_output_wav))
                nn_output = nn_output_wav[:min_len]
            else:
                nn_output = primary_mic - 0.5 * reference_mic 
            
            if self.use_lms:
                # LMS uses the reference mic to cancel stationary noise from the AI output
                final_output, _ = self.lms_filter.process(reference_mic[:len(nn_output)], nn_output)
            else:
                final_output = nn_output

            self.audio_interface.write_chunk(final_output)
            time.sleep(self.audio_interface.chunk_size / self.audio_interface.sample_rate * 0.5) 
            
            if (i+1) % 50 == 0:
                print(f"Processed {(i+1) * self.audio_interface.chunk_size} samples...")
                
        total_time = time.time() - start_time
        print(f"Pipeline finished. Total processing time: {total_time:.2f}s for {duration_sec}s of audio.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real-time ANC pipeline.")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to ONNX model.")
    parser.add_argument("--no-lms", action="store_true", help="Disable hybrid LMS filter.")
    parser.add_argument("--duration", type=int, default=5, help="Duration to run in seconds.")
    args = parser.parse_args()
    
    pipeline = RealtimePipeline(args.model, use_lms=not args.no_lms)
    pipeline.run(duration_sec=args.duration)
