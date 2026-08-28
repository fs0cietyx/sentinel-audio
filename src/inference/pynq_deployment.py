"""
PYNQ Edge Deployment Script for DRDO SIH26052
This script runs on the physical PYNQ board using the Vitis AI Runtime (VART).
It interfaces directly with the FPGA's DPU (Deep Learning Processor Unit) 
to run the XModel exported from our PyTorch QAT pipeline.
"""
import numpy as np
import time
import argparse
import sys
import os

try:
    import vart
    import xir
    import pynq
    from pynq.dpu import DpuOverlay
    ON_PYNQ = True
except ImportError:
    print("Warning: VART/PYNQ libraries not found. Simulating edge environment...")
    ON_PYNQ = False

from hybrid_filter import LMSFilter
import torch # using torch just for STFT in Python wrapper, FPGA does inference

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

class PYNQRealtimePipeline:
    def __init__(self, xmodel_path, use_lms=True):
        self.chunk_size = 512
        self.sample_rate = 16000
        self.use_lms = use_lms
        
        if ON_PYNQ:
            # Load the DPU overlay bitstream onto the FPGA
            self.overlay = DpuOverlay("dpu.bit")
            # Create a VART Runner for our specific XModel
            self.graph = xir.Graph.deserialize(xmodel_path)
            self.subgraphs = self.get_child_subgraph_dpu(self.graph)
            self.dpu_runner = vart.Runner.create_runner(self.subgraphs[0], "run")
            print("Successfully loaded XModel into FPGA DPU!")
        else:
            print(f"Mocking DPU runner for {xmodel_path}...")
            self.dpu_runner = None
            
        if self.use_lms:
            self.lms = LMSFilter(filter_length=64, step_size=0.005)

    def get_child_subgraph_dpu(self, graph):
        assert graph is not None, "Graph is none"
        root = graph.get_root_subgraph()
        assert root is not None, "Failed to get root subgraph"
        child_subgraphs = root.topological_sort()
        return [s for s in child_subgraphs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]

    def run_dpu_inference(self, input_tensor):
        if not ON_PYNQ:
            # Simulate DPU delay (usually ~2-5ms for this model on DPU)
            time.sleep(0.002)
            return input_tensor 
            
        # VART requires a list of numpy buffers
        input_data = [input_tensor]
        output_data = [np.empty((1, 2, 257, 3), dtype=np.float32)]
        
        # Execute asynchronously on the DPU
        job_id = self.dpu_runner.execute_async(input_data, output_data)
        self.dpu_runner.wait(job_id)
        
        return output_data[0]

    def process_live_stream(self, duration_sec=10):
        print("Starting live DPU-accelerated audio processing...")
        num_chunks = int(duration_sec * self.sample_rate / self.chunk_size)
        
        for i in range(num_chunks):
            # 1. Read hardware I/O (Simulated here)
            primary_mic = np.random.randn(self.chunk_size).astype(np.float32)
            reference_mic = np.random.randn(self.chunk_size).astype(np.float32)
            
            # 2. STFT
            wav_tensor = torch.tensor(primary_mic).unsqueeze(0)
            spec = wav_to_spec(wav_tensor).numpy()
            
            # 3. DPU AI Inference (Hardware Accelerated)
            dpu_out_spec = self.run_dpu_inference(spec)
            
            # 4. ISTFT
            out_tensor = torch.tensor(dpu_out_spec)
            ai_cleaned_wav = spec_to_wav(out_tensor).squeeze().numpy()
            
            # 5. Hybrid LMS (CPU/PS side)
            if self.use_lms:
                min_len = min(len(reference_mic), len(ai_cleaned_wav))
                final_wav, _ = self.lms.process(reference_mic[:min_len], ai_cleaned_wav[:min_len])
            else:
                final_wav = ai_cleaned_wav
                
            # 6. Write hardware I/O
            # ... write to audio codec ...

        print("Live stream processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/dccrn_anc.xmodel", help="Path to Vitis AI XModel")
    args = parser.parse_args()
    
    pipeline = PYNQRealtimePipeline(args.model)
    pipeline.process_live_stream(duration_sec=2)
