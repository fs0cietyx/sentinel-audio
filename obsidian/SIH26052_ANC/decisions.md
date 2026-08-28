# Architectural & Design Decisions

### 1. On-The-Fly Audio Mixing
- **Decision:** Dynamically mix LibriSpeech and MAD inside the PyTorch `__getitem__` function during training instead of pre-mixing static files.
- **Why:** Saves massive amounts of disk space and acts as a powerful data augmentation technique, preventing overfitting.

### 2. Time-Frequency Domain (STFT) Processing
- **Decision:** The network (DCCRN) operates on complex spectrograms rather than raw 1D audio waveforms.
- **Why:** ANC for non-stationary noise is mathematically easier to isolate in the frequency domain. Preserving the phase is critical for speech reconstruction.

### 3. FPGA Hardware Conformity (ReLU & Static ONNX)
- **Decision:** Use standard `ReLU` over `LeakyReLU` and enforce strictly static ONNX graphs (removing dynamic axes).
- **Why:** `LeakyReLU` wastes precious FPGA DSP slices. Static graphs are mandatory because FPGA HLS synthesis requires pre-allocated memory buffers.

### 4. Single-Channel Deep Learning vs. Dual-Mic LMS
- **Decision:** Abandon the legacy Dual-Mic LMS filter approach entirely in favor of a Single-Channel Deep Complex Convolutional Recurrent Network (DCCRN).
- **Why:** DRDO explicitly stated classical LMS fails on non-stationary impulsive noise. Deep Learning solves this. A Single-Channel architecture provides a massive tactical advantage because it can be deployed as a software update to existing single-mic legacy defense radios.

### 5. PYNQ / Vitis AI Quantization Prep
- **Decision:** Inject PyTorch `QuantStub` and `DeQuantStub` directly into the PyTorch module definition.
- **Why:** The AMD/Xilinx Vitis AI DPU only operates on INT8 logic. PyTorch must execute Quantization-Aware Training (QAT) to preserve STOI/PESQ metrics, and the compiler needs these stubs to identify hardware entry/exit points.
