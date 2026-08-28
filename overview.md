# Overview: AI/ML-Driven Noise Suppression System (PS ID: 26052)

## The Core Problem
In defense and mission-critical communication systems, reliable speech transmission is destroyed by two fundamentally different types of noise:
1. **Stationary Noise:** Constant hums (e.g., helicopter rotors, armored vehicle engines).
2. **Non-Stationary / Impulsive Noise:** Sudden, unpredictable explosions (e.g., gunshots, artillery fire).

Traditional signal processing (like classical LMS dual-mic filters) can handle stationary hums but fails catastrophically on impulsive gunshots because the noise isn't predictable. 

## Our Master Architecture (The Hybrid AI Solution)
To perfectly adhere to DRDO's problem statement, we designed a two-stage hybrid pipeline tailored for embedded hardware (FPGA / AI-enabled SoCs):

### Stage 1: The AI Engine (Completed in 36 Hours)
We built a **Single-Channel Deep Complex Convolutional Recurrent Network (DCCRN)**. 
- **Why Single-Channel AI?** Impulsive noises (gunshots) are too fast for dual-mic phase-delay algorithms. The AI must learn the actual *acoustic footprint* of the explosion to mask it. We process audio in the complex STFT domain to preserve phase for perfect speech reconstruction.
- **Hardware-First Engineering:** We didn't just build a Python model. We hardened it for FPGA DSP slices (AMD PYNQ / Vitis AI) by replacing expensive `LeakyReLU` with standard `ReLU`, enforcing strictly static memory allocation in our ONNX export, and injecting PyTorch Quantization-Aware Training (QAT) stubs for INT8 precision.
- **Data Engineering:** Instead of static files, our PyTorch pipeline dynamically mixes LibriSpeech and the MAD (Military Audio Dataset) on-the-fly at random SNRs (-5dB to 15dB), creating an infinitely robust training set.

### Stage 2: The Dual-Mic Edge Integration (Phase 2 Roadmap)
As requested by the problem statement, the AI engine acts as the brain within a larger dual-mic physical system.
- The **Reference Mic** feeds a lightweight classical LMS filter (which is computationally cheap on an SoC) to cancel the stationary helicopter hums.
- This frees up the FPGA's DSP slices to run the DCCRN purely on the remaining non-stationary gunshots/artillery.

## Expected DRDO Metrics
* **SNR:** > 15 dB
* **STOI:** > 0.85
* **PESQ:** > 2.5
* **Latency:** < 10-15ms (Real-time edge execution)
