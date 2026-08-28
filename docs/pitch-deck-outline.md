# Pitch Deck Outline (DRDO SIH26052)

**Slide 1: Title Slide**
- **Title:** Hybrid AI/ML-Driven Noise Suppression System
- **Subtitle:** Solving DRDO PS-26052 for Mission-Critical Defense Communications
- **Team Name & Member Roles**

**Slide 2: The Problem**
- **Core Challenge:** Defense comms are destroyed by two vastly different noises.
- **Stationary Noise:** Helicopter hums, engine drones (predictable).
- **Impulsive Noise:** Gunshots, artillery fire, explosions (unpredictable, sudden).
- **Why it's hard:** Traditional dual-mic filters handle stationary noise but fail catastrophically on impulsive noise. AI handles impulsive noise but is computationally heavy.

**Slide 3: Our Master Architecture (The Hybrid Solution)**
- **Stage 1 (The Brain):** Single-Channel AI (DCCRN) operating on the complex STFT domain to identify and erase the acoustic footprint of explosions.
- **Stage 2 (The Muscle):** A lightweight classical LMS filter using the reference mic to cancel the helicopter hums.
- **Why this wins:** We don't waste expensive AI compute on simple hums. We save the FPGA's DSP slices specifically for the hardest problem (gunshots).

**Slide 4: Hardware-First Engineering (Our Secret Weapon)**
- We didn't just build a "Python script." We built this for the Edge (FPGA/Jetson).
- **Optimization 1:** Removed hardware-expensive operations (`LeakyReLU` -> `ReLU`).
- **Optimization 2:** PyTorch Quantization-Aware Training (QAT). The AI was trained to simulate INT8 precision so it doesn't degrade when flashed to the board.
- **Optimization 3:** Strictly static memory ONNX export (required by Vitis AI / TensorRT).

**Slide 5: Training & Data Engineering**
- No static datasets. We built a dynamic data pipeline.
- We aggressively mixed Clean Speech (LibriSpeech) with Military Impulsive Noise (MAD) on-the-fly at random Signal-to-Noise Ratios (-5dB to 15dB).
- The model learned to survive extreme acoustic environments.

**Slide 6: Live Metrics & Results**
- **Latency:** Our pipeline operates on 512-sample chunks achieving < 15ms latency.
- **Speech Quality:** (Show a screenshot or graph of your Colab STOI > 0.85 and PESQ > 2.5 metrics).
- **SNR Improvement:** Consistently > 15 dB noise reduction on military test sets.

**Slide 7: Edge Deployment Roadmap (Phase 2)**
- How this actually goes onto the battlefield.
- The `fpga_model.onnx` is fed into the Vitis AI compiler.
- Real-time C++ inference wrapper deployed on an AMD PYNQ or NVIDIA Jetson AGX.
- Dual-mic hardware wiring.

**Slide 8: Conclusion & Q&A**
- We delivered a robust, hardware-ready, hybrid AI solution that strictly adheres to the DRDO requirements.
- Open for questions.
