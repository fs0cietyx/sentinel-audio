# DRDO SIH26052: Adaptive Noise Cancellation (ANC)

## What It Is
An AI/ML-enabled adaptive noise cancellation (ANC) system designed to effectively suppress stationary, non-stationary, and impulsive defense noises. 
To perfectly satisfy the DRDO requirements, we designed a **Two-Stage Hybrid Architecture**:
1. **Stage 1 (Core AI Engine):** A state-of-the-art Single-Channel Deep Complex Convolutional Recurrent Network (DCCRN) that uses deep learning to predict and obliterate unpredictable impulsive explosions (gunshots/artillery).
2. **Stage 2 (Edge Physical Integration):** The AI operates alongside a classical lightweight LMS filter. The LMS filter uses the reference mic to cancel stationary hums (helicopter rotors), freeing the FPGA's DSP slices to run the AI engine at maximum efficiency.

It is explicitly optimized for real-time inference on **FPGA hardware (AMD PYNQ / Vitis AI)**.

## Why It Matters
Defense communication is severely degraded by dynamic acoustic disturbances. Traditional signal processing (LMS) assumes stationary noise and fails catastrophically on impulsive sounds. Our deep learning approach perfectly complements the LMS filter to create an impenetrable acoustic shield.

## Who It Is For
Defense personnel, tactical operators, and DRDO judges evaluating hardware-optimized machine learning solutions.

## Key Targets
- **SNR Improvement:** > 15 dB
- **STOI (Speech Intelligibility):** > 0.85
- **PESQ (Speech Quality):** > 2.5
- **Latency:** Real-time processing (< 10ms per frame via FPGA)

## Useful Links
- [DRDO Problem Statement 26052](https://www.sih.gov.in/)
- [MAD: Military Audio Dataset](https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset)
- [LibriSpeech Clean Dataset](http://www.openslr.org/12)
