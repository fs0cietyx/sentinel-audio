# DRDO SIH26052: Hybrid AI/ML Noise Suppression System

## Overview
This repository contains the codebase for our Hybrid AI/ML Noise Suppression System, built to solve DRDO Problem Statement 26052. The system aims to suppress both stationary noise (e.g., helicopter hum) and impulsive noise (e.g., gunshots/artillery) with a real-time latency of < 15ms.

Our solution employs a **two-stage hybrid architecture**:
1. **Single-Channel Deep Complex Convolutional Recurrent Network (DCCRN)**: An AI engine operating in the complex STFT domain to eliminate highly unpredictable, non-stationary impulsive noise.
2. **Dual-Mic LMS Wrapper**: A lightweight classical adaptive filter acting on a reference microphone to eliminate stationary background noise.


## Tech Stack
* **Language/Framework:** Python, PyTorch
* **Model:** DCCRN (Deep Complex CRN)
* **Audio I/O:** `soundfile`
* **Evaluation:** `pystoi`, `pesq`
* **Deployment:** ONNX, Vitis AI, AMD PYNQ
* **Training Compute:** Google Colab (T4 GPU)
* **Datasets:** LibriSpeech, MAD (Military Audio Dataset)

## Key Features
- **Hardware-First Engineering**: Built with standard ReLUs and exported with static memory axes to perfectly map onto FPGA DSP slices (AMD PYNQ / Vitis AI).
- **Quantization-Aware Training (QAT)**: PyTorch INT8 simulation natively built into the training loop, ensuring STOI and PESQ metrics do not crash when compiled for edge AI boards.
- **Dynamic Training Pipeline**: On-the-fly random mixing of LibriSpeech and the MAD (Military Audio Dataset) at varying SNRs (-5dB to 15dB) during training.
- **Ultra-Low Latency Edge Inference**: The pipeline operates on 512-sample chunks (10-15ms) to achieve real-time performance.

## Project Structure
- `src/models/anc_network.py`: The DCCRN PyTorch implementation with QAT stubs.
- `src/training/train.py`: Training pipeline with custom MultiLoss (SI-SNR, L1 Spectrogram).
- `src/inference/export_onnx.py`: Static-shape ONNX exporter tuned for 3-frame STFT chunks.
- `src/inference/realtime_pipeline.py`: The mock Edge wrapper that orchestrates the ONNX AI engine alongside the LMS filter.
- `src/inference/compile_for_pynq.sh`: Vitis AI compilation script (ONNX -> XModel).
- `src/inference/pynq_deployment.py`: Final deployment script using VART / DPUOverlay for the PYNQ FPGA board.
- `src/inference/hybrid_filter.py`: The lightweight LMS adaptive filter.

## Getting Started

### 1. Training (Google Colab)
We recommend training the AI engine on a GPU instance (e.g., Colab T4). 
1. Zip this workspace and upload it to Colab.
2. Open `SIH26052_Colab_Training.ipynb` and follow the execution cells.

### 2. Desktop Simulation (Testing ONNX)
To simulate the edge environment processing the dual-mic streams on your PC:
```bash
python3 src/inference/realtime_pipeline.py --model checkpoints/fpga_model.onnx
```

### 3. FPGA PYNQ Deployment (Final Stage)
To run this on the physical AMD PYNQ Board:
1. Compile the ONNX model to an XModel inside the Vitis AI Docker container:
   ```bash
   ./src/inference/compile_for_pynq.sh
   ```
2. Move `dccrn_anc.xmodel` to your PYNQ board.
3. Run the hardware-accelerated DPU inference wrapper:
   ```bash
   python3 src/inference/pynq_deployment.py --model checkpoints/dccrn_anc.xmodel
   ```

## Authors
Created for SIH26052 (DRDO).
