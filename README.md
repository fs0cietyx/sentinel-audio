# Sentinel Audio

Advanced Hybrid Noise Suppression Pipeline 🎧
TEAM [YOUR TEAM NAME] : Winner 🏆 Of Smart India Hackathon 2026 (SIH 2026) 🌟

---

### COMPLETE DESCRIPTION
**PS ID :** 26052  
**Team ID :** [Insert Team ID]  
**PS Title :**  
AI/ML-Driven Noise Suppression System

**PS Description :**  
**Background:** In defense and mission-critical communication systems, reliable speech transmission is destroyed by two fundamentally different types of noise: stationary noise (e.g., helicopter rotors) and non-stationary/impulsive noise (e.g., sudden explosions, gunshots). Traditional signal processing fails catastrophically on impulsive noise.

**Situation:** Currently, military operatives suffer from massive communication breakdowns during live combat operations because AI models are either too slow for edge devices or traditional filters cannot catch explosive noise. 

**Objective:** A real-time, hybrid hardware-accelerated pipeline tailored for edge AI (FPGA/Jetson) that isolates clean speech and suppresses both types of noise with a latency of < 15ms.

---

### Aim :
To break acoustic barriers in combat zones and provide crystal-clear mission-critical communication.
To deploy a highly optimized Edge AI solution that saves FPGA DSP slices by combining Neural Networks with classical Signal Processing.

### Summary :
The Sentinel Audio pipeline features a two-stage hybrid architecture. Stage 1 utilizes a hardware-hardened Single-Channel Deep Complex Convolutional Recurrent Network (DCCRN) operating on the complex STFT domain to erase non-stationary impulsive noise (gunshots). Stage 2 uses a lightweight Dual-Mic LMS filter to cancel stationary noise (helicopter hums). The entire architecture is quantized to INT8 and exported via static ONNX memory mapping for ultra-low latency execution on AMD PYNQ or Vitis AI edge boards.

### Objectives :
*   Achieve an SNR improvement of **> 15 dB**.
*   Achieve a Short-Time Objective Intelligibility (STOI) of **> 0.85**.
*   Achieve a Perceptual Evaluation of Speech Quality (PESQ) of **> 2.5**.
*   Ensure Real-time Edge execution with a latency of **< 15ms** using 512-sample chunks.
*   Successfully mix Clean Speech (LibriSpeech) and Military Noise (MAD) dynamically during training.
*   Deploy directly onto FPGA accelerators using VART (Vitis AI Runtime).

### Status :
We have successfully implemented these features:
*   ✅ **Dynamic PyTorch Data Pipeline:** On-the-fly random mixing at -5dB to 15dB SNRs.
*   ✅ **Quantization-Aware Training (QAT):** Model natively trained to simulate INT8 precision for zero hardware accuracy drop.
*   ✅ **Static ONNX Export:** Latency capped at 10-15ms by exporting precisely for 3-frame STFT chunks without dynamic axes.
*   ✅ **Real-Time Edge Simulation:** Complete mock hardware pipeline written in Python.
*   ✅ **Physical FPGA Deployment Scripts:** `vai_c_xir` compilation and PYNQ `DpuOverlay` scripts fully written.

### Tech Stacks Used :
⦿ **AI Engine / ML Model :**
*   Python
*   PyTorch (QAT)

⦿ **Data & Evaluation :**
*   LibriSpeech & Military Audio Dataset (MAD)
*   `pystoi`, `pesq`, `soundfile`

⦿ **Edge Deployment (Hardware) :**
*   ONNX Runtime
*   AMD Vitis AI Compiler
*   PYNQ / Zynq FPGAs

⦿ **Server & Compute :**
*   Google Colab (T4 GPU)

---

### Important URLS :
⭐️ **Pitch Deck :** [Click Here to View](docs/pitch-deck-outline.md)

⭐️ **Architecture Overview :** [Click Here to View](docs/overview.md)

⭐️ **Build Plan :** [Click Here to View](docs/36-hour-build-plan.md)

---

### Project Created & Maintained By
❤ **Team [YOUR TEAM NAME]**
*   [Member 1 Name]
*   [Member 2 Name]
*   [Member 3 Name]
*   [Member 4 Name]
*   [Member 5 Name]
*   [Member 6 Name]

*(Add LinkedIn links here)*

---

### How-to-run

#### 1. Training (Google Colab)
1. Zip this workspace and upload it to Google Colab.
2. Open `SIH26052_Colab_Training.ipynb` and execute all cells using a T4 GPU.
3. Download the resulting `checkpoints/fpga_model.onnx`.

#### 2. Desktop Edge Simulation
To simulate the Jetson/FPGA edge environment processing the dual-mic streams on your PC:
1. Clone this repository.
2. Install requirements: `pip install -r requirements.txt`
3. Run the realtime pipeline:
   `python3 src/inference/realtime_pipeline.py --model checkpoints/fpga_model.onnx`

#### 3. Physical FPGA Deployment (PYNQ Board)
1. Compile the ONNX model to an XModel inside the Vitis AI Docker container:
   `./src/inference/compile_for_pynq.sh`
2. Move `dccrn_anc.xmodel` to your PYNQ board.
3. Run the hardware-accelerated DPU inference wrapper:
   `python3 src/inference/pynq_deployment.py --model checkpoints/dccrn_anc.xmodel`

---

### Support
💙 If you like this project, give it a ⭐ and share it with friends! 
🙏 THANK YOU 🙏
