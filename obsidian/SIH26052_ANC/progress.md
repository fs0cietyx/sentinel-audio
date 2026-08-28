# Project Diary

## [2026-08-28] - Project Initiation & Pipeline Scaffolding
- **What happened:** Initialized the hackathon project for DRDO SIH26052.
- **What changed:** Generated `dataset.py`, `anc_network.py` (DCCRN), `losses.py`, `train.py`, and ONNX deployment scripts.

## [2026-08-28] - Data Acquisition & Pipeline Debugging
- **What happened:** Acquired datasets (MAD, LibriSpeech) and debugged critical PyTorch multiprocessing/audio loading crashes.
- **What changed:** Bypassed macOS `torchaudio` ffmpeg crash by natively reading via `soundfile`. Enforced strict 3-second audio padding. Fixed DCCRN tensor dimension mismatch. Added `hann_window`.

## [2026-08-28] - Baseline Established & FPGA Preparation
- **What happened:** Successfully trained the baseline (STOI: 0.71). Hardened the network for FPGA.
- **What changed:** Replaced `LeakyReLU` with `ReLU`. Stripped dynamic axes from `export_onnx.py`. Successfully generated static `fpga_model.onnx`.

## [2026-08-28] - Pitch Deck Realignment & Colab Packaging
- **What happened:** Reviewed judge's feedback on the PPT deck. Discovered a dangerous misalignment where the deck pitched legacy Dual-Mic LMS while the codebase implemented state-of-the-art Single-Channel DCCRN.
- **What changed:** 
  - Rewrote the 6-slide pitch deck to aggressively pitch the Deep Learning (DCCRN) approach and the static FPGA pipeline.
  - Injected PyTorch `QuantStub` and `DeQuantStub` into `anc_network.py` to make it officially Vitis AI (PYNQ) compliant.
  - Packaged the codebase into `colab_workspace.zip` for immediate migration to cloud GPUs.
- **What should happen next:** Upload to Google Colab and commence scale-up training.
