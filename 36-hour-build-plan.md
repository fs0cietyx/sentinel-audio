# 36-Hour Build Plan

## The Concept (Hours 0–4)
- **Architecture Finalization:** Solidify the "Hybrid Narrative": Stage 1 is the Single-Channel DCCRN AI for impulsive noises. Stage 2 is the Dual-Mic LMS wrapper for stationary noises. 
- **Data Engineering:** Build the dynamic PyTorch `ANCDataset` to randomly mix LibriSpeech and MAD audio on-the-fly, bypassing macOS ffmpeg bugs via `soundfile`.

## The Core AI Engine (Hours 4–22)
- **Model Construction:** Build the Deep Complex Convolutional Recurrent Network (DCCRN). 
- **Hardware Hardening:** Replace hardware-expensive operations (LeakyReLU -> ReLU) to save FPGA DSP slices. Add PyTorch Quantization-Aware Training (QAT) stubs.
- **Baseline Training:** Run the first 5 epochs locally to establish baseline STOI/PESQ metrics and prove the pipeline works end-to-end.

## Cloud Scaling & FPGA Compilation (Hours 22–30)
- **Cloud Migration:** Package the `src/` codebase and upload to Google Colab.
- **Scale Up:** Download the 100-hour `train-clean-100` dataset directly in Colab and execute QAT training overnight to hit STOI > 0.85.
- **ONNX Export:** Export the trained weights to a strictly static `fpga_model.onnx` graph (stripping dynamic axes) for the Vitis AI compiler.

## Polish & Pitch (Hours 30–36)
- **Hybrid Mockup:** Write `realtime_pipeline.py` to simulate the Phase 2 Dual-Mic LMS wrapper interacting with the AI engine.
- **Presentation:** Build a pitch deck that clearly separates what we *built* (the extremely difficult AI engine) from the *roadmap* (the straightforward physical mic wiring). Shock the judges with the baseline metrics.
