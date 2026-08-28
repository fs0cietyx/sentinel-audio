# Edge Deployment & Hardware (Agent 3)

## Objective
Optimize the trained DCCRN model for real-time inference on FPGA/SoC hardware (AMD PYNQ / Vitis AI / Jetson) and establish the blueprint for Phase 2 dual-mic physical integration.

## Core Responsibilities
1. **FPGA Model Optimization:** Ensure the PyTorch model complies with High-Level Synthesis (HLS) rules.
   - Inject `QuantStub` for INT8 precision execution.
   - Enforce strictly static ONNX graphs (no dynamic memory allocations).
2. **Phase 2 Hybrid Architecture Mockup:** Build the conceptual Python wrapper (`realtime_pipeline.py`) that demonstrates how the AI interacts with the physical hardware. 
   - A lightweight adaptive filter (LMS) processes the reference mic to cancel stationary noise.
   - The AI processes the primary mic to predict and subtract impulsive noise.

## File Structure Built
* `src/inference/export_onnx.py`: Script to freeze the DCCRN and export `fpga_model.onnx` with strict static dimensions.
* `src/inference/hybrid_filter.py`: The classical LMS filter implementation (Phase 2 component).
* `src/inference/realtime_pipeline.py`: The simulation script demonstrating the Dual-Mic + AI integration.

## Integration Points
Takes the trained `best_anc_model.pth` from Colab, quantizes it, exports it to ONNX, and proves the latency remains under the DRDO real-time threshold.
