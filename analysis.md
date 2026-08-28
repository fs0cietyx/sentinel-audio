**Innovation Scope: Breakthrough**
* Uses advanced AI/ML coupled with edge-optimized deployment (TensorRT/ONNX) and a hybrid DSP approach.

**Invention Effort: High**
* Scored from the number of distinct components asked for: complex domain AI models, edge optimization (quantization/pruning), and physical dual-microphone hardware integration on specialized boards (Jetson AGX Orin).
* Effort score: 9
* This is a Hardware PS; budget real time for physical prototyping, sourcing parts, and testing, not just code.

**Competitive Landscape: Low Crowding**
* No other live SIH 2026 problem statement asks for a build this close to this one. The explicit requirement for edge hardware (Jetson) and physical microphone setups narrows the realistic field even further.
* **What most teams will build:** Many teams will stop at a Python/PyTorch script running on a laptop.
* **How to stand out:** Execute the full pipeline on actual edge hardware using ONNX/TensorRT, implement the dual-mic setup, and prove you hit the specific metrics (SNR > 15, STOI > 0.85, PESQ > 2.5).

**SWOT Snapshot**

* **Strengths:**
  * Expected Solution lists distinct requirements (metrics, loss functions, hardware) leaving no ambiguity.
  * Genuine modern-tech core (AI + Edge Compute) gives you real substance to demo.
* **Weaknesses:**
  * Requires specialized hardware (NVIDIA Jetson AGX Orin or similar) and dual microphones. Sourcing these and dealing with drivers is a risk.
* **Opportunities:**
  * Show a live dashboard hitting exact metric targets (SNR > 15 dB, STOI > 0.85, PESQ > 2.5) during the physical demo.
  * Implement the suggested "Hybrid" architecture (AI + LMS filter) to show deep understanding of signal processing.
* **Threats:**
  * Physical demos are unforgiving. Driver issues with edge hardware or microphone latency can ruin the presentation.
