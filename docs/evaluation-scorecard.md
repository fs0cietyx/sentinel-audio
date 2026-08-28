# Evaluation Scorecard

## Innovation
- **Breakthrough:** Uses advanced AI/ML combined with hardware-level edge optimization (TensorRT, quantization) and hybrid signal processing.

## Invention
- **High:** Asks for distinct, complex technical components: complex domain modeling, specific loss functions (SI-SNR, perceptual), edge optimization, and physical dual-mic hardware integration.

## Technical Feasibility
- **Workable:** Achievable in 36 hours *if* the team divides labor well between model training (cloud) and edge deployment/hardware wiring (Jetson).

## Impact and Benefits
- **Direct:** Enables reliable and intelligible communication in defense, aerospace, and high-noise environments. Directly measurable via requested metrics: SNR > 15 dB, STOI > 0.85, and PESQ > 2.5.

## Architecture
- **Multi-Layer:** Expect a physical sensor layer (primary + reference mics), an optimized edge-compute layer (Jetson AGX Orin running ONNX/TensorRT), and a hybrid processing stack (AI + LMS adaptive filter).

## Overall Verdict: GREEN
- Strong balance of real innovation, manageable effort and a workable competitive field, a good default pick for most teams.

**Biggest strength:** Scope is extremely defined (specific hardware, metrics, and optimization techniques). Less time lost interpreting what to build.

**Biggest risk:** Hardware integration and edge deployment. Moving a model from PyTorch to TensorRT on a Jetson board while dealing with live audio streams is notorious for latency and driver issues.

**Validate before you commit:** Ensure your team has access to an AI-enabled SoC/DSP (like Jetson) and dual microphones, and that someone knows how to deploy ONNX/TensorRT models.

## What The Evaluator Will Likely Ask
1. Walk me through your model optimization pipeline. How did you use quantization and TensorRT to reduce latency?
2. How is your system utilizing both the primary and reference microphones?
3. Show me the real-time metrics. Are we actually hitting STOI > 0.85 and PESQ > 2.5 right now?
4. What happens when connectivity drops, or input data is clipped or heavily reverberated? (Did you use the requested data augmentations?)
5. Explain the role of the lightweight adaptive filter (LMS) in your hybrid pipeline.
