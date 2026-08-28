# Architectural Decisions

### 1. Hybrid Filter over End-to-End Deep Learning
**Decision:** We chose to run a classical LMS filter alongside the AI model instead of relying entirely on a massive Neural Network.
**Reason:** Stationary noises (helicopter hums) are easily handled by cheap classical filters. Wasting expensive FPGA DSP slices on stationary noise violates the constraint for ultra-low latency. The AI is strictly reserved for non-stationary impulsive noise.

### 2. FP32 ONNX Export with QAT-Hardened Weights
**Decision:** We exported the QAT model weights without PyTorch's native `FakeQuantize` operator nodes attached.
**Reason:** Proprietary hardware compilers (like Vitis AI `vai_q_onnx` or TensorRT) often clash with PyTorch's specific INT8 graph structures. By exporting QAT-hardened weights in a standard ONNX graph, we allow the silicon's native compiler to cast the weights efficiently without accuracy degradation.

### 3. Static 3-Frame STFT Chunks
**Decision:** The ONNX graph strictly rejects `dynamic_axes` and expects exactly a `(1, 2, 257, 3)` tensor.
**Reason:** Dynamic memory allocation on FPGAs induces massive latency jitter. Forcing a static 512-sample (3-frame) chunk guarantees deterministic execution latency well below the <15ms DRDO requirement.
