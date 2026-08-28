# Model Architecture & Training (Agent 2)

## Objective
Design a state-of-the-art Deep Learning model capable of operating in the complex domain to perform active noise cancellation, alongside a robust training framework.

## Core Responsibilities
1. **Network Architecture:** Implement a time-frequency domain network (e.g., Deep Complex Convolutional Recurrent Network - DCCRN).
   - Must process both full-band and sub-band features.
   - Must operate in the complex domain to preserve phase information.
2. **Loss Functions:**
   - **SI-SNR (Scale-Invariant Signal-to-Noise Ratio):** Primary metric for waveform reconstruction.
   - **L1/L2 Loss:** For spectrogram magnitude constraints.
   - **Perceptual Loss:** To optimize for human hearing characteristics.
3. **Evaluation Metrics:** Integrate STOI (Speech Intelligibility) and PESQ (Speech Quality) calculations during the validation step.

## File Structure to Build
* `src/models/anc_network.py`: The PyTorch `nn.Module` definition.
* `src/training/train.py`: The main training loop, loss calculations, and optimizer setup.
* `src/training/losses.py`: Custom SI-SNR and Perceptual loss implementations.

## Integration Points
Receives data from the Data Engineer pipeline. Outputs trained `.pth` model weights for the Edge Deployment Specialist.
