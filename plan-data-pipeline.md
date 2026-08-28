# Data Engineering Pipeline (Agent 1)

## Objective
Build a scalable, dynamic data generation pipeline to mix clean speech with curated defense noises, mimicking real-world combat and industrial scenarios.

## Core Responsibilities
1. **Dynamic Mixing (On-the-fly):** Instead of saving massive mixed audio files, mix clean speech and noise during training. This provides infinite variations and prevents overfitting.
2. **SNR Scaling:** Randomly scale noise arrays to meet target SNRs (ranging from -10dB to +20dB).
3. **Data Augmentations:**
   - **Reverberation:** Apply Room Impulse Responses (RIRs) to simulate physical spaces.
   - **Clipping:** Simulate microphone peaking caused by loud impulsive noises (gunshots/artillery).
   - **Random Noise Mixing:** Combine stationary (wind) and non-stationary (sirens) noises.

## File Structure to Build
* `src/data/dataset.py`: PyTorch `Dataset` and `DataLoader` classes.
* `src/data/augmentations.py`: Audio processing functions for RIR convolution and clipping.

## Integration Points
Passes the `(Augmented Noisy Audio, Clean Target Speech)` tensor pairs directly to the ML Architect's training loop.
