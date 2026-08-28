import numpy as np

class LMSFilter:
    """
    Lightweight Least Mean Squares (LMS) adaptive filter for residual noise suppression.
    """
    def __init__(self, filter_length=64, step_size=0.01):
        self.filter_length = filter_length
        self.mu = step_size
        self.weights = np.zeros(self.filter_length)
        self.buffer = np.zeros(self.filter_length)

    def process(self, reference_signal, desired_signal):
        reference_signal = np.asarray(reference_signal).flatten()
        desired_signal = np.asarray(desired_signal).flatten()
        
        n_samples = len(desired_signal)
        error_signal = np.zeros(n_samples)
        estimated_noise = np.zeros(n_samples)
        
        for i in range(n_samples):
            self.buffer[1:] = self.buffer[:-1]
            self.buffer[0] = reference_signal[i]
            
            y = np.dot(self.weights, self.buffer)
            estimated_noise[i] = y
            
            e = desired_signal[i] - y
            error_signal[i] = e
            
            self.weights = self.weights + 2 * self.mu * e * self.buffer
            
        return error_signal, estimated_noise

    def reset(self):
        self.weights = np.zeros(self.filter_length)
        self.buffer = np.zeros(self.filter_length)

if __name__ == "__main__":
    filter_length = 32
    lms = LMSFilter(filter_length=filter_length, step_size=0.01)
    
    t = np.linspace(0, 1, 16000)
    noise = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(len(t))
    signal = np.sin(2 * np.pi * 1000 * t)
    mixed = signal + noise
    
    enhanced, _ = lms.process(noise, mixed)
    print(f"Processed {len(enhanced)} samples.")
