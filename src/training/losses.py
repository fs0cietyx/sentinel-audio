import torch
import torch.nn as nn

class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        """
        Args:
            preds: Predicted waveforms, shape (batch_size, num_samples)
            targets: Target clean waveforms, shape (batch_size, num_samples)
        Returns:
            Negative SI-SNR loss (to be minimized).
        """
        # Ensure zero-mean
        preds = preds - torch.mean(preds, dim=1, keepdim=True)
        targets = targets - torch.mean(targets, dim=1, keepdim=True)
        
        # <s, s>
        target_energy = torch.sum(targets ** 2, dim=1, keepdim=True) + self.eps
        # <s_hat, s>
        dot_product = torch.sum(preds * targets, dim=1, keepdim=True)
        
        # s_target = (<s_hat, s> / <s, s>) * s
        s_target = (dot_product / target_energy) * targets
        
        # e_noise = s_hat - s_target
        e_noise = preds - s_target
        
        # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True)
        e_noise_energy = torch.sum(e_noise ** 2, dim=1, keepdim=True)
        
        si_snr = 10 * torch.log10((s_target_energy + self.eps) / (e_noise_energy + self.eps))
        
        # We return the negative SI-SNR to minimize it
        return -torch.mean(si_snr)

class MultiLoss(nn.Module):
    """
    Combines SI-SNR, L1/L2 on spectrogram, and a placeholder Perceptual Loss.
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1):
        super().__init__()
        self.si_snr = SISNRLoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, pred_wav, target_wav, pred_spec, target_spec):
        # Time-domain SI-SNR loss
        loss_sisnr = self.si_snr(pred_wav, target_wav)
        
        # Frequency-domain magnitude L1 loss
        pred_mag = torch.sqrt(pred_spec[:, 0]**2 + pred_spec[:, 1]**2 + 1e-8)
        target_mag = torch.sqrt(target_spec[:, 0]**2 + target_spec[:, 1]**2 + 1e-8)
        loss_spec = self.l1(pred_mag, target_mag)
        
        # Placeholder for Perceptual Loss (could be PESQ approximation or feature matching)
        loss_perceptual = torch.tensor(0.0, device=pred_wav.device, requires_grad=True)
        
        total_loss = self.alpha * loss_sisnr + self.beta * loss_spec + self.gamma * loss_perceptual
        return total_loss, {"si_snr": loss_sisnr.item(), "spec_l1": loss_spec.item()}
