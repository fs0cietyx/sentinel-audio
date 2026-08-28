import torch
import torch.nn as nn

class SISNRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, target):
        # Subtract mean
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)
        
        # Scale-Invariant scaling factor
        alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + self.eps) / (torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps)
        target_scaled = alpha * target
        
        noise = preds - target_scaled
        
        val = (torch.sum(target_scaled ** 2, dim=-1) + self.eps) / (torch.sum(noise ** 2, dim=-1) + self.eps)
        val = 10 * torch.log10(val)
        return -torch.mean(val)

class MultiLoss(nn.Module):
    """
    State-of-the-Art Loss function for Speech Enhancement.
    Combines SI-SNR (for waveform structure) with L1 Magnitude Loss (for scale and phase anchoring).
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.si_snr = SISNRLoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred_wav, target_wav, pred_spec, target_spec):
        # 1. SI-SNR Loss on the waveform (handles structural shape)
        loss_sisnr = self.si_snr(pred_wav, target_wav)
        
        # 2. L1 Loss on the Spectrogram Magnitude
        # CRITICAL: This anchors the model's output scale so it doesn't output 40x louder audio!
        pred_mag = torch.abs(torch.complex(pred_spec[:,0], pred_spec[:,1]))
        target_mag = torch.abs(torch.complex(target_spec[:,0], target_spec[:,1]))
        loss_mag = self.l1(pred_mag, target_mag)
        
        return (self.alpha * loss_sisnr) + ((1 - self.alpha) * loss_mag)
