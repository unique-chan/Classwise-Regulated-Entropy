import torch
import torch.nn as nn
import torch.nn.functional as F


class CRE(nn.Module):
    def __init__(self, K, device, psi=1e-7):
        assert K > 0 and type(K) is int, 'Hyper-parameter "K" should be a integer (> 0).'
        self.K = K                                                   # K
        self.psi = psi                                               # ψ
        self.device = device                                         # {'cpu', 'cuda:0', 'cuda:1', ...}
        super(CRE, self).__init__()

    def forward(self, yHat, y):
        # [Pseudo code]
        # (i)       e = - (yHat / norm) log (yHat / norm)
        # (ii)      e += - K * ( (ψ / norm) log (ψ / norm) )
        # (iii)     e = e ⊙ yHat_zerohot (To ignore all ground truth classes)
        # (iv)      e = e ⊙ (yHat + γ) (⊙: Hadamard Product)
        # (v)       e = scalar_sum(e)
        # (vi)      e = e / N

        kush = 1e-7                                                  # γ
        C = yHat.shape[1]                                            # number of classes
        N = len(y)                                                   # batch size

        # For (i), (ii)
        yHat = F.softmax(yHat, dim=1)
        VP = torch.ones_like(yHat) * self.psi                        # VP: virtual distribution except for yHat
        norm = yHat + VP * self.K + kush
        e = (yHat / norm) * torch.log((yHat / norm) + kush)
        e += ((VP / norm) * torch.log((VP / norm) + kush)) * self.K

        # For (iii)
        yHat_zerohot = torch.ones(N, C).scatter_(1, y.view(N, 1).data.cpu(), 0)
        e *= yHat_zerohot.to(device=self.device)
        e = torch.sum(e, dim=1)

        # For (iv)
        yHat_gt = yHat.data * F.one_hot(y, C)
        yHat_gt = yHat_gt.data.max(dim=1).values
        e *= (yHat_gt + kush)

        # For (v), (vi)
        e = float(torch.sum(e))
        e /= N

        return e
