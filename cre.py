import torch
import torch.nn as nn
import torch.nn.functional as F


class ClasswiseRegulatedEntropy(nn.Module):
    def __init__(self, K, device, psi=1e-7):
        assert K > 0 and type(K) is int, 'Hyper-parameter "K" should be a integer (> 0).'
        self.K = K                                                   # K
        self.psi = psi                                               # ψ
        self.device = device                                         # {'cpu', 'cuda:0', 'cuda:1', ...}
        super(ClasswiseRegulatedEntropy, self).__init__()

    def forward(self, yHat, y):
        # [Pseudo code]
        # e = - (yHat / norm) log (yHat / norm)
        # e += - K * ( (ψ / norm) log (ψ / norm) )
        # e = e ⊙ (yHat + γ) (⊙: Hadamard Product)
        # e = e ⊙ yHat_zerohot (To ignore all ground truth classes)
        # e = scalar_sum(e)
        # e = e / N

        num_classes = yHat.shape[1]                                  # C
        batch_size = len(y)                                          # N
        yHat = F.softmax(yHat, dim=1)
        yHat_max = yHat.data.max(dim=1).values
        yHat_max = yHat_max.view([-1, 1])

        psi_distribution = torch.ones_like(yHat) * self.psi
        yHat_zerohot = torch.ones(batch_size, num_classes).scatter_(1, y.view(batch_size, 1).data.cpu(), 0)
        norm = yHat + psi_distribution * self.K + 1e-10
        classwise_entropy = (yHat / norm) * torch.log((yHat / norm) + 1e-10)
        classwise_entropy += ((psi_distribution / norm) * torch.log((psi_distribution / norm) + 1e-10)) * self.K
        gamma = 1e-10

        classwise_entropy *= (yHat_max + gamma)
        classwise_entropy *= yHat_zerohot.to(device=self.device)
        entropy = float(torch.sum(classwise_entropy))
        entropy /= batch_size
        return entropy
