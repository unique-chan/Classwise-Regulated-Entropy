import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfRegularizedEntropy(nn.Module):
    def __init__(self, C, K, psi=1e-7):
        assert K > 0 and type(K) is int, 'Hyper-parameter "alpha" should be a integer (> 0).'
        self.C = C                                                   # C (number of classes)
        self.K = K                                                   # K
        self.psi = psi                                               # ψ
        super(SelfRegularizedEntropy, self).__init__()

    def forward(self, yHat, y):
        batch_size = len(y)                                          # N
        yHat = F.softmax(yHat, dim=1)

        psi_distribution = torch.ones_like(yHat) * self.psi
        # yHat_zerohot = torch.ones(batch_size, self.C).scatter_(1, y.view(batch_size, 1).data.cpu(), 0)
        yHat_zerohot = torch.ones(batch_size, self.C).scatter_(1, y.view(batch_size, 1).data, 0)
        norm = yHat + psi_distribution * self.K + 1e-10
        # e = - (yHat / norm) log (yHat / norm)
        classwise_entropy = (yHat / norm) * torch.log((yHat / norm) + 1e-10)
        # e += - K * ( (ψ / norm) log (ψ / norm) )
        classwise_entropy += ((psi_distribution / norm) * torch.log((psi_distribution / norm) + 1e-10)) * self.K
        # e = e ⊙ (yHat + γ) (⊙: Hadamard Product)
        gamma = 0.3
        classwise_entropy *= (yHat + gamma)
        # e = e ⊙ yHat_zerohot (To ignore all ground truth classes)
        classwise_entropy *= yHat_zerohot
        # classwise_entropy *= yHat_zerohot.cuda()
        # e = scalar_sum(e)
        entropy = float(torch.sum(classwise_entropy))
        # e = e / N
        entropy /= batch_size
        return entropy
