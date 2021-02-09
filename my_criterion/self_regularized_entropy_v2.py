import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfRegularizedEntropy(nn.Module):
    def __init__(self, num_classes, alpha, child_prob=1e-7):
        assert alpha > 0 and type(alpha) is int, 'Hyper-parameter "alpha" should be a integer (> 0).'
        self.classes = num_classes
        self.alpha = alpha
        self.child_prob = child_prob
        super(SelfRegularizedEntropy, self).__init__()

    def forward(self, yHat, y):
        batch_size = len(y)
        yHat = F.softmax(yHat, dim=1)
        yHat_child = torch.ones_like(yHat) * self.child_prob
        # scatter_(dim, index, src)
        yHat_zerohot = torch.ones(batch_size, self.classes).scatter_(1, y.view(batch_size, 1).data.cpu(), 0)
        norm = yHat + yHat_child * self.alpha + 1e-10
        classwise_entropy = (yHat / norm) * torch.log((yHat / norm) + 1e-10)
        # for _ in range(self.alpha):
        #     classwise_entropy += (yHat_child / norm) * torch.log((yHat_child / norm) + 1e-10)
        # <=> classwise_entropy += ((yHat_child / norm) * torch.log((yHat_child / norm) + 1e-10)) * self.alpha
        classwise_entropy += ((yHat_child / norm) * torch.log((yHat_child / norm) + 1e-10)) * self.alpha
        classwise_entropy *= yHat_zerohot.cuda()
        entropy = float(torch.sum(classwise_entropy))
        entropy /= batch_size
        entropy /= (self.classes - 1)
        return entropy
