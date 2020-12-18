import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfRegularizedEntropy(nn.Module):
    def __init__(self, num_classes):
        self.classes = num_classes
        self.batch_size = 0
        super(SelfRegularizedEntropy, self).__init__()

    def forward(self, yHat, y):
        self.batch_size = len(y)
        yHat = F.softmax(yHat, dim=1)

        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7   # numerical trick
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.classes)
        return loss


