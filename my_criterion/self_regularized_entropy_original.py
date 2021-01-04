import torch
import torch.nn as nn
import torch.nn.functional as F


# commit test
class SelfRegularizedEntropy(nn.Module):
    def __init__(self, num_classes):
        self.classes = num_classes
        self.batch_size = 0
        super(SelfRegularizedEntropy, self).__init__()

    def forward(self, yHat, y):
        self.batch_size = len(y)
        yHat = F.softmax(yHat, dim=1)
        yHat_child_pair = yHat * (1. / (self.classes - 1))
        norm = yHat_child_pair + yHat + 1e-7
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        prob_yHat = yHat / norm
        prob_log_yHat = torch.log(prob_yHat + 1e-10)
        prob_yHat_child = yHat_child_pair / norm
        prob_log_yHat_child = torch.log(prob_yHat_child + 1e-10)
        output = (prob_yHat * prob_log_yHat + prob_yHat_child * prob_log_yHat_child) * y_zerohot.cuda()
        loss = float(torch.sum(output))
        loss = loss / self.batch_size
        loss = loss / (self.classes - 1)
        return loss
