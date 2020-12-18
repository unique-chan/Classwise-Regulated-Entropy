from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch import cuda, isinf
import torch.nn as nn


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class Trainer:
    def __init__(self, model, loader, lr, warmup_epochs=5):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        self.warmup_scheduler = WarmUpLR(self.optimizer, len(loader) * warmup_epochs)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()

    def train(self, epoch, loader, lr_warmup):
        self.model.train()
        train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            if lr_warmup:
                self.warmup_scheduler.step()

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.cross_entropy(outputs, targets)
            if isinf(loss):
                print('[Error] nan loss, stop training.')
                exit(1)

            # loss.backward(retain_graph=True)
            loss.backward()

            self.optimizer.step()
            train_loss = train_loss + loss.item()
