from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch import cuda, isinf, no_grad
import torch.nn as nn
from my_criterion import self_regularized_entropy
from my_utils import util


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class Trainer:
    device = 'cuda' if cuda.is_available() else 'cpu'

    def __init__(self, model, loader, lr, warmup_epochs=5):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        self.warmup_scheduler = WarmUpLR(self.optimizer, len(loader) * warmup_epochs)
        self.model.to(self.device)
        # loss
        self.train_loss_list, self.valid_loss_list, self.test_loss_list = [], [], []
        # loss function
        self.cross_entropy = nn.CrossEntropyLoss()
        self.regularized_entropy = self_regularized_entropy.SelfRegularizedEntropy(4)
        # accuracy
        self.total, self.top1_correct, self.top5_correct = 0, 0, 0

    def reset_acc_members(self):
        self.total, self.top1_correct, self.top5_correct = 0, 0, 0

    def measure_acc(self, top1_acc, top5_acc, num_samples):
        self.top1_correct += top1_acc
        self.top5_correct += top5_acc
        self.total += num_samples
        top1_acc_rate = 100. * (self.top1_correct / self.total)
        top5_acc_rate = 100. * (self.top5_correct / self.total)
        return top1_acc_rate, top5_acc_rate

    def one_epoch(self, loader, lr_warmup, front_msg='', cur_epoch=0):
        batch_loss = 0
        progress_bar = util.ProgressBar()
        for batch_idx, (inputs, targets) in enumerate(loader):
            if lr_warmup:
                self.warmup_scheduler.step()
            inputs, targets = inputs.to(Trainer.device), targets.to(Trainer.device)
            ### inference
            outputs = self.model(inputs)
            ### for measuring accuracy
            top1_acc, top5_acc = util.topk_acc(outputs, targets)
            top1_acc_rate, top5_acc_rate = self.measure_acc(top1_acc, top5_acc, num_samples=targets.size(0))
            ### for back-propagation
            self.optimizer.zero_grad()
            loss = self.cross_entropy(outputs, targets)
            # loss = self.regularized_entropy(outputs, targets)
            if isinf(loss) and front_msg == 'training':
                print('[Error] nan loss, stop {}.'.format(front_msg))
                exit(1)
            loss.backward()
            self.optimizer.step()
            batch_loss = batch_loss + loss.item()
            ### progress_bar
            progress_bar.progress_bar(front_msg, cur_epoch + 1, batch_idx, len(loader),
                                      msg='Loss: %.3f | Acc: [top-1] %.3f%%, [top-5] %.3f%%'
                                          % (loss / (batch_idx + 1), top1_acc_rate, top5_acc_rate))
        return batch_loss / len(loader)

    def train(self, cur_epoch, loader, lr_warmup):
        self.reset_acc_members()
        self.model.train()
        train_loss = self.one_epoch(loader, lr_warmup, front_msg='Training', cur_epoch=cur_epoch)
        self.train_loss_list.append(train_loss)

    def valid(self, cur_epoch, loader):
        self.reset_acc_members()
        self.model.eval()
        valid_loss = self.one_epoch(loader, lr_warmup=False, front_msg='Validation', cur_epoch=cur_epoch)
        self.valid_loss_list.append(valid_loss)

    def test(self, loader):
        self.reset_acc_members()
        self.model.eval()
        test_loss = self.one_epoch(loader, lr_warmup=False, front_msg='Test')
        self.test_loss_list.append(test_loss)

