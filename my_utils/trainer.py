from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch import cuda, isinf, no_grad
import torch.nn as nn
from my_criterion import complement_entropy, self_regularized_entropy
from my_utils import util
from math import isnan
import torch


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class Trainer:
    device = 'cuda' if cuda.is_available() else 'cpu'

    def __init__(self, model, loader, lr, num_classes, loss_function, lr_step, lr_step_gamma, warmup_epochs=5, clip=0):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        self.warmup_scheduler = WarmUpLR(self.optimizer, len(loader) * warmup_epochs)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_step, gamma=lr_step_gamma)
        self.model.to(self.device)
        # loss
        self.train_loss_list, self.valid_loss_list, self.test_loss = [], [], None
        # my loss list
        self.loss_function = loss_function
        self.cross_entropy = nn.CrossEntropyLoss()
        self.complement_entropy = complement_entropy.ComplementEntropy(num_classes)
        self.self_regularized_entropy = self_regularized_entropy.SelfRegularizedEntropy(num_classes)
        # accuracy
        self.total, self.top1_correct, self.top5_correct = 0, 0, 0
        self.train_top1_acc_list, self.valid_top1_acc_list, self.test_top1_acc = [], [], None
        self.train_top5_acc_list, self.valid_top5_acc_list, self.test_top5_acc = [], [], None
        # gradient clipping constant
        self.clip = clip

    def reset_acc_members(self):
        self.total, self.top1_correct, self.top5_correct = 0, 0, 0

    def measure_acc(self, top1_acc, top5_acc, num_samples):
        self.top1_correct += top1_acc
        self.top5_correct += top5_acc
        self.total += num_samples
        top1_acc_rate = 100. * (self.top1_correct / self.total)
        top5_acc_rate = 100. * (self.top5_correct / self.total)
        return top1_acc_rate, top5_acc_rate

    ### Insert your loss function here! ################################################################

    def select_loss_function(self):
        return eval('self.' + self.loss_function)

    def ERM(self, outputs, targets):
        return self.cross_entropy(outputs, targets)

    def COT(self, outputs, targets):
        return self.cross_entropy(outputs, targets) - self.complement_entropy(outputs, targets)

    def SRE(self, outputs, targets):  # proposed method
        return self.cross_entropy(outputs, targets) - self.self_regularized_entropy(outputs, targets)

    ####################################################################################################

    def one_epoch(self, loader, lr_warmup, front_msg='', cur_epoch=0):
        ### [] is not that important to training.
        batch_loss = 0
        progress_bar = util.ProgressBar()
        top1_acc_rate, top5_acc_rate = 0, 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            if lr_warmup:
                self.warmup_scheduler.step()
            inputs, targets = inputs.to(Trainer.device), targets.to(Trainer.device)
            ### inference
            outputs = self.model(inputs)
            ### [for measuring accuracy]
            top1_acc, top5_acc = util.topk_acc(outputs, targets)
            top1_acc_rate, top5_acc_rate = self.measure_acc(top1_acc, top5_acc, num_samples=targets.size(0))
            ### choose loss function (core)
            if front_msg == 'Train':
                loss = self.select_loss_function()(outputs, targets)
            else:
                loss = self.ERM(outputs, targets)
            # [if loss is nan...]
            if isnan(loss) and front_msg == 'Train':
                print('[Error] nan loss, stop <{}>.'.format(front_msg))
                exit(1)
            ### [for training]
            if front_msg == 'Train':
                ### zero_grad
                self.optimizer.zero_grad()
                ### back_propagation
                loss.backward()
                ### gradient clipping
                if self.clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                ### optimization
                self.optimizer.step()

            ### [loss memo.]
            batch_loss = batch_loss + loss.item()
            ### [progress_bar]
            progress_bar.progress_bar(front_msg, cur_epoch + 1, batch_idx, len(loader),
                                      msg='Loss: %.3f | Acc.: [top1] %.3f%%, [top5] %.3f%%'
                                          % (loss, top1_acc_rate, top5_acc_rate))
        return batch_loss, top1_acc_rate.item(), top5_acc_rate.item()

    def train(self, cur_epoch, loader, lr_warmup):
        self.reset_acc_members()
        self.model.train()
        train_loss, top1_acc_rate, top5_acc_rate = self.one_epoch(loader, lr_warmup,
                                                                  front_msg='Train', cur_epoch=cur_epoch)
        self.train_loss_list.append(train_loss)
        self.train_top1_acc_list.append(top1_acc_rate)
        self.train_top5_acc_list.append(top5_acc_rate)

        ### learning rate decay
        self.lr_scheduler.step()

    def valid(self, cur_epoch, loader):
        self.reset_acc_members()
        self.model.eval()
        with torch.no_grad():
            valid_loss, top1_acc_rate, top5_acc_rate = self.one_epoch(loader, lr_warmup=False,
                                                                      front_msg='Valid', cur_epoch=cur_epoch)
            self.valid_loss_list.append(valid_loss)
            self.valid_top1_acc_list.append(top1_acc_rate)
            self.valid_top5_acc_list.append(top5_acc_rate)

    def test(self, loader):
        self.reset_acc_members()
        self.model.eval()
        with torch.no_grad():
            self.test_loss, self.test_top1_acc, self.test_top5_acc = self.one_epoch(loader,
                                                                                lr_warmup=False, front_msg='Test')
