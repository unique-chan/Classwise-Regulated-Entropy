import time
from torch import manual_seed, cuda, backends
import numpy as np
import random


def fix_random_seed(seed=1234):
    # Ref.: https://github.com/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    backends.cudnn.deterministic = True


def topk_acc(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    topk_acc_list = [correct[:k].view(-1).float().sum(0, keepdim=True) for k in topk]
    return topk_acc_list


class ProgressBar:
    last_time = time.time()
    begin_time = last_time
    TOTAL_BAR_LENGTH = 10.

    def __init__(self):
        pass

    @staticmethod
    def format_time(seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)
        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D '
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h '
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm '
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's '
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms '
            i += 1
        if f == '':
            f = '0ms'
        return f

    @staticmethod
    def progress_bar(front_msg, epoch_num, current_batch_idx, batch_size, msg=None):
        if current_batch_idx == 0:
            ProgressBar.last_time = time.time()
            ProgressBar.begin_time = ProgressBar.last_time
        cur_len = int(ProgressBar.TOTAL_BAR_LENGTH * current_batch_idx / batch_size)
        rest_len = int(ProgressBar.TOTAL_BAR_LENGTH - cur_len) - 1
        print('%s | Epoch: %5d [' % (front_msg, epoch_num), end='')
        for i in range(cur_len):
            print('■', end='')
        print('▶', end='')
        for i in range(rest_len):
            print(' ', end='')
        print(']', end='')
        cur_time = time.time()
        step_time = cur_time - ProgressBar.last_time
        ProgressBar.last_time = cur_time
        tot_time = cur_time - ProgressBar.begin_time
        msg_list = list()
        msg_list.append('  Step: %s' % ProgressBar.format_time(step_time))
        msg_list.append(' | Total Time: %s' % ProgressBar.format_time(tot_time))
        if msg:
            msg_list.append(' | ' + msg)
        msg = ''.join(msg_list)
        print(msg, end='')
        print(' | Batch: %d/%d ' % (current_batch_idx + 1, batch_size), end='')
        if current_batch_idx < batch_size - 1:
            print('', end='\r')
        else:
            print('', end='\n')

    # def progress_bar(front_msg, epoch_num, current_batch_idx, batch_size, msg=None):
    #     if current_batch_idx == 0:
    #         ProgressBar.last_time = time.time()
    #         ProgressBar.begin_time = ProgressBar.last_time
    #     cur_len = int(ProgressBar.TOTAL_BAR_LENGTH * current_batch_idx / batch_size)
    #     rest_len = int(ProgressBar.TOTAL_BAR_LENGTH - cur_len) - 1
    #     sys.stdout.write('%s | Epoch: %5d [' % (front_msg, epoch_num))
    #     for i in range(cur_len):
    #         sys.stdout.write('■')
    #     sys.stdout.write('▶')
    #     for i in range(rest_len):
    #         sys.stdout.write(' ')
    #     sys.stdout.write(']')
    #     cur_time = time.time()
    #     step_time = cur_time - ProgressBar.last_time
    #     ProgressBar.last_time = cur_time
    #     tot_time = cur_time - ProgressBar.begin_time
    #     msg_list = list()
    #     msg_list.append('  Step: %s' % ProgressBar.format_time(step_time))
    #     msg_list.append(' | Total Time: %s' % ProgressBar.format_time(tot_time))
    #     if msg:
    #         msg_list.append(' | ' + msg)
    #     msg = ''.join(msg_list)
    #     sys.stdout.write(msg)
    #     sys.stdout.write(' | Batch: %d/%d ' % (current_batch_idx + 1, batch_size))
    #     if current_batch_idx < batch_size - 1:
    #         sys.stdout.write('\r')
    #     else:
    #         sys.stdout.write('\n')
    #     sys.stdout.flush()
