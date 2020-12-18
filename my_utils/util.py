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


def topk_acc(output, target, topk=(1,), sum_mode=True):
    '''
    topk = (1, )
    topk = (1, 5)
    '''
    maxk = max(topk)
    batch_size = target.size()[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        if sum_mode:
            res.append(correct_k)
        else: # 'avg_mode'
            res.append(correct_k.mul_(100.0 / batch_size))
    return res
