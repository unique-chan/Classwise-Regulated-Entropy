def topk_acc(output, target, topk=(1,), sum_mode=True):
    '''
    topk = (1, )
    topk = (1, 5)
    '''
    maxk = max(topk)
    batch_size = target.size(0)
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