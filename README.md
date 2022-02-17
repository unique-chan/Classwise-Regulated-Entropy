# Classwise-Regulated-Entropy (Pytorch)
**[Yechan Kim](https://github.com/unique-chan)**

[A paper will be uploaded via Arxiv!]()

🚧 Under Construction! (Do not fork already!!!)

## This repository contains:
* Code for Classwise-Regulated-Entropy (CRE) 
* For replication, we will soon provide a Pytorch implementation for CIFAR-10 classification with CRE.

## Prerequisites
* See `requirements.txt`.
```
torch
torchvision
```

## Code for the proposed method
* See `cre.py`.
* Loss = Cross Entropy - λ * Classwise Regulated Entropy (λ: modulating factor)
```python
class CRE(nn.Module):
    def __init__(self, K, device, psi=1e-7):
        assert K > 0 and type(K) is int, 'Hyper-parameter "K" should be a integer (> 0).'
        self.K = K                                                   # K
        self.psi = psi                                               # ψ
        self.device = device                                         # {'cpu', 'cuda:0', 'cuda:1', ...}
        super(CRE, self).__init__()

    def forward(self, yHat, y):
        # [Pseudo code]
        # (i)       e = - (yHat / norm) log (yHat / norm)
        # (ii)      e += - K * ( (ψ / norm) log (ψ / norm) )
        # (iii)     e = e ⊙ yHat_zerohot (To ignore all ground truth classes)
        # (iv)      e = e ⊙ (yHat + γ) (⊙: Hadamard Product)
        # (v)       e = scalar_sum(e)
        # (vi)      e = e / N

        kush = 1e-7                                                  # γ
        C = yHat.shape[1]                                            # number of classes
        N = len(y)                                                   # batch size

        # For (i), (ii)
        yHat = F.softmax(yHat, dim=1)
        VP = torch.ones_like(yHat) * self.psi                        # virtual distribution except for yHat
        norm = yHat + VP * self.K + kush
        e = (yHat / norm) * torch.log((yHat / norm) + kush)
        e += ((VP / norm) * torch.log((VP / norm) + kush)) * self.K

        # For (iii)
        yHat_zerohot = torch.ones(N, C).scatter_(1, y.view(N, 1).data.cpu(), 0)
        e *= yHat_zerohot.to(device=self.device)
        e = torch.sum(e, dim=1)

        # For (iv)
        yHat_gt = yHat.data * F.one_hot(y, C)
        yHat_gt = yHat_gt.data.max(dim=1).values
        e *= (yHat_gt + kush)

        # For (v), (vi)
        e = float(torch.sum(e))
        e /= N

        return e
```

## Script for generating SIFAR
* See `sifar.py`.
* Note that this work abbreviates as SIFAR for '<u>Semantically</u> similar samples from c<u>IFAR</u>-100.'
* You may need to prepare downloaded `CIFAR-100` before executing `sifar.py`. Kindly refer to [cifar2png](https://github.com/knjcode/cifar2png) and run `cifar2png` without superclass option.

## Contribution
If you find any bugs or have opinions for further improvements, please feel free to create a pull request or contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.

## Reference
1. Hao-Yun Chen, Pei-Hsin Wang, Chun-Hao Liu, Shih-Chieh Chang, Jia-Yu Pan, Yu-Ting Chen, Wei Wei, and Da-Cheng Juan. Complement objective training. arXiv preprint arXiv:1903.01182, 2019.
2. https://github.com/weiaicunzai/pytorch-cifar100
3. https://github.com/knjcode/cifar2png
