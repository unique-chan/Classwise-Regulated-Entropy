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
class ClasswiseRegulatedEntropy(nn.Module):
    def __init__(self, K, device, psi=1e-7):
        assert K > 0 and type(K) is int, 'Hyper-parameter "K" should be a integer (> 0).'
        self.K = K                                                   # K
        self.psi = psi                                               # ψ
        self.device = device                                         # {'cpu', 'cuda:0', ...}
        super(ClasswiseRegulatedEntropy, self).__init__()

   def forward(self, yHat, y):
        num_classes = yHat.shape[1]                                  # C
        batch_size = len(y)                                          # N
        yHat = F.softmax(yHat, dim=1)
        yHat_max = yHat.data.max(dim=1).values
        yHat_max = yHat_max.view([-1, 1])

        psi_distribution = torch.ones_like(yHat) * self.psi
        yHat_zerohot = torch.ones(batch_size, num_classes).\
            scatter_(1, y.view(batch_size, 1).data.cpu(), 0)
        norm = yHat + psi_distribution * self.K + 1e-10
        classwise_entropy = (yHat / norm) * torch.log((yHat / norm) + 1e-10)
        classwise_entropy += ((psi_distribution / norm) * 
                               torch.log((psi_distribution / norm) + 1e-10)) * self.K
        
        kush = 1e-10
        classwise_entropy *= (yHat_max + kush)
        classwise_entropy *= yHat_zerohot.to(device=self.device)
        entropy = float(torch.sum(classwise_entropy))
        entropy /= batch_size
        return entropy
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
