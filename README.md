# Classwise-Regulated-Entropy (Pytorch)
**[Yechan Kim](https://github.com/unique-chan)**

[A paper will be uploaded via Arxiv!]()

🚧 Under Construction!

## This repository contains:
- Classwise Cross Entropy (code) 
- For simplicity, classification code for visual recognition is provided separately in this [GitHub repo 🖱️](https://github.com/unique-chan/Simple-Image-Classification): you can easily use `Classwise Regulated Entropy` by passing `--loss_function='CRE'` for executing `train.py`. For details, please visit the above repository.

## Prerequisites
* See requirements.txt
```
torch
torchvision
```

## Code
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

## Contribution
If you find any bugs or have opinions for further improvements, please feel free to create a pull request or contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.

## Reference
1. Hao-Yun Chen, Pei-Hsin Wang, Chun-Hao Liu, Shih-Chieh Chang, Jia-Yu Pan, Yu-Ting Chen, Wei Wei, and Da-Cheng Juan. Complement objective training. arXiv preprint arXiv:1903.01182, 2019.
2. https://github.com/calmisential/Basic_CNNs_TensorFlow2
3. https://github.com/Hsuxu/Loss_ToolBox-PyTorch
4. https://github.com/weiaicunzai/pytorch-cifar100
