# Self-Regularized-Entropy
Robust Image Classification via Class-wise Softmax Suppression (PyTorch)

[Core Idea]
* Notation
    - <img src="https://render.githubusercontent.com/render/math?math=K">: Total number of classes
    - <img src="https://render.githubusercontent.com/render/math?math=N">: Total number of samples
    - <img src="https://render.githubusercontent.com/render/math?math=g">: Index of the ground-truth class
    - <img src="https://render.githubusercontent.com/render/math?math=\hat{y}">: Predicted probability vectors
    - <img src="https://render.githubusercontent.com/render/math?math=y">: Label vectors (One-hot encoded)
    - <img src="https://render.githubusercontent.com/render/math?math=(i)">: Index of the <img src="https://render.githubusercontent.com/render/math?math=i">th vector
    - <img src="https://render.githubusercontent.com/render/math?math=[j]">: Index of the probability mass on class <img src="https://render.githubusercontent.com/render/math?math=j">


* Cross Entropy
    - <img src="https://render.githubusercontent.com/render/math?math=H(y, \hat{y})=-\frac{1}{N}\sum_{i=1}^{N}log(\hat{y}^{(i)[g]})">

* Self-Regularized Entropy
    - <img src="https://render.githubusercontent.com/render/math?math=S(\hat{y})=-\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1, j \ne g}^{N}R(\hat{y}^{(i)[j]})">
    - <img src="https://render.githubusercontent.com/render/math?math=R(\hat{y}^{(i)[j]})=\frac{c(\hat{y}^{(i)[j]})}{c(\hat{y}^{(i)[j]})%2B\hat{y}^{(i)[j]}}log\frac{c(\hat{y}^{(i)[j]})}{c(\hat{y}^{(i)[j]}%2B\hat{y}^{(i)[j]}})%2B\frac{\hat{y}^{(i)[j]}}{c(\hat{y}^{(i)[j]})%2B\hat{y}^{(i)[j]}}log\frac{\hat{y}^{(i)[j]}}{c(\hat{y}^{(i)[j]})%2B\hat{y}^{(i)[j]}}">
    where <img src="https://render.githubusercontent.com/render/math?math=c(\hat{y}^{(i)[j]})"> denotes <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{K-1}\hat{y}^{(i)[j]}">.

* Proposed Training Loss Function
    - <img src="https://render.githubusercontent.com/render/math?math=H(y,\hat{y})-S(\hat{y})">
    