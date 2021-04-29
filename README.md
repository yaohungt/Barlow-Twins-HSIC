# Barlow Twins and HSIC

> Unofficial Pytorch implementation for Barlow Twins and HSIC_SSL on small datasets (CIFAR10, STL10, and Tiny ImageNet).

Correspondence to: 
  - Yao-Hung Hubert Tsai (yaohungt@cs.cmu.edu)

## Technical Report
[**A Note on Connecting Barlow Twins with Negative-Samples-Free Contrastive Learning**](https://arxiv.org/pdf/2104.13712.pdf)<br>
[Yao-Hung Hubert Tsai](https://yaohungt.github.io), [Shaojie Bai](https://jerrybai1995.github.io), [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)<br>

I hope this work will be useful for your research :smiling_face_with_three_hearts: 

## Usage

### Disclaimer
A large portion of the code is from [this repo](https://github.com/leftthomas/SimCLR), which is a great resource for academic development. Note that we do not perform extensive hyper-parameters grid search and hence you may expect a performance boost after tuning some hyper-parameters (e.g., the learning rate).

The official implementation of Barlow Twins can be found [here](https://github.com/facebookresearch/barlowtwins). We have also tried the HSIC_SSL in this official repo and we find similar performance (we tried on ImageNet-1K and CIFAR10) between HSIC_SSL and Barlow Twins' method. 

### Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- thop
```
pip install thop
```

### Supported Dataset
`CIFAR10`, `STL10`, and [`Tiny_ImageNet`](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4).


### Train and Linear Evaluation using Barlow Twins 
```
python main.py --lmbda 0.0078125 --corr_zero --batch_size 128 --feature_dim 128 --dataset cifar10
python linear.py --dataset cifar10 --model_path results/0.0078125_128_128_cifar10_model.pth
```
### Train and Linear Evaluation using HSIC
```
python main.py --lmbda 0.0078125 --corr_neg_one --batch_size 128 --feature_dim 128 --dataset cifar10
python linear.py --dataset cifar10 --model_path results/neg_corr_0.0078125_128_128_cifar10_model.pth
```
