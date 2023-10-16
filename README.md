# Attentive Co-Evolving ODE :: ACE-NODE

This study aims to predict more accurately by adding the attention technique to the neural ode.

<p align="center">
  <img align="middle" src="./assets/attentive.png" alt="neural_ODE" width="500" height="350" /> 
</p>
Neural ordinary differential equations (NODEs) presented a new paradigm to construct (continuous-time) neural networks. While showing several good characteristics in terms of the number of parameters and the flexibility in constructing neural networks, they also have a couple of well-known limitations: i) theoretically NODEs learn homeomorphic mapping functions only, and ii) sometimes NODEs show numerical instability in solving integral problems. To handle this, many enhancements have been proposed. To our knowledge, however, integrating attention into NODEs has been overlooked for a while. To this end, we present a novel method of attentive dual co-evolving NODE (ACE-NODE): one main NODE for a downstream machine learning task and the other for providing attention to the main NODE. Our ACE-NODE supports both pairwise and elementwise attention. In our experiments, our method outperforms existing NODE-based and non-NODE-based baselines in almost all cases by non-trivial margins.

# Our Baseline 

## Neural ODE
Those `examples` directory contains cleaned up code regarding the usage of adaptive ODE solvers in machine learning. The scripts in this directory assume that `torchdiffeq` is installed following instructions from the main directory.
MNIST
```
python3 atten_ode_mnist.py 
```
CIFAR10
```
python3 atten_ode_cifar10.py 
```
SVHN
```
python3 atten_ode_svhn.py 
```
## GRU-ODE-Baeyes
USHCN-daily
```
python3 climate_gruode_atten.py --lr 0.05 --weight_decay 0.001 --nepochs 200
```
If you want to know more details about USHCN-daily, go to USHCN-daily folder 

## Latent_ODE 
Physionet
```
python3 run_models.py  --niters 20 -n 8000 -l 20 --dataset physionet --attentive-ode --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.016 --classif

```
If you want to know more details about Physionet, go to Physionet folder 

