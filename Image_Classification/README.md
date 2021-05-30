# AttentiveODE


# Our Baseline 

Those `examples` directory contains cleaned up code regarding the usage of adaptive ODE solvers in machine learning. The scripts in this directory assume that `torchdiffeq` is installed following instructions from the main directory.

## Image Classification

### MNIST
|MNIST|Test Error|# Params|
|------|---|---|
|ResNet|0.41%(99.59)|0.60M|
|RK-Net|0.47%(99.53)|0.22M|
|Neural ODE(in Augmented ODE)|0.42%(99.58)|0.22M| 
|Neural ODE(we did)|0.34%(99.66)|0.22M| 
|Attentive ODE|*0.28%(99.72)*|0.28M|

#### MNIST Test Error(Final)
|MNIST(REAL)|Test Error|# Params|
|------|---|---|
|Attentive ODE(seed = 2022)|0.32(99.68)|0.28M|
|Attentive ODE(seed = 2021)|0.36(99.64)|0.28M|
|Attentive ODE(seed = 2020)|0.34(99.66)|0.28M|
|Attentive ODE(seed = 2019)|0.29(99.71)|0.28M|
|Attentive ODE|*0.28%(99.72)*|0.28M|

 
### CIFAR10
|CIFAR-10|Test Error|# Params|
|------|---|---|
|ResNet|13.47%(86.53)|0.57M|
|RK-Net|--|--|
|Neural ODE(in Augemented ODE)|46.3%(53.7)|0.20M|
|Neural ODE(we did)|14.8%(85.2)|0.20M|
|Attentive ODE|11.73%(88.27)|1.4M|
|Attentive ODE|11.88%(88.12)|1.2M|


#### CIFAR10 Test Error(Final)
|seed|Attentive ODE(1.27M)|Attentive ODE(0.36M)|Neural ODE|ResNet|RK-Net|
|----|---|---|---|---|---|
|2022|12.33(87.67)|13.86(86.14)|14.96(85.04)|---|15.5(84.50)|
|2021|11.88(88.12)|14.07(85.93)|15.07(84.93)|13.33(86.67)|15.01(84.99)|
|2020|12.30(87.70)|14.57(85.43)|13.62(86.38)|---|14.96(85.04)|
|2019|12.38(87.62)|14.42(85.58)|15.07(84.93)|---|15.14(84.86)|
|2018|12.50(87.50)|13.96(86.04)|14.53(85.47)|---|15.36(84.64)|


### SVHN
|SVHN|Test Error|# Params|
|------|---|---|
|ResNet|3.4%(96.6)|0.60M|
|RK-Net|3.48%(96.52)|0.22M|
|Neural ODE(in agumented ODE)|18.94%(81.06)|0.22M|
|Neural ODE(we did)|4.01%(95.99)|0.22M|
|Attentive ODE|4.06%(95.94)|0.32M|
|Attentive ODE|3.86%(96.14)|0.32M|


#### SVHN Test Error(Final)
|seed|Attentive ODE|Neural ODE|ResNet|RK-Net|
|----|---|---|---|---|
|2022|4.21(95.79)|4.57(95.43)|4.17(95.83)|9.58(90.42)|
|2021|3.86(96.14)|4.28(95.72)|4.27(95.73)|8.79(91.21)|
|2020|4.3(95.70)|4.59(95.41)|4.28(95.72)|9.55(90.45)|
|2019|4.08(95.92)|4.37(95.63)|4.2(95.80)|7.86(92.14)|
|2018|4.49(95.51)|4.51(95.49)|4.17(95.83)|9.2(90.80)|


