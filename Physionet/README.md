# Attentive ODEs for Irregularly-Sampled Time Series


## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

```
conda env create -f attentive_ode.yml
```


## Experiments on different datasets

By default, the dataset are downloadeded and processed when script is run for the first time. 

Raw datasets: 

[[Physionet]](https://physionet.org/physiobank/database/challenge/2012/)



### Running Attentive ODE 

* Attentive ODE 

```
sh test.sh

```

or 

```
python3 run_models.py  --niters 20 -n 8000 -l 20 --dataset physionet --attentive-ode --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.016 --classif

```

## Latent_ODE
- DATA : Physionet

|Time Series Latent(Seed)|AUC Score|
|------|---|
|1991 (91194)|0.8470|
|2022 (93669)|0.8430|
|2021 (53879)|0.8496|
|2020 (6152)|0.8454|
|2019 (51159)|0.8470|

Test MSE on PhysioNet.
Encoder-decoder models. 
|Model|Interp(x 10^-3)|Extrap(x 10^-3)|
|------|---|---|
|RNN-VAE|5.930±0.249|3.055±0.145|
|LatentODE (RNN enc.)|3.907±0.252|3.162±0.052|
|LatentODE (ODE enc.)|2.118±0.271|2.231±0.029|
|LatentODE + Poisson|2.789±0.771|2.208±0.050|
|ACE-NODE(ODE enc.)|--|2.045±0.039|


