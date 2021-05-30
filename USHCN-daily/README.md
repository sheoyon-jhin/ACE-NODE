# GRU-ODE-Bayes

### Requirements

The code uses Python3 and Pytorch as auto-differentiation package. The following python packages are required and will be automatically downloaded when installing the gru_ode_bayes package:

```
numpy
pandas
sklearn
torch
tensorflow (for logging)
tqdm
argparse
```

### Procedure

Install the main package :

```
pip install -e . 
```
And also the ODE numerical integration package : 
```
cd torchdiffeq
pip install -e .
```
## Run experiments
Experiments folder contains different cases study for the GRU-ODE-Bayes. Each trained model is then stored in the `trained_models` folder.


### USHCN daily (climate) data
For retraining the model, go to Climate folder and run 
```
python3 climate_gruode_atten.py --lr 0.05 --weight_decay 0.001 --nepochs 300
```

## Acknowledgements and References

The torchdiffeq package has been extended from the original version proposed by (Chen et al. 2018)

Chen et al. Neural ordinary differential equations, NeurIPS, 2018.

For climate dataset : 

Menne et al., Long-Term Daily Climate Records from Stations Across the Contiguous United States

## *attentive* GRU-ODE-BAYES 
|USHCN-Daily|MSE|NEGLL|
|------|---|---|
|NeuralODE-VAE|0.96|1.46|
|NeuralODE-VAE-MASK|0.83|1.36|
|Sequential VAE|0.83|1.37| 
|GRU-Simple|0.75|1.23|
|GRU-D|0.53|0.99|
|T-LSTM|0.59|1.67|
|GRU-ODE-Bayes|0.43|0.84|
|Attentive-GRU-ODE-Bayes|0.343|0.89|
