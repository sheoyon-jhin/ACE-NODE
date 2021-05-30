for seed in 1991 2019 2020 2021 2022
do
    CUDA_VISIBLE_DEVICES=0 python3 run_models.py --random-seed $seed --niters 20 -n 8000 -l 20 --dataset physionet --attentive-ode --rec-dims 40 --rec-layers 3 --gen-layers 3 --units 50 --gru-units 50 --quantization 0.016 --classif >/home/bigdyl/latent_ode/test2.csv
done