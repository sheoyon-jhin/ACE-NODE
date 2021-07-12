
for regul in l2 
do
    for lam in 1e-3
    do
        for lr in 0.01
        do
            for tol in 1e-5  
            do
                for adjoint in False
                do
                    for seed in 2021
                    do
                        CUDA_VISIBLE_DEVICES=0 python3 -u atten_ode_cifar10.py --seed $seed --nepochs 150 --regul $regul --lam $lam --lr $lr --tol $tol --adjoint $adjoint --save ./Cifar10_L2_final/seed_{$seed}_cifar10_param_2_{$lr}_{$tol}_{$adjoint}_{$lam}_{$regul}
                    done
                done
            done
        done
    done
done 



#python3 attentiveODE_CIFAR10_TRAIN.py --seed 2021 --nepochs 150 --regul l2 --lam 1e-3 --lr 0.01 --tol 1e-5 --adjoint False 