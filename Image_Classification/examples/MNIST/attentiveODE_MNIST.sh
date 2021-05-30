

for regul in l2 
do
    for lam in 1e-4
    do
        for lr in 0.01
        do
            for tol in 1e-4
            do
                for adjoint in True 
                do
                    for seed in 2021
                    do

                        python3 -u atten_ode_mnist3.py --nepochs 150 --regul $regul --lam $lam --lr $lr --tol $tol --adjoint $adjoint --seed $seed --save ./Attentive_mnist/seed_{$seed}_mnist_{$lr}_{$tol}_{$adjoint}_{$lam}_{$regul}
                    done
                done
            done
        done
    done
done 