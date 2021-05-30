for seed in 2022 2021 2020 2019 2018
do
    for regul in l1
    do
        
        for lam in 1e-3 
        do
            
            for lr in 0.05
            do
                for tol in 1e-5
                do
                    for adjoint in True 
                    do

                        python3 -u atten_ode_svhn.py --seed $seed --regul $regul --lam $lam --lr $lr --tol $tol --adjoint $adjoint --save ./svhn_final/seed_{$seed}_svhn_{$lr}_{$tol}_{$adjoint}_{$lam}_{$regul}
                    done
                done
            done
        done
    done 
done