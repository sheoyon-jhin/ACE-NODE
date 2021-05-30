
for seed in 2021
do
    for network in resnet odenet
    do    
        python3 odenet_mnist.py --seed $seed --network $network --save ./mnist_org_5case/seed_{$seed}_network_{$network}_mnist --lr 0.01 --tol 1e-4 --adjoint True
    done
done
