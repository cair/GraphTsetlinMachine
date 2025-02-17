#!/bin/bash

echo `date`, Setup the environment ...
set -e  # exit if error

models="graph_tm tm_classifier graph_nn"
dataset_noise_ratios="0.005 0.01 0.02 0.05 0.1 0.2"
num_iterations=10  # Number of times to repeat the experiments

for (( i=1; i<=num_iterations; i++ ))
do
    echo "Iteration $i of $num_iterations"

    for N in $dataset_noise_ratios; do
        echo `date`, Running Graph NN ...
        python3 graph_nn.py --dataset_noise_ratio $N

        echo `date`, Running Graph Tsetlin Machine ...
        python3 graph_tm.py --dataset_noise_ratio $N
        
        echo `date`, Running Tsetlin Machine Classifier ...
        python3 tm_classifier.py --dataset_noise_ratio $N
    done
done


