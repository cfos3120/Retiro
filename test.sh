#!/bin/bash
echo 'Hello World'

# Core
epochs=1000
batch=8
file_path="/project/MLFluids/steady_cavity_case_b200_maxU100ms_simple_normalized.npy"
sub_x=4
seed=42
multi_gpu=1 #(1/0 : True/False)

    a=(Train Monitor)  # Type of hybrid calculation
    b=(0 1)            # Initialize Weights
    c=(0 1)            # Dynamic Loss Balancing
    d=(0 100 200)      # Key only batches
    e=(0 1)            # Secondary Optimizer for Key only batches


# Some of these tests do not matter, but we just want to see if the code returns any errors
for i in "${a[@]}"; do
    for j in "${b[@]}"; do
        for k in "${c[@]}"; do
            for l in "${d[@]}"; do
                for m in "${e[@]}"; do 
                    echo "python3 main.py --test > log_test_a${i}_b${j}_c${k}_d${l}_e${m}";
                done
            done
        done
    done
done