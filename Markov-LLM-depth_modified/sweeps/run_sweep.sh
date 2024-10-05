#!/bin/bash

# Define the possible values for each parameter
chain_values=("icl")
order_values=(2 4 6)
n_layer_values=(2 3)
n_head_values=(1)
n_embd_values=(16 32)
batch_size_values=(16)
sequence_length_values=(512 128)
iterations_values=(10000)
seed_values=(1 2 3)

# Define the wandb project name
wandb_project="markov-LLM-depth-heads-attention-icl"

# Initialize iteration count
iteration_count=0

# Loop over all combinations of parameters
for chain in "${chain_values[@]}"; do
  for order in "${order_values[@]}"; do
    for n_layer in "${n_layer_values[@]}"; do
      for n_head in "${n_head_values[@]}"; do
        for n_embd in "${n_embd_values[@]}"; do
          for batch_size in "${batch_size_values[@]}"; do
            for sequence_length in "${sequence_length_values[@]}"; do
              for iterations in "${iterations_values[@]}"; do
                for seed in "${seed_values[@]}"; do
                  # Compute the device index (modulus 8)
                  device_index=$((iteration_count % 8))
                  
                  # Construct the argument string
                  args="--chain $chain --order $order --n_layer $n_layer --n_head $n_head --n_embd $n_embd --batch_size $batch_size --sequence_length $sequence_length --iterations $iterations --seed $seed --wandb_project $wandb_project --device cuda:$device_index"
                  
                  # Run the command
                  echo "Running: python src/main.py $args"
                  python src/main.py $args &
                  
                  # Increment iteration count
                  iteration_count=$((iteration_count + 1))
                  
                  # Wait for 30 seconds between runs
                  echo "Waiting for 30 seconds before the next run..."
                  sleep 30 &
                done
              done
            done
          done
        done
      done
    done
  done
done
