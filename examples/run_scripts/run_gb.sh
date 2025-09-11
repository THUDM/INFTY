#!/bin/bash

source activate infty

cd PILOT

opt=(zo_sgd zo_sgd_sign zo_sgd_conserve zo_adam zo_adam_sign zo_adam_conserve forward_grad)
method=(ease)


for method in ${method[@]}; do
  for opt in ${opt[@]}; do
    python main.py --inftyopt $opt --config exps/${method}.json
    done
done

