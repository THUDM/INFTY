#!/bin/bash

source activate infty

cd PILOT

opt=(pcgrad)
method=(icarl)


for method in ${method[@]}; do
  for opt in ${opt[@]}; do
    python main.py --inftyopt $opt --config exps/${method}.json
    done
done

