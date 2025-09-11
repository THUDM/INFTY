#!/bin/bash

source activate infty

cd PILOT

opt=(sam gsam looksam gam c_flat c_flat_plus)
method=(memo_scr)


for method in ${method[@]}; do
  for opt in ${opt[@]}; do
    python main.py --inftyopt $opt --config exps/${method}.json
    done
done

