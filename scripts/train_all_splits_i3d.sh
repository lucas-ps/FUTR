#!/bin/bash

# Loop for 50salads dataset splits
for split_num in $(seq 1 5)
do
   echo "Training 50salads on split ${split_num}"
   ./scripts/50s_train.sh ${split_num}
done

# Loop for Breakfast dataset splits
for split_num in $(seq 1 4)
do
   echo "Training Breakfast on split ${split_num}"
   ./scripts/bf_train.sh ${split_num}
done

