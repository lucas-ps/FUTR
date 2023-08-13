#!/bin/bash

# Loop for 50salads dataset splits
for split_num in $(seq 1 5)
do
   echo "Testing 50salads on split ${split_num}"
   ./scripts/50s_predict.sh ${split_num}
done
for split_num in $(seq 1 4)
do
   echo "Testing Breakfast on split ${split_num}"
   ./scripts/50s_predict_tsn.sh ${split_num}
done

# Loop for Breakfast dataset splits
for split_num in $(seq 1 4)
do
   echo "Testing Breakfast on split ${split_num}"
   ./scripts/bf_predict.sh ${split_num}
done
for split_num in $(seq 1 4)
do
   echo "Testing Breakfast on split ${split_num}"
   ./scripts/bf_predict_tsn.sh ${split_num}
done

