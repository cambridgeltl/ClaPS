#!/bin/bash
num_shots=16
task_name=sst2
model_name=google/flan-t5-base
description=claps
reward_type=cross_entropy
prune_type=reward
template_id=0
percentile=90
seed=42
method=greedy

python run_prune_search.py \
    --model_name $model_name \
    --dataset_name $task_name \
    --num_shots $num_shots \
    --method $method \
    --seed $seed \
    --reprune_vocab True \
    --prune_type $prune_type \
    --percentile $percentile \
    --template_id $template_id \
    --reward_type $reward_type \
    --save_path results/$task_name/$model_name/shots-$num_shots/$description-$percentile/$seed \
    --dict_path ./results/$task_name/$model_name/shots-$num_shots/$description-$reward_type-dict.json \
    --vocab_path ./vocabs/google-flan-t5-base-kmeans-vocab.json    