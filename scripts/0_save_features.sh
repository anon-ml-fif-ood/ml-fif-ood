#!/bin/bash
source .env.example
source .env

in_dataset_names=(
    "ILSVRC2012" \ 
)
out_dataset_names=( \
    "ILSVRC2012" \ 
    "MOS_SUN" \
    "MOS_INATURALIST" \
    "MOS_PLACES" \
    "TEXTURES" \
)

for in_dataset_name in ${in_dataset_names[*]}
do

    model="DENSENET121"
    model_name=${model}_${in_dataset_name}
    for out_dataset_name in ${out_dataset_names[*]}
    do
    python -m features.save_dn_features \
        --model $model_name  \
        --dataset $out_dataset_name
    done
    
    python -m features.save_dn_features \
        --model $model_name  \
        --dataset $in_dataset_name --train
    
    model="BITSR101"
    model_name=${model}_${in_dataset_name}
    for out_dataset_name in ${out_dataset_names[*]}
    do
    python -m features.save_bit_features \
        --model $model_name  \
        --dataset $out_dataset_name
    done
    
    python -m features.save_bit_features \
        --model $model_name  \
        --dataset $in_dataset_name --train
    
    model="VIT16L"
    model_name=${model}_${in_dataset_name}
    for out_dataset_name in ${out_dataset_names[*]}
    do
    python -m features.save_vit_features \
        --model $model_name  \
        --dataset $out_dataset_name
    done
    
    python -m features.save_vit_features \
        --model $model_name  \
        --dataset $in_dataset_name --train

done