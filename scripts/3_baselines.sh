#!/bin/bash
model=( \
    "VIT16L" \
    "BITSR101" \
    "DENSENET121" \
)
in_dataset_names=( \
    "ILSVRC2012" \
)
method_names=( \
    "odin" \
    "energy" \
    "gradnorm" \
)
out_dataset_names="MOS_SUN MOS_PLACES MOS_INATURALIST TEXTURES"
for model in ${model[*]}
do
    for in_dataset_name in ${in_dataset_names[*]}
    do
        model_name=${model}_${in_dataset_name}
        for method_name in ${method_names[*]}
        do
            for out_dataset_name in ${out_dataset_names[*]}
            do
                for temperature in 1 1000
                do

                    python -m benchmark.baselines \
                        --score $method_name \
                        -nn $model_name  \
                        -out $out_dataset_name \
                        -t $temperature \
                        -eps 0 \
                        --batch-size 10

                done
            done
        done
    done
done