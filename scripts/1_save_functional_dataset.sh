#!/bin/bash
model=( \
    "DENSENET121" \
    "BITSR101" \
    "VIT16L" \
)
in_dataset_names=( \
    "ILSVRC2012" \
)
out_dataset_names="MOS_INATURALIST MOS_PLACES MOS_SUN TEXTURES"
for model in ${model[*]}
do
    for in_dataset_name in ${in_dataset_names[*]}
    do
        model_name=${model}_${in_dataset_name}

        for method_name in "mahalanobis_mono_class" 
        do

            python -m benchmark.save_functional_dataset \
                -d $method_name \
                -nn $model_name  \
                -outs $out_dataset_names -bs 1000 -r none
           
        done
    done
done
