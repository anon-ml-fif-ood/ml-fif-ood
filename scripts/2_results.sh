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

            python -m benchmark.main \
                -d $method_name \
                -nn $model_name  \
                -outs $out_dataset_names \
                -bs 10000 \
                -agg CLASS_FIF \
                -fns features.transition1.pool features.transition2.pool features.transition3.pool flatten res_layer_1 res_layer_2 res_layer_3 layer_12 layer_13 layer_14 layer_15 layer_16 layer_17 layer_18 layer_19 layer_20 layer_21 layer_22 layer_23 encoder_output \
                --nseeds 1
                
        done
    done
done