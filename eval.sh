#! /bin/bash

for dtype in bf16 fp32; do
    for model in original lcm; do
        for vae in original light; do
            mkdir data/comlops/v2/model-${model}-vae-${vae}-dtype-${dtype}
            for step in 2 4 10 25 50 100; do
                out_dir=data/comlops/v2/model-${model}-vae-${vae}-dtype-${dtype}/n-${step}
                mkdir -p ${out_dir}
                python eval.py \
                    data/comlops/v2/images/ \
                    data/comlops/v2/depth/ \
                    ${out_dir} \
                    -n ${step} \
                    --model ${model} \
                    --vae ${vae} \
                    --precision ${dtype}
            done
        done
    done
done
