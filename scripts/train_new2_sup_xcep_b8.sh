#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits

dataset='roadseg_semi_new2'  # 使用的数据集
method='supervised'
backbone='xception'
batch_size=8

config=configs/${dataset}_${backbone}_${batch_size}.yaml
labeled_id_path=splits/$dataset/labeled.txt
unlabeled_id_path=splits/$dataset/unlabeled.txt
save_path=expr/$dataset/$method/${backbone}_${batch_size}

mkdir -p $save_path

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
