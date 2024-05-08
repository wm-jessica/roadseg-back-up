# UniMatch for road segmentation


## Results

Comparison utilizing exclusively labeled data.

| Method                      | MeanIou | back_iou | road_iou |
| :-------------------------: | :-------: | :-------: | :-------: |
| Xception(sup)               | 84.1      | 99.12     | 69.07      | 
| Resnet101(sup)              | 85.48      | 99.19      | 71.78     | 
| Resnet101(unimatch)          |    86.21   |  99.2    |  73.22 |


Comparison with various unlabeled data.

| Method                      | MeanIou | back_iou | road_iou |
| :-------------------------: | :-------: | :-------: | :-------: |
| Resnet101(unimatch)          |    86.21   |  99.2    |  73.22 |
| Resnet101(unimatch)          |  86.37          | 99.24     | 73.51      | 

The checkpoints are situated within the 'experiments' folder.


## Getting Started

### Installation

```bash
cd UniMatch
conda create -n unimatch python=3.10.4
conda activate unimatch
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone

```
├── ./pretrained
    ├── resnet101.pth
    └── xception.pth
```

### Dataset


``` 
├── data
    ├── roadseg_semi_new2
        └── train
        └── val
        └── txts
    
```

## Usage

### UniMatch

```bash
bash scripts/train_new2_uni_res_1k.sh 4 12360
```



### Supervised Baseline
```bash

bash scripts/train_new2_sup_res_b8.sh 4 12360
bash scripts/train_new2_sup_xcep_b8.sh 4 12360

```


