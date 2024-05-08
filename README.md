# UniMatch for road segmentation

## data and model download instruction

Dirty road extraction from GF-2 images by semi-supervised deep learning method for arid and semiarid regions of southern Mongolia The experimental setup is forked from UniMatch(https://github.com/LiheYoung/UniMatch).

Due to the limitation of upload data and model size, we store the training data set and the trained model in Baidu online disk, which is linked as https://pan.baidu.com/s/1vu2lD-qfuXxYij8r-MOSSg (Extraction code:u9rc). For training, please directly extract the three files from the zip file into your home directory. Then follow the training instruction in this instruction file.

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


