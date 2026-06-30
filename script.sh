#!/bin/bash





resnet50='cfgs/resnet50_baseline.yaml'
resnet50_sd='cfgs/resnet50_sd.yaml'
resnet50_augmix='cfgs/resnet50_augmix.yaml'
resnet50_augmix_sd='cfgs/resnet50_augmix+sd.yaml'


torchrun --nproc_per_node=2 --master_port='12348' train.py --yaml $resnet50
# torchrun --nproc_per_node=2 --master_port='12348' train.py --yaml $resnet50_sd

# torchrun --nproc_per_node=2 --master_port='12348' train.py --yaml $resnet50_augmix
# torchrun --nproc_per_node=2 --master_port='12348' train.py --yaml $resnet50_augmix_sd