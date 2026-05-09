#!/bin/bash

python train.py  -s datasets/cambridge/GreatCourt -m outputs/cambridge/GreatCourt -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu --sample_kpts --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001  --cfg  configs/ulfloc_cambridge.yaml

python train.py  -s datasets/cambridge/KingsCollege -m outputs/cambridge/KingsCollege -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu --sample_kpts --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001  --cfg  configs/ulfloc_cambridge.yaml

python train.py  -s datasets/cambridge/OldHospital -m outputs/cambridge/OldHospital -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu --sample_kpts --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001   --cfg  configs/ulfloc_cambridge.yaml

python train.py  -s datasets/cambridge/ShopFacade -m outputs/cambridge/ShopFacade -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu --sample_kpts --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001  --cfg  configs/ulfloc_cambridge.yaml

python train.py  -s datasets/cambridge/StMarysChurch -m outputs/cambridge/StMarysChurch -r 1  -f sp -g 3dgs --iterations 30000 --data_device cpu --sample_kpts --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001  --cfg  configs/ulfloc_cambridge.yaml
