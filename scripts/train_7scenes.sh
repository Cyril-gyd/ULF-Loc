python train.py -s datasets/7scenes/7scenes_reference_models/chess  -m outputs/7scenes/chess --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../chess --cfg  configs/ulfloc_7scenes.yaml

python train.py -s datasets/7scenes/7scenes_reference_models/heads -m outputs/7scenes/heads --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../heads --cfg  configs/ulfloc_7scenes.yaml

python train.py -s datasets/7scenes/7scenes_reference_models/fire -m outputs/7scenes/fire --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../fire --cfg  configs/ulfloc_7scenes.yaml

python train.py -s datasets/7scenes/7scenes_reference_models/office/ -m outputs/7scenes/office --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../office --cfg  configs/ulfloc_7scenes.yaml

python train.py -s datasets/7scenes/7scenes_reference_models/redkitchen/ -m outputs/7scenes/redkitchen --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../redkitchen --cfg  configs/ulfloc_7scenes.yaml

python train.py -s datasets/7scenes/7scenes_reference_models/pumpkin/ -m outputs/7scenes/pumpkin --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../pumpkin --cfg  configs/ulfloc_7scenes.yaml

python train.py -s datasets/7scenes/7scenes_reference_models/stairs/ -m outputs/7scenes/stairs --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../stairs --cfg  configs/ulfloc_7scenes.yaml

