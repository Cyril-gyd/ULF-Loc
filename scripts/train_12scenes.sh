python train.py -s datasets/12scenes/12scenes_reference_models/apt1/kitchen  -m outputs/12scenes/apt1/kitchen --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../apt1/kitchen  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/apt1/living  -m outputs/12scenes/apt1/living --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../apt1/living  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/apt2/bed  -m outputs/12scenes/apt2/bed --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../apt2/bed  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/apt2/kitchen  -m outputs/12scenes/apt2/kitchen --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../apt2/kitchen --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/apt2/living  -m outputs/12scenes/apt2/living --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../apt2/living --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/apt2/luke  -m outputs/12scenes/apt2/luke --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../apt2/luke  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/office1/gates362  -m outputs/12scenes/office1/gates362 --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../office1/gates362  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/office1/gates381  -m outputs/12scenes/office1/gates381 --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../office1/gates381  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/office1/lounge  -m outputs/12scenes/office1/lounge --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../office1/lounge  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/office1/manolis  -m outputs/12scenes/office1/manolis --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../office1/manolis  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/office2/5a  -m outputs/12scenes/office2/5a --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../office2/5a  --cfg  configs/ulfloc_12scenes.yaml

python train.py -s datasets/12scenes/12scenes_reference_models/office2/5b  -m outputs/12scenes/office2/5b --iterations 30000 --data_device cpu -f sp -g 3dgs --sample_kpts --images  ../../../office2/5b  --cfg  configs/ulfloc_12scenes.yaml

