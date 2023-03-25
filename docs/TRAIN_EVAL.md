# Prerequisites

**Please make sure you have prepared the environment and the nuScenes dataset.**

##  Train

Train UniAD with 8 GPUs 
```
./tools/uniad_dist_train.sh ./projects/configs/track_map/base_stage1.py 8
# For slurm users:
# ./tools/uniad_slurm_train.sh YOUR_PARTITION ./projects/configs/track_map/base_stage1.py 8
```

## Eval
Eval UniAD with 8 GPUs
```
./tools/uni_dist_eval.sh ./projects/configs/track_map/base_stage1.py ./path/to/ckpts.pth 8
# For slurm users:
# ./tools/uniad_slurm_eval.sh YOUR_PARTITION ./projects/configs/track_map/base_stage1.py ./path/to/ckpts.pth 8
```

