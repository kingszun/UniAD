import os
import json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

if __name__ == "__main__":
    root = '/home/senyao.du/E2EFormer/data/nuscenes'
    version = 'v1.0-trainval'
    work_dir = '/home/senyao.du/E2EFormer/data'

    nusc = NuScenes(version=version, dataroot=root, verbose=True)
    if version == 'v1.0-trainval':
        scenes = splits.val
    elif version == 'v1.0-test':
        scenes = splits.test
    else:
        raise ValueError("unknown")

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name']
        if scene_name not in scenes:
            continue

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True
        else:
            frame['first'] = False
        frames.append(frame)

    del nusc

    res_dir = os.path.join(work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)