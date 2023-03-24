import torch
file_path = '/mnt/nas37/yihan01.hu/models/ablation_ckpts/bevformer_r101_dcn_24ep.pth'
model = torch.load(file_path, map_location='cpu')
all = 0
# ['planning_head','motion_head', 'occ_head','seg_head']
for key in list(model['state_dict'].keys()):
    # if 'planning_head' in key or 'occ_head' in key or 'motion_head' in key or 'seg_head' in key:
    #     continue
    all += model['state_dict'][key].nelement()
print(all/1024/1024)
import pdb;pdb.set_trace()

# smaller 63374123
# v4 69140395
