import cv2
import os
import glob
import numpy as np
import mmcv
import matplotlib 
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils import splits
from pyquaternion import Quaternion
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import obtain_map_info
from projects.mmdet3d_plugin.datasets.map_api import NuScenesMap
from PIL import Image
import torch

color_mapping = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
])/255
print('Using color mapping', color_mapping)

class AgentPrediction:
    """
    Agent data class, includes bbox, traj, and occflow
    """
    def __init__(self,
        pred_score,
        pred_label,
        pred_center,
        pred_dim,
        pred_yaw,
        pred_vel,
        pred_traj,
        pred_traj_score,
        pred_track_id=None,
        pred_occ_map=None,
        is_sdc=False,
        past_pred_traj=None,
        command=None,
        attn_mask=None,
    ):
        self.pred_score = pred_score
        self.pred_label = pred_label
        self.pred_center = pred_center
        self.pred_dim = pred_dim 
        self.pred_yaw = -pred_yaw-np.pi/2
        self.pred_vel = pred_vel
        self.pred_traj = pred_traj
        self.pred_traj_score = pred_traj_score
        self.pred_track_id = pred_track_id
        self.pred_occ_map = pred_occ_map
        if self.pred_traj is not None:
            if isinstance(self.pred_traj_score, int):
                self.pred_traj_max = self.pred_traj
            else:
                self.pred_traj_max = self.pred_traj[self.pred_traj_score.argmax()]
        else:
            self.pred_traj_max = None
        self.nusc_box = Box(
            center=pred_center,
            size=pred_dim,
            orientation=Quaternion(axis=[0, 0, 1], radians=self.pred_yaw),
            label=pred_label,
            score=pred_score
        )
        if is_sdc:
            self.pred_center = [0, 0, -1.2+1.56/2]
        self.is_sdc = is_sdc
        self.past_pred_traj = past_pred_traj
        self.command = command
        self.attn_mask = attn_mask

class Visualizer:
    """
    BaseRender class
    """
    def __init__(
        self,
        dataroot='/mnt/petrelfs/yangjiazhi/e2e_proj/data/nus_mini',
        version='v1.0-mini',
        predroot=None,
        with_occ_map=False,
        with_map=False,
        with_planning=False,
        render_gt_boxes=False,
        render_lidar=False,
        show_command=False,
        show_hd_map=False,
        show_sdc_car=False,
        show_sdc_traj=False,
        show_legend=False,
        with_pred_box=True,
        with_pred_traj=False,
        with_occ_map_time_seq=False,
        vis_attn_mask=False):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.predict_helper = PredictHelper(self.nusc)
        self.with_occ_map = with_occ_map
        self.with_map = with_map
        self.with_planning = with_planning
        self.render_lidar = render_lidar
        self.show_command = show_command
        self.show_hd_map = show_hd_map
        self.show_sdc_car = show_sdc_car
        self.show_sdc_traj = show_sdc_traj
        self.show_legend = show_legend
        self.with_occ_map_time_seq = with_occ_map_time_seq
        self.with_pred_traj = with_pred_traj
        self.with_pred_box = with_pred_box
        self.vis_attn_mask = vis_attn_mask
        self.veh_id_list = [0, 1, 2, 3, 4, 6, 7]
        self.use_json = '.json' in predroot
        self.token_set = set()
        if self.use_json:
            self.predictions = self._parse_predictions_json(predroot)
        else:
            # self.predictions = self._parse_predictions_pkl(predroot, layer_subfix=layer_subfix)
            self.predictions = self._parse_predictions_multitask_pkl(predroot)
        self.bev_render = BEVRender(render_gt_boxes=render_gt_boxes)
        # self.attn_render = BEVAttnRender(render_gt_boxes=render_gt_boxes)
        self.cam_render = CameraRender(render_gt_boxes=render_gt_boxes)
        
        if self.show_hd_map:
            self.nusc_maps = {
                'boston-seaport': NuScenesMap(dataroot=dataroot, map_name='boston-seaport'),
                'singapore-hollandvillage': NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage'),
                'singapore-onenorth': NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth'),
                'singapore-queenstown': NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown'),
            }

    def _parse_predictions_multitask_pkl(self, predroot):
        outputs = mmcv.load(predroot)
        outputs = outputs['bbox_results']
        prediction_dict = dict()
        for k in range(len(outputs)):
            token = outputs[k]['token']
            self.token_set.add(token)
            if self.show_sdc_traj:
                outputs[k]['boxes_3d'].tensor= torch.cat([outputs[k]['boxes_3d'].tensor,outputs[k]['sdc_boxes_3d'].tensor], dim=0)
                outputs[k]['scores_3d']= torch.cat([outputs[k]['scores_3d'],outputs[k]['sdc_scores_3d']], dim=0)
                outputs[k]['labels_3d']= torch.cat([outputs[k]['labels_3d'], torch.zeros((1,), device=outputs[k]['labels_3d'].device)], dim=0)
            # detection
            bboxes = outputs[k]['boxes_3d']
            scores = outputs[k]['scores_3d']
            labels = outputs[k]['labels_3d']

            track_scores = scores.cpu().detach().numpy()
            track_labels = labels.cpu().detach().numpy()
            track_boxes = bboxes.tensor.cpu().detach().numpy()

            track_centers = bboxes.gravity_center.cpu().detach().numpy()
            track_dims =  bboxes.dims.cpu().detach().numpy()
            track_yaw =  bboxes.yaw.cpu().detach().numpy()

            if 'track_ids' in outputs[k]:
                track_ids = outputs[k]['track_ids'].cpu().detach().numpy()
            else:
                track_ids = None

            # speed
            track_velocity = bboxes.tensor.cpu().detach().numpy()[:,-2:]
            
            # trajectories
            trajs = outputs[k][f'traj'].numpy()
            traj_scores = outputs[k][f'traj_scores'].numpy()
            
            predicted_agent_list = []

            # occflow
            if self.with_occ_map:
                if 'topk_query_ins_segs' in outputs[k]['occ']:
                    occ_map = outputs[k]['occ']['topk_query_ins_segs'][0].cpu().numpy()
                else:
                    occ_map = np.zeros((1, 5, 200, 200))
            else:
                occ_map = None

            occ_idx = 0
            for i in range(track_scores.shape[0]):
                if track_scores[i] < 0.25:
                    continue
                if occ_map is not None and track_labels[i] in self.veh_id_list:
                    occ_map_cur = occ_map[occ_idx, :, ::-1]
                    occ_idx += 1
                else:
                    occ_map_cur = None
                if track_ids is not None:
                    if i < len(track_ids):
                         track_id = track_ids[i]
                    else:
                        track_id = 0
                else:
                    track_id = None
                # if track_labels[i] not in [0, 1, 2, 3, 4, 6, 7]:
                #     continue
                predicted_agent_list.append(
                    AgentPrediction(
                        track_scores[i], 
                        track_labels[i],
                        track_centers[i], 
                        track_dims[i], 
                        track_yaw[i],
                        track_velocity[i],
                        trajs[i],
                        traj_scores[i],
                        pred_track_id=track_id,
                        pred_occ_map=occ_map_cur,
                        past_pred_traj=None
                        )
                    )
            
            if self.with_map:
                map_thres = 0.7
                score_list = outputs[k]['pts_bbox']['score_list'].cpu().numpy().transpose([1,2,0])
                predicted_map_seg = outputs[k]['pts_bbox']['lane_score'].cpu().numpy().transpose([1,2,0]) # H, W, C
                predicted_map_seg[..., -1] =  score_list[..., -1]
                predicted_map_seg = (predicted_map_seg > map_thres) * 1.0
                predicted_map_seg = predicted_map_seg[::-1, :, :]
            else:
                predicted_map_seg = None
            
            if self.with_planning:
                # detection
                bboxes = outputs[k]['sdc_boxes_3d']
                scores = outputs[k]['sdc_scores_3d']
                labels = 0

                track_scores = scores.cpu().detach().numpy()
                track_labels = labels
                track_boxes = bboxes.tensor.cpu().detach().numpy()

                track_centers = bboxes.gravity_center.cpu().detach().numpy()
                track_dims =  bboxes.dims.cpu().detach().numpy()
                track_yaw =  bboxes.yaw.cpu().detach().numpy()
                track_velocity = bboxes.tensor.cpu().detach().numpy()[:,-2:]
                
                outputs[k]['planning_traj'][0][:, 0] = -outputs[k]['planning_traj'][0][:, 0]
                
                if self.vis_attn_mask:
                    attn_mask = [outputs[k]['weight_list'][ii].reshape(200,200).cpu().detach().numpy() for ii in range(3)]
                    attn_mask = sum(attn_mask)[::-1]
                else:
                    attn_mask = None
                if self.show_command:
                    command = outputs[k]['command'][0].cpu().detach().numpy()
                else:
                    command = None
                planning_agent = AgentPrediction(
                            track_scores[0], 
                            track_labels,
                            track_centers[0], 
                            track_dims[0], 
                            track_yaw[0],
                            track_velocity[0],
                            outputs[k]['planning_traj'][0].cpu().detach().numpy(),
                            1,
                            pred_track_id=-1,
                            pred_occ_map=None,
                            past_pred_traj=None,
                            is_sdc=True,
                            command=command,
                            attn_mask=attn_mask
                            )
                predicted_agent_list.append(planning_agent)
            else:
                planning_agent = None
            prediction_dict[token] = dict(predicted_agent_list=predicted_agent_list,
                                        predicted_map_seg=predicted_map_seg,
                                        predicted_planning=planning_agent)
        return prediction_dict

    def _parse_predictions_json(self, predroot):
        outputs = mmcv.load(predroot)
        prediction_dict = dict()
        for key in outputs['results'].keys():

            predicted_agent_list = []
            for i in range(len(outputs['results'][key])):
                # detection
                det_scores = np.array(outputs['results'][key][i]['detection_score'])
                det_labels = np.array(outputs['results'][key][i]['detection_name'])
                det_boxes = np.array(outputs['results'][key][i]['size'])

                det_centers = np.array(outputs['results'][key][i]['translation'])
                det_dims = np.array(outputs['results'][key][i]['size'])
                det_yaw = np.array(outputs['results'][key][i]['rotation'])

                # speed
                det_velocity = np.array(outputs['results'][key][i]['velocity'])
                
                # trajectories
                trajs = np.array(outputs['results'][key][i]['predict_traj'])
                traj_scores = np.array(outputs['results'][key][i]['predict_traj_score'])
                if det_scores < 0.25:
                    continue
                predicted_agent_list.append(
                    AgentPrediction(
                        det_scores, 
                        0,
                        det_centers, 
                        det_dims, 
                        0,
                        det_velocity,
                        trajs,
                        traj_scores,
                        )
                    )
            prediction_dict[key] = dict(predicted_agent_list=predicted_agent_list)
        return prediction_dict

    def visualize_bev(self, sample_token, out_filename, t=None):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()
        
        if self.render_lidar:
            self.bev_render.render_lidar_data(sample_token, self.nusc)
        if self.bev_render.render_gt_boxes:
            self.bev_render.render_anno_data(sample_token, self.nusc, self.predict_helper)
        if self.with_pred_box:
            self.bev_render.render_pred_box_data(self.predictions[sample_token]['predicted_agent_list'])
        if self.with_pred_traj:
            self.bev_render.render_pred_traj(self.predictions[sample_token]['predicted_agent_list'])
        if self.with_map:
            self.bev_render.render_pred_map_data(self.predictions[sample_token]['predicted_map_seg'])
        
        if self.with_occ_map_time_seq:
            self.bev_render.render_occ_map_data_time(self.predictions[sample_token]['predicted_agent_list'], t)
        elif self.with_occ_map:
            self.bev_render.render_occ_map_data(self.predictions[sample_token]['predicted_agent_list'])
        else:
            pass
        if self.with_planning:
            self.bev_render.render_pred_box_data([self.predictions[sample_token]['predicted_planning']])
            self.bev_render.render_planning_data(self.predictions[sample_token]['predicted_planning'], show_command=self.show_command)
            if self.vis_attn_mask:
                self.bev_render.render_planning_attn_mask(self.predictions[sample_token]['predicted_planning'])
        if self.show_hd_map:
            self.bev_render.render_hd_map(self.nusc, self.nusc_maps, sample_token)
        if self.show_sdc_car:
            self.bev_render.render_sdc_car()
        if self.show_legend:
            self.bev_render.render_legend()
        self.bev_render.save_fig(out_filename + '.jpg')
    
    def visualize_cam(self, sample_token, out_filename):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)
        self.cam_render.render_pred_track_bbox(self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc)
        self.cam_render.render_pred_traj(self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc, render_sdc=self.with_planning)
        self.cam_render.save_fig(out_filename + '_cam.jpg')

    def combine(self, out_filename):
        # pass
        bev_image = cv2.imread(out_filename + '.jpg')
        cam_image = cv2.imread(out_filename + '_cam.jpg')
        if os.path.exists(out_filename + '_attn_mask.jpg'):
            attn_mask = cv2.imread(out_filename + '_attn_mask.jpg')
            merge_image = cv2.hconcat([cam_image, bev_image, attn_mask])
            os.remove(out_filename + '_attn_mask.jpg')
        else:
            merge_image = cv2.hconcat([cam_image, bev_image])
        cv2.imwrite(out_filename + '.jpg', merge_image)
        os.remove(out_filename + '_cam.jpg')
        
    def to_video(self, folder_path, out_path, fps=4, downsample=1):
        imgs_path = glob.glob(os.path.join(folder_path, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height//downsample), interpolation = cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


class BaseRender:
    """
    BaseRender class
    """
    def __init__(
        self,
        figsize=(10,10)):
        self.figsize = figsize
        self.fig, self.axes = None, None

    def reset_canvas(self, dx=1, dy=1, tight_layout=False):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(dx, dy, figsize=self.figsize)
        if tight_layout:
            plt.tight_layout()
    
    def close_canvas(self):
        plt.close()

    def save_fig(self,filename):
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        print(f'saving to {filename}')
        plt.savefig(filename)

class CameraRender(BaseRender):
    """
    Render class for Camera View
    """
    def __init__(self,
                figsize=(53.3333,20),
                render_gt_boxes=False):
        super(CameraRender, self).__init__(figsize)
        self.cams = [
            'CAM_FRONT_LEFT',
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
        ]
        self.render_gt_boxes = render_gt_boxes

    def project_to_cam(self,
                        agent_prediction_list,
                        sample_data_token,
                        nusc,
                        lidar_cs_record,
                        project_traj=False,
                        cam=None,
                        ):
        _, cs_record, pose_record, cam_intrinsic, imsize = self.get_image_info(sample_data_token, nusc)
        boxes = []
        # import pdb;pdb.set_trace()
        for agent in agent_prediction_list:
            box = Box(agent.pred_center, agent.pred_dim, Quaternion(axis=(0.0, 0.0, 1.0), radians=agent.pred_yaw),
                        name=agent.pred_label, token='predicted')
            box.is_sdc = agent.is_sdc
            if project_traj:
                box.pred_traj = np.zeros((agent.pred_traj_max.shape[0]+1 ,3))
                box.pred_traj[:, 0] = agent.pred_center[0]
                box.pred_traj[:, 1] = agent.pred_center[1]
                box.pred_traj[:, 2] = agent.pred_center[2] - agent.pred_dim[2]/2
                box.pred_traj[1:, :2] += agent.pred_traj_max[: ,:2]
                box.pred_traj = (Quaternion(lidar_cs_record['rotation']).rotation_matrix @ box.pred_traj.T).T
                box.pred_traj += np.array(lidar_cs_record['translation'])[None, :]
            box.rotate(Quaternion(lidar_cs_record['rotation']))
            box.translate(np.array(lidar_cs_record['translation']))
            boxes.append(box)
        # Make list of Box objects including coord system transforms.
        
        box_list = []
        tr_id_list = []
        for i, box in enumerate(boxes):
            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            if project_traj:
                box.pred_traj += -np.array(cs_record['translation'])[None, :]
                box.pred_traj = (Quaternion(cs_record['rotation']).inverse.rotation_matrix @ box.pred_traj.T).T
                
            tr_id = agent_prediction_list[i].pred_track_id
            if box.is_sdc and cam == 'CAM_FRONT':
                box_list.append(box)
            if not box_in_image(box, cam_intrinsic, imsize):
                continue
            box_list.append(box)
            tr_id_list.append(tr_id)
        return box_list, tr_id_list, cam_intrinsic, imsize
    
    def render_image_data(self, sample_token, nusc):
        sample = nusc.get('sample', sample_token)
        for i, cam in enumerate(self.cams):
            sample_data_token = sample['data'][cam]
            data_path, _, _, _, _ = self.get_image_info(sample_data_token, nusc)
            image = np.array(Image.open(data_path))
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 60)
            fontScale = 2
            color = (0, 0, 0)
            thickness = 4
            image = cv2.putText(image, cam, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            self.axes[i//3, i%3].imshow(image)
            plt.axis('off')
            self.axes[i//3, i%3].axis('off')
            self.axes[i//3, i%3].grid(False)
    
    def render_pred_track_bbox(self, predicted_agent_list, sample_token, nusc):
        sample = nusc.get('sample', sample_token)
        lidar_cs_record = nusc.get('calibrated_sensor', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        for i,cam in enumerate(self.cams):
            sample_data_token = sample['data'][cam]
            box_list, tr_id_list, camera_intrinsic, imsize = self.project_to_cam(predicted_agent_list, sample_data_token, nusc, lidar_cs_record)
            for j, box in enumerate(box_list):
                if box.is_sdc:
                    continue
                tr_id = tr_id_list[j]
                if tr_id is None:
                    tr_id = 0
                c = color_mapping[tr_id  % len(color_mapping)]
                box.render(self.axes[i//3, i%3], view=camera_intrinsic, normalize=True, colors=(c, c, c))
            # plot gt
            if self.render_gt_boxes:
                data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token, selected_anntokens=sample['anns'])
                for j, box in enumerate(boxes):
                    c = [0, 1, 0]
                    box.render(self.axes[i//3, i%3], view=camera_intrinsic, normalize=True, colors=(c, c, c))
            self.axes[i//3, i%3].set_xlim(0, imsize[0])
            self.axes[i//3, i%3].set_ylim(imsize[1], 0)

    def render_pred_traj(self, predicted_agent_list, sample_token, nusc, render_sdc=False, points_per_step=10):
        sample = nusc.get('sample', sample_token)
        lidar_cs_record = nusc.get('calibrated_sensor', nusc.get('sample_data', sample['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        for i, cam in enumerate(self.cams):
            sample_data_token = sample['data'][cam]
            box_list, tr_id_list, camera_intrinsic, imsize = self.project_to_cam(predicted_agent_list, sample_data_token, nusc, lidar_cs_record, project_traj=True, cam=cam)
            for j, box in enumerate(box_list):
                traj_points = box.pred_traj[:, :3]
                
                total_steps = (len(traj_points)-1) * points_per_step + 1
                total_xy = np.zeros((total_steps, 3))
                for k in range(total_steps-1):
                    unit_vec = traj_points[k//points_per_step+1] - traj_points[k//points_per_step]
                    total_xy[k] = (k/points_per_step - k//points_per_step) * unit_vec + traj_points[k//points_per_step]
                in_range_mask = total_xy[:, 2] > 0.1
                traj_points = view_points(total_xy.T, camera_intrinsic, normalize=True)[:2, :]
                traj_points = traj_points[:2, in_range_mask]
                if box.is_sdc:
                    if render_sdc:
                        self.axes[i//3, i%3].scatter(traj_points[0], traj_points[1], color=(1, 0.5, 0), s=150)
                    else:
                        continue
                else:
                    tr_id = tr_id_list[j]
                    if tr_id is None:
                        tr_id = 0
                    c = color_mapping[tr_id  % len(color_mapping)]
                    self.axes[i//3, i%3].scatter(traj_points[0], traj_points[1], color=c, s=15)
            self.axes[i//3, i%3].set_xlim(0, imsize[0])
            self.axes[i//3, i%3].set_ylim(imsize[1], 0)


    def get_image_info(self, sample_data_token, nusc):
        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        data_path = nusc.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None
        return data_path, cs_record, pose_record, cam_intrinsic, imsize



class BEVRender(BaseRender):
    """
    Render class for BEV
    """
    def __init__(self,
        figsize=(20,20), 
        margin: float = 50, 
        view: np.ndarray = np.eye(4),
        render_gt_boxes=False):
        super(BEVRender, self).__init__(figsize)
        self.margin = margin
        self.view = view
        self.render_gt_boxes = render_gt_boxes
    
    def set_plot_cfg(self):
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        # self.axes.axis('off') 
        self.axes.set_aspect('equal')
        self.axes.grid(False)

    def render_sample_data(self, canvas, sample_token):
        pass

    def render_anno_data(
            self, 
            sample_token, 
            nusc,
            predict_helper):
        sample_record = nusc.get('sample', sample_token)
        assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'
        lidar_record = sample_record['data']['LIDAR_TOP']
        data_path, boxes, _ = nusc.get_sample_data(lidar_record, selected_anntokens=sample_record['anns'])
        for box in boxes:
            instance_token = nusc.get('sample_annotation', box.token)['instance_token']
            future_xy_local = predict_helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
            if future_xy_local.shape[0] > 0:
                trans = box.center
                rot = Quaternion(matrix=box.rotation_matrix)
                future_xy = convert_local_coords_to_global(future_xy_local, trans, rot)
                future_xy = np.concatenate([trans[None, :2], future_xy], axis=0)
                c = np.array([0, 0.8, 0])
                box.render(self.axes, view=self.view, colors=(c, c, c))
                self._render_traj(future_xy, line_color=c, dot_color=(0, 0, 0))
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])

    def render_lidar_data(
            self, 
            sample_token,
            nusc):            
        sample_record = nusc.get('sample', sample_token)
        assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'
        lidar_record = sample_record['data']['LIDAR_TOP']
        data_path, boxes, _ = nusc.get_sample_data(lidar_record, selected_anntokens=sample_record['anns'])
        LidarPointCloud.from_file(data_path).render_height(self.axes, view=self.view)
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        self.axes.axis('off') 
        self.axes.set_aspect('equal')

    def render_pred_box_data(self, agent_prediction_list):
        for pred_agent in agent_prediction_list:
            c = np.array([0, 1, 0])
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None: # this is true
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id  % len(color_mapping)]
            pred_agent.nusc_box.render(axis=self.axes, view=self.view, colors=(c,c,c))
            if pred_agent.is_sdc:
                c = np.array([1, 0, 0])
                pred_agent.nusc_box.render(axis=self.axes, view=self.view, colors=(c,c,c))

    def render_pred_traj(self, agent_prediction_list, top_k=3):
        for pred_agent in agent_prediction_list:
            if pred_agent.is_sdc:
                continue
            sorted_ind = np.argsort(pred_agent.pred_traj_score)[::-1] # from high to low
            num_modes = len(sorted_ind)
            sorted_traj = pred_agent.pred_traj[sorted_ind, :, :2]
            sorted_score = pred_agent.pred_traj_score[sorted_ind]
            # norm_score = np.sum(np.exp(sorted_score))
            norm_score = np.exp(sorted_score[0])
            
            sorted_traj = np.concatenate([np.zeros((num_modes, 1, 2)), sorted_traj], axis=1)
            trans = pred_agent.pred_center
            rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi/2)
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if pred_agent.pred_label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25
            # print(sorted_score)
            for i in range(top_k-1, -1, -1):
                viz_traj = sorted_traj[i, :, :2]
                viz_traj = convert_local_coords_to_global(viz_traj, trans, rot)
                traj_score = np.exp(sorted_score[i])/norm_score
                # traj_score = [1.0, 0.01, 0.01, 0.01, 0.01, 0.01][i]
                self._render_traj(viz_traj, traj_score=traj_score, colormap='winter', dot_size=dot_size)
            

    def render_pred_map_data(self, predicted_map_seg):
        # rendered_map = map_color_dict
        map_color_dict = np.array([(204, 128, 0), (102, 255, 102), (102, 255, 102)]) # divider, crossing, contour
        rendered_map = map_color_dict[predicted_map_seg.argmax(-1).reshape(-1)].reshape(200,200, -1)
        bg_mask = predicted_map_seg.sum(-1) == 0
        rendered_map[bg_mask, :] = 255
        self.axes.imshow(rendered_map, alpha=0.6, interpolation='nearest', extent=(-51.2, 51.2, -51.2, 51.2))

    def render_occ_map_data(self, agent_list):
        rendered_map = np.ones((200, 200, 3))
        rendered_map_hsv = matplotlib.colors.rgb_to_hsv(rendered_map)
        occ_prob_map = np.zeros((200, 200))
        for i in range(len(agent_list)):
            pred_agent = agent_list[i]
            if pred_agent.pred_occ_map is None:
                continue
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None: # this is true
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id  % len(color_mapping)]
            pred_occ_map = pred_agent.pred_occ_map.max(0)
            update_mask = pred_occ_map > occ_prob_map
            occ_prob_map[update_mask] = pred_occ_map[update_mask]
            pred_occ_map *= update_mask
            hsv_c = matplotlib.colors.rgb_to_hsv(c)
            rendered_map_hsv[pred_occ_map> 0.1] = (np.ones((200,200,1)) * hsv_c)[pred_occ_map> 0.1]
            max_prob = pred_occ_map.max()
            renorm_pred_occ_map = (pred_occ_map - max_prob) * 0.7 + 1
            sat_map = (renorm_pred_occ_map * hsv_c[1])
            rendered_map_hsv[pred_occ_map> 0.1, 1] = sat_map[pred_occ_map> 0.1]
            rendered_map = matplotlib.colors.hsv_to_rgb(rendered_map_hsv)
        self.axes.imshow(rendered_map, alpha=0.8, interpolation='nearest', extent=(-50, 50, -50, 50))
        
    def render_occ_map_data_time(self, agent_list, t):
        rendered_map = np.ones((200, 200, 3))
        rendered_map_hsv = matplotlib.colors.rgb_to_hsv(rendered_map)
        occ_prob_map = np.zeros((200, 200))
        for i in range(len(agent_list)):
            pred_agent = agent_list[i]
            if pred_agent.pred_occ_map is None:
                continue
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None: # this is true
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id  % len(color_mapping)]
            pred_occ_map = pred_agent.pred_occ_map[t]
            update_mask = pred_occ_map > occ_prob_map
            occ_prob_map[update_mask] = pred_occ_map[update_mask]
            pred_occ_map *= update_mask
            hsv_c = matplotlib.colors.rgb_to_hsv(c)
            rendered_map_hsv[pred_occ_map> 0.1] = (np.ones((200,200,1)) * hsv_c)[pred_occ_map> 0.1]
            max_prob = pred_occ_map.max()
            renorm_pred_occ_map = (pred_occ_map - max_prob) * 0.7 + 1
            sat_map = (renorm_pred_occ_map * hsv_c[1])
            rendered_map_hsv[pred_occ_map> 0.1, 1] = sat_map[pred_occ_map> 0.1]
            rendered_map = matplotlib.colors.hsv_to_rgb(rendered_map_hsv)
        self.axes.imshow(rendered_map, alpha=0.8, interpolation='nearest', extent=(-50, 50, -50, 50))

    def render_planning_data(self, predicted_planning, show_command=False):
        planning_traj = predicted_planning.pred_traj
        planning_traj = np.concatenate([np.zeros((1,2)), planning_traj], axis=0)
        self._render_traj(planning_traj, colormap='autumn', dot_size=50)
        if show_command:
            self._render_command(predicted_planning.command)
            
    def render_planning_attn_mask(self, predicted_planning):
        planning_attn_mask = predicted_planning.attn_mask
        planning_attn_mask = planning_attn_mask/planning_attn_mask.max()
        cmap_name = 'plasma'
        self.axes.imshow(planning_attn_mask, alpha=0.8, interpolation='nearest', extent=(-51.2, 51.2, -51.2, 51.2), vmax=0.2, cmap=matplotlib.colormaps[cmap_name])

    def render_hd_map(self, nusc, nusc_maps, sample_token):
        # import pdb;pdb.set_trace()
        sample_record = nusc.get('sample', sample_token)
        sd_rec = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        info = {
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'scene_token': sample_record['scene_token']
        }

        layer_names = ['road_divider', 'road_segment', 'lane_divider', 'lane',  'road_divider', 'traffic_light', 'ped_crossing']
        map_mask = obtain_map_info(nusc,
                            nusc_maps,
                            info,
                            patch_size=(102.4, 102.4),
                            canvas_size=(1024, 1024),
                            layer_names=layer_names)
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = map_mask[:, ::-1] > 0
        map_show = np.ones((1024,1024,3))
        map_show[map_mask[0], :] =  np.array([1.00, 0.50, 0.31])
        map_show[map_mask[1], :] =  np.array([159./255., 0.0, 1.0])
        self.axes.imshow(map_show, alpha=0.2, interpolation='nearest', extent=(-51.2, 51.2, -51.2, 51.2))

    def _render_traj(self, future_traj, traj_score=1, colormap='winter', points_per_step=20, line_color=None, dot_color=None, dot_size=25):
        total_steps = (len(future_traj)-1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors*traj_score + (1-traj_score)*np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps-1):
            unit_vec = future_traj[i//points_per_step+1] - future_traj[i//points_per_step]
            total_xy[i] = (i/points_per_step - i//points_per_step) * unit_vec + future_traj[i//points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)
        
    def _render_command(self, command):
        command_dict = ['TURN RIGHT', 'TURN LEFT', 'KEEP FORWARD']
        self.axes.text(-48, -45, command_dict[int(command)], fontsize=45)

    def render_sdc_car(self):
        sdc_car_png = cv2.imread('figs/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))

    def render_legend(self):
        legend = cv2.imread('figs/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(23, 51.2, -50, -40))
        

if __name__ == '__main__':
    # predroot = '/mnt/nas37/yihan01.hu/models/e2e/bevformer_exp/det_query_e2e_ablation_1025/tracking_map_traj_v1/out_mini.pkl'
    # predroot = '/mnt/nas37/yihan01.hu/models/e2e/bevformer_exp/det_query_e2e_planning/tracking_map_traj_occflow_planning_ep3_sin_embed_w_sdc_learned_posembed/out_mini.pkl'
    # predroot = '/mnt/nas37/yihan01.hu/models/ablation_ckpts/11-2-SOTA-BASE-V1-MaskAttn-R101-5Frames/out.pkl'
    predroot = '/mnt/nas20/yihan01.hu/outs/out_vov_mini.pkl'
    out_folder = 'viz/out_vov_mini/'
    demo_video = 'mini_val_final.avi'
    
    vis_attn_mask = False
    if vis_attn_mask:
        vis_cam = False
        render_cfg = dict(
            with_occ_map=False,
            with_occ_map_time_seq=False,
            with_map=False,
            with_planning=True,
            render_gt_boxes=False,
            render_lidar=False,
            show_command=True,
            show_hd_map=True,
            show_sdc_car=True,
            show_legend=False,
            with_pred_box=True,
            with_pred_traj=False,
            show_sdc_traj=False,
            vis_attn_mask=True,
        )
        subfix = '_attn_mask'
    else:
        vis_cam = True
        render_cfg = dict(
            with_occ_map=True,
            with_occ_map_time_seq=False,
            with_map=True,
            with_planning=True,
            render_gt_boxes=False,
            render_lidar=False,
            show_command=True,
            show_hd_map=False,
            show_sdc_car=True,
            show_legend=True,
            with_pred_box=True,
            with_pred_traj=True,
            show_sdc_traj=False,
            vis_attn_mask=False,
        )
        subfix = ''
    viser = Visualizer(version='v1.0-mini', predroot=predroot, dataroot='data/nuscenes', **render_cfg)
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # # viser = Visualizer(version='v1.0-trainval', predroot='/home/users/yihan01.hu/workspace/BEVFormer/test/bevformer_base_traj/Fri_Jul_22_01_24_48_2022/pts_bbox/results_nusc.json')
    # val_splits = splits.val
    # train_splits = splits.mini_train
    # scene_token_to_name = dict()
    # for i in range(len(viser.nusc.scene)):
    #     scene_token_to_name[viser.nusc.scene[i]['token']] = viser.nusc.scene[i]['name']
    # for i in range(len(viser.nusc.sample)):
    #     sample_token = viser.nusc.sample[i]['token']
    #     scene_token = viser.nusc.sample[i]['scene_token'] 
        
    #     if scene_token_to_name[scene_token] not in val_splits:
    #         continue
    #     if sample_token not in viser.token_set:
    #         print(i, sample_token, 'not in prediction pkl!')
    #         continue
    #     print(i, sample_token)
    #     if render_cfg['with_occ_map_time_seq']:
    #         for t in range(5):
    #             viser.visualize_bev(sample_token, out_folder + str(i).zfill(3) + f'_{t}', t)
    #     else:
    #         viser.visualize_bev(sample_token, out_folder + str(i).zfill(3) + subfix)
    #     if vis_cam:
    #         viser.visualize_cam(sample_token, out_folder + str(i).zfill(3))
    #         viser.combine(out_folder + str(i).zfill(3))
    viser.to_video(out_folder, demo_video, fps=4, downsample=2)