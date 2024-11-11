import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from torch import torch
from PIL import Image
from multipledispatch import dispatch

class UniADDataManagerMetadata:
    """
    UniAD 데이터의 metadata를 관리하는 클래스
        - metadata
            - total_frames: [0 : 2^31)
            - creation_time: "YYYY-MM-DD hh:mm:ss"
            - source: ""
            - target: ""
            - frame_rate: ""
            - units: ""
            - vehicle_info: ""
            - sensor_info: ""
    """

    def __init__(self) -> None:
        """
        관리하는 모든 data에 대한 명시적인 선언
        """
        self.metadata = {
            'total_frames': 0,
            'creation_time': "",
            'source': "",
            'target': "",
            'frame_rate': "",
            'units': "",
            'vehicle_info': "",
            'sensor_info': ""
        }

    def set_metadata(self, key: str, value: Any) -> None:
        """
        지정된 key의 메타데이터 값을 설정한다.
        """

    @dispatch(Any)
    def get_metadata(self, key: str) -> Any:
        """
        해당 key의 메타데이터 값을 반환한다.
        """

    @dispatch()
    def get_metadata(self) -> Dict[str, Any]:
        """
        모든 메타데이터를 반환한다.
        """

    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """
        메타데이터 값을 업데이트한다.
        """

    @dispatch(Any)
    def remove_metadata(self, key: str) -> None:
        """
        지정된 key의 메타데이터를 삭제한다.
        """

    @dispatch
    def remove_metadata(self) -> None:
        """
        모든 메타데이터를 삭제한다.
        """

    @dispatch(Any)
    def validate_metadata(self, key: str) -> bool:
        """
        해당 key의 메타데이터 필드가 올바른 형식과 범위를 가지고 있는지 검증한다.
        """

    @dispatch
    def validate_metadata(self, key: str) -> bool:
        """
        모든 메타데이터 필드가 올바른 형식과 범위를 가지고 있는지 검증한다.
        """


class UniADDataManagerCamera:
    """
    UniAD 학습에 필요한 camera data 를 관리하는 클래스
    이미지는 PIL 라이브러리와 numpy 라이브러리를 사용하여 관리한다.
        - camera
            - wd_cmr_fr: uint8, (3, 1536, 1920)
                - R: [0 : 255] 
                - G: [0 : 255] 
                - B: [0 : 255]
            - wd_cmr_lh: uint8, (3, 1536, 1920)
                - R: [0 : 255] 
                - G: [0 : 255] 
                - B: [0 : 255]
            - wd_cmr_rh: uint8, (3, 1536, 1920)
                - R: [0 : 255] 
                - G: [0 : 255] 
                - B: [0 : 255]
            - wd_cmr_rr: uint8, (3, 1536, 1920)
                - R: [0 : 255] 
                - G: [0 : 255] 
                - B: [0 : 255]
    """
    def __init__(self):
        """
        관리하는 data에 대한 명시적인 선언
        image 는 numpy 와 PIL.Image.Image 모두 갖고있는다.
        """
        self.camera: dict = {}
        self.camera_pil: dict = {}
        self.metadata: dict = {}

    def load_PILimages(self, camera_name: str, path_to_image: str) -> None:
        """
        path_to_image 를 camera_name 의 PIL.Image.Image 로 불러온다.
        """

    @dispatch(str)
    def add_image(self, camera_name: str) -> None:
        """
        자정된 camera_name 의 PIL.Image.Image 를 (H, W, C) 에서 (C, H ,W) 로 트랜스포즈하여 np.ndarray 로 변환 하여 caemra dict 에 추가한다. 
        """

    @dispatch()
    def add_image(self) -> None:
        """
        모든 camera_name 의 PIL.Image.Image 를 (H, W, C) 에서 (C, H ,W) 로 트랜스포즈하여 np.ndarray 로 변환 하여 caemra dict 에 추가한다. 
        """

    @dispatch(str)
    def remove_images(self, camera_name: str) -> None:
        """
        입력받은 camera_name 의 데이터를 삭제한다.
        """

    @dispatch()
    def remove_images(self) -> None:
        """
        모든 camera 의 데이티러를 삭제한다.
        """

    @dispatch(str)
    def get_images(self, camera_name: str) -> np.ndarray:
        """
        입력받은 camera_name 의 카메라 데이터를 반환 한다.       
        """

    @dispatch()
    def get_images(self) -> Dict[str, np.ndarray]:
        """
        모든 camera_name 의 카메라 데이터를 반환 한다.       
        """

    @dispatch(str)
    def get_pilimages(self, camera_name: str) -> Image.Image:
        """
        camera_name 의 PIL.Image.Image를 반환 한다.
        """

    @dispatch()
    def get_pilimages(self) -> Dict[str, Image.Image]:
        """
        모든 camera_name 의 PIL.Image.Image 데이터를 반환 한다.
        """

    def get_registered_cameras(self) -> List[str]:
        """
        self.camera 에 등록된 모든 camera_name 의 목록을 반환 한다.
        """

    @dispatch(str)
    def pil_to_numpy(self, camera_name: str) -> np.ndarray:
        """
        지정된 camera_name 의 PIL.Image.Image 를 np.array로 변환 한다
        """

    @dispatch()
    def pil_to_numpy(self, camera_name: str) -> Dict[str, np.ndarray]:
        """
        모든 camera_name 의 PIL.Image.Image 를 np.array로 변환 한다
        """

    @dispatch(str)
    def numpy_to_pil(self, camera_name: str) -> Image.Image:
        """
        지정된 camera_name 의 np.array 를 PIL.Image.Image로 변환 한다
        """

    @dispatch()
    def numpy_to_pil(self) -> Dict[str, Image.Image]:
        """
        모든 camera_name 의 np.array 를 PIL.Image.Image로 변환 한다
        """

    def to_tensor(self) -> Dict[str, torch.Tensor]:
        """
        image 데이터를 PyTorch 텐서로 변환 한다
        """

          

class UniADDataManagerInput(UniADDataManagerCamera):
    """
    UniAD 학습에 필요한 Input data를 관리하는 클래스
        - input
            - camera
                - wd_cmr_fr: uint8, (3, 1536, 1920)
                - wd_cmr_lh: uint8, (3, 1536, 1920)
                - wd_cmr_rh: uint8, (3, 1536, 1920)
                - wd_cmr_rr: uint8, (3, 1536, 1920)
    """

    def __init__(self):
        """
        UniADDataManagerCamera를 상속받아 초기화하고, 
        input 데이터 관리를 위한 추가적인 속성을 초기화한다.
        """
        super().__init__()
        self.input: dict = {}
        self.input_metadata = {}

    def add_camera(self):
        """
        input 에 UniADDataManagerCamera.camera 를 추가한다.
        """

    @dispatch(str)
    def remove_input(self, input_name: str) -> None:
        """
        입력받은 input_name 의 데이터 를 삭제한다.
        """

    @dispatch()
    def remove_input(self) -> None:
        """
        모든 input_name 의 데이터 를 삭제한다.
        """

    @dispatch(str)
    def get_input(self, input_name: str) -> Dict:
        """
        입력받은 input_name 의 데이터를 반환 한다.       
        """

    @dispatch()
    def get_input(self) -> Dict:
        """
        모든 input_name 의 데이터를 반환 한다.       
        """

    def get_registered_cameras(self) -> List[str]:
        """
        self.input 에 등록된 모든 input_name 의 목록을 반환 한다.
        """

    def to_tensor(self) -> Dict:
        """
        모든 input 을 PyTorch 텐서로 변환 한다
        """



class UniADDataManagerBboxes_3d:
    """
    UniAD 학습에 필요한 bboxes_3d data를 관리하는 클래스
        - bboxes_3d: (count_of_agents, 11)
            - count_of_agents: [0 : 255] 
                - center_x: [-200.0 : 200.0] 
                - center_y: [-200.0 : 200.0] 
                - center_z: [-200.0 : 200.0] 
                - width: [0.0 : 10.0] 
                - length: [0.0 : 20.0] 
                - height: [0.0 : 10.0] 
                - yaw: [-180.0 : 180.0] 
                - velocity_x: [-200.0 : 200.0] 
                - velocity_y: [-200.0 : 200.0]
                - occlusion: [0, 0.25, 0.5, 0.75] 
                - count_of_lidar_points: [0 : 2^31)
    """

    def __init__(self):
        """
        bboxes_3d 데이터와 관련 메타데이터를 초기화한다.
        """
        self.dtype = np.dtype([
            ('center_x', np.float32), ('center_y', np.float32), ('center_z', np.float32),
            ('width', np.float32), ('length', np.float32), ('height', np.float32),
            ('yaw', np.float32), ('velocity_x', np.float32), ('velocity_y', np.float32),
            ('occlusion', np.float32), ('count_of_lidar_points', np.int32)
        ])
        self.count_of_agents = 0
        self.agent_bbox_3d = np.zeros(11, dtype=self.dtype)
        self.bboxes_3d = np.empty(0, dtype=self.dtype)
        self.metadata = {}

    def add_agent_bbox_3d(self, agent_bbox_3d) -> None:
        """
        self.bboxes_3d 에 agent_bbox_3d 데이터를 추가한다.
        self.count_of_agents 를 1 증가한다.
        """

    @dispatch(int, str)
    def get_bboxes_3d(self, number_of_agent, agent_bbox_3d_features: str) -> np.ndarray:
        """
        self.bboxes_3d[number_of_agnet][agent_bbox_3d_features] 의 값을 반환한다.
        """

    @dispatch(int)
    def get_bboxes_3d(self, number_of_agent) -> np.ndarray:
        """
        self.bboxes_3d[number_of_agnet] 의 값을 반환한다.
        """

    @dispatch()
    def get_bboxes_3d(self) -> np.ndarray:
        """
        self.bboxes_3d 의 값을 반환한다.
        """

    @dispatch(int)
    def remove_bboxes_3d(self, number_of_agent: int) -> None:
        """
        self.bboxes_3d[number_of_agnet] 를 제거한다.
        """

    @dispatch()
    def remove_bboxes_3d(self, number_of_agent: int) -> None:
        """
        self.bboxes_3d 를 제거한다.
        """

    @dispatch(int, str, Any)
    def update_agent_bbox_3d(self, number_of_agent: int, agent_bbox_3d_features: str, data: Any) -> None:
        """
        self.bboxes_3d[number_of_agnet][agent_bbox_3d_features] 의 agent_bbox_3d 정보를 업데이트한다.
        """

    @dispatch(int, np.ndarray)
    def update_agent_bbox_3d(self, number_of_agent: int, agent_bbox_3d : np.ndarray) -> None:
        """
        self.bboxes_3d[number_of_agnet] 의 agent_bbox_3d 정보를 업데이트한다.
        """

    def get_count_of_agents(self) -> int:
        """
        bboxes_3d 의 agent 갯수 를 반환한다.
        """

    def to_tensor(self) -> Dict[str, torch.Tensor]:
        """
        bboxes_3d 데이터를 PyTorch 텐서로 변환 한다
        """

class UniADDataManagerLabels_3d:
    """
    UniAD 학습에 필요한 labels_3d data를 관리하는 클래스
        - labels_3d: uint8, (count_of_agents, 1)
            - count_of_agents: [0 : 255] 
                - [0 : 255]
    """

    def __init__(self):
        """
        labels_3d 데이터와 관련 메타데이터를 초기화한다.
        """
        self.dtype = np.dtype([
            ('label', np.uint8)
        ])
        self.count_of_agents = 0
        self.labels_3d = np.array([], dtype=np.uint8)
        self.metadata = {}

    def add_agent_bbox_3d(self, label) -> None:
        """
        self.labels_3d 에 label 데이터를 추가한다.
        self.count_of_agents 를 1 증가한다.
        """

    @dispatch(int, str)
    def get_labels_3d(self, number_of_agent: int, agent_labels_3d_features: str) -> np.ndarray:
        """
        self.labels_3d[number_of_agnet][agent_labels_3d_features] 의 값을 반환한다.
        """

    @dispatch(np.ndarray)
    def get_labels_3d(self, number_of_agent) -> np.ndarray:
        """
        self.labels_3d[number_of_agnet] 의 값을 반환한다.
        """

    @dispatch()
    def get_labels_3d(self) -> np.ndarray:
        """
        self.labels_3d 의 값을 반환한다.
        """

    @dispatch(int)
    def remove_labels_3d(self, number_of_agent: int) -> None:
        """
        self.labels_3d[number_of_agnet] 를 제거한다.
        """

    @dispatch()
    def remove_labels_3d(self, number_of_agent: int) -> None:
        """
        self.labels_3d 를 제거한다.
        """

    @dispatch(int, str, Any)
    def update_agent_bbox_3d(self, number_of_agent: int, agent_labels_3d_features: str, agent_labels_3d_features_data: Any) -> None:
        """
        self.labels_3d[number_of_agnet][agent_labels_3d_features] 의 정보를 업데이트한다.
        """

    @dispatch(int, np.ndarray)
    def update_agent_bbox_3d(self, number_of_agent: int, label : np.ndarray) -> None:
        """
        self.labels_3d[number_of_agnet] 의 label 정보를 업데이트한다.
        """

    def get_count_of_agents(self) -> int:
        """
        labels_3d 의 agnet 갯수 를 반환한다.
        """

    def to_tensor(self) -> Dict[str, torch.Tensor]:
        """
        labels_3d 데이터를 PyTorch 텐서로 변환 한다
        """



class UniADDataManagerPast_traj:
    def __init__(self, count_of_past_horizons: int, count_of_future_horizons: int) -> None:
        """
        past_traj 데이터와 관련 메타데이터를 초기화한다.
            - past_traj: float32, (count_of_agents, count_of_past_horizons + count_of_future_horizons, 2)
                - count_of_agents: [0 : 255] 
                    - count_of_past_horizons: [0 : 255] 
                    - count_of_future_horizons: [0 : 255] 
                    - count_of_total_horizons : [0 : 255]
                        - x: [-200.0 : 200.0] 
                        - y: [-200.0 : 200.0]
        """
        self.count_of_agents = 0
        self.count_of_past_horizons = count_of_past_horizons
        self.count_of_future_horizons = count_of_future_horizons
        self.count_of_totalhorizons = count_of_past_horizons + count_of_future_horizons
        self.dtype_coordinate = np.dtype([('x', np.float32), ('y', np.float32)])
        self.dtype = np.dtype([
            ('number_of_totalhorizon', self.dtype_coordinate, (self.count_of_totalhorizons, ))
        ])
        self.past_trajdict: dict = {}
        self.past_traj = np.empty(0, dtype=self.dtype)

    def add_agent_past_traj(self, agent_past_traj) -> None:
        """
        self.past_trajdict 에 agent_past_traj 데이터를 추가한다.
        self.count_of_agents 를 1 증가한다.
        """

    @dispatch(int, int)
    def get_bboxes_3d(self, number_of_agent: int, number_of_totalhorizon: int) -> np.ndarray:
        """
        self.past_trajdict[number_of_agnet][number_of_totalhorizon] 의 값을 반환한다.
        """

    @dispatch(int)
    def get_bboxes_3d(self, number_of_agent) -> np.ndarray:
        """
        self.past_trajdict[number_of_agnet] 의 값을 반환한다.
        """

    @dispatch()
    def get_bboxes_3d(self) -> dict:
        """
        self.past_trajdict 의 값을 반환한다.
        """


    @dispatch(int)
    def remove_bboxes_3d(self, number_of_agent: int) -> None:
        """
        self.past_trajdict[number_of_agnet] 를 제거한다.
        """

    @dispatch()
    def remove_bboxes_3d(self, number_of_agent: int) -> None:
        """
        self.past_trajdict 를 제거한다.
        """

    @dispatch(int, int, str, Any)
    def update_agent_past_traj(self, number_of_agent: int, number_of_totalhorizon: int, agent_past_traj_features: str, agent_past_traj_features_data : Any) -> None:
        """
        self.past_trajdict[number_of_agnet][number_of_totalhorizon][agent_past_traj_features] 의 agent_past_traj 정보를 업데이트한다.
        """

    @dispatch(int, str,)
    def update_agent_past_traj(self, number_of_agent: int, agent_past_traj_features: str, agent_past_traj_features_data : Any) -> None:
        """
        self.past_trajdict[number_of_agnet][agent_past_traj_features] 의 agent_past_traj 정보를 업데이트한다.
        """

    def get_count_of_agents(self) -> int:
        """
        past_trajdict 의 agnet 갯수 를 반환한다.
        """

    def to_tensor(self) -> Dict[str, torch.Tensor]:
        """
        past_trajdict 데이터를 PyTorch 텐서로 변환 한다
        """

        
class UniADDataManagerSdc_label:
    def __init__(self):
        """
        sdc_label 데이터를 초기화한다.
        """
        self.sdc_label = np.array([1], dtype=np.uint8)

    def get_sdc_label(self) -> np.ndarray:
        """
        현재 sdc_label 데이터를 반환한다.

        Returns:
            np.ndarray: sdc_label 데이터 (항상 [1])
        """
        return self.sdc_label

    def validate_data(self) -> bool:
        """
        현재 sdc_label 데이터가 유효한지 검증한다.

        Returns:
            bool: 데이터가 유효하면 True, 그렇지 않으면 False
        """
        return self.sdc_label.shape == (1,) and self.sdc_label[0] == 1

    def reset(self) -> None:
        """
        sdc_label 데이터를 기본값으로 재설정한다.
        """
        self.sdc_label = np.array([1], dtype=np.uint8)

    def to_tensor(self) -> 'torch.Tensor':
        """
        sdc_label 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            torch.Tensor: PyTorch 텐서로 변환된 sdc_label 데이터
        """
        return torch.tensor(self.sdc_label, dtype=torch.uint8)

    def from_tensor(self, tensor: 'torch.Tensor') -> None:
        """
        PyTorch 텐서를 sdc_label 데이터로 변환하여 설정한다.

        Args:
            tensor (torch.Tensor): 변환할 PyTorch 텐서
        """
        self.sdc_label = tensor.numpy().astype(np.uint8)

    def save_data(self, file_path: str) -> None:
        """
        현재 sdc_label 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """
        np.save(file_path, self.sdc_label)

    def load_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 sdc_label 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """
        loaded_data = np.load(file_path)
        if loaded_data.shape == (1,) and loaded_data[0] == 1:
            self.sdc_label = loaded_data
        else:
            raise ValueError("Invalid sdc_label data loaded")

    def get_info(self) -> str:
        """
        sdc_label 데이터에 대한 정보를 문자열로 반환한다.

        Returns:
            str: sdc_label 데이터 정보
        """
        return f"SDC Label: {self.sdc_label[0]}, Shape: {self.sdc_label.shape}, Dtype: {self.sdc_label.dtype}"

    def is_sdc(self) -> bool:
        """
        현재 데이터가 SDC(Self-Driving Car)를 나타내는지 확인한다.

        Returns:
            bool: SDC를 나타내면 True, 그렇지 않으면 False
        """
        return bool(self.sdc_label[0])

    def visualize(self, output_path: Optional[str] = None) -> None:
        """
        sdc_label 데이터를 시각화한다. (이 경우 단순히 콘솔에 출력)

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로 (이 메서드에서는 사용되지 않음)
        """
        print(f"SDC Label Visualization: {'SDC' if self.is_sdc() else 'Not SDC'}")


class UniADDataManagerMulti_object_tracking(UniADDataManagerLabels_3d, UniADDataManagerPast_traj, UniADDataManagerPast_traj_mask, UniADDataManagerSdc_label):
    def __init__(self):
        """
        멀티 오브젝트 트래킹 데이터와 관련 메타데이터를 초기화한다.
        """
        super().__init__()
        self.count_of_agents = 0
        self.count_of_past_horizons = 0
        self.count_of_future_horizons = 0

    def load_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 멀티 오브젝트 트래킹 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """

    def save_data(self, file_path: str) -> None:
        """
        현재 멀티 오브젝트 트래킹 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """

    def set_multi_object_tracking_data(self, bboxes_3d: np.ndarray, labels_3d: np.ndarray, 
                                       past_traj: np.ndarray, past_traj_mask: np.ndarray, 
                                       sdc_label: np.ndarray) -> None:
        """
        멀티 오브젝트 트래킹 데이터를 설정한다.

        Args:
            bboxes_3d (np.ndarray): 3D 바운딩 박스 데이터
            labels_3d (np.ndarray): 3D 레이블 데이터
            past_traj (np.ndarray): 과거 궤적 데이터
            past_traj_mask (np.ndarray): 과거 궤적 마스크 데이터
            sdc_label (np.ndarray): SDC 레이블 데이터
        """

    def get_multi_object_tracking_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        현재 멀티 오브젝트 트래킹 데이터를 반환한다.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (bboxes_3d, labels_3d, past_traj, past_traj_mask, sdc_label)
        """

    def add_agent(self, bbox_3d: np.ndarray, label_3d: np.ndarray, 
                  past_traj: np.ndarray, past_traj_mask: np.ndarray) -> None:
        """
        새로운 에이전트의 데이터를 추가한다.

        Args:
            bbox_3d (np.ndarray): 에이전트의 3D 바운딩 박스
            label_3d (np.ndarray): 에이전트의 3D 레이블
            past_traj (np.ndarray): 에이전트의 과거 궤적
            past_traj_mask (np.ndarray): 에이전트의 과거 궤적 마스크
        """

    def update_agent(self, agent_index: int, bbox_3d: Optional[np.ndarray] = None, 
                     label_3d: Optional[np.ndarray] = None, past_traj: Optional[np.ndarray] = None, 
                     past_traj_mask: Optional[np.ndarray] = None) -> None:
        """
        특정 에이전트의 데이터를 업데이트한다.

        Args:
            agent_index (int): 업데이트할 에이전트의 인덱스
            bbox_3d (np.ndarray, optional): 새로운 3D 바운딩 박스
            label_3d (np.ndarray, optional): 새로운 3D 레이블
            past_traj (np.ndarray, optional): 새로운 과거 궤적
            past_traj_mask (np.ndarray, optional): 새로운 과거 궤적 마스크
        """

    def remove_agent(self, agent_index: int) -> None:
        """
        특정 에이전트의 데이터를 제거한다.

        Args:
            agent_index (int): 제거할 에이전트의 인덱스
        """

    def get_agent_data(self, agent_index: int) -> Dict[str, np.ndarray]:
        """
        특정 에이전트의 모든 데이터를 반환한다.

        Args:
            agent_index (int): 조회할 에이전트의 인덱스

        Returns:
            Dict[str, np.ndarray]: 에이전트의 데이터 (bbox_3d, label_3d, past_traj, past_traj_mask)
        """

    def set_horizons(self, past_horizons: int, future_horizons: int) -> None:
        """
        과거 및 미래 시간 범위를 설정한다.

        Args:
            past_horizons (int): 과거 시간 범위
            future_horizons (int): 미래 시간 범위
        """

    def get_horizons(self) -> Tuple[int, int]:
        """
        현재 설정된 과거 및 미래 시간 범위를 반환한다.

        Returns:
            Tuple[int, int]: (과거 시간 범위, 미래 시간 범위)
        """

    def validate_data(self) -> bool:
        """
        현재 멀티 오브젝트 트래킹 데이터가 유효한지 검증한다.

        Returns:
            bool: 데이터가 유효하면 True, 그렇지 않으면 False
        """

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        멀티 오브젝트 트래킹 데이터의 통계 정보를 계산하여 반환한다.

        Returns:
            Dict[str, Dict[str, float]]: 각 데이터 유형별 통계 정보
        """

    def to_tensor(self) -> Dict[str, 'torch.Tensor']:
        """
        멀티 오브젝트 트래킹 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            Dict[str, torch.Tensor]: PyTorch 텐서로 변환된 데이터
        """

    def from_tensor(self, tensors: Dict[str, 'torch.Tensor']) -> None:
        """
        PyTorch 텐서를 멀티 오브젝트 트래킹 데이터로 변환하여 설정한다.

        Args:
            tensors (Dict[str, torch.Tensor]): 변환할 PyTorch 텐서들
        """

    def visualize_data(self, output_path: Optional[str] = None) -> None:
        """
        멀티 오브젝트 트래킹 데이터를 시각화한다.

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로
        """

    def filter_agents(self, condition: callable) -> List[int]:
        """
        주어진 조건에 맞는 에이전트들의 인덱스를 반환한다.

        Args:
            condition (callable): 필터링 조건 함수

        Returns:
            List[int]: 조건을 만족하는 에이전트들의 인덱스 리스트
        """

    def get_sdc_agent_index(self) -> int:
        """
        SDC(Self-Driving Car) 에이전트의 인덱스를 반환한다.

        Returns:
            int: SDC 에이전트의 인덱스
        """

    def interpolate_trajectories(self) -> None:
        """
        누락된 궤적 데이터를 보간한다.
        """

    def align_data(self) -> None:
        """
        모든 데이터의 시간 정렬을 수행한다.
        """

    def apply_coordinate_transform(self, transform_matrix: np.ndarray) -> None:
        """
        모든 공간 데이터에 좌표 변환을 적용한다.

        Args:
            transform_matrix (np.ndarray): 변환 행렬
        """

    def get_data_slice(self, start_time: int, end_time: int) -> Dict[str, np.ndarray]:
        """
        지정된 시간 범위의 데이터 슬라이스를 반환한다.

        Args:
            start_time (int): 시작 시간
            end_time (int): 종료 시간

        Returns:
            Dict[str, np.ndarray]: 시간 범위에 해당하는 데이터 슬라이스
        """

    def merge_data(self, other_data: 'UniADDataManagerMulti_object_tracking') -> None:
        """
        다른 UniADDataManagerMulti_object_tracking 객체의 데이터를 현재 객체와 병합한다.

        Args:
            other_data (UniADDataManagerMulti_object_tracking): 병합할 다른 데이터 관리자 객체
        """
        
class UniADDataManagerLane_masks:
    def __init__(self):
        """
        lane_masks 데이터와 관련 메타데이터를 초기화한다.
        """
        self.lane_masks = np.array([], dtype=np.uint8)
        self.count_of_instances = 0
        self.height_of_bev = 0
        self.width_of_bev = 0

    def load_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 lane_masks 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """

    def save_data(self, file_path: str) -> None:
        """
        현재 lane_masks 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """

    def set_lane_masks(self, lane_masks: np.ndarray) -> None:
        """
        lane_masks 데이터를 설정한다.

        Args:
            lane_masks (np.ndarray): 설정할 lane_masks 데이터
        """

    def get_lane_masks(self) -> np.ndarray:
        """
        현재 lane_masks 데이터를 반환한다.

        Returns:
            np.ndarray: lane_masks 데이터
        """

    def add_lane_mask(self, mask: np.ndarray) -> None:
        """
        새로운 차선 마스크를 추가한다.

        Args:
            mask (np.ndarray): 추가할 차선 마스크
        """

    def update_lane_mask(self, instance_index: int, mask: np.ndarray) -> None:
        """
        특정 인스턴스의 차선 마스크를 업데이트한다.

        Args:
            instance_index (int): 업데이트할 인스턴스의 인덱스
            mask (np.ndarray): 새로운 마스크 데이터
        """

    def remove_lane_mask(self, instance_index: int) -> None:
        """
        특정 인스턴스의 차선 마스크를 제거한다.

        Args:
            instance_index (int): 제거할 인스턴스의 인덱스
        """

    def get_lane_mask(self, instance_index: int) -> np.ndarray:
        """
        특정 인스턴스의 차선 마스크를 반환한다.

        Args:
            instance_index (int): 조회할 인스턴스의 인덱스

        Returns:
            np.ndarray: 차선 마스크 데이터
        """

    def set_bev_dimensions(self, height: int, width: int) -> None:
        """
        BEV(Bird's Eye View) 이미지의 높이와 너비를 설정한다.

        Args:
            height (int): BEV 이미지의 높이
            width (int): BEV 이미지의 너비
        """

    def get_bev_dimensions(self) -> Tuple[int, int]:
        """
        현재 설정된 BEV 이미지의 높이와 너비를 반환한다.

        Returns:
            Tuple[int, int]: (높이, 너비)
        """

    def validate_data(self) -> bool:
        """
        현재 lane_masks 데이터가 유효한지 검증한다.

        Returns:
            bool: 데이터가 유효하면 True, 그렇지 않으면 False
        """

    def get_statistics(self) -> Dict[str, float]:
        """
        lane_masks 데이터의 통계 정보를 계산하여 반환한다.

        Returns:
            Dict[str, float]: 통계 정보 (예: 평균 마스크 면적, 최대 마스크 값 등)
        """

    def to_tensor(self) -> 'torch.Tensor':
        """
        lane_masks 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            torch.Tensor: PyTorch 텐서로 변환된 lane_masks 데이터
        """

    def from_tensor(self, tensor: 'torch.Tensor') -> None:
        """
        PyTorch 텐서를 lane_masks 데이터로 변환하여 설정한다.

        Args:
            tensor (torch.Tensor): 변환할 PyTorch 텐서
        """

    def visualize_masks(self, output_path: Optional[str] = None) -> None:
        """
        lane_masks 데이터를 시각화한다.

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로
        """

    def apply_threshold(self, threshold: int) -> None:
        """
        lane_masks에 임계값을 적용하여 이진화한다.

        Args:
            threshold (int): 적용할 임계값 (0-255)
        """

    def merge_masks(self) -> np.ndarray:
        """
        모든 인스턴스의 마스크를 하나의 마스크로 병합한다.

        Returns:
            np.ndarray: 병합된 마스크
        """

    def split_masks(self, merged_mask: np.ndarray) -> None:
        """
        병합된 마스크를 개별 인스턴스 마스크로 분할한다.

        Args:
            merged_mask (np.ndarray): 분할할 병합된 마스크
        """

    def resize_masks(self, new_height: int, new_width: int) -> None:
        """
        모든 마스크의 크기를 조정한다.

        Args:
            new_height (int): 새로운 높이
            new_width (int): 새로운 너비
        """

    def rotate_masks(self, angle: float) -> None:
        """
        모든 마스크를 지정된 각도로 회전한다.

        Args:
            angle (float): 회전 각도 (도 단위)
        """

    def filter_masks(self, condition: callable) -> List[int]:
        """
        주어진 조건에 맞는 마스크들의 인덱스를 반환한다.

        Args:
            condition (callable): 필터링 조건 함수

        Returns:
            List[int]: 조건을 만족하는 마스크들의 인덱스 리스트
        """

    def get_mask_area(self, instance_index: int) -> int:
        """
        특정 인스턴스 마스크의 면적을 계산한다.

        Args:
            instance_index (int): 계산할 인스턴스의 인덱스

        Returns:
            int: 마스크의 면적 (픽셀 수)
        """

    def apply_mask_operation(self, operation: callable) -> None:
        """
        모든 마스크에 사용자 정의 연산을 적용한다.

        Args:
            operation (callable): 적용할 연산 함수
        """
        

class UniADDataManagerOnline_mapping(UniADDataManagerLane_masks):
    def __init__(self):
        """
        online_mapping 데이터와 관련 메타데이터를 초기화한다.
        """
        super().__init__()
        self.timestamp = None

    def load_online_mapping_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 online_mapping 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """

    def save_online_mapping_data(self, file_path: str) -> None:
        """
        현재 online_mapping 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """

    def update_online_mapping(self, new_lane_masks: np.ndarray) -> None:
        """
        새로운 lane_masks 데이터로 online_mapping을 업데이트한다.

        Args:
            new_lane_masks (np.ndarray): 새로운 lane_masks 데이터
        """

    def get_online_mapping(self) -> np.ndarray:
        """
        현재 online_mapping 데이터를 반환한다.

        Returns:
            np.ndarray: online_mapping 데이터
        """

    def set_timestamp(self, timestamp: float) -> None:
        """
        online_mapping 데이터의 타임스탬프를 설정한다.

        Args:
            timestamp (float): 설정할 타임스탬프
        """

    def get_timestamp(self) -> float:
        """
        현재 online_mapping 데이터의 타임스탬프를 반환한다.

        Returns:
            float: 타임스탬프
        """

    def merge_with_previous_mapping(self, previous_mapping: 'UniADDataManagerOnline_mapping') -> None:
        """
        이전 매핑 데이터와 현재 매핑 데이터를 병합한다.

        Args:
            previous_mapping (UniADDataManagerOnline_mapping): 이전 매핑 데이터
        """

    def filter_outdated_lanes(self, time_threshold: float) -> None:
        """
        지정된 시간 임계값보다 오래된 차선 데이터를 필터링한다.

        Args:
            time_threshold (float): 시간 임계값 (초 단위)
        """

    def interpolate_lanes(self) -> None:
        """
        누락된 차선 데이터를 보간한다.
        """

    def extrapolate_lanes(self, time_delta: float) -> None:
        """
        현재 차선 데이터를 기반으로 미래 차선 위치를 예측한다.

        Args:
            time_delta (float): 예측할 시간 간격 (초 단위)
        """

    def apply_confidence_threshold(self, confidence_threshold: float) -> None:
        """
        지정된 신뢰도 임계값 이하의 차선 데이터를 제거한다.

        Args:
            confidence_threshold (float): 신뢰도 임계값 (0.0 ~ 1.0)
        """

    def get_lane_change_probability(self) -> Dict[int, float]:
        """
        각 차선에 대한 차선 변경 확률을 계산한다.

        Returns:
            Dict[int, float]: 차선 인덱스를 키로, 차선 변경 확률을 값으로 하는 딕셔너리
        """

    def detect_new_lanes(self, previous_mapping: 'UniADDataManagerOnline_mapping') -> List[int]:
        """
        이전 매핑과 비교하여 새로 감지된 차선의 인덱스를 반환한다.

        Args:
            previous_mapping (UniADDataManagerOnline_mapping): 이전 매핑 데이터

        Returns:
            List[int]: 새로 감지된 차선의 인덱스 리스트
        """

    def remove_temporary_obstructions(self) -> None:
        """
        일시적인 장애물로 인한 차선 단절을 제거하고 연속성을 복원한다.
        """

    def estimate_lane_curvature(self) -> Dict[int, float]:
        """
        각 차선의 곡률을 추정한다.

        Returns:
            Dict[int, float]: 차선 인덱스를 키로, 곡률을 값으로 하는 딕셔너리
        """

    def classify_lane_types(self) -> Dict[int, str]:
        """
        각 차선의 유형을 분류한다 (예: 실선, 점선, 이중선 등).

        Returns:
            Dict[int, str]: 차선 인덱스를 키로, 차선 유형을 값으로 하는 딕셔너리
        """

    def validate_online_mapping(self) -> bool:
        """
        현재 online_mapping 데이터의 유효성을 검증한다.

        Returns:
            bool: 데이터가 유효하면 True, 그렇지 않으면 False
        """

    def get_mapping_statistics(self) -> Dict[str, float]:
        """
        online_mapping 데이터의 통계 정보를 계산하여 반환한다.

        Returns:
            Dict[str, float]: 통계 정보 (예: 평균 차선 수, 평균 차선 길이 등)
        """

    def visualize_online_mapping(self, output_path: Optional[str] = None) -> None:
        """
        online_mapping 데이터를 시각화한다.

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로
        """

    def to_tensor(self) -> 'torch.Tensor':
        """
        online_mapping 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            torch.Tensor: PyTorch 텐서로 변환된 online_mapping 데이터
        """

    def from_tensor(self, tensor: 'torch.Tensor') -> None:
        """
        PyTorch 텐서를 online_mapping 데이터로 변환하여 설정한다.

        Args:
            tensor (torch.Tensor): 변환할 PyTorch 텐서
        """


class UniADDataManagerSdc_planning:
    def __init__(self):
        """
        sdc_planning 데이터와 관련 메타데이터를 초기화한다.
        """
        self.sdc_planning = np.array([], dtype=np.float32)
        self.count_of_planning_horizons = 0

    def load_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 sdc_planning 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """

    def save_data(self, file_path: str) -> None:
        """
        현재 sdc_planning 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """

    def set_sdc_planning(self, planning_data: np.ndarray) -> None:
        """
        sdc_planning 데이터를 설정한다.

        Args:
            planning_data (np.ndarray): 설정할 sdc_planning 데이터
        """

    def get_sdc_planning(self) -> np.ndarray:
        """
        현재 sdc_planning 데이터를 반환한다.

        Returns:
            np.ndarray: sdc_planning 데이터
        """

    def add_planning_horizon(self, x: float, y: float, yaw: float) -> None:
        """
        새로운 계획 지점을 추가한다.

        Args:
            x (float): x 좌표
            y (float): y 좌표
            yaw (float): 방향각 (도 단위)
        """

    def update_planning_horizon(self, index: int, x: float, y: float, yaw: float) -> None:
        """
        특정 인덱스의 계획 지점을 업데이트한다.

        Args:
            index (int): 업데이트할 지점의 인덱스
            x (float): 새로운 x 좌표
            y (float): 새로운 y 좌표
            yaw (float): 새로운 방향각 (도 단위)
        """

    def remove_planning_horizon(self, index: int) -> None:
        """
        특정 인덱스의 계획 지점을 제거한다.

        Args:
            index (int): 제거할 지점의 인덱스
        """

    def get_planning_horizon(self, index: int) -> Tuple[float, float, float]:
        """
        특정 인덱스의 계획 지점 데이터를 반환한다.

        Args:
            index (int): 조회할 지점의 인덱스

        Returns:
            Tuple[float, float, float]: (x, y, yaw) 데이터
        """

    def get_count_of_planning_horizons(self) -> int:
        """
        현재 계획 지점의 수를 반환한다.

        Returns:
            int: 계획 지점의 수
        """

    def validate_data(self) -> bool:
        """
        현재 sdc_planning 데이터가 유효한지 검증한다.

        Returns:
            bool: 데이터가 유효하면 True, 그렇지 않으면 False
        """

    def get_statistics(self) -> Dict[str, float]:
        """
        sdc_planning 데이터의 통계 정보를 계산하여 반환한다.

        Returns:
            Dict[str, float]: 통계 정보 (예: 평균 속도, 최대 회전각 등)
        """

    def interpolate_planning(self, target_count: int) -> None:
        """
        현재 계획 데이터를 목표 개수로 보간한다.

        Args:
            target_count (int): 목표 계획 지점 수
        """

    def extrapolate_planning(self, additional_count: int) -> None:
        """
        현재 계획 데이터를 기반으로 추가 지점을 외삽한다.

        Args:
            additional_count (int): 추가할 계획 지점 수
        """

    def smooth_trajectory(self, smoothing_factor: float) -> None:
        """
        계획된 궤적을 부드럽게 만든다.

        Args:
            smoothing_factor (float): 스무딩 강도 (0.0 ~ 1.0)
        """

    def calculate_curvature(self) -> List[float]:
        """
        각 계획 지점에서의 곡률을 계산한다.

        Returns:
            List[float]: 각 지점의 곡률 리스트
        """

    def calculate_velocity(self, time_step: float) -> List[float]:
        """
        각 계획 지점 간의 속도를 계산한다.

        Args:
            time_step (float): 계획 지점 간의 시간 간격 (초 단위)

        Returns:
            List[float]: 각 구간의 속도 리스트
        """

    def calculate_acceleration(self, time_step: float) -> List[float]:
        """
        각 계획 지점 간의 가속도를 계산한다.

        Args:
            time_step (float): 계획 지점 간의 시간 간격 (초 단위)

        Returns:
            List[float]: 각 구간의 가속도 리스트
        """

    def check_collision(self, obstacles: List[Tuple[float, float, float, float]]) -> List[bool]:
        """
        계획된 경로와 주어진 장애물들 간의 충돌 여부를 확인한다.

        Args:
            obstacles (List[Tuple[float, float, float, float]]): 
                장애물 리스트 (x, y, 너비, 높이)

        Returns:
            List[bool]: 각 계획 지점에서의 충돌 여부 리스트
        """

    def optimize_trajectory(self, cost_function: callable) -> None:
        """
        주어진 비용 함수를 사용하여 궤적을 최적화한다.

        Args:
            cost_function (callable): 궤적 평가를 위한 비용 함수
        """

    def visualize_planning(self, output_path: Optional[str] = None) -> None:
        """
        sdc_planning 데이터를 시각화한다.

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로
        """

    def to_tensor(self) -> 'torch.Tensor':
        """
        sdc_planning 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            torch.Tensor: PyTorch 텐서로 변환된 sdc_planning 데이터
        """

    def from_tensor(self, tensor: 'torch.Tensor') -> None:
        """
        PyTorch 텐서를 sdc_planning 데이터로 변환하여 설정한다.

        Args:
            tensor (torch.Tensor): 변환할 PyTorch 텐서
        """

    def apply_safety_constraints(self, max_acceleration: float, max_yaw_rate: float) -> None:
        """
        안전 제약 조건을 적용하여 계획을 수정한다.

        Args:
            max_acceleration (float): 최대 허용 가속도
            max_yaw_rate (float): 최대 허용 회전 속도 (도/초)
        """

    def get_total_distance(self) -> float:
        """
        계획된 경로의 총 거리를 계산한다.

        Returns:
            float: 총 거리 (미터)
        """

    def get_total_rotation(self) -> float:
        """
        계획된 경로의 총 회전각을 계산한다.

        Returns:
            float: 총 회전각 (도)
        """
        
class UniADDataManagerFuture_boxes:
    def __init__(self):
        """
        future_boxes 데이터와 관련 메타데이터를 초기화한다.
        """
        self.future_boxes = np.array([], dtype=bool)
        self.count_of_planning_horizons = 0

    def load_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 future_boxes 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """

    def save_data(self, file_path: str) -> None:
        """
        현재 future_boxes 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """

    def set_future_boxes(self, future_boxes_data: np.ndarray) -> None:
        """
        future_boxes 데이터를 설정한다.

        Args:
            future_boxes_data (np.ndarray): 설정할 future_boxes 데이터
        """

    def get_future_boxes(self) -> np.ndarray:
        """
        현재 future_boxes 데이터를 반환한다.

        Returns:
            np.ndarray: future_boxes 데이터
        """

    def add_planning_horizon(self, x: bool, y: bool) -> None:
        """
        새로운 계획 지점의 future_box 데이터를 추가한다.

        Args:
            x (bool): x 축 데이터
            y (bool): y 축 데이터
        """

    def update_planning_horizon(self, index: int, x: bool, y: bool) -> None:
        """
        특정 인덱스의 future_box 데이터를 업데이트한다.

        Args:
            index (int): 업데이트할 지점의 인덱스
            x (bool): 새로운 x 축 데이터
            y (bool): 새로운 y 축 데이터
        """

    def remove_planning_horizon(self, index: int) -> None:
        """
        특정 인덱스의 future_box 데이터를 제거한다.

        Args:
            index (int): 제거할 지점의 인덱스
        """

    def get_planning_horizon(self, index: int) -> Tuple[bool, bool]:
        """
        특정 인덱스의 future_box 데이터를 반환한다.

        Args:
            index (int): 조회할 지점의 인덱스

        Returns:
            Tuple[bool, bool]: (x, y) 데이터
        """

    def get_count_of_planning_horizons(self) -> int:
        """
        현재 계획 지점의 수를 반환한다.

        Returns:
            int: 계획 지점의 수
        """

    def validate_data(self) -> bool:
        """
        현재 future_boxes 데이터가 유효한지 검증한다.

        Returns:
            bool: 데이터가 유효하면 True, 그렇지 않으면 False
        """

    def get_statistics(self) -> Dict[str, int]:
        """
        future_boxes 데이터의 통계 정보를 계산하여 반환한다.

        Returns:
            Dict[str, int]: 통계 정보 (예: x축 True 개수, y축 True 개수 등)
        """

    def interpolate_future_boxes(self, target_count: int) -> None:
        """
        현재 future_boxes 데이터를 목표 개수로 보간한다.

        Args:
            target_count (int): 목표 계획 지점 수
        """

    def extrapolate_future_boxes(self, additional_count: int) -> None:
        """
        현재 future_boxes 데이터를 기반으로 추가 지점을 외삽한다.

        Args:
            additional_count (int): 추가할 계획 지점 수
        """

    def apply_logical_operations(self, operation: str) -> np.ndarray:
        """
        x와 y 축 데이터에 대해 논리 연산을 수행한다.

        Args:
            operation (str): 수행할 논리 연산 ('and', 'or', 'xor')

        Returns:
            np.ndarray: 논리 연산 결과
        """

    def get_continuous_segments(self, axis: str) -> List[Tuple[int, int]]:
        """
        연속된 True 값을 가진 세그먼트를 찾는다.

        Args:
            axis (str): 분석할 축 ('x' 또는 'y')

        Returns:
            List[Tuple[int, int]]: 연속 세그먼트의 시작과 끝 인덱스 리스트
        """

    def find_pattern(self, pattern: List[Tuple[bool, bool]]) -> List[int]:
        """
        주어진 패턴과 일치하는 위치를 찾는다.

        Args:
            pattern (List[Tuple[bool, bool]]): 찾을 패턴

        Returns:
            List[int]: 패턴이 발견된 시작 인덱스 리스트
        """

    def apply_sliding_window(self, window_size: int, operation: callable) -> np.ndarray:
        """
        슬라이딩 윈도우 기법을 적용하여 데이터를 처리한다.

        Args:
            window_size (int): 윈도우 크기
            operation (callable): 윈도우에 적용할 연산

        Returns:
            np.ndarray: 처리된 데이터
        """

    def visualize_future_boxes(self, output_path: Optional[str] = None) -> None:
        """
        future_boxes 데이터를 시각화한다.

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로
        """

    def to_tensor(self) -> 'torch.Tensor':
        """
        future_boxes 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            torch.Tensor: PyTorch 텐서로 변환된 future_boxes 데이터
        """

    def from_tensor(self, tensor: 'torch.Tensor') -> None:
        """
        PyTorch 텐서를 future_boxes 데이터로 변환하여 설정한다.

        Args:
            tensor (torch.Tensor): 변환할 PyTorch 텐서
        """

    def compare_with_ground_truth(self, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        예측된 future_boxes와 실제 ground truth를 비교한다.

        Args:
            ground_truth (np.ndarray): 실제 future_boxes 데이터

        Returns:
            Dict[str, float]: 정확도, 정밀도, 재현율 등의 메트릭
        """

    def generate_random_future_boxes(self, count: int) -> None:
        """
        무작위 future_boxes 데이터를 생성한다.

        Args:
            count (int): 생성할 데이터 포인트 수
        """

    def apply_smoothing(self, window_size: int) -> None:
        """
        future_boxes 데이터에 스무딩을 적용한다.

        Args:
            window_size (int): 스무딩 윈도우 크기
        """

    def encode_run_length(self) -> Dict[str, List[int]]:
        """
        future_boxes 데이터를 런 길이 인코딩으로 변환 한다

        Returns:
            Dict[str, List[int]]: x축과 y축에 대한 런 길이 인코딩 결과
        """

    def decode_run_length(self, encoded_data: Dict[str, List[int]]) -> None:
        """
        런 길이 인코딩된 데이터를 future_boxes 형식으로 디코딩한다.

        Args:
            encoded_data (Dict[str, List[int]]): 런 길이 인코딩된 데이터
        """
        

class UniADDataManagerCommand:
    def __init__(self):
        """
        command 데이터와 관련 메타데이터를 초기화한다.
        """
        self.command = np.array([], dtype=np.uint8)

    def load_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 command 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """

    def save_data(self, file_path: str) -> None:
        """
        현재 command 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """

    def set_command(self, command: int) -> None:
        """
        command 데이터를 설정한다.

        Args:
            command (int): 설정할 command 값 (0-255 범위)
        """

    def get_command(self) -> int:
        """
        현재 command 데이터를 반환한다.

        Returns:
            int: command 값
        """

    def validate_command(self) -> bool:
        """
        현재 command 데이터가 유효한지 검증한다.

        Returns:
            bool: 데이터가 유효하면 True, 그렇지 않으면 False
        """

    def increment_command(self, step: int = 1) -> None:
        """
        command 값을 지정된 단계만큼 증가시킨다.

        Args:
            step (int): 증가시킬 단계 (기본값: 1)
        """

    def decrement_command(self, step: int = 1) -> None:
        """
        command 값을 지정된 단계만큼 감소시킨다.

        Args:
            step (int): 감소시킬 단계 (기본값: 1)
        """

    def reset_command(self) -> None:
        """
        command 값을 0으로 초기화한다.
        """

    def to_binary(self) -> str:
        """
        command 값을 8비트 이진 문자열로 변환 한다

        Returns:
            str: 8비트 이진 문자열
        """

    def from_binary(self, binary_str: str) -> None:
        """
        8비트 이진 문자열을 command 값으로 변환하여 설정한다.

        Args:
            binary_str (str): 8비트 이진 문자열
        """

    def to_one_hot(self) -> np.ndarray:
        """
        command 값을 원-핫 인코딩으로 변환 한다

        Returns:
            np.ndarray: 256 길이의 원-핫 인코딩 배열
        """

    def from_one_hot(self, one_hot: np.ndarray) -> None:
        """
        원-핫 인코딩 배열을 command 값으로 변환하여 설정한다.

        Args:
            one_hot (np.ndarray): 256 길이의 원-핫 인코딩 배열
        """

    def map_to_action(self, action_map: Dict[int, str]) -> str:
        """
        현재 command 값을 주어진 액션 맵에 따라 문자열 액션으로 변환 한다

        Args:
            action_map (Dict[int, str]): command 값과 액션 문자열의 매핑

        Returns:
            str: 매핑된 액션 문자열
        """

    def from_action(self, action: str, action_map: Dict[str, int]) -> None:
        """
        주어진 액션 문자열을 액션 맵에 따라 command 값으로 변환하여 설정한다.

        Args:
            action (str): 액션 문자열
            action_map (Dict[str, int]): 액션 문자열과 command 값의 매핑
        """

    def visualize_command(self, output_path: Optional[str] = None) -> None:
        """
        현재 command 값을 시각화한다.

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로
        """

    def to_tensor(self) -> 'torch.Tensor':
        """
        command 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            torch.Tensor: PyTorch 텐서로 변환된 command 데이터
        """

    def from_tensor(self, tensor: 'torch.Tensor') -> None:
        """
        PyTorch 텐서를 command 데이터로 변환하여 설정한다.

        Args:
            tensor (torch.Tensor): 변환할 PyTorch 텐서
        """

    def get_command_history(self) -> List[int]:
        """
        command 값의 변경 이력을 반환한다.

        Returns:
            List[int]: command 값 변경 이력
        """

    def clear_command_history(self) -> None:
        """
        command 값의 변경 이력을 초기화한다.
        """

    def is_command_in_range(self, start: int, end: int) -> bool:
        """
        현재 command 값이 지정된 범위 내에 있는지 확인한다.

        Args:
            start (int): 범위의 시작값
            end (int): 범위의 끝값

        Returns:
            bool: command 값이 범위 내에 있으면 True, 그렇지 않으면 False
        """

    def apply_command_mask(self, mask: int) -> int:
        """
        현재 command 값에 비트 마스크를 적용한다.

        Args:
            mask (int): 적용할 비트 마스크

        Returns:
            int: 마스크가 적용된 command 값
        """

    def generate_random_command(self) -> None:
        """
        무작위 command 값을 생성하여 설정한다.
        """

    def compare_commands(self, other_command: 'UniADDataManagerCommand') -> bool:
        """
        현재 command 값과 다른 UniADDataManagerCommand 인스턴스의 command 값을 비교한다.

        Args:
            other_command (UniADDataManagerCommand): 비교할 다른 command 인스턴스

        Returns:
            bool: 두 command 값이 같으면 True, 다르면 False
        """

class UniADDataManagerPlanning(UniADDataManagerSdc_planning, UniADDataManagerFuture_boxes, UniADDataManagerCommand):
    """
    UniAD 학습에 필요한 data 를 관리하는 클래스
        - planning
            - sdc_planning: float32, (count_of_planning_horizons, 3)
            - future_boxes: boolean, (count_of_planning_horizons, 2)
            - command: uint8, (1,)
    """
    
class UniADDataManagerPlanning(UniADDataManagerSdc_planning, UniADDataManagerFuture_boxes, UniADDataManagerCommand):
    def __init__(self):
        """
        planning 데이터와 관련 메타데이터를 초기화한다.
        """
        UniADDataManagerSdc_planning.__init__(self)
        UniADDataManagerFuture_boxes.__init__(self)
        UniADDataManagerCommand.__init__(self)

    def load_planning_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 전체 planning 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """

    def save_planning_data(self, file_path: str) -> None:
        """
        현재 전체 planning 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """

    def set_planning_data(self, sdc_planning: np.ndarray, future_boxes: np.ndarray, command: int) -> None:
        """
        전체 planning 데이터를 설정한다.

        Args:
            sdc_planning (np.ndarray): sdc_planning 데이터
            future_boxes (np.ndarray): future_boxes 데이터
            command (int): command 값
        """

    def get_planning_data(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        현재 전체 planning 데이터를 반환한다.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: (sdc_planning, future_boxes, command) 데이터
        """

    def validate_planning_data(self) -> bool:
        """
        현재 전체 planning 데이터가 유효한지 검증한다.

        Returns:
            bool: 모든 데이터가 유효하면 True, 그렇지 않으면 False
        """

    def add_planning_horizon(self, x: float, y: float, yaw: float, future_box_x: bool, future_box_y: bool) -> None:
        """
        새로운 계획 지점을 추가한다.

        Args:
            x (float): sdc_planning의 x 좌표
            y (float): sdc_planning의 y 좌표
            yaw (float): sdc_planning의 방향각
            future_box_x (bool): future_boxes의 x 축 데이터
            future_box_y (bool): future_boxes의 y 축 데이터
        """

    def update_planning_horizon(self, index: int, x: float, y: float, yaw: float, future_box_x: bool, future_box_y: bool) -> None:
        """
        특정 인덱스의 계획 지점을 업데이트한다.

        Args:
            index (int): 업데이트할 지점의 인덱스
            x (float): 새로운 sdc_planning의 x 좌표
            y (float): 새로운 sdc_planning의 y 좌표
            yaw (float): 새로운 sdc_planning의 방향각
            future_box_x (bool): 새로운 future_boxes의 x 축 데이터
            future_box_y (bool): 새로운 future_boxes의 y 축 데이터
        """

    def remove_planning_horizon(self, index: int) -> None:
        """
        특정 인덱스의 계획 지점을 제거한다.

        Args:
            index (int): 제거할 지점의 인덱스
        """

    def get_planning_horizon(self, index: int) -> Tuple[Tuple[float, float, float], Tuple[bool, bool]]:
        """
        특정 인덱스의 계획 지점 데이터를 반환한다.

        Args:
            index (int): 조회할 지점의 인덱스

        Returns:
            Tuple[Tuple[float, float, float], Tuple[bool, bool]]: ((x, y, yaw), (future_box_x, future_box_y))
        """

    def get_count_of_planning_horizons(self) -> int:
        """
        현재 계획 지점의 수를 반환한다.

        Returns:
            int: 계획 지점의 수
        """

    def visualize_planning_data(self, output_path: Optional[str] = None) -> None:
        """
        전체 planning 데이터를 시각화한다.

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로
        """

    def to_tensor(self) -> Dict[str, 'torch.Tensor']:
        """
        전체 planning 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            Dict[str, torch.Tensor]: 각 데이터 유형별 PyTorch 텐서
        """

    def from_tensor(self, tensors: Dict[str, 'torch.Tensor']) -> None:
        """
        PyTorch 텐서를 planning 데이터로 변환하여 설정한다.

        Args:
            tensors (Dict[str, torch.Tensor]): 변환할 PyTorch 텐서들
        """

    def interpolate_planning_data(self, target_count: int) -> None:
        """
        현재 planning 데이터를 목표 개수로 보간한다.

        Args:
            target_count (int): 목표 계획 지점 수
        """

    def extrapolate_planning_data(self, additional_count: int) -> None:
        """
        현재 planning 데이터를 기반으로 추가 지점을 외삽한다.

        Args:
            additional_count (int): 추가할 계획 지점 수
        """

    def apply_command_to_planning(self) -> None:
        """
        현재 command 값을 기반으로 planning 데이터를 조정한다.
        """

    def generate_random_planning_data(self, count: int) -> None:
        """
        무작위 planning 데이터를 생성한다.

        Args:
            count (int): 생성할 데이터 포인트 수
        """

    def apply_smoothing_to_planning(self, window_size: int) -> None:
        """
        planning 데이터에 스무딩을 적용한다.

        Args:
            window_size (int): 스무딩 윈도우 크기
        """

    def find_collision_points(self) -> List[int]:
        """
        sdc_planning과 future_boxes 데이터를 비교하여 충돌 가능성이 있는 지점을 찾는다.

        Returns:
            List[int]: 충돌 가능성이 있는 지점의 인덱스 리스트
        """

    def calculate_planning_statistics(self) -> Dict[str, float]:
        """
        planning 데이터의 통계 정보를 계산한다.

        Returns:
            Dict[str, float]: 통계 정보 (예: 평균 속도, 최대 회전각 등)
        """

    def export_to_json(self, file_path: str) -> None:
        """
        planning 데이터를 JSON 형식으로 내보낸다.

        Args:
            file_path (str): 저장할 JSON 파일의 경로
        """

    def import_from_json(self, file_path: str) -> None:
        """
        JSON 파일에서 planning 데이터를 가져온다.

        Args:
            file_path (str): 가져올 JSON 파일의 경로
        """

    def apply_transformation(self, translation: Tuple[float, float], rotation: float) -> None:
        """
        전체 planning 데이터에 변환(이동 및 회전)을 적용한다.

        Args:
            translation (Tuple[float, float]): (x, y) 이동량
            rotation (float): 회전각 (라디안)
        """

    def get_planning_summary(self) -> str:
        """
        현재 planning 데이터의 요약 정보를 문자열로 반환한다.

        Returns:
            str: planning 데이터 요약 정보
        """

    def compare_with_ground_truth(self, ground_truth: 'UniADDataManagerPlanning') -> Dict[str, float]:
        """
        예측된 planning 데이터와 실제 ground truth를 비교한다.

        Args:
            ground_truth (UniADDataManagerPlanning): 실제 planning 데이터

        Returns:
            Dict[str, float]: 각 구성 요소별 비교 메트릭
        """


class UniADDataManager(UniADDataManagerInput, UniADAnotationDataManager):
    def __init__(self):
        """
        UniAD 학습에 필요한 모든 데이터와 관련 메타데이터를 초기화한다.
        """
        UniADDataManagerInput.__init__(self)
        UniADAnotationDataManager.__init__(self)

    def load_data(self, file_path: str) -> None:
        """
        지정된 파일 경로에서 모든 UniAD 데이터를 로드한다.

        Args:
            file_path (str): 데이터 파일의 경로
        """

    def save_data(self, file_path: str) -> None:
        """
        현재 모든 UniAD 데이터를 지정된 파일 경로에 저장한다.

        Args:
            file_path (str): 저장할 파일의 경로
        """

    def get_input_data(self) -> Dict[str, Any]:
        """
        현재 입력 데이터를 반환한다.

        Returns:
            Dict[str, Any]: 입력 데이터
        """

    def get_annotation_data(self) -> Dict[str, Any]:
        """
        현재 어노테이션 데이터를 반환한다.

        Returns:
            Dict[str, Any]: 어노테이션 데이터
        """

    def set_input_data(self, input_data: Dict[str, Any]) -> None:
        """
        입력 데이터를 설정한다.

        Args:
            input_data (Dict[str, Any]): 설정할 입력 데이터
        """

    def set_annotation_data(self, annotation_data: Dict[str, Any]) -> None:
        """
        어노테이션 데이터를 설정한다.

        Args:
            annotation_data (Dict[str, Any]): 설정할 어노테이션 데이터
        """

    def validate_data(self) -> bool:
        """
        현재 모든 UniAD 데이터가 유효한지 검증한다.

        Returns:
            bool: 모든 데이터가 유효하면 True, 그렇지 않으면 False
        """

    def preprocess_data(self) -> None:
        """
        모든 UniAD 데이터에 대해 전처리를 수행한다.
        """

    def augment_data(self) -> None:
        """
        데이터 증강 기법을 적용하여 학습 데이터를 확장한다.
        """

    def visualize_data(self, output_path: Optional[str] = None) -> None:
        """
        모든 UniAD 데이터를 시각화한다.

        Args:
            output_path (str, optional): 시각화 결과를 저장할 파일 경로
        """

    def to_tensor(self) -> Dict[str, 'torch.Tensor']:
        """
        모든 UniAD 데이터를 PyTorch 텐서로 변환 한다

        Returns:
            Dict[str, torch.Tensor]: 각 데이터 유형별 PyTorch 텐서
        """

    def from_tensor(self, tensors: Dict[str, 'torch.Tensor']) -> None:
        """
        PyTorch 텐서를 UniAD 데이터로 변환하여 설정한다.

        Args:
            tensors (Dict[str, torch.Tensor]): 변환할 PyTorch 텐서들
        """

    def get_batch(self, batch_size: int) -> Dict[str, Any]:
        """
        지정된 배치 크기의 데이터를 반환한다.

        Args:
            batch_size (int): 배치 크기

        Returns:
            Dict[str, Any]: 배치 데이터
        """

    def shuffle_data(self) -> None:
        """
        모든 UniAD 데이터를 무작위로 섞는다.
        """

    def split_data(self, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[Any, Any, Any]:
        """
        데이터를 학습, 검증, 테스트 세트로 분할한다.

        Args:
            train_ratio (float): 학습 데이터 비율
            val_ratio (float): 검증 데이터 비율
            test_ratio (float): 테스트 데이터 비율

        Returns:
            Tuple[Any, Any, Any]: (학습 데이터, 검증 데이터, 테스트 데이터)
        """

    def normalize_data(self) -> None:
        """
        모든 수치형 데이터를 정규화한다.
        """

    def denormalize_data(self) -> None:
        """
        정규화된 데이터를 원래 스케일로 되돌린다.
        """

    def filter_data(self, condition: Callable[[Any], bool]) -> None:
        """
        주어진 조건에 따라 데이터를 필터링한다.

        Args:
            condition (Callable[[Any], bool]): 필터링 조건 함수
        """

    def merge_data(self, other_data: 'UniADDataManager') -> None:
        """
        다른 UniADDataManager 인스턴스의 데이터를 현재 인스턴스와 병합한다.

        Args:
            other_data (UniADDataManager): 병합할 다른 데이터 관리자 인스턴스
        """

    def export_to_csv(self, file_path: str) -> None:
        """
        모든 UniAD 데이터를 CSV 형식으로 내보낸다.

        Args:
            file_path (str): 저장할 CSV 파일의 경로
        """

    def import_from_csv(self, file_path: str) -> None:
        """
        CSV 파일에서 UniAD 데이터를 가져온다.

        Args:
            file_path (str): 가져올 CSV 파일의 경로
        """

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        모든 UniAD 데이터의 통계 정보를 계산한다.

        Returns:
            Dict[str, Any]: 각 데이터 유형별 통계 정보
        """

    def apply_transformation(self, transformation_func: Callable[[Any], Any]) -> None:
        """
        모든 UniAD 데이터에 변환 함수를 적용한다.

        Args:
            transformation_func (Callable[[Any], Any]): 적용할 변환 함수
        """

    def get_data_summary(self) -> str:
        """
        현재 UniAD 데이터의 요약 정보를 문자열로 반환한다.

        Returns:
            str: 데이터 요약 정보
        """

    def validate_consistency(self) -> bool:
        """
        입력 데이터와 어노테이션 데이터 간의 일관성을 검증한다.

        Returns:
            bool: 데이터가 일관성이 있으면 True, 그렇지 않으면 False
        """

    def generate_synthetic_data(self, num_samples: int) -> None:
        """
        지정된 수의 합성 데이터를 생성한다.

        Args:
            num_samples (int): 생성할 샘플 수
        """

    def apply_data_cleaning(self) -> None:
        """
        데이터 클리닝 작업을 수행하여 이상치나 잘못된 데이터를 처리한다.
        """

    def encode_categorical_data(self) -> None:
        """
        범주형 데이터를 인코딩한다.
        """

    def decode_categorical_data(self) -> None:
        """
        인코딩된 범주형 데이터를 원래 형태로 디코딩한다.
        """

    def calculate_feature_importance(self) -> Dict[str, float]:
        """
        각 특성의 중요도를 계산한다.

        Returns:
            Dict[str, float]: 특성별 중요도
        """

    def apply_dimensionality_reduction(self, method: str, n_components: int) -> None:
        """
        차원 축소 기법을 적용한다.

        Args:
            method (str): 사용할 차원 축소 방법 (예: 'PCA', 'tSNE')
            n_components (int): 축소할 차원 수
        """

    def handle_missing_data(self, strategy: str) -> None:
        """
        결측치를 처리한다.

        Args:
            strategy (str): 결측치 처리 전략 (예: 'mean', 'median', 'mode')
        """

    def create_time_series_features(self) -> None:
        """
        시계열 데이터에 대한 특성을 생성한다.
        """

    def apply_smoothing(self, window_size: int) -> None:
        """
        시계열 데이터에 스무딩을 적용한다.

        Args:
            window_size (int): 스무딩 윈도우 크기
        """

    def calculate_correlation_matrix(self) -> np.ndarray:
        """
        특성 간 상관 관계 행렬을 계산한다.

        Returns:
            np.ndarray: 상관 관계 행렬
        """

    def apply_feature_scaling(self, method: str) -> None:
        """
        특성 스케일링을 적용한다.

        Args:
            method (str): 스케일링 방법 (예: 'standard', 'minmax')
        """

    def generate_cross_validation_folds(self, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        교차 검증을 위한 폴드를 생성한다.

        Args:
            n_splits (int): 분할할 폴드 수

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: 학습 및 검증 인덱스 튜플의 리스트
        """

	