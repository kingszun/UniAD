## UniAD overview

본 fork (`kingszun/UniAD`) 는 upstream `OpenDriveLab/UniAD` `v2.0` (2025-10-29 release) 를 base 로 하는 reproduction + MotionFormer 분석 작업 공간이다. parent project `kingszun/krong-kak` 의 Kakao 채용 과제 (UniAD 분석 + MotionFormer → GameFormer 대체 제안) 의 일부.

### 작업 목적

- UniAD 가 본 환경에서 정상 동작함을 확인 (KAK-22 ~ KAK-25)
- MotionFormer module 의 위치 / interface 를 분석 가능한 수준까지 파악 (KAK-26)
- 위 결과를 parent 의 분석 Epic (`KAK-27` 비교, `KAK-28` 대체 검토) 에 사실 자료로 제공

### Repo 구조 (v2.0)

```
uniad/
├── projects/
│   ├── configs/
│   │   ├── _base_/                # shared config
│   │   ├── bevformer/             # BEV encoder pretrain
│   │   ├── stage1_track_map/      # perception (TrackFormer + MapFormer)
│   │   └── stage2_e2e/            # motion + occ + planning
│   └── mmdet3d_plugin/            # 모델 본체
├── tools/
│   ├── train.py, test.py          # mmcv-style entry
│   ├── uniad_dist_train.sh        # multi-GPU train wrapper
│   ├── uniad_dist_eval.sh         # multi-GPU eval wrapper
│   ├── uniad_slurm_*.sh           # slurm 환경
│   ├── uniad_vis_result.sh        # 시각화
│   ├── create_data.py + uniad_create_data.sh  # nuScenes info pkl 생성
│   ├── data_converter/
│   └── analysis_tools/visualize/run.py
├── docker/Dockerfile              # v2.0 단일 Dockerfile
├── docs/
│   ├── INSTALL.md                 # upstream
│   ├── DATA_PREP.md               # upstream
│   ├── TRAIN_EVAL.md              # upstream
│   └── 00-overview.md             # 본 문서 (fork 자체)
├── scripts/                       # host-side data download (KAK-24)
├── requirements.txt
├── LICENSE                        # All Rights Reserved (fork)
└── README.md                      # upstream
```

### Dependency stack (INSTALL.md 기준)

| 항목 | 버전 |
| --- | --- |
| python | 3.9 |
| torch | 2.0.1 |
| torchvision | 0.15.2 |
| torchaudio | 2.0.2 |
| cuda | 11.8 |
| mmcv-full | 1.6.1 |
| mmdet | 2.26.0 |
| mmsegmentation | 0.29.1 |
| mmdet3d | 1.0.0rc6 |

`requirements.txt`: opencv-python 4.8.0.76, einops 0.8.1, numpy 1.22.4, casadi 3.6.7, pytorch-lightning 1.2.5, pandas 1.2.2, torchmetrics 0.6.2, networkx 2.5, motmetrics<=1.1.3, ipython 8.12.3, yapf 0.40.1, google-cloud-bigquery.

### Entry point

| 작업 | command |
| --- | --- |
| training | `tools/uniad_dist_train.sh CONFIG N_GPUS` |
| eval | `tools/uniad_dist_eval.sh CONFIG CKPT N_GPUS` |
| visualization | `python tools/analysis_tools/visualize/run.py --predroot ... --out_folder ...` |
| data info pkl 생성 | `tools/uniad_create_data.sh` (또는 HuggingFace off-the-shelf 사용) |

### Config 단계

- `projects/configs/bevformer/bevformer_base.py` — BEV encoder (BEVFormer) 학습/평가. v2.0 부터 inline. checkpoint 외부 download 권장.
- `projects/configs/stage1_track_map/base_track_map.py` — TrackFormer + MapFormer (perception 단계)
- `projects/configs/stage2_e2e/*.py` — stage1 동결 후 motion + occupancy + planning 학습

### License

- 본 fork: All Rights Reserved (kingszun, 2024-2026). LICENSE 참조
- upstream v2.0: Apache 2.0

### Pretrained checkpoint

HuggingFace `OpenDriveLab/UniAD2.0_R101_nuScenes` 의 ckpts:
- `r101_dcn_fcos3d_pretrain.pth` — backbone init
- `bevformer_r101_dcn_24ep.pth` — BEV encoder
- `uniad_base_track_map.pth` — stage1 결과
- `uniad_base_e2e.pth` — stage2 (full) 결과

같은 repo 의 data/ 폴더:
- `nuscenes_infos_temporal_train.pkl`, `nuscenes_infos_temporal_val.pkl` — info pkl (off-the-shelf)
- `motion_anchor_infos_mode6.pkl` — motion anchor

### BEVFormer 결합

inline. 별도 submodule 아님. `projects/configs/bevformer/bevformer_base.py` 가 동일 repo 내 학습/평가 entry. ckpt 만 외부 (HuggingFace) 에서 download.

BEVFormer 교체 가능 — `bev_embed` + `bev_pos` 만 같은 shape 로 제공하면 다른 BEV encoder (LSS 등) 로 swap 가능 (upstream README 명시).

### Dataset 지원

- nuScenes V1.0 only (full + mini)
- v2.0 의 nuPlan / NAVSIM 은 tools 미release (2025Q2 ETA)

### GPU 요구사항 (TRAIN_EVAL.md 기준)

| 단계 | GPU memory | 권장 GPU 수 / 시간 (8×A100 기준) |
| --- | --- | --- |
| stage1 training | 50 GB (queue_length=5) / 30 GB (=3) | 6 epoch ≈ 2일 |
| stage2 training | 17 GB | 20 epoch ≈ 4일 |
| eval | 명시 없음 (training 보다 적음) | — |

queue_length=3 으로 줄이면 stage1 도 V100 32GB / A100 40GB 에서 가능.

### Data 용량

nuScenes 본체 (nuscenes.org)

| 항목 | 용량 | 필요 시점 |
| --- | --- | --- |
| Mini (10 scene) | 4 GB | KAK-23 local smoke |
| Trainval blobs (10 part) | ~280 GB | KAK-23 full eval / KAK-25 train |
| Trainval metadata | 0.4 GB | 위와 동일 |
| Test blob | ~25 GB | 본 과제 scope 외 |
| CAN bus expansion | 15 MB | 항상 필요 |
| Map expansion v1.3 | 150 MB | 항상 필요 |

UniAD 부속 (HuggingFace)

| 항목 | 용량 | 비고 |
| --- | --- | --- |
| ckpts 4종 | ~2 GB | 항상 필요 |
| info pkls (train+val) | ~700 MB | off-the-shelf 권장 |
| motion anchors | ~10 MB | 항상 필요 |

총 minimum (mini smoke): ~7 GB. full eval/train: ~290 GB + working space ~50 GB.

### Host data layout (KAK-24)

cost 절감을 위해 host 외장 mount 에 download 후 RunPod network volume 으로 1회 upload (gameformer 패턴 참고). RunPod GPU pod 는 즉시 학습/평가 진입 → idle download cost 회피.

권장 layout

```
/mnt/e/datasets/
├── nuscenes/                  # 일반 nuScenes raw
│   ├── samples/, sweeps/, maps/, can_bus/, lidarseg/
│   ├── v1.0-mini/, v1.0-trainval/, v1.0-test/  (test 는 선택)
└── uniad/                     # UniAD 전용
    ├── ckpts/                 # HuggingFace 4종
    ├── infos/                 # info pkls
    └── others/                # motion anchors
```

repo 의 `data/`, `ckpts/` 는 위 path 로 symlink (`scripts/03-link_to_repo.sh`). 둘 다 `.gitignore` 에 이미 포함.

download 흐름은 `scripts/README.md` 참조.

### MotionFormer 위치 (KAK-26 사전 인덱스)

세부 분석은 KAK-26 의 별도 산출물에서 다룬다. 본 문서는 시작 위치만 명시.

- `projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py` — MotionFormer entry class
- `projects/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/` — sub-module 들
  - `base_motion_head.py`
  - `modules.py`
  - `motion_deformable_attn.py` — deformable attention
  - `motion_optimization.py`
  - `motion_utils.py`
- 주변
  - `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` — TrackFormer + BEV encoder hook
  - `projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py` — OccFormer
  - `projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin.py` — Planner

### Parent 관계

- 본 fork 는 `kingszun/krong-kak` 의 submodule (`submodules/kingszun/uniad`)
- parent 의 짝 submodule: `kingszun/GameFormer` (대체 module source)
- parent 의 Epic / Task 는 `KAK-20` 산하 (`KAK-21` ~ `KAK-26`, `KAK-32`)
- 본 fork 의 변경은 본 repo 에서 commit + push 후 parent 의 submodule pointer update
