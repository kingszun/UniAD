## UniAD overview

본 fork (`kingszun/UniAD`) 는 upstream `OpenDriveLab/UniAD` `main` branch (v1, CVPR'23 release) 를 base 로 하는 reproduction + MotionFormer 분석 작업 공간이다. parent project `kingszun/krong-kak` 의 Kakao 채용 과제 (UniAD 분석 + MotionFormer → GameFormer 대체 제안) 의 일부.

### 작업 목적

- UniAD 가 본 환경에서 정상 동작함을 확인 (KAK-22 ~ KAK-25)
- MotionFormer module 의 위치 / interface 를 분석 가능한 수준까지 파악 (KAK-26)
- 위 결과를 parent 의 분석 Epic (`KAK-27` 비교, `KAK-28` 대체 검토) 에 사실 자료로 제공

### Base 결정 trace

KAK-32 에서 v2.0 (`609ee08`) 으로 전환했으나 다음 reproducibility 이슈가 누적

1. upstream `docker/Dockerfile` 가 v1 stack 그대로 — `INSTALL.md` 의 v2.0 stack 과 불일치
2. `mmcv-full==1.6.1` 의 prebuilt wheel 이 cu118/torch2.0 URL 에 부재 (가용 최저 1.7.2)
3. `mmdet3d==1.0.0rc6` 의 source build 가 uv strict isolation 에서 차단

KAK-35 에서 upstream/main (`532fc33`, v1 latest) 로 revert. 분석 보고서 입장에서도 paper 시점 코드와 정합성이 더 높다.

archive: `v1-archive` / `v1-archive-26-05-08` (revert 직전), `v2-archive` / `v2-archive-26-05-08` (v2.0 작업 보존). 둘 다 fork remote 에 push.

### Repo 구조

```
uniad/
├── projects/
│   ├── configs/
│   │   ├── _base_/                # shared config
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
├── docker/
│   ├── Dockerfile                 # fork (KAK-22, gameformer 패턴 + v1 stack)
│   └── entrypoint.sh              # PUBLIC_KEY 처리 + sshd
├── docs/
│   ├── INSTALL.md                 # upstream
│   ├── DATA_PREP.md               # upstream
│   ├── TRAIN_EVAL.md              # upstream
│   └── 00-overview.md             # 본 문서 (fork 자체)
├── scripts/                       # host-side data + container 운영
├── compose.yaml                   # fork
├── .env.example                   # fork
├── requirements.txt
├── LICENSE                        # All Rights Reserved (fork)
└── README.md                      # upstream
```

v2.0 의 `bevformer/` config 폴더는 v1 에는 없음 — BEVFormer pretrain 은 외부 ckpt 로 받음.

### Dependency stack (INSTALL.md 기준)

| 항목 | 버전 | 비고 |
| --- | --- | --- |
| python | 3.8 | INSTALL.md 권장 |
| cuda | 11.1.1 | conda or system |
| torch | 1.9.1+cu111 | torchvision 0.10.1, torchaudio 0.9.1 |
| mmcv-full | 1.4.0 | prebuilt cu111/torch1.9.0 |
| mmdet | 2.14.0 | pip |
| mmsegmentation | 0.14.1 | pip |
| mmdet3d | v0.17.1 | from source (git clone + checkout + pip install -e .) |
| torchmetrics | 0.8.2 | upstream Dockerfile 주석: "old version needed" |

`requirements.txt`: google-cloud-bigquery, motmetrics 1.1.3, einops 0.4.1, numpy 1.20.0, casadi 3.5.5, pytorch-lightning 1.2.5.

추가 build deps (Dockerfile 에 명시): scipy 1.7.3, scikit-image 0.20.0, setuptools 59.5.0, wheel, Cython.

### Entry point

| 작업 | command |
| --- | --- |
| training | `tools/uniad_dist_train.sh CONFIG N_GPUS` |
| eval | `tools/uniad_dist_eval.sh CONFIG CKPT N_GPUS` |
| visualization | `python tools/analysis_tools/visualize/run.py --predroot ... --out_folder ...` |
| data info pkl 생성 | `tools/uniad_create_data.sh` (또는 v1.0 release 의 off-the-shelf 사용) |

### Config 단계

- `projects/configs/stage1_track_map/base_track_map.py` — TrackFormer + MapFormer (perception)
- `projects/configs/stage2_e2e/*.py` — stage1 동결 후 motion + occupancy + planning

### License

- 본 fork: All Rights Reserved (kingszun, 2024-2026). LICENSE 참조
- upstream main: Apache 2.0

### Pretrained checkpoint (GitHub Releases)

| file | source | 용도 |
| --- | --- | --- |
| `bevformer_r101_dcn_24ep.pth` | `zhiqi-li/storage` v1.0 | BEV encoder, stage1 init |
| `uniad_base_track_map.pth` | `OpenDriveLab/UniAD` v1.0 | stage1 결과 |
| `uniad_base_e2e.pth` | `OpenDriveLab/UniAD` v1.0.1 | stage2 (full) 결과 |
| `nuscenes_infos_temporal_train.pkl` | `OpenDriveLab/UniAD` v1.0 | train info pkl (off-the-shelf) |
| `nuscenes_infos_temporal_val.pkl` | `OpenDriveLab/UniAD` v1.0 | val info pkl |
| `motion_anchor_infos_mode6.pkl` | `OpenDriveLab/UniAD` v1.0 | motion anchor |

`scripts/02-download_huggingface.sh` 가 위 6개를 자동 download. (script 이름은 v2.0 잔재 — v1 은 GitHub Releases 사용)

### BEVFormer 결합

inline. 별도 submodule 아님. checkpoint (`bevformer_r101_dcn_24ep.pth`) 만 외부 (GitHub Releases) 에서 받아 stage1 init 으로 사용.

BEVFormer 교체 가능 — `bev_embed` + `bev_pos` 만 같은 shape 로 제공하면 다른 BEV encoder (LSS 등) 로 swap 가능.

### Dataset 지원

nuScenes V1.0 only.

### GPU 요구사항 (TRAIN_EVAL.md 기준)

| 단계 | GPU memory | 권장 GPU 수 / 시간 (8×A100 기준) |
| --- | --- | --- |
| stage1 training | 50 GB (queue_length=5) / 30 GB (=3) | 6 epoch ≈ 2일 |
| stage2 training | 17 GB | 20 epoch ≈ 4일 |
| eval | 명시 없음 (training 보다 적음) | — |

`queue_length=3` 으로 줄이면 stage1 도 V100 32GB / A100 40GB 에서 가능.

cu111 + torch 1.9 의 GPU compute 지원 한계: 7.5 (T4 / 2080Ti), 8.0 (A100), 8.6 (Ampere consumer 3060 등). 4090(8.9) / H100(9.0) 미지원 — 해당 GPU 사용 시 v2 stack 필요 (단 v2 reproducibility 이슈 있음, KAK-35 참조).

### Data 용량

nuScenes 본체 (nuscenes.org)

| 항목 | 용량 | 필요 시점 |
| --- | --- | --- |
| Mini (10 scene) | 4 GB | KAK-23 local smoke (단 v1 release 의 mini info pkl 부재 — 자체 generate 필요) |
| Trainval blobs (10 part) | ~280 GB | KAK-23 full eval / KAK-25 train |
| Trainval metadata | 0.4 GB | 위와 동일 |
| Test blob | ~25 GB | 본 과제 scope 외 |
| CAN bus expansion | 15 MB | 항상 필요 |
| Map expansion v1.3 | 150 MB | 항상 필요 |

UniAD 부속 (GitHub Releases)

| 항목 | 용량 | 비고 |
| --- | --- | --- |
| ckpts 3종 | ~2 GB | bevformer + stage1 + stage2 |
| info pkls (train+val) | ~700 MB | off-the-shelf 권장 |
| motion anchors | ~10 MB | 항상 필요 |

총 minimum (mini smoke + reduced infos): mini 만 ~5 GB. full eval/train: ~290 GB + working space ~50 GB.

### Host data layout (KAK-24)

cost 절감을 위해 host 외장 mount 에 download 후 RunPod network volume 으로 1회 upload (gameformer 패턴). RunPod GPU pod 는 즉시 학습/평가 진입 → idle download cost 회피.

권장 layout

```
/mnt/e/datasets/
├── nuscenes/                  # 일반 nuScenes raw
│   ├── samples/, sweeps/, maps/, can_bus/
│   └── v1.0-mini/, v1.0-trainval/, v1.0-test/  (test 는 선택)
└── uniad/                     # UniAD 전용
    ├── ckpts/                 # GitHub Releases 3종
    ├── infos/                 # info pkls
    └── others/                # motion_anchor_infos_mode6.pkl
```

repo 의 `data/`, `ckpts/` 는 위 path 로 symlink (`scripts/03-link_to_repo.sh`). 둘 다 `.gitignore` 에 이미 포함.

download 흐름은 `scripts/README.md` 참조.

### MotionFormer 위치 (KAK-26 사전 인덱스)

세부 분석은 KAK-26 의 별도 산출물에서 다룬다. 본 문서는 시작 위치만 명시.

- `projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py` — MotionFormer entry class (v1 단일 file 구조, paper 와 동일)
- 주변
  - `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` — TrackFormer + BEV encoder hook
  - `projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py` — OccFormer
  - `projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin.py` — Planner

### Parent 관계

- 본 fork 는 `kingszun/krong-kak` 의 submodule (`submodules/kingszun/uniad`)
- parent 의 짝 submodule: `kingszun/GameFormer` (대체 module source)
- parent 의 Epic / Task 는 `KAK-20` 산하 (`KAK-21` ~ `KAK-26`, `KAK-32`, `KAK-35`)
- 본 fork 의 변경은 본 repo 에서 commit + push 후 parent 의 submodule pointer update
