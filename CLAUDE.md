## UniAD (project CLAUDE.md)

CVPR'23 Best Paper, OpenDriveLab/UniAD — planning-oriented end-to-end autonomous driving. 본 fork 는 reproduction + MotionFormer 분석 용도.

### 작업 컨벤션

- 한국어 + 기술 용어는 영어 (한글 음차 금지)
- bold (`**...**`), emoji 사용 금지
- file: lowercase, segment간 `-`, 단어간 `_`
- 시간 표기: `yy-mm-dd-hh:mm:ss`
- 설계 후 구현 (design-first); 구현 중 설계 변경 필요 시 일단 정지 → 설계 update → review → 재개

### Rules

추가 규칙은 `.claude/rules/` 하위 파일에 정의되어 있다.

- `kck-jira-workflow.md` — jira ticket workflow

### Jira

- project key: `KAK` (https://kingszun.atlassian.net/jira/software/projects/KAK)
- name: kakao
- 본 fork 의 작업은 Epic `KAK-20` (UniAD reproduction) 산하 Story (`KAK-21` ~ `KAK-26`) 에 매핑
- 모든 commit message 에 `[KAK-N]` 포함
- 자세한 규칙은 `.claude/rules/kck-jira-workflow.md` 참조

### 진입 시 먼저 읽을 것

본 fork 는 reproduction 시작 전 단계. 자체 docs (`docs/0X-*.md`) 는 아직 없고 upstream 문서를 그대로 사용한다.

- `README.md` — UniAD 모델 개요, 결과, getting started 안내
- `docs/INSTALL.md` — environment 설치 절차 (mmcv / mmdet / mmdet3d / nuScenes-devkit 의존성)
- `docs/DATA_PREP.md` — nuScenes data prep, info file 생성, motion / occupancy gt 생성
- `docs/TRAIN_EVAL.md` — stage 별 학습 / 평가 흐름, GPU 요구사항

자체 docs 는 KAK-21 ~ KAK-22 진행 시 추가 (`docs/00-overview.md`, `docs/01-environment.md` 등).

### 핵심 사실

- upstream: `OpenDriveLab/UniAD` (CVPR 2023 Best Paper, Award Candidate). 본 fork (`kingszun/UniAD`) 는 license 만 변경 (All Rights Reserved) + 일부 정리 commit.
- 학습 구조 — 2 stage:
  - stage 1: `track_map` (perception: TrackFormer + MapFormer)
  - stage 2: stage1 동결 후 motion (MotionFormer) + occupancy (OccFormer) + planning (Planner) 추가 학습
- 핵심 module 위치 — `projects/mmdet3d_plugin/uniad/`
  - `detectors/uniad_track.py` — TrackFormer 진입점, BEV encoder 결합 지점 (BEVFormer 교체 가능 hook)
  - `dense_heads/motion_head.py` — MotionFormer 정의 (KAK-26 분석 대상)
  - `dense_heads/occ_head.py` — OccFormer
  - `dense_heads/planning_head_plugin.py` — Planner
- BEV encoder 는 BEVFormer 가 default 지만 `bev_embed` + `bev_pos` 만 같으면 다른 encoder 로 교체 가능 (upstream README 명시).
- pretrained checkpoint — OpenDriveLab release (stage1, stage2 base / small).
- dataset — nuScenes (full ~350GB, mini ~4GB). 별도 motion / occupancy gt 생성 단계 필요.

### 작업 상태

현재 미수행 (KAK-20 Epic 시작 시점).

- env setup 미완료 — Dockerfile / venv 미정의 (KAK-22)
- smoke 미수행 — pretrained checkpoint inference 미실행 (KAK-23)
- training 미수행 (KAK-25)
- MotionFormer interface 미정리 (KAK-26)

`KAK-21` (repo 구조 / dependency / license 파악) 부터 진행.

### 자주 쓰는 command

env / smoke / training script 는 KAK-22 ~ KAK-25 진행 중 추가 예정. 현 시점은 upstream 의 README / TRAIN_EVAL.md 절차 그대로 시도.

### 실수 방지

- upstream README 의 알려진 이슈
  - stage1 reproduce 실패 위험 — `loss_past_traj` 와 `img_neck` / `BN` freeze 가 원인. 2023-06-12 bugfix 로 unfreeze + loss 제거 완료된 코드 사용 (현 fork 는 upstream merge 상태이므로 fix 포함).
  - planning 시각화 x-axis 반전 bug 는 2023-08-03 fix 완료.
- mmcv 버전 강결합 — torch 와 mmcv-full 의 호환 matrix 를 INSTALL.md 그대로 따르기 전에 build 시간 / cuda 버전 호환 점검.
- nuScenes mini 와 full 은 info file / occupancy gt 가 별도. mini smoke 후 바로 full 학습 시도하면 data 누락으로 실패.
- `git add -A` / `git add .` 금지 — 변경 file 개별 지정.
- commit 은 user 명시 지시 시에만. 작업 즉시 `git add` 까지만.

### upstream 호환성 patch

현재 없음. KAK-22 environment setup 진행 중 발견되는 patch 는 본 표에 추가 (file / line / 변경 / 사유).

### parent project 관계

- 본 fork 는 `kingszun/krong-kak` (Kakao 채용 과제 작업 공간) 의 submodule 로 사용된다.
- parent 의 과제 목적: UniAD 분석 + MotionFormer module 을 GameFormer (ICCV'23) 로 대체하는 제안.
- parent 의 짝 submodule: `kingszun/GameFormer` (대체 module source).
- parent 에 직접 commit 불가 — fork 자체의 변경은 본 repo 에서 commit + push 후 parent 의 submodule pointer 를 update.
