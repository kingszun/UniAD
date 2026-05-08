## UniAD (project CLAUDE.md)

CVPR'23 Best Paper, OpenDriveLab/UniAD — planning-oriented end-to-end autonomous driving. 본 fork 는 reproduction + MotionFormer 분석 용도, upstream `v2.0` (2025-10-29 release, `mmdet3d 1.x` + `torch 2.0.1+cu118` stack) 기준.

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
- 본 fork 의 작업은 Epic `KAK-20` (UniAD reproduction) 산하 ticket 에 매핑
- default issue type 은 Task (user-facing deliverable 일 때만 Story — `.claude/rules/kck-jira-workflow.md` 의 issue type 선택 가이드 참조)
- 모든 commit message 에 `[KAK-N]` 포함

### 진입 시 먼저 읽을 것

본 fork 는 reproduction 시작 전 단계. 자체 docs (`docs/0X-*.md`) 는 아직 없고 upstream v2.0 문서를 그대로 사용한다.

- `README.md` — UniAD 모델 개요, v2.0 release notes, getting started 안내
- `docs/INSTALL.md` — environment 설치 절차 (`mmdet3d 1.0.0rc6` / `torch 2.0.1+cu118` / cuda 11.8 기준)
- `docs/DATA_PREP.md` — nuScenes data prep (motion / occupancy gt 생성 포함). nuPlan / NAVSIM 은 v2.0 TODO 상태.
- `docs/TRAIN_EVAL.md` — stage 별 학습 / 평가 흐름, GPU 요구사항

### 핵심 사실

- upstream: `OpenDriveLab/UniAD` `v2.0` (default branch, 2025-10-29 release). 본 fork (`kingszun/UniAD`) 는 v2.0 base + LICENSE 변경 (All Rights Reserved) + 본 docs.
- v1 history 는 `v1-archive` branch + `v1-archive-26-05-08` tag 로 보존 (KAK-32).
- v2.0 framework — `mmdet3d 1.0.0rc6`, `torch 2.0.1+cu118`. paper 와 architecture 는 동일, framework 만 modernize.
- 학습 구조 — 2 stage:
  - stage 1: `track_map` (perception: TrackFormer + MapFormer)
  - stage 2: stage1 동결 후 motion (MotionFormer) + occupancy (OccFormer) + planning (Planner) 추가 학습
- 핵심 module 위치 — `projects/mmdet3d_plugin/uniad/`
  - `detectors/uniad_track.py` — TrackFormer 진입점, BEV encoder 결합 지점 (BEVFormer 교체 가능 hook: `bev_embed` + `bev_pos`)
  - `dense_heads/motion_head.py` + `dense_heads/motion_head_plugin/` — MotionFormer 정의 (KAK-26 분석 대상). v2.0 에서 plugin 디렉토리로 분리됨 (`base_motion_head.py`, `motion_deformable_attn.py`, `motion_optimization.py`, `motion_utils.py`).
  - `dense_heads/occ_head.py` — OccFormer
  - `dense_heads/planning_head_plugin.py` — Planner
- pretrained checkpoint — OpenDriveLab release (stage1, stage2 base/small).
- dataset — nuScenes (full ~350GB, mini ~4GB). v2.0 의 nuPlan / NAVSIM 은 tools 미release (2025Q2 ETA).

### 작업 상태

- KAK-32 fork base v2.0 전환 완료
- KAK-21 repo 구조 / dependency / license 파악 (진행 예정)
- KAK-22 env setup 미완료 (v2.0 의 `mmdet3d 1.x` + `torch 2.0.1` 기준으로 진행 — v1 대비 dep build 비용 절감 기대)
- KAK-23 inference smoke 미수행
- KAK-24 nuScenes full data prep 미수행
- KAK-25 training smoke 미수행
- KAK-26 MotionFormer interface 미정리 (`motion_head_plugin/` 분리 구조 반영 필요)

### 자주 쓰는 command

env / smoke / training script 는 KAK-22 ~ KAK-25 진행 중 추가 예정. 현 시점은 upstream 의 README / TRAIN_EVAL.md 절차 그대로 시도.

### 실수 방지

- `git add -A` / `git add .` 금지 — 변경 file 개별 지정.
- commit 은 user 명시 지시 시에만. 작업 즉시 `git add` 까지만.
- nuScenes mini 와 full 은 info file / occupancy gt 가 별도. mini smoke 후 바로 full 학습 시도하면 data 누락으로 실패.
- v2.0 release notes (`README.md` 의 `2025/10/29` UniAD 2.0 Release section + linked PR #251) 의 framework breaking change 는 INSTALL.md 따르기 전에 확인.
- v1 docs / v1 코드 reference 가 필요한 경우 `v1-archive` branch 또는 `v1-archive-26-05-08` tag 사용.

### upstream 호환성 patch

현재 없음. KAK-22 environment setup 진행 중 발견되는 patch 는 본 표에 추가 (file / line / 변경 / 사유).

### parent project 관계

- 본 fork 는 `kingszun/krong-kak` (Kakao 채용 과제 작업 공간) 의 submodule 로 사용된다.
- parent 의 과제 목적: UniAD 분석 + MotionFormer module 을 GameFormer (ICCV'23) 로 대체하는 제안.
- parent 의 짝 submodule: `kingszun/GameFormer` (대체 module source).
- parent 에 직접 commit 불가 — fork 자체의 변경은 본 repo 에서 commit + push 후 parent 의 submodule pointer 를 update.
