## UniAD (project CLAUDE.md)

CVPR'23 Best Paper, OpenDriveLab/UniAD — planning-oriented end-to-end autonomous driving. 본 fork 는 reproduction + MotionFormer 분석 용도, upstream `main` (v1, paper 시점 release) 기준.

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

- `docs/00-overview.md` — fork 자체 정리 (project 목표, repo 구조, dependency, entry point, GPU / data 용량, host data layout, MotionFormer 위치 등). KAK-21 산출물.
- `docs/01-motionformer_interface.md` — MotionFormer class / forward / input / output / loss / 데이터 흐름 / 결합 pain point. KAK-26 산출물.
- `README.md` — UniAD 모델 개요, getting started 안내
- `docs/INSTALL.md` — environment 설치 절차 (cuda 11.1 / torch 1.9.1+cu111 / mmcv-full 1.4.0 / mmdet 2.14.0 / mmsegmentation 0.14.1 / mmdet3d v0.17.1 from source)
- `docs/DATA_PREP.md` — nuScenes data prep (motion / occupancy gt 생성 포함)
- `docs/TRAIN_EVAL.md` — stage 별 학습 / 평가 흐름, GPU 요구사항

### 핵심 사실

- upstream: `OpenDriveLab/UniAD` `main` branch (v1, CVPR'23 release). 본 fork (`kingszun/UniAD`) 는 v1 base + LICENSE 변경 (All Rights Reserved) + 본 docs / scripts / Dockerfile.
- 이전 시도: v2.0 base 로 전환했으나 reproducibility 이슈 (Dockerfile stale, mmcv 1.6.1 wheel 부재, mmdet3d 1.0.0rc6 source build 차단) 누적되어 v1 latest 로 revert (KAK-35).
- archive: `v1-archive` / `v1-archive-26-05-08`, `v2-archive` / `v2-archive-26-05-08` (둘 다 fork remote 에 push).
- v1 framework — `cuda 11.1`, `python 3.8`, `torch 1.9.1+cu111`, `mmcv-full 1.4.0`, `mmdet 2.14.0`, `mmsegmentation 0.14.1`, `mmdet3d v0.17.1` (from source).
- 학습 구조 — 2 stage:
  - stage 1: `track_map` (perception: TrackFormer + MapFormer)
  - stage 2: stage1 동결 후 motion (MotionFormer) + occupancy (OccFormer) + planning (Planner) 추가 학습
- 핵심 module 위치 — `projects/mmdet3d_plugin/uniad/`
  - `detectors/uniad_track.py` — TrackFormer 진입점, BEV encoder 결합 지점 (BEVFormer 교체 가능 hook: `bev_embed` + `bev_pos`)
  - `dense_heads/motion_head.py` — MotionFormer 정의 (KAK-26 분석 대상). v1 에서는 단일 file (paper 와 동일 구조).
  - `dense_heads/occ_head.py` — OccFormer
  - `dense_heads/planning_head_plugin.py` — Planner
- pretrained checkpoint — GitHub Releases (OpenDriveLab/UniAD v1.0 / v1.0.1, zhiqi-li/storage v1.0).
- dataset — nuScenes V1.0.

### 작업 상태

- KAK-32 fork base v2.0 전환 완료 → KAK-35 로 revert (v2-archive 에 보존)
- KAK-35 revert v2.0 → v1 latest 진행 중 (현재 commit batch)
- KAK-21 repo 구조 / dependency / license 파악 완료 → `docs/00-overview.md` (v1 기준 재작성)
- KAK-22 env setup 완료 — `docker/Dockerfile` (v1 stack, uv venv + 3 patches), `compose.yaml`, `.env.example`, `scripts/04-06` + local 3060 import 검증 PASS
- KAK-23 inference smoke 미수행
- KAK-24 nuScenes full data prep 진행 중 (`scripts/01-03` 으로 host download. v1 ckpt URL 로 갱신)
- KAK-25 training smoke 미수행
- KAK-26 MotionFormer interface 정리 완료 → `docs/01-motionformer_interface.md`

### 자주 쓰는 command

#### host data preparation (KAK-24)

상세는 `scripts/README.md` 참조. 순서

```
cp scripts/nuscenes_urls_example.txt scripts/nuscenes_urls.txt
# scripts/nuscenes_urls.txt 에 nuScenes 세션의 signed URL paste
bash scripts/01-download_nuscenes.sh
bash scripts/02-download_huggingface.sh
bash scripts/03-link_to_repo.sh
```

#### env setup / 검증 (KAK-22)

```
cp .env.example .env
bash scripts/04-build_image.sh
bash scripts/05-up.sh
bash scripts/06-smoke_import.sh
```

container shell

```
docker compose exec uniad bash
```

GPU 상태

```
docker compose exec uniad nvidia-smi
```

train / eval 은 KAK-23 / KAK-25 진행 시 추가.

### 실수 방지

- `git add -A` / `git add .` 금지 — 변경 file 개별 지정.
- commit 은 user 명시 지시 시에만. 작업 즉시 `git add` 까지만.
- nuScenes mini 와 full 은 info file 이 별도 (mini 전용 info 가 v1 release 에 없음 — full trainval info 만 제공). mini smoke 는 mini metadata 기반 자체 inference 만 가능.
- v1 stack 은 cu111 — 4090(8.9) / H100(9.0) 미지원. 해당 GPU 가 필요하면 v2 stack 로 전환 (`v2-archive` branch 참고).
- mmdet3d v0.17.1 은 source build 필요 (Dockerfile 에 git clone + editable install 포함).

### upstream 호환성 patch

| file | 변경 | 사유 |
| --- | --- | --- |
| `docker/Dockerfile` | upstream Dockerfile (apt-installed python3.8 + system pip + global env) → fork Dockerfile (uv venv + sshd + dual-mode entrypoint, gameformer 패턴) (KAK-22) | RunPod cloud-native 운영 (sshd, isolated venv, image 안에 코드 박지 않고 host bind-mount) 을 위해 재작성. 기능적으로 동일한 stack. |
| Dockerfile build deps `scikit-image==0.20.0` → `0.19.3` (KAK-22) | 0.20.0 은 `numpy>=1.21.1` 요구하지만 `requirements.txt` 가 `numpy==1.20.0` 로 pin. upstream Dockerfile 은 pip 의 느슨한 resolver 로 우회했으나 uv 는 strict — 호환되는 0.19.3 으로 통일 |
| Dockerfile mmdet3d v0.17.1 `pip install -e .` → `pip install .` (non-editable) (KAK-22) | `setuptools 59.5.0` 은 PEP 660 의 `build_editable` hook 부재. uv 의 editable 설치가 실패. non-editable 은 venv site-packages 로 복사 — upstream Dockerfile 의 `cp -r mmdet3d` 와 동일 효과 |
| Dockerfile build deps 에 `Pillow<10` 추가 (KAK-22) | Pillow 10.x 부터 `PIL._typing` 이 `numpy.typing.NDArray` 호출. numpy 1.20 에는 부재 → import 시 `AttributeError`. torchvision 0.10.1 이 Pillow pin 없이 latest 가져와 발생 |

### parent project 관계

- 본 fork 는 `kingszun/krong-kak` (Kakao 채용 과제 작업 공간) 의 submodule (`submodules/kingszun/uniad`)
- parent 의 짝 submodule: `kingszun/GameFormer` (대체 module source)
- parent 에 직접 commit 불가 — fork 자체의 변경은 본 repo 에서 commit + push 후 parent 의 submodule pointer 를 update.
