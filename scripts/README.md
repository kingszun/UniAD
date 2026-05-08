## scripts

UniAD fork 의 host-side / container 운영 wrapper.

### 실행 순서

#### KAK-24: host-side data preparation

cost 절감 목적. nuScenes + UniAD ckpts / info pkls 를 host external mount 에 받아둔 후 RunPod network volume 에 1회 upload — GPU pod 가 download 시간 동안 idle 하는 비용을 회피.

1. `01-download_nuscenes.sh` — nuScenes 본체 (mini, trainval blobs, can_bus, map). user 가 fresh 한 signed URL 을 `scripts/nuscenes_urls.txt` 에 paste 후 실행. URL 은 로그인된 nuScenes 세션에서 7일 유효
2. `02-download_huggingface.sh` — UniAD ckpts 4종 + info pkls 2종 + motion anchors. auth 불필요, 자동
3. `03-link_to_repo.sh` — host external mount → repo `data/`, `ckpts/` symlink. mmcv config 가 기대하는 layout (`docs/DATA_PREP.md` 참조) 으로 노출

#### KAK-22: container build / 운영

4. `04-build_image.sh` — `docker compose build`. `.env` 가 없으면 `.env.example` 복사. 첫 build 는 mmcv 설치까지 약 10~20 분
5. `05-up.sh` — `docker compose up -d`. 기존 container 가 있으면 recreate. 종료는 `docker compose down`
6. `06-smoke_import.sh` — container 안에서 torch + cuda + mmcv / mmdet / mmsegmentation / mmdet3d / projects.mmdet3d_plugin import 검증. 마지막 줄에 `PASS` 출력 시 KAK-22 acceptance 충족

### 환경 변수

| 변수 | default | 의미 |
| --- | --- | --- |
| `HOST_DATA_ROOT` | `/mnt/e/datasets` | host external mount root |
| `NUSCENES_DIR` | `${HOST_DATA_ROOT}/nuscenes` | nuScenes raw |
| `UNIAD_ROOT` | `${HOST_DATA_ROOT}/uniad` | UniAD 전용 (ckpts / infos / others) |
| `DOWNLOAD_TMP` | `${HOST_DATA_ROOT}/_tmp_nuscenes` | 1번 script 의 임시 archive 적재 위치 |
| `NUSCENES_URLS_FILE` | `${SCRIPT_DIR}/nuscenes_urls.txt` | 1번 script 의 URL 목록 |

### 의존성

- `aria2c` 권장 (parallel download, 빠름). 없으면 `wget` 으로 fallback
- `tar`, `unzip`, `ln` (표준)

### 결과 layout

```
${HOST_DATA_ROOT}/
├── nuscenes/                  # raw nuScenes
│   ├── samples/, sweeps/, maps/, can_bus/, lidarseg/
│   └── v1.0-mini/, v1.0-trainval/
└── uniad/
    ├── ckpts/                 # 4 ckpts files
    ├── infos/                 # train/val info pkls
    └── others/                # motion_anchor_infos_mode6.pkl
```

repo (3번 실행 후):

```
submodules/kingszun/uniad/
├── data/                      # gitignored
│   ├── nuscenes -> ${NUSCENES_DIR}
│   ├── infos    -> ${UNIAD_ROOT}/infos
│   └── others   -> ${UNIAD_ROOT}/others
└── ckpts -> ${UNIAD_ROOT}/ckpts  # gitignored
```

### 주의

- `scripts/nuscenes_urls.txt` 는 signed URL (token 포함) 이므로 git tracking 금지. `.gitignore` 에 추가됨
- nuScenes 가입 + dataset license accept 가 선결 — 미동의 시 download 페이지 자체가 보이지 않음
- URL 만료 시 (7일) nuScenes 세션에서 재발급 후 paste 갱신
