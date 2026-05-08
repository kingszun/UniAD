## MotionFormer interface (KAK-26)

UniAD v1 의 MotionFormer module 정의 위치 / 입출력 / 학습 loss / 주변 module 결합 지점을 file_path:line_number 단위로 정리. parent project 의 분석 Epic (`KAK-27` 비교, `KAK-28` 대체 검토) 의 입력 자료.

### 1. Class 정의

| 항목 | 값 |
| --- | --- |
| class | `MotionHead` |
| parent | `BaseMotionHead` (`projects/mmdet3d_plugin/uniad/dense_heads/motion_head_plugin/base_motion_head.py`) |
| 정의 file | `projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py:21-43` |
| `__init__` | `motion_head.py:43-86` |

핵심 constructor arg

| arg | default / 의미 |
| --- | --- |
| `predict_steps` | 12 (10 Hz, 1.2s horizon) |
| `transformerlayers` | `MotionTransformerDecoder` config |
| `bev_h`, `bev_w` | 30 (기본), config 에서 200 으로 override |
| `embed_dims` | 256 |
| `num_anchor` | 6 (mode 수 / class group) |
| `det_layer_num` | 6 |
| `group_id_list` | class → mode group 매핑 (config: `[[0,1,2,3,4],[6,7],[8],[5,9]]`) |
| `pc_range` | `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` |
| `anchor_info_path` | `data/others/motion_anchor_infos_mode6.pkl` |
| `vehicle_id_list` | `[0,1,2,3,4,6,7]` (post-hoc filter) |
| `loss_traj` | `TrajLoss` config (config: `use_variance=True, cls_loss_weight=0.5, nll_loss_weight=0.5, loss_weight_minfde=0.25`) |

### 2. forward signature

`motion_head.py:211-236`, `@auto_fp16` decorator.

```
def forward(self,
            bev_embed,          # (h*w, B, D) = (40000, 1, 256) for 200x200 BEV
            track_query,        # (B, A_track, D) = (1, 900, 256), TrackFormer agent embeddings
            lane_query,         # (B, M, D), MapFormer 의 lane / centerline embeddings
            lane_query_pos,     # (B, M, D)
            track_bbox_results) # List[Tuple[Tensor]] per frame
```

return — `Dict[str, Tensor]`

| key | shape | 의미 |
| --- | --- | --- |
| `all_traj_scores` | `(num_dec, B, A, P)` = (3, 1, A, 6) | log-softmax mode prob |
| `all_traj_preds` | `(num_dec, B, A, P, T, 5)` = (3, 1, A, 6, 12, 5) | (Δx, Δy, σx, σy, ρ) per timestep |
| `valid_traj_masks` | `(B, A)` | 유효성 mask (현 구현은 전부 True) |
| `traj_query` | `(num_dec, B, A, P, D)` = (3, 1, A, 6, 256) | decoder layer 별 intermediate query |
| `track_query` | `(B, A, D)` | pass-through |
| `track_query_pos` | `(B, A, D)` | agent center 의 positional embedding |
| `sdc_traj_query` | `(1, P, D)` | SDC (ego) trajectory query — Planner 입력 |
| `sdc_track_query` | `(D,)` | SDC tracking embedding — Planner 입력 |

### 3. Sub-component

| component | file:line | 역할 |
| --- | --- | --- |
| `MotionTransformerDecoder` | `motion_head_plugin/modules.py:19-172` | 3-layer iterative decoder |
| `IntentionInteraction` | `modules.py:33` | mode 간 self-attention (anchor-anchor) |
| `TrackAgentInteraction` | `modules.py:34-35` | agent-agent attention (TransformerDecoderLayer × 6 / decoder layer) |
| `MapInteraction` | `modules.py:36-37` | agent-lane attention (TransformerDecoderLayer × 3) |
| BEV deformable attention | `motion_deformable_attn.py:25-172` (`MotionTransformerAttentionLayer`) | trajectory reference point 기반 BEV sampling |
| `agent_level_embedding_layer` | `base_motion_head.py:68-87` | agent-frame anchor embed |
| `scene_level_ego_embedding_layer` | `base_motion_head.py:68-87` | ego-frame transformed anchor embed |
| `scene_level_offset_embedding_layer` | `base_motion_head.py:68-87` | rotation-only transformed anchor embed |
| `boxes_query_embedding_layer` | `base_motion_head.py:68-87` | agent center position embed |
| `traj_cls_branches` | `base_motion_head.py:89-116` | 3-layer MLP × num_dec, output `(B, A, P, 1)` |
| `traj_reg_branches` | `base_motion_head.py:89-116` | 3-layer MLP × num_dec, output `(B, A, P, T*5)` |
| `learnable_motion_query_embedding` | `base_motion_head.py:58-59` | `nn.Embedding(num_anchor*num_groups, D)` = `(24, 256)` |

### 4. Input 상세

#### TrackFormer 산출 (`projects/mmdet3d_plugin/uniad/detectors/uniad_track.py`)

| input | shape | dtype | source |
| --- | --- | --- | --- |
| `track_query` | `(B, 1, A, D)` → squeeze `(B, A, D)` | float32 | `uniad_track.py:116` — `outs_track['track_query_embeddings'][None, None, ...]` |
| `track_bbox_results` | `List[Tuple[bbox(A,10), scores(A,), labels(A,), idx, mask)]` | float32 / int64 | `uniad_track.py:118, 128-131` |

#### BEVFormer 산출

| input | shape | dtype | source |
| --- | --- | --- | --- |
| `bev_embed` | `(h*w, B, D)` = (40000, 1, 256) | float32 | `uniad_track.py:345, 352` — `get_bev_features()` permute |
| `bev_pos` | shape 동일 | float32 | (현 motion_head 는 직접 사용 안 함, occ/planner 가 사용) |

#### MapFormer (seg_head) 산출

| input | shape | dtype | source |
| --- | --- | --- | --- |
| `lane_query` | `(B, M, D)`, M ≈ 1000-2000 | float32 | `motion_head.py:133` — `outs_seg['args_tuple'][3]` |
| `lane_query_pos` | `(B, M, D)` | float32 | `motion_head.py:133` — `outs_seg['args_tuple'][5]` |

#### 내부 생성 (motion_head.py:240-294)

| element | shape | source |
| --- | --- | --- |
| kmeans anchors | `(num_groups, num_anchor, T, 2)` = (4, 6, 12, 2) | `_load_anchors()` (`base_motion_head.py:33-45`) ← `data/others/motion_anchor_infos_mode6.pkl` |
| scene-level ego anchors | `(B, A, G, P, T, 2)` | `motion_head.py:259` — `anchor_coordinate_transform(..., with_translation_transform=True)` |
| scene-level offset anchors | `(B, A, G, P, T, 2)` | `motion_head.py:260` — rotation-only |
| `init_reference` (reference traj) | `(B, A, P, T, 2)` | `motion_head.py:296` — `group_mode_query_pos()` (line 281) class label 별 선택 |
| `track_query_pos` | `(B, A, D)` | `motion_head.py:248` — `pos2posemb2d()` + `boxes_query_embedding_layer` |

### 5. Output 상세 + 다운스트림 소비처

| output | shape | dtype | consumer |
| --- | --- | --- | --- |
| `all_traj_scores` | `(num_dec, B, A, P)` = (3, 1, A, 6) | log-softmax | `TrajLoss` + `planning_head` (mode 선택) |
| `all_traj_preds` | `(num_dec, B, A, P, T, 5)` | float32 | `(Δx, Δy, σx, σy, ρ)` per timestep — `planning_head` 가 trajectory filter 에 사용 |
| `traj_query` | `(num_dec, B, A, P, D)` | float32 | layer 별 intermediate. `motion_head.py:152, 155` 에서 SDC 와 agent 분리 |
| `sdc_traj_query` | `(1, P, D)` | float32 | `planning_head.py:118` |
| `sdc_track_query` | `(D,)` | float32 | `planning_head.py:119` |

motion_head 는 `vehicle_id_list` ([0,1,2,3,4,6,7]) 만 통과시키므로 OccFormer / Planner 는 vehicle agent 만 본다 (`motion_head.py:139-149, 160`). OccFormer 는 motion output 을 직접 받지 않고 BEV context 로 동작.

### 6. Loss 구성

`projects/mmdet3d_plugin/losses/traj_loss.py` 의 `TrajLoss` (line 16-95).

forward (`traj_loss.py:41-95`)

| 구성 | 정의 file:line | 식 / reduction |
| --- | --- | --- |
| mode classification (`l_class`) | `traj_loss.py:87` | `NLL = -log(p_best)`, p_best = best-matching mode 의 probability |
| trajectory regression (`l_reg`) | `traj_loss.py:82` (variance) / `traj_loss.py:84` (min-ADE) | `use_variance=True` 시 5-param Gaussian NLL (`traj_loss.py:123-165`) |
| min-ADE (`l_minade`) | `traj_loss.py:97-121` | best-matching mode 의 timestep 평균 euclidean |
| min-FDE (`l_minfde`) | `traj_loss.py:167-200` | final timestep euclidean |
| miss-rate (`l_mr`) | `traj_loss.py:203-233` | max(distance) > 2.0m 비율 |

Gaussian NLL 식 (`traj_loss.py:123-165`)

```
nll = 0.5 / (1 - ρ²) * [σx²(x - μx)² + σy²(y - μy)² - 2ρσxσy(x - μx)(y - μy)] - log(σx * σy)
```

config (`projects/configs/stage2_e2e/base_e2e.py:408-413`)

```
loss_traj=dict(
    type='TrajLoss',
    use_variance=True,
    cls_loss_weight=0.5,
    nll_loss_weight=0.5,
    loss_weight_minade=0.,
    loss_weight_minfde=0.25)
```

motion_head 의 layer 별 loss aggregation: `motion_head.py:419-422` (per-layer), `motion_head.py:425-430` (final + intermediate, `d0.loss_traj` ~ `d2.loss_traj`).

### 7. 데이터 흐름

stage 1 (perception, freeze 후 stage 2 동안 update 없음)
1. images `(B, N_cam, 3, H, W)` → BEVFormer encoder → `bev_embed (h*w, B, D)`, `bev_pos`
2. images → TrackFormer detector → `track_query (B, A, D)`, `track_bbox_results`
3. images → MapFormer (seg_head) → `lane_query (B, M, D)`, `lane_query_pos`

stage 2 (motion / occ / planning)

4. `_load_anchors()` → kmeans anchors `(G, P, T, 2)`. class label 별 group 선택 후 ego / offset frame 으로 transform → scene-level anchors `(B, A, G, P, T, 2)`
5. agent center → `boxes_query_embedding_layer` → `track_query_pos (B, A, D)`
6. anchor embedding 3종 (agent-level, ego-level, offset-level) 결합 → mode expansion `(B, A, D)` → `(B, A, P, D)`
7. `MotionTransformerDecoder` 3 layer iteration (각 layer 마다)
   - intention interaction (anchor-anchor)
   - track-agent interaction (cross-attn to `track_query`)
   - map interaction (cross-attn to `lane_query`)
   - BEV deformable attention (reference traj 위치에서 sampling)
   - 4 attention 융합 → `traj_reg_branch` → trajectory delta 예측 → reference traj update
8. final layer: `traj_cls_branches[-1]` → mode score, `traj_reg_branches[-1]` → trajectory params
9. `TrajLoss` 적용 (gt 매칭은 TrackFormer 의 `matched_idxes` 재사용)
10. SDC (ego) trajectory query 분리 → `sdc_traj_query`, `sdc_track_query` 가 `planning_head.py:118-119` 입력
11. agent trajectory 는 vehicle filter 통과 후 occ_head / planning 의 vehicle context 로 사용

### 8. 결합 pain point — GameFormer 대체 시 고려 사항

#### anchor system

- mode 가 class-group-specific (4 groups × 6 modes). agent frame anchor 를 ego / offset frame 으로 dynamic transform.
- 대체 시 보존 필요: anchor pkl 의 coordinate frame 의미 + class-group 매핑.
- GameFormer 는 mode 수 (M=6) 를 그대로 사용하지만 class group 개념 부재 — 평행 매핑 또는 단일 group 으로 단순화 결정 필요.

#### query 구조 (4D)

- track_query 가 mode 차원으로 expand: `(B, A, D)` → `(B, A, P, D)`.
- 3종 anchor embedding 이 별도 fusion 됨.
- GameFormer 의 level-K decoder 는 agent 별 다중 mode 출력하지만 4D query layout 이 자연스럽지 않음 — adapter 필요.

#### deformable attention 의 dynamic reference

- BEV feature 를 trajectory reference point 에서 sampling. reference 는 3 layer iteration 동안 update.
- GameFormer 는 ego-centric BEV-free input (agent history + map polylines) 사용 — BEV deformable attention 자체가 unmatched.
- 대체 시 BEV feature 를 agent / map / ego state 형태로 변환하거나, BEV sampling stage 만 유지하고 GameFormer 는 그 결과를 받는 hybrid 가능.

#### vehicle post-hoc filter

- `vehicle_id_list` 로 vehicle 만 통과. OccFormer / Planner 는 vehicle agent 만 본다.
- GameFormer 출력에도 동일 filter 적용 필요. 또는 filter 단계에서 GameFormer 의 score 를 사용해 추가 ranking.

#### static + dynamic 혼합

- static intention embedding (`modules.py:101`) 은 3 layer 동안 constant.
- dynamic ego embedding 은 trajectory 갱신마다 재계산 (`modules.py:154-167`).
- GameFormer 의 InteractionDecoder 는 level k 마다 trajectory 를 input 으로 받아 갱신 — 의미적으로는 유사하나 input/output 형식 mapping 이 필요.

#### loss layer-별 weight

- decoder layer 3개에 대해 `d0`, `d1`, final 각각 `TrajLoss` 적용.
- GameFormer 의 level 수 (K) 가 다를 경우 loss config 수정 필요.

#### frozen BEV encoder

- stage 2 에서 BEVFormer 는 freeze (`base_e2e.py:109` `freeze_bev_encoder=True`).
- BEV embedding shape / 의미가 바뀌면 motion_head 재학습 필요. GameFormer 는 BEV 사용 안 하므로 이 종속성은 사라짐.

### 9. 요약

대체 설계 시 결정 변수

1. anchor 의 class-group 구조를 유지할지 (preserve) vs 평탄화할지 (flatten)
2. BEV deformable attention 출력 → GameFormer 입력 변환 pathway
3. mode expansion 의 layout 을 유지할지 (`(B, A, P, D)`) vs GameFormer 의 native `(B, A, M, T, 4)` 로 매핑할지
4. layer-별 loss aggregation 을 그대로 둘지 (level k 별로) vs final-only 로 단순화할지
5. vehicle filter 의 위치 (motion 출력 vs occ/planner 입력)

이 결정들은 `KAK-28` 대체 가능성 검토 ticket 에서 다룬다.
