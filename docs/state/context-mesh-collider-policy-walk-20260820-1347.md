# Context — mesh collider, policy walk, visualize

> **Phiên:** 2026-08-20 13:47 · nhánh `v3d-rewrite`
> **Phạm vi:** cắt vật thể (OBB-anchored SAM 3) → scene robot MuJoCo → policy đi bộ → thư mục visualize
> **Thay thế:** `context-room-robot.md` (bản cũ, viết trước khi đổi sang mesh collider)

Tài liệu phương pháp: [`../guides/room-to-robot-scene.md`](../guides/room-to-robot-scene.md),
[`../guides/obb-anchored-segmentation.md`](../guides/obb-anchored-segmentation.md)
Danh mục lỗi: [`../reference/bugs-found.md`](../reference/bugs-found.md) B17–B28

---

## 1. Sai lầm lớn nhất của phiên này, và cách sửa

Tôi đã lấy một bản tái dựng 300.000 mặt của DA3 + PGSR — thứ mà người dùng đánh giá
là rất tốt — rồi **vứt đi và thay bằng 6 hộp nguyên thuỷ + 79 hộp từ annotation**.
Sau đó tôi đo chính đống hộp đó và báo cáo như thể đó là căn phòng.

Hệ quả đo được: chỉ **8% sàn đi được** ở clearance 0,45 m. Con số đó không nói gì
về căn phòng thật — nó là hệ quả của việc vẽ hộp thẳng trục quanh ống mềm và dầm
trần, những thứ trong thực tế không chắn lối đi.

**Phép sửa:** chính mesh tái dựng trở thành collider, qua CoACD.

```text
room_300k.glb (300k mặt)
  → decimate 40k mặt (chỉ để phân rã; visual giữ nguyên 300k)
  → CoACD threshold=0.06, max_convex_hull=64
  → 64 khối lồi, 4.444 mặt, 33 giây
  → 64 mesh geom trong MuJoCo, group 0, rgba alpha=0
  + 1 plane dự phòng dưới đáy (mesh sàn có lỗ ở chỗ scan không tới)
```

Cờ `--box-collider` vẫn giữ đường cũ để đối chiếu.

**Bài học chung:** khi tầng dưới cho ra thứ tốt, đừng thay nó bằng thứ xấp xỉ rồi
đo cái xấp xỉ. Người dùng nhìn thấy sự xuống cấp trước khi tôi đo được nó.

---

## 2. Pipeline hiện tại

```bash
make room-scene     # mesh phòng → collider CoACD + robot → MJCF, 2 phép kiểm, render
make room-play      # nạp policy ONNX, mặc định đứng yên
make room-serve     # 3 video + phục vụ http://127.0.0.1:8080
make room-view      # mở MuJoCo viewer
make visualize      # gom mọi artefact + cắt video 7s + 2 trang index
make video-index    # chỉ dựng lại visualize/videos.html
make trim-videos SECS=7
```

Cờ chạy trực tiếp:

```
--box-collider        quay lại hộp nguyên thuỷ (để so sánh)
--max-hulls 64        số khối lồi tối đa
--hull-threshold 0.06 ngưỡng lõm CoACD; thấp hơn = bám sát hơn
--no-furniture        bỏ hộp đồ đạc (chỉ dùng khi đang ở chế độ hộp)
--no-test / --no-render
```

---

## 3. Số đo đã kiểm chứng

### Hình học phòng

```
sàn      z = [-0.204, 0.132]  →  z = 0        nguồn: GT ins=15
trần     z = [-3.731, -2.922] →  z = +3.29    nguồn: GT ins=2
cửa      cao 2.37–2.48 m                       ⇒ k ≈ 1
sàn 8.15 × 8.26 m, cao 3.42 m
mesh: 300.000 mặt, 688 mảnh rời, watertight=False, Euler=−19566
```

### Hai phép kiểm vật lý (với mesh collider)

| | kết quả |
|---|---|
| quả cầu control | 1,5 m → **0,33 m**, bị chặn bởi `collider_013` — **PASS** |
| robot, home pose | thân thấp nhất `z = 0,0`, 41 contact, không chìm — **PASS** |
| spawn | `[3.96, −5.19]`, clearance **3,965 m** |

### Policy `policy_12000.onnx` (đo với collider HỘP — cần chạy lại)

| scene | lệnh | quãng đường | tốc độ | kết quả |
|---|---|---|---|---|
| có đồ đạc | `0 0 0` | 0,05 m | — | **STANDING**, upright 0,9995 |
| không đồ đạc | `0.5 0 0` | 3,85 m | 0,385 m/s | **WALKING** |
| có đồ đạc | `0.5 0 0` | 1,20 m | 0,120 m/s | **BLOCKED** bởi `obs_136_wood` |

### 8 hướng đi thẳng (cũng đo với collider HỘP)

| hướng | quãng đường | tốc độ | upright | kết quả |
|---|---|---|---|---|
| **+315°** | **2,99 m** | 0,249 m/s | 1,00 | **chạy hết 12 s** |
| +270° | 3,48 m | 1,281 m/s | 0,87 | loại — lao người, không phải đi |
| +90° | 1,08 m | 0,392 m/s | 1,00 | chặn ở 2,8 s |
| +0° | 1,20 m | 0,294 m/s | 1,00 | chặn ở 4,1 s |
| +45° | 1,02 m | 0,193 m/s | 0,99 | chặn ở 5,3 s |
| +225° | 0,82 m | 0,210 m/s | 0,97 | chặn ở 3,9 s |
| +135° / +180° | 0,40 / 0,58 m | — | 0,19 / −0,25 | **ngã** |

---

## 4. Hợp đồng policy (`deploy_12000.yaml`)

```text
obs 475 = base_ang_vel 3×5 + projected_gravity 3×5 + velocity_commands 3×5
        + joint_pos_rel 28×5 + joint_vel_rel 28×5 + last_action 28×5 + gait_phase 2×5
actions 28,  step_dt 0.02 s  →  10 substep ở timestep 0.002
```

Ba điều bắt buộc, sai thì **không có gì báo lỗi**:

1. **History cũ trước mới sau**, mỗi term làm phẳng riêng rồi nối theo thứ tự khai báo.
2. **Actuator là mô-men** (`gainprm=1, biasprm=0`) → PD làm ngoài bằng
   `stiffness`/`damping`. *(Đã thử actuator implicit của MuJoCo: kết quả trùng khít.)*
3. **Đứng yên là pha gait riêng** — `mdp.phase` zero cả hai kênh khi `‖cmd‖ < 0.1`.

**Policy không có cảm biến vật cản.** Cả 475 chiều là bản thể + lệnh + gait. Nó mù
trước đồ đạc và sẽ đâm vào — đúng thiết kế.

Giới hạn lệnh: `lin_vel_x [-0.5, 1.4]`, `lin_vel_y [-0.8, 0.8]`, `ang_vel_z [-0.5, 0.5]`.
`ang_vel_z` chỉ 0,5 rad/s → quay một vòng mất 12 giây, nên bộ lái phải **quay tại
chỗ trước rồi mới đi**.

---

## 5. Cắm terrain vào `play.py` của mjlab

Đã kiểm chứng trong venv của repo robot: `nbody 1, ngeom 86, **nq 0**`, collider ở
**group 0**, `TerrainEntityCfg(spec_fn=...)` nhận.

```python
import sys; sys.path.insert(0, ".../OUTPUT_ROOM")
from room_terrain import spec_fn, SPAWN_XY
from mjlab.terrains import TerrainEntityCfg

env_cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane", spec_fn=spec_fn)
env_cfg.scene.entities["robot"].init_state.pos = (*SPAWN_XY, 0.841)
```

- `nq = 0` bắt buộc — terrain mang free body sẽ làm lệch mọi địa chỉ `qpos`.
- **group 0** vì `terrain_scan`/`foot_height_scan` lọc `include_geom_groups=(0,)`.
- **Phải đổi spawn** — mặc định mjlab là gốc toạ độ, chỗ đó nằm trong máy.
- Nên tắt `out_of_terrain_bounds` và `terrain_levels` (giả định terrain lưới ô).
- Mô hình robot khớp: mjlab và scene này đều **82,6 kg**, `nq 35`.
- **Warp trên máy này chỉ có CPU** (`CUDA not enabled`). Người dùng xác nhận máy họ chạy được.

---

## 6. Nguồn dữ liệu — không phải video

`00777c41d4` là scene benchmark **ScanNet++**: **261 ảnh DSLR**
`DSC00850.JPG` → `DSC01111.JPG`, 1752×1168, 121 MB.

Bằng chứng: `cameras_optimised.json` có 261 camera, cái đầu tên `DSC00850`; mọi
`object.json` trích frame `DSC00973`, `DSC01015`.

```
outputs-full-factory/delivery/INPUT/scannetppv2/data/00777c41d4/dslr/resized_undistorted_images/
```

Các video trong `input/` là **scene khác**, chưa sinh ra output nào đã giao:
`small_factory_final.mp4` (57,7 s, 1080p), `factory_scene_4k/.../video.mp4`
(56,1 s, 4K HEVC), 5 clip `full-apartment/`.

---

## 7. Mọi lỗi im lặng đã gặp

Không cái nào ném exception. Mỗi cái tạo ra một kết quả chạy được nhưng sai.

| # | Lỗi | Triệu chứng đo được |
|---|---|---|
| B17 | Cắt thừa + thiếu cùng lúc | text-prompt SAM 3 mất chân bàn; `A ∪ B` rembg thêm 3 hộp đen |
| B18 | Gate `leak` fail open | không ứng viên nào đạt → xếp theo `fill`; mask nuốt cả ảnh có `fill=1.0` thắng |
| B19 | `leak` vô nghĩa khi OBB phủ khung | hull 99,9% ảnh, mask 99,7%, `leak=0.001` |
| B20 | Trừ hàng xóm theo chồng lấn 2D | mask bàn 210.914 → 7.261 px, 4 view về 0 |
| B21 | Laplacian xếp vật ở xa là nét nhất | view xa 455 vs cận cảnh 35 |
| B22 | Slot MV gán hệ world | `view_embed` học được nên nhãn slot là conditioning thật |
| B23 | PNG nền trắng | `.convert("RGBA")` → alpha=255; model coi cả ô 518 là vật thể |
| B24 | Trọng lực ngược chiều | Z tái dựng chỉ xuống |
| B25 | Mesh phòng làm collider | bao lồi = khối đặc, robot bị đẩy ra ngoài |
| B26 | Keyframe ghép tay | 2 freejoint, `nq=42` ≠ 35 |
| B27 | Texture | xatlas treo; chiếu hộp thì sàn và mặt bàn tranh texel |
| B28 | Spawn ở tâm phòng | trong `obs_55_machine`: 88 contact, xuyên 0,57 m |
| — | Verdict thiếu quãng đường | in "STANDING" cho lệnh đi tới |
| — | Tốc độ trung bình toàn cuộc | gộp 6 s bị chặn → đo vật cản chứ không đo policy |
| — | Chọn hướng tốt nhất theo quãng đường | +270° thắng với 1,281 m/s (lệnh 0,5) — lao người |
| — | Trim video bị hoàn tác | collector copy bản gốc **sau khi** trim |
| — | Control giả định sàn phẳng | với mesh thật, cầu dừng trên mặt bàn z=0,33 → FAIL sai |

---

## 8. Bất biến

1. **Mesh tái dựng là collider.** Không thay bằng hộp trừ khi đang cố tình so sánh.
2. **Xoay 180° quanh X, không lật Z.** `det` phải `+1`.
3. **Mesh visual luôn `contype=0`**; hull luôn `rgba alpha=0` (chúng trùng chỗ nhau).
4. **Quả cầu control luôn có** — tách "collider hỏng" khỏi "robot ngã".
5. **Keyframe ghi sau khi compile**, theo `jnt_qposadr` thật.
6. **Không include `scene.xml` của robot** — nó mang sàn thứ hai.
7. **Độ cao sàn từ GT hoặc phân vị 1%**, không `min()`.
8. **Spawn phải tìm và ghi lại clearance.**
9. **Chạy control PD trước khi tin policy.**
10. **Observation dựng từ `deploy_*.yaml`**, không hardcode.
11. **Verdict phải đối chiếu LỆNH** — đứng và đi cần hai tiêu chí; phân biệt "bị
    chặn" với "không chịu đi"; loại run có tốc độ ≫ lệnh.
12. **Trim video sau khi copy**, trước khi dựng index.

---

## 9. Thư mục visualize

```
visualize/            129 MB, 38 artefact, 20 video (mọi video ≤ 7 s)
├── index.html            mọi thứ theo stage
├── videos.html           21 video một trang, kèm thời lượng + verdict
├── 1_reconstruction/     7 video + 6 ảnh
├── 2_object_crops/       contact sheet 37 object
├── 3_assembled_scene/    5 render
├── 4_robot_scene/        room_robot_check.png
├── 5_policy/             walk_4s, walk_tour, stills
├── 6_headings/           8 video, mỗi hướng một cái
├── 7_objects_3d/         3 object: model-viewer + turntable + GLB
└── reports/              5 JSON
```

`obj_0052_machine` đã bị bỏ theo yêu cầu.

**Object mesh render xám vì chúng xám thật** — mọi đỉnh là `[102,102,102]`.
Hunyuan3D sinh hình dạng, không sinh diện mạo.

---

## 10. Còn treo — theo thứ tự tác động

1. **Occlusion-only completion (việc chính chưa làm).** Người dùng yêu cầu: cắt vật
   thể ra, chỉ điền phần bị che, giữ nguyên bề mặt đã quan sát. Hiện `OUTPUT_G` là
   38 mesh Hunyuan3D **thay thế toàn bộ** — đúng kiểu thay thế mà người dùng phản đối.
2. **Chạy lại walk + headings với mesh collider.** Mọi số ở mục 3 đo với hộp, giờ đã lỗi thời.
3. **Stage [2] calibrate mét chưa chạy.** `OUTPUT_B2/` chưa có,
   `00_metric_calibration.py` chưa viết. Có sẵn 223 OBB ground-truth để giải `k`.
4. **Cắt tốc độ ở lần chạm đầu tiên** thay vì trung bình toàn cuộc.
5. **DA3 có thể vẫn chạy depth ở độ phân giải thấp rồi upsample** —
   xem `docs/reference/audit-prompt-da3-resolution.md`. Nếu đúng, chất lượng chưa chạm trần.

---

## 11. File

```text
scripts/sim/
├── make_room_scene.py     mesh → CoACD collider + MJCF + terrain export, 2 kiểm, render
├── play_policy.py         chạy policy ONNX, phán quyết theo lệnh
├── walk_headings.py       đi thẳng nhiều hướng, so sánh
├── walk_tour.py           A* + pure pursuit (chưa dùng được với collider hộp)
└── serve_room.py          render 3 video + phục vụ HTTP

scripts/tools/
├── collect_visualize.py   gom artefact + trim + 2 index
├── make_video_index.py    trang video
├── trim_videos.py         cắt về N giây (re-encode, không stream copy)
└── visualize_objects.py   render object 3D qua MuJoCo (trimesh cần pyglet, không có)

scripts/adapter/
├── 03_resegment_obb_sam3.py          GPU, segment lại neo OBB
└── 02_quality_gate_and_adapter_a.py  bỏ rembg, đọc masks_v2, xuất RGBA
```

Phụ thuộc thêm phiên này: `onnxruntime`, `coacd` (đã khai báo trong `requirements.txt`).
