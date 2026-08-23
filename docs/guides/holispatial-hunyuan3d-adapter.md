# Holi-Spatial → Adapter → Hunyuan3D: Pipeline chia nhỏ theo từng output

> **Mục tiêu:** giữ nguyên kiến trúc Holi-Spatial và Hunyuan3D, chỉ thêm adapter để hoàn thiện các vật thể bị thiếu mặt sau.
>
> **Nguyên tắc:**  
> - Holi-Spatial: **không sửa model**.  
> - Hunyuan3D: **không sửa model**.  
> - Chỉ viết **Adapter A** (Holi → Hunyuan) và **Adapter B** (Hunyuan → world coordinate của Holi).  
> - Version đầu tiên **không convert mesh trở lại 3DGS**. Final scene là **hybrid: Holi 3DGS + completed object meshes**.
> - **Scale được khoá một lần, ở cấp scene, trước mọi thứ khác** (Stage [2]). Không object nào được scale riêng lẻ theo phỏng đoán.
> - Đích cuối không phải là ảnh render đẹp, mà là **scene chạy được trong physics engine và robot dùng được** (Stage [10]–[12]).

---

# 0. Toàn bộ pipeline ở mức cao

Pipeline đầy đủ gồm **12 stage**, chia làm ba khối:

- **Stage [1]–[2] — Scene thật.** Holi-Spatial dựng scene, rồi *một lần duy nhất* khoá toàn scene về mét.
- **Stage [3]–[9] — Completion.** Bù mặt khuất của từng object bằng Hunyuan3D, verify lại trên chính camera thật, ghép thành hybrid scene.
- **Stage [10]–[12] — Robot.** Biến scene đó thành thứ physics engine chạy được, rồi chứng minh robot dùng được nó.

```text
4K RGB VIDEO
    ↓
[1] HOLI-SPATIAL                                  → PHẦN B
    ├─ Camera poses
    ├─ Depth
    ├─ 3DGS
    ├─ Masks / semantics
    └─ 3D OBB
    ↓
[2] GLOBAL METRIC CALIBRATION                     → PHẦN B2
    ├─ 1 known physical distance / marker / RGB-D reference
    ├─ compute metres-per-unit
    └─ rescale toàn scene về mét
    ↓
[3] C1 — OBJECT PACKAGE                           → PHẦN C
    ├─ best_rgb.png
    ├─ best_mask.png
    ├─ observed_points.ply
    ├─ multi-view RGB
    ├─ masks
    ├─ depth
    ├─ cameras
    └─ object.json
         └─ metric OBB
    ↓
[4] OBJECT QUALITY GATE                           → PHẦN D
    ├─ đủ coverage
    │     ↓
    │   KEEP HOLI
    │
    └─ thiếu coverage
          ↓
      COMPLETE
    ↓
[5] ADAPTER A                                     → PHẦN E
    RGB + mask
        ↓
    crop object
        ↓
    SAM3 ∪ Rembg
        ↓
    white background
        ↓
    center + resize 518×518
        ↓
    front / left / back / right
    ↓
[6] HUNYUAN3D-2MV                                 → PHẦN F
    ↓
Generated complete mesh
mesh_raw.glb
    ↓
[7] OBJECT SCALE + INITIAL ALIGNMENT              → PHẦN G
    ├─ normalize generated mesh
    ├─ lấy Holi metric OBB làm reference
    ├─ scale generated mesh
    │
    │   sx = Holi_width  / Hunyuan_width
    │   sy = Holi_height / Hunyuan_height
    │   sz = Holi_depth  / Hunyuan_depth
    │
    ├─ ưu tiên dimensions đã quan sát chắc chắn
    ├─ translate → OBB center
    └─ yaw candidates:
         0° / 90° / 180° / 270°
    ↓
[8] MULTI-VIEW VERIFICATION                       → PHẦN H
    render candidate về camera thật
        ↓
    compare:
        ├─ Mask IoU
        ├─ Depth error
        ├─ Dimension error
        └─ observed-surface consistency
        ↓
    PASS?
      ├─ YES → accept completed mesh
      └─ NO  → fallback Holi geometry
    ↓
[9] FINAL HYBRID SCENE                            → PHẦN I
    Holi 3DGS / reconstructed scene
          +
    verified completed meshes
    ↓                        (optional: Hunyuan3D-Paint → PHẦN J)
[10] ROBOT-READY SCENE COMPILER                   → PHẦN R1
    ├─ TSDF / explicit mesh
    ├─ simplified collision mesh
    ├─ floor / wall / obstacle semantics
    ├─ heightfield
    ├─ static / dynamic tags
    └─ URDF / MJCF / USD
    ↓
[11] ROBOT SIMULATION                             → PHẦN R2
    ├─ MuJoCo
    ├─ Isaac / PhysX
    └─ later: Newton
    ↓
[12] ROBOT VALIDATION                             → PHẦN R3
    ├─ collision
    ├─ clearance
    ├─ navigation
    ├─ traversability
    ├─ success rate
    └─ sim2sim / later sim2real
```

## Bảng ánh xạ stage → phần → script → output

| Stage | Nội dung | PHẦN | Script | Output | Trạng thái |
|---|---|---|---|---|---|
| [1] | Holi-Spatial baseline | B | *(Holi gốc, không sửa)* | `OUTPUT_B1/` | ✅ |
| [2] | Global metric calibration | B2 | `scripts/adapter/00_metric_calibration.py` | `OUTPUT_B2/` | ⏳ chưa viết |
| [3] | Object package C1 | C | `scripts/adapter/01_normalize_holi_output.py` | `OUTPUT_C1/` | ✅ |
| [3b] | OBB-anchored re-segmentation | C3 | `scripts/adapter/03_resegment_obb_sam3.py` *(GPU)* | `OUTPUT_C3/` | ✅ |
| [4] | Object quality gate | D | `scripts/adapter/02_quality_gate_and_adapter_a.py` | `OUTPUT_D_v2/` | ✅ |
| [5] | Adapter A | E | `scripts/adapter/02_quality_gate_and_adapter_a.py` | `OUTPUT_D_v2/adapter_a/` | ✅ |
| [6] | Hunyuan3D-2mv shape | F | `scripts/adapter/04_run_hunyuan_shape.py` | `OUTPUT_F/` | ✅ |
| [7] | Scale + initial alignment | G | `scripts/adapter/05_align_and_verify_adapter_b.py` | `OUTPUT_G/` | ✅ |
| [8] | Multi-view verification | H | `scripts/adapter/05_align_and_verify_adapter_b.py` | `OUTPUT_G/` | ✅ |
| [9] | Final hybrid scene | I | `scripts/adapter/06_assemble_final_scene.py` | `OUTPUT_I/` | ✅ |
| — | Texture *(optional)* | J | `scripts/adapter/07_texture_bake_objects.py` | `OUTPUT_J/` | ✅ |
| [10] | Robot-ready scene compiler | R1 | `scripts/adapter/10_robotics_manipulation_gate.py` + `scripts/sim/make_unitree_scene.py` | `OUTPUT_I/*.mjcf` | ✅ |
| [11] | Robot simulation | R2 | `scripts/sim/render_unitree_scene.py` | `OUTPUT_K/` | ⏳ một phần |
| [12] | Robot validation | R3 | `scripts/sim/validate_unitree_scene.py` | `OUTPUT_L/` | ⏳ một phần |

> **Ranh giới quan trọng nhất của toàn pipeline là Stage [2].**
> Mọi con số vật lý ở Stage [10]–[12] — khối lượng, lực, clearance, step height, tốc độ — đều nhân trực tiếp với `metres_per_unit`.
> Sai scale ở [2] thì không có gì phía sau đúng được, và **lỗi đó không nhìn thấy được trên ảnh render**: scene sai scale trông y hệt scene đúng scale.


---

# PHẦN A — Chuẩn bị project

## Step A1 — Tạo cấu trúc thư mục

```text
project/
│
├── data/
│   └── scene_001/
│       └── rgb/
│           ├── 000000.jpg
│           ├── 000001.jpg
│           └── ...
│
├── stage01_holi_raw/            # [1]
├── stage01b_metric_calibration/ # [2]  ← khoá scene về mét
├── stage02_holi_bridge/         # [3]
├── stage03_quality_gate/        # [4]
├── stage04_adapter_a/           # [5]
├── stage05_hunyuan_shape/       # [6]
├── stage06_adapter_b/           # [7]
├── stage07_validation/          # [8]
├── stage08_texture/             # optional (Paint)
├── final_scene/                 # [9]
├── stage09_robot_scene/         # [10] MJCF / URDF / USD + collision
├── stage10_robot_sim/           # [11] rollout logs, video, contact stats
└── stage11_robot_validation/    # [12] validation report
```

### OUTPUT của Step A1

```text
project/
└── data/scene_001/rgb/*.jpg
```

### Checkpoint

- [ ] Các frame cùng một scene.
- [ ] Không bị đảo thứ tự.
- [ ] Không có frame hỏng.
- [ ] Camera di chuyển đủ quanh các object cần reconstruct.
- [ ] Nếu có thể, giữ input 4K gốc; resize chỉ khi model/pipeline yêu cầu.

---

# PHẦN B — Chạy Holi-Spatial nguyên bản

## Step B1 — Chạy Holi-Spatial

**Không sửa code/model của Holi ở bước này.**

```text
RGB frames
    ↓
SfM / camera recovery
    ↓
Depth prior
    ↓
3DGS optimization
    ↓
VLM + SAM3
    ↓
2D masks
    ↓
2D → 3D lifting
    ↓
3D OBB
    ↓
Multi-view merge / confidence refinement
```

### OUTPUT tối thiểu cần lấy từ Holi

```text
stage01_holi_raw/
│
├── scene_3dgs/
│   └── <native Holi/3DGS files>
│
├── cameras/
│   └── <camera intrinsics + extrinsics>
│
├── depth/
│   └── <rendered/refined depth per frame>
│
├── masks/
│   └── <2D masks per object/per frame>
│
├── objects/
│   └── <3D OBB / instance metadata>
│
└── best_views/
    └── <best source frame per object>
```

> Tên file thật của Holi có thể khác. Không cần đổi Holi. Bước kế tiếp sẽ **normalize output** sang format adapter của riêng ta.

### Checkpoint

Ta phải nhìn thấy được ít nhất:

1. Scene 3DGS.
2. Camera pose.
3. Refined/rendered depth.
4. Mask của object.
5. 3D OBB.
6. Object ID.
7. Frame tốt nhất của object.

Nếu thiếu một trong **mask / camera / OBB**, chưa chạy Hunyuan.

---

# PHẦN B2 — Global Metric Calibration (Stage [2])

> **Chạy đúng một lần cho cả scene, ngay sau Holi và trước mọi thứ khác.**

## Step B2.0 — Vấn đề

Holi-Spatial (và DA3 phía dưới nó) trả về scene trong **world unit**, không phải mét. Scene đó đúng *về hình dạng* nhưng chỉ đúng *về kích thước* sai khác một hệ số toàn cục chưa biết `k`.

Hệ quả nếu bỏ qua bước này:

```text
OBB extent sai × k       → mesh Hunyuan bị scale sai × k
mass ∝ volume            → sai × k³
lực / moment             → sai × k³
clearance, step height   → sai × k
```

Và điểm nguy hiểm nhất: **render ra ảnh thì scene sai scale trông y hệt scene đúng scale.** Lỗi chỉ lộ ra ở stage robot, sau khi đã tốn toàn bộ compute phía trên.

Vì vậy scale phải được giải **một lần, ở cấp scene** — không bao giờ được sửa lẻ tẻ từng object ở stage sau. Scale từng object riêng biệt sẽ phá vỡ tính nhất quán không gian giữa các object.

## Step B2.1 — Nguồn tham chiếu (chọn 1, theo thứ tự ưu tiên)

| # | Nguồn | Cách lấy | Sai số điển hình |
|---|---|---|---|
| 1 | **RGB-D / LiDAR reference** | 1 frame có depth metric thật | < 1% |
| 2 | **Fiducial marker** | ArUco / checkerboard cạnh biết trước đặt trong cảnh khi quay | 1–2% |
| 3 | **Known physical distance** | Đo bằng thước 1 cạnh bất kỳ nhìn thấy trong video (khung cửa, mép bàn, chiều cao tủ) | 2–5% |
| 4 | **DA3 metric prior** | Không đo gì, tin ước lượng metric của DA3 | **không kiểm chứng** |

> Nguồn (4) là fallback, không phải calibration. Nó phải được đánh dấu `UNVERIFIED` và cờ đó phải đi theo tới tận stage robot.

## Step B2.2 — Tính hệ số

```text
metres_per_unit = L_real_metres / L_scene_units
```

Trong đó `L_scene_units` là khoảng cách *giữa đúng hai điểm đó* đo trong point cloud/mesh của Holi.

## Step B2.3 — Áp dụng vào những gì

```text
RESCALE (× metres_per_unit):
    ├─ camera translation (extrinsics t)
    ├─ depth map
    ├─ point cloud / observed_points.ply
    ├─ 3DGS: vị trí μ và scale Σ của từng gaussian
    └─ OBB: center_world và extent_xyz

KHÔNG ĐỔI:
    ├─ camera rotation R
    ├─ camera intrinsics K
    ├─ mask 2D
    └─ mọi đại lượng vô hướng bất biến scale (IoU, coverage, confidence)
```

## Step B2.4 — Verify bằng một tham chiếu thứ hai, độc lập

Bắt buộc. Dùng chính vật đã đo để verify là kiểm chứng vòng tròn.

```text
predicted = L2_scene_units × metres_per_unit
residual_pct = |predicted − L2_real| / L2_real × 100
```

Sanity check nhanh bằng vật có kích thước chuẩn trong ảnh:

```text
khung cửa      ≈ 2.00 – 2.10 m cao
mặt bàn        ≈ 0.70 – 0.80 m cao
mặt ngồi ghế   ≈ 0.42 – 0.48 m cao
pallet EUR     = 1.20 × 0.80 m
trần nhà xưởng ≈ 2.5 – 6 m
```

## Step B2.5 — Decision rule

```text
residual_pct <= 5%   → VERIFIED     → được phép chạy tới stage robot
5% < residual <= 15% → RECALIBRATE  → đo lại, hoặc đổi sang nguồn tham chiếu tốt hơn
residual_pct > 15%   → FAIL         → sai ở khâu chọn điểm hoặc scene reconstruction lỗi
không có tham chiếu  → UNVERIFIED   → geometry chạy được; mass/force ở [10] là ƯỚC LƯỢNG
```

## Step B2.6 — Output

```text
stage01b_metric_calibration/ (hoặc OUTPUT_B2/)
├── metric_calibration.json
├── scene_metric/                # bản đã rescale của toàn bộ artefact ở [1]
│   ├── cameras/
│   ├── depth/
│   ├── points_metric.ply
│   └── scene_3dgs_metric/
└── calibration_evidence/
    ├── reference_frame.png      # frame có đánh dấu 2 điểm đã đo
    └── reference_overlay.png
```

### `metric_calibration.json`

```json
{
  "scene_id": "scene_001",
  "metres_per_unit": 1.0,
  "source": "tape_measure",
  "reference": {
    "description": "chiều rộng khung cửa xưởng",
    "frame_id": "000120",
    "measured_metres": 1.20,
    "scene_units": 1.20
  },
  "verification": {
    "description": "chiều cao mặt bàn",
    "real_metres": 0.75,
    "predicted_metres": 0.78,
    "residual_pct": 4.0
  },
  "status": "VERIFIED",
  "applied_to": ["cameras", "depth", "points", "3dgs", "obb"]
}
```

> `"metres_per_unit": 1.0` **không** có nghĩa là "chưa làm gì". Nó có nghĩa là **"tin ước lượng metric của DA3"** — một quyết định, và phải đi kèm `"source": "da3_prior_unverified"`, `"status": "UNVERIFIED"`.
> Cùng một cờ này là thứ mà `scripts/sim/make_unitree_scene.py --metres-per-unit` đọc để rescale mọi độ dài, khối lượng và lực trong MJCF.

### Checkpoint

- [ ] Đúng **một** `metres_per_unit` cho cả scene.
- [ ] Tham chiếu verify là **độc lập** với tham chiếu calibrate.
- [ ] `residual_pct` đã ghi ra file, không phải chỉ in ra terminal.
- [ ] Mọi OBB extent ở PHẦN C là **mét**, không phải unit.
- [ ] Sanity check: không có vật nào cao hơn trần, không có ghế cao 4 m.
- [ ] Nếu `UNVERIFIED`: cờ đã được truyền xuống tới `OUTPUT_I` và validation report.

---

# PHẦN C — Chuẩn hóa output Holi thành interface của adapter

## Step C1 — Tạo `HoliObjectPackage`

Với mỗi object `obj_XXXX`, lưu tối đa **10 góc nhìn** (`max_views = 10`) cùng mask, depth và camera tương ứng:

> **Mask mà C1 ghi ra chỉ là mask thô của Holi, không phải mask cuối cùng.**
> Holi gọi SAM 3 bằng text prompt thuần (`sam3.py:151` — `set_text_prompt(label)`), không có ràng buộc hình học nào, rồi lifting 3D cắt gọt tiếp. Kết quả đo được trên scene nhà xưởng: mask `obj_0219_table` chỉ phủ **mặt bàn**, mất toàn bộ chân, thanh giằng và mặt hộc kéo.
> Mask dùng thật cho Adapter A đến từ **PHẦN C3**. `best_rgb.png` / `best_mask.png` của C1 cũng không còn quyết định best view — luật chọn view nằm ở PHẦN E.

```text
stage02_holi_bridge/ (hoặc OUTPUT_C1/)
└── obj_0001/
    │
    ├── object.json
    ├── best_rgb.png          # View nguyên vẹn, không chạm viền, diện tích lớn nhất
    ├── best_mask.png         # Binary mask giải nén từ RLE
    ├── observed_points.ply
    │
    ├── views/                # Tối đa 10 views tốt nhất (không lưu thừa)
    │   ├── frame_000120.png
    │   └── ...
    ├── masks/                # Tối đa 10 masks tương ứng
    │   ├── frame_000120.png
    │   └── ...
    ├── depth/                # Tối đa 10 depth tương ứng (đơn vị: MÉT)
    │   ├── frame_000120.npy
    │   └── ...
    └── cameras/              # Tối đa 10 camera JSONs tương ứng
        ├── frame_000120.json
        └── ...
```

## Step C2 — Schema của `object.json`

```json
{
  "instance_id": "obj_0001",
  "semantic_class": "chair",
  "confidence": 0.93,

  "best_frame_id": "000120",

  "units": "metres",
  "metres_per_unit": 1.0,
  "metric_status": "VERIFIED",

  "obb": {
    "center_world": [0.0, 0.0, 0.0],
    "extent_xyz": [1.0, 1.0, 1.0],
    "rotation_world_3x3": [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ]
  },

  "dimension_confidence_xyz": [0.91, 0.88, 0.34],

  "source_views": [
    "000120",
    "000150"
  ]
}
```

> Các số phía trên chỉ là **schema minh họa**. Adapter phải điền dữ liệu thật từ Holi.

### Ba trường mới, và vì sao chúng bắt buộc

- `units` / `metres_per_unit` / `metric_status`: kế thừa trực tiếp từ Stage [2]. `extent_xyz` phải đã là **mét**. Object package nào thiếu ba trường này thì Adapter B không có quyền tin OBB của nó.
- `dimension_confidence_xyz`: độ tin cậy **riêng cho từng trục** của OBB, tính từ camera coverage của trục đó và tỉ lệ observed points phủ dọc trục:

```text
conf[axis] = 0.5 × bin_coverage(axis) + 0.5 × point_span(axis) / extent(axis)
```

  Trục nào camera chưa bao giờ nhìn thấy hai đầu thì extent của nó là **cận dưới**, không phải phép đo. Đây chính là trục mà Stage [7] không được phép scale theo. Thiếu trường này là nguyên nhân gốc của lỗi mesh bị bóp dẹt theo chiều sâu.

### OUTPUT của PHẦN C

```text
stage02_holi_bridge/obj_XXXX/
```

Đây là **interface chuẩn** giữa Holi và toàn bộ code adapter phía sau.

### Checkpoint

- [ ] `best_rgb.png` đúng object.
- [ ] `best_mask.png` khớp với `best_rgb.png`.
- [ ] `object.json` có OBB.
- [ ] `observed_points.ply` nằm đúng world coordinate.
- [ ] Camera metadata khớp đúng frame.

---

# PHẦN C3 — Segment lại bằng SAM 3 neo vào OBB

> Script: `scripts/adapter/03_resegment_obb_sam3.py` · chạy trên GPU · output `OUTPUT_C3/`

## Step C3.0 — Vì sao phải segment lại

Holi prompt SAM 3 **bằng chữ**, không bằng hình học. Với `"table"`, concept head bám vào vùng phẳng nổi bật nhất và trả về mặt bàn. Bản vá cũ ở PHẦN E là hợp mask với rembg (`A ∪ B`), nhưng union chỉ biết **cộng**: nó kéo thêm ba cái hộp đen đứng trên bàn và một mảng sàn, mà vẫn không lấy lại được chân bàn.

Thứ chưa ai dùng: mỗi object đã có **OBB mét** trong `object.json` và **camera đầy đủ** trong `cameras/`. Chiếu OBB xuống ảnh cho một **bao hình học chính xác** — vừa chặn phần thừa vừa lộ ra phần thiếu.

## Step C3.1 — Phép chiếu

`rotation` trong camera JSON là **camera→world**, nên world→camera là chuyển vị:

```text
X_cam  = Rᵀ (X_world − C)
u      = fx · X_cam.x / X_cam.z + width/2
v      = fy · X_cam.y / X_cam.z + height/2
hull   = convexHull(project(8 góc OBB))
```

`rotation_world_3x3` của OBB map **local→world**, `extent_xyz` là full extent, nên `local = R_obbᵀ(X_world − center)` và 8 góc nằm đúng tại `±extent/2`.

## Step C3.2 — Prompt bằng hình học, không bằng chữ

SAM 3 có sẵn `Sam3Processor.add_geometric_prompt(box, label, state)` nhận **box dương và box âm** ở dạng `[cx, cy, w, h]` chuẩn hoá — đường mà `sam3.py` chưa từng gọi.

```text
positive box  = bbox(hull)
negative box  = bbox(hull) của mọi instance j thoả:
                  OBB_j ⊄ OBB_i   (không phải bộ phận)
                  và OBB_j ∩ OBB_i ≠ ∅   (thật sự đè lên)
```

Bốn tổ hợp được thử cho mỗi view — `obb_box`, `obb_box_neg`, `text_obb_box`, `text_only` — cộng thêm **chính mask cũ của Holi** làm ứng viên. Mask cũ dự thi bằng đúng thước đo, nên bước này **không bao giờ làm object tệ đi so với hiện tại**.

## Step C3.3 — Chọn ứng viên bằng tiêu chí 3D

```text
fill = |mask ∩ hull| / |hull|      phủ được bao nhiêu phần OBB
leak = |mask \ hull| / |mask|      thoát ra ngoài OBB bao nhiêu
```

`leak` là **gate cứng** (mặc định `0.05`): OBB là biên vật lý, pixel ngoài nó sai theo định nghĩa. `fill` **chỉ để xếp hạng**, không làm gate — vật rỗng như kệ, thang, ống lấp rất ít thể tích bao của chính nó một cách hoàn toàn hợp lệ.

Thứ tự xếp hạng phải tách hai nhánh:

```python
if leak <= leak_max:  key = (1,  fill, -leak)
else:                 key = (0, -leak,  fill)
```

Nhánh `else` bắt buộc xếp theo `-leak`. Nếu xếp theo `fill` như nhánh trên, một mask nuốt trọn khung hình sẽ có `fill = 1.0` một cách tầm thường và **thắng tuyệt đối** — gate fail open. Đo thực tế trước khi sửa: `obj_0211_shelf` nhận mask `leak = 0.984`, phình từ 134 lên 11.867 px.

## Step C3.4 — Bỏ phiếu chéo giữa các view

Mỗi view segment độc lập. Sau đó view nào có `fill < median − 2·MAD` so với các view còn lại của cùng object thì bị đánh dấu outlier — nó đã nhìn thấy thứ khác chứ không phải object.

Một view chỉ `view_ok` khi thoả **cả bốn**:

```text
toàn bộ 8 góc OBB nằm trong ảnh, lề ≥ 16 px   (không bị cắt cụt)
mask không chạm viền ảnh trong 16 px
không phải fill-outlier
leak ≤ leak_max
```

## Step C3.5 — Output

```text
OUTPUT_C3/
├── resegment_report.json      # fill / leak / variant / view_ok từng view
└── obj_0219_table/
    └── masks_v2/
        ├── DSC01068.png
        └── ...
```

### Checkpoint

- [ ] `leak` của mọi view được chọn ≤ 0.05.
- [ ] Không view nào thắng bằng `text_only` với `leak` cao.
- [ ] Object nào mask mới thua mask cũ thì `variant == "holi_original"`.

---

# PHẦN D — Quality Gate: object nào mới cần Hunyuan?

## Step D1 — Bộ lọc loại trừ (Exclude Classes) & Phân loại đối tượng

Không chạy Hunyuan3D cho toàn bộ các nhãn kiến trúc phẳng, vật thể trong suốt, hoặc chi tiết gắn tường/sàn:

### ❌ Các nhãn LOẠI TRỪ HOÀN TOÀN (Excluded Classes):
```text
floor, wall, ceiling, background, large structural surfaces
door, glass, wheel, window, windows, brush
light, lamp, pipe, pipes, tube, tubing, bottle, bottles, can, cans
sticker, label, paper, cord, knob, handle, hinge, sign, electrical outlet, electrical_outlet
wood, panel, rubber, plastic, base, support
```

### ✅ Các 3D Assets độc lập THỰC SỰ ĐƯỢC GIỮ LẠI:
```text
chair, table, cabinet, sofa, machine, toolbox, equipment, shelf, workbench, box, ladder, trash can, container, tool
```

## Step D2 — Tính camera coverage quanh 8 góc Azimuth

Chia góc nhìn quanh object thành 8 azimuth bins (mỗi bin 45°):

```text
                 0° (Front)
                   ↑

           315°       45°

  270° (Right)    O       90° (Left)

           225°       135°

                   ↓
                180° (Back)
```

Tính tỉ lệ bao phủ:
```text
coverage = observed_bins / 8.0
```

### Tiêu chí quyết định (Decision Rules)
```text
Class thuộc danh sách Exclude:
    → SKIP_SMALL_OBJECT / EXCLUDED (Không đưa vào sinh 3D)

coverage >= 0.75:
    → KEEP_HOLI (Đã quan sát đủ 360 độ từ video quét thực tế)

0.50 <= coverage < 0.75:
    → REVIEW_OPTIONAL (Cân nhắc bù đắp nếu bị khuyết mặt sau)

coverage < 0.50:
    → COMPLETE_WITH_HUNYUAN (Thiếu mặt sau/các góc bên → bắt buộc chạy Hunyuan3D)
```

## Step D3 — Output của Phần D

```text
stage03_quality_gate/ (hoặc OUTPUT_D/)
├── coverage_report.csv          # Báo cáo chi tiết từng object (ID, Class, Coverage, Decision)
├── complete_objects.json        # Danh sách KEEP_HOLI
├── incomplete_objects.json      # Danh sách COMPLETE_WITH_HUNYUAN
└── skipped_small_objects.json   # Danh sách đã lọc bỏ
```

---

# PHẦN E — Adapter A: Holi → Hunyuan3D (neo vào OBB, xuất RGBA)

> Script: `scripts/adapter/02_quality_gate_and_adapter_a.py` · chạy CPU · output `OUTPUT_D_v2/adapter_a/`

## Step E1 — Input Adapter A

* `OUTPUT_C3/obj_XXXX/masks_v2/*.png` — mask đã neo OBB (PHẦN C3).
* `OUTPUT_C3/resegment_report.json` — `fill` / `leak` / `view_ok` từng view.
* `OUTPUT_C1/obj_XXXX/{views,cameras}/` — ảnh gốc và camera.

Không còn dùng `best_rgb.png` / `best_mask.png`: best view do PHẦN E chọn lại.

## Step E2 — Bỏ vật thể đè lên, vá lỗ

Với mỗi instance `j` khác:

```text
OBB_j ⊆ OBB_i                   → GIỮ   (bộ phận: ê-tô bắt vào bàn, tay nắm hộc kéo)
OBB_j ∩ OBB_i ≠ ∅, không lồng   → TRỪ   mask_j đã dilate 3 px, rồi INPAINT_TELEA vá lỗ
không giao nhau trong 3D        → BỎ QUA
```

Phép kiểm tra giao nhau **trong 3D** là bắt buộc. Nếu chỉ xét chồng lấn trên ảnh, cái máy đứng phía sau bàn cũng bị coi là vật đè và bị trừ khỏi mask bàn. Đo thực tế khi thiếu phép kiểm tra này: mask `obj_0219_table` tụt từ 210.914 xuống 7.261 px, bốn view khác về 0.

Thêm một chốt chặn: nếu phép trừ ăn quá **30%** mask, dữ liệu instance sai chứ không phải vật thể bị che — giữ nguyên mask còn hơn đưa một mảnh vụn cho Hunyuan3D.

## Step E3 — Bỏ đốm rời

Giữ thành phần liên thông lớn nhất, cùng mọi thành phần ≥ **10%** diện tích của nó. Đốm nhỏ nằm lọt trong OBB vượt được mọi phép kiểm tra hình học, nhưng trên nền alpha trong suốt thì Hunyuan3D đọc chúng thành hình khối rời. Một cái chân bàn hay bậc thang thật đều vượt xa ngưỡng này.

## Step E4 — Chọn best view

Chỉ view có `view_ok` mới được dự tuyển. Xếp hạng:

```text
score = fill × area × (1 nếu đủ nét, 0.05 nếu không)
```

Độ nét đo bằng **phương sai Laplacian trên vùng vật thể đã resize về 256×256**, không đo trên pixel thô. Trên pixel thô, đại lượng này tỉ lệ với chi tiết-trên-mỗi-pixel: một vật ở xa nhồi chi tiết vào ít pixel sẽ ăn đứt ảnh cận cảnh sạch của cùng vật đó. Đo được trên `obj_0219_table`: view xa 19.879 px cho 455, view cận cảnh 210.914 px cho 35 — chuẩn hoá xong cả hai về khoảng 164–337 và thứ hạng đảo lại cho đúng.

Nếu không view nào `view_ok`, lấy view **ít leak nhất** trong số còn lại, với điều kiện `leak ≤ 0.15` và mask không phủ quá 60% khung hình. Không có view nào đạt thì **bỏ object và báo cáo**, chứ không đưa mask hỏng xuống dưới.

## Step E5 — Gán slot multi-view theo hệ riêng của OBB

```python
R_obb    = obj['obb']['rotation_world_3x3']       # local → world
d_local  = R_obb.T @ (cam_position − obb_center)
up       = trục local gần world-Z nhất
front    = trục ngang có extent NHỎ hơn           # nhìn vào mặt dài nhất
az       = atan2(d_local[side], d_local[front]) mod 360

front 315–45°  |  left 45–135°  |  back 135–225°  |  right 225–315°
```

Phải đo trong hệ riêng của OBB, không phải hệ world. `conditioner.py:174-180` cho thấy Hunyuan3D-2mv có **`view_embed` học được, đánh index theo slot** — nhãn slot mang ý nghĩa thật với model. Đo bằng azimuth world thì "mặt trước" của mỗi object phụ thuộc vào việc nó tình cờ xoay thế nào trong phòng, và model nhận conditioning mâu thuẫn.

`≥2` slot → chạy `Hunyuan3D-2mv`. Đúng `1` slot → chạy `Hunyuan3D-2` single-view.

## Step E6 — Xuất ảnh RGBA

```text
mask → bbox → pad vuông 10% → resize 518×518
kênh alpha = mask;  RGB dưới alpha để trắng cho dễ xem bằng mắt
```

**Alpha là bắt buộc, không phải nền trắng đặc.** `04_run_hunyuan_shape.py:116` gọi `.convert("RGBA")`; trên PNG nền trắng nó cho alpha = 255 khắp nơi, và model hiểu "vật thể" là trọn ô vuông 518×518 thay vì cái bàn. Hunyuan3D crop và căn giữa dựa trên alpha.

**rembg đã bị bỏ hẳn.** Hợp alpha của nó với mask segmentation chính là thứ kéo vật thể hàng xóm và sàn nhà vào ảnh crop.

## Step E7 — Cấu trúc Output

```text
OUTPUT_D_v2/
├── coverage_report.csv
├── contact_sheet.html          # trước/sau toàn scene, dựng bằng
│                               # scripts/adapter/tools/make_contact_sheet.py
└── adapter_a/
    ├── adapter_a_manifest.json
    └── obj_0219_table/
        ├── hunyuan_input.png   # 518×518 RGBA, alpha = mask
        ├── original_crop.png
        ├── adapter_a_meta.json
        └── mv_inputs/
            └── front.png       # chỉ những slot có view thật
```

### `adapter_a_meta.json`

```json
{
  "instance_id": "obj_0219_table",
  "semantic_class": "table",
  "source_frame": "DSC01068",
  "mode": "singleview",
  "azimuth_frame": "obb_local",
  "eligible_views": ["DSC00973", "DSC00976", "DSC01015"],
  "canonical_multiview_slots": {
    "front": { "view_id": "DSC01068", "angle_deg": 2.0, "file": "mv_inputs/front.png", "pixel_area": 210914 }
  },
  "num_canonical_views": 1
}
```

### Checkpoint

`hunyuan_input.png` phải:

- [ ] là RGBA, alpha khớp đúng biên vật thể;
- [ ] chứa **trọn** object — có chân, có thân, không cụt ở mép ảnh;
- [ ] **không** chứa vật thể khác đứng đè lên, không nền xưởng, không sàn;
- [ ] không có đốm rời trôi trên nền trong suốt.


# PHẦN F — Chạy Hunyuan3D-DiT: Shape Generation

## Step F1 — Hỗ trợ cả Multi-View (`Hunyuan3D-2mv`) và Single-View

* **Multi-View mode (`Hunyuan3D-2mv` - Khuyên dùng):**
  Nạp từ điển các góc trực giao `images = {"front": ..., "left": ..., "back": ..., "right": ...}` từ thư mục `mv_inputs/`. Mô hình sẽ tận dụng thông tin đa hướng thực tế để sinh khối 3D chính xác $360^\circ$ mà không bị méo lệch góc sau.
* **Single-View mode (`Hunyuan3D-2`):**
  Nạp duy nhất ảnh `hunyuan_input.png` nếu vật thể chỉ quan sát được từ 1 góc nhìn duy nhất.

```text
mv_inputs/ (hoặc hunyuan_input.png)
       │
       ▼
Hunyuan3D-DiT (Hunyuan3D-2mv)
       │
       ▼
complete bare 3D mesh (mesh_raw.glb)
```

## Step F2 — Script thực thi (`scripts/adapter/04_run_hunyuan_shape.py`)

```bash
# Chạy Multi-View cho toàn bộ danh mục trong Step E:
python scripts/adapter/04_run_hunyuan_shape.py \
  --adapter-manifest outputs-full-factory/delivery/OUTPUT_D/adapter_a/adapter_a_manifest.json \
  --output-dir outputs-full-factory/delivery/OUTPUT_F \
  --mode multiview

# Chạy thử nghiệm MVP trên 1 vật thể cụ thể:
python scripts/adapter/04_run_hunyuan_shape.py \
  --target-object obj_0219_table \
  --output-dir outputs-full-factory/delivery/OUTPUT_F
```

## Step F3 — Output của Phần F

```text
stage05_hunyuan_shape/ (hoặc OUTPUT_F/)
├── stage05_manifest.json
└── obj_0219_table/
    ├── mesh_raw.glb            # Bare 3D mesh khép kín (đã có mặt sau)
    ├── mesh_raw.obj            # Bare OBJ mesh
    └── generation_meta.json    # Thời gian sinh, số đỉnh/mặt
```

Chỉ cần giữ một mesh format nếu implementation của bạn chỉ sinh một format.

### Checkpoint

Mesh phải:

- [ ] kín hơn geometry Holi cũ;
- [ ] có backside;
- [ ] đúng loại object;
- [ ] không bị topology phá hỏng nghiêm trọng.

### Lưu ý

```text
mesh_raw
```

**chưa ở world coordinate của Holi**.

Không chèn mesh này trực tiếp vào scene.

---

# PHẦN G — Adapter B.1: Initial Alignment

## Step G1 — Input

```text
Hunyuan:
    mesh_raw

Holi:
    OBB
    observed_points
    camera poses
    masks
    depth
```

## Step G2 — Normalize Hunyuan mesh

Tính mesh center:

```text
c_gen
```

đưa mesh về origin:

```text
V0 = V - c_gen
```

## Step G3 — Scale theo Holi metric OBB (per-axis, có trọng số tin cậy)

Hunyuan extent (sau normalize):

```text
e_gen = [w_gen, h_gen, d_gen]
```

Holi extent — **đã là mét nhờ Stage [2]**:

```text
e_holi = [w_holi, h_holi, d_holi]
```

### G3.a — Tỉ lệ từng trục

```text
sx = w_holi / w_gen
sy = h_holi / h_gen
sz = d_holi / d_gen
```

Dùng **per-axis** chứ không dùng uniform scale, vì tỉ lệ của mesh Hunyuan sinh ra là *học được từ prior*, không phải đo được từ cảnh này. Bàn thật trong xưởng dài 2.4 m và bàn Hunyuan sinh ra tỉ lệ 1:1 sẽ không bao giờ khớp OBB nếu chỉ được scale đều.

### G3.b — Ưu tiên dimensions đã quan sát chắc chắn

Đây là phần bắt buộc, không phải tinh chỉnh. `sz` thường là trục **mặt sau** — đúng cái trục Holi không nhìn thấy, và cũng đúng cái lý do ta phải gọi Hunyuan ngay từ đầu. Tin `d_holi` trên trục đó là tin một cận dưới.

Dùng `dimension_confidence_xyz` từ `object.json` (PHẦN C):

```text
τ = 0.6                                  # ngưỡng "quan sát chắc chắn"

C = { axis | conf[axis] >= τ }           # tập trục đáng tin

nếu C rỗng:
    → không đủ căn cứ scale. Fallback: giữ tỉ lệ gốc của Hunyuan,
      scale đều theo đường chéo OBB, và đánh dấu LOW_CONFIDENCE.

ngược lại:
    s_ref = median{ s_axis | axis ∈ C }

    s_axis_final =  s_axis   nếu axis ∈ C        # trục đo được → tin Holi
                    s_ref    nếu axis ∉ C        # trục khuất   → giữ tỉ lệ Hunyuan
```

Ý nghĩa: **trục nào Holi đo được thì nghe Holi, trục nào Holi không thấy thì nghe Hunyuan.** Không trục nào bị đoán.

### G3.c — Chốt chặn anisotropy

```text
a = max(s_final) / min(s_final)

a <= 1.5   → nhận
a >  1.5   → OBB nhiều khả năng sai (mask dính vật khác, lifting lỗi)
             → fallback uniform s_ref + cờ ANISOTROPY_CLAMPED
```

Không có chốt chặn này, một OBB hỏng sẽ kéo mesh dãn thành hình thù vô nghĩa mà silhouette IoU vẫn có thể chấp nhận được ở một vài view.

## Step G4 — Translation

Dùng:

```text
t = Holi OBB center
```

## Step G5 — Rotation candidates

Lấy OBB orientation làm initialization và thử:

```text
yaw = base_yaw + 0°
yaw = base_yaw + 90°
yaw = base_yaw + 180°
yaw = base_yaw + 270°
```

## Step G6 — Output

```text
stage06_adapter_b/ (hoặc OUTPUT_G/)
└── obj_0001/
    ├── candidate_000.obj
    ├── candidate_090.obj
    ├── candidate_180.obj
    ├── candidate_270.obj
    └── transforms_candidates.json
```

### `transforms_candidates.json`

```json
{
  "instance_id": "obj_0001",
  "units": "metres",
  "scale_xyz": [1.02, 0.97, 1.01],
  "scale_source_per_axis": ["holi", "holi", "hunyuan_ratio"],
  "dimension_confidence_xyz": [0.91, 0.88, 0.34],
  "anisotropy": 1.05,
  "flags": [],
  "translation_xyz": [0.0, 0.0, 0.0],
  "candidates": [
    { "yaw_deg": 0,   "rotation_3x3": [[1,0,0],[0,1,0],[0,0,1]] },
    { "yaw_deg": 90,  "rotation_3x3": [[0,-1,0],[1,0,0],[0,0,1]] },
    { "yaw_deg": 180, "rotation_3x3": [[-1,0,0],[0,-1,0],[0,0,1]] },
    { "yaw_deg": 270, "rotation_3x3": [[0,1,0],[-1,0,0],[0,0,1]] }
  ]
}
```

> `scale_source_per_axis` là trường quan trọng nhất khi debug. Nó trả lời trực tiếp câu hỏi "kích thước này ở đâu ra" cho từng trục một.

---

# PHẦN H — Adapter B.2: Multi-view Verification

## Step H1 — Render candidate về camera của Holi

Với mỗi candidate mesh:

```text
aligned candidate
       │
       ▼
render từ Holi camera i
       │
       ├── silhouette_i
       └── depth_i
```

## Step H2 — So sánh silhouette

```text
IoU_i = IoU(
    rendered_silhouette_i,
    Holi_mask_i
)
```

Tính:

```text
mean_mask_iou
```

## Step H3 — So sánh depth

Chỉ tính trong object mask:

```text
depth_error_i =
    mean(
        abs(
            rendered_mesh_depth_i
            -
            Holi_depth_i
        )
    )
```

Có thể normalize theo object depth để có:

```text
relative_depth_error
```

## Step H3b — Dimension error

Chỉ tính trên các trục **confident** (`conf[axis] >= τ`). Trục khuất không có ground truth nên chấm điểm nó là tự lừa mình.

```text
dimension_error =
    mean over axis ∈ C of
        | extent_final[axis] − extent_holi[axis] | / extent_holi[axis]
```

Metric này bắt lỗi mà IoU không bắt được: một mesh nhỏ hơn thực tế nhưng đặt gần camera hơn vẫn có thể cho silhouette khớp.

## Step H3c — Observed-surface consistency

So mesh candidate với **điểm đã thực sự quan sát được**, không so với render:

```text
với mỗi p ∈ observed_points.ply:
    d(p) = khoảng cách từ p tới bề mặt mesh candidate gần nhất

surface_rmse     = sqrt( mean( d(p)² ) )                [mét]
surface_rmse_rel = surface_rmse / ||OBB diagonal||
inlier_ratio     = tỉ lệ điểm có d(p) <= 0.02 m
```

Đây là ràng buộc chặt nhất trong bốn metric, vì `observed_points.ply` là dữ liệu 3D thật, không qua bước render nào. Nó bắt được candidate đúng silhouette nhưng lệch theo phương chiều sâu — chính là kiểu lỗi mà yaw 180° hay tạo ra ở vật thể gần đối xứng.

## Step H4 — Score

```text
score =
      w1 * mean_mask_iou
    − w2 * relative_depth_error
    − w3 * dimension_error
    − w4 * surface_rmse_rel

mặc định:  w1 = 1.0,  w2 = 0.5,  w3 = 0.3,  w4 = 0.5
```

> Bốn trọng số là hyperparameter của adapter, không phải từ paper. Nhưng thứ tự quan trọng thì có căn cứ: `mask_iou` chọn *hướng*, `surface_rmse_rel` chọn *vị trí*, `dimension_error` chọn *kích thước*.

## Step H5 — Chọn candidate tốt nhất

```text
best_candidate
    =
candidate có score cao nhất
```

Nếu candidate tốt nhất và candidate nhì cách nhau `< 0.02` score, object gần đối xứng và hướng chưa được quyết định chắc chắn → ghi cờ `AMBIGUOUS_YAW` vào `alignment_metrics.json`. Với vật thể đi vào stage manipulation, cờ này phải được người xem lại.

## Step H6 — Output

```text
stage07_validation/
└── obj_0001/
    ├── mesh_aligned_final.obj
    ├── transform_final.json
    ├── alignment_metrics.json
    │
    └── renders/
        ├── view_000120_rgb.png
        ├── view_000120_mask.png
        ├── view_000120_depth.png
        └── ...
```

### `transform_final.json`

```json
{
  "instance_id": "obj_0001",
  "units": "metres",
  "scale_xyz": [1.02, 0.97, 1.01],
  "rotation_3x3": [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ],
  "translation_xyz": [0, 0, 0],
  "yaw_deg": 0
}
```

### `alignment_metrics.json`

```json
{
  "instance_id": "obj_0001",
  "mean_mask_iou": 0.0,
  "relative_depth_error": 0.0,
  "dimension_error": 0.0,
  "surface_rmse_m": 0.0,
  "surface_rmse_rel": 0.0,
  "inlier_ratio": 0.0,
  "score": 0.0,
  "runner_up_gap": 0.0,
  "decision": "ACCEPT",
  "fail_reasons": [],
  "flags": []
}
```

## Threshold accept đề xuất cho MVP

> Đây là threshold khởi đầu để debug, không phải ground truth của paper.

```text
mean_mask_iou          >= 0.65
AND relative_depth_error <= 0.10
AND dimension_error      <= 0.15      # chỉ trên các trục confident
AND surface_rmse_rel     <= 0.05
AND inlier_ratio         >= 0.70
```

Nếu fail bất kỳ điều kiện nào:

```text
REJECT HUNYUAN MESH
→ giữ object Holi ban đầu
→ ghi lý do fail (metric nào, giá trị bao nhiêu) vào alignment_metrics.json
```

Fallback im lặng là thứ phải tránh: một object bị reject mà không ghi lý do sẽ xuất hiện ở scene cuối như thể nó đã pass.

---

# PHẦN I — Không merge topology với 3DGS ở Version 1

## Step I1 — Giữ Holi scene nguyên bản

```text
Holi scene 3DGS
```

## Step I2 — Chỉ thêm mesh object đã align

```text
Final scene
=
Holi 3DGS
+
aligned Hunyuan meshes
```

### OUTPUT

```text
final_scene/
├── holi_3dgs/
│   └── <copy/link native Holi 3DGS scene>
│
├── completed_meshes/
│   ├── obj_0001.glb
│   ├── obj_0007.glb
│   └── ...
│
└── scene_manifest.json
```

### `scene_manifest.json`

```json
{
  "scene_id": "scene_001",

  "background_representation": "Holi-Spatial 3DGS",

  "units": "metres",
  "metres_per_unit": 1.0,
  "metric_status": "VERIFIED",

  "completed_objects": [
    {
      "instance_id": "obj_0001",
      "mesh": "completed_meshes/obj_0001.glb",
      "transform": "identity_world",
      "source": "Hunyuan3D completion"
    }
  ],

  "fallback_objects": [
    {
      "instance_id": "obj_0007",
      "source": "Holi geometry",
      "reason": "surface_rmse_rel 0.11 > 0.05"
    }
  ]
}
```

> Đây là **hybrid scene**, chưa phải một file GLB duy nhất.
>
> Ba trường `units` / `metres_per_unit` / `metric_status` được Stage [10] đọc trực tiếp. Thiếu chúng thì scene compiler không có cách nào biết 1 đơn vị trong file này là bao nhiêu mét, và mọi khối lượng nó tính ra sẽ sai theo luỹ thừa ba.

---

# PHẦN J — Optional: Hunyuan3D-Paint

Chỉ làm sau khi Shape + Alignment chạy ổn.

## Step J1

Input:

```text
mesh_aligned_final
+
best_rgb
```

## Step J2

```text
Hunyuan3D-Paint
        ↓
textured mesh
```

## Step J3 — Output

```text
stage08_texture/
└── obj_0001/
    ├── mesh_textured.glb
    ├── texture_basecolor.png
    └── material_meta.json
```

Sau đó thay:

```text
mesh_aligned_final
```

bằng:

```text
mesh_textured
```

trong `final_scene/scene_manifest.json`.

---

# PHẦN K — Evaluation: bắt buộc phải lưu Before / After

## Step K1 — Baseline

Render Holi object từ các view:

```text
before/
├── front.png
├── side.png
├── back.png
└── top.png
```

## Step K2 — Sau completion

```text
after/
├── front.png
├── side.png
├── back.png
└── top.png
```

## Step K3 — Metrics

```text
evaluation/
└── obj_0001_metrics.json
```

Ví dụ schema:

```json
{
  "instance_id": "obj_0001",

  "camera_coverage": 0.375,
  "metric_status": "VERIFIED",
  "dimension_confidence_xyz": [0.91, 0.88, 0.34],

  "before": {
    "missing_backside": true,
    "mean_mask_iou": 0.0,
    "surface_rmse_rel": 0.0
  },

  "after": {
    "mean_mask_iou": 0.0,
    "relative_depth_error": 0.0,
    "dimension_error": 0.0,
    "surface_rmse_rel": 0.0,
    "inlier_ratio": 0.0,
    "alignment_pass": false
  }
}
```

Adapter phải điền giá trị thật.

### Kết quả quan trọng nhất cần báo cáo

```text
1. Backside còn hole hay không?
2. Mesh có đúng position không?
3. Mesh có đúng scale không — TÍNH BẰNG MÉT, không phải unit?
4. Mesh có đúng orientation không?
5. Front observed surface có bị lệch nhiều không?
6. Mask IoU trước/sau.
7. Depth error trước/sau.
8. Surface RMSE so với observed_points trước/sau.
9. Trục nào của kích thước là ĐO ĐƯỢC, trục nào là SUY RA?
```

> Mục `9` là thứ hay bị bỏ sót nhất khi báo cáo. Một mesh khớp OBB hoàn hảo trên trục mà camera chưa bao giờ nhìn thấy không chứng minh điều gì cả — nó chỉ đang khớp với một phỏng đoán.

---

# PHẦN R1 — Robot-Ready Scene Compiler (Stage [10])

> Từ đây trở đi, scene không còn phục vụ mắt người mà phục vụ solver.

## Step R1.0 — Vì sao cần hẳn một stage riêng

3DGS **không mô phỏng được**. Gaussian không có bề mặt, không có thể tích, không có contact normal. Physics engine cần ba thứ mà scene ở Stage [9] chưa có:

```text
1. bề mặt tường minh   : mesh / heightfield / primitive
2. khối lượng & quán tính
3. ma sát & vật liệu
```

Stage này là nơi scene "đẹp để nhìn" trở thành scene "đúng để va chạm".

## Step R1.1 — Tách visual geometry khỏi collision geometry

Mỗi thân giữ **hai** biểu diễn:

```text
visual    : mesh gốc — nhiều tam giác, có texture, chỉ để render
collision : convex hull / convex decomposition / primitive (box, cylinder, sphere)
```

Lý do không dùng chung: solver chỉ nhanh và ổn định trên hình **lồi**. Nhét mesh lõm nguyên bản vào collision cho ra contact sai, vật xuyên nhau, và chậm hơn hàng chục lần.

Thứ tự ưu tiên khi chọn collision shape:

```text
primitive (box / cylinder)   ← rẻ nhất, ổn định nhất. Dùng khi OBB fit tốt.
convex hull                  ← vật lồi hoặc gần lồi.
convex decomposition (VHACD) ← chỉ khi hình lõm thật sự quan trọng (kệ, thùng hở, ghế).
```

## Step R1.2 — Nền scene: TSDF chứ không phải 3DGS

```text
depth Holi (đã rescale ở [2])
        ↓
TSDF fusion
        ↓
mesh nền tường minh
        ↓
        ├─→ heightfield   (locomotion, terrain)
        └─→ collision mesh (tường, cột, vật cố định)
```

## Step R1.3 — Ba quyết định đã trả giá bằng đo đạc

Ba mục dưới đây **không phải lựa chọn thẩm mỹ**. Mỗi mục là hệ quả của một lần làm sai và đo được hậu quả.

### (a) Mỗi máy chỉ được biểu diễn ĐÚNG MỘT LẦN

Heightfield tái dựng **đã chứa** các máy: 97% ô nội thất cao nằm dưới OBB của một object đã detect. Bản export trước đó thêm 43 object box lên trên → mỗi box nằm lọt trong lòng đất đặc → **63 cặp overlap, peak |qvel| = 48.8, scene không bao giờ settle**.

```text
mặc định               : máy nằm trong terrain, KHÔNG thêm body
cần manipulation       : --objects-as-bodies
                         → khoét máy khỏi terrain
                         → thêm lại dưới dạng hull body
```

Hai lựa chọn đều nhất quán. **Trộn hai lựa chọn thì không.**

### (b) Tường là box, và phải được khoét khỏi terrain trước

Heightfield **không thể có lỗ** ⇒ không thể có cửa khi tường vẫn còn là terrain. Bốn dải tường được lấp xuống mức sàn rồi thay bằng slab tường minh — slab thì mở cửa được.

### (c) Vị trí cửa lấy từ dữ liệu, kích thước cửa thì không

Ba door instance chiếu đúng lên ba mảng tường tái dựng khác nhau, một cái lệch 0.002 ⇒ **vị trí** của chúng có căn cứ. **Extent** thì không: cùng pass lift3d đó under-segment sàn đủ tệ để trượt Stage J, và một "door" chỉ còn là mảnh 0.42.

⇒ Đặt cửa **ở đúng chỗ door**, kích thước lấy theo **yêu cầu robot đi lọt**, và **ghi rõ điều đó trong file** để người đọc sau không tưởng con số đó là đo được.

## Step R1.4 — Degeneracy filter (trước khi cấp tư cách rigid body)

```text
min_extent < 0.015 m               → REJECT
min_extent / max_extent < 0.035    → REJECT
```

Mesh phẳng 2D làm dynamic body sẽ rung, lọt qua sàn, hoặc làm nổ solver. Chúng là mảnh reconstruction lỗi, không phải vật thể.

## Step R1.5 — Static vs Dynamic

```text
STATIC COLLIDER  : machine, radiator, cabinet, shelf, ladder, workbench,
                   tường, sàn, trần
DYNAMIC LIFTABLE : box, tool, cup, drawer, part, wheel, ball, crate
                   → gắn <freejoint/>
```

Quy tắc bổ sung, ưu tiên cao hơn danh sách nhãn:

```text
mass ước tính > tải trọng robot   → ép về STATIC
```

Một cái tủ 300 kg gắn `<freejoint/>` là scene sai, dù nhãn của nó nằm ở nhóm nào.

## Step R1.6 — Physics metadata

```text
mass     = volume × density(class)              [kg]
inertia  = 1/12 · m · (h² + d²)   (I_xx)        # xấp xỉ box đặc
           1/12 · m · (w² + d²)   (I_yy)
           1/12 · m · (w² + h²)   (I_zz)
friction : sliding / torsional / rolling theo vật liệu
restitution : 0.0 – 0.1 cho scene công nghiệp
```

Mọi độ dài ở đây là **mét thật**, nhờ Stage [2]. Nếu `metric_calibration.status == "UNVERIFIED"`, toàn bộ mass và inertia phải mang cờ `"estimated": true` — và cờ đó phải xuất hiện trong validation report ở Stage [12], không được nuốt mất giữa đường.

## Step R1.7 — Export

```text
MJCF  (MuJoCo)      ← primary
URDF  (ROS / PhysX)
USD   (Isaac Sim)
```

## Step R1.8 — Output

```text
stage09_robot_scene/ (hoặc OUTPUT_I/)
├── scene_factory.xml               # MJCF
├── scene_factory.urdf
├── scene_factory.usd
├── scene_factory_terrain.npy       # heightfield
├── scene_factory_terrain.png
├── scene_factory_texture.png       # top-down colour
├── scene_factory_report.json       # cell size, extents, quyết định đã chọn
├── visual/
│   └── obj_XXXX_visual.glb
├── collision/
│   ├── obj_XXXX_collision.obj      # hull / VHACD parts
│   └── obj_XXXX_parts/
└── physics_metadata.json
```

### `physics_metadata.json`

```json
{
  "scene_id": "scene_001",
  "metres_per_unit": 1.0,
  "metric_status": "VERIFIED",
  "terrain_carries_objects": true,
  "objects": [
    {
      "instance_id": "obj_0001",
      "semantic_class": "toolbox",
      "body_type": "DYNAMIC_LIFTABLE",
      "mass_kg": 4.2,
      "inertia_diag": [0.031, 0.044, 0.052],
      "friction": [0.8, 0.005, 0.0001],
      "restitution": 0.05,
      "collision_shape": "convex_hull",
      "estimated": false
    }
  ]
}
```

### Checkpoint

- [ ] Không cặp geom nào overlap tại `t = 0`.
- [ ] Mỗi máy xuất hiện đúng **một** lần (trong terrain **hoặc** là body — không cả hai).
- [ ] Cửa/lối đi đủ rộng cho robot, và file ghi rõ kích thước đó là *yêu cầu*, không phải *đo đạc*.
- [ ] Không còn mesh phẳng nào trong danh sách dynamic.
- [ ] Mọi `mass > 0`, mọi thành phần `inertia > 0`.
- [ ] MJCF compile được: `mujoco.MjModel.from_xml_path(...)` không ném lỗi.
- [ ] `metres_per_unit` trong `physics_metadata.json` khớp `metric_calibration.json`.

---

# PHẦN R2 — Robot Simulation (Stage [11])

## Step R2.1 — Backend

```text
MuJoCo        ← primary (nhẹ, deterministic, MJCF sẵn từ R1)
Isaac / PhysX ← secondary, dùng để sim2sim cross-check
Newton        ← later
```

## Step R2.2 — Spawn robot

```text
load MJCF
    ↓
chọn spawn pose từ free-space map (KHÔNG hard-code)
    ↓
settle: chạy N bước không điều khiển
    ↓
rollout: chạy policy / trajectory
```

### Cạm bẫy đã gặp: probe xuống sàn chạm vào chính robot

Ray bắn thẳng xuống từ phía trên gốc toạ độ chạm **đầu của chính robot**, và báo terrain ở `1.718` trong khi sàn ở `0`.

⇒ **Mọi probe terrain phải bỏ qua các hit không-phải-terrain trước khi tin kết quả.** Đây là lỗi im lặng: nó không crash, nó chỉ đặt robot lơ lửng hoặc chôn robot trong máy.

## Step R2.3 — Tham số chạy

```text
timestep      : 0.002 s  (locomotion)  /  0.001 s  (manipulation contact-rich)
solver        : Newton, iterations >= 50 cho scene nhiều contact
seed          : ghi lại — mọi rollout phải tái lập được
settle_time   : >= 2.0 s trước khi bắt đầu đo bất cứ thứ gì
```

## Step R2.4 — Output

```text
stage10_robot_sim/ (hoặc OUTPUT_K/)
├── rollout_000/
│   ├── qpos.npy
│   ├── qvel.npy
│   ├── contacts.jsonl
│   ├── video.mp4
│   └── run_meta.json          # seed, timestep, solver, robot model, thời gian
├── rollout_001/
└── sim_summary.json
```

### Checkpoint

- [ ] Robot đứng **trên** sàn, không lún, không lơ lửng.
- [ ] Scene settle: `peak |qvel|` hội tụ, không tăng dần.
- [ ] Contact xuất hiện đúng chỗ (chân ↔ sàn), không phải contact rác giữa các máy.
- [ ] Rollout tái lập được với cùng seed.

---

# PHẦN R3 — Robot Validation (Stage [12])

> Stage này là nơi được phép nói "pipeline chạy đúng". Trước đó tất cả chỉ là artefact.

## Step R3.1 — Danh mục kiểm tra và ngưỡng

```text
[1] collision t=0     : 0 cặp interpenetrating
[2] stability         : peak |qvel| < 5.0 sau 2 s settle
                        (tham chiếu lỗi cũ: 48.8 khi máy bị đếm hai lần)
[3] clearance         : min(door_width, corridor_width) >= robot_width + 0.10 m
[4] navigation        : tồn tại path start → goal trên free-space map
[5] traversability    : slope <= slope_max(robot)
                        step  <= step_max(robot)
[6] success rate      : >= 80% trên N >= 20 trial, seed khác nhau
[7] sim2sim           : sai lệch trajectory MuJoCo vs Isaac <= 10% cùng control
[8] scale sanity      : metric_calibration.status == "VERIFIED"
```

Điều kiện `[8]` **không chặn** pipeline, nhưng nếu là `UNVERIFIED` thì `[3]`, `[5]`, `[6]` phải được báo cáo là **chưa kiểm chứng** — vì cả ba đều là đại lượng có đơn vị mét.

## Step R3.2 — Kiểm tra hướng heightfield: chỉ đặt câu hỏi ở chỗ trả lời được

So chiều cao ray với mảng nguồn **chỉ well-posed trên nền phẳng cục bộ**. Ở mặt bên thẳng đứng của một cột máy, ray cắt vào vách cột và rơi lệch hơn một mét — đó là tính chất của **phép probe**, không phải của terrain.

⇒ Check hướng chỉ chấm điểm trên các ô phẳng, và **báo cáo riêng** phần đuôi vách dựng đứng. Trộn hai loại ô vào một con số trung bình sẽ tạo ra một lỗi không có thật, rồi tốn thời gian đi sửa terrain vốn không sai.

## Step R3.3 — Output

```text
stage11_robot_validation/ (hoặc OUTPUT_L/)
├── validation_report.json
├── validation_report.md
├── clearance_map.png
├── navigation_path.png
├── qvel_settle_curve.png
└── failures/
    └── <ảnh + log của từng check trượt>
```

### `validation_report.json`

```json
{
  "scene_id": "scene_001",
  "robot": "unitree_g1",
  "metric_status": "VERIFIED",
  "checks": {
    "collision_t0":    { "pass": true,  "interpenetrating_pairs": 0 },
    "stability":       { "pass": true,  "peak_qvel": 0.0 },
    "clearance":       { "pass": true,  "min_clearance_m": 0.0, "robot_width_m": 0.0 },
    "navigation":      { "pass": true,  "path_found": true, "path_length_m": 0.0 },
    "traversability":  { "pass": true,  "max_slope_deg": 0.0, "max_step_m": 0.0 },
    "success_rate":    { "pass": true,  "trials": 20, "rate": 0.0 },
    "sim2sim":         { "pass": null,  "note": "chưa chạy Isaac" }
  },
  "overall": "PASS",
  "unverified_because_of_scale": []
}
```

### Checkpoint

- [ ] Mỗi check **có khả năng fail thật** — check không bao giờ fail được là check vô nghĩa.
- [ ] Check nào chưa chạy thì ghi `null`, **không** ghi `true`.
- [ ] Mọi con số có đơn vị đều là mét, và truy được về `metres_per_unit`.
- [ ] Ảnh failure được lưu lại, không chỉ có dòng log.

---

# PHẦN L — Script nên viết theo thứ tự

Đừng viết một script 2,000 dòng.

Tách thành:

```text
scripts/
├── 00_metric_calibration.py            # [2]      ⏳ CHƯA VIẾT
├── 01_normalize_holi_output.py         # [3]      ✅
├── 02_quality_gate_and_adapter_a.py    # [4]+[5]  ✅
├── 04_run_hunyuan_shape.py             # [6]      ✅
├── 05_align_and_verify_adapter_b.py    # [7]+[8]  ✅
├── 06_assemble_final_scene.py          # [9]      ✅
├── 07_texture_bake_objects.py          # optional ✅
├── 10_robotics_manipulation_gate.py    # [10] physics metadata + static/dynamic  ✅
├── make_unitree_scene.py               # [10] MJCF + terrain + tường/cửa         ✅
├── render_unitree_scene.py             # [11] rollout + video                    ✅
└── validate_unitree_scene.py           # [12] validation report                  ✅
```

> Đây là **tên file thật trong repo**, không phải tên lý tưởng. Một vài script gộp hai stage (`02_` gộp [4]+[5], `05_` gộp [7]+[8]) — chấp nhận được vì hai stage đó luôn chạy cùng nhau và dùng chung dữ liệu đã nạp.

## Output của từng script

| Script | Stage | Input | Output |
|---|---|---|---|
| `00_metric_calibration.py` | [2] | Holi raw + 1 phép đo thật | `OUTPUT_B2/metric_calibration.json` + scene đã rescale |
| `01_normalize_holi_output.py` | [3] | Holi raw output (metric) | `OUTPUT_C1/` object packages |
| `02_quality_gate_and_adapter_a.py` | [4]+[5] | object package | `coverage_report.csv`, `OUTPUT_D/adapter_a/` |
| `04_run_hunyuan_shape.py` | [6] | `mv_inputs/` hoặc `hunyuan_input.png` | `OUTPUT_F/mesh_raw.*` |
| `05_align_and_verify_adapter_b.py` | [7]+[8] | mesh + metric OBB + cameras/masks/depth/points | `OUTPUT_G/mesh_aligned_final.*` + metrics |
| `06_assemble_final_scene.py` | [9] | accepted meshes + nền | `OUTPUT_I/` hybrid scene |
| `07_texture_bake_objects.py` | opt | aligned mesh + RGB | `OUTPUT_J/` textured mesh |
| `10_robotics_manipulation_gate.py` | [10] | `OUTPUT_G/` | static/dynamic tag, mass, inertia, friction |
| `make_unitree_scene.py` | [10] | scene + terrain + `--metres-per-unit` | `scene_factory.xml` (MJCF) |
| `render_unitree_scene.py` | [11] | MJCF + robot | `OUTPUT_K/` rollout, video |
| `validate_unitree_scene.py` | [12] | MJCF + report + terrain | `OUTPUT_L/validation_report.json` |

---

# PHẦN M — MVP tối thiểu nên làm trước

Không làm toàn bộ ngay.

## MVP-0 — Metric calibration (làm trước tất cả)

Rẻ nhất, và là thứ duy nhất không thể sửa sau. Một phép đo bằng thước là đủ.

```text
Holi scene
    ↓
đo 1 khoảng cách thật + 1 khoảng cách verify
    ↓
metric_calibration.json  (status = VERIFIED)
```

### OUTPUT bắt buộc

```text
MVP_0/
├── metric_calibration.json
├── reference_overlay.png
└── sanity_check.txt        # chiều cao cửa / bàn / ghế đo lại từ scene
```

**Nếu không làm MVP-0, mọi MVP sau vẫn chạy được — nhưng không con số mét nào trong chúng đáng tin.**

---

## MVP-1

Chỉ chạy **1 object**:

```text
Holi output
    ↓
obj_0001
    ↓
Adapter A
    ↓
Hunyuan Shape
    ↓
Adapter B: OBB scale/rotation/translation
    ↓
render comparison
```

### OUTPUT bắt buộc

```text
MVP_1/
├── holi_before.png
├── hunyuan_input.png
├── mesh_raw.obj
├── mesh_aligned.obj
├── aligned_front.png
├── aligned_side.png
├── aligned_back.png
└── metrics.json
```

**Nếu MVP-1 chưa align đúng thì dừng. Không thêm Paint.**

---

## MVP-2

Thêm:

```text
multi-view silhouette IoU
+
depth verification
```

### OUTPUT

```text
MVP_2/
├── mesh_aligned_final.obj
├── transform_final.json
└── alignment_metrics.json
```

---

## MVP-3

Sau khi MVP-2 ổn mới thêm:

```text
Hunyuan3D-Paint
```

### OUTPUT

```text
MVP_3/
└── mesh_textured.glb
```

---

## MVP-4 — Robot-ready (Stage [10]–[12])

Chỉ bắt đầu khi MVP-2 pass và MVP-0 ở trạng thái `VERIFIED`.

```text
hybrid scene
    ↓
collision + physics metadata
    ↓
MJCF compile
    ↓
spawn robot + settle
    ↓
validation report
```

### OUTPUT bắt buộc

```text
MVP_4/
├── scene_factory.xml
├── physics_metadata.json
├── settle_qvel_curve.png
└── validation_report.json
```

**Tiêu chí dừng của MVP-4:** `peak |qvel| < 5.0` sau 2 s settle và `interpenetrating_pairs == 0`. Chưa đạt hai điều đó thì chưa cần bàn tới navigation hay success rate.

---

# PHẦN N — Tiêu chí quyết định pipeline đã chạy đúng

## N.1 — Ở mức object

Pipeline được xem là **PASS** cho một object nếu:

```text
[1] Adapter A crop đúng object.
[2] Hunyuan sinh đúng loại object.
[3] Mesh nằm đúng 3D metric OBB.
[4] Không bị lật front/back.
[5] Silhouette tương đối khớp ở nhiều observed views.
[6] Depth không lệch nghiêm trọng.
[7] Kích thước khớp trên các trục ĐÃ QUAN SÁT ĐƯỢC (dimension_error).
[8] Mesh bám vào observed_points thật (surface_rmse_rel, inlier_ratio).
[9] Backside đã được completion.
[10] Nếu alignment fail → tự động fallback về Holi object, VÀ ghi rõ lý do fail.
```

## N.2 — Ở mức scene

```text
[11] Đúng một metres_per_unit cho cả scene, có tham chiếu verify độc lập.
[12] Mọi độ dài trong scene cuối là mét, truy được về metric_calibration.json.
[13] Mỗi vật thể vật lý xuất hiện đúng một lần (terrain HOẶC body).
[14] MJCF/URDF/USD compile được.
```

## N.3 — Ở mức robot

```text
[15] 0 cặp geom interpenetrating tại t = 0.
[16] peak |qvel| < 5.0 sau 2 s settle.
[17] Robot đứng trên sàn, không lún, không lơ lửng.
[18] Clearance đủ cho robot đi qua mọi lối bắt buộc.
[19] Tồn tại path start → goal.
[20] Success rate >= 80% trên >= 20 trial.
```

> Đạt `[1]`–`[10]` chỉ có nghĩa là **object đã được completion đúng**. Chỉ khi đạt tới `[20]` mới được nói **"pipeline chạy đúng"**.

---

# PHẦN O — Điều KHÔNG làm ở Version 1

Để tránh phức tạp không cần thiết:

```text
❌ Không sửa Holi network
❌ Không fine-tune Holi
❌ Không sửa Hunyuan network
❌ Không fine-tune Hunyuan
❌ Không thêm neural adapter
❌ Không merge mesh topology trực tiếp với 3DGS
❌ Không convert Hunyuan mesh thành Gaussian ngay
❌ Không chạy Paint trước khi geometry alignment ổn
```

Version đầu chỉ cần:

```text
Holi
  ↓
Metric Calibration
  ↓
Adapter A
  ↓
Hunyuan Shape
  ↓
Adapter B
  ↓
Hybrid Scene
  ↓
Robot-Ready Scene
```

## Hai điều KHÔNG được bỏ qua, kể cả ở Version 1

```text
✅ PHẢI có metric calibration (Stage [2]) — dù chỉ là một phép đo bằng thước.
   Không có nó thì Stage [10]–[12] không có nghĩa, và lỗi không lộ ra
   cho tới khi đã tốn toàn bộ compute.

✅ PHẢI tách visual mesh khỏi collision mesh (Stage [10]).
   Nhét mesh render thẳng vào collision là con đường ngắn nhất tới
   scene không settle được.
```

---

# PHẦN P — Final checklist

## Input

- [ ] RGB sequence/video sẵn sàng.
- [ ] Holi-Spatial chạy baseline thành công.
- [ ] Có ít nhất **một** phép đo thật trong cảnh (hoặc marker, hoặc frame RGB-D).

## Holi output

- [ ] 3DGS.
- [ ] Camera intrinsics.
- [ ] Camera extrinsics.
- [ ] Depth.
- [ ] 2D masks.
- [ ] 3D OBB.
- [ ] Best source image.

## Metric calibration (Stage [2])

- [ ] `metres_per_unit` tính từ tham chiếu thật.
- [ ] Verify bằng tham chiếu **thứ hai, độc lập**.
- [ ] `residual_pct <= 5%`.
- [ ] Đã rescale: cameras, depth, points, 3DGS, OBB.
- [ ] `status` (`VERIFIED` / `UNVERIFIED`) được truyền xuống mọi stage sau.

## Adapter A

- [ ] Crop object.
- [ ] Remove background.
- [ ] Center.
- [ ] 518×518.
- [ ] Lưu metadata.

## Hunyuan

- [ ] Sinh bare mesh.
- [ ] Mesh có backside.
- [ ] Lưu raw mesh.

## Adapter B

- [ ] Normalize mesh.
- [ ] Scale **per-axis** (sx, sy, sz) theo metric OBB.
- [ ] Trục low-confidence dùng `s_ref`, không dùng extent Holi.
- [ ] Chốt chặn anisotropy `<= 1.5`.
- [ ] Translate theo OBB center.
- [ ] Test yaw candidates.
- [ ] Multi-view mask check.
- [ ] Depth check.
- [ ] Dimension error check (chỉ trên trục confident).
- [ ] Observed-surface consistency check.
- [ ] Lưu final transform + lý do accept/reject.

## Final scene

- [ ] Accepted mesh → add vào hybrid scene.
- [ ] Rejected mesh → fallback về Holi, có ghi lý do.
- [ ] Render before/after.
- [ ] Lưu metrics.

## Robot-ready (Stage [10])

- [ ] Visual mesh tách khỏi collision mesh.
- [ ] Collision là primitive / hull / VHACD, không phải mesh lõm nguyên bản.
- [ ] Degeneracy filter đã loại mesh phẳng khỏi dynamic.
- [ ] Static / dynamic tag đúng, có kiểm tra theo tải trọng robot.
- [ ] Mass / inertia / friction / restitution đầy đủ và dương.
- [ ] Mỗi vật thể xuất hiện đúng một lần.
- [ ] Tường là slab tường minh, cửa mở được.
- [ ] MJCF / URDF / USD compile được.

## Robot simulation (Stage [11])

- [ ] Spawn pose lấy từ free-space map.
- [ ] Probe terrain bỏ qua hit không-phải-terrain.
- [ ] Settle >= 2 s trước khi đo.
- [ ] Seed + timestep + solver được ghi lại.

## Robot validation (Stage [12])

- [ ] 0 cặp interpenetrating tại t = 0.
- [ ] `peak |qvel| < 5.0`.
- [ ] Clearance / navigation / traversability đã đo.
- [ ] Success rate trên >= 20 trial.
- [ ] Check chưa chạy ghi `null`, không ghi `true`.
- [ ] Nếu scale `UNVERIFIED`: đã liệt kê rõ những kết luận nào chưa kiểm chứng.

---

# Kết luận triển khai

Thứ tự nên làm thực tế:

```text
DAY/PHASE 0
Holi baseline
    ↓
ĐO 1 khoảng cách thật → metric calibration
    ↓
normalize output (metric)

DAY/PHASE 1
chọn 1 object lỗi
    ↓
Adapter A

DAY/PHASE 2
Hunyuan Shape
    ↓
raw mesh

DAY/PHASE 3
Adapter B initial alignment (per-axis scale)
    ↓
candidate meshes

DAY/PHASE 4
multi-view verification
(mask IoU + depth + dimension + observed-surface)
    ↓
accepted aligned mesh

DAY/PHASE 5
Hybrid scene
    ↓
before / after evaluation

DAY/PHASE 6
Robot-ready compiler
    ↓
collision + physics + MJCF

DAY/PHASE 7
Robot simulation + validation
    ↓
validation_report.json

ONLY AFTER THAT
    ↓
Hunyuan3D-Paint
```

> Paint nằm cuối cùng vì texture không ảnh hưởng tới bất kỳ kết luận nào ở Stage [10]–[12]. Làm Paint trước validation là đổi compute lấy ảnh đẹp trong khi geometry có thể vẫn sai.

**Output quan trọng nhất của toàn bộ Version 1:**

```text
OUTPUT_B2/
└── metric_calibration.json          ← không có file này thì mọi con số phía dưới là vô đơn vị

final_scene/ (OUTPUT_I/)
├── holi_3dgs/
├── completed_meshes/
├── scene_manifest.json
├── scene_factory.xml                ← MJCF, robot-ready
└── physics_metadata.json

OUTPUT_L/
└── validation_report.json           ← thứ duy nhất được phép dùng để nói "chạy đúng"
```

và với từng object:

```text
obj_XXXX/
├── hunyuan_input.png
├── mesh_raw.obj
├── mesh_aligned_final.obj
├── transform_final.json        # scale_xyz per-axis
└── alignment_metrics.json      # iou / depth / dimension / surface + lý do accept-reject
```

Đây là version ít thay đổi kiến trúc nhất, dễ debug nhất, và đủ để trả lời hai câu hỏi tách bạch:

1. **Hunyuan3D có sửa được lỗi backside của object trong Holi-Spatial không?** → trả lời bởi Stage [8], `alignment_metrics.json`.
2. **Scene sau khi sửa có dùng được cho robot không?** → trả lời bởi Stage [12], `validation_report.json`.

Câu hỏi thứ nhất là về **completion**. Câu hỏi thứ hai là về **tính dùng được**. Trả lời được câu một mà không trả lời câu hai thì kết quả mới chỉ là một mesh đẹp hơn, chưa phải một scene chạy được.
