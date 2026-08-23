# Context — room mesh → robot scene → policy walking

> **Phiên:** 2026-08-20 · nhánh `v3d-rewrite`
> **Đầu vào:** `outputs-full-factory/delivery/OUTPUT_B1/mesh/room_300k.glb`
> **Đầu ra:** `outputs-full-factory/delivery/OUTPUT_ROOM/`
> **Mục tiêu:** một căn phòng thật, có robot đứng trong đó, vật lý thật, rồi đi bằng policy.

Tài liệu phương pháp: [`../guides/room-to-robot-scene.md`](../guides/room-to-robot-scene.md)
Danh mục lỗi: [`../reference/bugs-found.md`](../reference/bugs-found.md) B24–B28

---

## 1. Pipeline

```bash
make room-scene     # mesh phòng + collider + robot → MJCF, kiểm, render
make room-play      # nạp policy ONNX, mặc định đứng yên
make room-serve     # render 3 video + phục vụ http://127.0.0.1:8080
make room-view      # mở MuJoCo viewer
```

Tham số ghi đè:

```bash
make room-scene ROOM=path/to.glb ROOM_GT=path/to_gt.json
make room-play  POLICY=/path/policy_13999.onnx CMD="0.5 0 0"
make room-serve PORT=9000
```

Cờ chỉ có ở lời gọi trực tiếp:

```
--no-furniture        chỉ dựng shell, không collider đồ đạc — đây là PHÉP CONTROL
--no-test             bỏ hai phép kiểm vật lý
--no-render           bỏ ảnh kiểm tra
--gt                  tuỳ chọn; thiếu thì độ cao sàn lấy theo phân vị 1%
```

### Sơ đồ

```text
room_300k.glb  (Z chỉ xuống, 688 mảnh rời, không kín)
      │
      ├─ xoay 180° quanh X  (det=+1, KHÔNG lật Z)
      ├─ hạ sàn về z=0      (độ cao lấy từ GT, không lấy min)
      └─ assert: trần phải ở z > 0.5                    ← chặn scene lộn ngược
      │
      ├──► visual   room_visual.obj    contype=0, không bao giờ va chạm
      │
      └──► collider 1 plane + 5 box shell
                    + 79 box đồ đạc từ 223 OBB ground-truth
      │
      ├─ find_spawn()  lưới 120×120 → điểm thoáng nhất + con số clearance
      │
      ├──► room_scene.xml    robot + quả cầu control + keyframe (ghi sau khi compile)
      └──► room_terrain.xml  chỉ terrain, nq=0, cho mjlab spec_fn
                 + room_terrain.py   spec_fn() và SPAWN_XY
```

---

## 2. Số đo đã kiểm chứng

### Hình học phòng

```
sàn      z = [-0.204, 0.132]  →  z = 0        nguồn: GT ins=15
trần     z = [-3.731, -2.922] →  z = +3.29    nguồn: GT ins=2
cửa      cao 2.37–2.48 m                       ⇒ k ≈ 1, scene gần đúng mét
kích thước sàn 8.15 × 8.26 m,  cao 3.42 m
mesh: 300.000 mặt, 688 mảnh rời, watertight=False, Euler=−19566
```

### Phép kiểm vật lý

| | kết quả |
|---|---|
| quả cầu control | 1,5 m → **0,0996 m** (bán kính 0,1), `v=0` — **PASS** |
| robot, home pose | thân thấp nhất `z=0,0`, không chìm, 17 contact — **PASS** |
| spawn | `[4.7571, −2.0306]`, clearance **1,007 m**, contact lúc t=0: **2** (hai bàn chân) |

### Policy `policy_12000.onnx`, 10 giây

| scene | lệnh | quãng đường | tốc độ | uprightness | kết quả |
|---|---|---|---|---|---|
| có đồ đạc | `0 0 0` | 0,05 m | — | 0,9995 | **STANDING** |
| **không đồ đạc** | `0.5 0 0` | **3,85 m** | **0,385 m/s** | 0,999 | **WALKING** |
| không đồ đạc | `1.0 0 0` | 3,84 m | 0,384 m/s | 0,997 | bão hoà ~0,38 |
| có đồ đạc | `0.5 0 0` | 1,20 m | 0,120 m/s | 0,994 | **BLOCKED** |

Ca cuối: đi 1,2 m trong 4 giây đầu (**0,3 m/s**), rồi đụng `obs_134_wood`,
`obs_136_wood`, `obs_162_panel` và dừng hẳn 6 giây còn lại.

**Policy đi được.** Nó dừng vì **va chạm hoạt động đúng**, không phải vì policy dở.

---

## 3. Hợp đồng policy (`deploy_12000.yaml`)

```text
obs 475 = base_ang_vel 3×5 + projected_gravity 3×5 + velocity_commands 3×5
        + joint_pos_rel 28×5 + joint_vel_rel 28×5 + last_action 28×5
        + gait_phase 2×5
actions 28,  step_dt 0.02 s  →  10 substep ở timestep 0.002
```

Ba điều bắt buộc, sai thì **không có gì báo lỗi**:

1. **History cũ trước mới sau**, mỗi term làm phẳng riêng rồi nối theo thứ tự khai
   báo. Sai thứ tự vẫn đủ 475 số, mạng vẫn chạy, robot chỉ ngã hợp lý.
2. **Actuator là mô-men** (`gainprm=1, biasprm=0`) → PD phải làm ngoài, bằng
   `stiffness`/`damping` trong YAML: `tau = kp(action·scale + offset − q) − kd·qd`.
   *(Đã thử actuator implicit của MuJoCo: kết quả trùng khít, không phải nguyên
   nhân của việc đi chậm.)*
3. **Đứng yên là pha gait riêng.** `mdp.phase` zero cả hai kênh khi
   `‖command‖ < 0.1`.

**Policy không có cảm biến vật cản.** Cả 475 chiều đều là bản thể + lệnh + gait.
Nó mù trước đồ đạc và sẽ đâm vào — đúng thiết kế, cần biết trước khi diễn giải.

---

## 4. Cắm vào `play.py` của mjlab

Đã kiểm chứng trong venv của repo robot: terrain compile ra `nbody 1, ngeom 86,
**nq 0**`, 85 collider ở **group 0**, và `TerrainEntityCfg(spec_fn=...)` nhận.

```python
import sys; sys.path.insert(0, ".../OUTPUT_ROOM")
from room_terrain import spec_fn, SPAWN_XY
from mjlab.terrains import TerrainEntityCfg

env_cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane", spec_fn=spec_fn)
env_cfg.scene.entities["robot"].init_state.pos = (*SPAWN_XY, 0.841)
```

Ba ràng buộc:

- `nq = 0` là bắt buộc. Terrain mang một free body sẽ làm lệch mọi địa chỉ `qpos`
  mjlab tính.
- **group 0** vì `terrain_scan` và `foot_height_scan` lọc `include_geom_groups=(0,)`.
- **Phải đổi spawn.** Mặc định của mjlab là gốc toạ độ, chỗ đó nằm trong máy.

Nên tắt `out_of_terrain_bounds` (termination) và `terrain_levels` (curriculum) —
cả hai giả định terrain sinh theo lưới ô.

Mô hình robot khớp: mjlab và scene này đều cho **82,6 kg**, `nq 35`.

---

## 5. Bảy lỗi im lặng đã gặp

Không cái nào ném exception. Mỗi cái đều tạo ra một scene chạy được nhưng sai.

| # | Lỗi | Triệu chứng đo được |
|---|---|---|
| B24 | Trọng lực ngược chiều | Z tái dựng chỉ xuống; `gravity -9.81` đẩy robot lên trần |
| B25 | Mesh phòng làm collider | MuJoCo lấy bao lồi → khối đặc, robot bị đẩy ra ngoài |
| B26 | Keyframe ghép tay | 2 freejoint, `nq=42` ≠ 35; home pose rơi vào quả cầu |
| B27 | Texture | xatlas treo (688 mảnh); chiếu hộp thì sàn và mặt bàn tranh texel |
| B28 | Spawn ở tâm phòng | Trong `obs_55_machine`: 88 contact, xuyên 0,57 m |
| — | Verdict thiếu quãng đường | In "STANDING" cho lệnh đi tới mà không đo di chuyển |
| — | Tốc độ trung bình toàn cuộc | Gộp 6 giây bị chặn → đo vật cản chứ không đo policy |

---

## 6. Bất biến

1. **Xoay 180° quanh X, không lật Z.** `det` phải `+1`; lật một trục là phản chiếu.
2. **Mesh phòng luôn `contype=0`.**
3. **Quả cầu control luôn có trong scene** — nó tách "sàn hỏng" khỏi "robot ngã".
4. **Keyframe ghi sau khi compile**, theo `jnt_qposadr` thật.
5. **Không include `scene.xml` của robot** — nó mang sàn thứ hai.
6. **Độ cao sàn từ GT hoặc phân vị**, không `min()`.
7. **Spawn phải tìm và ghi lại clearance.**
8. **Chạy control PD trước khi tin policy.**
9. **Observation dựng từ `deploy_*.yaml`**, không hardcode.
10. **Verdict phải đối chiếu với LỆNH.** Đứng và đi cần hai tiêu chí khác nhau;
    và phải phân biệt "bị chặn" với "không chịu đi".

---

## 7. Còn treo

- **Chưa calibrate mét (Stage [2]).** `OUTPUT_B2/` chưa tồn tại,
  `00_metric_calibration.py` chưa viết. Sàn 8,15 × 8,26 m và cửa 2,37–2,48 m
  gợi ý `k ≈ 1`, nhưng đó là *trông quen*, không phải phép đo. Có sẵn 223 OBB
  ground-truth để giải `k` kèm phân bố sai số.
- **Đồ đạc là hộp bao**, robot không luồn được xuống gầm bàn.
- **Tường/trần là hộp bao quanh footprint**, không phải hình dạng thật.
- **Tốc độ tính trung bình toàn cuộc.** `BLOCKED` chưa nói "trước khi bị chặn thì
  đi 0,3 m/s" — nên cắt ở lần chạm vật cản đầu tiên.
- **Lệnh 1.0 chỉ ra 0,384 m/s.** Đặc tính policy, không phải lỗi harness.
- **Warp trên máy này chỉ có CPU** (`CUDA not enabled`), nên `play.py` qua
  `mujoco_warp` sẽ chậm. Người dùng xác nhận máy của họ chạy được.

---

## 8. File

```text
scripts/sim/
├── make_room_scene.py     mesh → MJCF + terrain export, hai phép kiểm, render
├── play_policy.py         chạy policy ONNX, phán quyết theo lệnh
└── serve_room.py          render 3 video + phục vụ HTTP

outputs-full-factory/delivery/OUTPUT_ROOM/
├── room_scene.xml         robot + control sphere + keyframe
├── room_terrain.xml       chỉ terrain, nq=0, cho mjlab
├── room_terrain.py        spec_fn() + SPAWN_XY
├── room_scene_report.json orientation, shell, obstacles, spawn, hai phép kiểm
├── policy_play_report.json quãng đường, tốc độ, uprightness, blocked_by
├── room_robot_check.png   4 khung dựng scene
├── policy_play.png        4 thời điểm chạy policy
├── policy_{robot,room,topdown}.mp4
├── index.html             trang xem trên :8080
└── assets/                room_visual.obj + 31 STL robot
```

Phụ thuộc thêm ở phiên này: `onnxruntime` (đã khai báo trong `requirements.txt`
và `pyproject.toml`).
