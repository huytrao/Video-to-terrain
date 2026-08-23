# Từ mesh phòng đến scene robot đứng được — có vật lý thật

Cách biến `room_300k.glb` thành một scene MuJoCo mà robot đứng **trong** phòng,
với trọng lực đúng chiều và va chạm dùng được. Một lệnh:

```bash
make room-scene
```

Chạy trên scene nhà xưởng `00777c41d4`. Script: `scripts/sim/make_room_scene.py`.

---

## 1. Ba thứ khiến mesh phòng không dùng được làm physics, và cả ba đều im lặng

### 1.1 Trục Z chỉ XUỐNG

Đo trực tiếp trên `ground_truth_boxes.json`:

```text
floor    ins=15   z = [-0.204,  0.132]
ceiling  ins= 2   z = [-3.731, -2.922]
door     ins= 6   z = [-2.455,  0.027]   cao 2.48 m
```

Sàn ở `z ≈ 0`, trần ở `z ≈ −3,3`, và cửa chạy từ sàn **lên** tới `z = −2,46`.
Trục Z của bản tái dựng chỉ xuống đất.

Nên `gravity="0 0 -9.81"` trong `factory_scene.mjcf` đang **đẩy robot về phía
trần**. Không có gì báo lỗi; robot chỉ đơn giản bay lên và trông như "vật lý hỏng".

**Phép sửa phải là xoay 180° quanh X**, `(x, y, z) → (x, −y, −z)`:

```python
FLIP_X180 = [[1, 0, 0, 0],
             [0,-1, 0, 0],
             [0, 0,-1, 0],
             [0, 0, 0, 1]]
```

Không phải lật riêng Z. Lật một trục là **phép phản chiếu** (`det = −1`) — nó biến
thế giới thành ảnh trong gương, và một robot thuận tay phải bước ra thành thuận
tay trái. Xoay 180° quanh X có `det = +1`, giữ nguyên chiều thuận.

Sau khi xoay, hạ sàn về `z = 0`. Độ cao sàn lấy từ ground truth chứ **không** lấy
đỉnh thấp nhất của mesh — đỉnh thấp nhất của một phòng tái dựng thường là nhiễu
lơ lửng dưới sàn. Không có GT thì dùng phân vị 1%, không dùng min.

Có một assert chặn ngay tại đây: sau biến đổi, trần phải ở `z > 0,5`. Sai chiều
thì script dừng, chứ không đẩy một scene lộn ngược xuống các stage sau.

### 1.2 MuJoCo coi mọi mesh geom là BAO LỒI

Đây là điểm chí mạng, và nó không có trong bất kỳ thông báo lỗi nào.

Bao lồi của một căn phòng là một **khối đặc**. Nạp thẳng `room_300k.glb` làm
collider thì robot không đứng trong phòng — nó bị đẩy ra ngoài khối.

Nên phải tách hai vai:

```text
mesh phòng   →  chỉ để NHÌN     contype="0" conaffinity="0" group="2"
collider     →  primitive       1 plane (sàn) + 5 box (4 tường + trần)
```

Đây chính là quy tắc visual-vs-collision mà `architecture/initial-setup.md` đã đặt
ra từ đầu, và là lý do một căn phòng không bao giờ nạp thẳng vào physics engine
được.

### 1.3 Không dán texture — và vì sao

`room_300k.glb` mang màu **theo từng đỉnh** (163.808 đỉnh), đó là cách
photogrammetry xuất màu. MuJoCo chỉ nhận `<material>` + `<texture>` + toạ độ UV,
nên vertex color bị bỏ qua **trong im lặng** và phòng render ra xám trơn.

Đã thử hai đường để đưa màu vào, cả hai đều bỏ:

**`xatlas` unwrap.** Là câu trả lời đúng sách, và sai công cụ ở đây. Mesh này là
bản tái dựng chứ không phải model: **688 mảnh rời**, `is_watertight = False`,
`Euler = −19.566`. xatlas sinh hàng nghìn chart tí hon rồi tắc ở khâu đóng gói —
hơn 4 phút trên 80.000 tam giác, vẫn chưa xong.

**Phép chiếu hộp 6 mặt.** Nhanh (3,8 giây) và đúng về màu, nhưng nhoè nặng. Lý do
mang tính cấu trúc, không sửa được bằng tinh chỉnh: trong ô `+Z` nhìn từ trên, mặt
sàn và **mọi mặt bàn** đều chiếu chồng lên cùng một texel. Chúng tranh nhau, và
kết quả xấu hơn là không có màu.

Nên phòng để **flat-shaded**. Hình khối đọc rõ hơn hẳn, và không có gì trong
physics phụ thuộc vào màu. Nếu sau này cần ảnh thật, đường đúng là **bake lại từ
ảnh gốc qua camera pose** (đã có sẵn trong `OUTPUT_C1/*/cameras/`), chứ không phải
tham số hoá lại một mesh vỡ vụn.

---

## 2. Đặt robot: cái bẫy keyframe

Robot Unitree đi kèm `<key name="home" qpos="0 0 0.841 1 0 0 0 ...">` — pelvis cao
0,841 m, rồi 28 góc khớp. Tổng 35 số.

Ghép tay vector 35 số đó vào scene thì **sai**, vì scene có **hai** freejoint:
quả cầu control và pelvis robot. `nq = 42`, không phải 35. Quả cầu khai báo trước
nên nó chiếm `qpos[0:7]`, và toàn bộ home pose của robot rơi vào quả cầu — còn
pelvis nhận một lát góc khớp làm toạ độ.

Triệu chứng khi đó: `pelvis z = 0.0` thay vì `0.841`, và không có cảnh báo nào,
vì vector chỉ đơn giản là **ngắn**.

Phép sửa là biên dịch scene trước, rồi ghi keyframe theo địa chỉ thật:

```python
model = mujoco.MjModel.from_xml_path(xml)          # pass 1
root  = model.jnt_qposadr[free_joint_of("pelvis")]
qpos  = model.qpos0.copy()
qpos[root:root+7] = home[:7]
qpos[root+7:]     = home[7:]                        # rồi ghi lại XML
```

Cùng lý do đó, `data.ctrl` phải đọc qua `actuator_trnid` → `jnt_qposadr` chứ không
cắt `qpos[7:7+nu]`.

Và **không include `scene.xml` của robot**: nó mang theo một `floor` plane riêng,
sẽ đánh nhau với sàn của phòng. Chỉ lấy `_rl.xml`, `_sensors.xml`, `_actuators.xml`,
và gỡ `<compiler meshdir="../meshes/">` khỏi chúng vì đường dẫn đó tính theo file
gốc, không còn đúng khi file được copy sang chỗ khác.

---

## 3. Hai phép kiểm, và vì sao cần cả hai

### 3.1 Quả cầu control

Một quả cầu thả từ 1,5 m. Nó phải dừng ở đúng bán kính của nó.

```text
sphere 1.5 → 0.0996 m   (bán kính 0.1)   |v| = 0   PASS
```

**Đây là control, không phải trang trí.** Nếu quả cầu rơi xuyên sàn thì collider
sai, và mọi thứ robot làm sau đó không chứng minh được điều gì. Kiểm robot mà
không có control là kiểm hai thứ cùng lúc và không biết cái nào hỏng.

### 3.2 Robot

Giữ khớp ở home pose rồi sim 3 giây. Phép kiểm **cố tình không** yêu cầu robot
đứng vững — không có bộ điều khiển thăng bằng thì humanoid phải ngã, và điều đó
chẳng nói gì về căn phòng.

Thứ được hỏi là **sàn có đỡ nổi không**:

```text
pelvis 0.841 → 0.274 m      ngã, đúng như dự kiến
lowest body z = 0.0         không chìm một mm nào
contacts = 14               tiếp xúc thật
diverged = False            không nổ
```

Robot chìm xuống sàn hay bị bắn ngang phòng là lỗi va chạm, không phải lỗi điều
khiển. Đó là ranh giới mà phép kiểm này vạch ra.

### 3.3 Nhìn ảnh

`room_robot_check.png` — bốn khung: robot t=0, robot t=3s, cả phòng kèm collider
trong suốt, và nhìn từ trên xuống.

Một scene biên dịch được và qua cả hai phép kiểm vẫn có thể sai theo cách chỉ mắt
bắt được: robot quay mặt vào tường, phòng lộn trong ra ngoài, texture map nhầm
chart. Ba lỗi trong dự án này được tìm ra đúng bằng cách mở ảnh lên sau khi mọi
kiểm tra tự động đã báo xanh.

---

## 4. Va chạm đồ đạc, và điểm spawn phải được tìm

### 4.1 Mesh phòng không cho được collider đồ đạc

300.000 tam giác của nó là **một thành phần liên thông duy nhất** trải khắp phòng
— tường, sàn và mọi cái máy dính liền thành một mặt. Tách thành phần cho ra đúng
một khối; bao lồi khối đó lại chính là cục đặc mà mục 1.2 đã tránh.

Nhưng annotation thì **đã là** một danh sách hộp có hướng, tính bằng mét. Mỗi hộp
thành một `box` geom. Đó cũng là độ chi tiết đúng: robot cần không đi xuyên qua
cái bàn, chứ không cần thò tay vào hộc kéo.

Lọc bỏ nhãn không phải vật cản (`sticker`, `label`, `light`, `glass`, `sign`…, và
`floor`/`wall`/`ceiling` vì shell đã lo), bỏ hộp nhỏ hơn 0,20 m hoặc dưới
0,01 m³. Còn lại **79 hộp** trên scene này.

### 4.2 Tâm phòng không phải sàn trống

Spawn robot ở tâm phòng đặt nó **bên trong `obs_55_machine`**: 88 điểm tiếp xúc,
xuyên sâu tới **0,57 m**, mọi bộ phận từ chậu hông tới cổ tay đều nằm trong cái
máy. Solver bắn nó ra và nó ngã.

Điều nguy hiểm là **triệu chứng đọc như lỗi policy**. Robot ngã, và nếu không đo
thì kết luận tự nhiên là "policy không đứng nổi" — trong khi policy còn chưa được
hỏi gì.

Nên điểm spawn phải được **tìm và đo**:

```text
lấy lưới 120×120 trên sàn
clearance = khoảng cách trong mặt bằng tới tường và tới mọi hộp
            có dải Z cắt vào khoảng [0,05 m, 1,8 m] mà thân robot đi qua
chọn điểm clearance lớn nhất, và ghi lại con số đó
```

Khoảng cách tới hộp xoay tính trong hệ riêng của hộp, ở đó hình chữ nhật thẳng
trục nên phép đo tách rời theo từng chiều. Trên scene này: `[4.76, −2.03]` với
**1,007 m** thoáng, và số điểm tiếp xúc lúc spawn tụt từ 88 xuống **2** — đúng hai
bàn chân.

---

## 5. Nạp policy: chạy được và đứng được

```bash
make room-play                      # đứng yên, lệnh vận tốc 0
make room-play CMD="0.5 0 0"        # đi tới
make room-play POLICY=/đường/dẫn/policy_13999.onnx
```

Script: `scripts/sim/play_policy.py`. Nó dựng observation **từ `deploy_*.yaml`**
chứ không hardcode, vì sai layout thì không có gì báo lỗi: vector vẫn đúng độ
dài, mạng vẫn chạy, robot chỉ ngã một cách trông rất hợp lý.

### 5.1 Ba thứ YAML quy định mà rất dễ làm sai

**Thứ tự history.** Circular buffer của IsaacLab trả về **cũ trước, mới sau**. Mỗi
term được làm phẳng riêng rồi mới nối theo đúng thứ tự khai báo:

```text
base_ang_vel 3×5=15 | projected_gravity 15 | velocity_commands 15
joint_pos_rel 28×5=140 | joint_vel_rel 140 | last_action 140 | gait_phase 2×5=10
                                                            tổng = 475
```

475 khớp đúng chữ ký ONNX. Sai một term là lệch hết phần sau, mà vẫn đủ 475.

**Actuator là mô-men, không phải servo vị trí.** Kiểm tra được:
`gainprm = 1`, `biasprm = 0`. Nên vòng PD phải làm ở đây bằng `stiffness` và
`damping` trong YAML, chứ không giao cho MuJoCo:

```python
tau = kp * (action * scale + offset − q) − kd * qd
```

**Đứng yên là một pha gait riêng.** `mdp.phase` zero cả hai kênh khi
`‖command‖ < 0.1`. Lệnh bằng 0 phải cho `[0, 0]`, không phải sin/cos của một đồng
hồ đang chạy.

Ngoài ra `step_dt = 0.02` với timestep 0.002 → **10 substep** mỗi bước policy, và
giữ nguyên target suốt 10 substep đó.

### 5.2 Control trước, policy sau

Trước khi tin bất kỳ con số nào của policy, chạy **giữ PD ở home pose, không hỏi
policy**. Nó tách "sàn hỏng" khỏi "policy dở":

| | tâm phòng | spawn đã tìm |
|---|---|---|
| tiếp xúc lúc spawn | 88 | 2 |
| PD hold t=0 | `upright 0.950` | `upright 1.000` |
| PD hold sau 2 s | đã ngã | vẫn đứng, rồi mới từ từ đổ |

PD thuần đổ sau ~2 giây là **đúng** — không có bộ thăng bằng thì humanoid phải
đổ. Điều nó chứng minh là sàn và điểm spawn không có lỗi.

### 5.3 Kết quả — `policy_12000.onnx`, lệnh `0 0 0`, 10 giây

```text
pelvis        0.841 → 0.8317 m   (min 0.8317)
uprightness   0.9995             (min 0.9994)
trôi 1 s cuối 0.0001 m
                                  STANDING
```

Tiêu chí `standing` cần cả ba: còn gần độ cao ban đầu, còn thẳng đứng, và **đã
đứng yên**. Một robot đã ngã trượt cả ba.

---

## 6. Pipeline

```bash
make room-scene     # dựng phòng + collider + robot, kiểm, render
make room-play      # nạp policy, đứng yên 10 giây
make room-view      # mở trong MuJoCo viewer
```

Đổi tham số:

```bash
make room-scene ROOM_ROBOT=.../unitree_robots/vr_m3_1m6
make room-scene ROOM=path/to/other_room.glb ROOM_GT=path/to/gt.json
```

Chạy trực tiếp để lấy thêm cờ:

```bash
.venv/bin/python scripts/sim/make_room_scene.py \
  --room  outputs-full-factory/delivery/OUTPUT_B1/mesh/room_300k.glb \
  --gt    outputs-full-factory/delivery/OUTPUT_B1/metadata/ground_truth_boxes.json \
  --out   outputs-full-factory/delivery/OUTPUT_ROOM \
  --robot /path/to/unitree_robots/vr_h3_2 \
  --no-test                # bỏ hai phép kiểm vật lý
  --no-render              # bỏ ảnh kiểm tra
```


`--gt` là tuỳ chọn. Không có nó, độ cao sàn lấy theo phân vị 1% của đỉnh — chạy
được với video bất kỳ, chỉ kém chắc hơn.

### Output

```text
OUTPUT_ROOM/
├── room_scene.xml            # nạp thẳng vào MuJoCo
├── room_scene_report.json    # mọi số đo của hai phép kiểm
├── room_robot_check.png      # bốn khung để duyệt mắt
├── policy_play.png           # bốn thời điểm khi chạy policy
├── policy_play_report.json   # số đo của lần chạy policy
├── vr_h3_2_{rl,sensors,actuators}.xml
└── assets/
    ├── room_visual.obj       # đã xoay, sàn ở z=0
    └── *.stl                 # 31 mesh robot
```

Thư mục tự chứa: copy đi đâu cũng chạy, không phụ thuộc đường dẫn tới repo robot.

---

## 7. Giới hạn còn lại

- **Tường và trần là hộp bao chữ nhật quanh footprint**, không phải hình dạng
  tường thật. Robot không lọt ra ngoài, nhưng phòng không vuông thì có khe hở
  giữa collider và mesh.
- **Đồ đạc là hộp bao, không phải hình dạng thật.** 79 collider lấy từ OBB
  annotation nên robot không đi xuyên bàn máy, nhưng nó cũng không luồn được
  xuống gầm bàn — hộp là khối đặc. Cần hình dạng thật thì phải convex-decompose
  từng vật, mà mesh phòng lại không tách rời được (mục 4.1).
- **Chỉ có annotation mới cho được đồ đạc.** Không có `--gt` thì scene chỉ còn
  shell, robot đi xuyên mọi thứ trong phòng.
- **Chưa calibrate mét.** Stage [2] chưa chạy, `OUTPUT_B2/` chưa tồn tại. Sàn đo
  được 8,15 × 8,26 m và cửa 2,37–2,48 m nên `k ≈ 1`, nhưng đó là *ước lượng dựa
  trên vật trông quen*, không phải phép đo. Xem PHẦN B2 của
  `holispatial-hunyuan3d-adapter.md`.
- **Mới kiểm chứng ở lệnh đứng yên.** `CMD="0.5 0 0"` chạy được nhưng chưa ai đo:
  đi lại trong phòng chật với 79 vật cản là bài toán khác hẳn đứng một chỗ.

---

## 8. Bất biến — đừng phá

1. **Xoay 180° quanh X, không lật Z.** `det` phải bằng `+1`, nếu không thế giới
   bị phản chiếu.
2. **Mesh phòng luôn `contype="0"`.** Cho nó va chạm là biến căn phòng thành một
   khối đặc.
3. **Quả cầu control phải luôn có trong scene.** Nó là thứ phân biệt "collider
   hỏng" với "robot ngã".
4. **Keyframe phải viết sau khi biên dịch**, theo `jnt_qposadr` thật. Ghép tay là
   đặt robot vào nhầm body, im lặng.
5. **Không include `scene.xml` của robot** — nó mang một sàn thứ hai.
6. **Độ cao sàn lấy từ ground truth hoặc phân vị**, không lấy `min()`.
7. **Điểm spawn phải được tìm và đo, không mặc định tâm phòng.** Ghi lại
   `clearance` — nó là bằng chứng robot không sinh ra bên trong một cái máy.
8. **Chạy control PD trước khi tin policy.** Nó tách "sàn/spawn hỏng" khỏi
   "policy dở"; hai thứ đó trông giống hệt nhau khi nhìn robot ngã.
9. **Observation dựng từ `deploy_*.yaml`, không hardcode.** Sai layout vẫn đủ
   475 số và mạng vẫn chạy — không có gì báo lỗi.
