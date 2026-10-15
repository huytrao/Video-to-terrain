# Từ mesh phòng đến scene robot đứng được — có vật lý thật

Tài liệu kỹ thuật hướng dẫn chuyển đổi mesh không gian tái dựng (`room_300k.glb` hoặc bản đồ độ cao $512 \times 510$) thành môi trường mô phỏng vật lý [MuJoCo](https://mujoco.org/) hoàn chỉnh. Đảm bảo robot hình người (**Unitree Humanoid / Bipedal Robot**) đứng vững trong phòng với trọng lực đúng chiều, không nổ tiếp xúc (collision explosion), và cửa đi thông suốt.

Chạy toàn bộ quy trình bằng một lệnh:

```bash
make room-scene
```

Môi trường kiểm chuẩn chính: Nhà xưởng công nghiệp (`small_factory`, $6.01 \times 3.73 \times 6.35\,\text{m}$).  
Script điều phối cốt lõi: [`scripts/sim/make_unitree_scene.py`](../../scripts/sim/make_unitree_scene.py), [`scripts/sim/validate_unitree_scene.py`](../../scripts/sim/validate_unitree_scene.py), [`scripts/sim/play_policy.py`](../../scripts/sim/play_policy.py).

---

## 1. Ba lỗi cốt tử của mesh tái dựng và nguyên lý khắc phục

Khi nhập trực tiếp mesh thị giác từ photogrammetry hoặc TSDF vào physics engine, hệ thống mô phỏng hầu như luôn gặp lỗi nghiêm trọng mà không xuất hiện bất kỳ thông báo cảnh báo tường minh nào:

### 1.1 Trục Z chỉ XUỐNG và phép xoay bảo toàn tính thuận chirality ($\det(R) = +1$)

Đo đạc trực tiếp trên bounding box thực tế (`ground_truth_boxes.json`):

```text
floor    ins=15   z = [-0.204,  0.132]
ceiling  ins= 2   z = [-3.731, -2.922]
door     ins= 6   z = [-2.455,  0.027]   chiều cao 2.48 m
```

Sàn ở $z \approx 0$, trần ở $z \approx -3.33\,\text{m}$, và cửa chạy từ sàn **lên** tới $z = -2.46\,\text{m}$. Trục $Z$ của hệ toạ độ tái dựng hướng thẳng xuống đất.

Nếu giữ nguyên hệ trục này và khai báo `gravity="0 0 -9.81"` trong MuJoCo, **trọng lực sẽ hút robot bay về phía trần**. Không có ngoại lệ nào được ném ra; robot đơn giản là lơ lửng và bay mất.

#### Phép sửa sai lầm: Lật riêng trục $Z$ ($z \to -z$)
Phép biến đổi $\operatorname{diag}(1, 1, -1)$ có định thức $\det = -1$. Đây là **phép phản chiếu qua gương** (chiral reflection). Nó làm đảo nghịch toàn bộ quy tắc bàn tay phải, biến một robot thuận tay phải thành robot thuận tay trái, làm đảo dấu tích có hướng trong phương trình Coriolis và ma sát nón tiếp xúc.

#### Phép sửa chuẩn xác: Xoay $180^\circ$ quanh trục $X$
Áp dụng ma trận quay SO(3):

$$R_X(\pi) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad \det(R_X(\pi)) = +1$$

Chuyển đổi toạ độ: $(x, y, z) \to (x, -y, -z)$. Định thức $\det = +1$ bảo toàn tuyệt đối tính thuận của không gian.

Sau khi xoay, tịnh tiến toàn bộ scene để đưa mặt sàn về $z = 0$. Cao độ sàn phải lấy từ phân vị 10% của các đỉnh sàn phẳng ($\operatorname{percentile}_{10}(h)$) hoặc từ bounding box sàn được gán nhãn, **tuyệt đối không lấy giá trị nhỏ nhất $\min(z)$**. Giá trị $\min(z)$ trong mesh quét thực tế luôn chứa các điểm nhiễu trôi nổi sâu dưới lòng đất.

Kiểm tra chặn (assertion gate): Sau biến đổi, trần nhà bắt buộc phải nằm ở $z > 0.5\,\text{m}$. Nếu sai chiều, quy trình dừng ngay lập tức.

### 1.2 MuJoCo coi mọi mesh geom là BAO LỒI (Convex Hull)

MuJoCo tối ưu hoá va chạm bằng cách tự động tính **bao lồi** cho các phần tử `<geom type="mesh">`.

Bao lồi của toàn bộ một căn phòng là một **khối đặc khổng lồ bao kín cả không gian bên trong**. Nếu nạp thẳng `room_300k.glb` làm collider va chạm, robot không thể đứng bên trong phòng — solver sẽ coi robot đang nằm sâu trong lòng khối đặc và tác dụng lực đẩy cực lớn bắn robot văng ra ngoài.

Giải pháp bắt buộc là áp dụng quy tắc kép [Visual-vs-Collision](../../docs/architecture/architecture-graph.md#7-package-architecture-srcv3d--scriptssim):
- **Mesh phòng**: Chỉ dùng để render thị giác (`contype="0" conaffinity="0" group="2"`).
- **Collider vật lý**: Xây dựng từ các hình khối nguyên sinh (primitive) giải tích: 1 mặt phẳng sàn (`plane`) và các khối hộp chữ nhật (`box`) cho 4 bức tường và trần.

### 1.3 Màu đỉnh (Vertex Colors) và Flat Shading

Mesh photogrammetry (`room_300k.glb`) lưu trữ màu theo từng đỉnh ($163,808$ đỉnh). MuJoCo chỉ hỗ trợ texture map qua toạ độ UV và thẻ `<material>`. Do đó màu đỉnh bị bỏ qua trong im lặng và phòng sẽ render ra màu xám trơn.

Hai phương pháp tạo UV đã được thử nghiệm và loại bỏ vì lý do kỹ thuật:
1. **`xatlas` UV unwrap**: Mesh quét gồm **688 mảnh rời rạc**, không kín nước (`is_watertight = False`, đặc trưng Euler $\chi = -19,566$). Thuật toán sinh ra hàng nghìn mảnh chart nhỏ li ti và tắc nghẽn ở khâu đóng gói (chạy hơn 4 phút trên 80,000 tam giác vẫn không hoàn thành).
2. **Chiếu hộp 6 mặt (Box projection)**: Chạy nhanh (3.8 giây) nhưng gây nhoè nặng. Nhìn từ trên xuống ($+Z$), mặt sàn và mặt bàn máy móc chiếu đè lên cùng một texel, tạo ra hiện tượng tranh chấp vân bề mặt loang lổ.

Giải pháp tối ưu: Sử dụng **flat-shaded visual mesh** kết hợp ánh sáng định hướng, hoặc sử dụng bản đồ ảnh chiếu từ trên xuống (top-down orthophoto $nu \times nv$) dán trực tiếp lên địa hình heightfield.

---

## 2. 💥 Khắc phục hiện tượng "Collision Explosion" trong MuJoCo

### 2.1 Hiện tượng nổ tiếp xúc do biểu diễn kép (Double Representation)
Khi đưa mesh quét TSDF hoặc bản đồ độ cao vào MuJoCo cùng với các bounding box vật thể:
- Phép tái dựng TSDF là bề mặt liên tục (continuous 2-manifold surface), đã bao gồm sẵn hình khối của các máy móc và bàn thao tác: $97\%$ các ô địa hình cao trong phòng nằm trực tiếp dưới hộp bao OBB của các vật thể được phát hiện.
- Nếu đặt thêm 43 khối collider (hộp OBB hoặc convex hull) đè lên mặt sàn liên tục này, mỗi khối collider sẽ bị chìm sâu trong lòng đất: tạo ra **63 cặp va chạm xuyên nhau** với độ xuyên thấu ban đầu lên tới **$0.57\,\text{m}$**.
- Tại thời điểm $t=0$, solver tiếp xúc của MuJoCo (`implicitfast` / PGS) cố gắng giải phóng cùng lúc 63 điểm xuyên thấu này. Lực pháp tuyến vọt lên $> 10^5\,\text{N}$, generalized velocity $|q_{\text{vel}}|$ vọt lên **$48.8\,\text{rad/s}$**, dẫn đến hiện tượng nổ mô phỏng (`qvel became non-finite`) hoặc robot bị bắn văng khỏi scene.

```text
Biểu diễn lỗi (Nổ va chạm):
Terrain mesh:        ═══════════╦══════════════╦═══════════   (Địa hình đã chứa máy)
Collider vật thể:               │ ▓▓▓▓▓▓▓▓▓▓▓▓ │               (Đặt đè lên -> Xuyên thấu sâu 0.57m)
                                └──────────────┘
MuJoCo solver:       63 cặp xuyên thấu -> Lực > 10^5 N -> |qvel| = 48.8 -> NỔ VẬT LÝ!
```

### 2.2 Thuật toán: Phân tách mặt phẳng RANSAC và đục khoét đa giác Footprint

Quy trình giải quyết triệt để tại gốc rễ:

1. **Phân tách mặt phẳng sàn bằng RANSAC**:  
   Trích xuất mặt phẳng sàn $\mathbf{n}_{\text{floor}} \cdot \mathbf{x} + d = 0$ với vector pháp tuyến $\mathbf{n} \cdot \hat{\mathbf{z}} > 0.99$. Xác định cao độ sàn chuẩn $z_{\text{floor}} = \operatorname{percentile}_{10}(h_{\text{keep}})$.

2. **Chiếu footprint đa giác OBB 2D (`obb_footprint_mask`)**:  
   Với mỗi vật thể có tâm $\mathbf{c}$, kích thước $\mathbf{e}$ và ma trận hướng $A \in \mathrm{SO}(3)$, tính 8 đỉnh trong không gian 3D:
   $$\mathbf{p}_k = \mathbf{c} + \left( \mathbf{s}_k \odot \mathbf{e} \right) A, \quad \mathbf{s}_k \in \{-0.5, +0.5\}^3$$
   Chiếu 8 đỉnh này lên mặt phẳng lưới toạ độ $(u, v)$:
   $$u_k = \mathbf{p}_k \cdot \mathbf{a}_x - u_0, \quad v_k = \mathbf{p}_k \cdot \mathbf{a}_y - v_0$$
   Tính bao lồi 2D $\operatorname{ConvHull}(\{(u_k, v_k)\})$ và rasterize các nửa không gian (half-space inequalities $\mathbf{P} \cdot \mathbf{n}_h + d_h \le 0$) để đánh dấu các ô lưới bị che phủ.  
   *(Lưu ý: Nếu rasterize theo hộp bao thẳng trục AABB, diện tích chiếm chỗ bị thổi phồng tới $75\%$ đối với các máy đặt chéo, khiến thuật toán nhận nhầm $79\%$ diện tích sàn là vật cản thay vì con số thực tế là $45.5\%$).*

3. **Nội suy hạ sàn bằng biến đổi khoảng cách Euclid (`fill_from_floor`)**:  
   Vùng địa hình nằm dưới đáy vật thể được đục khoét (carve) khỏi heightfield và lấp phẳng về cao độ sàn.  
   Không dùng phép gán láng giềng gần nhất (nearest-neighbour) vì sẽ kéo mép cao của máy móc ra sàn xung quanh. Thuật toán dùng `scipy.ndimage.distance_transform_edt` chỉ lấy mẫu từ các ô sàn phẳng thực sự ($h < z_{\text{floor}} + 0.15\,\text{m}$).

4. **Cắt gọt collider tĩnh (`carve_hulls`)**:  
   Đối với bài toán thao tác (manipulation) cần vật thể chuyển động độc lập (`--objects-as-bodies`), các khối collider tĩnh của phòng bị cắt gọt bởi các mặt phẳng biên của OBB vật thể, loại bỏ hoàn toàn phần giao cắt.

#### Kết quả thực nghiệm:
- Số cặp va chạm xuyên thấu ban đầu: giảm từ 63 xuống **0** (hoặc đúng 2 tiếp xúc bàn chân khi spawn robot).
- Vận tốc lắng tối đa của hệ: giảm từ $|q_{\text{vel}}| = 48.8$ xuống **$8.87\,\text{rad/s}$**. Mô phỏng đạt trạng thái tĩnh ổn định tuyệt đối.

---

## 3. 🚪 Giới hạn 2.5D của Heightfield và giải pháp Wall-Slab tường minh

### 3.1 Bản chất toán học của hạn chế 2.5D
Bản đồ độ cao (heightfield elevation grid) là một hàm **đơn trị**:

$$z = f(x, y)$$

Mỗi toạ độ mặt bằng $(x, y)$ chỉ có thể mang đúng một giá trị độ cao $z$. Do đó, heightfield **về mặt tô-pô không thể biểu diễn được lỗ khoét ngang, đường hầm hoặc cửa đi có dầm/lanh-tô bắc ngang bên trên**.

Hậu quả khi nạp tường vào heightfield:
- Tường nhà trở thành một con đập đất dựng đứng kéo dài liên tục.
- Vị trí cửa đi bị bít kín hoàn toàn bởi khối đất đặc dâng cao ngang tường.
- Robot hoặc người giả lập bị nhốt trong phòng, không thể di chuyển qua các gian phòng khác.

### 3.2 Giải pháp: Bóc tách dải tường và tái tạo tấm Slab giải tích

Thuật toán giải phóng cửa đi trong `scripts/sim/make_unitree_scene.py`:

```
          ┌──────────────────────────────────────────────┐
          │     Lanh-tô Lintel Slab [h_door, h_room]      │
          ├──────────────┐                ┌──────────────┤
          │ Trụ Jamb 0   │    CỬA ĐI      │ Trụ Jamb 1   │
          │ [s_lo,       │   THÔNG SUỐT   │ [a + w/2,    │
          │  a - w/2]    │   (PASSABLE)   │  s_hi]       │
══════════╧══════════════╧════════════════╧══════════════╧══════════ (Sàn phẳng z=0)
```

1. **Gom cụm mặt phẳng tường (`cluster_walls`)**:  
   Mặt phẳng tường từ RANSAC ($n \cdot x + d = 0$) thường bị phân mảnh (ví dụ 7 mặt phẳng cho 4 bức tường). Thuật toán gom cụm các mặt phẳng có cùng hướng pháp tuyến và khoảng cách dọc trục trong ngưỡng sai số $\delta \le 0.35\,\text{m}$.

2. **Quét bề dày tường bằng profile trung vị (`wall_band_mask`)**:  
   Để không cắt lẹm vào máy móc đặt sát tường hoặc để sót gờ tường, thuật toán quét dọc theo mặt cắt độ cao trung vị:
   $$\operatorname{profile}(i) = \operatorname{median}_j (H(j, i))$$
   Mở rộng từ tâm tường ra hai phía cho đến khi độ cao tụt về mức sàn ($h \le 0.30\,\text{m}$). Nhờ đó đo được chính xác bề dày dải tường thực tế trong dữ liệu quét.

3. **Đục dải tường khỏi Heightfield**:  
   Toàn bộ dải tường và cửa đi trên heightfield được xoá bỏ và hạ về cao độ sàn $z_{\text{floor}}$ bằng EDT infill, tạo thành mặt sàn phẳng liền mạch.

4. **Định vị cửa đi theo toạ độ đo đạc**:  
   Toạ độ cửa từ module Lift3D được chiếu vuông góc vào bức tường gần nhất ($\Delta_{\perp} \le 0.05\,\text{m}$). Kích thước cửa được chuẩn hoá thành $1.20 \times 2.05\,\text{m}$ để đảm bảo robot hình người đi lọt.

5. **Sinh các khối Slab giải tích (`wall_with_optional_door`)**:  
   Mỗi bức tường có cửa được tạo thành từ 3 khối `<geom type="box">` giải tích:
   - **Trụ bên trái (Jamb 0)**: Kéo dài từ mép tường $s_{\text{lo}}$ đến mép cửa $a_{\text{door}} - w/2$, chiều cao trần $h_{\text{room}}$.
   - **Trụ bên phải (Jamb 1)**: Kéo dài từ mép cửa $a_{\text{door}} + w/2$ đến hết tường $s_{\text{hi}}$, chiều cao trần $h_{\text{room}}$.
   - **Lanh-tô trên (Lintel)**: Đặt ngay trên cửa, chiều rộng $w$, giới hạn độ cao $z \in [h_{\text{door}}, h_{\text{room}}]$.

6. **Kiểm tra độ thông suốt bằng Ray-Casting (`validate_unitree_scene.py`)**:  
   Hệ thống kiểm tra tự động bắn tia quét ngang xuyên qua cửa ở 3 cao độ ($z \in \{0.25, 0.90, 1.60\}\,\text{m}$) và 3 độ lệch ngang ($\Delta \in \{-0.35, 0.0, +0.35\}\,\text{m}$) trên khoảng cách $1.8\,\text{m}$. Nếu bất kỳ tia nào chạm vào vật cản, scene sẽ bị đánh rớt (FAIL).

---

## 4. Thiết lập động học Robot: Bài toán Freejoint và tìm điểm Spawn tối ưu

### 4.1 Cái bẫy địa chỉ Keyframe khi có nhiều Freejoint
Robot Unitree / Bipedal humanoid đi kèm keyframe tư thế chuẩn (`home`):
`qpos="0 0 0.841 1 0 0 0 ..."` (pelvis cao $0.841\,\text{m}$ và 28 góc khớp, tổng 35 giá trị).

Nếu ghép chuỗi 35 số này vào scene có sẵn quả cầu probe hoặc anchor, scene sẽ có **hai freejoint** ($nq = 42$). Quả cầu khai báo trước nên chiếm `qpos[0:7]`, toàn bộ home pose của robot bị gán nhầm cho quả cầu, còn pelvis robot nhận nhầm góc khớp làm toạ độ. Hậu quả: `pelvis z = 0.0`, robot bị ép lún nửa người vào sàn mà không hề có thông báo lỗi cú pháp.

**Giải pháp**: Biên dịch mô hình trong bộ nhớ, tra cứu địa chỉ động từ `MjModel`:
```python
free_joints = [j for j in range(model.njnt)
               if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
               and model.jnt_bodyid[j] == pelvis_id]
root_qpos = model.jnt_qposadr[free_joints[0]]
```

### 4.2 Tìm điểm Spawn thoáng bằng Distance Transform
Nếu mặc định spawn robot ở tâm hình học của phòng, trên scene nhà xưởng này robot sẽ rơi trúng **bên trong cỗ máy `obs_55_machine`**: 88 điểm va chạm sâu tới $0.57\,\text{m}$. Solver sẽ lập tức đẩy văng robot làm nó ngã gục.

Thuật toán tìm điểm spawn tối ưu:
1. Tạo ma trận nhị phân các vùng cấm: Ô có độ cao $h > 0.25\,\text{m}$ (máy móc) hoặc nằm ngoài chu vi phòng.
2. Áp dụng phép biến đổi khoảng cách Euclidean (`scipy.ndimage.distance_transform_edt`): Mỗi ô sàn nhận giá trị là khoảng cách ngắn nhất tới vật cản gần nhất.
3. Lọc các ô có độ biến thiên cao độ phẳng ($\max(h) - \min(h) < 0.06\,\text{m}$ trong cửa sổ $25 \times 25$).
4. Chọn toạ độ có khoảng cách thoáng lớn nhất: Tìm được toạ độ $[4.76, -2.03]$ với bán kính an toàn **$1.007\,\text{m}$**.

Tại điểm spawn này, số điểm tiếp xúc lúc $t=0$ giảm từ 88 xuống đúng **2 điểm** (hai mặt đế bàn chân).

---

## 5. 🤖 Real-to-Sim Validation với Robot hình người và Playback Policy RSL-RL

Hỗ trợ mô hình robot hình người kích thước thật:
- **Unitree H1 / Humanoid**: Robot hình người hai chân.
- **Unitree Biped / Quadruped**: Các mô hình robot tiêu chuẩn.

Quy trình kiểm chứng độ tin cậy mô phỏng qua hai bước độc lập:

```bash
make room-play                      # Đứng yên, lệnh vận tốc [0, 0, 0]
make room-play CMD="0.5 0 0"        # Tiến thẳng với vận tốc 0.5 m/s
make room-play POLICY=/path/to/policy_12000.onnx
```

### 5.1 Kiến trúc tầng chạy Policy (`scripts/sim/play_policy.py`)

Bộ điều khiển tải trực tiếp mạng nơ-ron xuất từ **RSL-RL / IsaacLab** (`policy_*.onnx`) và cấu hình triển khai (`deploy_*.yaml`):

1. **Bộ đệm lịch sử (Circular History Buffer)**:  
   IsaacLab trả về dữ liệu quan sát theo thứ tự **cũ trước, mới sau** (oldest-first, newest-last). Mỗi đại lượng được làm phẳng và ghép nối theo đúng thứ tự khai báo:
   ```text
   base_ang_vel (3x5=15) | projected_gravity (15) | velocity_commands (15)
   joint_pos_rel (28x5=140) | joint_vel_rel (140) | last_action (140) | gait_phase (2x5=10)
                                                             Tổng cộng = 475 chiều
   ```
2. **Gia tốc góc và trọng lực chiếu trong hệ quy chiếu thân (Pelvis Body Frame)**:  
   Gia tốc góc: $\boldsymbol{\omega}_{\text{body}} = R_{\text{pelvis}}^T \boldsymbol{\omega}_{\text{world}}$.  
   Trọng lực chiếu: $\mathbf{g}_{\text{proj}} = R_{\text{pelvis}}^T [0, 0, -1]^T$.
3. **Pha bước đi (Gait Phase)**:  
   Khi lệnh vận tốc $\| \mathbf{v}_{\text{cmd}} \| < 0.1\,\text{m/s}$, pha bước đi bị triệt tiêu về $[0, 0]$ (trạng thái đứng yên), không dùng hàm $\sin/\cos$ của đồng hồ chạy tự do.
4. **Vòng lặp mô-men xoắn PD cấp thấp**:  
   Các actuator trong file mô hình là động cơ mô-men thuần (`gainprm=1, biasprm=0`). Vòng lặp PD được tính toán tường minh:
   $$\boldsymbol{\tau} = \mathbf{K}_p (\mathbf{a}_t \odot \mathbf{s}_{\text{scale}} + \mathbf{o}_{\text{offset}} - \mathbf{q}) - \mathbf{K}_d \dot{\mathbf{q}}$$
   Mỗi bước suy luận policy ($50\,\text{Hz}, \Delta t = 0.02\,\text{s}$) tương ứng với **10 substep** tích phân vật lý MuJoCo ($500\,\text{Hz}, \Delta t = 0.002\,\text{s}$).

### 5.2 Kết quả thực nghiệm kiểm chuẩn trên Robot hình người (10 giây)

```text
pelvis        0.8410 -> 0.8317 m   (độ lún ổn định 9.3 mm)
uprightness   0.9995               (độ thẳng đứng duy trì > 0.9994)
trôi 1 s cuối 0.0001 m             (vận tốc trôi < 0.1 mm/s)
trạng thái    STANDING (PASS)
```

Tiêu chí đạt chuẩn yêu cầu đồng thời cả 3 điều kiện:
- Chiều cao pelvis ổn định quanh mức danh định.
- Độ nghiêng trục thân $\mathbf{z}_{\text{pelvis}} \cdot \hat{\mathbf{z}}_{\text{world}} > 0.995$.
- Độ trôi vị trí ở giây cuối cùng $< 5\,\text{mm}$.

---

## 6. Lệnh thực thi và Cấu trúc thư mục Delivery

### 6.1 Lệnh CLI chuẩn

```bash
# 1. Dựng scene đầy đủ: phòng, collider, cửa đi, robot humanoid
make room-scene

# 2. Tuỳ biến tham số môi trường và đường dẫn robot
.venv/bin/python scripts/sim/make_unitree_scene.py \
  --assets            outputs-smallroom/terrain \
  --unitree           ${UNITREE_MUJOCO_DIR:-/path/to/unitree_mujoco_si} \
  --robot             biped_robot \
  --door-width        1.20 \
  --door-height       2.05 \
  --metres-per-unit   1.0

# 3. Chạy kiểm tra va chạm và độ ổn định vật lý
.venv/bin/python scripts/sim/validate_unitree_scene.py

# 4. Chạy kiểm thử điều khiển kín với policy RSL-RL
.venv/bin/python scripts/sim/play_policy.py \
  --scene             outputs-smallroom/OUTPUT_ROOM/room_scene.xml \
  --policy            configs/policies/policy_12000.onnx \
  --deploy            configs/policies/deploy_12000.yaml
```

### 6.2 Cấu trúc thư mục Output tự chứa (`OUTPUT_ROOM/`)

```text
OUTPUT_ROOM/
├── room_scene.xml            # File scene chính nạp trực tiếp vào MuJoCo
├── room_scene_report.json    # Báo cáo định lượng các phép đo kiểm va chạm
├── room_robot_check.png      # Ảnh chụp trực giao 4 góc kiểm tra tiếp xúc
├── policy_play.png           # Biểu đồ trạng thái khi policy vận hành
├── policy_play_report.json   # Chỉ số ổn định, độ trôi và lực tiếp xúc
├── robot_{rl,sensors,actuators}.xml
└── assets/
    ├── factory_hfield.png    # Bản đồ độ cao 16-bit 512x510 đã đục dải tường và máy móc
    ├── factory_texture.png   # Ảnh vân bề mặt chiếu từ trên xuống
    └── *.stl                 # 31 file mesh STL của robot
```

Thư mục hoàn toàn độc lập, có thể sao chép sang máy tính khác mà không bị lỗi đường dẫn tuyệt đối.

---

## 7. Giới hạn kỹ thuật hiện tại

1. **Tường bao hình học giải tích**: Tường ngoại vi được gán là các hộp bao chữ nhật bao quanh footprint phòng. Với phòng có hình dáng chữ L hoặc vát góc, sẽ xuất hiện khe hở giữa collider hộp và mesh thị giác.
2. **Khối va chạm đồ đạc là OBB/Convex Hull**: 43 vật thể được biểu diễn dưới dạng hộp có hướng hoặc bao lồi. Robot không thể luồn tay hoặc chui gầm bàn thao tác.
3. **Phụ thuộc vào nhãn phát hiện vật thể**: Nếu không có bounding box từ module Stage F, scene chỉ chứa địa hình nền phẳng và robot sẽ đi xuyên qua máy móc.
4. **Độ tin cậy thang đo mét**: DA3 ước lượng chiều cao phòng là $2.307\,\text{m}$. Đây là thang đo ước lượng đơn góc nhìn. Khi triển khai tại nhà xưởng thực tế, cần một phép đo thước dây để truyền tham số `--metres-per-unit` chuẩn hoá toàn bộ lực và khối lượng.

---

## 8. Bất biến kỹ thuật bất khả xâm phạm (Engineering Invariants)

1. **Xoay $180^\circ$ quanh X, không bao giờ lật riêng trục Z**: Định thức $\det(R)$ bắt buộc phải bằng $+1$.
2. **Mesh thị giác của phòng luôn gán `contype="0" conaffinity="0"`**: Tránh biến toàn bộ căn phòng thành một khối bao lồi đặc.
3. **Luôn giữ quả cầu probe kiểm chuẩn (control sphere)**: Phân biệt rõ ràng lỗi do địa hình hỏng hay lỗi do policy điều khiển.
4. **Tra cứu địa chỉ Keyframe qua `jnt_qposadr`**: Tuyệt đối không gán cứng mảng `qpos[0:7]`.
5. **Không nạp đè `scene.xml` gốc của robot**: File gốc chứa sẵn một mặt sàn plane riêng, sẽ gây xung đột với sàn của nhà xưởng.
6. **Cao độ sàn lấy theo phân vị 10%**: Không lấy giá trị nhỏ nhất $\min(z)$ của các đỉnh mesh.
7. **Luôn tìm điểm spawn thoáng bằng phép biến đổi khoảng cách**: Không mặc định spawn ở tâm phòng.
8. **Chạy kiểm thử giữ vị trí PD trước khi chạy policy RL**: Tách biệt lỗi tiếp xúc cơ học với lỗi điều khiển thăng bằng.
9. **Cấu trúc quan sát (Observation) phải xây dựng từ `deploy_*.yaml`**: Đảm bảo đúng thứ tự oldest-first và đủ 475 chiều.
