# OBB-anchored segmentation — cắt vật thể chuẩn cho Hunyuan3D

Cách thay mask text-prompt của Holi bằng mask neo vào OBB mét, và cách chạy lại
nó từ đầu. Chạy thật trên scene nhà xưởng `00777c41d4`, 38 object, 250 view.

Tài liệu đi kèm: `holispatial-hunyuan3d-adapter.md` PHẦN C3 + E là **đặc tả**;
tài liệu này là **runbook và hồ sơ vì sao**. `bugs-found.md` mục B14–B19 ghi
từng lỗi riêng.

---

## 1. Vấn đề

Ảnh `hunyuan_input.png` sai ở cả hai đầu cùng lúc — vừa thừa vừa thiếu.

Đo trên `obj_0219_table`, view `DSC01071`:

| | đo được |
|---|---|
| `best_mask.png` của C1 | 212.467 px — **chỉ mặt bàn**, không chân, không hộc kéo, không ê-tô |
| mask chạm mép dưới ảnh | y = 1158 / 1168, dù C1 tuyên bố "không chạm viền" |
| sau khi hợp với rembg | crop nở lên tới y = 521, **kéo thêm 3 hộp đen** đứng trên bàn và một mảng sàn |

Hai nguyên nhân nằm ở hai chỗ khác nhau:

**`source-reference/holi-spatial/sam3.py:151`** — Holi gọi SAM 3 bằng đúng một
dòng, và đó là text prompt thuần:

```python
output = processor.set_text_prompt(state=state, prompt=label)
```

Với `"table"`, concept head bám vào vùng phẳng nổi bật nhất. Không có một ràng
buộc hình học nào bảo nó rằng cái bàn còn có chân.

**`02_quality_gate_and_adapter_a.py:160`** (bản cũ) — bản vá cho việc đó là hợp
mask với rembg:

```python
union_mask = cv2.bitwise_or(sam_mask_full, rembg_mask_full)
```

Union chỉ biết **cộng**. Nó không lấy lại được chân bàn (rembg cũng coi vùng tối
dưới gầm là nền), nhưng lại thêm vào mọi thứ rembg cho là "foreground nổi bật" —
tức cả ba cái hộp đen.

> Có một phát hiện phụ đáng ghi: chạy SAM 3 text prompt trực tiếp cho **312.416
> px**, trong khi `best_mask.png` lưu trong C1 chỉ có 212.467 px và thủng lỗ chỗ.
> Mask trong C1 **không phải** output thô của SAM 3 — nó đã đi qua bước lifting
> 3D của Holi và bị bước đó làm hỏng thêm.

---

## 2. Thứ đang bị bỏ phí

Mỗi object đã có sẵn, trong chính `OUTPUT_C1`:

```text
object.json     obb.center_world, obb.extent_xyz (mét), obb.rotation_world_3x3
cameras/*.json  position, rotation (camera→world), fx, fy, width, height
```

Đủ để chiếu OBB xuống từng ảnh và có một **bao hình học chính xác**. Không dòng
code nào trước đây dùng đến nó.

Quy ước đã kiểm chứng bằng số, không phải suy đoán:

```text
R_cam   là camera→world   →  X_cam = R_camᵀ (X_world − C)
R_obb   là local→world    →  local = R_obbᵀ (X_world − center),  8 góc tại ±extent/2
```

Kiểm chứng trên `obj_0219_table`: `extent/2 = [1.0521, 0.4113, 0.4101]` và
`max|local| = [1.0521, 0.4113, 0.4101]` — khớp tuyệt đối.

Chọn phép chiếu sai chiều thì thấy ngay: dùng `R_cam` thay vì `R_camᵀ` cho ra
`z` âm và `v = −11444`.

---

## 3. Cách làm

### 3.1 SAM 3 nhận prompt hình học, không phải chữ

Bản SAM 3 vendor sẵn có `Sam3Processor.add_geometric_prompt(box, label, state)`
nhận **box dương và box âm** dạng `[cx, cy, w, h]` chuẩn hoá — đường mà `sam3.py`
chưa từng gọi.

```text
positive  = bbox(convexHull(chiếu 8 góc OBB))
negative  = bbox tương tự của mọi instance j thoả:
              OBB_j ⊄ OBB_i          không phải bộ phận
              OBB_j ∩ OBB_i ≠ ∅      thật sự đè lên
```

Bốn tổ hợp thi đấu cho mỗi view — `obb_box`, `obb_box_neg`, `text_obb_box`,
`text_only` — **cộng thêm chính mask cũ của Holi** làm ứng viên. Mask cũ dự thi
bằng đúng thước đo, nên bước này không bao giờ làm một object tệ đi.

Đo trên `obj_0219_table / DSC01071`:

| prompt | score | pixels |
|---|---|---|
| `text_only` (Holi hiện tại) | 0.609 | 212.467 |
| `obb_box` | **0.965** | **318.631** |

### 3.2 Chấm điểm bằng tiêu chí 3D

```text
fill = |mask ∩ hull| / |hull|
leak = |mask \ hull| / |mask|
```

`leak` là **gate cứng** (0.05). OBB là biên vật lý; pixel ngoài nó sai theo định
nghĩa. Đây là thứ union `A ∪ B` không bao giờ có được.

`fill` **chỉ xếp hạng**, không làm gate — kệ, thang, ống lấp rất ít thể tích bao
của chính chúng một cách hoàn toàn hợp lệ.

### 3.3 Một view chỉ dùng được khi thoả cả bốn

```text
toàn bộ 8 góc OBB nằm trong ảnh, lề ≥ 16 px
mask không chạm viền ảnh trong 16 px
không phải fill-outlier  (fill < median − 2·MAD so với các view khác cùng object)
leak ≤ 0.05
```

### 3.4 Tầng local

Trừ vật đè (bắt buộc kiểm tra giao OBB **trong 3D**) → inpaint lỗ → bỏ đốm rời
(< 10% thành phần lớn nhất) → chọn best view theo `fill × area` → gán slot theo
azimuth trong **hệ riêng của OBB** → xuất RGBA 518×518 với `alpha = mask`.

Chi tiết từng bước ở PHẦN E của `holispatial-hunyuan3d-adapter.md`.

---

## 4. Sáu cái bẫy, với số đo

Ba cái đầu là lỗi có sẵn. Ba cái sau do chính code mới sinh ra và bị bắt bằng số
đo, không phải bằng mắt — ghi lại vì chúng sẽ quay lại nếu ai đó viết lại tầng này.

| # | Bẫy | Triệu chứng đo được |
|---|---|---|
| 1 | Text prompt thuần | mask bàn chỉ có mặt trên |
| 2 | `A ∪ B` với rembg | hộp đen + sàn bị gộp vào |
| 3 | Nền trắng đặc thay vì alpha | `.convert("RGBA")` cho alpha = 255 khắp nơi; model hiểu cả ô 518×518 là vật thể |
| 4 | Gate `leak` **fail open** | không ứng viên nào đạt ngưỡng → code xếp theo `fill`, mà mask nuốt cả ảnh có `fill = 1.0`; `obj_0211_shelf` nhận mask `leak = 0.984`, phình 134 → 11.867 px |
| 5 | `leak` vô nghĩa khi OBB phủ khung hình | `obj_0053 / DSC01081`: hull = 99.9% ảnh, mask = 99.7% ảnh, `leak = 0.001` |
| 6 | Trừ hàng xóm thiếu kiểm tra giao OBB 3D | chồng lấn *trên ảnh* không phải chồng lấn *trong không gian*; mask bàn tụt 210.914 → 7.261 px, bốn view khác về 0 |

Hai cái nữa không phải bug nhưng cùng loại — đại lượng đo sai thứ mình tưởng:

**Phương sai Laplacian thiên vị vật ở xa.** Nó tỉ lệ với chi tiết-trên-mỗi-pixel,
nên đo mật độ chi tiết chứ không đo độ nét. Trên `obj_0219_table`: view xa 19.879
px cho **455**, view cận cảnh sạch nhất 210.914 px cho **35** — và view xa thắng.
Chuẩn hoá bằng cách resize vùng vật thể về 256×256 trước khi đo: cả 7 view về
khoảng 164–337, thứ hạng đảo lại cho đúng.

**Azimuth hệ world không phải azimuth của vật thể.** `Hunyuan3D-2/hy3dgen/shapegen/models/conditioner.py:174-180`
cho thấy 2mv có `view_embed` **học được, đánh index theo slot**, nên nhãn slot
mang ý nghĩa thật với model. Đo bằng `atan2(cam.y − obj.y, cam.x − obj.x)` thì
"mặt trước" của mỗi object phụ thuộc vào việc nó tình cờ xoay thế nào trong
phòng. Phải đưa về hệ OBB bằng `R_obbᵀ`.

---

## 5. Runbook

### 5.1 Kernel

Chỉ cần **RUN** (RTX PRO 6000, offline). Lấy proxy URL mới từ panel
⋮ → "Kaggle Jupyter Server" của notebook, ghi vào `.env` dưới
`KAGGLE_PROXY_RTX6000_NO_INTERNET`.

Kiểm tra sống:

```bash
for v in KAGGLE_PROXY_RTX6000_NO_INTERNET KAGGLE_PROXY_HAS_INTERNET_NO_GPU; do
  u=$(grep -E "^$v *=" .env | cut -d= -f2- | tr -d '"'"'"' ')
  echo "$v -> HTTP $(curl -s -o /dev/null -w '%{http_code}' -m 15 "$u/api/kernels")"
done
```

Dataset phải attach sẵn:

```text
/kaggle/input/datasets/traoanhuy/v3d-sam31-multiplex/
├── ckpt/sam3.1_multiplex.pt     3.5 GB
└── src/sam3/                    có cả sam3/train/
/kaggle/input/datasets/traoanhuy/v3d-build-cache-sm120/   wheel cache
```

### 5.2 Ba bẫy môi trường

**Thiếu `iopath`, `ftfy`.** Không có trong wheel cache, và RUN không có mạng.
Tải ở máy local rồi đẩy lên:

```bash
/usr/bin/python3 -m pip download iopath ftfy portalocker wcwidth --no-deps -d /tmp/wheels
for f in /tmp/wheels/*; do
  .venv/bin/python kaggle-runtime-toolkit-agent-skills/kaggle_executor.py \
    upload "$f" "wheels/$(basename $f)"
done
```

rồi trên kernel: `pip install --no-index --no-deps /kaggle/working/wheels/*`.

**Kaggle tự giải nén `.gz` khi tạo dataset.** `bpe_simple_vocab_16e6.txt.gz` đến
nơi thành `.txt`, mà tokenizer gọi `gzip.open()`. Nén lại rồi truyền qua
`bpe_path`:

```python
import gzip, shutil, os
src = '/kaggle/input/datasets/traoanhuy/v3d-sam31-multiplex/src/sam3/assets/bpe_simple_vocab_16e6.txt'
dst = '/kaggle/working/assets/bpe_simple_vocab_16e6.txt.gz'
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(src, 'rb') as fi, gzip.open(dst, 'wb') as fo:
    shutil.copyfileobj(fi, fo)
```

**Phải bật bf16 autocast.** Không bật thì Linear đầu tiên báo
`mat1 and mat2 must have the same dtype, but got BFloat16 and Float`:

```python
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
```

`missing_keys = backbone.vision_backbone.convs.3.*` khi nạp checkpoint là **bình
thường** — `convs.3` chỉ thuộc nhánh `enable_inst_interactivity`, mà đường
geometric prompt không dùng đến.

### 5.3 Đẩy dữ liệu lên

`OUTPUT_C1` nguyên bản là 682 MB. Contents API gửi mỗi request một file nên 576
file sẽ rất chậm. Nén `views/` sang JPEG q95 (chúng vốn là PNG lưu lại từ ảnh
JPG gốc, nên không mất gì thực chất), giữ `masks/` ở PNG, rồi tar thành **một**
gói: 682 MB → **119 MB**.

Script đọc view chấp nhận cả `.png` và `.jpg`, nên không cần đổi gì khác.

### 5.4 Chạy

```bash
# 1. Segment lại — trên kernel GPU
.venv/bin/python kaggle-runtime-toolkit-agent-skills/kaggle_executor.py \
  upload scripts/adapter/03_resegment_obb_sam3.py "code/03_resegment_obb_sam3.py"

.venv/bin/python kaggle-runtime-toolkit-agent-skills/kaggle_executor.py exec "
import sys, runpy, warnings; warnings.filterwarnings('ignore')
sys.argv = ['x', '--c1-dir', '/kaggle/working/C1',
                 '--output-dir', '/kaggle/working/out_c3_all']
_ = runpy.run_path('/kaggle/working/code/03_resegment_obb_sam3.py', run_name='__main__')
" --timeout 5400
```

> Gán `_ =` là bắt buộc. `runpy.run_path` trả về dict globals, và IPython sẽ in
> trọn cái dict đó ra, đẩy toàn bộ log thật ra khỏi màn hình.

```bash
# 2. Kéo về và chạy Adapter A — local, CPU
tar xzf out_c3_all.tgz -C outputs-full-factory/delivery/

.venv/bin/python scripts/adapter/02_quality_gate_and_adapter_a.py \
  --c1-dir        outputs-full-factory/delivery/OUTPUT_C1 \
  --output-dir    outputs-full-factory/delivery/OUTPUT_D_v2 \
  --resegment-dir outputs-full-factory/delivery/OUTPUT_C3

# 3. Trang duyệt trước/sau
.venv/bin/python scripts/adapter/tools/make_contact_sheet.py
```

Thời gian đo được: segment 38 object / 250 view mất vài phút trên RTX 6000;
nạp checkpoint 4.1 s.

---

## 6. Kết quả trên scene `00777c41d4`

38 object, 228 view segment được, **140 view qua gate**.

| | |
|---|---|
| mask lớn hơn | **29** |
| mask nhỏ đi | **0** — cơ chế fallback bảo đảm điều này |
| giữ mask cũ (`variant = holi_original`) | 8, tất cả đều là mảnh vụn < 4000 px |
| bị từ chối | 1 (`obj_0210_shelf`) |
| đủ ≥2 slot cho 2mv | 7 |

Vài trường hợp tiêu biểu:

| object | cũ | mới | tỉ lệ |
|---|---|---|---|
| `obj_0053_machine` | vụn nát, không nhận ra | máy bào liền khối | 12,13× |
| `obj_0192_cabinet` | chỉ mặt phẳng trước | tủ có chiều sâu | 8,33× |
| `obj_0213_ladder` | thang thủng lỗ | thang đủ bậc | 4,33× |
| `obj_0180_workbench` | **gộp nhầm cả thùng rác** | chỉ cái máy | 2,53× |
| `obj_0219_table` | mặt bàn + 3 hộp đen + sàn | bàn đủ chân, hộc kéo, ê-tô | 1,63× |

---

## 7. Giới hạn còn lại

Không sửa được ở tầng này, cần sửa ở tầng trên hoặc ở khâu quay:

- **Chỉ 7/37 object đủ ≥2 slot cho 2mv.** Camera chỉ quét một cung ~40° quanh mỗi
  vật — cả 10 view của `obj_0219_table` nằm trong azimuth 324°–5°. Đây là giới
  hạn của dữ liệu quay, không phải của logic gán slot. Muốn dùng được 2mv thật
  thì phải quay vòng quanh vật thể.
- **`obj_0052` / `obj_0055` còn dính giá đỡ.** Chúng nằm trong OBB của máy nhưng
  Holi không đăng ký thành instance riêng, nên luật trừ hàng xóm không có gì để
  trừ.
- **`obj_0179_workbench` và `obj_0219_table` là cùng một cái bàn**, bị Holi tách
  thành hai instance.
- **7/38 object có OBB dưới 1000 px**, một cái chỉ dài 4 cm. Chúng không nên vào
  Hunyuan3D ngay từ đầu; quality gate hiện lọc theo tên lớp chứ không theo kích
  thước thật.

---

## 8. Bất biến — đừng phá

1. **`leak` là gate, `fill` chỉ để xếp hạng.** Đảo lại là vật rỗng chết hết.
2. **Nhánh vượt ngưỡng leak phải xếp theo `−leak`, không theo `fill`.** Xếp theo
   `fill` là gate fail open (bẫy #4).
3. **`hull / frame ≤ 0.60` là điều kiện tiên quyết để `leak` có nghĩa** (bẫy #5).
4. **Trừ hàng xóm phải kiểm tra giao OBB trong 3D**, cộng chốt chặn 30% (bẫy #6).
5. **Mask cũ của Holi luôn được dự thi.** Đó là thứ bảo đảm "không bao giờ tệ
   hơn hiện tại".
6. **Xuất RGBA với `alpha = mask`.** Nền trắng đặc phá conditioning của Hunyuan3D.
7. **Không dùng lại rembg ở tầng này.** Nó là nguồn gốc của cắt thừa.
