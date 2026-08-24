# Hướng dẫn xem sản phẩm — Video → 3D

Toàn bộ kết quả nằm trong thư mục `report/`. Mọi trang đều là HTML tĩnh,
**không cần cài đặt gì** — chỉ cần mở đúng cách bên dưới.

## Cách mở (quan trọng)

Các trang 3D cần tải file `.glb` cùng thư mục, nên **không mở file
bằng double-click** (`file://` sẽ bị trình duyệt chặn). Thay vào đó:

```bash
cd report
python3 -m http.server 8790
```

rồi mở trình duyệt tại `http://127.0.0.1:8790/`. Tắt server bằng `Ctrl+C`.

## Trang nào làm gì

| URL | Nội dung | Dùng khi |
| :--- | :--- | :--- |
| `/index.html` | Báo cáo tổng thể: quy trình, số liệu kiểm chứng, bảng so sánh với LiDAR/dựng tay | Đọc đầu tiên — 10 phút là hiểu toàn bộ pipeline |
| `/viewer/index.html` | Scene 3D tương tác: căn phòng + 8 vật rời có vật lý thật, kéo–thả được | Demo trực tiếp cho người xem; kéo vật thả lên bàn là ấn tượng nhất |
| `/objects3d/index.html` | 3 vật hoàn thiện nhất (bàn nguội, lưỡi dao, thùng rác) dựng lại bằng Hunyuan3D, quay vòng 360° | Cho thấy bước "hoàn thiện vật thể" hoạt động ra sao |
| `/KICH_BAN_QUAY.md` | Kịch bản quay video demo 4 phút, chia theo từng mốc thời gian | Khi cần thu video giới thiệu thay vì demo trực tiếp |

Thứ tự gợi ý khi demo: `index.html` (lướt nhanh) → `viewer/` (tương tác)
→ trong viewer bấm nút **"3 vật hoàn thiện"** để sang `objects3d/`.

## Trong viewer, thao tác được gì

- **Kéo chuột trái / vuốt**: xoay quanh cảnh.
- **Con lăn**: zoom.
- **Click tên vật ở bảng bên phải**: camera bay tới sát vật đó.
- **Kéo một vật rồi thả**: vật rơi và *nằm đúng bề mặt* dưới nó (sàn,
  mặt bàn…) nhờ raycast va chạm — không phải vị trí đặt sẵn.
- **"Đưa tất cả về chỗ cũ"**: reset toàn bộ về trạng thái ban đầu.

## Nếu muốn chạy lại pipeline từ đầu

Phần code đầy đủ (kèm hướng dẫn kỹ thuật tiếng Anh) nằm ở cấp trên của
zip này: đọc `README.md` → `make setup && make test`. Toàn bộ quy trình
chạy lại từ một video đầu vào mới theo `docs/guides/pipeline-runbook.md`.
