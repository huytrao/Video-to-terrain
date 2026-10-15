# Hướng dẫn Khảo sát & Đánh giá Hệ thống — Monocular Video to 3D Digital Twin Pipeline

**Dự án:** Tái tạo Digital Twin và Môi trường Mô phỏng Động học Robot từ Video Điện thoại Đơn sắc (Monocular Video to 3D Digital Twin Pipeline for Sim-to-Real Robotics)  
**Kỹ sư nghiên cứu thực hiện:** **Trảo An Huy (Trao An Huy)** — VinUniversity  
**Cố vấn hướng dẫn:** **Nguyễn Doãn Hoàng**  
**Lĩnh vực:** Computer Vision · 3D Scene Reconstruction · Reinforcement Learning · Sim-to-Real Robotics  

---

## 1. Tổng quan & Phương thức Khởi chạy

Toàn bộ hệ thống báo cáo và các ứng dụng trình diễn 3D tương tác được đóng gói dưới dạng các giao diện WebGL/Three.js tĩnh độc lập trong thư mục `output/REPORT/` (hoặc thư mục bundle `report/`).

> [!IMPORTANT]
> **Phương thức khởi chạy bắt buộc qua HTTP Server:**  
> Do các thành phần WebGL/Three.js tải các mô hình 3D nhị phân (`.glb`) và tệp siêu dữ liệu JSON cục bộ thông qua Fetch API, trình duyệt web sẽ kích hoạt cơ chế bảo mật CORS nếu mở trực tiếp bằng giao thức `file://`. Quý Thầy/Cô và các nhà nghiên cứu vui lòng khởi chạy qua HTTP server nội bộ đơn giản bằng lệnh sau:

### Lệnh khởi chạy:

Nếu đang ở thư mục gốc của repository:
```bash
python3 -m http.server 8790 --directory output/REPORT
```

Hoặc nếu đang mở trực tiếp thư mục `output/REPORT` (hoặc giải nén từ zip):
```bash
cd output/REPORT
python3 -m http.server 8790
```

Sau đó, truy cập bằng trình duyệt web tại địa chỉ: **`http://127.0.0.1:8790/`**  
*(Nhấn `Ctrl + C` trên terminal để dừng server sau khi hoàn tất khảo sát).*

---

## 2. Mục lục Giao diện & Các Phân hệ Nghiên cứu

| Đường dẫn (URL) | Phân hệ / Nội dung | Ý nghĩa Khoa học & Mục đích Kiểm chứng |
| :--- | :--- | :--- |
| **`/index.html`** | **Báo cáo Khoa học Toàn diện** (Slide deck tương tác) | Trình bày toàn bộ cấu trúc 8 giai đoạn của pipeline, cơ sở toán học, phân tích so sánh với LiDAR/CAD thủ công, số liệu thực nghiệm và phát hiện lỗi im lặng. |
| **`/full_factory/index.html`** | **Full Factory Scene Showcase** (13 slides chuyên sâu) | Trình chiếu toàn cảnh phân xưởng thực tế với **38 thực thể vật thể** hoàn thiện, phân loại động học robot (`DYNAMIC_LIFTABLE`, `STATIC_OBSTACLE`), độ bao phủ (coverage), kiểm chứng căn chỉnh IoU và khả năng nạp vào MuJoCo. |
| **`/viewer/index.html`** | **Scene 3D Tương tác Vật lý Thời gian thực** | Trực quan hóa mô phỏng tương tác: căn phòng quét thực tế + 8 vật thể rời rạc tích hợp trọng trường và va chạm raycast BVH; cho phép kéo-thả và kiểm tra tính tựa ổn định. |
| **`/objects3d/index.html`** | **Khảo sát Vật thể 3D Hoàn thiện (Hunyuan3D)** | Khảo sát chi tiết 3D đa hướng (Google `<model-viewer>` 360°, turntable video, ma trận 4 góc nhìn) của các vật thể điển hình hoàn thiện bù đắp mặt khuất. |
| **`media/full_factory_contact_sheet.html`** | **Bảng Tiếp xúc (Contact Sheet) 38 Vật thể** | Tổng hợp ảnh trực giao và trạng thái hoàn thiện của toàn bộ 38 vật thể trích xuất từ nhà máy. |
| **`KICH_BAN_QUAY.md`** | **Kịch bản Thuyết trình Kỹ thuật (6 phút)** | Khung cấu trúc trình bày chuẩn mực theo từng mốc thời gian dành cho báo cáo nghiệm thu và hội thảo chuyên môn. |

---

## 3. Thứ tự Khảo sát Khuyến nghị cho Hội đồng Đánh giá

1. **Khảo sát Báo cáo Tổng thể (`/index.html`):** Nắm bắt bài toán, nguyên lý trích xuất từ monocular RGB video, cấu trúc dữ liệu Twin Digital Data và số liệu đối sánh.
2. **Khảo sát Full Factory Scene (`/full_factory/index.html`):** Xem xét bằng chứng mở rộng toàn diện trên 38 vật thể thực tế trong nhà máy, ma trận kiểm chứng góc xoay IoU và phân loại phục vụ robot manipulation.
3. **Thực nghiệm Tương tác Vật lý (`/viewer/index.html`):** Kiểm tra tính nhất quán hình học và phản ứng vật lý va chạm đa vật thể.
4. **Kiểm tra Độ mịn & Độ kín Mô hình (`/objects3d/index.html`):** Đánh giá chất lượng tái tạo bề mặt watertight và texture synthesis.

---

## 4. Cơ chế Tính toán Vật lý trong Phân hệ Viewer (`/viewer/index.html`)

Phân hệ viewer không sử dụng hoạt ảnh đặt sẵn (pre-baked keyframes), mà tích hợp engine tính toán vật lý tương tác trực tiếp:
- **Tăng tốc Truy vấn Không gian (BVH Acceleration):** Sử dụng `three-mesh-bvh` để xây dựng Bounding Volume Hierarchy trên mesh của toàn bộ căn phòng và các vật thể xung quanh.
- **Phát hiện Va chạm Thực (Continuous Contact Raycasting):** Khi người dùng nhấc và thả một vật thể, hệ thống phóng chùm tia raycast từ các điểm chân đế (`feetOf`) để xác định chính xác bề mặt đỡ thấp nhất (mặt sàn bê tông hoặc mặt bàn làm việc `obj_0179_workbench`).
- **Động học Rơi Tự do & Tiêu hao Năng lượng:** Tính toán theo gia tốc trọng trường $g = 9.81\text{ m/s}^2$, lực cản không khí bậc hai $F_d = \frac{1}{2}\rho C_d A v^2$, và hệ số đàn hồi va chạm $\varepsilon = 0.12$.
- **Thao tác Điều khiển:**
  - **Chuột trái (kéo trên vật thể):** Nhấc vật thể theo mặt phẳng đối diện camera; hệ thống tự động nâng cao độ nếu vật thể đi qua vật cản.
  - **Thả chuột:** Kích hoạt pha rơi tự do cho tới khi tiếp xúc ổn định trên bề mặt đỡ.
  - **Chuột phải / Kéo nền:** Xoay tự do xung quanh tâm quan sát (OrbitControls).
  - **Con lăn chuột:** Thu phóng khoảng cách (Zoom).
  - **Bảng điều khiển bên trái:** Chọn nhanh vật thể để camera tự động căn khung (frame-in), theo dõi động năng va chạm (Joule), vận tốc rơi (m/s) và độ cao so với mặt đỡ.
  - **Nút "Đưa tất cả về chỗ cũ":** Khôi phục trạng thái ban đầu của toàn bộ scene.

---

## 5. Tái lập & Huấn luyện Pipeline từ Mã nguồn

Mã nguồn hoàn chỉnh của pipeline và các tài liệu kỹ thuật chuyên sâu được lưu trữ tại repository:
- Tài liệu hướng dẫn thực thi đầy đủ: `docs/guides/pipeline-runbook.md`
- Hướng dẫn thiết lập môi trường: `README.md` (`make setup && make test`)
- Cấu hình mô phỏng robot: `output/REPORT/twindata/` và các định dạng MJCF/MuJoCo.
