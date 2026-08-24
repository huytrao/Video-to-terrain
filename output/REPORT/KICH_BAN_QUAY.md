# Kịch bản quay video demo — 6 phút

**Mục tiêu:** cho người xem thấy *toàn bộ* những gì pipeline trích ra từ một
video điện thoại — không chỉ mesh đẹp, mà từng lớp dữ liệu dùng được riêng.

Bốn tab mở sẵn:

- **Tab 1 — Báo cáo:** http://127.0.0.1:8790/index.html
- **Tab 2 — Scene tương tác:** http://127.0.0.1:8790/viewer/index.html
- **Tab 3 — 3 vật hoàn thiện:** http://127.0.0.1:8790/objects3d/index.html

Quay màn hình, không cần webcam. Nói theo gạch đầu dòng, đừng đọc nguyên văn.
Mỗi đoạn phim nhúng trong báo cáo dài 3–7 giây, bấm play là đủ.

---

## 0:00–0:30 · Mở đầu — đặt câu hỏi trước, đừng khoe kết quả

**Màn hình:** Tab 1, tiêu đề + sơ đồ "Vào một thứ, ra một kho".

> "Em đặt một câu hỏi: một video quay tay bằng điện thoại chứa được bao nhiêu
> thông tin, nếu mình khai thác đến cùng?
>
> Câu trả lời: **bảy lớp dữ liệu**, tất cả sinh tự động từ đúng một video —
> camera poses, độ sâu, gaussian splatting, mesh có màu, cắt ngang kiểm chứng,
> tách vật thể, và scene robot. Không LiDAR, không dựng tay."

**Thao tác:** cuộn chậm qua sơ đồ, dừng 3 giây ở dòng "ra một kho".

---

## 0:30–2:00 · Bảy lớp dữ liệu — phần chính của video

Mục "Đọc kết quả" trong Tab 1 có 4 video nhúng; mỗi cái play 3–5 giây rồi pause,
vừa phát vừa nói. Đây là chỗ thể hiện breadth — đừng lướt.

### Lớp 1 — Đầu vào (video `01_dau_vao_video.mp4`, 7 giây)

> "Đây là chất lượng thật của đầu vào: quay tay, không đèn, không bảng hiệu chuẩn.
> Anh để ý máy hút bụi xanh, cửa gỗ cam, dãy cửa sổ nhìn ra cây — em sẽ đối chiếu
> mấy chi tiết này ở các lớp sau."

### Lớp 2 — Độ sâu (video `02_do_sau_da3.mp4`)

> "Từ mỗi khung hình, Depth-Anything-3 suy ra khoảng cách tới từng bề mặt —
> không cảm biến, thuần học từ ảnh. Đây là tầng dữ liệu đầu tiên."

### Lớp 3 — Gaussian splatting (video `03_gaussian_splatting.md`)

> "Song song đó là gaussian splatting: cảnh được biểu diễn bằng hàng triệu hạt
> quang thực, nhìn từ góc mới vẫn photorealistic. Dùng để render novel views,
> làm ground truth hình ảnh cho robot."

### Lớp 4 — Mesh có màu (video `04_mesh_3d.mp4`)

> "Gộp depth + splatting lại thành mesh 20,6 triệu tam giác, màu lấy trực tiếp
> từ ảnh gốc — em không tô lại gì cả."

### Lớp 5 — Cắt ngang (video `05_cat_ngang_phong.mp4`)

> "Cắt bổ đôi phòng để xác nhận hình học bên trong đặc, không rỗng — đây là
> bước kiểm định, không phải kết quả."

### Lớp 6 — Tách vật thể (mục "Vật thể tách được", hình năm vật)

> "Phần khó nhất: tách đồ khỏi phòng. Em nói thẳng con số — tự động ra 46 khối,
> 28 khối đặc; phần demo sau là 8 vật đã kiểm tay. Sếp research sẽ tự kiểm, nên
> em nói số thật ngay từ đầu."

### Lớp 7 — Hộp vật lý (video `06_hop_vat_the.mp4`)

> "Cuối cùng, mỗi vật được bọc hộp va chạm đúng kích thước — đây chính là thứ
> robot dùng để chạm, nắm, di chuyển đồ. Video này là hộp đè lên phòng, nhìn từ
> nhiều góc."

---

## 2:00–2:40 · Đối chiếu ảnh gốc ↔ mesh — chỗ người xem tự thuyết phục mình

**Màn hình:** mục "Căn phòng dựng được" — hình ba góc phòng.

> "Giờ đối chiếu với ảnh gốc lúc nãy: cửa gỗ cam hai đầu, dãy cửa sổ với cây,
> sàn xanh lá, đèn trần — đều có trong mesh. Màu lấy từ ảnh gốc nên nhìn như
> photo, nhưng đây là hình học thật, đo đạc được."

**Thao tác:** rê chuột chỉ đúng những chi tiết đã nhắc ở lớp 1. Đừng lướt nhanh.

---

## 2:40–3:40 · Demo tương tác — ấn tượng nhất

**Màn hình:** Tab 2 (viewer scene cũ, 8 vật).

> "Cái này chạy trực tiếp trong trình duyệt — tám vật rời có vật lý thật."

**Thao tác theo thứ tự:**

1. Xoay camera một vòng chậm quanh phòng — 5 giây
2. Bấm tên một vật ở danh sách — camera bay tới, vật sáng lên
3. **Kéo một vật lên cao rồi thả** — nó rơi, chạm mặt, nảy nhẹ
4. Kéo vật **đặt lên nóc bàn nguội** — dừng trên mặt bàn, không xuyên qua
5. **"Đưa tất cả về chỗ cũ"** — reset về vị trí ban đầu

> "Vật rơi xuống sàn hay xuống bàn là do raycast va chạm tính thật, không phải
> đặt sẵn độ cao. Đây là bước đệm đưa scene vào MuJoCo cho robot huấn luyện."

**Chuyển sang Tab 3 — 3 vật hoàn thiện:**

6. Kéo xoay `obj_0179_workbench` một vòng
7. Chỉ vào ảnh bốn góc bên dưới

> "Ba vật này đã được Hunyuan3D sinh nốt mặt camera không thấy, rồi ghép ngược
> về đúng vị trí trong phòng. Bàn nguội 613 nghìn tam giác, kín hoàn toàn —
> thả vào mô phỏng vật lý được ngay."

---

## 3:40–4:20 · So sánh với cách làm khác

**Màn hình:** mục "So với các cách làm khác".

> "So với quét LiDAR hay dựng tay Blender thì thế nào? LiDAR thắng về độ chính
> xác hình học; dựng tay thắng về tách vật sạch. Nhưng hàng cuối bảng mới là
> điểm khác biệt thật: thêm một cảnh với cách này chỉ tốn **một lần đi bộ quay
> video**, còn hai cách kia tốn thiết bị hoặc tuần nhân công. Với bài toán cần
> hàng trăm cảnh đa dạng cho robot, đây mới là con đường khả thi."

---

## 4:20–5:10 · Đóng góp kỹ thuật — tìm ra lỗi im lặng

**Màn hình:** quay lại Tab 1, hai biểu đồ ở mục "Bốn lỗi im lặng".

> "Phần đóng góp của em không phải chạy được pipeline — mà là bắt được bốn lỗi
> chạy trơn nhưng trả số sai mà không phép đo nào bắt được.
>
> Biểu đồ trái: mô hình bị chấm trên chính ảnh nó học. Em giữ riêng 106 ảnh,
> giấu suốt 4 tiếng huấn luyện — điểm thật 24,17 chứ không phải 25,08. Và nhờ
> có tập giữ riêng mới đọc được đường cong: sau 70.000 vòng gần như không cải
> thiện, cắt ở đó tiết kiệm 1 tiếng 20 phút GPU mỗi lần chạy.
>
> Biểu đồ phải: mô hình ngôn ngữ bịa 'cây' ở 91% khung, trong xưởng kín. Em
> phát hiện bằng cách vẽ 'nhãn xuất hiện lần đầu ở đâu rồi bám dai bao lâu'.
> Sau sửa còn 12% — và lần này có thật, nhìn xuyên cửa sổ."

**Thao tác:** dừng 5 giây mỗi biểu đồ, rê chuột theo đường cong khi nói.

---

## 5:10–5:40 · Giới hạn — nói trước để báo cáo được tin

**M màn hình:** mục "Giới hạn hiện tại".

> "Hai giới hạn em nêu rõ. Thứ nhất, tỉ lệ chưa hiệu chuẩn tuyệt đối — em thử
> suy từ kích thước cửa, thang, xô nhưng không hội tụ, nên em không dán số cho
> có. Thứ hai, 334 vùng khuyết vì camera không nhìn được mặt sau máy móc; em đã
> thử lấp nhưng bỏ hướng đó vì tạo hình học bịa."

---

## 5:40–6:00 · Kết — quay về câu hỏi đầu

**Màn hình:** Tab 1, cuộn về sơ đồ "Vào một thứ, ra một kho".

> "Quay lại câu hỏi đầu: bao nhiêu thông tin nằm trong một video? Câu trả lời
> của em là bảy lớp — mỗi lớp một file, một định dạng chuẩn, dùng độc lập cho
> việc khác nhau: render novel views, huấn luyện policy, mô phỏng vật lý, hay
> làm ground truth. Việc tiếp theo: khoá tỉ lệ mét, sinh mặt khuất, chạy cảnh
> thứ hai để kiểm tính khái quát, rồi xuất MJCF cho MuJoCo."

---

## Ba điều nên tránh

**Đừng nói "hệ thống hoàn hảo".** Sếp research sẽ tìm chỗ hở ngay. Báo cáo này
mạnh vì tự nêu giới hạn trước.

**Đừng bỏ qua lớp gaussian splatting.** Nhiều người chưa biết kỹ thuật này —
một câu "photorealistic từ mọi góc nhìn, không cần mesh" là đủ gây ấn tượng,
và nó chứng minh breadth của data stack.

**Đừng đọc số liệu như đọc bảng.** Mỗi con số kèm "vì sao quan trọng": 70.000
vòng không phải số đẹp, mà là 1 giờ 20 phút GPU tiết kiệm mỗi lần chạy.

---

## Nếu bị hỏi vặn

**"Sao PSNR thấp hơn báo cáo cũ?"**
> Vì con số cũ đo sai — chấm trên ảnh đã học nên chỉ nói mô hình thuộc bài đến
> đâu. 24,17 thấp hơn nhưng là con số so sánh được giữa các lần chạy.

**"46 vật mà chỉ 28 dùng được, ít quá?"**
> Với một video quay tay không thiết bị thì đây là mức hiện tại. Điểm quan trọng
> là quy trình tự động — thêm một cảnh chỉ tốn một lần đi bộ quay, nên mở rộng
> được. Chất lượng từng vật cải thiện bằng bước sinh mặt khuất.

**"Splatting dùng làm gì nếu đã có mesh?"**
> Hai cái phục vụ việc khác nhau. Mesh cho vật lý — va chạm cần bề mặt kín. Splatting
> cho thị giác — render photorealistic từ góc camera chưa quay, dùng làm môi trường
> huấn luyện vision hay ground truth so sánh.

**"Chạy được trên cảnh khác không?"**
> Mọi tham số đều suy từ chính cảnh, không hardcode — em đã sửa đúng chỗ đó. Về
> nguyên tắc chạy được, nhưng em chưa kiểm chứng và ghi nó là việc ưu tiên.
