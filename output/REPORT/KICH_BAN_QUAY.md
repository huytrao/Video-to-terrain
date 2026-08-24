# Kịch bản quay video demo — 4 phút

Hai tab đã mở sẵn:

- **Tab 1 — Báo cáo:** http://127.0.0.1:8790/index.html
- **Tab 2 — Scene tương tác:** http://127.0.0.1:8790/viewer/index.html
- **Tab 3 — 3 vật hoàn thiện:** http://127.0.0.1:8790/objects3d/index.html

Quay màn hình, không cần webcam. Nói theo gạch đầu dòng, đừng đọc nguyên văn.

---

## 0:00–0:25 · Mở đầu — đặt câu hỏi trước, đừng khoe kết quả

**Màn hình:** Tab 1, phần tiêu đề.

> "Em đặt một câu hỏi: một đoạn video quay tay bằng điện thoại thì chứa được bao
> nhiêu thông tin 3D, nếu mình khai thác đến cùng?
>
> Đây là câu trả lời. Đầu vào là 852 khung hình quay tay. Đầu ra là bảy lớp dữ
> liệu — không LiDAR, không dựng tay."

**Thao tác:** cuộn chậm xuống sơ đồ vào/ra, dừng 3 giây.

---

## 0:25–0:55 · Đầu vào thật sự trông thế nào

**Màn hình:** cuộn tới hình bốn khung hình đầu vào.

> "Đây là chất lượng thật của đầu vào — không dàn dựng, không đèn, không bảng
> hiệu chuẩn. Anh để ý mấy chi tiết này: máy hút bụi xanh, cửa gỗ cam, dãy cửa
> sổ nhìn ra cây, sàn xanh lá."

**Thao tác:** rê chuột chỉ từng vật khi nhắc tên. Chi tiết này quan trọng — nó
cho người xem cái để đối chiếu ở bước sau.

---

## 0:55–1:30 · Căn phòng dựng lại — để người xem tự kiểm chứng

**Màn hình:** cuộn tới hình ba góc phòng.

> "Và đây là mesh dựng lại, 20,6 triệu tam giác. Anh so lại mấy chi tiết vừa
> nãy: cửa gỗ cam ở hai đầu, dãy cửa sổ với cây phía sau, sàn xanh lá, đèn trần.
>
> Màu lấy trực tiếp từ ảnh gốc, em không tô lại gì cả."

**Thao tác:** rê chuột chỉ đúng những vật vừa nhắc ở bước trước. Đây là khoảnh
khắc thuyết phục nhất — đừng lướt nhanh.

---

## 1:30–2:00 · Vật thể tách rời — và nói thật về chất lượng

**Màn hình:** cuộn tới hình năm vật thể.

> "Phần khó nhất là tách vật thể ra khỏi phòng. Đây là năm cái nhận ra được
> bằng mắt: máy công cụ, bàn nguội có ngăn kéo, cửa, máy khoan đứng, giá đỡ.
>
> Em nói thẳng con số: quy trình tách tự động ra 46 khối, trong đó 28 khối có
> hình dạng đặc. Số còn lại là mảnh vỏ bóc khỏi tường — dùng được cho va chạm
> nhưng không phải đồ vật. Phần em demo ngay sau đây là 8 vật đã kiểm bằng tay."

**Vì sao nói vậy:** sếp làm research sẽ tự kiểm. Nói trước con số thật thì phần
còn lại của báo cáo được tin; giấu đi mà bị phát hiện thì mất hết.

---

## 2:00–2:50 · Demo tương tác — phần gây ấn tượng nhất

**Màn hình:** chuyển sang Tab 2 (scene tương tác).

> "Cái này chạy trực tiếp trong trình duyệt, tám vật rời trong phòng."

**Thao tác theo thứ tự:**

1. Xoay camera một vòng chậm quanh phòng — 5 giây
2. Bấm một vật trong danh sách bên trái — camera bay tới, vật sáng lên
3. **Kéo một vật** lên cao rồi thả — nó rơi, chạm mặt đỡ, nảy nhẹ
4. Kéo một vật **đặt lên nóc bàn nguội** — nó dừng trên mặt bàn chứ không xuyên qua
5. Bấm **Đưa tất cả về chỗ cũ** — mọi vật về đúng vị trí ban đầu

> "Vật rơi xuống sàn hay xuống mặt bàn là do va chạm tính thật, không phải đặt sẵn
> độ cao. Đây là bước đệm để đưa scene này vào MuJoCo cho robot huấn luyện."

**Chuyển sang Tab 3 — 3 vật hoàn thiện:**

6. Kéo xoay `obj_0179_workbench` một vòng
7. Chỉ vào ảnh bốn góc bên dưới

> "Ba vật này đã được sinh nốt mặt camera không nhìn thấy, rồi ghép ngược về đúng
> vị trí trong phòng. Bàn nguội 613 nghìn tam giác, kín hoàn toàn — cái này thả vào
> mô phỏng vật lý được ngay."

---

## 2:50–3:30 · Đóng góp kỹ thuật — phần thể hiện năng lực

**Màn hình:** quay lại Tab 1, cuộn tới hai biểu đồ.

> "Phần đóng góp của em không phải chạy được pipeline, mà là tìm ra bốn lỗi
> im lặng — loại vẫn chạy trơn và vẫn trả số trông hợp lý.
>
> Biểu đồ trái: mô hình được chấm điểm bằng chính những ảnh nó đã học. Em giữ
> riêng 106 ảnh, giấu suốt 4 tiếng huấn luyện. Điểm thật là 24,17 chứ không phải
> 25,08. Và có tập giữ riêng thì mới đọc được đường cong — sau 70.000 vòng gần
> như không cải thiện, cắt ở đó tiết kiệm 1 tiếng 20 phút GPU mỗi lần chạy.
>
> Biểu đồ phải: mô hình ngôn ngữ bịa ra 'cây' ở 91% số khung, trong một xưởng
> kín. Em phát hiện bằng cách vẽ đường 'nhãn xuất hiện lần đầu ở đâu rồi bám dai
> bao lâu'. Sau khi sửa còn 12% — và em hỏi lại thì hoá ra lần này có thật,
> nhìn qua cửa sổ."

**Thao tác:** dừng ở mỗi biểu đồ 5 giây, rê chuột theo đường cong khi giải thích.

---

## 3:30–4:00 · Giới hạn và bước tiếp theo

**Màn hình:** cuộn tới mục giới hạn rồi mục việc tiếp theo.

> "Hai giới hạn em nói rõ. Thứ nhất, tỉ lệ chưa hiệu chuẩn tuyệt đối — em có thử
> suy hệ số từ kích thước chuẩn của cửa, thang, xô, nhưng các ước lượng không
> hội tụ nên em không dán một con số vào cho có.
>
> Thứ hai, có 334 vùng khuyết vì camera không nhìn được mặt sau máy móc. Em đã
> thử lấp nhưng bỏ hướng đó — nó tạo ra hình học bịa.
>
> Việc tiếp theo em xếp theo thứ tự: khoá tỉ lệ mét, sinh mặt khuất, chạy thử
> trên cảnh thứ hai để kiểm tính khái quát, rồi xuất MJCF cho MuJoCo."

---

## Ba điều nên tránh

**Đừng nói "hệ thống hoàn hảo".** Sếp research sẽ tìm chỗ hở ngay. Báo cáo này
mạnh vì nó tự nêu giới hạn trước.

**Đừng lướt nhanh phần đối chiếu ảnh gốc ↔ mesh.** Đó là chỗ người xem tự
thuyết phục mình, mạnh hơn mọi con số.

**Đừng đọc số liệu như đọc bảng.** Mỗi con số nên kèm "vì sao nó quan trọng" —
ví dụ 70.000 vòng không phải con số đẹp, mà là 1 tiếng 20 phút GPU tiết kiệm
được mỗi lần chạy.

---

## Nếu bị hỏi vặn

**"Sao PSNR thấp hơn báo cáo cũ?"**
> Vì con số cũ đo sai. Nó chấm trên ảnh đã học nên chỉ nói mô hình thuộc bài đến
> đâu. 24,17 thấp hơn nhưng là con số dùng được để so sánh giữa các lần chạy.

**"46 vật mà chỉ 28 dùng được, có ít quá không?"**
> Với một video quay tay không thiết bị thì đây là mức em đạt được hiện tại. Điểm
> quan trọng là quy trình tự động — thêm một cảnh chỉ tốn một lần đi bộ, nên số
> lượng mở rộng được. Chất lượng từng vật thì cải thiện bằng bước sinh mặt khuất.

**"Chạy được trên cảnh khác không?"**
> Mọi tham số đều suy từ chính cảnh chứ không hardcode — em đã sửa đúng chỗ đó.
> Nên về nguyên tắc là chạy được, nhưng em chưa kiểm chứng, và em ghi nó là việc
> ưu tiên trong danh sách tiếp theo.
