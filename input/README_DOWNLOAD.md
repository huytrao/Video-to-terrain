# Hướng dẫn tải dữ liệu từ Hugging Face DL3DV-10K

Thư mục này chứa script và tài liệu để tải các scenes từ bộ dữ liệu **DL3DV-10K** trên Hugging Face.

---

## 1. Yêu cầu trước khi tải
1. Token Hugging Face đã được lưu trong file [`.env`](.env) ở thư mục gốc:
   ```env
   HF_TOKEN=hf_...
   ```
2. Tài khoản Hugging Face của bạn đã bấm **Accept / Access** trên các repo tương ứng:
   - [DL3DV-ALL-video](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-video) (Để tải video gốc `.mp4`)
   - [DL3DV-ALL-4K](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-4K) (Để tải frames 4K + camera poses)
   - [DL3DV-ALL-2K](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-2K)
   - [DL3DV-ALL-960P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P)
   - [DL3DV-ALL-480P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P)

---

## 2. Cách dùng nhanh bằng script `input/download.sh`

### Tải video factory scene mặc định:
```bash
./input/download.sh
```

### Tải tùy biến theo tham số:
```bash
./input/download.sh <HASH> <SUBSET> <FILE_TYPE> <RESOLUTION> <OUTPUT_DIR>
```

**Ví dụ:**
- Tải **video** của scene factory 4K:
  ```bash
  ./input/download.sh 23cdcd0cd5b44abb350ccbc6c780712d6b0c7af8143f8292f1d3e8ecfc9f4edd 1K video 4K input/factory_scene_4k
  ```

- Tải **ảnh 4K và camera poses**:
  ```bash
  ./input/download.sh 23cdcd0cd5b44abb350ccbc6c780712d6b0c7af8143f8292f1d3e8ecfc9f4edd 1K images+poses 4K input/factory_scene_4k
  ```

- Tải **colmap cache**:
  ```bash
  ./input/download.sh 23cdcd0cd5b44abb350ccbc6c780712d6b0c7af8143f8292f1d3e8ecfc9f4edd 1K colmap_cache 4K input/factory_scene_4k
  ```

---

## 3. Cách gọi trực tiếp bằng `download.py`

```bash
./.venv/bin/python download.py \
  --odir input/factory_scene_4k \
  --subset 1K \
  --resolution 4K \
  --file_type video \
  --hash 23cdcd0cd5b44abb350ccbc6c780712d6b0c7af8143f8292f1d3e8ecfc9f4edd
```
