# T1 · Camera degradation health score — Nhóm GTAA

**Nền tảng:** xe ADAS · **Sensor:** camera trước · **Tính năng:** phát hiện vật thể (YOLOv8n)

**Claim:** khi camera bị nhòe, thiếu sáng hoặc nhiễu mạnh hơn, các chỉ số sức khỏe ảnh (blur score,
saturation ratio, entropy) thay đổi và mAP@0.5 của detector giảm. Câu hỏi kèm theo: chỉ số sức khỏe
nào báo trước được mAP giảm, và độ tự tin của detector có tự nhận ra ảnh hỏng không?

Thành viên: xem [`TEAMMATES.md`](TEAMMATES.md).

## Cấu trúc

```
benchmark.ipynb         notebook chạy chính, đã lưu sẵn output (bằng chứng chạy): bảng, plot, ảnh trước/sau
src/benchmark.py        toàn bộ logic benchmark (tạo lỗi -> đo sức khỏe ảnh -> chạy YOLOv8n -> bảng + plot)
src/check_s5_port.py    kiểm tra phần tạo lỗi viết lại khớp với code gốc của repo S5
results/<tag>/          kết quả: summary.csv, per_image.csv, topk_share.csv, plot_*.png, examples.jpg, run_info.json
reports/                báo cáo riêng của từng thành viên
data/bdd100k/           dữ liệu (không đưa lên git)
```

## Cài đặt

**1. Môi trường** (Python 3.10, đã thử với các phiên bản trong `requirements.txt`):
```bash
pip install -r requirements.txt
```

**2. Dữ liệu** — BDD100K, 10.000 ảnh tập val, bản FiftyOne trên HuggingFace
([`dgural/bdd100k`](https://huggingface.co/datasets/dgural/bdd100k), khoảng 700 MB). Chỉ cần
`samples.json` (nhãn box + thời tiết + thời điểm) và thư mục `data/` (ảnh):
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('dgural/bdd100k', repo_type='dataset', allow_patterns=['samples.json', 'data/*'], local_dir='data/bdd100k')"
```

Đã có BDD100K ở chỗ khác thì khỏi tải, chỉ cần trỏ biến môi trường tới đó:
```bash
export BDD100K_ROOT=/đường/dẫn/tới/bdd100k     # thư mục chứa samples.json và data/
```

**3. Weights:** không cần làm gì — lần chạy đầu ultralytics tự tải `yolov8n.pt` (6 MB, COCO).
Muốn dùng file khác: `export YOLO_WEIGHTS=/đường/dẫn/yolov8n.pt`.

## Chạy

Cách 1 — mở `benchmark.ipynb` (Jupyter / VS Code / Colab) và *Run All*.

Cách 2 — dòng lệnh:
```bash
python src/benchmark.py --n 8 --smoke --device cpu      # thử nhanh ~30 s, ghi vào results/smoke/
python src/benchmark.py --n 200                         # benchmark chính -> results/main/
python src/benchmark.py --n 200 --families overexpose --no-night --tag glare   # mở rộng: chói sáng
```

Tham số hay dùng: `--device cpu|0|mps` (bỏ trống thì tự chọn GPU nếu có, không thì CPU), `--seed`
(mặc định 0), `--families motion_blur dark noise overexpose`, `--tag` (tên thư mục trong `results/`).

### Chạy ở đâu cũng được

Toàn bộ benchmark là **một file Python** (`src/benchmark.py`), không phụ thuộc hệ thống nào:
đường dẫn lấy từ biến môi trường hoặc mặc định tương đối trong repo, weights tự tải, GPU tự phát hiện.
Thời gian chạy `--n 200` (11 điều kiện, khoảng 2.200 lượt suy luận): vài phút trên GPU, chừng 10–15 phút trên CPU.

| Hệ thống | Cách chạy |
|---|---|
| Laptop / PC (Linux, macOS, Windows) | Làm theo "Cài đặt" rồi `python src/benchmark.py --n 200`. Windows đặt biến môi trường bằng `set BDD100K_ROOT=...` (cmd) hoặc `$env:BDD100K_ROOT="..."` (PowerShell) |
| Google Colab | `!git clone <repo>` → `%cd <repo>` → `!pip install -r requirements.txt` → chạy lệnh tải dữ liệu ở trên → `!python src/benchmark.py --n 200` |
| Kaggle Notebook | Bật GPU trong Settings, rồi làm như Colab (`!git clone`, `!pip install`, tải dữ liệu, `!python src/benchmark.py --n 200`) |
| Máy chủ / server qua SSH | Như laptop; chạy nền để không mất khi đóng terminal: `nohup python src/benchmark.py --n 200 > run.log 2>&1 &` |

## Thiết kế benchmark

| Thành phần | Cách làm |
|---|---|
| Dữ liệu | 200 ảnh BDD100K **ban ngày, trời quang**, bốc ngẫu nhiên với seed 0; nhãn GT 7 lớp đổi sang id COCO (car, truck, bus, person/rider, traffic light, bicycle, motorcycle) |
| Baseline | Ảnh gốc, chỉ qua nén lại JPEG q95 như mọi điều kiện khác |
| Lỗi theo S5 (severity 1 / 3 / 5) | **zoom blur** trước-sau, phóng to tới **1,02 / 1,06 / 1,10** (xe lao về phía trước, `ImageMotionBlurFrontBack`) · **nhiễu Gauss σ = 20,4 / 45,9 / 96,9** trên thang 0–255 (ImageNet-C qua `ImageAddGaussianNoise`) |
| Lỗi nhóm tự thêm (S5 không có) | thiếu sáng gain **×0,5 / ×0,25 / ×0,1** · *(mở rộng)* chói gain **×1,5 / ×2,5 / ×4** rồi cắt ở 255 |
| Quy tắc | mỗi lần đổi **một** yếu tố, cùng 200 ảnh, cùng cách đo; nhiễu có seed theo từng ảnh |
| Lỗi thật | 200 ảnh **ban đêm** trời quang của BDD100K — *ảnh khác* nên chỉ để đối chiếu, không cùng baseline |
| Detector | YOLOv8n COCO, không huấn luyện lại, imgsz 640 |

| Metric | Định nghĩa | Đơn vị · chiều tốt | Đo cái gì |
|---|---|---|---|
| `blur_score` | variance của Laplacian trên ảnh xám (trung vị theo ảnh) | mức xám² · cao = nét | sức khỏe ảnh |
| `saturation_ratio` | tỉ lệ pixel ≤ 5 hoặc ≥ 250 | 0–1 · thấp = tốt | sức khỏe ảnh (mất thông tin do quá tối/sáng) |
| `entropy_bits` | entropy Shannon histogram xám | bit (tối đa 8) · cao = nhiều thông tin | sức khỏe ảnh |
| `noise_est` | ước lượng σ nhiễu Immerkær (1996), trung vị theo ảnh — *cải tiến nhóm thêm* | mức xám · thấp = tốt | sức khỏe ảnh (nhiễu) |
| `mAP50`, `mAP50_95` | mAP của YOLOv8n so với nhãn GT (`ultralytics val`) | 0–1 · cao = tốt | **chất lượng thuật toán, đo thật** |
| `RCE` | (mAP50_sạch − mAP50_lỗi) / mAP50_sạch — đúng công thức của S5 | 0–1 · thấp = bền | mức suy giảm tương đối |
| `uncertainty` | 1 − max confidence (predict ở conf 0,05; ảnh không có box -> 0) | 0–1 | độ tự tin của detector, tín hiệu "least confidence" của active learning |
| `share_of_topk` | trộn baseline + 3 mức lỗi, lấy top-200 theo `uncertainty`, tỉ lệ từng mức | 0–1 · 25 % = không phân biệt | uncertainty có ưu tiên ảnh hỏng không |

## Nguồn tham khảo

- **[S5]** Dong và cộng sự, *Benchmarking Robustness of 3D Object Detection to Common Corruptions in
  Autonomous Driving*, CVPR 2023 — [arXiv 2303.11040](https://arxiv.org/abs/2303.11040) ·
  repo [thu-ml/3D_Corruptions_AD](https://github.com/thu-ml/3D_Corruptions_AD) @ commit `48c23f7`.
  - *Nguồn làm gì:* input là point cloud LiDAR + ảnh camera (KITTI-C, nuScenes-C, Waymo-C); áp 27 loại
    lỗi chia 5 nhóm (thời tiết, sensor, chuyển động, vật thể, lệch calibration/thời gian) × 5 mức severity
    lên 24 detector 3D (LiDAR-only, camera-only, fusion). Metric: AP dưới lỗi và **RCE = (AP_sạch −
    AP_lỗi) / AP_sạch**. Kết luận nguồn nêu trong abstract: lỗi mức chuyển động nguy hiểm nhất; fusion bền
    hơn đơn cảm biến; detector chỉ dùng camera rất dễ hỏng trước lỗi ảnh.
  - *Limitation (theo phạm vi nguồn):* lỗi là mô phỏng, đánh giá trên detector 3D và ba dataset trên.
  - *Nhóm dùng gì:* **đúng định nghĩa severity** của hai lỗi camera — `ImageMotionBlurFrontBack` (zoom
    blur) và `ImageAddGaussianNoise` — trong `Camera_corruptions.py`, viết lại bằng OpenCV vì file gốc
    cần imgaug + mmdet3d. `src/check_s5_port.py` so với logic gốc: lệch tối đa 1 mức xám
    (`results/check_s5_port.txt`).
  - *Nhóm không làm:* không chạy detector 3D, không dùng KITTI/nuScenes (quá 120 phút và cần mmdet3d).
    Số của paper (3D, KITTI/nuScenes) và số nhóm đo (2D, BDD100K, YOLOv8n) **không so trực tiếp được**.

## Kết quả (`results/main/`, 200 ảnh/điều kiện, seed 0)

| Điều kiện | blur score | saturation | noise_est | uncertainty | mAP50 | RCE |
|---|---|---|---|---|---|---|
| clean | 327 | 0,009 | 0,75 | 0,208 | 0,295 | 0 |
| zoom blur s1 / s3 / s5 | 70 / 26 / 16 | ≈ 0,005 | 0,44 / 0,34 / 0,30 | 0,21 / 0,28 / 0,38 | 0,236 / 0,110 / 0,078 | 0,20 / 0,63 / 0,74 |
| nhiễu s1 / s3 / s5 | **3.989 / 15.984 / 47.793** | ≈ 0,015 | 13,7 / 27,9 / 48,8 | 0,23 / 0,33 / 0,46 | 0,246 / 0,171 / 0,039 | 0,17 / 0,42 / 0,87 |
| tối ×0,5 / ×0,25 / ×0,1 | 84 / 22 / 4,4 | 0,013 / 0,049 / 0,279 | 0,43 / 0,26 / 0,15 | 0,21 / 0,23 / 0,28 | 0,280 / 0,260 / 0,192 | 0,05 / 0,12 / 0,35 |
| đêm thật (ảnh khác) | 60 | 0,166 | 0,21 | 0,357 | 0,146 | — |

- mAP50 giảm đơn điệu theo mức lỗi ở cả ba họ; nặng nhất là nhiễu severity 5 (RCE 0,87).
- **Failure case:** nhiễu làm blur score tăng 146 lần và entropy tăng — chỉ số sức khỏe báo ảnh "nét hơn" đúng lúc
  mAP sập; cổng blur + saturation chỉ gắn cờ 1,5–3,5 % ảnh nhiễu.
- **Cải tiến đã thử:** thêm `noise_est` (Immerkær 1996) → cổng gắn cờ 100 % ảnh nhiễu, đổi lại báo nhầm trên ảnh sạch
  tăng 9,5 % → 14,5 %.
- Uncertainty của detector ưu tiên ảnh hỏng: ở mức nặng nhất, ảnh lỗi chiếm 35–54 % top-200 (25 % nếu không liên quan).

Phân tích đầy đủ, failure case và engineering decision: [`reports/report_Nguyen_Son_Giang.md`](reports/report_Nguyen_Son_Giang.md).
