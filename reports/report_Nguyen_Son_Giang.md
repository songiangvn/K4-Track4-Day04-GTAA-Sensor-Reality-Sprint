# Báo cáo — T1 · Camera degradation health score

**Người viết:** Nguyễn Sơn Giang · 2A202602747 · **Nhóm:** GTAA (xem `TEAMMATES.md`)
**Bằng chứng chạy:** `benchmark.ipynb` (đã lưu output) · số liệu `results/main/summary.csv`, `per_image.csv`,
`topk_share.csv` · plot `results/main/plot_*.png` · ảnh trước/sau `results/main/examples.jpg` · cấu hình
`results/main/run_info.json` · lệnh chạy lại: `python src/benchmark.py --n 200` (seed 0)

> Quy ước trong báo cáo: **[Đo]** = nhóm tự đo trong lab · **[Nguồn]** = paper/repo nói · **[Giả thuyết]** = suy luận
> kỹ thuật, chưa đo.

---

## 1. Problem

- **Nền tảng / tính năng / sensor:** xe ADAS · phát hiện vật thể (xe, người, xe đạp/máy, đèn giao thông) · camera trước.
- **Failure thực tế:** camera vẫn trả ảnh nhưng ảnh xấu đi — nhòe khi xe chạy nhanh (ego-motion), nhiễu cảm biến khi
  tăng gain, thiếu sáng. Detector vẫn chạy và vẫn trả box, nên hệ thống **không tự biết frame nào đáng tin**.
- **Claim ban đầu:** mức lỗi tăng → chỉ số sức khỏe ảnh (blur score, saturation ratio, entropy) đổi theo và mAP@0.5
  giảm. Câu hỏi kèm theo: (a) chỉ số sức khỏe nào báo trước được mAP giảm, (b) độ tự tin của detector có tự nhận ra
  ảnh hỏng không — điều quan trọng khi dùng uncertainty để chọn ảnh gán nhãn trong data loop.

## 2. Method

**Nguồn — [S5] Dong et al., *Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous
Driving*, CVPR 2023** — [arXiv 2303.11040](https://arxiv.org/abs/2303.11040) · repo
[thu-ml/3D_Corruptions_AD](https://github.com/thu-ml/3D_Corruptions_AD) @ commit `48c23f7`.

| | |
|---|---|
| Input → output | Point cloud LiDAR + ảnh camera (KITTI-C, nuScenes-C, Waymo-C) → AP của 24 detector 3D dưới 27 loại lỗi × 5 mức severity |
| Metric | AP dưới lỗi; **RCE = (AP_sạch − AP_lỗi) / AP_sạch** |
| Kết luận **[Nguồn]** (abstract) | lỗi mức chuyển động nguy hiểm nhất; fusion LiDAR–camera bền hơn đơn cảm biến; detector chỉ dùng camera rất dễ hỏng trước lỗi ảnh |
| Limitation (phạm vi nguồn) | lỗi là mô phỏng; detector 3D; ba dataset trên |

**Nhóm tái hiện phần nào:** dùng **đúng định nghĩa severity 1/3/5** của hai lỗi camera trong
`Camera_corruptions.py` — zoom blur trước-sau `ImageMotionBlurFrontBack` (lỗi motion blur camera mặc định của repo
trên KITTI, `kitti_dataset.py:182`) và nhiễu Gauss `ImageAddGaussianNoise` (ImageNet-C) — và metric **RCE**. File gốc
phụ thuộc imgaug + mmdet3d nên nhóm viết lại bằng OpenCV; `src/check_s5_port.py` so với logic gốc: **lệch tối đa 1
mức xám** (`results/check_s5_port.txt`). Nhóm thêm một họ lỗi S5 không có: thiếu sáng (nhân pixel ×0,5/×0,25/×0,1).

**Không tái hiện:** detector 3D và KITTI/nuScenes (cần mmdet3d, quá 120 phút). Thay bằng YOLOv8n (weights COCO,
không huấn luyện lại) trên BDD100K. Vì vậy **số của nhóm không so trực tiếp với số của paper** — khác detector,
dataset, 2D/3D.

## 3. Benchmark

- **Dữ liệu:** BDD100K val (bản FiftyOne `dgural/bdd100k`), 200 ảnh **ban ngày, trời quang** (seed 0), 3.026 box GT
  thuộc 7 lớp. Đối chiếu: 200 ảnh **đêm thật** (ảnh khác, 2.739 box).
- **Cấu hình:** mỗi điều kiện đổi **một** yếu tố trên cùng 200 ảnh; mọi điều kiện cùng qua JPEG q95; YOLOv8n imgsz 640.
- **Metric:** mAP@0.5 so với GT (thuật toán, đo thật) và RCE; sức khỏe ảnh không cần nhãn: blur score (variance
  Laplacian), saturation ratio (tỉ lệ pixel ≤5 hoặc ≥250), entropy; uncertainty = 1 − max confidence.

**Kết quả [Đo]** (`results/main/summary.csv`; blur score là trung vị, còn lại trung bình theo ảnh):

| Điều kiện | Severity S5 | blur score | saturation | entropy (bit) | uncertainty | **mAP50** | **RCE** |
|---|---|---|---|---|---|---|---|
| clean (baseline) | — | 327 | 0,009 | 7,49 | 0,208 | **0,295** | 0 |
| zoom blur 1,02 | 1 | 70 | 0,008 | 7,47 | 0,211 | 0,236 | 0,20 |
| zoom blur 1,06 | 3 | 26 | 0,006 | 7,44 | 0,278 | 0,110 | 0,63 |
| zoom blur 1,10 | 5 | 16 | 0,004 | 7,41 | 0,379 | 0,078 | 0,74 |
| nhiễu σ 20,4 | 1 | 3.989 | 0,012 | 7,68 | 0,234 | 0,246 | 0,17 |
| nhiễu σ 45,9 | 3 | 15.984 | 0,017 | 7,80 | 0,329 | 0,171 | 0,42 |
| nhiễu σ 96,9 | 5 | **47.793** | 0,019 | **7,90** | 0,455 | **0,039** | **0,87** |
| tối ×0,5 | — | 84 | 0,013 | 6,50 | 0,214 | 0,280 | 0,05 |
| tối ×0,25 | — | 22 | 0,049 | 5,51 | 0,229 | 0,260 | 0,12 |
| tối ×0,1 | — | 4,4 | 0,279 | 4,22 | 0,279 | 0,192 | 0,35 |
| đêm thật *(ảnh khác)* | — | 60 | 0,166 | 5,81 | 0,357 | 0,146 | 0,50* |

\* RCE của ảnh đêm so với ảnh ngày *khác*, chỉ để tham khảo.

**Nhóm quan sát được [Đo]:**
1. Claim đúng về hướng: mAP50 giảm đơn điệu theo mức lỗi ở cả 3 họ. Ở severity 5, zoom blur mất 74 % và nhiễu mất
   87 % mAP50 so với baseline; thiếu sáng ×0,1 chỉ mất 35 %.
2. Trong từng họ blur/tối, blur score giảm cùng chiều với mAP — nhưng **gộp cả 10 điều kiện mô phỏng**, tương quan
   hạng Spearman giữa blur score và mAP50 gần 0 (ρ = −0,02); entropy ρ = −0,29 (sai dấu); uncertainty ρ = −0,89.
3. Khi trộn baseline với 3 mức lỗi rồi lấy top-200 ảnh có uncertainty cao nhất, mức nặng nhất chiếm **53,5 %**
   (nhiễu), **44,5 %** (zoom blur), **35 %** (tối) — so với 25 % nếu uncertainty không liên quan tới lỗi; ảnh sạch
   chỉ còn 9,5–21,5 %.

Baseline mAP50 = 0,295 thấp vì YOLOv8n COCO chạy thẳng trên BDD100K (khác miền, nhiều vật nhỏ như đèn giao thông);
nhóm so sánh **tương đối** (RCE) trên cùng baseline, không so mức tuyệt đối.

## 4. Failure case

**Nhiễu cảm biến đánh lừa chỉ số sức khỏe ảnh.** Điều kiện `noise_sigma96.9` (S5 severity 5), 200 ảnh, xem hàng 7
bảng trên, `plot_noise.png` và ô thứ 3 hàng 2 của `examples.jpg`:

- **[Đo]** mAP50 sập từ 0,295 xuống **0,039** (RCE 0,87); 31,5 % ảnh không còn box nào kể cả ở ngưỡng confidence 0,05.
- **[Đo]** cùng lúc đó blur score **tăng 146 lần** (327 → 47.793) và entropy **tăng** (7,49 → 7,90): theo hai chỉ số
  này, ảnh trông *nét hơn và giàu thông tin hơn* ảnh sạch. Saturation gần như không đổi (0,009 → 0,019).
- **[Đo]** cổng sức khỏe dựa trên blur score + saturation (ngưỡng phân vị 5 %/95 % của ảnh sạch) chỉ gắn cờ
  **1,5–3,5 %** ảnh nhiễu ở cả 3 mức — tức để lọt gần như toàn bộ.
- **Nguyên nhân:** variance của Laplacian đo năng lượng tần số cao; nhiễu Gauss *chính là* năng lượng tần số cao,
  nên chỉ số không phân biệt được cạnh thật với hạt nhiễu. Entropy tăng vì nhiễu trải đều histogram.
- **Hệ quả [Giả thuyết]:** một hệ thống dùng blur score để quyết định có tin camera hay không sẽ *tăng* độ tin camera
  đúng lúc detector gần như mù — nguy hiểm hơn là không có health check. Trên xe thật, nhiễu mạnh thường đi cùng
  thiếu sáng (camera tăng gain), nên hai lỗi chồng nhau còn tệ hơn từng lỗi riêng; nhóm chưa đo tổ hợp này.

Một chỗ lệch thứ hai cùng gốc: blur score cũng nhạy với **độ sáng** — tối ×0,1 có blur score 4,4 (thấp hơn cả zoom
blur severity 5 là 16) nhưng mAP chỉ mất 35 % so với 74 %; cổng dựa trên blur score gắn cờ 64,5 % ảnh tối ×0,5 dù mAP
chỉ giảm 5 %. Variance Laplacian tỉ lệ với bình phương độ tương phản, nên nó trộn "nhòe" với "tối".

**Paper cho biết [Nguồn]:** detector chỉ dùng camera rất dễ hỏng trước lỗi ảnh, và lỗi chuyển động nguy hiểm nhất.
Kết quả của nhóm cùng chiều (zoom blur và nhiễu severity 5 làm mất 74–87 % mAP50), nhưng khác detector/dataset nên
không đặt hai con số cạnh nhau như so sánh trực tiếp.

## 5. Engineering decision

**Cải tiến đã thử nhanh [Đo]:** thêm chỉ số **ước lượng nhiễu** `noise_est` (Immerkær 1996, *Fast noise variance
estimation*: lọc bằng mặt nạ Laplacian bậc hai, triệt tiêu cạnh và vùng phẳng, chỉ giữ nhiễu). Trung vị theo điều
kiện: sạch 0,75 → nhiễu 13,7 / 27,9 / 48,8, trong khi blur và tối đều *giảm* (0,30 / 0,15) — tức nó tách được nhiễu
khỏi hai lỗi kia. Cổng B = cổng cũ **hoặc** `noise_est` > phân vị 95 % ảnh sạch (mục 8 của notebook):

| | Cổng A (blur + saturation) | Cổng B (+ noise_est) |
|---|---|---|
| Gắn cờ ảnh nhiễu σ 20,4 / 45,9 / 96,9 | 2,5 % / 3,5 % / 1,5 % | **100 % / 100 % / 100 %** |
| Gắn cờ zoom blur s1 / s3 / s5 | 72,5 % / 99,5 % / 100 % | như cổng A |
| Gắn cờ ảnh đêm thật | 83,5 % | 83,5 % |
| Báo nhầm trên ảnh sạch | 9,5 % | 14,5 % |

**Nếu đưa vào xe thật:**
1. **Camera health gate trên mỗi frame** với ba chỉ số blur / exposure (saturation) / noise; khi gắn cờ, hạ trọng số
   camera trong fusion và dựa nhiều hơn vào radar/LiDAR — đúng hướng paper cho biết fusion bền hơn [Nguồn]. Chi phí
   thấp: bốn chỉ số tính trong khoảng **16–20 ms/frame** trên 1 luồng CPU với ảnh 1280×720 [Đo, đo nhanh — ô cuối notebook; dao động giữa các lần chạy], không cần GPU;
   có thể giảm thêm bằng cách tính trên ảnh thu nhỏ.
2. **Không dùng một ngưỡng tuyệt đối cho blur score:** chuẩn hoá theo độ tương phản/độ sáng và đặt ngưỡng **theo
   từng camera** từ log chạy bình thường (độ lệch chuẩn blur score giữa các ảnh sạch đã là 226 trên trung bình 373).
3. **Data loop:** uncertainty thuần ưu tiên ảnh hỏng do sensor (top-k chiếm 35–54 % ở mức nặng nhất). Ảnh bị cổng
   gắn cờ nên vào một nhóm riêng *"sensor fault"* có hạn mức, không trộn với ảnh khó do nội dung, để ngân sách gán
   nhãn không bị tiêu vào ảnh mà chính con người cũng khó gán.

**Trade-off — khi nào nên / không nên:**
- *Nên* bật cổng B khi camera là nguồn chính cho tính năng an toàn (AEB, giữ làn) và có sensor dự phòng để chuyển sang.
- *Không nên* gắn cứng ngưỡng: báo nhầm trên ảnh sạch tăng 9,5 % → 14,5 %; trên xe chỉ có camera, mỗi lần báo nhầm là
  một lần hạ chức năng không cần thiết. Cổng hiện tại cũng gắn cờ 64,5 % ảnh tối ×0,5 dù mAP chỉ giảm 5 % — quá nhạy.
- *Chưa phủ:* ảnh đêm thật của BDD có `noise_est` thấp (0,21) vì ảnh đã qua nén JPEG mạnh — cổng bắt được ảnh đêm nhờ
  saturation chứ không nhờ nhiễu. Trên camera RAW thật, mức nhiễu ban đêm có thể khác hẳn [Giả thuyết].

**Vòng thử tiếp theo — đo gì:** (1) AUROC của từng cổng trong việc phát hiện frame mà detector *thật sự* hỏng
(recall theo ảnh giảm), thay vì chỉ tỉ lệ gắn cờ; (2) chạy trên dữ liệu thời tiết/đêm thật có nhãn (BDD100K rain/night
đầy đủ, hoặc ACDC) thay cho lỗi mô phỏng; (3) tổ hợp tối + nhiễu; (4) latency của cổng trên phần cứng nhúng.

## Hạn chế của benchmark

Lỗi chủ yếu là mô phỏng; 200 ảnh/điều kiện và một seed; ngưỡng cổng đặt trên chính 200 ảnh sạch (in-sample) nên tỉ
lệ báo nhầm là lạc quan; detector không huấn luyện lại trên BDD100K (mAP tuyệt đối thấp); chưa đo latency end-to-end.
