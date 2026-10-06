# Báo cáo cá nhân — Sensor Reality Sprint, chủ đề T1

**Họ tên:** Vũ Thương Tín · **MSSV:** 2A202602955 · **Nhóm:** GTAA

**Phần tôi phụ trách:** mở rộng benchmark sang lỗi **chói / thừa sáng (overexpose)** và phân tích giới hạn của cổng sức khỏe camera.

Bằng chứng dùng trong báo cáo:

- Benchmark chung của nhóm: `benchmark.ipynb`, `results/main/` (lệnh `python src/benchmark.py --n 200`)
- Phần mở rộng của tôi: `results/glare/` — `summary.csv`, `plot_overexpose.png`, `examples.jpg`, `run_info.json`
  (lệnh `python src/benchmark.py --n 200 --families overexpose --no-night --tag glare`, seed 0)

Ký hiệu: **[Đo]** kết quả nhóm tự chạy · **[Nguồn]** paper/repo nói · **[Giả thuyết]** suy luận chưa kiểm chứng.

## 1. Bài toán

Camera trước của xe ADAS dùng cho phát hiện xe, người, đèn giao thông. Khi xe hướng thẳng về phía mặt trời, hoặc
ra khỏi hầm, ảnh bị cháy sáng: nhiều vùng thành trắng xoá (pixel = 255) và mất chi tiết. Camera không báo lỗi gì, nó
vẫn gửi frame bình thường. Câu hỏi của tôi: **mức chói nào bắt đầu làm detector kém đi, và chỉ số sức khỏe ảnh nào
nhận ra được lỗi này?**

## 2. Phương pháp và nguồn tham khảo

**[Nguồn]** Tôi tham khảo Dong và cộng sự, *Benchmarking Robustness of 3D Object Detection to Common Corruptions in
Autonomous Driving*, CVPR 2023 ([bài báo](https://arxiv.org/html/2303.11040v1),
[mã nguồn](https://github.com/thu-ml/3D_Corruptions_AD) @ `48c23f7`). Công trình thử 27 loại lỗi, mỗi loại 5 mức,
trên KITTI-C / nuScenes-C / Waymo-C với 24 detector 3D. Paper dùng
**RCE theo từng điều kiện = (AP_sạch − AP_lỗi) / AP_sạch** và nhận thấy các mô hình chỉ dùng camera dễ suy giảm trước
lỗi ảnh, còn mô hình kết hợp LiDAR–camera nhìn chung bền hơn.

**[Đo]** Benchmark chính của nhóm dùng lại mức zoom blur và nhiễu cùng cách tính RCE; detector là YOLOv8n với trọng số
COCO, đánh giá trên 200 ảnh BDD100K ban ngày, trời quang (seed 0). Phần tôi thêm là **tăng sáng toàn ảnh**: nhân từng
giá trị pixel với 1,5 / 2,5 / 4 rồi cắt ở 255. Cách này khác lỗi *Strong Sunlight* của paper, vốn mô phỏng mặt trời
cục bộ trong ảnh. Tôi giữ nguyên 200 ảnh, nhãn, detector và phép đo của benchmark chính để so sánh các mức lỗi trong
lab; kết quả AP của paper không thể so trực tiếp vì khác dữ liệu và bài toán phát hiện 3D.

`blur_score` là trung vị variance của Laplacian trên ảnh xám; `saturation_ratio` là tỉ lệ pixel xám **≤ 5 hoặc ≥ 250**
trung bình theo ảnh. `uncertainty` là trung bình của `1 − max confidence` (ảnh không có box nhận giá trị 0);
`mAP50` tính trên toàn bộ 200 ảnh có nhãn.

## 3. Mở rộng: chói sáng [Đo]

| Điều kiện | blur_score | saturation_ratio | mAP50 | uncertainty | entropy (bit) | RCE |
|---|---|---|---|---|---|---|
| `clean` | 327,0917 | 0,0094 | 0,2953 | 0,2076 | 7,4869 | 0,0000 |
| `overexpose_gain1.5` | 596,5371 | 0,1416 | 0,3020 | 0,2140 | 7,2806 | −0,0228 |
| `overexpose_gain2.5` | 716,8585 | 0,4276 | 0,2780 | 0,2379 | 5,7297 | 0,0585 |
| `overexpose_gain4` | 554,2057 | **0,6348** | **0,1962** | 0,3228 | 4,1702 | **0,3355** |

Nguồn số: `results/glare/summary.csv`; có thể xem xu hướng trong `plot_overexpose.png` và ảnh minh họa trong
`examples.jpg`. RCE ở đây áp công thức của paper cho **mAP50 của lab**, lấy `clean` làm mốc.

Trả lời hai câu hỏi trong phần việc:

- **Chỉ số nào báo chói rõ nhất?** `saturation_ratio`: tăng đều từ 0,0094 lên 0,1416 / 0,4276 / 0,6348. Đây là
  tỉ lệ pixel xám gần **hai đầu** dải sáng, nên 0,6348 không đồng nghĩa chính xác với 63,48% pixel trắng bị cắt.
  `blur_score` tăng từ 327 lên 597 rồi 717 và chỉ giảm về 554 ở mức ×4; do phụ thuộc độ tương phản và các vùng bị
  cắt sáng, nó không phản ánh mức chói theo một chiều nhất quán.
- **Chói hay tối làm mAP50 giảm nhiều hơn ở mức nặng nhất?** Tối ×0,1 giảm **nhỉnh hơn**: mAP50 là 0,1916
  (RCE 0,3510) so với chói ×4 là 0,1962 (RCE 0,3355), đối chiếu `results/main/summary.csv`. Chênh lệch mAP50
  chỉ 0,0046 trên mẫu này; chưa có khoảng tin cậy để kết luận khác biệt có ý nghĩa thống kê.

## 4. Failure case: chỉ số sức khỏe báo động thừa

Tôi áp lại **cổng A** của nhóm lên từng ảnh trong `results/glare/per_image.csv`: gắn cờ nếu `blur_score` dưới phân vị
5% của 200 ảnh sạch (95,2042) **hoặc** `saturation_ratio` trên phân vị 95% (0,0390). Các ngưỡng và cách đặt cổng
được ghi ở mục 8 của `benchmark.ipynb`.

| | clean | chói ×1,5 | chói ×2,5 | chói ×4 |
|---|---|---|---|---|
| Cổng A gắn cờ | 19/200 (9,5%) | **193/200 (96,5%)** | 200/200 | 200/200 |
| mAP50 của điều kiện | 0,2953 | **0,3020** | 0,2780 | 0,1962 |

**Quan sát [Đo]:** ở chói ×1,5, cổng gắn cờ 96,5% ảnh dù mAP50 của điều kiện đo được là 0,3020, cao hơn
baseline 0,2953. Điều kiện saturation tự nó gắn cờ đúng 193/200 ảnh; điều kiện blur thấp gắn cờ 3/200 ảnh và
đều nằm trong 193 ảnh ấy. Đây là dấu hiệu cổng quá nhạy **ở cấp điều kiện thử nghiệm**; mAP tổng hợp không cho biết
từng ảnh bị gắn cờ có phát hiện đúng hay sai.

**Vì sao [Giả thuyết]:** ngưỡng saturation 0,0390 đặt theo phân vị của ảnh sạch có thể quá chặt với ảnh tăng sáng.
Vùng trời sáng có thể khiến chỉ số toàn ảnh vượt ngưỡng, trong khi vùng chứa xe hoặc người vẫn còn đủ chi tiết.
Giả thuyết về vị trí vật thể và nguyên nhân gắn cờ cần được kiểm tra bằng metric theo vùng và lỗi phát hiện từng ảnh.

**Hệ quả [Giả thuyết]:** nếu dùng cổng này để hạ trọng số camera, hệ thống có nguy cơ hạ chức năng trong nhiều
khung hình tăng sáng nhẹ dù mAP50 gộp vẫn giữ được trên tập thử. Chưa thể suy rộng sang mọi cảnh nắng gắt ngoài đời.

## 5. Trade-off và đề xuất

- **Đề xuất:** tính thêm tỉ lệ pixel gần trắng riêng trong vùng đường và vùng chứa vật thể, rồi đánh giá khả năng
  dự báo lỗi phát hiện theo từng ảnh. Chọn ngưỡng trên tập hiệu chỉnh riêng theo mức suy giảm phát hiện chấp nhận
  được; bốn giá trị trung bình của bảng trên chưa đủ để chốt một ngưỡng vận hành.
- **Trade-off:** ngưỡng thấp có thể gắn cờ nhiều ảnh còn dùng được; ngưỡng cao có thể bỏ qua ảnh đã mất chi tiết vật
  thể. Cần đo cả hai loại sai số trước khi dùng cờ để điều chỉnh trọng số camera trong hệ đa cảm biến.
- **Vòng thử tiếp theo:** thử ảnh chói *thật* khi ngược nắng hoặc ra hầm, so tỉ lệ gần trắng theo vùng với lỗi phát
  hiện từng ảnh và mAP của toàn tập, rồi kiểm tra cổng trên tập ảnh chưa dùng để đặt ngưỡng.

## Hạn chế

Chói được mô phỏng bằng nhân độ sáng toàn ảnh, chưa có lóa ống kính hay mặt trời cục bộ; chỉ 200 ảnh và một seed;
ngưỡng cổng đặt trên chính ảnh sạch của benchmark; detector không huấn luyện lại trên BDD100K. Báo cáo chưa có độ
bất định thống kê của mAP hoặc đánh giá trực tiếp chất lượng cổng theo từng ảnh.
