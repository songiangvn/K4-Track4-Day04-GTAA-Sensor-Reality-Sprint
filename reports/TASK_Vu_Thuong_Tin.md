# Phần việc của Tín (khoảng 30 phút)

Benchmark chính (motion blur, thiếu sáng, nhiễu, đêm thật) đã chạy xong ở `results/main/`.
Phần của Tín là **mở rộng thêm một loại lỗi camera: chói / thừa sáng (overexpose)** rồi viết
bản báo cáo riêng để nộp VLearn.

## 1. Điền MSSV
Mở `TEAMMATES.md`, thay `_(điền sau)_` bằng MSSV của bạn.

## 2. Chạy phần mở rộng chói sáng (một lệnh)
Cài môi trường và tải dữ liệu theo mục "Cài đặt" trong `README.md`, rồi:
```bash
python src/benchmark.py --n 200 --families overexpose --no-night --tag glare
```
Lệnh này nhân giá trị pixel với 1,5 / 2,5 / 4 rồi cắt ở 255 (mô phỏng nắng chiếu thẳng vào
camera), dùng **cùng 200 ảnh và cùng cách đo** với benchmark chính. Kết quả ghi vào
`results/glare/`: `summary.csv`, `plot_overexpose.png`, `examples.jpg`, `run_info.json`.

## 3. Điền bảng kết quả (copy số từ `results/glare/summary.csv`)
| Điều kiện | blur_score | saturation_ratio | mAP50 | uncertainty |
|---|---|---|---|---|
| clean | | | | |
| overexpose_gain1.5 | | | | |
| overexpose_gain2.5 | | | | |
| overexpose_gain4 | | | | |

Trả lời ngắn 2 câu:
- Metric sức khỏe nào báo lỗi chói rõ nhất: blur score hay saturation ratio? Vì sao?
- So với `dark` trong `results/main/summary.csv`, chói hay tối làm mAP50 giảm nhiều hơn ở mức nặng nhất?

## 4. Viết báo cáo riêng
Copy `reports/report_Nguyen_Son_Giang.md` thành `reports/report_Vu_Thuong_Tin.md`, đổi tên
người viết, **viết lại bằng lời của mình** và thêm mục "Mở rộng: chói sáng" với bảng ở bước 3.
Nhớ tách rõ: câu nào nhóm tự đo, câu nào paper nói.
