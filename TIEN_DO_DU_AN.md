# Tiến độ dự án Chili Pepper Phenotyping

> Cập nhật lần cuối: 12/09/2026
> Mục đích: ghi lại nội dung chính, trạng thái hiện tại và các việc nên làm tiếp theo để thuận tiện bàn giao/phát triển.

## 1. Tổng quan

Dự án xử lý ảnh quả ớt theo hai giai đoạn chính:

1. Phát hiện từng quả ớt trong ảnh đầu vào.
2. Cắt từng quả, phân đoạn các bộ phận và trích xuất đặc trưng kiểu hình.

Luồng chạy chính hiện dùng YOLOv8 segmentation cho 5 lớp:

- `seed` (hạt)
- `outer` (phần vỏ ngoài)
- `placenta` (nhau thai/lõi)
- `endocarp` (khoang trong)
- `chili` (toàn bộ quả)

Dự án có hai điểm vào:

- Web UI: `python app.py`
- CLI: `python predictor.py --input_path <ảnh hoặc thư mục>`

## 2. Trạng thái nhanh

| Hạng mục | Trạng thái | Ghi chú |
| --- | --- | --- |
| Phát hiện quả bằng threshold | Đã có, đang là mặc định | Phụ thuộc mạnh vào bố cục ảnh; bỏ cố định 540 px phía trên |
| Phát hiện quả bằng YOLO | Có mã và weight | Nhánh tùy chọn chưa được xác nhận chạy đúng với API/định dạng bbox hiện tại |
| Phân đoạn 5 lớp | Đã có | Runtime chính dùng `yolov8_segment_v2.pt` |
| Hậu xử lý mask | Đã có | Làm sạch mask chili, xử lý endocarp, tách cuống khỏi placenta |
| Chuẩn hóa hướng quả | Đã có | Dựa trên vùng xanh của cuống, hỗ trợ xoay 90°/180° |
| Hiệu chỉnh kích thước bằng QR | Đã có | QR được xem là cạnh 10 mm; nếu không đọc được dùng trung bình dataset `0,094305839 mm/pixel` và gắn cờ fallback |
| Trích xuất đặc trưng | Đã có | Web và CLI đều xuất wrinkle cùng thickness min/trung bình/max |
| Giao diện kiểm tra pipeline | Đã có | Hiển thị ROI, mask, bbox, crop, segmentation và bảng kết quả |
| Docker | Đã có | Python 3.10, chạy Gradio ở cổng 7860 |
| Notebook nghiên cứu | Đã có | Tiền xử lý, dò tham số, đánh giá pipeline và hậu xử lý endocarp |
| Thickness/wrinkle mới | Đã tích hợp runtime | Dùng chung module hình học giữa `Chili`, Web và CLI |
| Kiểm thử tự động | Đã có | Có synthetic tests cho skeleton, ray-cast và chia contour; chưa có CI |

## 3. Luồng xử lý runtime hiện tại

```text
Ảnh đầu vào
  -> đọc QR để tính mm/pixel (không có QR: dùng trung bình dataset và cảnh báo)
  -> bỏ 540 px ở mép trên
  -> phát hiện contour quả bằng ngưỡng kênh đỏ
  -> lọc contour có diện tích >= 10.000 px²
  -> tạo bbox và crop có padding 50 px
  -> tìm vùng xanh của cuống để xoay quả về hướng chuẩn
  -> YOLOv8 phân đoạn seed/outer/placenta/endocarp
  -> class chili lấy từ threshold hoặc YOLO
  -> làm sạch, lấp lỗ và hợp nhất các mask
  -> tách stem khỏi placenta
  -> dựng wall/skeleton, tính thickness và wrinkle mới
  -> tính các đặc trưng hình học, màu sắc và hạt còn lại
  -> trả bảng kết quả/ảnh trực quan
```

Các ngưỡng quan trọng đang được hard-code:

- Vùng bỏ phía trên ảnh: `540 px`.
- Threshold detector: mode của histogram kênh đỏ + `22`.
- Diện tích contour tối thiểu khi detect: `10.000 px²`.
- Padding quanh bbox: `50 px`.
- Segmentation: `imgsz=640`, `conf=0.1`, `iou=0.7`.
- Riêng lớp seed: confidence tối thiểu `0.35`.
- Threshold mặc định để tạo class chili: offset `16`.

## 4. Các đặc trưng đang được tính

`src/object.py` hiện tính và lưu các nhóm chỉ số sau cho từng quả:

- Bbox: chiều rộng, chiều cao.
- Kích thước quả: chiều rộng min/max/trung bình; chiều dài theo spine; chiều rộng/chiều dài theo 9 đoạn.
- Diện tích: toàn quả, phần thịt, thân quả và placenta.
- Độ dày thành ớt: min/trung bình/max từ các lát cắt ray-cast cách nhau 1 mm trên longest backbone.
- Hạt: số lượng, tổng/min/max/trung bình/độ lệch chuẩn diện tích; dùng watershed để tách cụm.
- Màu: redness theo kênh `a*` của CIELAB và độ đồng đều màu.
- Hình thái: wrinkle, curvature index, asymmetry index.
- Tỷ lệ: flesh/placenta và seed density.
- Tỷ lệ quy đổi theo QR (`mm/pixel`) nếu đọc được.

Web UI trả bảng đặc trưng đầy đủ. CLI xuất thêm thickness min/trung bình/max,
wrinkle, tỷ lệ quy đổi và nguồn tỷ lệ.

## 5. Cấu trúc và vai trò các file

- `app.py`: Gradio UI, nạp model một lần, chạy pipeline và dựng toàn bộ ảnh kiểm tra trung gian.
- `predictor.py`: CLI chạy một ảnh hoặc cả thư mục, có tùy chọn ghi `runs/predict/output.csv`.
- `src/detect.py`: phát hiện từng quả bằng threshold hoặc YOLO.
- `src/segment.py`: wrapper YOLOv8 segmentation, lọc detection và chuyển mask thành polygon.
- `src/object.py`: lớp `Fruits`/`Chili`, crop, xoay, dựng mask và tính toàn bộ đặc trưng.
- `pipeline.ipynb`: thử nghiệm detection, segmentation, accuracy và app demo.
- `preprocess-data.ipynb`: chuẩn bị COCO/YOLO, cắt dữ liệu, bổ sung nhãn và thử SAHI.
- `get_parameter.ipynb`: thử nghiệm công thức/đặc trưng kiểu hình.
- `sub_process.ipynb`: nghiên cứu hậu xử lý endocarp theo dạng quả dài và quả chuông.
- `wrinkle_thickness_experiment.ipynb`: thử nghiệm phương pháp mới cho độ dày thành ớt và độ nhăn mép ngoài.
- `report.md`: mô tả kỹ thuật và công thức; một số nội dung chưa đồng bộ với runtime.
- `Dockerfile`: môi trường triển khai Gradio.
- `requirements.txt`: thư viện Python, hiện chưa khóa phiên bản.

## 6. Dữ liệu, model và kết quả có sẵn

- `data/`: 22 ảnh JPG và 20 file annotation JSON. Hai ảnh chưa có JSON tương ứng là `030_IT158894_1.jpg` và `035_IT163492_1.jpg`.
- `pretrain/`:
  - `yolo26n_detection_v1.pt` (~5,14 MB)
  - `yolov8_segment_v1.pt` (~52,26 MB)
  - `yolov8_segment_v2.pt` (~52,26 MB) — bản runtime đang dùng
  - `yolov8_segment_v3.pt` (~52,27 MB)
- `runs/`: có CSV và ảnh so sánh cho các thử nghiệm hậu xử lý endocarp, cùng `runs/predict/output.csv`.

Các thư mục `data/`, `pretrain/` và `runs/` đang nằm trong `.gitignore`; máy mới cần nhận các tài sản này bằng kênh riêng.

## 7. Thử nghiệm thickness và wrinkle mới

Notebook `wrinkle_thickness_experiment.ipynb` đã được tạo và chạy trên quả đầu tiên của `029_IT250232_2.jpg` bằng `yolov8_segment_v2.pt`.

### Mask thành ớt

Mask thành ớt được xác định bằng:

```text
wall = chili - union(endocarp, outer, placenta, seed)
```

Các phép đo hình học dùng tỷ lệ QR `0,094787 mm/pixel` trên ảnh thử nghiệm. Notebook dừng nếu không đọc được QR để không gắn nhãn pixel thành mm.

### Độ dày mới

- Baseline hiện tại dùng `2 × distanceTransform` tại các điểm skeleton, tức giả định skeleton luôn nằm chính giữa hai mép.
- Phương pháp mới sắp từng component skeleton thành polyline và nội suy điểm theo arc length mỗi `1 mm`, thay cho cách chia ô lưới không gian.
- Skeleton mới ưu tiên wall đã closing `1,4 mm` và opening `0,6 mm`, rồi chuyển skeleton thành graph 8-neighbor có trọng số theo khoảng cách pixel.
- Với mỗi connected component có nhánh, `longest_backbone_skeleton()` luôn chỉ giữ đường geodesic dài nhất giữa hai endpoint. Component thẳng/vòng không có nhánh được giữ nguyên.
- Đường hở giữ cả hai endpoint; các khoảng thường đúng 1 mm và chỉ khoảng cuối được phép ngắn hơn. Đường vòng lấy mẫu từ điểm trên-trái, không lặp điểm đầu.
- Baseline luôn dùng skeleton thô của wall gốc. Phương án cắt spur cũ đã được loại khỏi notebook.
- Tại mỗi điểm lấy mẫu, thuật toán tìm đoạn thẳng ngắn nhất đi qua skeleton và cắt một mép trong cùng một mép ngoài.
- Góc được quét bước `1°`, sau đó tinh chỉnh bước `0,1°`.
- Giao điểm với hai mép được nội suy dưới pixel; độ dài được đổi sang mm bằng QR.
- Các lát cắt gần cuống, điểm nhánh hoặc ngoại lệ cục bộ về độ dày/hướng được loại khỏi thống kê mới. Nếu số điểm sau lọc dưới 30% dữ liệu thô (tối thiểu 20 điểm), notebook dùng dữ liệu thô và ghi rõ `filter_mode=fallback_raw`.
- Điểm dùng để tính được vẽ cyan; vị trí ray-cast lỗi hoặc bị lọc vẫn được vẽ xám để kiểm tra trực quan tính đều của chuỗi lấy mẫu.

Kết quả thử nghiệm:

| Chỉ số | Giá trị |
| --- | ---: |
| Baseline thickness trung bình | 3,7455 mm |
| Baseline thickness lớn nhất | 11,5866 mm |
| Thickness mới trung bình | 6,7149 mm |
| Thickness mới trung vị | 6,1676 mm |
| Thickness mới nhỏ nhất/lớn nhất sau lọc | 3,6024 / 15,6486 mm |
| Điểm lấy mẫu yêu cầu | 422 |
| Điểm ray-cast hợp lệ | 368/422 |
| Điểm được dùng sau lọc | 304/368 |
| Điểm ray-cast lỗi / bị lọc | 54 / 64 |
| Coverage ray-cast | 87,20% |
| Sai số khoảng cách thường lớn nhất | `< 0,01 mm` |
| Tỷ lệ skeleton giữ lại | 93,24% |
| Chế độ skeleton / lọc | `smoothed_wall`, `longest_geodesic` / `clean` |

So với dữ liệu trước lọc (`mean=6,6719 mm`, `min=0,5437 mm`), bộ lọc đã loại các lát cắt cực ngắn bất thường. Backbone còn 3.527 pixel và là skeleton duy nhất dùng để tính thickness mới. Coverage chưa đạt 100% và giá trị lớn nhất vẫn cao, nên phương pháp vẫn cần ground truth cùng nhiều ảnh hơn dù đã được tích hợp runtime.

### Độ nhăn mới

- Chỉ tính trên các run thuộc mép ngoài của wall; không dùng toàn bộ contour chili như baseline.
- `WRINKLE_SEGMENT_THRESHOLD_MM` là tham số hình học độc lập và có thể thay đổi; giá trị thử nghiệm hiện tại là `5 mm`. Nó không liên quan threshold kênh đỏ dùng để detect/tạo mask chili.
- Hai đầu mỗi đoạn được nội suy chính xác sao cho chord đạt ngưỡng, thay vì chỉ chọn vertex có sẵn trên contour.
- Độ cong dư mỗi đoạn là `arc/chord - 1`; wrinkle là trung bình độ cong dư của các đoạn.
- Đoạn dư cuối được chia lại với đoạn trước để hai chord cuối gần đều nhau và không vượt threshold.
- Mép đi qua cuống bị loại trước khi chia đoạn. Vùng cấm được áp sau bước làm mượt nhãn để thuật toán không nối lại đường qua cuống.

Kết quả thử nghiệm sau khi loại vùng cuống:

| Chỉ số | Giá trị |
| --- | ---: |
| Baseline wrinkle | 0,066527 |
| Wrinkle mới | 0,131135 |
| Mean `arc/chord` | 1,131135 |
| Số đoạn contour | 74 |
| Số outer run | 6 |
| Tổng chiều dài contour được dùng | 401,592 mm |
| Chord lớn nhất | 5,000 mm |

Notebook có ảnh trực quan cho wall, mép trong/ngoài, vùng cuống bị loại, skeleton, lát cắt độ dày, heatmap và các đoạn wrinkle. Synthetic tests cho thành thẳng/nghiêng, contour thẳng/zigzag, giới hạn chord và invariant loại cuống đều đã qua. Lần chạy xác nhận gần nhất dùng ratio `0,094787 mm/pixel`, phát hiện 23 seed, 4 outer, 1 placenta và 1 endocarp.

Trạng thái: **đã thay thế `get_thickness()` và `get_wrinkle()` trong runtime.**
Các phép toán được đặt trong `src/geometric_features.py`; `Chili` cache wall data để
hai đặc trưng không lặp lại bước dựng mask/skeleton. Gradio có slider wrinkle
riêng mặc định `5 mm`; CLI có `--wrinkle-segment-threshold-mm`.

## 8. Vấn đề đã biết và điểm chưa đồng bộ

### Ưu tiên cao

1. **Tùy chọn class chili trên UI/CLI đang không có hiệu lực.** `Fruits` nhận `chili_method` và `chili_threshold`, nhưng `Fruits.get_item()` không truyền chúng vào `Chili`. Vì vậy lựa chọn của người dùng bị mất và `SegmentModel` dùng cấu hình mặc định.
2. **Nhánh YOLO detection cần sửa/xác nhận.** Mã đang đọc `results.bboxes` trực tiếp từ kết quả `model.predict()`, trong khi API Ultralytics thông thường trả danh sách `Results`. Bbox từ ảnh đã crop cũng cần được đổi đúng định dạng và cộng lại offset 540 px trước khi crop trên ảnh gốc.
3. **Detector threshold có thể lỗi khi ảnh không tạo ra contour.** `get_mask()` gọi `max(contour_areas)` mà chưa xử lý danh sách rỗng.

### Đồng bộ tài liệu và khả năng tái lập

4. README hướng dẫn weight segmentation `v1`, nhưng `app.py` và `predictor.py` mặc định dùng `v2`.
5. `report.md` mô tả HSV Saturation + Otsu, ngưỡng diện tích 4.000 px² và SAHI tại inference; runtime hiện dùng histogram kênh đỏ + offset, ngưỡng 10.000 px² và không dùng SAHI trong luồng chính.
6. `predictor.py --save_csv` không tự tạo thư mục `runs/predict/`.
7. Requirements chưa khóa phiên bản; `torch` được import trực tiếp nhưng không được khai báo trực tiếp trong `requirements.txt`.
8. Đã có synthetic unit tests cho hình học nhưng chưa có CI hoặc bộ metric ground truth.
9. Khi không có QR, các số đo dùng tỷ lệ trung bình dataset; đây là giá trị ước lượng và được đánh dấu `dataset_mean_fallback` trong Web/CSV.
10. Thickness/wrinkle mới hiện mới có regression tham chiếu trên một quả ớt chuông; chưa có ground truth vật lý hoặc đánh giá trên toàn bộ dạng quả.

## 9. Kiểm tra đã thực hiện cho bản ghi này

- Đã đọc các entry point, ba mô-đun trong `src/`, README, report, Dockerfile và requirements.
- Đã thống kê dữ liệu, pretrained weights, kết quả trong `runs/` và nội dung chính của bốn notebook.
- Đã parse AST toàn bộ file Python: `app.py`, `predictor.py`, `src/detect.py`, `src/segment.py`, `src/object.py` đều hợp lệ về cú pháp.
- Đã chạy end-to-end notebook thickness/wrinkle mới trên `029_IT250232_2.jpg`; synthetic tests và real-image checks đều đạt.
- Đã tích hợp hai đặc trưng vào runtime và thêm test tự động trong `tests/test_geometric_features.py`.
- Đã chạy regression runtime trên `029_IT250232_2.jpg` sau tích hợp: quả đầu dùng QR thật
  `0,09478673 mm/pixel`, thickness min/trung bình/max là `3,80 / 6,71 / 15,65 mm`
  và wrinkle là `0,12202` với threshold `5 mm`. Pipeline xử lý đủ hai quả, không lỗi.
- Đã chạy 11 synthetic/unit tests: tất cả đạt. Môi trường hiện cảnh báo bộ phiên bản
  SciPy/NumPy chưa tương thích hoàn toàn (`scipy` yêu cầu `NumPy < 1.25`, đang có `1.26.4`).
- Chưa đánh giá độ chính xác model hoặc hai đặc trưng mới trên toàn bộ dataset/ground truth.

## 10. Đề xuất công việc tiếp theo

Thứ tự ưu tiên đề xuất:

1. Đánh giá thickness/wrinkle mới trên nhiều quả dài và quả chuông; đối chiếu thickness với số đo vật lý thủ công.
2. Đo ground truth độ dày để tinh chỉnh bộ lọc lát cắt và đánh giá coverage.
3. Thêm smoke test end-to-end khi môi trường test có `ultralytics` và model weight.
4. Thêm test cho hai chế độ `chili_method`/`chili_threshold` đã được truyền từ `Fruits` xuống `Chili`.
5. Sửa, chuẩn hóa và kiểm thử nhánh YOLO detection (kiểu kết quả, định dạng bbox, offset ROI).
6. Bổ sung xử lý ảnh rỗng/không contour/ảnh thấp hơn 540 px cho cả web và CLI.
7. Đồng bộ README/report với weight `v2` và thuật toán runtime thực tế.
8. Đồng bộ schema đầu ra giữa CLI và web; tự tạo thư mục output trước khi ghi CSV.
9. Thêm test unit cho mask/feature và một smoke test end-to-end trên ảnh mẫu cố định.
10. Khóa phiên bản dependency và xây bộ đánh giá có ground truth cho các số đo vật lý.

## 11. Lệnh sử dụng nhanh

```bash
pip install -r requirements.txt
python app.py
```

CLI:

```bash
python predictor.py \
  --input_path data/001_IT135673_1.jpg \
  --weight_seg pretrain/yolov8_segment_v2.pt
```

Docker:

```bash
docker build -t chili-phenotyping-app .
docker run -p 7860:7860 chili-phenotyping-app
```
