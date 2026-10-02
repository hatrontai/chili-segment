Báo cáo này tổng hợp chi tiết toàn bộ các phương pháp kỹ thuật, quyết định thiết kế thuật toán, và logic Toán học chuyên sâu được áp dụng trong hệ thống trích xuất kiểu hình ớt (Chili Pepper Phenotyping Pipeline).

---

## 1. Image Preprocessing & Object Localization
*   **HSV Saturation Thresholding:** Hình ảnh được chuyển sang không gian màu HSV. Kênh bão hòa màu **(Saturation - S)** được trích xuất. Vì nền đen gần như không có độ bão hòa, trong khi trái ớt (bất kể xanh, đỏ, hay vàng) đều có độ bão hòa cao, kênh S tạo ra độ tương phản hoàn hảo. Ngưỡng phân tách (threshold) được tự động tính toán bằng phương pháp Otsu.
*   **Morphological Cleanup:** Mặt nạ (mask) sau đó được đóng (Morphological Closing) với kernel $10 \times 10$ và lọc diện tích ($>4,000 \text{ px}^2$) để loại bỏ vụn nhiễu trước khi cắt (crop) thành các ảnh cục bộ.

## 2. Instance Segmentation & Boundary Correction
*   **Deep Learning & SAHI:** Khung mô hình YOLO phân đoạn 5 nhãn (`Chili`, `Outer`, `Endocarp`, `Placenta`, `Seed`). Tại khâu suy luận, kỹ thuật cắt lát SAHI (Sliced Aided Hyper Inference) được tích hợp nhằm phát hiện chính xác các quả ớt lai giống siêu nhỏ ở độ phân giải gốc mà không làm mất chi tiết.
*   **Post-segmentation Stem Separation:** Cuống ớt (Stem) thường bị gộp chung vào lớp Placenta. Hệ thống dùng bộ lọc không gian màu HSV (chọn dải Hue xanh lục) để tự động bóc tách phần cuống này ra khỏi Nhau thai, thu được mặt nạ thuần khiết ($M_{placenta} = M_{placenta\_raw} \setminus M_{stem}$) nhằm tránh làm lạm phát diện tích khoang rỗng.
*   **Outer Boundary Elimination:** Lớp `Outer` (phần vỏ lồi ra do cấu trúc 3D của quả) được thuật toán nhận diện và loại trừ, đảm bảo diện tích mặt cắt phẳng được tính toán chính xác tuyệt đối mà không bị cộng dồn phần vỏ ở nền.

---

## 3. Advanced Phenotypic Trait Extraction Logic
Hệ thống không sử dụng các phương pháp đo lườngBounding Box xấp xỉ truyền thống. Dưới đây là giải thích chi tiết về cơ sở Toán học đằng sau quá trình trích xuất từng chỉ số:

*   **Fruit Length — FL (Chiều dài quả):**
    Để đo chính xác chiều dài dọc theo độ cong tự nhiên của quả ớt, thuật toán tạo một bộ xương hình thái học (morphological skeleton) từ mặt nạ quả ớt. Các tọa độ trung tâm được chiết xuất và khớp (curve-fitting) với một phương trình đa thức bậc 3 ($x = ay^3 + by^2 + cy + d$). Chiều dài thực được tính chính xác bằng tích phân vi phân chiều dài cung (Arc Length Integral):
    $$FL = \int_{y_{min}}^{y_{max}} \sqrt{1 + \left(\frac{dx}{dy}\right)^2} dy$$

*   **Fruit Width — FW (Chiều rộng quả):**
    Thay vì lấy chiều rộng Bounding Box, thuật toán cắt bỏ 5% diện tích nhiễu ở phần chóp và cuống. Sau đó, nó quét lặp qua từng hàng ngang (row) của mặt nạ, tính khoảng cách tuyệt đối giữa tọa độ x nhỏ nhất và lớn nhất để trả về giá trị chiều rộng Cực đại, Cực tiểu và Trung bình.

*   **Mesocarp Thickness — MT (Độ dày trung quả bì):**
    Áp dụng phép biến đổi khoảng cách Euclidean Distance Transform (EDT) lên mặt nạ thịt quả tinh khiết (Flesh mask = Tổng quả trừ đi khoang rỗng, vỏ lồi, cuống). Thuật toán EDT gán cho mỗi pixel bên trong cùi thịt một giá trị bằng đúng khoảng cách ngắn nhất từ nó tới biên gần nhất. Độ dày tối đa và trung bình chính bằng đường kính của các vòng tròn nội tiếp này ($Thickness = 2 \times \text{EDT}$).

*   **Mesocarp Area Fraction — MAF (Tỷ lệ diện tích thịt quả):**
    Tính bằng $MAF=MA/FBA$, trong đó $MA$ là diện tích mesocarp và $FBA$ là diện tích thân quả sau khi loại outer và cuống. Đây là tỷ lệ mặt cắt 2D được chiếm bởi thịt quả, dùng để mô tả cấu trúc tương đối giữa các mẫu. Chỉ số nằm trong [0, 1]; nếu thân quả rỗng thì trả về `None`. Chưa có kiểm chứng trực tiếp về dự đoán độ cay, năng suất hay khối lượng thịt.

*   **Seed-to-Cavity Area Ratio — SCR (Tỷ lệ diện tích hạt/khoang):**
    Định lượng tỷ lệ không gian bị hạt chiếm dụng bên trong khoang rỗng trung tâm. Để khắc phục sai số vẽ viền của AI, mặt nạ khoang rỗng (mẫu số) được thuật toán cẩn thận tái tạo lại bằng phép hợp (Union): $M_{endocarp} \cup M_{placenta} \cup M_{seed}$.

*   **Adaptive Marker-based Watershed (Đếm hạt dính chùm):**
    Giải quyết triệt để lỗi AI gộp chung các cụm hạt (cluster) dày đặc. Thuật toán tự động định cỡ Watershed bằng cách:
    1. Tính số hạt kỳ vọng ($E_i$) cho từng cụm: Lấy diện tích cụm chia cho Trung vị ($\tilde{A}_{seed}$) của một hạt độc lập ($E_i = \text{round}(A_{cluster} / \tilde{A}_{seed})$).
    2. Chạy biến đổi EDT trên cụm để tìm các đỉnh cục bộ (peaks).
    3. Thiết lập vòng lặp thu hẹp khoảng cách tối thiểu giữa các đỉnh (min\_distance) liên tục cho đến khi thuật toán Watershed tìm ra ĐÚNG số hạt kỳ vọng $E_i$.

*   **Fruit Curvature Index — FCI (Chỉ số độ cong quả):**
    Đo lường mức độ uốn cong của quả ớt bằng cách so sánh chiều dài đường thẳng Euclidean nối 2 điểm chóp/cuống (Straight Line) với chiều dài cung vi phân thực tế (Arc Length Integral). Chỉ số tiến về 1 đối với ớt thẳng và giảm dần với ớt cong gắt.

*   **Fruit Asymmetry Index — FAI (Chỉ số bất đối xứng quả):**
    Đánh giá độ phình lệch của quả ớt bằng cách chiếu các đường thẳng vuông góc với trục xương sống (spine) ở mọi điểm. Thuật toán tính tích phân tổng độ chênh lệch tuyệt đối về bề rộng giữa nửa trái ($W_L$) và nửa phải ($W_R$) chia cho tổng bề rộng.

*   **Fruit Ripeness & Uniformity (Độ chín & Đồng đều màu):**
    Đánh giá quang phổ trên mặt nạ thịt ớt tinh khiết. Hình ảnh RGB được chuyển sang hệ màu CIELAB. Thuật toán phân tích kênh $a^*$ (trục Xanh lục - Đỏ). Trị số Trung bình (Mean) của kênh $a^*$ phản ánh chuẩn xác giai đoạn chín sinh lý của quả ớt, trong khi Phương sai (Variance) của các pixel đo lường mức độ chín đồng đều (Color Uniformity).
