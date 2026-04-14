# CHƯƠNG 1: GIỚI THIỆU

---

## 1.1 Lý do chọn đề tài

### 1.1.1 Bối cảnh thực tiễn

Trong kỷ nguyên số hóa và trí tuệ nhân tạo (AI), nhận dạng và phân tích khuôn mặt đã trở thành một trong những lĩnh vực nghiên cứu trọng tâm của Thị giác Máy tính (Computer Vision). Khả năng tự động trích xuất thông tin từ ảnh khuôn mặt — bao gồm tuổi, giới tính, sắc tộc, biểu cảm — đang được ứng dụng rộng rãi trong nhiều ngành công nghiệp:

- **An ninh và giám sát**: Hệ thống nhận diện khuôn mặt tại sân bay, nhà ga, trung tâm thương mại giúp xác định đối tượng nghi vấn, phân tích hành vi đám đông, và hỗ trợ lực lượng an ninh ra quyết định nhanh chóng.
- **Marketing và quảng cáo**: Các hệ thống biển quảng cáo thông minh (Digital Signage) có khả năng nhận diện thuộc tính khách hàng (tuổi, giới tính) để hiển thị nội dung quảng cáo phù hợp, tăng hiệu quả tiếp thị mục tiêu.
- **Dịch vụ y tế và chăm sóc sức khỏe**: Ước lượng tuổi từ ảnh khuôn mặt hỗ trợ nghiên cứu lão hóa, đánh giá tình trạng sức khỏe da, và phát triển các ứng dụng làm đẹp dựa trên AI.
- **Thương mại điện tử**: Hệ thống gợi ý sản phẩm dựa trên nhân khẩu học khuôn mặt, ứng dụng thử đồ ảo (Virtual Try-On) dựa trên đặc điểm khuôn mặt.
- **Giải trí và mạng xã hội**: Các bộ lọc khuôn mặt (face filters), hiệu ứng biến đổi tuổi (age transformation), và các công cụ chỉnh sửa ảnh thông minh.

### 1.1.2 Thách thức kỹ thuật

Mặc dù có nhiều ứng dụng tiềm năng, bài toán dự đoán thuộc tính khuôn mặt vẫn đặt ra nhiều thách thức kỹ thuật quan trọng:

1. **Tính đa dạng của dữ liệu**: Khuôn mặt con người vô cùng đa dạng về hình dáng, biểu cảm, góc nhìn, điều kiện ánh sáng, và mức độ che khuất. Một mô hình cần có khả năng tổng quát hóa tốt trên nhiều điều kiện khác nhau.

2. **Bài toán đa nhiệm (Multi-Task)**: Dự đoán đồng thời nhiều thuộc tính (tuổi, giới tính, sắc tộc) đòi hỏi mô hình phải cân bằng giữa các nhiệm vụ có bản chất khác nhau — hồi quy (regression) cho tuổi và phân loại (classification) cho giới tính, sắc tộc.

3. **Mất cân bằng lớp (Class Imbalance)**: Trong các bộ dữ liệu thực tế, phân bố mẫu thường không đều giữa các nhóm sắc tộc, nhóm tuổi. Điều này dẫn đến hiện tượng mô hình nghiêng về lớp đa số (majority bias).

4. **Yêu cầu thời gian thực**: Ứng dụng camera trực tiếp đòi hỏi tốc độ xử lý cao (>15 FPS), trong khi vẫn duy trì độ chính xác chấp nhận được — một thách thức lớn về cân bằng giữa accuracy và latency.

5. **Sự lão hóa phi tuyến**: Quá trình lão hóa khuôn mặt diễn ra khác nhau giữa các sắc tộc, giới tính, và chịu ảnh hưởng bởi nhiều yếu tố ngoại sinh (môi trường, lối sống, di truyền), khiến bài toán dự đoán tuổi đặc biệt phức tạp ở nhóm tuổi trung niên và cao tuổi.

### 1.1.3 Động lực nghiên cứu

Xuất phát từ thực tiễn và thách thức trên, đồ án này đề xuất xây dựng hệ thống **FaceVision AI** — một hệ thống end-to-end cho bài toán dự đoán thuộc tính khuôn mặt, tích hợp các kỹ thuật tiên tiến nhất trong lĩnh vực Học sâu (Deep Learning) và Xử lý ảnh số (Digital Image Processing), với mục tiêu đạt độ chính xác cao và khả năng ứng dụng thực tế thông qua giao diện web hiện đại.

---

## 1.2 Mục tiêu đồ án

### 1.2.1 Mục tiêu tổng quát

Xây dựng một hệ thống hoàn chỉnh từ đầu đến cuối (end-to-end) cho bài toán dự đoán đồng thời ba thuộc tính khuôn mặt: **Tuổi** (Age — hồi quy), **Giới tính** (Gender — phân loại nhị phân) và **Sắc tộc** (Race — phân loại đa lớp) từ ảnh chân dung, sử dụng kiến trúc CNN đa nhiệm kết hợp các kỹ thuật huấn luyện và suy luận tiên tiến.

### 1.2.2 Mục tiêu cụ thể

| STT | Mục tiêu | Chỉ tiêu đo lường |
|-----|----------|-------------------|
| 1 | Xây dựng mô hình CNN đa nhiệm | ResNet50 + SE Attention + 3 prediction heads |
| 2 | Đạt độ chính xác cao trên dự đoán tuổi | Age MAE ≤ 5 năm |
| 3 | Đạt độ chính xác cao trên phân loại giới tính | Gender Accuracy ≥ 90% |
| 4 | Đạt độ chính xác cao trên phân loại sắc tộc | Race Accuracy ≥ 80% |
| 5 | Phát hiện khuôn mặt tự động | MTCNN + Haar Cascade fallback |
| 6 | Xử lý ảnh số cải thiện chất lượng đầu vào | Pipeline CLAHE + Gaussian Blur + Resize |
| 7 | Hỗ trợ suy luận thời gian thực | Camera live ≥ 15 FPS |
| 8 | Xây dựng giao diện web hiện đại | Streamlit 3 chế độ (ảnh, camera, biểu đồ) |
| 9 | Đánh giá toàn diện với nhiều chỉ số | Confusion Matrix, P/R/F1, Radar, Age Error by Group |
| 10 | Triển khai trên cả CPU và GPU | Tương thích CUDA 11.8+ (GPU) và CPU-only |

---

## 1.3 Phạm vi nghiên cứu

### 1.3.1 Phạm vi đề tài

- **Bộ dữ liệu**: UTKFace (Large Scale Face Dataset) gồm 23.704 ảnh khuôn mặt, với nhãn tuổi (0–116), giới tính (Male/Female), và sắc tộc (White/Black/Asian/Indian/Others — được gộp lại thành 4 lớp: White/Black/Asian/Others).
- **Kiến trúc mô hình**: ResNet50 (pretrained trên ImageNet) kết hợp SE Attention và 3 prediction heads chuyên biệt.
- **Bài toán**: Dự đoán đồng thời 3 thuộc tính (tuổi, giới tính, sắc tộc) từ ảnh đơn lẻ.
- **Nền tảng**: Ứng dụng web Streamlit chạy trên máy tính cá nhân (local), hỗ trợ cả Windows, macOS, Linux.

### 1.3.2 Giới hạn

- Hệ thống chỉ xử lý ảnh chân dung khuôn mặt (frontal face), chưa hỗ trợ tốt khuôn mặt nghiêng > 45° hoặc bị che khuất nhiều.
- Phân loại sắc tộc chỉ gồm 4 lớp, chưa phản ánh đầy đủ sự đa dạng sắc tộc trên toàn cầu.
- Dự đoán tuổi ở nhóm 56+ có sai số cao hơn do dữ liệu ít và biến thiên lớn trong quá trình lão hóa.
- Chưa triển khai trên nền tảng đám mây (cloud deployment) hoặc thiết bị di động (mobile).
- Chưa tích hợp bảo mật dữ liệu khuôn mặt hoặc cơ chế chống giả mạo (anti-spoofing).

---

## 1.4 Ý nghĩa khoa học và thực tiễn

### 1.4.1 Ý nghĩa khoa học

- **Đóng góp phương pháp luận**: Đồ án trình bày một cách tiếp cận toàn diện cho bài toán Multi-Task Learning trong Face Attribute Prediction, kết hợp nhiều kỹ thuật tiên tiến (Wing Loss, Focal Loss, Adaptive Uncertainty Weighting, Progressive Resizing, Mixup Augmentation) vào một pipeline thống nhất.
- **Nghiên cứu thực nghiệm**: Cung cấp kết quả đánh giá chi tiết trên từng nhóm tuổi, từng lớp sắc tộc, từng giới tính — giúp hiểu rõ điểm mạnh và yếu của mô hình, làm cơ sở cho các nghiên cứu tiếp theo.
- **So sánh phiên bản**: Ghi nhận quá trình cải tiến mô hình qua các phiên bản (v2 → v3+), minh chứng hiệu quả của từng kỹ thuật bổ sung.

### 1.4.2 Ý nghĩa thực tiễn

- **Ứng dụng trực tiếp**: Hệ thống có thể được sử dụng ngay trong các bài toán phân tích nhân khẩu học, giám sát, marketing thông minh.
- **Mã nguồn mở**: Toàn bộ mã nguồn được công bố theo giấy phép MIT, phục vụ cộng đồng nghiên cứu và phát triển.
- **Giao diện thân thiện**: Giao diện Streamlit 3 chế độ giúp người dùng không chuyên cũng có thể sử dụng hệ thống mà không cần kiến thức lập trình.
- **Tài liệu đầy đủ**: Cung cấp tài liệu chi tiết về cách cài đặt, huấn luyện lại mô hình, và mở rộng hệ thống.

---

## 1.5 Cấu trúc báo cáo

Báo cáo đồ án tốt nghiệp được tổ chức thành 6 chương chính:

| Chương | Tên chương | Nội dung tóm tắt |
|--------|-----------|-------------------|
| **1** | Giới thiệu | Lý do chọn đề tài, mục tiêu, phạm vi, ý nghĩa khoa học và thực tiễn |
| **2** | Cơ sở lý thuyết và Công nghệ sử dụng | Các khái niệm nền tảng: CNN, ResNet50, SE Attention, Multi-Task Learning, Wing Loss, Focal Loss, MTCNN, CLAHE, và công nghệ sử dụng (PyTorch, Streamlit, OpenCV) |
| **3** | Phân tích yêu cầu và Thiết kế hệ thống | Use Case, Activity Diagram, kiến trúc tổng thể 4 tầng, thiết kế dữ liệu, thiết kế mô hình CNN, thiết kế pipeline huấn luyện và suy luận, thiết kế giao diện |
| **4** | Thiết kế chi tiết và Cài đặt | Mô tả chi tiết từng module mã nguồn (7 module trong `src/`, 4 scripts, 1 ứng dụng web), giải thích logic từng hàm, từng lớp, từng giải thuật chính |
| **5** | Kết quả thực hiện và Đánh giá | Kết quả huấn luyện 50 epochs, đánh giá test set (Age MAE, Gender Acc, Race Acc), Confusion Matrix, Precision/Recall/F1, sai số theo nhóm tuổi, radar hiệu năng, ảnh chụp giao diện |
| **6** | Kết luận và Hướng phát triển | Tổng kết kết quả, đóng góp, hạn chế và đề xuất hướng phát triển tương lai |

Ngoài ra, báo cáo còn bao gồm **Tài liệu tham khảo** (7 bài báo khoa học + tài liệu kỹ thuật) và **Phụ lục** (hướng dẫn cài đặt, mã nguồn chính, cấu trúc dữ liệu, danh sách biểu đồ).
