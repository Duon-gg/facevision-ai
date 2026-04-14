# BÁO CÁO ĐỒ ÁN TỐT NGHIỆP

---

<p align="center">
  <strong>BỘ GIÁO DỤC VÀ ĐÀO TẠO</strong><br/>
  <strong>TRƯỜNG ĐẠI HỌC [TÊN TRƯỜNG]</strong><br/>
  <strong>KHOA CÔNG NGHỆ THÔNG TIN</strong><br/>
  <br/><br/>
  <strong>──────── ❖ ────────</strong>
</p>

<br/>

<p align="center">
  <strong style="font-size: 1.5em;">ĐỒ ÁN TỐT NGHIỆP</strong><br/><br/>
  <strong style="font-size: 1.8em;">HỆ THỐNG DỰ ĐOÁN THUỘC TÍNH KHUÔN MẶT<br/>SỬ DỤNG HỌC SÂU ĐA NHIỆM</strong><br/><br/>
  <em>(FaceVision AI — Multi-Task Face Attribute Prediction: Age, Gender, Race)</em>
</p>

<br/><br/>

<p align="center">
  <strong>Ngành:</strong> Công nghệ Thông tin / Thị giác Máy tính<br/>
  <strong>Chuyên ngành:</strong> Trí tuệ Nhân tạo — Học Máy<br/>
</p>

<br/>

<p align="center">
  <strong>Sinh viên thực hiện:</strong> Nguyễn Văn Tùng Dương<br/>
  <strong>Mã số sinh viên:</strong> [MSSV]<br/>
  <strong>Lớp:</strong> [Tên lớp]<br/>
  <strong>Giảng viên hướng dẫn:</strong> [Tên GVHD]<br/>
</p>

<br/><br/>

<p align="center">
  <strong>[Tên thành phố] — 2025</strong>
</p>

---

# LỜI CAM ĐOAN

Tôi xin cam đoan đây là công trình nghiên cứu của riêng tôi dưới sự hướng dẫn của [Tên GVHD]. Các số liệu, kết quả trình bày trong đồ án là trung thực và chưa từng được công bố trong bất kỳ công trình nghiên cứu nào trước đây.

Tôi xin cam đoan rằng mọi sự giúp đỡ cho việc thực hiện đồ án này đã được cảm ơn và các thông tin trích dẫn trong đồ án đã được ghi rõ nguồn gốc.

Toàn bộ mã nguồn, dữ liệu thí nghiệm, mô hình huấn luyện và kết quả đánh giá trong đồ án này đều do chính tôi thiết kế, xây dựng và thực hiện trong suốt quá trình làm đồ án tốt nghiệp.

<p align="right"><em>[Tên thành phố], ngày …… tháng …… năm 2025</em></p>
<p align="right"><strong>Sinh viên thực hiện</strong></p>
<p align="right"><em>Nguyễn Văn Tùng Dương</em></p>

---

# LỜI CẢM ƠN

Lời đầu tiên, tôi xin bày tỏ lòng biết ơn sâu sắc đến **[Tên GVHD]** — người đã trực tiếp hướng dẫn, định hướng nghiên cứu và tận tình giúp đỡ tôi trong suốt quá trình thực hiện đồ án tốt nghiệp này. Sự chỉ dẫn của thầy/cô không chỉ giúp tôi hoàn thành tốt đề tài mà còn giúp tôi hiểu sâu hơn về lĩnh vực Thị giác Máy tính và Học sâu.

Tôi xin chân thành cảm ơn quý thầy cô trong **Khoa Công nghệ Thông tin** và **Bộ môn Trí tuệ Nhân tạo** của trường đã truyền đạt cho tôi những kiến thức quý báu trong suốt thời gian học tập tại trường, làm nền tảng vững chắc để tôi có thể thực hiện đề tài này.

Tôi cũng xin gửi lời cảm ơn đến:

- **Nhóm nghiên cứu UTKFace** (University of Tennessee, Knoxville) vì đã công bố bộ dữ liệu UTKFace phục vụ nghiên cứu học thuật.
- **Cộng đồng mã nguồn mở PyTorch**, đặc biệt các tác giả của thư viện `facenet-pytorch`, `torchvision`, `streamlit` và `plotly` — những công cụ trụ cột trong quá trình phát triển hệ thống.
- **Google Colab** vì đã cung cấp tài nguyên GPU miễn phí (Tesla T4/A100) hỗ trợ quá trình huấn luyện mô hình.

Cuối cùng, tôi xin gửi lời tri ân sâu sắc đến gia đình và bạn bè đã luôn động viên, hỗ trợ tôi trong suốt quá trình học tập và thực hiện đồ án.

Mặc dù đã cố gắng hết sức, đồ án không tránh khỏi những thiếu sót. Tôi rất mong nhận được sự góp ý chân thành từ quý thầy cô và bạn đọc để đồ án được hoàn thiện hơn.

<p align="right"><em>Xin chân thành cảm ơn!</em></p>

---

# TÓM TẮT (ABSTRACT)

## Tiếng Việt

Đồ án "Hệ thống dự đoán thuộc tính khuôn mặt sử dụng Học sâu Đa nhiệm" (FaceVision AI) trình bày việc nghiên cứu, thiết kế và xây dựng một hệ thống end-to-end cho bài toán dự đoán đồng thời ba thuộc tính khuôn mặt: **Tuổi** (Age), **Giới tính** (Gender) và **Sắc tộc** (Race) từ ảnh chân dung.

Hệ thống sử dụng kiến trúc **Multi-Task Convolutional Neural Network (CNN)** với backbone **ResNet50** được tiền huấn luyện trên ImageNet, kết hợp với cơ chế chú ý kênh **Squeeze-and-Excitation (SE) Attention** để tăng cường khả năng trích xuất đặc trưng. Mô hình được huấn luyện trên bộ dữ liệu **UTKFace** gồm 23.704 ảnh khuôn mặt, sử dụng hàm mất mát đa nhiệm kết hợp **Wing Loss** (cho hồi quy tuổi), **Focal Loss** (cho phân loại sắc tộc với lớp thiểu số) và **Adaptive Uncertainty Weighting** (tự động học trọng số nhiệm vụ theo phương pháp Kendall et al., 2018).

Pipeline huấn luyện áp dụng nhiều kỹ thuật nâng cao: **Freeze-then-Unfreeze** backbone với Differential Learning Rate, **Gradient Accumulation** (effective batch size = 128), **Mixup Augmentation** (α = 0.2), **Progressive Resizing** (160 → 192 → 224 pixel), **WeightedRandomSampler** (cân bằng Race + Age group) và **Early Stopping** (patience = 10). Pipeline suy luận (inference) tích hợp phát hiện khuôn mặt hai tầng (**MTCNN** làm chính, **Haar Cascade** làm dự phòng), tiền xử lý ảnh số (CLAHE histogram equalization trên không gian màu LAB) và **Test-Time Augmentation (TTA)** với 5 phép biến đổi để tăng độ chính xác.

**Kết quả đánh giá trên tập test** (3.556 mẫu): Age MAE đạt **4,49 năm**, Gender Accuracy đạt **93,53%**, Race Accuracy đạt **85,69%**. Giao diện web Streamlit cung cấp ba chế độ: dự đoán ảnh tĩnh, camera trực tiếp (15–25 FPS) và biểu đồ đánh giá mô hình tương tác.

**Từ khóa:** Dự đoán thuộc tính khuôn mặt, Học sâu đa nhiệm, ResNet50, SE Attention, Wing Loss, Focal Loss, MTCNN, CLAHE, Test-Time Augmentation, Streamlit.

## English (Abstract)

This thesis presents the research, design, and implementation of **FaceVision AI** — an end-to-end system for simultaneously predicting three face attributes: **Age**, **Gender**, and **Race** from portrait images.

The system employs a **Multi-Task CNN** architecture with **ResNet50** backbone (pretrained on ImageNet), augmented with **Squeeze-and-Excitation (SE) Attention** for channel-wise feature recalibration. The model is trained on the **UTKFace** dataset (23,704 face images) using a combined loss function of **Wing Loss** (age regression), **Focal Loss** (race classification with class imbalance), and **Adaptive Uncertainty Weighting** (Kendall et al., 2018).

Advanced training techniques include: Freeze-then-Unfreeze backbone with Differential LR, Gradient Accumulation (effective batch = 128), Mixup Augmentation (α = 0.2), Progressive Resizing (160 → 192 → 224), WeightedRandomSampler (Race + Age group balancing), and Early Stopping (patience = 10). The inference pipeline integrates dual-level face detection (**MTCNN** primary + **Haar Cascade** fallback), Digital Image Processing (CLAHE on LAB color space), and **Test-Time Augmentation (TTA)** with 5 augmentations.

**Test set results** (3,556 samples): Age MAE = **4.49 years**, Gender Accuracy = **93.53%**, Race Accuracy = **85.69%**. The Streamlit web interface provides three modes: static image prediction, live camera (~15–25 FPS), and interactive model evaluation charts.

**Keywords:** Face Attribute Prediction, Multi-Task Learning, ResNet50, SE Attention, Wing Loss, Focal Loss, MTCNN, CLAHE, Test-Time Augmentation, Streamlit.

---

# DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Thuật ngữ đầy đủ | Giải thích tóm tắt |
|---|---|---|
| **AI** | Artificial Intelligence | Trí tuệ nhân tạo |
| **AMP** | Automatic Mixed Precision | Huấn luyện pha trộn độ chính xác tự động |
| **API** | Application Programming Interface | Giao diện lập trình ứng dụng |
| **BN** | Batch Normalization | Chuẩn hóa theo batch |
| **BGR** | Blue Green Red | Không gian màu BGR (OpenCV) |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization | Cân bằng lược đồ xám thích nghi có giới hạn tương phản |
| **CNN** | Convolutional Neural Network | Mạng nơ-ron tích chập |
| **CSV** | Comma-Separated Values | Tệp giá trị phân tách bằng dấu phẩy |
| **CUDA** | Compute Unified Device Architecture | Kiến trúc tính toán song song của NVIDIA |
| **DIP** | Digital Image Processing | Xử lý ảnh số |
| **DL** | Deep Learning | Học sâu |
| **DLR** | Differential Learning Rate | Tốc độ học phân biệt (backbone vs. heads) |
| **ERD** | Entity Relationship Diagram | Sơ đồ thực thể – quan hệ |
| **FC** | Fully Connected | Lớp kết nối đầy đủ |
| **FL** | Focal Loss | Hàm mất mát tiêu điểm |
| **FPS** | Frames Per Second | Số khung hình trên giây |
| **GPU** | Graphics Processing Unit | Bộ xử lý đồ họa |
| **GUI** | Graphical User Interface | Giao diện người dùng đồ họa |
| **HE** | Histogram Equalization | Cân bằng lược đồ xám |
| **HTML** | HyperText Markup Language | Ngôn ngữ đánh dấu siêu văn bản |
| **JSON** | JavaScript Object Notation | Định dạng trao đổi dữ liệu JSON |
| **LAB** | Lightness A B color space | Không gian màu LAB |
| **LR** | Learning Rate | Tốc độ học (hệ số học) |
| **MAE** | Mean Absolute Error | Sai số tuyệt đối trung bình |
| **ML** | Machine Learning | Học máy |
| **MTCNN** | Multi-task Cascaded Convolutional Networks | Mạng CNN phân tầng đa nhiệm (phát hiện khuôn mặt) |
| **MTL** | Multi-Task Learning | Học đa nhiệm |
| **NaN** | Not a Number | Giá trị không phải số |
| **O-Net** | Output Network | Mạng đầu ra (giai đoạn 3 MTCNN) |
| **P-Net** | Proposal Network | Mạng đề xuất (giai đoạn 1 MTCNN) |
| **PIL** | Python Imaging Library | Thư viện xử lý ảnh Python (Pillow) |
| **R-Net** | Refine Network | Mạng tinh chỉnh (giai đoạn 2 MTCNN) |
| **ReLU** | Rectified Linear Unit | Hàm kích hoạt tuyến tính chỉnh lưu |
| **RGB** | Red Green Blue | Không gian màu RGB |
| **SE** | Squeeze-and-Excitation | Nén và kích hoạt (cơ chế chú ý kênh) |
| **TTA** | Test-Time Augmentation | Tăng cường dữ liệu lúc suy luận |
| **UI** | User Interface | Giao diện người dùng |
| **UML** | Unified Modeling Language | Ngôn ngữ mô hình hóa thống nhất |
| **UX** | User Experience | Trải nghiệm người dùng |

---

# MỤC LỤC

- **LỜI CAM ĐOAN**
- **LỜI CẢM ƠN**
- **TÓM TẮT (ABSTRACT)**
- **DANH MỤC TỪ VIẾT TẮT**
- **DANH MỤC HÌNH ẢNH**
- **DANH MỤC BẢNG BIỂU**
- **MỤC LỤC**

---

### **Chương 1: GIỚI THIỆU**
- 1.1 Lý do chọn đề tài
- 1.2 Mục tiêu đồ án
- 1.3 Phạm vi nghiên cứu
- 1.4 Ý nghĩa khoa học và thực tiễn
- 1.5 Cấu trúc báo cáo

### **Chương 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG**
- 2.1 Tổng quan về Thị giác Máy tính
- 2.2 Mạng nơ-ron tích chập (CNN)
- 2.3 Kiến trúc ResNet50
- 2.4 Cơ chế chú ý Squeeze-and-Excitation (SE)
- 2.5 Học đa nhiệm (Multi-Task Learning)
- 2.6 Hàm mất mát chuyên biệt
  - 2.6.1 Wing Loss
  - 2.6.2 Focal Loss
  - 2.6.3 Adaptive Uncertainty Weighting
- 2.7 Phát hiện khuôn mặt (Face Detection)
  - 2.7.1 MTCNN
  - 2.7.2 Haar Cascade
- 2.8 Xử lý ảnh số (DIP)
  - 2.8.1 CLAHE
  - 2.8.2 Gaussian Blur
- 2.9 Các kỹ thuật huấn luyện nâng cao
  - 2.9.1 Transfer Learning & Freeze/Unfreeze
  - 2.9.2 Mixup Augmentation
  - 2.9.3 Progressive Resizing
  - 2.9.4 Gradient Accumulation
  - 2.9.5 WeightedRandomSampler
  - 2.9.6 Test-Time Augmentation (TTA)
- 2.10 Công nghệ và thư viện sử dụng

### **Chương 3: PHÂN TÍCH YÊU CẦU VÀ THIẾT KẾ HỆ THỐNG**
- 3.1 Phân tích yêu cầu chức năng
- 3.2 Phân tích yêu cầu phi chức năng
- 3.3 Biểu đồ Use Case
- 3.4 Biểu đồ hoạt động (Activity Diagram)
- 3.5 Kiến trúc tổng thể hệ thống
- 3.6 Thiết kế cấu trúc thư mục (Project Structure)
- 3.7 Thiết kế dữ liệu
  - 3.7.1 Bộ dữ liệu UTKFace
  - 3.7.2 Cấu trúc file CSV nhãn
  - 3.7.3 Phân chia Train/Val/Test
- 3.8 Thiết kế kiến trúc mô hình
- 3.9 Thiết kế pipeline Huấn luyện
- 3.10 Thiết kế pipeline Suy luận (Inference)
- 3.11 Thiết kế giao diện người dùng (UI/UX)

### **Chương 4: THIẾT KẾ CHI TIẾT VÀ CÀI ĐẶT**
- 4.1 Module Models — Kiến trúc CNN
  - 4.1.1 Lớp SEBlock
  - 4.1.2 Lớp FaceAttributeModel
- 4.2 Module Data — Dataset và DataLoader
  - 4.2.1 Lớp UTKFaceDataset
  - 4.2.2 Mixup Augmentation
  - 4.2.3 Module loader.py
- 4.3 Module Losses — Hàm mất mát
  - 4.3.1 Lớp FocalLoss
  - 4.3.2 Lớp WingLoss
  - 4.3.3 Lớp MultiTaskLoss
- 4.4 Module Detection — Phát hiện khuôn mặt
- 4.5 Module Preprocessing — Xử lý ảnh số
- 4.6 Module Inference — Pipeline suy luận
  - 4.6.1 Lớp FaceAnalyzer
  - 4.6.2 Test-Time Augmentation
  - 4.6.3 Batch Prediction
- 4.7 Module Utils — Hằng số và cấu hình
- 4.8 Script huấn luyện (train_v3.py)
- 4.9 Script đánh giá (eval.py)
- 4.10 Script tổ chức dữ liệu (organize_data.py)
- 4.11 Script xuất biểu đồ (export_charts.py)
- 4.12 Ứng dụng web (app.py)
  - 4.12.1 Cấu hình Streamlit
  - 4.12.2 Thiết kế CSS
  - 4.12.3 Tab Ảnh tĩnh
  - 4.12.4 Tab Camera trực tiếp
  - 4.12.5 Tab Biểu đồ đánh giá
- 4.13 Notebook huấn luyện trên Google Colab

### **Chương 5: KẾT QUẢ THỰC HIỆN VÀ ĐÁNH GIÁ**
- 5.1 Môi trường thực nghiệm
- 5.2 Quá trình huấn luyện
- 5.3 Kết quả đánh giá tổng quan
- 5.4 Đánh giá chi tiết dự đoán Tuổi
- 5.5 Đánh giá chi tiết dự đoán Giới tính
- 5.6 Đánh giá chi tiết dự đoán Sắc tộc
- 5.7 Phân tích Confusion Matrix
- 5.8 Radar hiệu năng tổng quan
- 5.9 Ảnh chụp giao diện ứng dụng
- 5.10 Đánh giá hiệu suất thời gian thực
- 5.11 So sánh với các phiên bản trước

### **Chương 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**
- 6.1 Kết luận
- 6.2 Đóng góp của đồ án
- 6.3 Hạn chế
- 6.4 Hướng phát triển tương lai

### **TÀI LIỆU THAM KHẢO**

### **PHỤ LỤC**
- Phụ lục A: Hướng dẫn cài đặt và chạy hệ thống
- Phụ lục B: Mã nguồn chính
- Phụ lục C: Cấu trúc cơ sở dữ liệu
- Phụ lục D: Danh sách biểu đồ đã xuất

---

# DANH MỤC HÌNH ẢNH

| STT | Tên hình | Mô tả | Trang |
|-----|----------|-------|-------|
| 1.1 | Ứng dụng dự đoán thuộc tính khuôn mặt | Ví dụ minh họa bài toán Face Attribute Prediction | |
| 2.1 | Kiến trúc mạng CNN tổng quát | Các lớp tích chập, pooling và fully connected | |
| 2.2 | Kiến trúc ResNet50 | Sơ đồ residual block và skip connection | |
| 2.3 | Khối Squeeze-and-Excitation | Cơ chế nén và kích hoạt kênh | |
| 2.4 | Pipeline MTCNN | 3 giai đoạn P-Net → R-Net → O-Net | |
| 2.5 | CLAHE histogram equalization | So sánh trước và sau cân bằng sáng | |
| 3.1 | Biểu đồ Use Case | Các tác nhân và chức năng hệ thống | |
| 3.2 | Kiến trúc tổng thể | 4 tầng: Input → Detection → Preprocessing → Prediction | |
| 3.3 | Flowchart pipeline Inference | Luồng xử lý từ ảnh đầu vào đến kết quả | |
| 3.4 | Phân bố dữ liệu UTKFace | Biểu đồ phân bố theo giới tính, sắc tộc, nhóm tuổi | |
| 4.1 | Kiến trúc FaceAttributeModel | ResNet50 + SE + 3 heads | |
| 5.1 | Training curves | Biểu đồ Loss, MAE, Gender Acc, Race Acc qua 50 epochs | |
| 5.2 | Gender Confusion Matrix | Ma trận nhầm lẫn giới tính (Male/Female) | |
| 5.3 | Race Confusion Matrix | Ma trận nhầm lẫn sắc tộc (4 lớp) | |
| 5.4 | Sai số tuổi theo nhóm | MAE cho 5 nhóm tuổi: Child → Senior | |
| 5.5 | Radar hiệu năng | Biểu đồ radar 5 chỉ số | |
| 5.6 | Score card | Điểm tổng hợp và xếp hạng model | |
| 5.7 | Screenshot Tab 1 | Giao diện dự đoán ảnh tĩnh | |
| 5.8 | Screenshot Tab 2 | Giao diện camera trực tiếp | |
| 5.9 | Screenshot Tab 3 | Giao diện biểu đồ đánh giá | |

---

# DANH MỤC BẢNG BIỂU

| STT | Tên bảng | Mô tả | Trang |
|-----|----------|-------|-------|
| 2.1 | Công nghệ và thư viện sử dụng | Danh sách thư viện Python với phiên bản | |
| 3.1 | Thống kê bộ dữ liệu UTKFace | Phân bố mẫu theo split (Train/Val/Test) | |
| 3.2 | Ánh xạ nhãn sắc tộc | Race remap 5 lớp → 4 lớp | |
| 3.3 | Cấu hình mặc định | Các hyperparameter cho huấn luyện | |
| 4.1 | Số lượng tham số mô hình | Backbone, SE, Age head, Gender head, Race head | |
| 4.2 | Cấu hình Data Augmentation | Các phép biến đổi trong training transform | |
| 5.1 | Kết quả đánh giá tổng quan | Age MAE, Gender Acc, Race Acc | |
| 5.2 | Sai số tuổi theo nhóm chi tiết | MAE, Median, số mẫu cho mỗi nhóm tuổi | |
| 5.3 | Classification Report — Gender | Precision, Recall, F1-Score cho Male/Female | |
| 5.4 | Classification Report — Race | Precision, Recall, F1-Score cho 4 lớp sắc tộc | |
| 5.5 | So sánh kết quả v2 và v3+ | Delta metrics giữa các phiên bản | |

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
