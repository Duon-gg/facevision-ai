# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

---

## 6.1 Kết luận

Đồ án "Hệ thống dự đoán thuộc tính khuôn mặt sử dụng Học sâu Đa nhiệm" (FaceVision AI) đã hoàn thành đầy đủ các mục tiêu đề ra từ ban đầu. Cụ thể:

### 6.1.1 Về mặt kỹ thuật

1. **Kiến trúc mô hình**: Đã xây dựng thành công một mô hình CNN đa nhiệm (Multi-Task CNN) sử dụng backbone ResNet50 pretrained, kết hợp cơ chế chú ý kênh Squeeze-and-Excitation (SE) và 3 prediction heads riêng biệt cho Tuổi (regression), Giới tính (binary classification), Sắc tộc (multi-class classification). Tổng số tham số: ~26,2 triệu.

2. **Hàm mất mát đa nhiệm**: Đã thiết kế và cài đặt hệ thống hàm mất mát kết hợp Wing Loss (cho hồi quy tuổi — nhạy cảm với sai số nhỏ), Focal Loss (cho phân loại sắc tộc — giải quyết class imbalance), và Adaptive Uncertainty Weighting (tự động học trọng số nhiệm vụ theo phương pháp Kendall et al., 2018).

3. **Pipeline huấn luyện**: Đã tích hợp 9 kỹ thuật huấn luyện nâng cao: Freeze-then-Unfreeze backbone, Differential Learning Rate, Gradient Accumulation (effective batch 128), Mixup Augmentation (α=0.2), Progressive Resizing (160→192→224), WeightedRandomSampler (multi-factor: Race + Age group), AMP (Mixed Precision Training), Gradient Clipping, và Early Stopping.

4. **Pipeline suy luận**: Đã xây dựng pipeline inference hoàn chỉnh với phát hiện khuôn mặt hai tầng (MTCNN primary + Haar Cascade fallback), tiền xử lý ảnh số CLAHE trên không gian màu LAB, và Test-Time Augmentation (TTA) với 5 phép biến đổi.

5. **Giao diện ứng dụng**: Đã xây dựng ứng dụng web Streamlit chuyên nghiệp với 3 chế độ: dự đoán ảnh tĩnh (upload/webcam/clipboard), camera trực tiếp (15-25 FPS), và bảng biểu đồ đánh giá mô hình tương tác (13 biểu đồ Plotly).

### 6.1.2 Về mặt kết quả

Tất cả chỉ tiêu đánh giá đều đạt hoặc vượt mục tiêu đề ra:

| Chỉ tiêu | Mục tiêu | Thực tế | Đánh giá |
|----------|----------|---------|---------|
| Age MAE | ≤ 5,0 năm | **4,49 năm** | ✅ Đạt (vượt 10,2%) |
| Gender Accuracy | ≥ 90% | **93,53%** | ✅ Đạt (vượt 3,9%) |
| Race Accuracy | ≥ 80% | **85,69%** | ✅ Đạt (vượt 7,1%) |
| Camera FPS | ≥ 15 FPS | **17-25 FPS** | ✅ Đạt |
| Điểm tổng hợp | ≥ 80/100 | **~90/100** | ✅ Xếp hạng B+ |

### 6.1.3 Về mặt phương pháp luận

Đồ án đã chứng minh tính hiệu quả của các kỹ thuật sau trong bài toán Face Attribute Prediction:

- **Multi-Task Learning**: Huấn luyện đồng thời 3 nhiệm vụ (Age, Gender, Race) trên cùng một backbone không chỉ tiết kiệm tài nguyên mà còn cải thiện khả năng tổng quát hóa nhờ chia sẻ biểu diễn đặc trưng.
- **Wing Loss**: Hiệu quả hơn SmoothL1 Loss ~0,5 năm MAE cho bài toán age regression trên khuôn mặt.
- **Focal Loss**: Cải thiện đáng kể accuracy của lớp thiểu số (Black: +5,2%, Asian: +3,8%) so với Cross Entropy thuần.
- **Adaptive Uncertainty Weighting**: Tự động cân bằng trọng số 3 task mà không cần grid search, tiết kiệm thời gian hyperparameter tuning.
- **Progressive Resizing**: Giảm ~20% thời gian training ở giai đoạn đầu mà không ảnh hưởng đến kết quả cuối cùng.

---

## 6.2 Đóng góp của đồ án

### 6.2.1 Đóng góp học thuật

1. **Pipeline toàn diện**: Cung cấp một pipeline end-to-end hoàn chỉnh từ thu thập dữ liệu, tiền xử lý, huấn luyện, đánh giá đến triển khai giao diện — có thể tái sử dụng cho các bài toán tương tự.

2. **Tổng hợp kỹ thuật**: Kết hợp 9+ kỹ thuật tiên tiến vào một hệ thống thống nhất với đánh giá chi tiết hiệu quả từng kỹ thuật.

3. **Đánh giá chi tiết**: Phân tích chi tiết sai số theo nhóm tuổi (5 nhóm), confusion matrix per class, classification report (Precision/Recall/F1) — giúp hiểu rõ điểm mạnh/yếu của mô hình.

### 6.2.2 Đóng góp thực tiễn

1. **Mã nguồn mở**: Toàn bộ code theo giấy phép MIT, cấu trúc modular rõ ràng (7 module), dễ mở rộng và tùy chỉnh.

2. **Giao diện thân thiện**: Ứng dụng Streamlit 3 chế độ giúp cả người dùng kỹ thuật lẫn phi kỹ thuật đều có thể sử dụng.

3. **Tài liệu đầy đủ**: README.md, PROJECT_STRUCTURE.md, data/README.md, comment code tiếng Việt — hỗ trợ việc bảo trì và tiếp tục phát triển.

4. **Hỗ trợ Google Colab**: Notebook training sẵn sàng cho người dùng không có GPU mạnh.

---

## 6.3 Hạn chế

Mặc dù đã đạt được kết quả tốt, đồ án vẫn còn một số hạn chế cần được lưu ý:

### 6.3.1 Hạn chế về dữ liệu

1. **Mất cân bằng lớp**: Nhóm Others (Indian + Others gộp lại) chiếm ~23,9% mẫu nhưng F1-Score chỉ đạt 68,10% — thấp nhất trong 4 lớp. Nguyên nhân: nhóm gộp có ngoại hình quá đa dạng, khó phân biệt.

2. **Thiên lệch tuổi**: 43,8% mẫu test thuộc nhóm 20-35 → mô hình thiên về nhóm tuổi này. Nhóm Teen (13-19) chỉ có 186 mẫu (5,2%) → thiếu đại diện.

3. **Đa dạng sắc tộc**: 4 lớp sắc tộc chưa phản ánh đầy đủ sự đa dạng toàn cầu (thiếu Latino/Hispanic, Middle Eastern, Pacific Islander, v.v.).

4. **Chất lượng ảnh**: UTKFace có nhiều ảnh chất lượng thấp, mờ, thiếu sáng — ảnh hưởng đến accuracy nhóm tuổi cao.

### 6.3.2 Hạn chế về mô hình

1. **Sai số nhóm tuổi cao**: Nhóm 56+ có MAE = 8,44 năm (gấp 5× nhóm 0-12) — quá trình lão hóa phi tuyến và đa dạng khiến dự đoán khó chính xác.

2. **Nhầm lẫn Others ↔ White**: 290 mẫu nhầm lẫn giữa Others và White — do Indian có ngoại hình gần gũi với White/South Asian.

3. **ResNet50 backbone**: Mặc dù mạnh, ResNet50 (~23,5M params) có thể quá nặng cho triển khai mobile/edge. Các backbone nhẹ hơn (MobileNetV3, EfficientNet-B0) chưa được thử nghiệm.

### 6.3.3 Hạn chế về triển khai

1. **Chỉ hỗ trợ local**: Chưa triển khai trên cloud (AWS, GCP, Azure) hoặc web server công khai.
2. **Chưa có API**: Chưa cung cấp REST API cho tích hợp với ứng dụng khác.
3. **Chưa anti-spoofing**: Không có cơ chế phát hiện ảnh giả (printed photo, screen photo), dễ bị đánh lừa.
4. **Chưa hỗ trợ multi-face tracking**: Camera live detect face mỗi frame độc lập, không track ID theo thời gian.

---

## 6.4 Hướng phát triển tương lai

### 6.4.1 Cải thiện mô hình

| Hướng | Mô tả | Kỳ vọng |
|-------|-------|---------|
| **EfficientNet backbone** | Thay ResNet50 bằng EfficientNet-B3/B4 — nhẹ hơn nhưng accuracy tương đương | Giảm 50% params, tăng FPS |
| **Vision Transformer (ViT)** | Sử dụng ViT-Small/DeiT cho global attention thay vì local features | Tăng Race Acc ~3-5% |
| **ArcFace embedding** | Thêm ArcFace loss cho race classification | Tăng inter-class separation |
| **Ordinal Regression** | Thay age regression bằng ordinal regression (CORAL) | Giảm outlier errors |
| **Label Distribution Learning** | Dự đoán phân bố tuổi thay vì điểm đơn | Giảm MAE nhóm 56+ |

### 6.4.2 Mở rộng dữ liệu

| Hướng | Mô tả |
|-------|-------|
| **MORPH II** | Bộ dữ liệu 55.000 ảnh, chất lượng cao hơn UTKFace |
| **IMDB-WIKI** | 500.000+ ảnh từ IMDB và Wikipedia (noisy labels, cần lọc) |
| **FairFace** | 108.000 ảnh, cân bằng 7 sắc tộc — giải quyết class imbalance |
| **APPA-REAL** | 7.591 ảnh với apparent age annotation (nhiều annotator) |
| **Cross-dataset evaluation** | Train trên UTKFace, test trên MORPH — đánh giá generalization |

### 6.4.3 Triển khai sản phẩm

| Hướng | Công nghệ | Mô tả |
|-------|-----------|-------|
| **REST API** | FastAPI / Flask | Cung cấp API endpoint cho tích hợp microservice |
| **Cloud Deployment** | Docker + AWS/GCP | Triển khai trên cloud với auto-scaling |
| **Mobile App** | TensorFlow Lite / ONNX | Export model cho Android/iOS |
| **Edge Computing** | NVIDIA Jetson / Coral | Triển khai trên thiết bị edge (camera thông minh) |
| **Web App** | Gradio / Streamlit Cloud | Deploy public cho demo |

### 6.4.4 Tính năng bổ sung

| Tính năng | Mô tả |
|----------|-------|
| **Face Anti-Spoofing** | Phát hiện ảnh giả (liveness detection) |
| **Expression Recognition** | Thêm task dự đoán biểu cảm (Happy, Sad, Angry, ...) |
| **Face Landmark Detection** | Detect 68 facial landmarks cho alignment |
| **Multi-Face Tracking** | Track ID khuôn mặt liên tục trong video |
| **Age Transformation** | Biến đổi khuôn mặt theo tuổi (aging/de-aging) sử dụng GAN |
| **Privacy Protection** | Mã hóa/ẩn danh khuôn mặt trước khi lưu trữ |

### 6.4.5 Nghiên cứu tiếp theo

1. **Fairness Analysis**: Đánh giá công bằng (fairness) của mô hình giữa các nhóm sắc tộc — liệu accuracy có bias theo sắc tộc nào không?
2. **Explainability**: Sử dụng Grad-CAM / SHAP để giải thích vùng ảnh nào ảnh hưởng đến dự đoán.
3. **Continual Learning**: Cho phép mô hình tiếp tục học từ dữ liệu mới mà không quên kiến thức cũ.
4. **Knowledge Distillation**: Nén mô hình ResNet50 vào MobileNet bằng knowledge distillation, giữ ~95% accuracy.
