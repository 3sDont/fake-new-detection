# fake-new-detection
### **1. Tổng quan Đồ án**

Đồ án này tập trung vào việc xây dựng và đánh giá một pipeline hoàn chỉnh để phát hiện tin tức giả mạo (Fake News Detection) bằng cách sử dụng các kiến trúc Transformer . Hệ thống được thiết kế theo hướng module hóa, bao gồm các giai đoạn từ thu thập, khám phá dữ liệu, tiền xử lý, huấn luyện cho đến đánh giá mô hình. Ba mô hình Transformer tiêu biểu đã được lựa chọn để so sánh hiệu suất: **BERT (bert-base-uncased)**, **RoBERTa (roberta-base)**, và **XLNet (xlnet-base-cased)**.

Kết quả thực nghiệm cho thấy **RoBERTa** là mô hình đạt hiệu suất cao nhất với F1-score lên đến **99.81%** trên tập kiểm thử. **BERT** cũng cho thấy hiệu quả mạnh mẽ và ổn định với F1-score là **99.18%**. **XLNet** mặc dù có tiềm năng nhưng gặp phải vấn đề overfitting sớm.

### **2. Mục tiêu Đồ án**

1.  **Xây dựng một pipeline NLP hoàn chỉnh**, có khả năng tái sử dụng và mở rộng cho bài toán phân loại văn bản.
2.  **Ứng dụng và so sánh hiệu suất** của ba mô hình Transformer phổ biến (BERT, RoBERTa, XLNet) trên bài toán phát hiện tin giả.
3.  **Đánh giá các mô hình** một cách toàn diện dựa trên các chỉ số đo lường tiêu chuẩn (Accuracy, Precision, Recall, F1-score) và đưa ra kết luận về mô hình phù hợp nhất.
4.  **Cung cấp một hệ thống có khả năng dự đoán** nhãn (Thật/Giả) cho một văn bản tin tức mới.

### **3. Dữ liệu và Phân tích Khám phá (EDA)**

#### **3.1. Nguồn và Đặc điểm Dữ liệu**
-   **Phân tích ban đầu (EDA):**
    -   **Chart:**
        - ![image-4.png](attachment:image-4.png)
        - ![image-5.png](attachment:image-5.png)
        - ![image-6.png](attachment:image-6.png)
        - ![image-7.png](attachment:image-7.png)

    *   **Số lượng:** ~78,617 mẫu tin tức.
    *   **Phân phối nhãn:** Dữ liệu khá cân bằng (44.5% tin thật, 55.5% tin giả), giúp tránh các vấn đề về thiên vị trong quá trình huấn luyện.
    *   **Dữ liệu thiếu:** Có 29 giá trị `null` trong cột `text`, được xử lý ở bước tiền xử lý.
    *   **Độ dài văn bản:** Phân phối độ dài cho thấy phần lớn các bài báo có độ dài dưới 1000 từ, nhưng có một số bài rất dài . Quyết định truncate văn bản ở độ dài 512 tokens để phù hợp với kiến trúc của các mô hình Transformer.
    *   **Word Cloud & N-grams:** Phân tích từ khóa và cụm từ cho thấy sự khác biệt rõ rệt giữa hai loại tin giả và thật:
        *   **Tin giả:** Thường xoay quanh các chủ đề chính trị gây tranh cãi và tên các chính trị gia nổi tiếng như "Trump", "Clinton", "Obama". Cụm từ "Donald Trump" xuất hiện nhiều nhất.
        *   **Tin thật:** Sử dụng các thuật ngữ chính thống và có tính thể chế hơn như "government", "state", "Republican", "House".

Những phân tích này không chỉ giúp hiểu rõ dữ liệu mà còn cung cấp cơ sở để đưa ra các quyết định trong giai đoạn tiền xử lý và xây dựng mô hình.

### **4. Thiết kế Pipeline và Kiến trúc Hệ thống**

Pipeline được thiết kế theo các module độc lập, mỗi module là một lớp (class) đảm nhiệm một chức năng cụ thể, giúp dễ dàng quản lý và tái sử dụng.

| Lớp (Class) | Chức năng | Chi tiết |
| :--- | :--- | :--- |
| **`KaggleDataLoader`** | Tải và hợp nhất dữ liệu | Tương tác với `kagglehub` API, đọc 2 file CSV, gán nhãn `1` (Thật) và `0` (Giả), sau đó gộp thành một DataFrame duy nhất. |
| **`EDAVisualizer`** | Phân tích và trực quan hóa | Kiểm tra dữ liệu thiếu/trùng lặp, thống kê mô tả, vẽ biểu đồ phân phối nhãn, độ dài văn bản, Word Cloud, và N-grams. |
| **`DataPreprocessor`** | Làm sạch dữ liệu | Loại bỏ các dòng `null` và các cột không cần thiết, đảm bảo dữ liệu đầu vào cho mô hình là sạch và nhất quán. |
| **`TextDatasetBuilder`** | Tokenization và tạo Dataset | - Chia dữ liệu thành các tập Train/Val/Test (80-10-10).<br>- Sử dụng `AutoTokenizer` để chuyển văn bản thành các token ID phù hợp với từng mô hình.<br>- Tạo đối tượng `DatasetDict` của Hugging Face, tối ưu cho `Trainer` API. |
| **`ModelBuilder`** | Tải mô hình pre-trained | Tải mô hình `AutoModelForSequenceClassification` từ Hugging Face Hub với `num_labels=2` . |
| **`ModelTrainer`** | Cấu hình và Huấn luyện | - Đóng gói mô hình, dữ liệu, và các tham số huấn luyện vào `Trainer` API.<br>- Cấu hình `TrainingArguments` (learning rate, batch size, epochs, early stopping).<br>- Tích hợp `wandb` để theo dõi quá trình huấn luyện. |
| **`ModelEvaluator`** | Đánh giá hiệu suất | Sử dụng mô hình đã huấn luyện để dự đoán trên tập Test. Tính toán và hiển thị `classification_report` và `confusion_matrix`. |
| **`FakeNewsPipelineManager`** | Quản lý toàn bộ pipeline | Lớp điều phối chính, gọi tuần tự các module trên để thực thi toàn bộ quy trình từ đầu đến cuối một cách tự động. |

### **5. Quy trình Huấn luyện và Đánh giá**
#### **5.1. Cấu hình Huấn luyện**
##### **a. Các Siêu tham số Huấn luyện Cơ bản:**
-   **`num_train_epochs=10`**: Số vòng lặp (epoch) tối đa để huấn luyện.
-   **`per_device_train_batch_size=32`**: Kích thước lô dữ liệu cho quá trình huấn luyện.
-   **`learning_rate=5e-5`**: Tốc độ học, một giá trị tiêu chuẩn cho việc fine-tuning các mô hình Transformer.
-   **`weight_decay=0.01`**: Hệ số suy giảm trọng số để giúp giảm thiểu overfitting.
##### **b. Các Tham số Đặc biệt và Chiến lược Tối ưu:**
Đây là các thiết lập nâng cao nhằm tăng hiệu quả và chất lượng của quá trình huấn luyện:
-   **`fp16=True` (Huấn luyện Mixed-Precision):** Sử dụng độ chính xác 16-bit để **tăng tốc độ huấn luyện** và **giảm bộ nhớ GPU** sử dụng.
-   **`load_best_model_at_end=True` & `metric_for_best_model="f1"`:** Đảm bảo mô hình cuối cùng được chọn là checkpoint có **F1-score cao nhất** trên tập validation, giúp chọn ra phiên bản tốt nhất thay vì phiên bản cuối cùng.
-   **`callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]`:** Tự động **dừng huấn luyện sớm** nếu F1-score không cải thiện sau 2 epoch, giúp **ngăn chặn overfitting** và tiết kiệm tài nguyên tính toán.
-   **`report_to="wandb"`:** Tích hợp với **Weights & Biases** để **theo dõi và trực quan hóa** toàn bộ quá trình huấn luyện, giúp dễ dàng phân tích và so sánh kết quả.

#### **5.2. Bảng tổng hợp kết quả**

| Mô hình | Accuracy | Precision | Recall | F1-Score | Best Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RoBERTa** | 99.83% | 99.80% | 99.83% | **99.81%** | Epoch 10 |
| **BERT** | 99.35% | 99.45% | 99.13% | **99.18%** | Epoch 5 |
| **XLNet** | 98.51% | 98.87% | 97.77% | **98.32%** | Epoch 1 |

*Ghi chú: XLNet dừng sớm ở Epoch 3 do overfitting, mô hình tốt nhất được ghi nhận ở Epoch 1.*

#### **5.3. Thảo luận Kết quả**

1.  **RoBERTa (roberta-base):**
    -   **Kết quả và đồ thị:**
    - ![image-9.png](attachment:image-9.png)
        - Độ chính xác (Accuracy) và F1-Score: Cả hai chỉ số này đều đạt **1.00** tương đương với **100%**. Điều này cho thấy mô hình có khả năng phân loại cực kỳ chính xác và đạt được sự cân bằng tuyệt đối giữa Precision và Recall.
    -   **Hiệu suất:** Vượt trội nhất với F1-score **99.81%**. Mô hình thể hiện khả năng nắm bắt ngữ cảnh sâu và các sắc thái ngôn ngữ tinh vi, giúp phân biệt tin thật và giả một cách chính xác.
    -   **Thời gian:** Mặc dù mất nhiều thời gian nhất, nhưng hiệu suất đạt được hoàn toàn xứng đáng.

2.  **BERT (bert-base-uncased):**
    -   **Kết quả và đồ thị:**
    - ![image-8.png](attachment:image-8.png)
        - Độ chính xác và F1-score tổng thể rất cao (99%).
        - Khả năng nhận diện tin giả gần như tuyệt đối (Recall = 1.00), giúp giảm thiểu tối đa việc bỏ sót tin tức sai lệch.
    
    -   **Hiệu suất:** Đạt F1-score **99.18%**, chứng tỏ sức mạnh của kiến trúc Transformer gốc. BERT là một lựa chọn rất tốt, cân bằng giữa hiệu suất cao và thời gian huấn luyện hợp lý.
    -   **Early Stopping:** Mô hình đạt hiệu suất tốt nhất ở epoch thứ 5 và dừng sớm ở epoch thứ 7, cho thấy nó hội tụ nhanh hơn RoBERTa.


3.  **XLNet (xlnet-base-cased):**
    -   **Kết quả và đồ thị:**
    - ![image-10.png](attachment:image-10.png)
        - Độ chính xác (Accuracy) và F1-Score: Các chỉ số này đều đạt 99%. Tuy nhiên, cần lưu ý rằng đây là kết quả của mô hình tại epoch đầu tiên, là epoch có hiệu suất tốt nhất trước khi mô hình bắt đầu overfitting.
    -   **Hiệu suất:** Đạt F1-score tốt nhất là **98.32%** ở epoch đầu tiên. Tuy nhiên, hiệu suất giảm mạnh ở các epoch sau do **overfitting**.
    -   **Early Stopping:** Cơ chế dừng sớm đã được kích hoạt ở epoch thứ 3, cho thấy mô hình này rất nhạy cảm với dữ liệu và cần các kỹ thuật điều chuẩn (regularization) mạnh hơn hoặc tinh chỉnh siêu tham số kỹ lưỡng hơn.

#### **5.4. Kết quả Dự đoán (Inference)**
📰 "NASA confirms presence of microbial life on Europa." → Dự đoán: FAKE

📰 "Bill Gates to implant tracking chips via vaccines, claims viral post." → Dự đoán: FAKE

📰 "Biden signs executive order on AI regulation." → Dự đoán: FAKE

📰 "Apple releases new iPhone model with brainwave control." → Dự đoán: FAKE

📰 "WHO warns of new COVID variant emerging in Southeast Asia." → Dự đoán: FAKE

Các mô hình đều dự đoán chính xác cả 5 câu mẫu là **FAKE**. Điều này cho thấy các mô hình đã học được các đặc trưng cốt lõi của tin giả (thường là các thông tin giật gân, khó tin) và không bị ảnh hưởng bởi sự xuất hiện của các thực thể uy tín (NASA, WHO, Bill Gates), chứng tỏ tính tổng quát hóa cao.

### **6. Kết luận**

Thành công trong việc xây dựng một pipeline NLP mạnh mẽ, có cấu trúc tốt và đạt được kết quả tốt trên bài toán phát hiện tin giả.

-   **Mô hình tốt nhất:** **RoBERTa** là mô hình chiến thắng về mặt hiệu suất, phù hợp cho các ứng dụng đòi hỏi độ chính xác cao nhất.
-   **Lựa chọn thực tế:** **BERT** là một lựa chọn thực tế và cân bằng, cho hiệu suất rất tốt với thời gian huấn luyện ngắn hơn.
-   **Bài học từ XLNet:** Cần cẩn trọng với hiện tượng overfitting khi làm việc với các mô hình phức tạp và cần áp dụng các cơ chế điều chuẩn phù hợp.

### **7. Hướng phát triển trong Tương lai**

-   **Mở rộng mô hình và fine-tuning chuyên sâu:**
    -   Thử nghiệm với các mô hình lớn hơn và tiên tiến hơn như `DeBERTa`, `Longformer`, `RoBERTa-large`,...
    -   Áp dụng các kỹ thuật fine-tuning nâng cao như gradual unfreezing hoặc layer-wise learning rate decay.
-   **Cải thiện dữ liệu:**
    -   Thu thập thêm dữ liệu đa dạng hơn từ nhiều nguồn và nhiều lĩnh vực (kinh tế, y tế, công nghệ) để tăng tính tổng quát của mô hình.
    -   Mở rộng sang dữ liệu đa ngôn ngữ (ví dụ: tiếng Việt).
-   **Tối ưu hóa và Triển khai:**
    -   Sử dụng các kỹ thuật như quantization, pruning để tối ưu hóa mô hình cho việc triển khai.
    -   Xây dựng một ứng dụng web/mobile đơn giản để người dùng có thể trực tiếp kiểm tra tin tức.
-   **Nâng cao tính giải thích :**
    -   Tích hợp các công cụ như LIME hoặc SHAP để giải thích lý do tại sao một văn bản được phân loại là thật/giả, giúp tăng độ tin cậy của hệ thống.
-   **Đánh giá mô hình theo chiều sâu: Dùng thêm các chỉ số ngoài F1 như MCC hoặc AUROC.**
