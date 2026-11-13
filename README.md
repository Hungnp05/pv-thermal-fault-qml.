# Chẩn đoán lỗi tấm pin quang điện từ ảnh nhiệt: so sánh học sâu cổ điển và mô hình lai lượng tử – cổ điển

> *Reliable and Generalizable Photovoltaic Fault Diagnosis Using Thermal Images: A Comparative Study of Classical Deep Learning and Hybrid Quantum-Classical Models*

Đồ án tốt nghiệp đại học. Bài toán: phân loại lỗi trên ảnh nhiệt của **từng tấm pin quang điện (PV)**, chụp từ drone. Đề tài trả lời ba câu hỏi:

1. Mạng CNN nào phù hợp nhất với ảnh nhiệt PV?
2. Học máy lượng tử (QML) có phân loại được lỗi không? Nó có tốt hơn bộ phân loại cổ điển khi **dùng chung một bộ đặc trưng** không?
3. Mô hình có **đáng tin cậy** không: độ tin cậy có khớp với tỉ lệ đúng thực tế, mô hình có biết lúc nào nên từ chối trả lời, và có chịu được nhiễu ảnh cũng như dữ liệu từ ngày chụp khác không?

---

## 1. Bối cảnh

Nhà máy điện mặt trời có hàng chục nghìn tấm pin. Tấm bị lỗi (cell nóng, ngắn mạch, nứt vỡ, bẩn, bị che bóng, ...) thường nóng hơn bình thường, nên có thể phát hiện bằng camera nhiệt gắn trên drone. Ảnh thu về rất nhiều, nên cần mô hình tự động phân loại loại lỗi. Mô hình cũng cần **báo khi không chắc chắn** để chuyển cho kỹ thuật viên kiểm tra.

Song song, học máy lượng tử đang được quảng bá là hướng mới cho bài toán phân loại. Đề tài kiểm tra điều đó một cách **công bằng**: mọi bộ phân loại lượng tử và cổ điển dùng cùng đặc trưng, cùng bước giảm chiều, cùng dữ liệu và cùng cách chia fold.

## 2. Dữ liệu

| Bộ dữ liệu | Số ảnh | Số lớp | Kích thước | Vai trò |
|---|---|---|---|---|
| **PVF-10** (Wang và cộng sự, 2024, *Applied Energy*) | 5.579 | 10 | khoảng 60×110, ảnh màu giả nhiệt | Bộ chính |
| **InfraredSolarModules** (RaptorMaps) | 20.000 | 12 | 24×40, ảnh xám | Bộ kiểm chứng độc lập, không gộp với PVF-10 |

Dữ liệu **không có trong repo**. Cần tự tải và đặt đúng chỗ như sau:

```
data/raw/PVF-10.zip                         # tải từ https://github.com/wangbobby1026/PVF-Dataset
data/raw/PVF_10_Ori/ , PVF_10_110x60/ , PVF_10_112x112/     # giải nén zip trực tiếp vào data/raw/
data/raw/ism/InfraredSolarModules/          # giải nén từ https://github.com/RaptorMaps/InfraredSolarModules
    images/*.jpg
    module_metadata.json
```

### Phát hiện quan trọng: cách chia gốc của PVF-10 bị rò rỉ dữ liệu

Mỗi ảnh drone chứa nhiều tấm pin, và mỗi tấm được cắt ra thành một ảnh riêng. Vì vậy 5.579 ảnh của PVF-10 thực ra chỉ đến từ **1.236 khung hình**. Trong cách chia train/test gốc, **396 trong 422 khung hình của tập test cũng xuất hiện trong tập train**. Mô hình được kiểm tra trên các tấm pin nằm cạnh, cùng ánh sáng và cùng góc chụp với tấm nó đã học, nên kết quả bị đẩy lên cao.

Đề tài dùng ba cách chia:

| Cách chia | Cách tạo | Mục đích |
|---|---|---|
| `official` | Train/test gốc của tác giả | Chỉ để tái lập bài báo gốc và minh họa rò rỉ |
| `session` (chính) | Cắt dòng thời gian chụp ở các khoảng trống hơn 300 giây để tạo 143 phiên bay. Gộp các phiên có ảnh trùng hệt nhau, còn 136 nhóm. Sau đó chia 5 fold bằng StratifiedGroupKFold. | Benchmark chính, không rò rỉ |
| `date` | Mỗi ngày chụp (18 ngày) nằm trọn trong một fold | Phép thử thay thế cho khả năng tổng quát hóa sang nơi chụp khác (RQ4) |

Với fold k: tập test là fold k, tập validation là fold (k+1) mod 5, phần còn lại là tập train.

InfraredSolarModules không có thông tin khung hình hay thời gian chụp. Vì vậy bộ này được chia 5 fold phân tầng theo lớp, và 22 nhóm ảnh trùng hệt nhau được giữ trong cùng một fold.

## 3. Phương pháp

```
Ảnh nhiệt ──► CNN (fine-tune, timm) ──► đặc trưng 512–2048 chiều (lưu ra đĩa)
                    │                              │
                    ▼                              ▼
          phân loại trực tiếp       StandardScaler → PCA (4 hoặc 8 chiều) → MinMax về [0, π]
          (nhóm A)                                 │
                                   ┌───────────────┴────────────────┐
                                   ▼                                ▼
                     bộ phân loại cổ điển (nhóm C)    bộ phân loại lượng tử (nhóm B)
                     Linear, MLP, tanh, SVM RBF        VQC, quantum kernel + SVM
```

- **Nhóm A, CNN:** MobileNetV3-Small, ResNet18, EfficientNet-B0, DenseNet121, ResNet50 và CoaT-Lite-Small (transformer).
  - Mô hình được huấn luyện ImageNet sẵn. Lúc đầu chỉ huấn luyện lớp cuối, sau đó fine-tune toàn mạng bằng AdamW với lịch học cosine.
  - Hàm mất mát là cross-entropy có trọng số theo lớp, vì dữ liệu rất mất cân bằng. Mô hình được chọn ở epoch có macro-F1 trên validation cao nhất.
  - Ảnh được thêm viền đen thành hình vuông rồi resize về 224. Không lật dọc ảnh, vì một số lỗi như bẩn ở mép dưới phụ thuộc hướng ảnh.
- **Nhóm B, lượng tử (PennyLane, mô phỏng):**
  - **VQC:** mã hóa góc RY, sau đó là các lớp RY–RZ trên từng qubit, rồi chuỗi CNOT nối thẳng hoặc nối vòng. Kết quả đo giá trị kỳ vọng ⟨Z⟩ của từng qubit được đưa vào một lớp Linear để ra 10 lớp. Có 6 cấu hình:

    | Cấu hình | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
    |---|---|---|---|---|---|---|
    | Qubit | 4 | 4 | 4 | 8 | 8 | 8 |
    | Độ sâu | 1 | 2 | 3 | 1 | 2 | 3 |
    | Nối CNOT | thẳng | thẳng | vòng | thẳng | thẳng | vòng |

  - **Quantum kernel:** kernel fidelity |⟨ψ(x)|ψ(x′)⟩|², đưa vào SVM. Kernel được tính từ statevector mô phỏng, nên chỉ hợp lệ trên máy mô phỏng.
- **Nhóm C, đối chứng cổ điển:** Linear, MLP, SVM RBF, và `tanh`. Bộ `tanh` có đầu ra bị chặn trong [−1, 1] giống VQC, dùng để kiểm tra xem lợi thế của VQC có phải chỉ do đầu ra bị chặn hay không.
- **Nhóm D, độ tin cậy:**
  - softmax gốc, temperature scaling (fit trên tập validation), MC Dropout, và dự đoán chọn lọc (chỉ trả lời khi đủ tự tin).
  - Đo độ bền với 5 loại biến đổi ảnh × 5 mức: nhiễu Gauss, làm mờ, giảm tương phản, tăng độ sáng, giảm độ phân giải.

**Chỉ số đánh giá:**
- Macro-F1 (chỉ số chính), accuracy, balanced accuracy, recall từng lớp.
- ECE, Brier, NLL: đo độ khớp giữa độ tin cậy và tỉ lệ đúng.
- AURC, độ chính xác ở từng mức coverage: đo khả năng dự đoán chọn lọc.
- Thời gian huấn luyện và số tham số.

## 4. Kết quả chính (PVF-10)

### Rò rỉ dữ liệu làm kết quả cao hơn thực tế

| ResNet18 | Macro-F1 |
|---|---|
| Cách chia gốc (bị rò rỉ) | 0,911 |
| Chia theo phiên bay (5 fold) | 0,741 ± 0,051 |
| Chia theo ngày chụp (5 fold) | 0,496 ± 0,081 |

### RQ1: so sánh CNN

| CNN | Macro-F1 theo phiên bay | ECE | Macro-F1 theo ngày chụp | Giữ lại khi ảnh bị nhiễu |
|---|---|---|---|---|
| CoaT-Lite-S | **0,783** | 0,069 | **0,590** | **93%** |
| DenseNet121 | 0,772 | 0,074 | 0,560 | 84% |
| EfficientNet-B0 | 0,747 | 0,082 | 0,485 | 82% |
| ResNet50 | 0,741 | 0,056 | 0,489 | 77% |
| ResNet18 | 0,741 | **0,038** | 0,496 | 79% |
| MobileNetV3-S | 0,731 | 0,088 | 0,516 | 84% |

*Cột cuối: macro-F1 trung bình trên 25 điều kiện nhiễu ảnh, chia cho macro-F1 trên ảnh sạch (chia theo phiên bay).*

- CoaT-Lite tốt nhất về độ chính xác, khả năng tổng quát hóa và độ bền.
- Lớp khó nhất là *string short circuit*, hay bị nhầm với *short circuit panel* và *hot cell*.
- Giảm độ phân giải là biến đổi gây hại nhất.

### RQ2/RQ3: lượng tử so với cổ điển trên cùng đặc trưng (8 chiều, 15 lần chạy)

| Bộ phân loại | Đặc trưng ResNet18 | Đặc trưng DenseNet121 | Đặc trưng CoaT-Lite |
|---|---|---|---|
| SVM RBF | 0,738 | 0,747 | 0,790 |
| Quantum kernel | 0,738 | 0,756 | 0,790 |
| MLP | 0,726 | 0,730 | 0,784 |
| VQC tốt nhất | 0,711 | 0,704 | 0,772 |

- **Quantum kernel ngang SVM RBF** trên cả ba backbone.
- **VQC thua các bộ phân loại cổ điển** khoảng 0,02–0,05 macro-F1, và huấn luyện chậm hơn hàng chục đến hàng trăm lần trên máy mô phỏng: khoảng 150–280 giây, so với dưới 5 giây của MLP và SVM.
- VQC có ECE thấp hơn rõ rệt trên đặc trưng ResNet18, kể cả so với đối chứng `tanh` (p ≤ 0,01). Nhưng **kết quả này không lặp lại** trên DenseNet121 và CoaT-Lite. Vì vậy không thể kết luận rằng VQC cho độ tin cậy sát thực tế hơn.
- Chọn backbone tốt quan trọng hơn chọn bộ phân loại: đổi từ ResNet18 sang CoaT-Lite tăng khoảng 0,05 F1 cho mọi bộ phân loại.

### RQ4/RQ5: độ tin cậy và tổng quát hóa

- Khi gặp ngày chụp chưa thấy lúc huấn luyện, mọi CNN mất 0,19–0,26 macro-F1. Thứ hạng giữa các CNN cũng thay đổi.
- Temperature scaling giảm ECE, nhất là khi ảnh bị nhiễu. MC Dropout gần như không có tác dụng.
- Khi chia theo ngày chụp, dù chỉ trả lời 20% số ảnh mô hình tự tin nhất, tỉ lệ sai vẫn khoảng 15%. Nghĩa là khi gặp dữ liệu lạ, độ tin cậy của mô hình không còn đáng tin để quyết định từ chối.

### Hạn chế

- PVF-10 không có nhãn nhà máy, nên phép thử theo ngày chụp chỉ là phép thử thay thế. Mức sụt hiệu năng khi chia theo ngày chụp lẫn cả sự thay đổi phân bố lớp. Đề tài **không** khẳng định khả năng tổng quát hóa sang nhà máy khác.
- Mạch lượng tử chạy trên máy mô phỏng. Thời gian chạy và cách tính kernel không áp dụng được cho phần cứng lượng tử thật.
- InfraredSolarModules không chia được theo phiên bay, và có 6 nhóm ảnh trùng mang nhãn mâu thuẫn.

## 5. Cấu trúc mã nguồn

```
src/pvf/
  paths.py        đường dẫn dữ liệu và kết quả
  metadata.py     đọc tên lớp và tên file DJI_<thời gian>_<khung hình>_T_<số thứ tự>
  data.py         bảng dữ liệu, chia fold, tiền xử lý ảnh, Dataset
  models.py       tạo CNN từ timm
  quantum.py      VQC và quantum kernel (PennyLane)
  metrics.py      macro-F1, ECE, Brier, NLL, AURC, ...
  gradcam.py      HiResCAM: vùng ảnh mô hình dựa vào
  utils.py        seed, thiết bị, lưu kết quả
scripts/
  audit_data.py         kiểm tra dữ liệu, tìm ảnh trùng   -> data/metadata.csv
  make_splits.py        tạo fold theo phiên bay và theo ngày chụp -> data/splits.csv
  prepare_ism.py        tạo bảng và fold cho InfraredSolarModules
  train_cnn.py          huấn luyện một CNN trên một fold
  extract_features.py   trích và lưu đặc trưng CNN
  train_head.py         huấn luyện một bộ phân loại cổ điển hoặc lượng tử trên đặc trưng
  run_benchmark.py      chạy hàng loạt, tự bỏ qua lần chạy đã xong
  calibrate.py          temperature scaling và dự đoán chọn lọc
  evaluate_reliability.py  MC Dropout, độ bền với nhiễu ảnh
  summarize.py          bảng mean ± std
  make_figures.py       vẽ hình cho báo cáo
  predict.py            dự đoán ảnh mới, đánh giá nếu ảnh xếp theo thư mục lớp
  app.py                giao diện demo (Streamlit)
```

## 6. Cài đặt

Yêu cầu: Python 3.12, GPU NVIDIA có CUDA. Dự án được phát triển trên RTX 4050 6 GB.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # hoặc requirements-lock.txt để có đúng phiên bản
.venv/Scripts/python -m pip install -e .
```

## 7. Chạy thí nghiệm

Chạy từ thư mục gốc. Nên đặt biến môi trường `PYTHONWARNINGS=ignore` để tắt cảnh báo của timm và PennyLane.

```bash
# Chuẩn bị dữ liệu và fold (chạy một lần)
.venv/Scripts/python scripts/audit_data.py
.venv/Scripts/python scripts/make_splits.py
.venv/Scripts/python scripts/prepare_ism.py

# Chạy thử nhanh trên 10% dữ liệu
.venv/Scripts/python scripts/train_cnn.py --model resnet18 --subset 0.1 --head-epochs 1 --epochs 2 --tag _smoke

# Nhóm A: CNN trên 5 fold (thêm --scheme date hoặc --scheme ism để đổi cách chia hoặc đổi bộ dữ liệu)
.venv/Scripts/python scripts/run_benchmark.py cnn --models resnet18 densenet121 --seeds 0

# Nhóm B và C: bộ phân loại lượng tử và cổ điển trên cùng đặc trưng
.venv/Scripts/python scripts/run_benchmark.py heads --feature-model resnet18 --seeds 0 1 2

# Nhóm D: độ tin cậy và độ bền
.venv/Scripts/python scripts/calibrate.py --scheme session
.venv/Scripts/python scripts/evaluate_reliability.py --models resnet18 densenet121

# Bảng và hình
.venv/Scripts/python scripts/summarize.py
.venv/Scripts/python scripts/make_figures.py --head-features resnet18_finetuned
```

Kết quả được lưu trong `outputs/`:
- `cnn/`: checkpoint và logits của CNN.
- `features/`: đặc trưng đã trích.
- `heads/`: kết quả của các bộ phân loại.
- `results/`: các file CSV tổng hợp.
- `figures/`: hình PNG và PDF.

## 8. Dự đoán và giao diện demo

```bash
# Dự đoán ảnh mới. Ảnh có độ tin cậy dưới ngưỡng bị đánh dấu "uncertain".
.venv/Scripts/python scripts/predict.py <ảnh hoặc thư mục> --model resnet18 --threshold 0.7 --out preds.csv

# Đánh giá trên bộ ảnh riêng đã xếp theo thư mục tên lớp, ví dụ my_images/hot_cell/*.png
.venv/Scripts/python scripts/predict.py my_images --labels-from-folders --out outputs/preds/my_images.csv

# Giao diện web, mở tại http://localhost:8501
.venv/Scripts/streamlit run scripts/app.py
```

Giao diện có ba tab:
- **Diagnosis:** tải ảnh lên hoặc lấy ảnh ngẫu nhiên từ fold test. Kết quả gồm lớp lỗi, độ tin cậy sau temperature scaling, top-3 lớp, heatmap HiResCAM, và cảnh báo khi mô hình không chắc chắn.
- **Experiment results:** bảng tổng hợp và các hình cho báo cáo.
- **Quantum vs classical:** so sánh F1 và ECE giữa các bộ phân loại lượng tử và cổ điển theo từng backbone và số qubit.

Để dự đoán, cần có checkpoint trong `outputs/cnn/`. Checkpoint được tạo ra bằng cách huấn luyện theo mục 7.
