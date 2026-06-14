# ARPD: Adaptive Evidence Retrieval with Paraphrase-Robust Training for Fake News Detection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-LIAR%20%2B%20FEVER-red.svg)](https://huggingface.co/datasets/liar)

**A Lightweight Speaker-Aware Ensemble Framework with Adaptive Evidence Retrieval for Fake News Detection**

*Submitted to Expert Systems with Applications (Elsevier)*

</div>

---

## 📋 Tóm tắt

ARPD là một framework nhẹ (22.8M params) phát hiện tin giả kết hợp:
- **Adaptive Evidence Retrieval**: Tự động điều chỉnh số lượng evidence từ Wikipedia theo độ mơ hồ của claim
- **Speaker-Aware Context**: Khai thác metadata người phát biểu để cải thiện accuracy
- **Ensemble Learning**: Kết hợp TF-IDF+LR (90%) và MiniLM+MLP (10%)
- **Paraphrase Augmentation**: Tăng cường robustness bằng synonym substitution

### Kết quả chính

| Dataset | Model | Accuracy | F1-Macro | F1-FAKE |
|---------|-------|----------|----------|---------|
| LIAR | TF-IDF+LR (baseline) | 64.96% | 62.08% | 51.63% |
| LIAR | **ARPD (ours)** | **65.19%** | **62.59%** | **52.73%** |
| FEVER | TF-IDF+LR (baseline) | 69.94% | 69.92% | 69.16% |
| FEVER | **ARPD (ours)** | **70.64%** | **70.52%** | **68.63%** |

> **Statistical significance**: F1-FAKE p=0.0021, F1-Macro p=0.0325 (5 random seeds, one-sided paired t-test)

---

## 🏗️ Kiến trúc hệ thống

```
INPUT CLAIM
      ↓
[1] Uncertainty Scorer    → k_adaptive ∈ {1,2,3,4,5}
      ↓
[2] Adaptive Retriever    → k Wikipedia passages
      ↓
[3] Paraphrase Augmentor  → (Training only) augmented samples
      ↓
[4] MiniLM Encoder        → 384-dim vector
    [speaker][subject] claim [SEP] evidence
      ↓
[5] Ensemble Classifier   → FAKE / REAL
    0.90 × TF-IDF+LR + 0.10 × ImprovedMLP
```

---

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- RAM 8GB+ (16GB khuyến nghị)
- Dung lượng ổ cứng: 3GB+ (cho wiki corpus)
- GPU: Không bắt buộc (chạy được CPU)

### Bước 1 — Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/ARPD-fake-news-detection.git
cd ARPD-fake-news-detection
```

### Bước 2 — Tạo môi trường ảo

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (macOS/Linux)
source venv/bin/activate
```

### Bước 3 — Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4 — Download LIAR dataset

```bash
# Tạo thư mục data
mkdir -p data/raw

# Download thủ công từ link sau và đặt vào data/raw/
# Link: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
# Hoặc mirror: https://github.com/thiagorainmaker77/liar_dataset
```

Sau khi download, giải nén:
```bash
cd data/raw
unzip liar_dataset.zip
cd ../..
```

Kiểm tra cấu trúc:
```
data/raw/
├── train.tsv
├── valid.tsv
├── test.tsv
└── README
```

---

## 💻 Chạy thực nghiệm

### Option A — Google Colab (Khuyến nghị, miễn phí)

1. Vào [colab.research.google.com](https://colab.research.google.com)
2. Upload file `notebooks/ARPD_Final_Experiments.ipynb`
3. Chọn **Runtime → Change runtime type → T4 GPU**
4. Chạy từng section theo thứ tự từ trên xuống

> **Lưu ý**: Notebook đã được chia thành 7 sections rõ ràng, mỗi section có comment giải thích.

### Option B — Chạy local

```bash
# Chạy baselines
python experiments/run_baselines.py

# Chạy ARPD với seed mặc định (42)
python experiments/run_experiment.py --seed 42

# Chạy với nhiều seeds để statistical significance
for seed in 42 123 456 789 2025; do
    python experiments/run_experiment.py --seed $seed
done

# Chạy không có Wikipedia retrieval (nhanh hơn để debug)
python experiments/run_experiment.py --seed 42 --no-retrieve
```

### Xem kết quả

```bash
# Kết quả được lưu trong thư mục results/
cat results/seed_42.json
```

---

## 📊 Tái tạo kết quả trong paper

Để tái tạo đầy đủ kết quả trong paper:

```bash
# Bước 1: Chạy 5 seeds cho statistical significance
for seed in 42 123 456 789 2025; do
    python experiments/run_experiment.py --seed $seed
done

# Bước 2: Tổng hợp kết quả
python experiments/aggregate_results.py

# Kết quả sẽ xuất ra results/FINAL_RESULTS.json
```

---

## 🧪 Chạy unit tests

```bash
# Chạy toàn bộ tests
pytest tests/ -v

# Chạy test cụ thể
pytest tests/test_pipeline.py::test_uncertainty_scorer -v
```

Kết quả mong đợi:
```
tests/test_pipeline.py::test_data_loader ✅ PASSED
tests/test_pipeline.py::test_uncertainty_scorer ✅ PASSED
tests/test_pipeline.py::test_encoder ✅ PASSED
tests/test_pipeline.py::test_classifier ✅ PASSED
tests/test_pipeline.py::test_augmentor ✅ PASSED
tests/test_pipeline.py::test_end_to_end ✅ PASSED
```

---

## 📁 Cấu trúc dự án

```
ARPD-fake-news-detection/
│
├── src/                          # Source code chính
│   ├── __init__.py
│   ├── data_loader.py            # Load và preprocess LIAR dataset
│   ├── uncertainty_scorer.py     # Tính k_adaptive từ linguistic features
│   ├── adaptive_retriever.py     # Query Wikipedia, filter evidence
│   ├── paraphrase_augmentor.py   # Synonym substitution augmentation
│   ├── encoder.py                # MiniLM encoder (22.7M params)
│   ├── classifier.py             # ImprovedMLP + ARPDTrainer
│   └── pipeline.py               # Kết hợp toàn bộ 5 components
│
├── experiments/                  # Scripts chạy thực nghiệm
│   ├── run_experiment.py         # Script chính: train + evaluate ARPD
│   └── run_baselines.py          # Chạy TF-IDF+LR, SVM, MiniLM+MLP
│
├── tests/                        # Unit tests
│   └── test_pipeline.py          # 25 tests cho tất cả components
│
├── notebooks/                    # Jupyter notebooks
│   └── ARPD_Final_Experiments.ipynb  # Notebook đầy đủ cho Colab
│
├── data/                         # Dữ liệu (không commit lên GitHub)
│   ├── raw/                      # LIAR dataset gốc
│   └── processed/                # Dữ liệu đã xử lý
│
├── results/                      # Kết quả thực nghiệm
│   └── FINAL_RESULTS.json        # Kết quả cuối cùng
│
├── requirements.txt              # Dependencies
├── README.md                     # File này
└── .gitignore                    # Bỏ qua data/, results/, etc.
```

---

## 📦 Dependencies chính

```
torch>=2.0.0
sentence-transformers>=2.2.0
transformers>=4.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
nltk>=3.8.0
wikipedia-api>=0.6.0
scipy>=1.11.0
pytest>=7.4.0
```

Xem đầy đủ trong `requirements.txt`.

---

## 🗄️ Datasets

### LIAR Dataset
- **Nguồn**: Wang (2017), ACL 2017
- **Link**: https://aclanthology.org/P17-2067/
- **Kích thước**: 12,836 claims từ PolitiFact.com
- **Nhãn**: 6 mức → binarize thành FAKE/REAL
- **Đặc điểm**: Có metadata speaker, subject, party

### FEVER Dataset
- **Nguồn**: Thorne et al. (2018), NAACL 2018
- **Link**: https://fever.ai/
- **Kích thước**: 185,445 claims từ Wikipedia
- **Nhãn**: SUPPORTS/REFUTES/NOT ENOUGH INFO → binarize
- **Đặc điểm**: Evidence Wikipedia annotate sẵn, coverage 99.2%

---

## 🔧 Sử dụng ARPD cho claim mới

```python
from src.pipeline import ARPDPipeline

# Khởi tạo pipeline
pipeline = ARPDPipeline.load('results/arpd_model.pt')

# Dự đoán một claim mới
claim = "The unemployment rate has never been lower."
speaker = "donald-trump"
subject = "economy"

result = pipeline.predict(claim, speaker=speaker, subject=subject)
print(f"Label: {result['label']}")        # FAKE hoặc REAL
print(f"Confidence: {result['prob']:.2%}") # Xác suất
print(f"Evidence: {result['evidence']}")   # Wikipedia evidence
```

---

## 📈 Kết quả chi tiết

### LIAR Dataset (Test set = 1,267 samples)

| Model | Accuracy | F1-Macro | F1-FAKE | F1-REAL |
|-------|----------|----------|---------|---------|
| TF-IDF + LR | 64.96% | 62.08% | 51.63% | 72.52% |
| TF-IDF + SVM | 63.30% | 60.19% | 49.07% | 71.31% |
| MiniLM + MLP | 61.33% | 61.15% | 58.54% | 63.76% |
| **ARPD (ours)** | **65.19%** | **62.59%** | **52.73%** | **72.45%** |

### Statistical Significance (5 seeds)

| Metric | ARPD | TF-IDF+LR | p-value |
|--------|------|-----------|---------|
| F1-Macro | 0.6264 ± 0.0045 | 0.6208 ± 0.0000 | **p=0.0325** ✅ |
| F1-FAKE | 0.5337 ± 0.0120 | 0.4984 ± 0.0000 | **p=0.0021** ✅ |

### Ablation Study

| Variant | F1-Macro | Drop |
|---------|----------|------|
| Full ARPD | 62.59% | — |
| Bỏ Ensemble | 61.33% | -1.26% |
| Bỏ Speaker Context | 59.98% | -2.61% |
| Bỏ Retrieval | 62.56% | -0.03% |

---

## 🙏 Acknowledgements

- LIAR dataset: Wang (2017), ACL 2017
- FEVER dataset: Thorne et al. (2018), NAACL 2018
- MiniLM encoder: Reimers & Gurevych (2019), EMNLP 2019
- Adversarial augmentation insight: Park et al. (2025), WWW 2025
- RAG gap identification: Ferraz et al. (2026), PROPOR 2026
