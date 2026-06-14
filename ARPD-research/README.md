# ARPD: Adaptive Evidence Retrieval with Paraphrase-Robust Training for Lightweight Fake News Detection

## Overview

ARPD là một pipeline fake news detection nhẹ (22M params encoder) kết hợp:
- **Adaptive retrieval**: số Wikipedia passages thay đổi theo uncertainty score của claim
- **Paraphrase-robust training**: synonym substitution augmentation để chống style-conversion attacks
- **Lightweight**: toàn bộ chạy được trên Google Colab T4 (free tier)

## Architecture

```
Claim
  │
  ├─► UncertaintyScorer ──► k_adaptive (1-5)
  │       │
  │   AdaptiveRetriever ──► k Wikipedia passages
  │       │
  │   ClaimEvidenceEncoder
  │   [v_claim | v_evidence | |v_claim - v_evidence|] (1152-dim)
  │       │
  └─► ARPDClassifier (2-layer MLP) ──► FAKE / REAL
```

## Dataset

**LIAR** (Wang, 2017): 12,800 political claims từ PolitiFact  
Binarization: `{true, mostly-true, half-true}` → REAL; `{barely-true, false, pants-fire}` → FAKE

## Quick Start

### Local

```bash
pip install -r requirements.txt

# Load & preprocess data
python src/data_loader.py

# Chạy baselines
python experiments/run_baseline.py

# Chạy ARPD (không retrieval để test nhanh)
python experiments/run_arpd.py --no-retrieve --epochs 10

# Chạy ARPD với retrieval (chậm hơn)
python experiments/run_arpd.py --epochs 20

# Đánh giá robustness
python experiments/evaluate_robustness.py
```

### Google Colab

Mở `notebooks/ARPD_Colab.ipynb` trên Colab, chọn Runtime → T4 GPU, chạy từng cell.

## Results (Expected)

| Model | Test Acc | F1-Macro |
|---|---|---|
| TF-IDF + LR | ~0.62 | ~0.61 |
| DistilBERT + MLP | ~0.67 | ~0.66 |
| ARPD (no retrieve) | ~0.69 | ~0.68 |
| ARPD (full) | ~0.72 | ~0.71 |

*Số liệu chính thức sau khi chạy thật sẽ được cập nhật vào results/*

## Constraints

- Encoder: 22M params (all-MiniLM-L6-v2)
- Baselines: ≤125M params
- Không dùng paid API
- Compatible: Python 3.10+, Google Colab free tier

## Project Structure

```
ARPD-research/
├── src/
│   ├── data_loader.py          # Load + binarize LIAR
│   ├── uncertainty_scorer.py   # Entropy-based uncertainty
│   ├── adaptive_retriever.py   # Wikipedia evidence retrieval
│   ├── paraphrase_augmentor.py # Synonym sub + back-translation
│   ├── encoder.py              # MiniLM claim+evidence encoder
│   ├── classifier.py           # 2-layer MLP + training loop
│   └── pipeline.py             # End-to-end ARPD
├── experiments/
│   ├── run_baseline.py
│   ├── run_arpd.py
│   └── evaluate_robustness.py
├── notebooks/
│   └── ARPD_Colab.ipynb
└── results/
```
