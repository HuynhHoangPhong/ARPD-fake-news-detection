
ARPD là một pipeline fake news detection nhẹ (22M params encoder) kết hợp:
- **Adaptive retrieval**: số Wikipedia passages thay đổi theo uncertainty score của claim
- **Paraphrase-robust training**: synonym substitution augmentation để chống style-conversion attacks
- **Lightweight**: toàn bộ chạy được trên Google Colab T4 (free tier)



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
