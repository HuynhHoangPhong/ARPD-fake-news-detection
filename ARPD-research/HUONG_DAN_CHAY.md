# Hướng dẫn chạy thực nghiệm đầy đủ (ARPD-research)

Tài liệu này mô tả từng bước để chạy ma trận thực nghiệm 5-seed trên Google Colab,
từ rebuild evidence cache đến tổng hợp kết quả cuối.

---

## Bước 0: Chuẩn bị môi trường (Colab)

```python
# Cell 1: Clone repo và cài dependencies
!git clone <repo_url> ARPD-research
%cd ARPD-research
!pip install -q sentence-transformers scikit-learn tqdm wikipedia-api pandas numpy torch scipy
```

---

## Bước 1: Rebuild evidence cache (chỉ cần làm 1 lần)

**Thời gian ước tính (đo thực tế trên LIAR dataset):**
- `--sleep 0.5` (mặc định, an toàn): ~10-15 giờ trên Colab T4
- `--sleep 0.1` (nhanh hơn, có thể bị rate-limit): ~3-5 giờ trên Colab T4

**Fill rate kỳ vọng:** 60-80% claims có ít nhất 1 passage (đo được 65% trên 20-claim probe)

```bash
# Build val + test trước (~30 phút) để kiểm tra pipeline
python build_evidence_cache.py --splits val test --sleep 0.1

# Build train (dài nhất — chạy qua đêm)
python build_evidence_cache.py --splits train --sleep 0.1
```

**Nếu bị ngắt giữa chừng:** script tự resume từ checkpoint gần nhất (ghi mỗi 50 claims).
Chỉ cần chạy lại đúng lệnh trên, không cần `--no-resume`.

**Sau khi xong, verify:**
```python
import pandas as pd
for split, n in [('train', 10240), ('val', 1284), ('test', 1267)]:
    df = pd.read_csv(f'cached_{split}_evidence.csv')
    fill = (df['retrieved_evidence'].fillna('').str.strip() != '').sum()
    print(f'{split}: {len(df)}/{n} rows, fill={fill}/{len(df)} ({100*fill/len(df):.1f}%)')
```

---

## Bước 2: Chạy baseline multi-seed

```bash
for seed in 42 123 456 789 2025; do
    python experiments/run_baseline.py --seed $seed
done
```

Kết quả append vào `results/baseline_results.csv`.

---

## Bước 3: Chạy ARPD multi-seed (ma trận đầy đủ)

Mỗi lần chạy mất ~15-30 phút (encoding + training + robustness).

```bash
# Cấu hình đầy đủ (Full ARPD): retrieve + speaker + ensemble + augmentation
for seed in 42 123 456 789 2025; do
    python experiments/run_arpd.py \
        --augmentation synonym \
        --seed $seed \
        --epochs 20
done

# Ablation: không retrieve (để đo contribution của retrieval)
for seed in 42 123 456 789 2025; do
    python experiments/run_arpd.py \
        --augmentation synonym \
        --no-retrieve \
        --seed $seed \
        --epochs 20
done

# Ablation: không speaker context
for seed in 42 123 456 789 2025; do
    python experiments/run_arpd.py \
        --augmentation synonym \
        --no-speaker \
        --seed $seed \
        --epochs 20
done

# Ablation: không ensemble
for seed in 42 123 456 789 2025; do
    python experiments/run_arpd.py \
        --augmentation synonym \
        --no-ensemble \
        --seed $seed \
        --epochs 20
done
```

---

## Bước 4: Tổng hợp kết quả và kiểm định thống kê

```bash
python experiments/aggregate_results.py
```

Output gồm:
- `results/significance_report.csv` — F1 trung bình ± std, p-value vs baseline
- Bảng in ra console với paired one-sided t-test (H1: ARPD > baseline)

**Đọc kết quả:**
- `p_ttest_onesided < 0.05` → reject H0, ARPD có cải thiện đáng kể
- `p_ttest_onesided >= 0.05` → không đủ bằng chứng; báo cáo Outcome B (negative/neutral result)

---

## Bước 5: Điền vào RESULTS_SUMMARY.md

Sau khi có số liệu từ aggregate, cập nhật bảng trong `RESULTS_SUMMARY.md` với:
- F1-macro trung bình (5 seed) cho mỗi cấu hình
- p-value từ paired t-test
- Ghi rõ fill rate thực tế của evidence cache

---

## Kiểm tra nhanh sau mỗi bước

```bash
# Sau bước 2:
python -c "import pandas as pd; df=pd.read_csv('results/baseline_results.csv'); print(df[['model','seed','test_f1_macro']])"

# Sau bước 3:
python -c "import pandas as pd; df=pd.read_csv('results/arpd_results.csv'); print(df[['augmentation','retrieve','use_speaker','use_ensemble','seed','ensemble_weight','test_f1_macro']])"

# Kiểm tra không có NaN:
python -c "
import pandas as pd
df = pd.read_csv('results/significance_report.csv')
assert df['f1_macro_mean'].isna().sum() == 0, 'NaN in significance report!'
print(df.to_string(index=False))
"
```

---

## Lưu ý khoa học

- Hyperparameter (`ensemble_weight`) LUÔN được chọn trên VAL set, không bao giờ trên test set
- Báo cáo cả Outcome A (ARPD tốt hơn) lẫn Outcome B (không tốt hơn) — không giấu kết quả âm
- Evidence fill rate phải được ghi trong paper (không phải 100%)
- Baseline TF-IDF+LR chạy trên text thô (không có retrieved evidence) để so sánh công bằng
