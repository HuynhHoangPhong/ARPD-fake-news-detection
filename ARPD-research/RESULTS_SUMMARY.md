# Results Summary

## Critical finding: all existing results use broken retriever (97.5% empty caches)

The three evidence cache files (`cached_train_evidence.csv`, `cached_val_evidence.csv`,
`cached_test_evidence.csv`) were built on Google Colab (notebook `ARPD1.2-share.ipynb`,
Cell 5) **before** the User-Agent header fix. Wikipedia returned HTTP 403 to requests
without a User-Agent, which was silently caught as an empty result. Measured fill rates:

| Split | Rows | Non-empty evidence | Fill rate |
|---|---|---|---|
| train | 10,240 | 253 | **2.5%** |
| val | 1,284 | 31 | **2.4%** |
| test | 1,267 | 36 | **2.8%** |

**Consequence:** Every Colab result that used these caches ("Full ARPD", "No-Aug + Full-Ret",
etc.) was functionally running in zero-evidence mode. None of those numbers represent
genuine evidence-augmented classification.

---

## Colab results (broken retriever — archived for reference only)

These numbers are from `results/*_before_retrieval_fix.csv`. Do NOT cite them as
"ARPD with retrieval" results — they are mislabeled zero-evidence runs.

### Baselines (valid — no retrieval used)

| Model | Test Acc | Test Macro-F1 |
|---|---|---|
| TF-IDF + LR | 0.6369 | 0.6317 |
| Frozen DistilRoBERTa + MLP | 0.6235 | 0.6211 |

### ARPD ablation — Cell 8 (invalid — retriever broken)

| Variant | Test Acc | Test Macro-F1 | Note |
|---|---|---|---|
| No-Aug + Full-Ret | 0.6377 | 0.6338 | Cache 97.5% empty → actually no-retrieval |
| Full-Aug + No-Ret | 0.6188 | 0.6094 | Synonym augmentation only |
| No-Aug + No-Ret | 0.6298 | 0.6255 | Frozen MiniLM + MLP, no aug, no ret |

### Multi-seed trial — Cell 10 (invalid — retriever broken)

| Seed | Naive Baseline F1 | Full ARPD F1 |
|---|---|---|
| 42 | 0.6098 | 0.6193 |
| 1337 | 0.6158 | 0.5982 |
| 2026 | 0.6107 | 0.6118 |
| **Mean ± Std** | **0.6121 ± 0.0032** | **0.6098 ± 0.0107** |

Paired t-test: T = -0.2934, **p = 0.797** → NOT significant.

**Interpretation:** "Naive Baseline" here is no-aug + no-ret (frozen MiniLM + MLP).
"Full ARPD" is synonym-aug + cache that is 97.5% empty. The comparison is
essentially "synonym augmentation" vs "no augmentation" — not evidence retrieval.
The p=0.797 is meaningless as an evidence-retrieval significance test.

---

## Post-fix comparison table (to be filled in after rebuilding caches)

**The caches must be rebuilt** using the fixed `build_evidence_cache.py` (which has the
User-Agent header fix). Expected fill rate after fix: ~70–80% based on Phase 2 sample.

Run on Colab (Tesla T4 recommended) — estimated time: ~2.5 hours for train split.

```bash
# In Colab, after cloning the repo with the fix applied:
python build_evidence_cache.py --splits train val test --sleep 0.5
```

Then fill in this table (run across 5 seeds: 42, 123, 456, 789, 2025):

| Model | Clean F1 (mean ± std) | Adv F1 @p=0.30 | p-value vs best baseline |
|---|---|---|---|
| TF-IDF + LR | — | — | — |
| Frozen DistilRoBERTa + MLP | — | — | — |
| ARPD: No-Aug + No-Ret | — | — | — |
| ARPD: Aug-Only + No-Ret | — | — | — |
| ARPD: No-Aug + Full-Ret | — | — | — |
| ARPD: Full (Aug + Ret) | — | — | — |

---

## Interim verdict (before cache rebuild)

Based on existing data and Phase 2 diagnosis:

The LIAR dataset has a structural limitation for Wikipedia-based evidence retrieval:
claims are short, decontextualised political quotes. Wikipedia summaries address
the general topic but rarely the specific statistic being fact-checked. Even with
a working retriever at 70–80% fill rate, it is unclear whether evidence passages
will add a meaningful signal beyond what the frozen encoder already captures.

**Two honest outcomes remain possible:**

**Outcome A** — After proper cache rebuild, "No-Aug + Full-Ret" (0.6338 with broken
cache) may improve further with real 70–80% evidence. If it significantly beats
TF-IDF (0.6317) and Frozen DistilRoBERTa+MLP (0.6211) at p < 0.05 across 5
seeds, the thesis claim is supportable.

**Outcome B** — Even with working retrieval, gains remain small and insignificant
(p > 0.05). In this case the contribution is a rigorously-measured negative result
plus a clear hypothesis: LIAR claims are too decontextualised for
Wikipedia-summary-level evidence. The natural follow-up is FEVER (where claims
are Wikipedia-grounded and retrieval can directly verify/contradict the claim).

---

## How to run Phase 2 experiments on Colab

### Step 1: Push the User-Agent fix to GitHub (already done on `feature/offline-cache`)

Make sure the branch includes:
- `src/adaptive_retriever.py` — User-Agent header in `_fetch_passages()`
- `src/adaptive_retriever.py` — `_CANDIDATE_SRLIMIT = 10` over-fetch strategy
- `build_evidence_cache.py` — resume-capable cache builder

### Step 2: Rebuild caches in Colab

```python
# In Colab, replace Cell 5 with:
import subprocess
result = subprocess.run(
    ['python', 'build_evidence_cache.py', '--splits', 'train', 'val', 'test', '--sleep', '0.5'],
    capture_output=True, text=True
)
print(result.stdout)
```

### Step 3: Run the full 5-seed ablation matrix

```bash
# From ARPD-research/ directory (after cache rebuild):
SEEDS="42 123 456 789 2025"

for seed in $SEEDS; do
    # Ablation: no augmentation, no retrieval (MLP-only baseline)
    python experiments/run_arpd.py --augmentation none --no-retrieve --no-ensemble --seed $seed

    # Ablation: augmentation only, no retrieval
    python experiments/run_arpd.py --augmentation synonym --no-retrieve --no-ensemble --seed $seed

    # Ablation: no augmentation, full retrieval
    python experiments/run_arpd.py --augmentation none --no-ensemble --seed $seed

    # Full ARPD: augmentation + retrieval + ensemble + speaker context
    python experiments/run_arpd.py --augmentation synonym --seed $seed
done

# Aggregate and compute significance
python experiments/aggregate_results.py
```

ARPD-specific flags for Colab: pass `--seed`, `--augmentation`, `--no-retrieve`,
`--no-speaker`, `--no-ensemble` as needed. Results auto-appended to `results/arpd_results.csv`.
