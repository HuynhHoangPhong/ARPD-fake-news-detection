# Limitations

This document records the known limitations of the ARPD system honestly, for
thesis reviewers and future researchers.

---

## 1. Residual empty-retrieval rate

Even after the User-Agent header fix (which was the primary cause of 97–98%
empty evidence in cached results), some claims remain unanswerable from
Wikipedia summaries. In a 20-claim validation sample after the fix, 6/20
(30%) still returned no evidence above the similarity threshold (0.25). These
are typically hyper-local political claims (e.g. county-level tax or spending
claims) for which Wikipedia has no dedicated article.

**Expected full-dataset fill rate:** ~70–80% of claims will get at least one
evidence passage; ~20–30% will remain zero-evidence and fall back to
claim-only encoding. This is a meaningful limitation because zero-evidence
claims effectively reduce ARPD to the no-retrieval baseline for that subset.

---

## 2. The uncertainty scorer is a hand-written heuristic, not a learned model

The `UncertaintyScorer` computes `k_adaptive` using six linguistic features
(token length, presence of numbers/percentages/dollars, hedge words, named
entity proxy via capitalisation, vague quantifiers, short-sentence penalty)
combined with hand-chosen weights (0.20, 0.25, 0.15, 0.15, 0.15, 0.10). These
coefficients were chosen by the authors based on intuition, not fit to data.

The scorer is described in the codebase as a deliberate design choice (the
embedding-based entropy alternative collapsed to k=5 for 88% of claims because
LIAR claims cluster tightly in MiniLM space), but it should be understood as a
rule-based heuristic, not a trained uncertainty model. It cannot be cited as
evidence of learned adaptive behaviour.

---

## 3. Pre-fix results (archived in results/*_before_retrieval_fix.csv) are unreliable

All results generated before June 2026 (including `arpd_results.csv`,
`arpd_ablation_results.csv`, `arpd_scientific_trials.csv`) were produced
**without a working retriever**. A missing `User-Agent` header in
`AdaptiveRetriever._fetch_passages()` caused Wikipedia's API to return HTTP
403 responses, which were silently caught and returned as empty passage lists.
The effective retrieval rate was ~0–2.5%.

As a result:
- "Full ARPD" in those results was functionally identical to "No-Retrieval ARPD".
- Any comparison between "with retrieval" and "without retrieval" in pre-fix
  results is meaningless.
- The pre-fix numbers are archived for reference but should **not** be cited
  as evidence of the pipeline's performance.

---

## 4. Seeds where ARPD underperformed baselines (pre-fix)

In `arpd_scientific_trials_before_retrieval_fix.csv`, on seed 1337 the
ARPD pipeline (F1 = 0.598) was outperformed by the naive TF-IDF baseline
(F1 = 0.616). Since retrieval was broken in all pre-fix runs, this difference
reflects variance in MLP training across seeds, not a retrieval effect. With
a working retriever, this comparison needs to be re-run.

---

## 5. Evidence relevance vs. LIAR claim structure

LIAR claims are short, decontextualised quotes from politicians
(e.g. "We have less Americans working now than in the 70s."). Wikipedia
evidence can address the general topic (employment trends, US economy) but
rarely the specific cited statistic. This means retrieved passages provide
background knowledge rather than direct fact-check evidence, which may limit
the downstream classification gain.

This is a dataset-level limitation: ARPD may work better on claims that
reference specific checkable propositions (like FEVER's Wikipedia-grounded
claims) than on LIAR's rhetorical/statistical assertions. See RESULTS_SUMMARY.md
for the recommended framing.

---

## 6. Synonym substitution quality

Even with the POS-tagging fix and blocklist, `synonym_substitute` uses
context-free WordNet lookups. A word like "bill" in "Congress passed a bill"
may still substitute to a beak-related synonym ("nib", "peak") because the
system cannot disambiguate the legislative sense from the anatomical sense
without full context. The POS fix eliminates cross-POS hallucinations (e.g.
the noun "ampere" replacing the adjective "major") and the blocklist removes
vulgar terms, but within-POS semantic drift remains.

---

## 7. No end-to-end fine-tuning of the encoder

The MiniLM-L6-v2 encoder is used with frozen weights. Only the downstream
ImprovedMLP classifier (~190K parameters) is trained. End-to-end fine-tuning
of the encoder on LIAR could improve performance but was excluded due to
compute constraints (fine-tuning 22M params on CPU is impractical).

---

## 8. Single-dataset evaluation

All results are on LIAR (Wang 2017). Generalisation to other fake-news
datasets (FEVER, FakeNewsNet, PHEME) is not demonstrated.
