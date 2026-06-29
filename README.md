# ARPD: Adaptive Retrieval-and-Paraphrase Defense for Fake News Detection

**Status:** Complete experimental cycle (post retrieval-fix) — final results, with effect sizes, multiple-comparisons correction, and a power analysis, not just raw significance flags.
**Dataset:** LIAR (Wang, 2017) — 12,791 fact-checked political statements.
**Runtime environment:** Google Colab (Tesla T4, free tier).
**One-line summary of the finding:** model ensembling is a confirmed, large, statistically robust contributor; Wikipedia-style retrieval and speaker-name conditioning are confirmed to _not_ help (and show a measurable negative effect) on this dataset; the comparison against simple baselines is genuinely inconclusive due to low statistical power at n=5 seeds, not evidence of "no difference." See Section 13.0 for the full claim-by-claim accounting.

---

## Table of Contents

1. [Summary & Motivation](#1-summary--motivation)
2. [Research Hypothesis](#2-research-hypothesis)
3. [Related Work](#3-related-work)
4. [Dataset](#4-dataset)
5. [System Architecture](#5-system-architecture)
6. [Implementation Details by Component](#6-implementation-details-by-component)
7. [Experimental Design](#7-experimental-design)
8. [A Critical Bug Found and Fixed](#8-a-critical-bug-found-and-fixed)
9. [Results](#9-results)
10. [Statistical Significance Testing](#10-statistical-significance-testing)
11. [Robustness Evaluation](#11-robustness-evaluation)
12. [Discussion & Interpretation](#12-discussion--interpretation)
13. [Contributions of This Work](#13-contributions-of-this-work)
14. [Limitations](#14-limitations)
15. [Future Work](#15-future-work)
16. [Codebase Structure](#16-codebase-structure)
17. [Reproducing the Experiments](#17-reproducing-the-experiments)
18. [References](#18-references)

---

## 1. Summary & Motivation

Modern fake-news detection systems built on neural language models tend to share two structural weaknesses:

1. **Lack of external grounding.** Most classifiers learn to predict veracity purely from the surface style and vocabulary of a statement, without checking it against any external knowledge. This leaves them exposed to statements that are factually wrong but _sound_ credible.
2. **Lack of paraphrase robustness.** A false claim can be re-worded with synonyms or restructured sentences while remaining false — a model that only memorizes surface patterns may fail to recognize the paraphrased version as the same claim.

**ARPD (Adaptive Retrieval-and-Paraphrase Defense)** is designed to address both weaknesses jointly by combining:

- **Adaptive retrieval:** for each claim, the system estimates a linguistic _uncertainty score_ to decide how many Wikipedia passages to retrieve as supporting evidence — vague or hedged claims pull in more evidence, specific/numeric claims pull in less (or none).
- **Paraphrase-robust augmentation:** during training, every claim is duplicated with a synonym-substituted version (WordNet-based), forcing the encoder to learn a representation that is not overly tied to exact wording.
- **Speaker-aware context:** the speaker and subject metadata available in LIAR are prepended to the claim text, since the credibility of a statement is, in practice, partly conditioned on _who_ said it.
- **Two-model ensemble:** a lexical model (TF-IDF + Logistic Regression) is combined with a semantic model (frozen sentence-encoder embeddings + MLP), so the system can exploit both surface lexical cues and deeper semantic signal simultaneously.

The entire pipeline is deliberately lightweight, runnable on free-tier infrastructure: the encoder has only 22M parameters (MiniLM-L6-v2), is never fine-tuned, and only a small MLP head (~190K parameters) is trained on top of frozen embeddings.

---

## 2. Research Hypothesis

> **H1:** On the LIAR dataset, a pipeline combining adaptive Wikipedia evidence retrieval, paraphrase augmentation, speaker-aware context, and ensembling ("Full ARPD") achieves a statistically significantly higher F1-macro (p < 0.05, one-sided paired t-test) than two simple baselines: TF-IDF + Logistic Regression, and a frozen sentence-encoder + MLP.

The experimental design was set up to yield two equally informative, honest outcomes:

- **Outcome A:** Full ARPD significantly outperforms both baselines → H1 is supported.
- **Outcome B:** Full ARPD does _not_ significantly outperform the baselines → a rigorously measured negative result, made informative by an ablation study that isolates which sub-component (if any) actually contributes value.

**A design limitation acknowledged up front:** five seeds per configuration was chosen for computational tractability on free-tier hardware (Section 13.5), not from an a-priori power analysis. As Section 10.4 quantifies retrospectively, this gives the study strong power to detect only large effects (Cohen's _d_ ≳ 1.3) and materially under-powers the comparison against the two simple baselines for the small-to-medium effect sizes actually observed. This is stated here, before the results are presented, so that the baseline comparison in Section 9–10 is read with the correct expectations from the outset, rather than as a post-hoc excuse for a null result.

---

## 3. Related Work

ARPD sits at the intersection of three research threads: evidence-based fact verification, adaptive retrieval-augmented generation, and adversarial/paraphrase robustness for misinformation detection. This section situates the design choices of ARPD against that literature.

### 3.1. Evidence-based fact verification and fake-news detection

The idea of grounding a veracity judgment in retrieved external evidence rather than the claim text alone goes back to early evidence-based approaches such as **DeClarE** (Popat et al., 2018), which jointly represents news content together with retrieved evidence articles, and to attention-based architectures that learn news–evidence interactions, e.g. **HAN** (Ma et al., 2019) and **EHIAN** (Wu et al., 2021). More recent work pushes this further with multi-step retrieval: **MUSER** (Liao et al., 2023) retrieves Wikipedia evidence in an iterative, multi-round process guided by a relevance threshold, mirroring how a human fact-checker would re-query when the first batch of evidence is insufficient — conceptually close to ARPD's adaptive _k_, but operating at the level of _iterative_ retrieval rounds rather than a single up-front uncertainty estimate. **GERE** (Chen et al., 2022) instead frames evidence retrieval generatively, learning to directly generate the title and sentence identifiers of supporting Wikipedia evidence. **RoE-FND** (2025) frames evidence usage as case-based reasoning over a self-built knowledge base. On the LLM side, **Re-Search** (2024) shows that multi-round, retrieval-augmented LLMs can outperform single-shot retrieval pipelines on real-world fake-news datasets by issuing follow-up queries when evidence is judged insufficient — the same underlying intuition (variable evidence budget per claim) that motivates ARPD's `UncertaintyScorer`, but implemented with an LLM-driven controller instead of a fixed linguistic heuristic.

ARPD differs from this family of work in scale and ambition: rather than an LLM-driven, multi-round retrieval controller, it uses a single up-front, rule-based uncertainty estimate to fix a retrieval budget _k_ ∈ {1..5}, retrieves once, and feeds the result into a small frozen-embedding classifier. This is a much lighter-weight design point, explicitly chosen to remain trainable on free-tier hardware, at the cost of giving up the iterative refinement loop that MUSER, RoE-FND, and Re-Search rely on for evidence sufficiency.

### 3.2. Evidence retrieval pipelines and the FEVER benchmark

A large share of the evidence-retrieval literature is built and evaluated on **FEVER** (Thorne et al., 2018), a fact-verification benchmark in which every claim is, by construction, _generated from_ a Wikipedia sentence and is therefore directly checkable against that same sentence. **BERT-based retrieval-and-verification pipelines** (Soleimani et al., 2020) report a sentence-retrieval recall of 87.1 and a FEVER score of 69.7 on this benchmark by training two separate BERT models for evidence retrieval and claim verification respectively. ARPD's evidence pipeline is architecturally much simpler (TF-IDF keyword extraction → Wikipedia Search API → batch extract → cosine-similarity filtering, with no learned retriever), and — critically — is evaluated on **LIAR**, whose claims are short, decontextualized political quotes _not_ generated from Wikipedia sentences. As discussed in Sections 12 and 14, this dataset-level mismatch (FEVER claims are Wikipedia-grounded by construction; LIAR claims are not) is one of the central explanatory hypotheses for why retrieval does not help in this work's results, and is exactly the contrast the Future Work section (§15) proposes to test directly by re-running ARPD on FEVER.

### 3.3. Adaptive retrieval depth

ARPD's core idea — let the _amount_ of retrieved evidence vary per input instead of using a fixed top-_k_ — mirrors a broader and currently very active line of work in retrieval-augmented generation (RAG) for LLMs, generally referred to as **Adaptive-RAG**. **Jeong et al. (2024)**, in the paper that the term is most associated with, train a small classifier to route a query to no-retrieval, single-step, or iterative multi-step retrieval based on predicted _query complexity_ — conceptually the same goal as ARPD's `UncertaintyScorer` (route to more or less retrieval based on an estimate of how much external help the input needs), but realized with a _learned_ classifier rather than a hand-written linguistic heuristic. Other recent work in this space includes dynamic context-size classifiers that predict the optimal number of passages per query (addressing the same "fixed top-_k_ is suboptimal" problem ARPD targets), and dynamic passage-selection methods that pick a _minimal sufficient_ evidence set instead of a fixed-size one. **Self-RAG** (Asai et al., 2023) takes a related but distinct approach, training a single LM to decide _on-demand_, token by token, whether retrieval is needed at all, via special reflection tokens, rather than fixing a retrieval budget up front per input.

ARPD's `UncertaintyScorer` is a much smaller-scale, fully rule-based instance of this same idea: it is not learned, requires no training data or labels, and is restricted to five interpretable linguistic features (sentence length, presence of numeric/percentage/dollar tokens, hedging words, capitalization-based entity proxy, and vague-quantifier words) rather than a trained classifier or LLM-based complexity estimator. The explicit motivation recorded in the codebase for _not_ using a learned/embedding-based uncertainty estimate is also documented in Section 6.1 and Section 14.2: an embedding-distance-based alternative was tried first and collapsed to _k_=5 for 88% of LIAR claims, because short political claims cluster too tightly in MiniLM's embedding space to produce a meaningfully separated uncertainty signal — a finding broadly consistent with the general observation in the Adaptive-RAG literature that naive embedding-based complexity signals are often unreliable, motivating learned or rule-based alternatives instead.

### 3.4. Speaker / metadata-aware fake-news detection on LIAR

LIAR was published together with a rich set of speaker metadata (job, party, state, historical truthfulness counts, venue), and a substantial line of follow-up work on this dataset specifically studies whether conditioning on that metadata improves detection. The original LIAR paper itself reports that combining a text-CNN with a separate metadata branch (speaker, party, state, job, and historical truthfulness counts processed through a Bi-LSTM) outperforms text-only models (Wang, 2017). Goldani, Momtazi, & Safabakhsh (2021) test a capsule-network architecture with each LIAR metadata field added individually, and find that conditioning on speaker **history** yields the model's best result, outperforming prior hybrid CNN and attention-LSTM baselines by 3.1 points on validation and 1.0 point on test F1. Jain, Kaliyar, Goswami, Narang, & Sharma (2021/2022), in **AENeT**, report an 11% accuracy improvement specifically attributable to incorporating the speaker's credit history into an attention-based architecture, identifying it as the single most informative metadata field among those tested. Trueman, Kumar, Narayanasamy, & Vidya (2021), in an attention-based convolutional BiLSTM model (**AC-BiLSTM**), similarly show that combining statement content with full speaker-context attention improves over text-only baselines on LIAR. The common thread across this literature is that the largest documented metadata gains come specifically from a speaker's **historical truthfulness record** (counts of past true/false statements), not merely their name, job title, or party affiliation in isolation.

This is directly relevant to interpreting ARPD's own ablation result for speaker context (Section 9, Section 12): ARPD's `use_speaker_context` only prepends the speaker's _name_ and the claim's _subject_ string as a textual prefix (`"[speaker] [subject] claim"`) — it does **not** encode the speaker's historical truthfulness counts that the literature above identifies as the actually load-bearing signal. The absence of a measurable improvement from speaker context in this work's ablation results (Section 9) is therefore plausibly explained, at least in part, by the fact that ARPD's speaker conditioning omits exactly the metadata field (credit history) that prior work isolates as responsible for most of the documented speaker-related gains, rather than necessarily implying that speaker information _in general_ is unhelpful for this task.

### 3.5. Paraphrase and adversarial robustness in misinformation detection

A growing body of work documents that fake-news and misinformation detectors are vulnerable to paraphrasing and synonym-substitution perturbations, motivating ARPD's training-time augmentation strategy. Nakamura et al. (2020) and Cui & Lee (2020) report substantial accuracy drops for transformer-based detectors under paraphrasing attacks; a recent survey on multilingual/multimodal misinformation detection explicitly lists synonym replacement (via multilingual WordNet) as a standard defensive augmentation technique, the same general strategy ARPD applies via `paraphrase_augmentor.py`. More sophisticated recent defenses go further than static WordNet substitution: **AdStyle** (2024) uses an LLM to iteratively generate style-conversion adversarial prompts targeted at the detector's current decision boundary, training against an evolving, model-aware adversary rather than a fixed perturbation distribution; **J-Guard** (2023) evaluates robustness under both character-level (Cyrillic homoglyph injection) and PLM-generated paraphrase attacks; and AdSent (2026) studies sentiment-targeted adversarial rewrites that preserve factual content while reframing affect. ARPD's augmentation (Section 6.3) is intentionally the simplest member of this family — fixed-probability, context-free WordNet synonym substitution, in the spirit of the general-purpose **EDA** augmentation framework (Wei & Zou, 2019), rather than an adversarially-optimized or LLM-driven perturbation strategy. This positions ARPD's robustness evaluation (Section 11) as a basic sanity check rather than a state-of-the-art adversarial robustness benchmark; Section 15 proposes upgrading toward the model-aware adversarial augmentation strategies surveyed here as a direct extension.

### 3.6. Summary of positioning

Relative to this literature, ARPD's specific contribution is not a new state-of-the-art retrieval or robustness mechanism in isolation, but a **lightweight, fully-reproducible integration** of adaptive-retrieval-style evidence budgeting, LIAR-specific speaker conditioning, and WordNet-based paraphrase augmentation into a single small pipeline runnable end-to-end on free-tier hardware — combined with an honest, statistically-tested ablation that reports which of these components actually moves the needle on this specific dataset (Section 9–10), rather than only reporting an aggregate "ours vs. baseline" number.

---

## 4. Dataset

**LIAR** (Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, ACL 2017) consists of short statements by U.S. politicians, each rated by PolitiFact on a 6-point truthfulness scale.

### 4.1. Label binarization

| Original label (6 classes)           | Binary label |
| ------------------------------------ | ------------ |
| `true`, `mostly-true`, `half-true`   | **REAL (1)** |
| `barely-true`, `false`, `pants-fire` | **FAKE (0)** |

### 4.2. Post-processing statistics

| Split      | Samples | REAL  | FAKE  | % REAL |
| ---------- | ------- | ----- | ----- | ------ |
| Train      | 10,240  | 5,752 | 4,488 | 56.2%  |
| Validation | 1,284   | 668   | 616   | 52.0%  |
| Test       | 1,267   | 714   | 553   | 56.4%  |

The dataset has a mild class imbalance toward REAL across all three splits; this is corrected via **automatic class weighting** in the MLP loss function (Section 6.5).

### 4.3. Per-sample fields

- `claim`: the statement text (short, ~15–20 tokens on average)
- `speaker`: the person who made the statement (e.g. "barack-obama")
- `subject`: the topic (e.g. "health-care", "taxes")
- `label`: binary label after mapping

---

## 5. System Architecture

```
                         ┌──────────────────────┐
                         │     Input claim       │
                         └──────────┬─────────────┘
                                    │
                  ┌─────────────────┼──────────────────┐
                  ▼                 ▼                  ▼
        ┌──────────────────┐ ┌─────────────┐  ┌──────────────────┐
        │ UncertaintyScorer│ │  Speaker +  │  │ ParaphraseAugmentor│
        │  -> k_adaptive   │ │  Subject    │  │ (train only):      │
        │  (1..5 passages) │ │  prefix     │  │ WordNet synonym sub │
        └─────────┬────────┘ └──────┬──────┘  └─────────┬─────────┘
                  ▼                 │                    │
        ┌──────────────────┐        │                    │
        │ AdaptiveRetriever│        │                    │
        │ (Wikipedia API)  │        │                    │
        │ -> k passages    │        │                    │
        └─────────┬────────┘        │                    │
                  │                 │                    │
                  ▼                 ▼                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │   ClaimEvidenceEncoder (MiniLM-L6-v2, 22M params,       │
        │   frozen)                                                │
        │   "[speaker] [subject] claim [SEP] evidence" -> R^384   │
        └───────────────────────────┬───────────────────────────────┘
                                    │
                ┌───────────────────┼────────────────────┐
                ▼                                        ▼
      ┌───────────────────┐                  ┌───────────────────────┐
      │ TF-IDF (1-3 gram)  │                  │     ImprovedMLP        │
      │ + LogisticReg.     │                  │  384->256->128->64->2 │
      │ on context text    │                  │  BatchNorm + Dropout  │
      │  P_lr(REAL)         │                  │  + residual           │
      └─────────┬───────────┘                  └───────────┬───────────┘
                │                                          │
                └───────────────► ENSEMBLE ◄────────────────┘
                       P_final = (1-w)*P_lr + w*P_mlp
                       (w grid-searched on the validation set)
                                    │
                                    ▼
                          Predicted label: FAKE / REAL
```

The pipeline is implemented in `ARPDPipeline` (`src/pipeline.py`), with every component **independently toggleable** through configuration flags — this is the technical basis for the ablation study in Section 7.2.

---

## 6. Implementation Details by Component

### 6.1. Uncertainty Scorer — deciding "how much evidence is needed"

File: `src/uncertainty_scorer.py`

**Design rationale:** an initial embedding-based entropy approach (k-nearest-neighbor similarity in MiniLM space) was tried and **discarded**, because LIAR claims are short and politically domain-specific, causing them to cluster too tightly in embedding space — entropy was nearly uniform across claims, and the resulting _k_ collapsed to 5 for 88% of inputs, defeating the purpose of "adaptivity."

Instead, `UncertaintyScorer` uses **five hand-engineered linguistic features**, requiring no training:

| Feature        | Formula                                                       | Rationale                                                                       |
| -------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `length_score` | `min(token_count / 25, 1.0)`                                  | Longer sentences tend to carry more claims to verify                            |
| `specificity`  | mean of (has number, has %, has $)                            | Claims with concrete figures are more directly checkable, needing less evidence |
| `hedge_score`  | count of hedge words (allegedly, reportedly, might, …)        | Hedged claims are more ambiguous about their source, needing more evidence      |
| `entity_score` | count of mid-sentence capitalized tokens (named-entity proxy) | Concrete named entities are easier to look up                                   |
| `vague_score`  | count of vague quantifiers (many, several, often, …)          | Vague quantification needs more supporting evidence                             |

Combined formula:

```
uncertainty = 0.20*length_score + 0.25*(1-specificity) + 0.15*hedge_score
            + 0.15*(1-entity_score) + 0.15*vague_score + 0.10*short_penalty
```

Mapped to the number of passages to retrieve (`k_adaptive`):

| uncertainty | k   |
| ----------- | --- |
| < 0.20      | 1   |
| 0.20 – 0.35 | 2   |
| 0.35 – 0.50 | 3   |
| 0.50 – 0.65 | 4   |
| ≥ 0.65      | 5   |

**Important scientific caveat:** this is a **hand-written, rule-based heuristic, not a trained model** — the feature weights (0.20, 0.25, 0.15…) were chosen by intuition, not fit to data. This should not be cited as evidence of "learned adaptive behavior" (see also Section 3.3 and Section 14.2). It is a deliberately lightweight, interpretable alternative to the learned complexity classifiers used in the Adaptive-RAG literature.

### 6.2. Adaptive Retriever — fetching evidence from Wikipedia

File: `src/adaptive_retriever.py`

The evidence-retrieval pipeline for each claim has four steps:

1. **Keyword extraction:** TF-IDF (English stop-words removed) extracts the 5 most informative keywords from the claim, used as the search query.
2. **Wikipedia search (Search API):** a single call to `en.wikipedia.org/w/api.php` (action=`query`, list=`search`) retrieves up to 10 candidate article titles (`srlimit=10`).
3. **Batch content extraction (Extract API):** instead of issuing one API call per article (high rate-limit risk), the system sends **a single batched request** with all 10 titles (`titles="A|B|C|..."`, `prop=extracts`, `exintro=1`, `explaintext=1`) to fetch the plaintext intro summary of all candidate articles in one network round-trip.
4. **Chunking & similarity filtering:** each intro summary is split into 100-word chunks, embedded with MiniLM, and scored by cosine similarity against the original claim; only chunks with similarity ≥ 0.25 are kept, up to the `k` requested by the Uncertainty Scorer.

Every request includes a `User-Agent` header compliant with the Wikimedia Foundation's API etiquette policy — this exact detail was the source of a serious bug, documented in Section 8.

### 6.3. Paraphrase Augmentor — training-time robustness

File: `src/paraphrase_augmentor.py`

During training (not applied at test time, except specifically for the robustness evaluation in Section 11), every training claim is duplicated: the original is kept, and a synonym-substituted version is added.

- For each word, there is a `p=0.15` probability of substitution with a WordNet synonym.
- Substitution is **part-of-speech constrained** — nouns are only replaced with nouns, verbs with verbs, etc. — to avoid ungrammatical replacements.
- A **blocklist** removes vulgar synonyms and a few observed "degenerate" WordNet senses (e.g. "major" → "ampere" via an electrical-engineering synset, or "bill" → beak-related senses).

The codebase also includes (but does not use in the "Full ARPD" configuration) an **EN→VI→EN back-translation** module using MarianMT (Helsinki-NLP/opus-mt), plus `random_deletion` and `random_swap` helpers used specifically to construct stronger adversarial test sets for the robustness evaluation (Section 11). This overall design follows the general spirit of **EDA** (Wei & Zou, 2019) — synonym replacement, random deletion, random swap — while restricting the _training-time_ augmentation specifically to synonym substitution.

### 6.4. Encoder — claim + evidence + speaker representation

File: `src/encoder.py`

Uses `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384-dim output), based on the Sentence-BERT architecture (Reimers & Gurevych, 2019), **kept fully frozen** (never fine-tuned).

Primary encoding mode (`use_speaker_context=True`, the default for Full ARPD): all available information is concatenated into a **single string** and encoded in one pass:

```
"[speaker] [subject] claim [SEP] evidence_passage_1 evidence_passage_2 evidence_passage_3"
```

(up to 3 evidence passages are concatenated; with no evidence, the string reduces to `"[speaker] [subject] claim"`).

A legacy encoding mode (`use_speaker_context=False`) is also retained: claim and evidence are encoded separately and combined as a weighted sum, `0.7 * v_claim + 0.3 * mean(v_evidence)` — used only for historical ablation comparisons, not the main pipeline.

### 6.5. Classifier — ImprovedMLP

File: `src/classifier.py`

A 3-block architecture with BatchNorm, Dropout, a residual connection, and Xavier initialization:

```
Input (384)
  -> Linear(384,256) -> BatchNorm -> ReLU -> Dropout(0.4)   [Block 1]
  -> Linear(256,128) -> BatchNorm -> ReLU -> Dropout(0.3)   [Block 2]  (+ residual from input via Linear(384,128))
  -> Linear(128,64)  -> BatchNorm -> ReLU -> Dropout(0.2)   [Block 3]
  -> Linear(64,2)                                            [Head: 2-class logits]
```

- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4.
- Loss: CrossEntropyLoss with **automatic class weighting**, computed from train-label frequencies, to mitigate the REAL/FAKE imbalance noted in Section 4.2.
- Trained for up to 20 epochs with **early stopping** on validation F1-macro (patience = 5 epochs with no improvement).
- The best-validation-F1 checkpoint is kept, not the final-epoch weights.

### 6.6. Ensemble — combining TF-IDF/LR and MLP

File: `src/pipeline.py` (`_ensemble_predict_proba`)

```
P_final(REAL) = (1 - w) * P_TFIDF_LR(REAL) + w * P_MLP(REAL)
```

- **TF-IDF + Logistic Regression branch:** text vectorized with TF-IDF (1–3 grams, vocabulary capped at 15,000 terms, including the speaker/subject prefix), Logistic Regression trained with `C=0.5`, `max_iter=1000`.
- **MLP branch:** the probability output of `ImprovedMLP` (Section 6.5), based on the semantic embedding.
- **Weight `w`** is **grid-searched on the validation set** (7 values: 0.05 → 0.35 in steps of 0.05), selecting the value with the highest validation F1-macro — the test set is **never** touched during this search, to avoid leakage.

---

## 7. Experimental Design

### 7.1. Baselines

| Name                           | Description                                                                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| **TF-IDF + LR**                | TF-IDF (1–2 grams) + Logistic Regression on the raw claim text only — no evidence, no speaker context.                               |
| **Frozen DistilRoBERTa + MLP** | Frozen DistilRoBERTa embeddings fed into an MLP of similar architecture to ARPD's, with no evidence, speaker context, or ensembling. |

Both baselines are trained and evaluated across **5 independent random seeds**: 42, 123, 456, 789, 2025.

### 7.2. Ablation study

To identify **exactly which ARPD component contributes value**, four configurations are run, each across 5 seeds:

| Configuration   | retrieve | use_speaker | use_ensemble  | augmentation |
| --------------- | -------- | ----------- | ------------- | ------------ |
| **Full ARPD**   | ✅       | ✅          | ✅            | synonym      |
| **No-Retrieve** | ❌       | ✅          | ✅            | synonym      |
| **No-Speaker**  | ✅       | ❌          | ✅            | synonym      |
| **No-Ensemble** | ✅       | ✅          | ❌ (MLP only) | synonym      |

### 7.3. Robustness evaluation

The Full ARPD checkpoint (seed 42) is additionally evaluated under **three levels of paraphrase attack** (synonym substitution at higher probability than training: 15%, 30%, 50%) to measure performance degradation when claims are re-worded.

### 7.4. Statistical testing

All comparisons between Full ARPD and (baseline / ablation) use a **one-sided paired t-test** (H1: Full ARPD > comparator), paired by seed.

---

## 8. A Critical Bug Found and Fixed

During an earlier phase of this project, it was discovered that **all previously reported "ARPD with retrieval" results were invalid**, due to a silent technical failure.

**Root cause:** `AdaptiveRetriever._fetch_passages()` was sending requests to the Wikipedia API **without a `User-Agent` header**. Per Wikimedia Foundation policy, requests lacking this header are rejected with **HTTP 403 (Forbidden)**. This failure was caught by a generic `try/except` block and **silently converted into an empty result list**, with no warning printed anywhere.

**Consequence:** the evidence caches (`cached_train/val/test_evidence.csv`) built before this fix had a **fill rate of only ~2.4–2.8%**:

| Split      | Rows   | Non-empty evidence | Fill rate |
| ---------- | ------ | ------------------ | --------- |
| Train      | 10,240 | 253                | 2.5%      |
| Validation | 1,284  | 31                 | 2.4%      |
| Test       | 1,267  | 36                 | 2.8%      |

This means every "Full ARPD" result reported prior to the fix was, in practice, running in near-zero-evidence mode while being labeled as "with retrieval." Any earlier comparison between "with retrieval" and "without retrieval" was therefore **scientifically meaningless**.

**Fix:** a valid `User-Agent` header was added to the `AdaptiveRetriever`'s HTTP session, and the number of search candidates (`srlimit`) was increased to 10 to provide more fallback articles before similarity filtering. The pre-fix results are archived in `results/*_before_retrieval_fix.csv` **for historical reference only** and are excluded from all reported numbers below.

**Post-fix fill rates** (measured on the full dataset, not a small sample):

| Split      | Rows   | Non-empty evidence | Fill rate |
| ---------- | ------ | ------------------ | --------- |
| Train      | 10,240 | 8,501              | **83.0%** |
| Validation | 1,284  | 1,081              | **84.2%** |
| Test       | 1,267  | 1,057              | **83.4%** |

This ~83–84% fill rate exceeds the original design target of 70–80%, and is the basis for every result reported from Section 9 onward.

---

## 9. Results

> All numbers below are measured on the **test set** (1,267 samples), after the retrieval fix described in Section 8, with every configuration run independently across **5 seeds: 42, 123, 456, 789, 2025**.

### 9.1. Baselines (5 seeds)

| Model                        | Test Accuracy (mean) | Test F1-macro (mean, 95% CI)             |
| ---------------------------- | -------------------- | ---------------------------------------- |
| TF-IDF + Logistic Regression | 0.6440               | 0.6150 (not meaningful — see note below) |
| Frozen DistilRoBERTa + MLP   | 0.6194               | 0.6178 [0.6124, 0.6232]                  |

_(TF-IDF+LR produces nearly identical results across seeds — this is the expected behavior of a convex linear model with a fixed `random_state`, not a bug, but it also means a 95% CI computed from these 5 values would be a near-zero-width artifact rather than a meaningful estimate of sampling variability; see Section 10.3 for the full statistical consequence of this.)_

### 9.2. ARPD — Full & ablations (5 seeds each)

| Configuration                                 | Test Accuracy (mean ± std) | Test F1-macro (mean ± std) | F1-macro 95% CI  |
| --------------------------------------------- | -------------------------- | -------------------------- | ---------------- |
| **Full ARPD** (retrieve + speaker + ensemble) | 0.6355 ± 0.0077            | 0.6118 ± 0.0077            | [0.6023, 0.6214] |
| No-Retrieve (Wikipedia retrieval disabled)    | 0.6497 ± 0.0046            | 0.6284 ± 0.0043            | [0.6231, 0.6338] |
| No-Speaker (speaker context disabled)         | 0.6437 ± 0.0041            | 0.6264 ± 0.0070            | [0.6178, 0.6351] |
| No-Ensemble (MLP only, no TF-IDF/LR branch)   | 0.5511 ± 0.0122            | 0.5466 ± 0.0113            | [0.5325, 0.5606] |

95% confidence intervals are computed from the 5-seed sample mean and standard error using the _t_-distribution (df=4), not a normal approximation, since n is small. Note that **Full ARPD's CI does not overlap with either No-Retrieve's or No-Ensemble's CI** — a visual, assumption-light corroboration of the t-test results in Section 10 (formal hypothesis tests remain the primary evidence; the non-overlapping CIs are a useful sanity check, not a substitute, since CI overlap and significance testing are not strictly equivalent procedures).

### 9.3. Per-seed breakdown (Full ARPD)

| Seed | Test Acc | Test F1-macro | Optimal ensemble weight (w) |
| ---- | -------- | ------------- | --------------------------- |
| 42   | 0.6306   | 0.6066        | 0.20                        |
| 123  | 0.6275   | 0.6019        | 0.30                        |
| 456  | 0.6417   | 0.6199        | 0.35                        |
| 789  | 0.6456   | 0.6185        | 0.05                        |
| 2025 | 0.6322   | 0.6123        | 0.25                        |

---

## 10. Statistical Significance Testing

### 10.1. Primary test: one-sided paired t-test

**One-sided paired t-test** (H1: Full ARPD > comparator), paired by seed, on F1-macro, together with the paired effect size (Cohen's _d_, computed from the same 5-seed differences used in the t-test):

| Comparison                                              | t           | two-sided p | one-sided p (H1: ARPD > X) | Cohen's _d_ (paired) | Effect size  |
| ------------------------------------------------------- | ----------- | ----------- | -------------------------- | -------------------- | ------------ |
| Full ARPD (0.6118) vs Frozen DistilRoBERTa+MLP (0.6178) | -1.365      | 0.2441      | 1.0000                     | -0.61                | medium       |
| Full ARPD (0.6118) vs TF-IDF+LR (0.6150)                | -0.917      | 0.4109      | 1.0000                     | -0.41                | small–medium |
| Full ARPD (0.6118) vs No-Retrieve (0.6284)              | -5.005      | 0.0075      | 1.0000                     | **-2.24**            | very large   |
| Full ARPD (0.6118) vs No-Speaker (0.6264)               | -3.206      | 0.0327      | 1.0000                     | **-1.43**            | large        |
| Full ARPD (0.6118) vs No-Ensemble (0.5466)              | **+11.186** | **0.0004**  | **0.0002**                 | **+5.00**            | very large   |

**How to read the one-sided p-value column:** because the test only checks "Full ARPD is better," a negative t-statistic (Full ARPD scoring lower) is conventionally assigned p ≈ 1.0 in that column. Read in isolation, this column makes the No-Retrieve and No-Speaker rows look like "no difference was found," which is **misleading**. The two-sided p-value and Cohen's _d_ columns tell the real story: the difference between Full ARPD and No-Retrieve is not only statistically significant (two-sided p = 0.0075) but has a _very large_ effect size (|d| = 2.24) — it is simply significant in the direction opposite to H1. This is the single most important table-reading caveat in this entire report, and Sections 10.2–10.3 below quantify exactly how much weight this evidence can bear.

### 10.2. Multiple-comparisons correction

Five comparisons are run against the same Full ARPD sample, which inflates the family-wise false-positive rate if each is judged against the conventional α = 0.05 in isolation. Applying a **Bonferroni correction** for 5 comparisons gives a corrected threshold of α_corrected = 0.05 / 5 = **0.01**. Re-checking each result against this stricter threshold:

| Comparison                  | two-sided p | Significant at α=0.05? | Significant at α_Bonferroni=0.01? |
| --------------------------- | ----------- | ---------------------- | --------------------------------- |
| vs Frozen DistilRoBERTa+MLP | 0.2441      | No                     | No                                |
| vs TF-IDF+LR                | 0.4109      | No                     | No                                |
| vs No-Retrieve              | 0.0075      | Yes                    | **Yes**                           |
| vs No-Speaker               | 0.0327      | Yes                    | No                                |
| vs No-Ensemble              | 0.0004      | Yes                    | **Yes**                           |

The headline result — Full ARPD vs. No-Ensemble — **survives** Bonferroni correction. The No-Retrieve result also survives. The No-Speaker result (p = 0.0327), while nominally significant at the conventional uncorrected α = 0.05, **does not survive** the multiple-comparisons correction, and should accordingly be reported with more hedging than the other two — "a large, suggestive, but not multiple-comparisons-robust effect" rather than "a significant effect."

### 10.3. A specific statistical caveat: the TF-IDF + Logistic Regression comparison is not a genuine 5-sample paired test

Section 9.1 already notes, as an observation, that TF-IDF+LR's test-set score is identical across all 5 "seeds" (because a convex linear model with a fixed `random_state=42` inside `ARPDPipeline.lr` has no stochasticity left to vary). This has a direct and under-stated statistical consequence: **the paired t-test against TF-IDF+LR is mathematically a one-sample t-test of Full ARPD's 5 scores against a fixed constant (0.614996), not a comparison between two independently-varying 5-seed samples.** This can be verified directly: the standard deviation of the five paired differences (Full ARPD − TF-IDF+LR) is, to machine precision, identical to the standard deviation of Full ARPD's own 5 scores, because the subtracted baseline term contributes exactly zero variance. This does not invalidate the comparison — the point estimate (0.6118 vs 0.6150) is still a fair comparison of means — but it means the "n=5 paired samples" framing implicitly oversells the independence of the baseline side of this specific test, and the resulting p-value should be read as describing the _variability of Full ARPD across seeds relative to a fixed target_, not "variability of the difference between two independently-resampled systems."

### 10.4. Statistical power of the 5-seed design

A one-sided paired t-test with **n = 5** has limited statistical power to detect anything but very large effects. Using the observed test results to calibrate this concretely:

- The minimum effect size detectable with 80% power, at n=5 and one-sided α=0.05, is **d ≈ 1.36** (a "very large" effect by conventional standards).
- The effect sizes actually observed against the two simple baselines were |d| = 0.41 (TF-IDF+LR) and |d| = 0.61 (DistilRoBERTa+MLP) — both in the "small-to-medium" range.
- At n=5, the **achieved power** to detect an effect of that magnitude is only **≈19–31%** — meaning that even if a true underlying effect of exactly this size existed in the population, an experiment of this size would fail to detect it as significant roughly 70–80% of the time, purely due to sample size, independent of whether ARPD genuinely helps.
- Reaching 80% power to detect an effect of d=0.41 (the TF-IDF+LR magnitude) at one-sided α=0.05 would require **n ≈ 38 seeds**; for d=0.61 (the DistilRoBERTa+MLP magnitude), **n ≈ 18 seeds**.

**Implication:** the "Full ARPD does not significantly beat the two simple baselines" conclusion (Section 9.1, Section 12.1) should be read as **"this 5-seed experiment did not have the statistical power to detect a small-to-medium effect against the simple baselines, even though a small-to-medium-sized true effect cannot be ruled out."** This is a materially different and more honest claim than "there is no effect." By contrast, the No-Ensemble and No-Retrieve comparisons are not subject to this caveat — their effect sizes (|d| = 5.00 and 2.24 respectively) are so large that even an underpowered 5-seed design detects them reliably, which is exactly why those two specific conclusions are reported with confidence in Section 12, while the baseline comparisons are reported as inconclusive rather than as "no effect."

---

## 11. Robustness Evaluation

The Full ARPD checkpoint (seed 42, clean test accuracy = 0.6306) was attacked with synonym substitution at three intensities:

| Attack                                 | Accuracy | F1-macro | Robustness Drop\* |
| -------------------------------------- | -------- | -------- | ----------------- |
| Clean (no perturbation)                | 0.6306   | 0.6066   | 0.0000            |
| synonym_p15 (15% of words substituted) | 0.6472   | 0.6285   | **-0.0166**       |
| synonym_p30 (30% of words substituted) | 0.6361   | 0.6154   | **-0.0055**       |
| synonym_p50 (50% of words substituted) | 0.6338   | 0.6125   | **-0.0032**       |

\*Robustness Drop = Accuracy(clean) − Accuracy(adversarial). A negative value means the model performed **better** on the perturbed data than on the original — see Section 12.4 for discussion.

**Statistical context for these numbers (computed, not estimated):** the test set has n = 1,267 items. Treating a single accuracy measurement as a sample proportion, the standard error at accuracy ≈ 0.63 is SE = √(0.63×0.37/1267) ≈ 0.0136, giving a normal-approximation 95% margin of error of **±0.0266 (≈2.66 accuracy points)** for a _single_ accuracy measurement on this test set. All three observed "robustness improvements" (0.32 to 1.66 accuracy points) are **smaller than this single-measurement margin of error** — i.e., they are quantitatively consistent with measurement noise on a test set of this size, even before considering that clean and adversarial predictions are _paired_ on the same 1,267 items (which would call for a McNemar-style paired test on row-level correct/incorrect outcomes, not an independent-proportions comparison — see Section 14.7 for why that exact test could not be reconstructed retroactively from the aggregated CSV outputs alone).

---

## 12. Discussion & Interpretation

### 12.1. Hypothesis H1 is not supported — but the evidence is asymmetric, not uniformly negative

The experimental design anticipated two possible, equally valid outcomes (Section 2): Outcome A (ARPD significantly beats baselines) or Outcome B (it does not, but the ablation explains why). The literal result lands in **Outcome B**: Full ARPD shows no statistically significant difference from either simple baseline at the conventional threshold. However, Section 10.4 shows this null result must be qualified rather than stated flatly: at n=5 seeds, this design has only **≈19–31% power** to detect an effect of the magnitude actually observed (|d| = 0.41–0.61). A more precise and more honest statement of this part of the finding is therefore: **"this study could not confirm that Full ARPD beats the two simple baselines, and was statistically underpowered to do so even if a small-to-medium true effect exists."** This is different from, and weaker than, "Full ARPD does not beat the baselines" — the latter claim is not supported by this data either, in either direction.

By contrast, the **No-Retrieve** and **No-Ensemble** comparisons are not subject to this power caveat: their effect sizes (|d| = 2.24 and 5.00 respectively, both "very large" by conventional standards) are large enough that a 5-seed design detects them with high confidence regardless of the design's general power limitations. This is why Section 12.2 below treats the ablation-internal findings with considerably more confidence than the ARPD-vs-baseline findings — the asymmetry in statistical power is itself part of the finding, not just a caveat appended after the fact.

### 12.2. The ablation tells a more detailed, and more statistically defensible, story than the headline comparison

The four configurations reveal very different contribution levels among ARPD's components, and — unlike the baseline comparisons in 12.1 — these differences are large enough to be measured reliably even with only 5 seeds:

- **Ensembling (combining TF-IDF/LR with the MLP) is the only component with a contribution that is both large and statistically robust to multiple-comparisons correction** (two-sided p = 0.0004, surviving Bonferroni correction at α=0.01; Cohen's d = +5.00). Disabling it drops F1-macro from 0.6118 to 0.5466 — by far the largest effect size in the entire ablation. This confirms the value of combining surface lexical signal (TF-IDF) with deeper semantic signal (embedding + MLP), and is the one claim in this entire report that can be made with high confidence.
- **Wikipedia evidence retrieval does not improve performance — and shows a large effect in the opposite direction.** The No-Retrieve configuration (0.6284) outperforms Full ARPD (0.6118), with a _very large_ effect size (d = -2.24, two-sided p = 0.0075) that also survives Bonferroni correction. This is a confident finding, not an inconclusive one: retrieval is reliably associated with _worse_, not better, performance in this specific configuration.
- **Speaker context shows the same directional pattern, with a large but less robust effect.** No-Speaker (0.6264) outperforms Full ARPD, d = -1.43, two-sided p = 0.0327 — large enough to be taken seriously, but this specific comparison does _not_ survive the Bonferroni correction in Section 10.2, and should be reported as suggestive rather than confirmed.

### 12.3. Why might retrieval and speaker context not help — or even hurt?

A few plausible explanations (not independently verified in this study, and worth testing directly in follow-up work):

1. **LIAR's claim structure is a poor match for Wikipedia-summary retrieval.** LIAR claims are short, decontextualized political quotes (e.g. "We have fewer Americans working now than in the seventies."). A Wikipedia summary can address the general topic (US employment trends) but rarely the specific statistic under scrutiny — retrieved passages provide background knowledge rather than direct verification evidence. As discussed in Section 3.2, this is precisely the structural property that distinguishes LIAR from FEVER, where claims _are_ Wikipedia-grounded by construction.
2. **~17% of claims still receive no evidence** (fill rate ~83%), so for that subset "Full ARPD" effectively degrades to the No-Retrieve configuration — this partial inconsistency may also explain the larger standard deviation observed for Full ARPD (0.0077) versus No-Retrieve (0.0043): Full ARPD is, in effect, a noisy mixture of two different operating regimes (with-evidence and zero-evidence) rather than a single consistent one.
3. **The small MLP (~190K parameters) may lack the capacity** to exploit the more complex signal introduced by concatenating longer evidence text — longer input strings may dilute the claim's own semantic signal within the fixed 384-dimensional embedding.
4. **ARPD's speaker conditioning only uses the speaker's name and subject string**, not the historical-truthfulness metadata that prior LIAR-specific literature (Section 3.4) identifies as the actual source of most documented speaker-related gains — so the absence of improvement here is consistent with, rather than contradictory to, that literature.
5. **The `k_adaptive` heuristic (Section 6.1) is not learned from data**, so it may be allocating the "evidence budget" suboptimally across claim types.

None of these five explanations is mutually exclusive, and this study does not isolate which one (if any) is the dominant cause — distinguishing between them would require, for example, stratifying the No-Retrieve-vs-Full-ARPD comparison by evidence-fill status (explanation 2) or by claim specificity (explanation 1), neither of which was logged at the per-claim level in the current `results/*.csv` outputs. This is listed explicitly as a methodological gap in Section 14.8.

### 12.4. The robustness result is a genuine open question, not a confirmed finding either way

The typical expectation is that a model performs **worse** on paraphrased inputs (since perturbation adds noise to the signal). The measured result instead shows a **slight accuracy increase** at all three attack intensities, most pronounced at the mildest level (p=15%: +1.66 accuracy points). Section 11 shows this is not merely "small" in a qualitative sense — it is **quantitatively smaller than the single-measurement 95% margin of error** (±2.66 accuracy points) implied by a 1,267-item test set. This does not prove the robustness effect is _only_ noise — the clean and adversarial predictions are paired on the same items, and a properly paired test (McNemar's test on row-level outcomes) could in principle detect a real effect smaller than the unpaired margin of error suggests — but that paired test could not be reconstructed retroactively from the aggregated CSV outputs available for this report (Section 14.7, Section 14.8). The honest conclusion is that **this result is currently neither confirmed nor refuted**, and should not be cited as evidence of either "robustness" or "fragility" until the appropriate paired significance test is run on row-level prediction data in a future replication.

### 12.5. Overall conclusion

> Within the scope of this experimental configuration (LIAR, frozen MiniLM-L6-v2, a small MLP trained on frozen embeddings), **ensembling is the only ARPD component whose contribution is both large and statistically robust** (Section 12.2). Two further findings are confidently established in the _opposite_ direction from the original hypothesis: Wikipedia-based evidence retrieval and, with somewhat less statistical robustness, speaker-aware context, are both associated with a measurable _decrease_ in performance on this dataset rather than the hypothesized increase — these are not "no effect" results, they are large, mostly Bonferroni-robust effects pointing the wrong way. The comparison against the two simple baselines, by contrast, remains genuinely inconclusive rather than negative: the 5-seed design lacked the statistical power to detect an effect of the magnitude actually observed (Section 10.4), so "Full ARPD versus TF-IDF+LR / DistilRoBERTa+MLP" should be reported as an open question requiring a larger-n replication, not as evidence that ARPD provides no benefit over simple baselines. Taken together, this is a meaningful, multi-part finding about the **limits of generic Wikipedia-style retrieval augmentation and shallow speaker conditioning on non-Wikipedia-grounded claim data such as LIAR**, combined with a confirmed, large positive result for model ensembling — and it motivates testing the same pipeline on a dataset where retrieved evidence can be checked directly against the claim, such as FEVER (Thorne et al., 2018), with a larger seed count to adequately power the baseline comparison — see Section 15.

---

## 13. Contributions of This Work

A contribution, in the sense used here, is not simply "a technique we built" — it is **a claim about the world that is more justified after this work than before it, together with an honest account of how much weight that claim can bear.** This section is organized around that standard, distinguishing explicitly between contributions that are methodological (reusable regardless of the specific numbers), and contributions that are empirical (specific, quantified claims, each tagged with its actual epistemic status rather than presented as uniformly "confirmed").

### 13.0. Summary: what this work allows you to claim, and what it does not

| Claim                                                                                                                              | Epistemic status                                                       | Basis                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Ensembling TF-IDF/LR with a frozen-embedding MLP improves F1-macro on LIAR                                                         | **Confirmed, large effect, Bonferroni-robust**                         | d=+5.00, two-sided p=0.0004 (Section 10.2)                                            |
| Wikipedia-summary retrieval, as implemented here, _does not_ improve F1-macro on LIAR and is associated with a measurable decrease | **Confirmed, very large effect, Bonferroni-robust**                    | d=-2.24, two-sided p=0.0075 (Section 10.2)                                            |
| Name+subject speaker conditioning, as implemented here, is associated with a decrease in F1-macro on LIAR                          | **Suggestive, large effect, not Bonferroni-robust**                    | d=-1.43, two-sided p=0.0327 (Section 10.2)                                            |
| Full ARPD beats simple TF-IDF+LR / frozen-encoder+MLP baselines                                                                    | **Inconclusive — underpowered, not negative**                          | 19–31% achieved power at n=5 (Section 10.4)                                           |
| Full ARPD is more robust to synonym-substitution paraphrase than expected                                                          | **Open question — within measurement noise, no paired test available** | Effect (≤1.66 pts) below the ±2.66 pt single-measurement margin of error (Section 11) |

This table is itself part of the contribution: it is the kind of explicit claim-by-claim epistemic accounting that a single aggregate "our F1 vs. their F1" comparison — the norm in much of the applied fake-news-detection literature surveyed in Section 3 — typically does not provide.

### 13.1. A reproducible, fully-decomposable pipeline, not a monolithic model

Unlike many fake-news detection papers that report a single end-to-end number for a tightly coupled architecture, every component of ARPD — retrieval, speaker context, augmentation, and ensembling — is implemented as an **independently toggleable flag** in `ARPDPipeline` (Section 5, Section 6). This made it possible to run a full ablation matrix (Section 7.2) using the _exact same_ underlying codebase and random seeds for every configuration, rather than re-implementing each ablation as a separate one-off script. The practical value of this design choice is direct and load-bearing for everything else in this report: it is what makes the component-level attribution in Section 12.2 — and, specifically, the ability to isolate ensembling as the one confidently-positive component — possible at all, rather than only being able to report an aggregate "ARPD vs. baseline" delta that would have left the ensembling effect, the retrieval effect, and the speaker effect entangled and individually invisible.

### 13.2. A documented, root-caused, and fixed silent failure mode in evidence retrieval

Section 8 documents a failure mode that is easy to overlook and likely under-reported in similar retrieval-augmented pipelines: a missing `User-Agent` header caused the Wikipedia API to silently return empty evidence for ~97.5% of claims, while the system continued to label its output as "retrieval-augmented." This is not merely a one-off bug-fix note — it is evidence for a broader methodological point with relevance beyond this project: **retrieval-augmented systems should report and monitor evidence fill rate as a first-class metric**, not assume that "the retriever ran without throwing an exception" is equivalent to "the retriever worked." A system that silently degrades to zero-evidence mode and a system that genuinely retrieves no useful evidence are operationally indistinguishable without this metric, yet they imply completely different conclusions about whether retrieval _itself_ was tested. The before/after fill-rate comparison in Section 8 (2.4–2.8% → 83.0–84.2%) is offered as a concrete, reusable diagnostic template for catching this class of error in other retrieval-augmented research pipelines — and is also a partial methodological caution about this literature generally: any retrieval-augmented result that does not report fill rate alongside accuracy should be read with this failure mode specifically in mind.

### 13.3. A statistically rigorous, component-level, and power-aware account of retrieval's value for LIAR-style claims

Most evidence-retrieval papers for fake-news detection are evaluated on FEVER-style data where claims are Wikipedia-grounded by construction (Section 3.2), so retrieval almost mechanically helps. This work instead provides a **statistically tested, ablation-isolated, effect-size-quantified measurement of retrieval's marginal value specifically on LIAR-style, decontextualized political claims** (Section 10.1–10.2) — and finds, with a very large, Bonferroni-robust effect size (d=-2.24), that retrieval does not help and is reliably associated with mild _harm_ in this specific configuration. Critically, this conclusion is reported with a precision that distinguishes it from the (separately underpowered, and therefore inconclusive) comparison against simple baselines — the contribution here is not just "we found a negative result," but "we found a negative result large enough that even a small, computationally cheap study can state it with confidence, while simultaneously being honest that a _different_ comparison in the same experiment (vs. baselines) cannot be stated with the same confidence." This is a useful, specific negative data point for the sub-question "does generic Wikipedia-summary retrieval help on short, decontextualized political claims," distinct from the broader and already well-supported claim "evidence retrieval helps fact verification in general" (well established on FEVER-style data, Section 3.2).

### 13.4. A worked example connecting LIAR speaker-conditioning results to the literature's actual locus of gain

Section 3.4 surveys prior LIAR-specific work and identifies that the documented gains from speaker metadata are concentrated specifically in **historical truthfulness counts**, not speaker identity or topic alone. By explicitly noting that ARPD's `use_speaker_context` only encodes speaker _name_ and claim _subject_ — and observing a measurable, large (though not Bonferroni-robust) _decrease_ in performance from it (Section 9.2, Section 10.2) — this work offers a clean, literature-grounded explanation for a negative ablation result, rather than leaving it as an unexplained anomaly. This distinction (name/topic vs. historical credibility record) is a reusable, falsifiable observation for any future LIAR-based system deciding which metadata fields are worth the added encoding complexity, and Section 15 turns it directly into a concrete next experiment (encode credit-history counts and re-run the same ablation slot).

### 13.5. A fully free-tier-reproducible experimental protocol

Every experiment in this work — including the full ablation matrix, both baselines, and the robustness evaluation — was executed end-to-end on a free-tier Google Colab T4 GPU, using a frozen 22M-parameter encoder and a ~190K-parameter trainable head. This is a deliberate scope decision (Section 14.5), but it also means the entire experimental protocol in Section 17 is realistically reproducible by other students or researchers without institutional compute budgets, which is not true of many of the LLM-based retrieval-augmented baselines surveyed in Section 3. This matters for the _kind_ of contribution this work can make: it demonstrates that a statistically careful, multi-seed, fully-ablated evaluation protocol — of the sort that is often skipped in resource-constrained student research precisely because it appears to require large compute budgets — is achievable on hardware available to essentially any researcher.

### 13.6. Treating statistical rigor itself as part of the contribution, not just as reporting hygiene

Sections 10.2–10.4 and 13.0 are, in one sense, "just" correct statistical practice — Bonferroni correction, effect sizes, and power analysis are standard tools. They are listed here as a contribution specifically because their _absence_ is common in the broader applied fake-news-detection literature surveyed in Section 3, where single-seed or small-n results are frequently reported as definitive without effect-size or power context. By explicitly computing and reporting (a) which of the five hypothesis tests survive multiple-comparisons correction, (b) the effect size and achieved power for every comparison, and (c) the exact mathematical sense in which the TF-IDF+LR comparison is a one-sample rather than two-sample test (Section 10.3), this work aims to model a standard of evidentiary precision that is directly transferable to any other small-seed-count ablation study — independent of whether ARPD itself turns out to be useful.

### 13.7. Honest, explicitly-bounded reporting of an unresolved robustness result

Rather than omitting or over-interpreting the counter-intuitive robustness finding (Section 11, Section 12.4) — that synonym-perturbed claims scored _higher_, not lower, than clean claims — this work reports it directly alongside a quantified bound (the result is smaller than the single-measurement margin of error implied by the test-set size) and an explicit statement of which paired statistical test _would_ resolve the question, and why it could not be run retroactively from the data already collected (Section 14.7–14.8). This is offered as a methodological contribution in its own right: a template for how to report a surprising result by bounding it numerically and naming the specific missing data and test needed to resolve it, rather than either suppressing the surprise or resolving it with unsupported speculation — and a concrete, falsifiable target for the follow-up evaluation proposed in Section 15.

---

## 14. Limitations

This section is written for thesis reviewers and future researchers, in the interest of full transparency.

### 14.1. Residual empty-retrieval rate (~17%)

Even after the User-Agent fix, roughly 17% of claims receive no evidence above the 0.25 similarity threshold — mostly claims about hyper-local political topics (e.g. county-level tax or spending figures) for which Wikipedia has no dedicated article. For this subset, "Full ARPD" effectively degrades to zero-evidence mode.

### 14.2. The Uncertainty Scorer is a hand-written heuristic, not a learned model

As noted in Section 6.1 and Section 3.3, the weights of the five linguistic features (0.20, 0.25, 0.15…) were chosen by the research team's intuition, not fit to data. This mechanism should not be cited as evidence of "learned adaptive behavior" — it is an interpretable, explicit rule set whose optimality has not been independently verified, and is a much lighter-weight design point than the learned complexity classifiers used in the broader Adaptive-RAG literature (Section 3.3).

### 14.3. LIAR's claim structure may be a poor fit for Wikipedia-summary-style retrieval

As discussed in Section 3.2 and Section 12.3, LIAR consists of short, decontextualized quotes. Wikipedia evidence typically provides background knowledge about the general topic rather than direct verification of the specific figure or assertion in the claim. This is a dataset-level limitation rather than a retrieval-implementation bug.

### 14.4. WordNet synonym-substitution quality is limited

Despite the part-of-speech constraint and blocklist, `synonym_substitute` performs **context-free** WordNet lookups — a word like "bill" in "Congress passed a bill" can still occasionally be replaced with a beak-related sense, since the substitution mechanism cannot disambiguate the legislative sense from the anatomical one using part-of-speech alone. As discussed in Section 3.5, this places ARPD's augmentation at the simplest end of the paraphrase-robustness spectrum, well below LLM-driven, context-aware adversarial augmentation methods such as AdStyle (2024).

### 14.5. No end-to-end encoder fine-tuning

MiniLM-L6-v2 is kept fully frozen; only the downstream `ImprovedMLP` (~190K parameters) is trained. End-to-end fine-tuning of the full 22M-parameter encoder on LIAR might improve performance, but was excluded due to the compute constraints of the free-tier Colab environment this project deliberately targets (Section 13.5).

### 14.6. Single-dataset evaluation

All results are measured on LIAR (Wang, 2017). Generalization to other fake-news datasets (FEVER, FakeNewsNet, PHEME) has not been demonstrated.

### 14.7. The robustness evaluation lacks formal, paired significance testing

The robustness evaluation (Section 11) was run on a single checkpoint (seed 42) and has not been repeated across multiple seeds. More specifically, although clean and adversarial predictions are _paired_ observations on the same 1,267 test items (the correct statistical test would be McNemar's test, or an equivalent paired test, on row-level correct/incorrect outcomes), only aggregate accuracy and F1-macro were exported to `results/robustness_results.csv` — individual per-item predictions were not retained. As a result, this report can only bound the result using an _unpaired_ margin-of-error approximation (Section 11), which is conservative but not the statistically correct test for this paired design. The "slight improvement under perturbation" result (Section 12.4) should accordingly be read as an open, currently untestable-from-existing-outputs question, not a confirmed conclusion in either direction.

### 14.8. Per-claim diagnostic data was not retained, limiting causal attribution within the ablation

The explanations offered in Section 12.3 for why retrieval and speaker context underperform (evidence-fill status, claim specificity, claim length) are plausible but not directly tested, because `run_arpd.py` and `aggregate_results.py` only persist aggregate per-seed metrics (accuracy, F1-macro, F1-fake, F1-real) to `results/arpd_results.csv`, not per-claim prediction correctness joined with per-claim evidence-fill status or linguistic features. A stratified re-analysis — e.g., "does No-Retrieve's advantage over Full ARPD disappear specifically on the ~83% of claims that _did_ receive evidence?" — would directly test explanation 2 in Section 12.3, but requires re-running evaluation with row-level outputs retained, which the current experimment pipeline does not do by default.

### 14.9. The "no significant difference from baselines" result is underpowered, not confirmatory of equivalence

As quantified in Section 10.4, the 5-seed design has only ≈19–31% power to detect the effect sizes actually observed against the two simple baselines (|d| = 0.41–0.61). This is listed here as a limitation, not only as a statistical footnote, because it constrains what can honestly be claimed in Section 12.5 and in any abstract or summary of this work: the correct claim is "this study could not establish a significant advantage of Full ARPD over simple baselines, and was underpowered to do so," not "Full ARPD provides no advantage over simple baselines." A formal equivalence test (e.g., a two-one-sided-tests / TOST procedure) — which would be needed to actually support a claim of "no meaningful difference" — was not run, and is suggested as a concrete next step in Section 15.

---

## 15. Future Work

Ordered roughly by how directly each item follows from a specific gap identified in Sections 10–14, rather than by topic:

1. **Re-run the baseline comparison with an a-priori-powered seed count.** Section 10.4 shows that detecting the observed effect sizes (|d| = 0.41–0.61) against TF-IDF+LR and DistilRoBERTa+MLP at 80% power would require **n ≈ 18–38 seeds**, not 5. This is the single most direct, well-specified next experiment implied by this report: it would convert the currently-inconclusive baseline comparison (Section 12.1, Section 14.9) into either a confirmed positive or a genuine equivalence result.
2. **Run a formal equivalence test (TOST)** alongside the larger-n replication in (1), rather than relying on a non-significant superiority test alone — this is the correct tool for actually supporting a future claim of "no meaningful difference from baselines," which the current design cannot support either way (Section 14.9).
3. **Retain row-level predictions** (per-claim correctness, for both clean and adversarial inputs, and for both retrieval-on and retrieval-off configurations) in future experiment runs. This single change to the logging pipeline would directly enable: a McNemar-style paired significance test for the robustness result (Section 14.7), and a stratified re-analysis of the retrieval ablation by evidence-fill status and claim specificity (Section 14.8) — both of which the current aggregate-only CSV outputs cannot support retroactively.
4. **Evaluate on FEVER** (Thorne et al., 2018) — a dataset where claims are, by construction, grounded directly in specific Wikipedia sentences, to test whether retrieval shows measurable value when claim and evidence share the same semantic grounding, in contrast to LIAR's decontextualized political quotes (Section 3.2, Section 12.3).
5. **Encode the speaker's historical truthfulness counts**, not just their name and the claim's subject, given the literature evidence (Section 3.4) that this specific metadata field — not speaker identity alone — is responsible for most of the documented speaker-conditioning gains on LIAR, and re-run the same No-Speaker ablation slot to test whether this changes the currently-negative (though not Bonferroni-robust) result.
6. **Replace the hand-written Uncertainty Scorer with a learned complexity estimator**, in the spirit of Jeong et al.'s (2024) learned query-complexity classifier (Section 3.3) — e.g. by training a small model to predict whether retrieving additional evidence actually changes the prediction on held-out validation data, rather than relying on fixed linguistic-feature weights.
7. **Fine-tune the encoder (or use a lightweight adapter/LoRA)** so the embedding learns a representation specialized for the political fake-news domain, instead of relying on a general-purpose frozen MiniLM encoder.
8. **Upgrade the retriever** — e.g. dense passage retrieval (DPR-style) instead of TF-IDF-keyword + Wikipedia Search API, or a domain-specific evidence source (PolitiFact, FactCheck.org) instead of general-purpose Wikipedia summaries.
9. **Upgrade the augmentation strategy** toward model-aware adversarial augmentation, in the spirit of AdStyle (2024, Section 3.5), rather than fixed-probability, context-free WordNet substitution.

---

## 16. Codebase Structure

```
ARPD-research/
├── src/
│   ├── data_loader.py          # LIAR loading + label binarization (6-class -> binary)
│   ├── uncertainty_scorer.py   # Linguistic heuristic -> k_adaptive (1..5)
│   ├── adaptive_retriever.py   # Wikipedia retrieval (Search + Batch Extract API)
│   ├── paraphrase_augmentor.py # WordNet synonym substitution + back-translation
│   ├── encoder.py               # MiniLM-L6-v2: claim + evidence + speaker -> R^384
│   ├── classifier.py            # ImprovedMLP (384->256->128->64->2) + training loop
│   └── pipeline.py               # ARPDPipeline: end-to-end, with ablation flags
├── experiments/
│   ├── run_baseline.py          # TF-IDF+LR and Frozen-Encoder+MLP, multi-seed
│   ├── run_arpd.py               # Full ARPD + ablation configurations, multi-seed
│   ├── evaluate_robustness.py    # Evaluation under paraphrase attack (15/30/50%)
│   └── aggregate_results.py      # Mean/std aggregation by configuration + paired t-test
├── build_evidence_cache.py       # Wikipedia evidence cache builder (resume-capable)
├── notebooks/                     # Google Colab execution notebook
├── data/
│   ├── raw/                       # Original LIAR (TSV zip)
│   └── processed/                 # Binarized LIAR, CSV format (claim, speaker, subject, label)
├── results/
│   ├── baseline_results.csv
│   ├── arpd_results.csv
│   ├── significance_report.csv
│   ├── full_significance_tests.csv
│   ├── robustness_results.csv
│   └── checkpoints/               # Saved model + pipeline state
├── tests/                          # Unit tests per module
├── LIMITATIONS.md                  # Detailed limitations (source document for Section 14)
├── RESULTS_SUMMARY.md              # History of the retrieval bug + before/after results
└── requirements.txt
```

---

## 17. Reproducing the Experiments

### 17.1. Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `nltk`, `pandas`, `numpy`. A GPU-enabled environment is recommended (validated on Google Colab, Tesla T4, free tier).

### 17.2. Data preparation

```bash
python -m src.data_loader   # downloads LIAR, binarizes labels, saves to data/processed/
```

### 17.3. Building the evidence cache (the most time-consuming step)

```bash
python build_evidence_cache.py --splits train val test
```

The script is resume-capable — re-running the same command after an interruption continues from where it left off instead of restarting.

### 17.4. Training baselines (5 seeds)

```bash
for seed in 42 123 456 789 2025; do
    python experiments/run_baseline.py --seed $seed
done
```

### 17.5. Training Full ARPD + ablations (5 seeds each)

```bash
SEEDS="42 123 456 789 2025"

for seed in $SEEDS; do
    python experiments/run_arpd.py --augmentation synonym --seed $seed --epochs 20                       # Full ARPD
    python experiments/run_arpd.py --augmentation synonym --no-retrieve --seed $seed --epochs 20         # No-Retrieve
    python experiments/run_arpd.py --augmentation synonym --no-speaker --seed $seed --epochs 20          # No-Speaker
    python experiments/run_arpd.py --augmentation synonym --no-ensemble --seed $seed --epochs 20         # No-Ensemble
done
```

### 17.6. Robustness evaluation

```bash
python experiments/evaluate_robustness.py --retrieve
```

### 17.7. Aggregating results + significance testing

```bash
python experiments/aggregate_results.py
```

The aggregated summary is saved to `results/significance_report.csv`; t-test details are printed to console (see Section 10 for interpretation).

---

## 18. References

- Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). _Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection_. arXiv:2310.11511.
- Chen, J., Bao, R., Zhang, H., Wang, N., Li, J., & Yang, D. (2022). _GERE: Generative Evidence Retrieval for Fact Verification_. SIGIR 2022.
- Cohen, J. (1988). _Statistical Power Analysis for the Behavioral Sciences_ (2nd ed.). Lawrence Erlbaum Associates. (Source of the effect-size and power conventions used in Section 10.)
- Cui, L., & Lee, D. (2020). _Coaid: Covid-19 healthcare misinformation dataset_. arXiv:2006.00885. (Cited for documented paraphrase-vulnerability of misinformation detectors, Section 3.5.)
- Goldani, M. H., Momtazi, S., & Safabakhsh, R. (2021). _Detecting fake news with capsule neural networks_. Applied Soft Computing, 101, 106991.
- Jain, V., Kaliyar, R. K., Goswami, A., Narang, P., & Sharma, Y. (2022). _AENeT: an attention-enabled neural architecture for fake news detection using contextual features_. Neural Computing and Applications, 34(1), 771–782.
- Jeong, S., Baek, J., Cho, S., Hwang, S. J., & Park, J. C. (2024). _Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity_. Proceedings of NAACL-HLT 2024, pages 7036–7050.
- Lakens, D. (2017). _Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses_. Social Psychological and Personality Science, 8(4), 355–362. (Source of the TOST equivalence-testing procedure proposed as future work in Section 15.)
- Liao, H., Peng, J., Huang, Z., Zhang, W., Li, G., Shu, K., & Xie, X. (2023). _MUSER: A Multi-Step Evidence Retrieval Enhancement Framework for Fake News Detection_. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2023).
- Ma, J., Gao, W., & Wong, K.-F. (2019). _Hierarchical Attention Networks for evidence-aware fake news detection_. (HAN.)
- Nakamura, K., Levy, S., & Wang, W. Y. (2020). _r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection_. Proceedings of LREC 2020. (Cited for paraphrase-attack vulnerability findings, Section 3.5.)
- Popat, K., Mukherjee, S., Yates, A., & Weikum, G. (2018). _DeClarE: Debunking Fake News and False Claims using Evidence-Aware Deep Learning_. Proceedings of EMNLP 2018.
- Reimers, N., & Gurevych, I. (2019). _Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks_. Proceedings of EMNLP-IJCNLP 2019. (Architectural basis for `sentence-transformers/all-MiniLM-L6-v2`.)
- Soleimani, A., Monz, C., & Worring, M. (2020). _BERT for Evidence Retrieval and Claim Verification_. Advances in Information Retrieval (ECIR 2020). arXiv:1910.02655.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). _FEVER: a large-scale dataset for Fact Extraction and VERification_. Proceedings of NAACL-HLT 2018. (Recommended benchmark for future work, Section 15.)
- Miller, G. A. (1995). _WordNet: A Lexical Database for English_. Communications of the ACM, 38(11), 39–41. (Basis for synonym substitution.)
- Trueman, T. E., Kumar, A. J., Narayanasamy, P., & Vidya, J. (2021). _Attention-based C-BiLSTM for fake news detection_. Applied Soft Computing, 110, 107600.
- Wang, W. Y. (2017). _"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection_. Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 422–426.
- Wei, J., & Zou, K. (2019). _EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks_. Proceedings of EMNLP-IJCNLP 2019.
- Wu, L., Rao, Y., Yu, H., Wang, Y., & Nazir, A. (2021). _EHIAN: Evidence-aware Hierarchical Interactive Attention Network for fake news detection_. (Cited in Section 3.1 via RoE-FND survey of evidence-news interaction architectures.)
- Wikimedia Foundation. _MediaWiki Action API documentation_ (`en.wikipedia.org/w/api.php`) — evidence retrieval source for this project.
- (2024). _AdStyle: Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection_. arXiv:2406.11260.
- (2023). _J-Guard: Journalism Guided Adversarially Robust Detection of AI-generated News_. arXiv:2309.03164.
- (2025). _RoE-FND: A Case-Based Reasoning Approach with Dual Verification for Fake News Detection via LLMs_. arXiv:2506.11078.
- (2024). _Re-Search for The Truth: Multi-round Retrieval-augmented Large Language Models are Strong Fake News Detectors_. arXiv:2403.09747.

---

_This document is based on real experimental results, executed on Google Colab (Tesla T4) after fixing the Wikipedia retrieval bug described in Section 8. All numbers in Sections 9–11 are taken directly from the `results/_.csv` files of the final experimental run — no estimated or placeholder values are used.\*
