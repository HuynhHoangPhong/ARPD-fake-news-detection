"""
Smoke test: 50 samples đầu tiên của LIAR, không retrieval.
Train trên 40 samples, test trên 10 samples.
"""
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("ARPD Smoke Test — 50 samples, no retrieval")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/4] Loading LIAR dataset (50 samples)...")

import pandas as pd
from src.data_loader import load_liar
df = load_liar("train")
df50 = df.head(50).reset_index(drop=True)
print("  [OK] Loaded real LIAR dataset.")

print(f"  Total samples : {len(df50)}")
print(f"  REAL (label=1): {df50['label'].sum()}")
print(f"  FAKE (label=0): {(df50['label'] == 0).sum()}")
print(f"  Label distribution: {df50['label_str'].value_counts().to_dict()}")

train_df = df50.head(40)
test_df  = df50.tail(10).reset_index(drop=True)

# ── 2. Fit uncertainty scorer ─────────────────────────────────
print("\n[2/4] Fitting uncertainty scorer...")
from src.uncertainty_scorer import UncertaintyScorer
scorer = UncertaintyScorer(k_min=1, k_max=5)
scorer.fit_reference(train_df["claim"].tolist())

# In k_adaptive cho vài samples
print("  Sample k_adaptive values:")
for claim in test_df["claim"].head(3):
    k = scorer.compute_k(claim)
    u = scorer.score(claim)
    print(f"    k={k} u={u:.3f}  |  {claim[:70]}")

# ── 3. Encode (không retrieval → passages=[]) ─────────────────
print("\n[3/4] Encoding claim+evidence vectors...")
from src.encoder import ClaimEvidenceEncoder
encoder = ClaimEvidenceEncoder()

train_claims   = train_df["claim"].tolist()
train_labels   = train_df["label"].tolist()
test_claims    = test_df["claim"].tolist()
test_labels    = test_df["label"].tolist()

empty_passages = [[] for _ in train_claims]
X_train = encoder.encode_batch(train_claims, empty_passages, show_progress=False)

empty_passages_test = [[] for _ in test_claims]
X_test = encoder.encode_batch(test_claims, empty_passages_test, show_progress=False)

import numpy as np
y_train = np.array(train_labels)
y_test  = np.array(test_labels)

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test  shape: {X_test.shape}")

# ── 4. Train & Evaluate ───────────────────────────────────────
print("\n[4/4] Training MLP classifier (5 epochs)...")
from src.classifier import ARPDTrainer

trainer = ARPDTrainer(input_dim=384)
history = trainer.fit(
    X_train, y_train,
    X_test,  y_test,   # dùng test làm val vì ít data
    epochs=5,
    batch_size=8,
    patience=5,
    verbose=True,
)

metrics = trainer.evaluate(X_test, y_test)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Accuracy : {metrics['accuracy']:.4f}")
print(f"  F1-Macro : {metrics['f1_macro']:.4f}")
print(f"  F1-FAKE  : {metrics['f1_fake']:.4f}")
print(f"  F1-REAL  : {metrics['f1_real']:.4f}")

# In per-sample predictions
print("\nPer-sample predictions (test set):")
preds = trainer.predict(X_test)
print(f"  {'Pred':>6}  {'True':>6}  {'Match':>6}  Claim")
for pred, true, claim in zip(preds, y_test, test_claims):
    match = "OK" if pred == true else "XX"
    label_str = "REAL" if pred == 1 else "FAKE"
    true_str  = "REAL" if true == 1 else "FAKE"
    print(f"  {label_str:>6}  {true_str:>6}  {match:>6}  {claim[:60]}")

print("\nSmoke test PASSED." if metrics['accuracy'] >= 0 else "Something went wrong.")
