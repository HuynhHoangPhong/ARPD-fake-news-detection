"""
MLP Classifiers cho binary fake news detection.

ARPDClassifier  — 2-layer MLP baseline (giữ để backward compat).
ImprovedMLP     — BatchNorm + Dropout + residual + Xavier init.

ImprovedMLP architecture (input_dim=384):
  Input (384)
    -> Linear(384,256) -> BN(256) -> ReLU -> Dropout(0.4)   [block 1]
    -> Linear(256,128) -> BN(128) -> ReLU -> Dropout(0.3)   [block 2]  + residual
    -> Linear(128,64)  -> BN(64)  -> ReLU -> Dropout(0.2)   [block 3]
    -> Linear(64, 2)                                         [head]
  Residual: projected input (384->128) added to block-2 output.
  Output: logits (B, 2) -- dung CrossEntropyLoss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


class ARPDClassifier(nn.Module):
    """2-layer MLP binary classifier (legacy)."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden1: int = 256,
        hidden2: int = 64,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


class ImprovedMLP(nn.Module):
    """
    3-block MLP voi BatchNorm, Dropout, residual connection, Xavier init.
    Output: logits (B, 2) cho CrossEntropyLoss.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 64,
        dropout1: float = 0.4,
        dropout2: float = 0.3,
        dropout3: float = 0.2,
    ) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
        )

        self.block3 = nn.Sequential(
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(dropout3),
        )

        # Residual: project input -> hidden2, add to block2 output
        self.residual_proj = nn.Linear(input_dim, hidden2, bias=False)

        self.head = nn.Linear(hidden3, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        h1 = self.block1(x)
        h2 = self.block2(h1) + residual
        h3 = self.block3(h2)
        return self.head(h3)  # (B, 2)


class ARPDTrainer:
    """Training wrapper -- supports both ARPDClassifier and ImprovedMLP."""

    def __init__(
        self,
        input_dim: int = 384,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
        use_improved: bool = True,
        class_weight: torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            input_dim: Feature vector dimension.
            lr: Learning rate.
            weight_decay: L2 regularization.
            device: "cuda" | "cpu" | None (auto-detect).
            use_improved: True -> ImprovedMLP + CrossEntropyLoss;
                          False -> legacy ARPDClassifier + BCEWithLogitsLoss.
            class_weight: Tensor shape (2,) for imbalanced data.
                          If None, auto-computed from y_train in fit().
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_improved = use_improved
        self._class_weight = class_weight

        if use_improved:
            self.model = ImprovedMLP(input_dim=input_dim).to(device)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weight.to(device) if class_weight is not None else None
            )
        else:
            self.model = ARPDClassifier(input_dim=input_dim).to(device)
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        # ImprovedMLP: CE loss needs long labels; legacy: BCE needs float
        y_t = torch.tensor(y, dtype=torch.long if self.use_improved else torch.float32)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

    def _preds_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.use_improved:
            return logits.argmax(dim=-1)               # (B,2) -> (B,)
        return (torch.sigmoid(logits) >= 0.5).long()   # (B,)  binary

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_improved:
            return self.criterion(logits, y)           # CE: y is long
        return self.criterion(logits, y.float())       # BCE: y is float

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self._compute_loss(self.model(X_batch), y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> dict:
        """Evaluate on X, y. Returns accuracy, f1_macro, f1_fake, f1_real."""
        self.model.eval()
        loader = self._make_loader(X, y, batch_size, shuffle=False)
        all_preds, all_labels = [], []
        for X_batch, y_batch in loader:
            preds = self._preds_from_logits(
                self.model(X_batch.to(self.device))
            ).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1_fake":  f1_score(all_labels, all_preds, pos_label=0, average="binary", zero_division=0),
            "f1_real":  f1_score(all_labels, all_preds, pos_label=1, average="binary", zero_division=0),
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        patience: int = 5,
        verbose: bool = True,
        compute_class_weight: bool = True,
    ) -> list[dict]:
        """
        Training loop with early stopping (monitors val f1_macro).

        Args:
            compute_class_weight: Auto-compute class weights from y_train
                                  when use_improved=True and no weight was
                                  passed to the constructor.
        Returns:
            history: list[dict] per epoch.
        """
        if self.use_improved and compute_class_weight and self._class_weight is None:
            counts = np.bincount(y_train.astype(int), minlength=2)
            total = counts.sum()
            weights = torch.tensor(
                [total / (2.0 * max(c, 1)) for c in counts], dtype=torch.float32
            )
            self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))

        loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        history, best_f1, best_state, no_improve = [], 0.0, None, 0

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(loader)
            val_metrics = self.evaluate(X_val, y_val, batch_size)
            history.append({"epoch": epoch, "loss": loss, **val_metrics})

            if verbose:
                print(
                    f"  Epoch {epoch:3d}/{epochs} | loss={loss:.4f}"
                    f" | acc={val_metrics['accuracy']:.4f}"
                    f" | f1_macro={val_metrics['f1_macro']:.4f}"
                )

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Return binary predictions (0=FAKE, 1=REAL)."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for (X_batch,) in DataLoader(
                TensorDataset(torch.tensor(X, dtype=torch.float32)),
                batch_size=batch_size,
            ):
                preds.extend(
                    self._preds_from_logits(self.model(X_batch.to(self.device))).cpu().numpy()
                )
        return np.array(preds)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved -> {path}")

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        print(f"Model loaded <- {path}")


if __name__ == "__main__":
    np.random.seed(42)
    N_train, N_val = 500, 100
    X_train = np.random.randn(N_train, 384).astype(np.float32)
    y_train = np.random.randint(0, 2, N_train)
    X_val = np.random.randn(N_val, 384).astype(np.float32)
    y_val = np.random.randint(0, 2, N_val)

    trainer = ARPDTrainer(use_improved=True)
    print(f"Device: {trainer.device}")
    print(f"ImprovedMLP params: {sum(p.numel() for p in trainer.model.parameters()):,}")

    out = trainer.model(torch.randn(8, 384))
    assert out.shape == (8, 2), f"Expected (8,2), got {out.shape}"
    print(f"Forward pass shape: {out.shape}  OK")

    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=3, verbose=True)
    print(f"Val metrics: {trainer.evaluate(X_val, y_val)}")
