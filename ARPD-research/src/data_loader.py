"""
Load và preprocess LIAR dataset.

LIAR có 6 nhãn gốc; ta binarize:
  REAL (1): true, mostly-true, half-true
  FAKE (0): barely-true, false, pants-fire

Nguồn: Download TSV gốc từ GitHub mirror của LIAR dataset (Wang, 2017).
HuggingFace datasets >= 2.x không còn hỗ trợ script-based datasets như liar.py.
"""

import urllib.request
import zipfile
import io
import pandas as pd
from pathlib import Path


LABEL_MAP = {
    "true": 1,
    "mostly-true": 1,
    "half-true": 1,
    "barely-true": 0,
    "false": 0,
    "pants-fire": 0,
}

# Cột trong file TSV gốc của LIAR (không có header)
_TSV_COLS = [
    "id", "label_str", "claim", "subject", "speaker",
    "speaker_job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_fire_counts", "context",
]

# File TSV trong zip
_SPLIT_FILES = {
    "train": "train.tsv",
    "validation": "valid.tsv",
    "test": "test.tsv",
}

# Mirror công khai — file zip gốc từ repo LIAR paper
_LIAR_ZIP_URL = (
    "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/"
    "master/liar_dataset.zip"
)


def _get_liar_zip(cache_dir: Path) -> Path:
    """
    Trả về path tới liar_dataset.zip.
    Ưu tiên file local; chỉ download nếu chưa có.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "liar_dataset.zip"
    if zip_path.exists():
        return zip_path
    print("  Downloading LIAR dataset...")
    urllib.request.urlretrieve(_LIAR_ZIP_URL, zip_path)
    print(f"  Saved → {zip_path}")
    return zip_path


def _parse_tsv_from_zip(zip_path: Path, filename: str) -> pd.DataFrame:
    """Đọc một file TSV từ bên trong zip, trả về DataFrame."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Tìm file trong zip — hỗ trợ cả root lẫn subfolder
        candidates = [n for n in zf.namelist() if n == filename or n.endswith("/" + filename)]
        if not candidates:
            raise FileNotFoundError(f"{filename} not found in {zip_path}. Contents: {zf.namelist()}")
        with zf.open(candidates[0]) as f:
            content = f.read().decode("utf-8")

    df = pd.read_csv(
        io.StringIO(content),
        sep="\t",
        header=None,
        names=_TSV_COLS,
        on_bad_lines="skip",
    )
    return df


def load_liar(split: str = "train", cache_dir: str | None = None) -> pd.DataFrame:
    """
    Load một split của LIAR dataset và trả về DataFrame đã binarize.

    Args:
        split: "train" | "validation" | "test"
        cache_dir: Thư mục cache (None = data/raw/ kế bên src/)

    Returns:
        DataFrame với các cột: claim, speaker, label (0/1), label_str
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"split phải là một trong {list(_SPLIT_FILES.keys())}")

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data" / "raw"
    cache_dir = Path(cache_dir)

    zip_path = _get_liar_zip(cache_dir)
    df_raw = _parse_tsv_from_zip(zip_path, _SPLIT_FILES[split])

    # Binarize nhãn
    df_raw = df_raw[df_raw["label_str"].isin(LABEL_MAP)].copy()
    df_raw["label"] = df_raw["label_str"].map(LABEL_MAP)
    df_raw["claim"] = df_raw["claim"].astype(str).str.strip()
    df_raw = df_raw.dropna(subset=["claim"]).reset_index(drop=True)

    # Fill NaN in speaker/subject so callers can safely format strings
    df_raw["speaker"] = df_raw["speaker"].fillna("").astype(str).str.strip()
    df_raw["subject"] = df_raw["subject"].fillna("").astype(str).str.strip()

    return df_raw[["claim", "speaker", "subject", "label", "label_str"]]


def load_all_splits(cache_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Trả về dict {"train": df, "validation": df, "test": df}."""
    return {
        split: load_liar(split, cache_dir)
        for split in ["train", "validation", "test"]
    }


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Lưu DataFrame ra CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_processed(path: str | Path) -> pd.DataFrame:
    """Load CSV đã lưu."""
    return pd.read_csv(path)


if __name__ == "__main__":
    print("Loading LIAR dataset...")
    splits = load_all_splits()

    for name, df in splits.items():
        n_real = df["label"].sum()
        n_fake = len(df) - n_real
        print(f"  {name:12s}: {len(df):5d} samples | REAL={n_real} FAKE={n_fake}")

    # Lưu ra processed/
    base = Path(__file__).parent.parent / "data" / "processed"
    for name, df in splits.items():
        save_processed(df, base / f"liar_{name}.csv")
        print(f"  Saved → {base / f'liar_{name}.csv'}")
