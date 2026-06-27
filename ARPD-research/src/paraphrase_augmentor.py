"""
Paraphrase Augmentor — tạo paraphrase negatives để train robust model.

Hai phương pháp:
  1. Back-translation EN→VI→EN dùng Helsinki-NLP/opus-mt-* (free, HuggingFace).
     - EN→VI: Helsinki-NLP/opus-mt-en-vi (~74M params, trong giới hạn 125M)
     - VI→EN: Helsinki-NLP/opus-mt-vi-en (~74M params)
     Lưu ý: load cả 2 model tốn ~1.5GB RAM. Trên Colab T4 vẫn ổn.

  2. Synonym substitution dùng WordNet (nltk) — nhẹ, không cần GPU.

Pipeline mặc định: back-translation trước, sau đó synonym sub với xác suất p.
"""

from __future__ import annotations

import random
import re

import nltk
from nltk.corpus import wordnet

# Lazy-load để tránh import transformer lúc không cần thiết
_bt_models: dict | None = None

# WordNet POS tag → NLTK POS constant mapping
_POS_MAP = {
    "NN": wordnet.NOUN, "NNS": wordnet.NOUN, "NNP": wordnet.NOUN, "NNPS": wordnet.NOUN,
    "VB": wordnet.VERB, "VBD": wordnet.VERB, "VBG": wordnet.VERB,
    "VBN": wordnet.VERB, "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
    "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
    "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV,
}

# Blocklist: lemmas that are slang, vulgar, or domain-inappropriate
_SYNONYM_BLOCKLIST = {
    "fuck", "shit", "ass", "bastard", "crap", "damn", "hell",
    "piss", "whore", "bitch", "cock", "dick", "pussy", "cunt",
    "homo", "retard", "moron", "idiot",
    # Degenerate substitutions observed on political LIAR claims
    "ampere",   # "major" → "ampere" via electrical engineering synset
    "neb",      # "bill" → "neb" (archaic beak meaning)
    "pecker",   # "bill" → "pecker" (vulgar)
    "billhook",
}


def _load_bt_models() -> dict:
    """Load back-translation models lần đầu tiên dùng."""
    global _bt_models
    if _bt_models is not None:
        return _bt_models

    from transformers import MarianMTModel, MarianTokenizer

    print("Loading back-translation models (lần đầu có thể mất vài phút)...")
    en_vi_name = "Helsinki-NLP/opus-mt-en-vi"
    vi_en_name = "Helsinki-NLP/opus-mt-vi-en"

    _bt_models = {
        "en_vi_tok": MarianTokenizer.from_pretrained(en_vi_name),
        "en_vi_mdl": MarianMTModel.from_pretrained(en_vi_name),
        "vi_en_tok": MarianTokenizer.from_pretrained(vi_en_name),
        "vi_en_mdl": MarianMTModel.from_pretrained(vi_en_name),
    }
    return _bt_models


def _translate(text: str, tokenizer, model) -> str:
    """Dịch một câu dùng MarianMT."""
    import torch

    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def back_translate(text: str, use_gpu: bool = False) -> str:
    """
    Back-translation EN → VI → EN.

    Args:
        text: Câu tiếng Anh gốc.
        use_gpu: Có dùng GPU không (nếu available).

    Returns:
        Câu paraphrase tiếng Anh sau back-translation.
    """
    import torch

    models = _load_bt_models()
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    for key in ["en_vi_mdl", "vi_en_mdl"]:
        models[key] = models[key].to(device)

    vi_text = _translate(text, models["en_vi_tok"], models["en_vi_mdl"])
    en_text = _translate(vi_text, models["vi_en_tok"], models["vi_en_mdl"])
    return en_text


def synonym_substitute(text: str, p: float = 0.15, seed: int | None = None) -> str:
    """
    Thay thế ngẫu nhiên ~p% từ bằng synonym từ WordNet.

    Uses POS-tagging to restrict substitutions to the same part of speech,
    and filters out a blocklist of slang/vulgar/degenerate lemmas.

    Args:
        text: Câu gốc.
        p: Xác suất thay thế mỗi từ.
        seed: Random seed để reproducible.

    Returns:
        Câu sau khi substitute.
    """
    rng = random.Random(seed)
    words = text.split()

    # POS-tag for same-POS constraint; fall back gracefully if unavailable
    try:
        pos_tags = nltk.pos_tag(words)
    except Exception:
        pos_tags = [(w, "NN") for w in words]  # default to noun if tagger fails

    result = []
    for word, pos in pos_tags:
        if rng.random() < p:
            wn_pos = _POS_MAP.get(pos)  # None if not a content POS
            if wn_pos is not None:
                syns = wordnet.synsets(word.lower(), pos=wn_pos)
                lemmas = [
                    lem.name().replace("_", " ")
                    for syn in syns
                    for lem in syn.lemmas()
                    if (lem.name().lower() != word.lower()
                        and lem.name().lower() not in _SYNONYM_BLOCKLIST
                        and "_" not in lem.name())  # skip multi-word phrases
                ]
                if lemmas:
                    result.append(rng.choice(lemmas))
                    continue
        result.append(word)

    return " ".join(result)


def augment(
    text: str,
    method: str = "both",
    p_synonym: float = 0.15,
    use_gpu: bool = False,
    seed: int | None = None,
) -> str:
    """
    Tạo một paraphrase của text.

    Args:
        text: Câu gốc.
        method: "backtranslate" | "synonym" | "both"
        p_synonym: Xác suất synonym substitution.
        use_gpu: Có dùng GPU cho back-translation không.
        seed: Random seed.

    Returns:
        Câu paraphrase.
    """
    result = text

    if method in ("backtranslate", "both"):
        result = back_translate(result, use_gpu=use_gpu)

    if method in ("synonym", "both"):
        result = synonym_substitute(result, p=p_synonym, seed=seed)

    return result


def augment_dataset(
    claims: list[str],
    labels: list[int],
    method: str = "synonym",
    p_synonym: float = 0.15,
    use_gpu: bool = False,
) -> tuple[list[str], list[int]]:
    """
    Augment toàn bộ dataset: trả về claims + augmented_claims với cùng labels.
    Dùng để double kích thước training set.
    """
    aug_claims = [
        augment(c, method=method, p_synonym=p_synonym, use_gpu=use_gpu, seed=i)
        for i, c in enumerate(claims)
    ]
    return claims + aug_claims, labels + labels


def random_deletion(text: str, p: float = 0.1, seed: int | None = None) -> str:
    """
    Randomly delete each non-stopword token with probability p.

    Stopwords (a, the, is, ...) are always kept to preserve grammaticality.
    If all content words would be deleted, returns original text.

    Args:
        text: Input sentence.
        p: Deletion probability per token.
        seed: Random seed for reproducibility.

    Returns:
        Sentence with some words removed.
    """
    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "if", "as", "not", "that", "this",
    }
    rng = random.Random(seed)
    words = text.split()
    if len(words) <= 3:
        return text  # too short to delete from

    kept = [w for w in words if w.lower() in _STOPWORDS or rng.random() > p]
    return " ".join(kept) if kept else text


def random_swap(text: str, n: int = 2, seed: int | None = None) -> str:
    """
    Randomly swap n pairs of adjacent words.

    Args:
        text: Input sentence.
        n: Number of swap operations.
        seed: Random seed.

    Returns:
        Sentence with some word pairs swapped.
    """
    rng = random.Random(seed)
    words = text.split()
    if len(words) < 2:
        return text

    for _ in range(n):
        i = rng.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def combined_augment(
    text: str,
    p_synonym: float = 0.15,
    p_deletion: float = 0.10,
    n_swap: int = 2,
    seed: int | None = None,
) -> str:
    """
    Apply all three augmentations sequentially:
      1. synonym_substitute (p_synonym)
      2. random_deletion    (p_deletion)
      3. random_swap        (n_swap pairs)

    Stronger than any single method; used for adversarial attack evaluation.

    Args:
        text: Input sentence.
        p_synonym: Synonym substitution probability.
        p_deletion: Word deletion probability.
        n_swap: Number of swap operations.
        seed: Base random seed (each op gets seed+offset for independence).

    Returns:
        Augmented sentence.
    """
    seed0 = seed if seed is not None else 0
    text = synonym_substitute(text, p=p_synonym, seed=seed0)
    text = random_deletion(text, p=p_deletion, seed=seed0 + 1)
    text = random_swap(text, n=n_swap, seed=seed0 + 2)
    return text


def ensure_nltk_data() -> None:
    """Download WordNet and POS tagger data if not present."""
    for resource, kind in [("wordnet", "corpora"), ("omw-1.4", "corpora"),
                           ("averaged_perceptron_tagger_eng", "taggers")]:
        try:
            nltk.data.find(f"{kind}/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


if __name__ == "__main__":
    ensure_nltk_data()

    sample = "The president signed a major healthcare reform bill last Tuesday."
    print(f"Original : {sample}")

    syn = synonym_substitute(sample, p=0.2, seed=42)
    print(f"Synonym  : {syn}")

    # Back-translation membutuhkan download model; skip jika offline
    try:
        bt = back_translate(sample)
        print(f"BackTrans: {bt}")
    except Exception as e:
        print(f"BackTrans: (skipped — {e})")
