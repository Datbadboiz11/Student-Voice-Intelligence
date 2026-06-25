from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


STOPWORDS = {
    "anh", "ban", "bi", "cua", "cho", "co", "con", "cung", "da", "dang", "de", "den", "duoc",
    "gi", "hay", "hoc", "khong", "khi", "la", "lam", "lien", "luc", "mot", "nhung", "o", "phong",
    "qua", "rat", "sinh", "su", "tai", "the", "thuong", "truong", "va", "ve", "vien", "voi", "yeu",
}
POSITIVE_NETWORK_TERMS = {"khoe", "muot", "ngon", "on", "tot", "lag"}
NEGATIVE_NETWORK_TERMS = {"yeu", "mat", "khong", "cham", "roi", "loi"}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return value.replace("đ", "d").replace("wi-fi", "wifi")


def tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalize(value)) if len(token) > 1}


def label(query: str, text: str) -> int:
    query_terms = tokens(query) - STOPWORDS
    text_terms = tokens(text)
    overlap = query_terms & text_terms
    query_normalized = normalize(query)
    text_normalized = normalize(text)
    if "wifi" in query_terms and "wifi" in text_terms:
        has_negative_intent = bool(NEGATIVE_NETWORK_TERMS & text_terms)
        has_positive_intent = bool(POSITIVE_NETWORK_TERMS & text_terms)
        if has_positive_intent and not has_negative_intent:
            return 0
    if len(overlap) >= 3:
        return 2
    if len(overlap) == 2:
        return 2
    if len(overlap) == 1:
        term = next(iter(overlap))
        return 2 if term in {"wifi", "chieu", "hocphi", "cantin", "dieuhoa", "vesinh"} else 1
    if any(phrase in text_normalized for phrase in ("may chieu", "hoc phi", "cau lac bo", "hoc phan")) and any(
        phrase in query_normalized for phrase in ("may chieu", "hoc phi", "cau lac bo", "hoc phan")
    ):
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply AI-assisted preliminary relevance labels.")
    parser.add_argument("--judgments", type=Path, default=Path("data/evaluation/retrieval_judgments.csv"))
    args = parser.parse_args()
    with args.judgments.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = handle.seek(0) or []
    if not rows:
        raise ValueError("Judgment file is empty.")
    fieldnames = list(rows[0])
    for row in rows:
        row["relevance"] = str(label(row["query"], row["text"]))
        row["review_note"] = "AI-assisted preliminary label; requires human audit."
    with args.judgments.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Applied AI-assisted preliminary labels to {len(rows)} candidates in {args.judgments}.")


if __name__ == "__main__":
    main()
