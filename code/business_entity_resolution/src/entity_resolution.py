"""Scalable, no-external-data entity resolution utilities for the challenge."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
NUMBER_RE = re.compile(r"\d+[A-Za-z]?")
STOPWORDS = {"the", "and", "of", "for", "inc", "llc", "ltd", "limited", "company", "co"}


def normalize(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(TOKEN_RE.findall(text))


def tokens(value: object) -> Set[str]:
    return set(normalize(value).split())


def informative_tokens(value: object) -> Set[str]:
    return {token for token in tokens(value) if token not in STOPWORDS and len(token) > 1}


def number_tokens(value: object) -> Set[str]:
    return set(NUMBER_RE.findall(normalize(value)))


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["name_norm"] = result["business_name"].map(normalize)
    result["address_norm"] = result["business_address"].map(normalize)
    result["name_tokens"] = result["business_name"].map(informative_tokens)
    result["address_tokens"] = result["business_address"].map(informative_tokens)
    result["address_numbers"] = result["business_address"].map(number_tokens)
    result["name_key"] = result["name_norm"].str.replace(" ", "", regex=False)
    result["country_norm"] = result["country"].map(normalize)
    result["postal_key"] = result["address_norm"].str.extract(r"(\d{4,6})", expand=False).fillna("")
    return result


def _add(index: Dict[str, List[int]], key: str, row_id: int) -> None:
    if key:
        index[key].append(row_id)


def build_block_index(target: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    indexes = {name: defaultdict(list) for name in ("name", "name_prefix", "postal", "number")}
    for row_id, row in target.reset_index(drop=True).iterrows():
        name = row["name_key"]
        _add(indexes["name"], f"{row.country_norm}|{name}", row_id)
        _add(indexes["name_prefix"], f"{row.country_norm}|{name[:6]}", row_id)
        _add(indexes["postal"], f"{row.country_norm}|{row.postal_key}", row_id)
        for number in row.address_numbers:
            _add(indexes["number"], f"{row.country_norm}|{number}", row_id)
    return indexes


def candidate_rows(source: pd.Series, target: pd.DataFrame, indexes: Mapping[str, Mapping[str, Sequence[int]]], max_candidates: int = 250) -> List[int]:
    country = source.country_norm
    ids: Set[int] = set()
    name = source.name_key
    ids.update(indexes["name"].get(f"{country}|{name}", ()))
    if len(name) >= 6:
        ids.update(indexes["name_prefix"].get(f"{country}|{name[:6]}", ()))
    if source.postal_key:
        ids.update(indexes["postal"].get(f"{country}|{source.postal_key}", ()))
    for number in source.address_numbers:
        ids.update(indexes["number"].get(f"{country}|{number}", ()))
    if len(ids) <= max_candidates:
        return sorted(ids)
    ranked = sorted(ids, key=lambda i: pair_features(source, target.iloc[i]), reverse=True)
    return ranked[:max_candidates]


def pair_features(left: pd.Series, right: pd.Series) -> np.ndarray:
    name_tokens_left = left.name_tokens
    name_tokens_right = right.name_tokens
    address_tokens_left = left.address_tokens
    address_tokens_right = right.address_tokens
    name_union = name_tokens_left | name_tokens_right
    address_union = address_tokens_left | address_tokens_right
    name_jaccard = len(name_tokens_left & name_tokens_right) / max(len(name_union), 1)
    address_jaccard = len(address_tokens_left & address_tokens_right) / max(len(address_union), 1)
    number_overlap = len(left.address_numbers & right.address_numbers) / max(len(left.address_numbers | right.address_numbers), 1)
    return np.array([
        fuzz.ratio(left.name_norm, right.name_norm) / 100.0,
        fuzz.token_set_ratio(left.name_norm, right.name_norm) / 100.0,
        name_jaccard,
        fuzz.ratio(left.address_norm, right.address_norm) / 100.0,
        fuzz.token_set_ratio(left.address_norm, right.address_norm) / 100.0,
        address_jaccard,
        number_overlap,
        float(left.country_norm == right.country_norm),
        float(bool(left.postal_key and left.postal_key == right.postal_key)),
        float(left.name_key == right.name_key),
    ], dtype=np.float32)


def make_pairs(source1: pd.DataFrame, targets: pd.DataFrame, max_candidates: int = 250) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    indexes = build_block_index(targets)
    rows: List[np.ndarray] = []
    pair_ids: List[Tuple[str, str]] = []
    candidate_map: Dict[str, List[str]] = {}
    targets = targets.reset_index(drop=True)
    for _, source in source1.iterrows():
        candidate_idx = candidate_rows(source, targets, indexes, max_candidates)
        candidate_map[str(source.entity_id)] = [str(targets.iloc[i].entity_id) for i in candidate_idx]
        for idx in candidate_idx:
            rows.append(pair_features(source, targets.iloc[idx]))
            pair_ids.append((str(source.entity_id), str(targets.iloc[idx].entity_id)))
    features = pd.DataFrame(np.vstack(rows) if rows else np.empty((0, 10), dtype=np.float32))
    features.insert(0, "source1_entity_id", [x[0] for x in pair_ids])
    features.insert(1, "candidate_entity_id", [x[1] for x in pair_ids])
    return features, candidate_map


def labelled_pairs(features: pd.DataFrame, ground_truth: pd.DataFrame, negative_ratio: int = 5, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    truth = {str(row.source1_entity_id): set(str(row.matched_entity_ids).split(",")) if str(row.matched_entity_ids) else set() for _, row in ground_truth.iterrows()}
    labels = np.array([int(candidate in truth.get(source, set())) for source, candidate in zip(features.source1_entity_id, features.candidate_entity_id)], dtype=np.int8)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    keep_negatives = rng.choice(negatives, size=min(len(negatives), max(len(positives) * negative_ratio, 1)), replace=False) if len(negatives) else np.array([], dtype=int)
    keep = np.concatenate([positives, keep_negatives])
    return features.iloc[keep].reset_index(drop=True), labels[keep]


def train_pair_model(features: pd.DataFrame, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
    model.fit(features.iloc[:, 2:].to_numpy(), labels)
    return model


def best_f05_threshold(labels: np.ndarray, probabilities: np.ndarray, groups: Iterable[str]) -> Tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.linspace(0.30, 0.95, 66):
        score = macro_f05(labels, probabilities >= threshold, groups)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def macro_f05(labels: np.ndarray, predictions: np.ndarray, groups: Iterable[str]) -> float:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        grouped[str(group)].append(idx)
    scores = []
    for indices in grouped.values():
        y_true = np.asarray(labels[indices], dtype=bool)
        y_pred = np.asarray(predictions[indices], dtype=bool)
        tp = int(np.sum(y_true & y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall:
            scores.append(1.25 * precision * recall / (0.25 * precision + recall))
        else:
            scores.append(float(not y_true.any() and not y_pred.any()))
    return float(np.mean(scores)) if scores else 0.0


def write_submission(source1: pd.DataFrame, features: pd.DataFrame, probabilities: np.ndarray, threshold: float, candidate_map: Mapping[str, Sequence[str]], matching_path: str, candidate_path: str) -> None:
    source_ids = source1.entity_id.astype(str).tolist()
    matches: Dict[str, List[str]] = {source_id: [] for source_id in source_ids}
    for (source_id, candidate_id), probability in zip(features[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None), probabilities):
        if probability >= threshold:
            matches[source_id].append(candidate_id)
    match_frame = pd.DataFrame({"source1_entity_id": source_ids, "matched_entity_ids": [",".join(sorted(set(matches[s]))) for s in source_ids]})
    candidate_frame = pd.DataFrame({"source1_entity_id": source_ids, "candidate_entity_ids": [",".join(sorted(set(candidate_map.get(s, [])))) for s in source_ids]})
    match_frame.to_csv(matching_path, sep="\t", index=False)
    candidate_frame.to_csv(candidate_path, sep="\t", index=False)
