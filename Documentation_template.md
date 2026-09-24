# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
This solution treats the task as high-precision multi-match entity resolution. It normalizes business names and addresses, generates country-aware candidates using exact name, name-prefix, postal-code, and address-number blocks, and scores each candidate with string-similarity and token-overlap features. A balanced logistic pair classifier and a validation-selected F0.5 threshold produce final matches while preserving empty predictions for likely singletons.

---

## 2. Methodology

### 2.1 Problem Analysis
The records contain business names, addresses, countries, and source-prefixed IDs. Noise includes punctuation, casing, legal suffixes, transliteration, abbreviations, reordered address components, typos, and missing address fragments. Countries are open-set labels, so France is handled without a fixed vocabulary. There are no image files or image references in the schema, so image encoders and multimodal towers would add no signal.

### 2.2 Solution Strategy
Normalize each source, union several conservative blocking keys, compute pairwise similarities for blocked pairs, fit a balanced logistic classifier, and select its threshold against macro F0.5 on a fixed validation split.

**Approach Type:** Blocking + pairwise classifier  
**Core Innovation:** Multiple independent country-aware blocks preserve recall while character and token similarities keep precision high under macro F0.5.

---

## 3. Candidate Generation (Blocking)
The final candidate set is the exact set passed to the pair model and is written to `candidate_pairs.tsv`.

- **Blocking keys used:** Country plus exact normalized name, six-character compact-name prefix, detected 4-6 digit postal code, or each address number.
- **Candidate pairs generated:** Printed by the validation and inference notebooks for the selected run.
- **How you ensured true matches were not lost:** Independent keys are unioned before ranking. Empty postal keys are ignored, country values remain open-set strings, and oversized blocks are limited to 250 candidates after preliminary similarity ranking.

---

## 4. Matching Model

**Features used:**
- Name features: normalized character ratio, token-set ratio, informative-token Jaccard, and exact compact-name equality.
- Address features: normalized character ratio, token-set ratio, informative-token Jaccard, address-number overlap, and postal-code equality.
- Other: country equality.

**Model type:** Balanced logistic regression over pair features. It is small and auditable for the large candidate set; positives and a deterministic 5:1 sample of negatives are used.  
**Threshold selection method:** Candidate thresholds from 0.30 through 0.95 are evaluated on a fixed validation split, using one F0.5 score per Source 1 entity so singleton behavior is represented.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** Printed by `02_pair_features_and_validation.ipynb` for the current run.
- **Common false positives (wrong merges):** Common names sharing a country and address number, especially when the number is a building or road number rather than a unique location.
- **Common false negatives (missed matches):** Records with both heavily corrupted names and incomplete addresses can miss every blocking key; the candidate cap can affect very large common-name blocks.

---

## 6. Conclusion
The pipeline is precision-oriented because the official metric penalizes false merges more than missed matches. It uses only supplied records, supports unseen country labels, and emits both the scored match file and the auditable candidate file.

---

## Appendix

### A. Code Artefacts
The complete implementation is under `code/business_entity_resolution/`. Install `requirements.txt`, run the three JSON notebooks in numeric order, and validate using `utils/validate_submission.py`. No embedding scripts are included because the actual dataset contains structured entity-resolution fields and no image modality.

### B. Additional Results
The EDA notebook reports counts, missingness, country frequencies, normalized-name uniqueness, duplicate IDs, and ground-truth match cardinalities. The validation notebook displays high-scoring validation pairs for manual error analysis.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
