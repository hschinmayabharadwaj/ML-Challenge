# Business Entity Resolution

This submission uses notebooks as the executable entry points. It is designed for the supplied challenge only and performs no external lookup.

## Notebooks

1. `01_eda_and_preprocessing.ipynb` profiles the TSVs, normalizes names/addresses, and checks ground-truth leakage.
2. `02_pair_features_and_validation.ipynb` creates blocked candidate pairs, trains a precision-oriented pair classifier, and selects a threshold on a deterministic validation split.
3. `03_inference_and_submission.ipynb` fits on all training pairs and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

The shared implementation is `src/entity_resolution.py`. Run notebooks from the repository root, or set `REPO_ROOT` in the first cell. For a quick smoke test, reduce `MAX_ROWS` in the notebook configuration; use `None` for the full data.

## Reproduce

```bash
python3 -m pip install -r code/business_entity_resolution/requirements.txt
jupyter lab
```

Open and run the notebooks in numeric order. Validate the generated files:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
