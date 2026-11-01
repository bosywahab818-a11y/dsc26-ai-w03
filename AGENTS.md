# Tawseel agent rules

## Verify before changing
- Run `uv run pytest` from the repository root; the suite is configured in `pyproject.toml`.
- Use Python 3.12 and the exact dependencies in `pyproject.toml`; do not add or upgrade packages without checking the DSC grader constraints.
- The grader runs offline and replaces `tawseel/data.py` with its own copy, so do not depend on local-only data behavior.
- Read `README.md` for the task contract and hidden-test expectations before changing model behavior.
- Keep implementation in `tawseel/model.py`; public tests are in `tests/test_public.py`.
- Record baseline metrics and the plain-language interpretation in `RESULTS.md`.
- Record AI assistance and leakage checks in `AI_USAGE.md`.

## Data and time boundaries
- Generate development data with `tawseel.data.make_parcels(n=4000, seed=...)`; do not treat generated rows as real delivery records.
- Customer history must use only strictly earlier `created_at` values; equal timestamps are ties, not prior events.
- Keep the chronological split: test size is ceil(n * fraction), and every row at the boundary timestamp belongs to test.
- Only use fields known when a parcel is created; exclude `courier_notes`, `delivered_at`, `failed`, and `parcel_id` from model features.
- Fit preprocessing and estimators on training data only; never fit transformations on the full dataset before splitting.
- Preserve input row order, index, and data when adding history features; do not mutate caller-owned frames.

## Models, evaluation, and artifacts
- Approved baseline model: scikit-learn `LogisticRegression`; comparator: `DummyClassifier(strategy="prior")`.
- Use scikit-learn 1.9.1 as pinned in `pyproject.toml`; do not silently substitute model families or versions.
- Evaluate on the held-out chronological test set with ROC-AUC, average precision (PR-AUC), Brier score, base rate, and sample count.
- Compare model and dummy on identical train/test partitions; report PR-AUC lift as model minus dummy.
- Data generator/source lives in `tawseel/data.py`; model code lives in `tawseel/model.py`; results live in `RESULTS.md`.
- No persisted model artifact is currently part of the repository; do not claim a model file exists or commit generated binaries without a task requirement.
- Recheck multiple seeds and inspect for leakage; ROC-AUC near 0.99 on this synthetic task is a leakage warning, not a success.

## Prompts and changes
- There is no established prompt library yet; if reusable prompts are added, store them under `prompts/` and add tests or examples that verify their required inputs and output format.
- Keep changes scoped to the requested task; do not edit the DSC-provided data generator or dependency pins.
- Before finishing, run `uv run pytest` and report the actual command result; do not claim unrun checks passed.
