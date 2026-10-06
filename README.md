# Human Activity Recognition: Dimensionality Reduction

Can 561 smartphone-sensor features be represented in fewer dimensions while retaining useful activity structure?

The project compares a reusable linear compressor (PCA) with two nonlinear
embeddings (t-SNE and UMAP). It is an unsupervised analysis: `Activity` is
never used as an input or a tuning target. It is revealed only after fitting
to interpret the maps.

## Dataset

[`data/train.csv`](data/train.csv) contains 7,352 recordings and
[`data/test.csv`](data/test.csv) contains 2,947 held-out recordings from
[Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones).

Each recording has 561 numeric sensor-derived features, a `subject` identifier,
and an `Activity` reference label. The analysis uses only the 561 features.
There are no missing feature values or exact duplicate training rows.

## Method and verified choices

1. **Preprocessing:** `StandardScaler` is fitted on training features and then
   applied to test features.
2. **PCA:** 95% explained variance selects 102 components. The 2D PCA chart
   retains 57.36% of variance and is used only for visual inspection.
3. **t-SNE:** a fixed, label-free 3,000-recording sample compares
   perplexities 5, 15, 30, 50, and 75 using trustworthiness. Perplexity 75 is
   selected and checked with seeds 42 and 7.
4. **UMAP:** a label-free search selects `n_neighbors=5` and `min_dist=0.0`.
   The final map is fitted on all training recordings and transforms the
   held-out test recordings. Test trustworthiness is 0.9163 for both checked
   seeds (42 and 7).

## Findings

- `LAYING` is the most distinct activity family.
- `SITTING` and `STANDING` substantially overlap.
- Walking-related activities form a more varied dynamic-motion region with
  meaningful local subgroups.
- PCA is the reusable linear compression method; t-SNE gives a local visual
  snapshot; UMAP provides a reusable nonlinear map for unseen data.

The 2D charts are diagnostic, not classifier scores. t-SNE and UMAP axes have
no physical interpretation, and distances between far-apart groups should not
be treated as literal activity distances.

## Reproduce

From the project directory:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
uv run --python .venv/bin/python jupyter nbconvert --to notebook --execute --inplace \
  dimensionality-reduction-har.ipynb --ExecutePreprocessor.timeout=600
```

The executed notebook contains the audit, parameter-search tables, plots, and
held-out UMAP check.

Raw CSV files are intentionally excluded from Git. Download the dataset from
the source linked above and place `train.csv` and `test.csv` in `data/` before
running the notebook.
