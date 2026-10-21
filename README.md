# Reliability reweighting of noisily projected NER labels

Translation-based cross-lingual transfer for token classification requires projecting labels from source to target text, and the projection step is error-prone. The paper "The devil is in the word alignment details: On translation-based cross-lingual transfer for token classification tasks" (2025) shows that details of this projection strongly affect downstream accuracy. This experiment asks a narrower question on a public NER stand-in: can a model that estimates how reliable each projected label is, and downweights or drops unreliable ones, beat a standard classifier trained on every projected label?

**Question:** Can a cross-validated reliability reweighting of noisily projected token labels beat a standard token classifier trained on all projected labels, on a public NER stand-in?

**Dataset:** WikiANN English NER, loaded over HTTP from the Hugging Face parquet files, using a fixed random subset of sentences. This is not the paper's own dataset. Label projection is simulated by randomly corrupting a fraction of training BIO labels; test labels stay clean.

**Baseline:** a per-token logistic regression on window features (word shape, affixes, neighbouring words), trained on all corrupted labels with no filtering.

**Improved model:** a LightGBM on the same features. Out-of-fold predicted probabilities give each training token a reliability score. The model drops the least reliable tokens, weights the rest by reliability, and averages its probabilities with the baseline's.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results are written to results.json (and figures/entity_f1.png).

## Results (latest run)

On WikiANN English NER (1,500 clean test sentences), entity-level micro-F1 was 30.5% for a logistic regression trained on all synthetically corrupted labels and 40.6% for my reliability-reweighted LightGBM blend, a single-split result.

| Metric | Value |
|---|---|
| baseline_score | 0.3049 |
| baseline_score_std | 0.005863 |
| baseline_f1 | 0.3049 |
| baseline_acc | 0.7623 |
| baseline_f1_PER | 0.3553 |
| baseline_f1_ORG | 0.1731 |
| baseline_f1_LOC | 0.3863 |
| improved_score | 0.4064 |
| improved_score_std | 0.007138 |
| improved_f1 | 0.4068 |
| improved_acc | 0.8034 |
| improved_f1_PER | 0.4754 |
| improved_f1_ORG | 0.2421 |
| improved_f1_LOC | 0.5028 |
| improvement | 0.1015 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Corruption is synthetic uniform label noise, not real translation-based projection. The corruption rate, drop fraction and blend weight were fixed a priori; no hyper-parameter search was run, because clean training labels are unavailable for selection. Only one sentence split is used; seed std reflects corruption and model randomness only. Test labels are clean. The paper's own numbers are not reproduced here and no comparison to them is claimed.

## Reference

This project is motivated by the following paper:

> B Ebing, G Glavaš (2025). *The devil is in the word alignment details: On translation-based cross-lingual transfer for token classification tasks*. Findings of the Association for Computational. https://aclanthology.org/2025.findings-acl.931/

Paper author: Prof. Goran Glavaš, Julius-Maximilians-Universität Würzburg (goran.glavas@uni-wuerzburg.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
