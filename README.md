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

## Reference

This project is motivated by the following paper:

> B Ebing, G Glavaš (2025). *The devil is in the word alignment details: On translation-based cross-lingual transfer for token classification tasks*. Findings of the Association for Computational. https://aclanthology.org/2025.findings-acl.931/

Paper author: Prof. Goran Glavaš, Julius-Maximilians-Universität Würzburg (goran.glavas@uni-wuerzburg.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
