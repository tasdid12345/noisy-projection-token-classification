"""Reliability-reweighted token tagger vs. a standard LR tagger on noisily projected WikiANN-en labels."""
import io, json, os, re, sys, time
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split

SEED = 42
SEEDS = [42, 43, 44]          # randomised parts (label corruption, LightGBM) run with each seed
N_SENT = 6000                 # fixed subset of sentences (keeps runtime small)
TEST_SIZE = 0.25              # sentence-level split, done once
NOISE = 0.3                   # fraction of training tokens whose label is corrupted (projection noise)
DROP_Q = 0.2                  # lowest-reliability fraction of training tokens dropped by the improved model
BASE = "https://huggingface.co/datasets/unimelb-nlp/wikiann/resolve/main/en/"
FILES = ["train-00000-of-00001.parquet", "validation-00000-of-00001.parquet", "test-00000-of-00001.parquet"]
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]  # WikiANN ClassLabel order
TYPES = ["PER", "ORG", "LOC"]


def fetch(url, tries=5):
    """HTTP GET with timeout and exponential backoff on 429/5xx."""
    for k in range(tries):
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return r.content
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** k)
            continue
        r.raise_for_status()
    raise RuntimeError(f"download failed: {url}")


def load_subset():
    """Load English WikiANN parquet splits and draw a fixed random subset of sentences."""
    df = pd.concat([pd.read_parquet(io.BytesIO(fetch(BASE + f))) for f in FILES], ignore_index=True)
    df = df.sample(n=min(N_SENT, len(df)), random_state=SEED).reset_index(drop=True)
    return [list(t) for t in df.tokens], [[int(x) for x in t] for t in df.ner_tags]


def shape(w):
    """Word shape with runs collapsed, e.g. 'Smith' -> 'Xx', '1998' -> 'd'."""
    s = "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in w)
    return re.sub(r"(.)\1+", r"\1", s)


def sent_feats(toks):
    """Per-token feature dicts: word, shape, affixes, flags, +-2 neighbour words, +-1 neighbour shapes."""
    low = [t.lower() for t in toks]
    sh = [shape(t) for t in toks]
    out = []
    for i, t in enumerate(toks):
        d = {"bias": 1, "w": low[i], "sh": sh[i], "p3": low[i][:3], "s2": low[i][-2:], "s3": low[i][-3:],
             "title": t.istitle(), "upper": t.isupper(), "digit": any(c.isdigit() for c in t)}
        for o in (-2, -1, 1, 2):
            j = i + o
            d[f"w{o}"] = low[j] if 0 <= j < len(toks) else "<pad>"
            if abs(o) == 1:
                d[f"sh{o}"] = sh[j] if 0 <= j < len(toks) else "<pad>"
        out.append(d)
    return out


def corrupt(y, rng):
    """Replace a NOISE fraction of labels with a different label chosen uniformly (mimics projection errors)."""
    y = y.copy()
    idx = rng.random(len(y)) < NOISE
    shift = rng.integers(1, len(LABELS), size=len(y))
    y[idx] = (y[idx] + shift[idx]) % len(LABELS)
    return y


def lgb_model(seed):
    """Small LightGBM multiclass model; identical settings for reliability estimation and the improved model."""
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, colsample_bytree=0.5,
                              subsample=0.8, subsample_freq=1, random_state=seed, verbose=-1)


def run_seed(seed, Xtr, y_clean, groups, Xte):
    """Return test probabilities of the baseline and the improved model for one seed."""
    y_tr = corrupt(y_clean, np.random.default_rng(seed))
    # BASELINE: per-token logistic regression on all (noisy) labels, no filtering
    p_base = LogisticRegression(max_iter=500).fit(Xtr, y_tr).predict_proba(Xte)
    # IMPROVED step 1: reliability = out-of-fold LightGBM probability of each token's observed label
    oof = np.zeros(len(y_tr))
    for tr, va in GroupKFold(n_splits=3).split(Xtr, y_tr, groups):  # folds grouped by sentence
        m = lgb_model(seed).fit(Xtr[tr], y_tr[tr])
        oof[va] = m.predict_proba(Xtr[va])[np.arange(len(va)), y_tr[va]]
    # IMPROVED step 2: drop the least reliable tokens, weight the rest by reliability
    keep = oof >= np.quantile(oof, DROP_Q)
    p_lgb = lgb_model(seed).fit(Xtr[keep], y_tr[keep], sample_weight=oof[keep]).predict_proba(Xte)
    # IMPROVED step 3: average with the baseline's probabilities
    return p_base, 0.5 * (p_base + p_lgb)


def entities(seqs):
    """Set of (sentence id, start, end, type) spans from BIO tag sequences; an I- tag starts a span if needed."""
    out = set()
    for sid, tags in enumerate(seqs):
        start = typ = None
        for i, t in enumerate(list(tags) + ["O"]):
            if start is not None and (t == "O" or t.startswith("B-") or t[2:] != typ):
                out.add((sid, start, i, typ))
                start = None
            if t != "O" and start is None:
                start, typ = i, t[2:]
    return out


def f1_of(gs, ps):
    """Entity-level F1 of predicted span set ps against gold span set gs."""
    tp = len(gs & ps)
    pr = tp / len(ps) if ps else 0.0
    rc = tp / len(gs) if gs else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def score(gold, pred):
    """Entity-level micro F1, per-type F1 and their macro mean."""
    g, p = entities(gold), entities(pred)
    res = {"micro": f1_of(g, p)}
    for ty in TYPES:
        res[ty] = f1_of({x for x in g if x[3] == ty}, {x for x in p if x[3] == ty})
    res["macro"] = float(np.mean([res[ty] for ty in TYPES]))
    return res


def main():
    t0 = time.time()
    seqs_tok, seqs_tag = load_subset()
    print(f"[data] loaded {len(seqs_tok)} sentences from WikiANN-en")
    tr, te = train_test_split(np.arange(len(seqs_tok)), test_size=TEST_SIZE, random_state=SEED)

    # Features: fit vectoriser on training tokens only; drop features seen fewer than twice in training
    dv = DictVectorizer()
    Xall = dv.fit_transform([f for i in tr for f in sent_feats(seqs_tok[i])]).tocsr()
    cols = np.asarray((Xall != 0).sum(axis=0)).ravel() >= 2
    Xtr = Xall[:, cols]
    Xte = dv.transform([f for i in te for f in sent_feats(seqs_tok[i])]).tocsr()[:, cols]
    y_clean = np.array([t for i in tr for t in seqs_tag[i]])
    groups = np.array([s for s, i in enumerate(tr) for _ in seqs_tag[i]])
    y_te = np.array([t for i in te for t in seqs_tag[i]])
    te_lens = [len(seqs_tag[i]) for i in te]
    gold_te = [[LABELS[t] for t in seqs_tag[i]] for i in te]
    print(f"[data] train {len(tr)} sents / test {len(te)} sents, {Xtr.shape[1]} features")

    def evaluate(p):
        """Entity micro-F1 (main), macro-type F1 and token accuracy on the clean test set."""
        pred = p.argmax(1)
        tags = [[LABELS[t] for t in part] for part in np.split(pred, np.cumsum(te_lens)[:-1])]
        r = score(gold_te, tags)
        r["acc"] = float((pred == y_te).mean())
        return r

    runs = {"baseline": [], "improved": []}
    for s in SEEDS:
        print(f"[seed {s}] corrupting training labels and fitting both models")
        p_b, p_i = run_seed(s, Xtr, y_clean, groups, Xte)
        runs["baseline"].append(evaluate(p_b))
        runs["improved"].append(evaluate(p_i))
        print(f"  micro-F1 baseline={runs['baseline'][-1]['micro']:.4f} improved={runs['improved'][-1]['micro']:.4f}")

    # Aggregate over seeds
    m = {}
    for name in runs:
        v = [r["micro"] for r in runs[name]]
        m[f"{name}_score"] = float(np.mean(v))
        m[f"{name}_score_std"] = float(np.std(v))
        m[f"{name}_f1"] = float(np.mean([r["macro"] for r in runs[name]]))
        m[f"{name}_acc"] = float(np.mean([r["acc"] for r in runs[name]]))
        for ty in TYPES:
            m[f"{name}_f1_{ty}"] = float(np.mean([r[ty] for r in runs[name]]))
    m["improvement"] = m["improved_score"] - m["baseline_score"]  # higher micro-F1 is better
    if not all(np.isfinite(list(m.values()))) or m["baseline_score"] <= 0 or m["improved_score"] <= 0:
        print("no meaningful metrics produced")
        sys.exit(1)

    # Chart
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    names = ["baseline", "improved"]
    ax.bar(names, [m[f"{n}_score"] for n in names], yerr=[m[f"{n}_score_std"] for n in names],
           capsize=4, color=["grey", "steelblue"])
    ax.set_ylabel("entity micro-F1 (clean test)")
    ax.set_title("Mean ± std over seeds")
    fig.tight_layout()
    fig.savefig("figures/entity_f1.png", dpi=120)

    results = {
        "question": "Can a cross-validated reliability reweighting of noisily projected token labels beat a "
                    "standard token classifier trained on all projected labels, on a public NER stand-in?",
        "method": "Random BIO-label corruption of training tokens mimics projection noise; baseline = logistic "
                  "regression on all noisy labels; improved = LightGBM reliability scores from out-of-fold "
                  "predictions, drop lowest-reliability tokens, reliability-weighted LightGBM, averaged with baseline.",
        "dataset": {
            "name": "WikiANN (English) NER, fixed sentence subset",
            "source_url": BASE,
            "n_rows": len(seqs_tok),
            "n_train": len(tr),
            "n_test": len(te),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "entity-level micro-F1 on clean test set",
        "higher_is_better": True,
        "baseline": "Per-token logistic regression on window features trained on all corrupted training labels.",
        "improved": "LightGBM on the same features, trained on tokens above the reliability cut-off and weighted by "
                    "out-of-fold reliability; probabilities averaged with the baseline.",
        "metrics": m,
        "notes": "Corruption is synthetic uniform label noise, not real translation-based projection. The "
                 "corruption rate, drop fraction and blend weight were fixed a priori; no hyper-parameter search "
                 "was run, because clean training labels are unavailable for selection. Only one sentence split "
                 "is used; seed std reflects corruption and model randomness only. Test labels are clean. The "
                 "paper's own numbers are not reproduced here and no comparison to them is claimed.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] baseline={m['baseline_score']:.4f} improved={m['improved_score']:.4f} "
          f"in {time.time() - t0:.0f}s; wrote results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # non-zero exit when no meaningful metrics can be produced
        print(f"error: {e}")
        sys.exit(1)
