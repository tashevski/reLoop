"""
Read-only data + evaluation harness for the regulatory-genomics autoresearch
experiment.

Task: binary classification of human DNA sequences -- is a 500 bp sequence an
ENHANCER (label 1) or a control region (label 0)? Dataset: `human_enhancers_cohn`
from the Genomic Benchmarks collection (Grešová et al., 2023). The sequences are
derived from the human reference genome -- a public benchmark, no PHI.

DO NOT MODIFY THIS FILE. It defines:
  - the LOCKED train / validation / test splits, and
  - the ground-truth metrics (primary AUROC, guardrails, per-GC-bin AUROC).
Editing it = gaming the metric and/or leaking the held-out test set.

Run it once to download + cache the data and sanity-check the splits:
    uv run prepare.py
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Fixed, LOCKED configuration -- the contract the agent develops against.
# --------------------------------------------------------------------------
CACHE_DIR = os.path.expanduser("~/.cache/autoresearch-genomics")
DATASET = "human_enhancers_cohn"
HF_REPO = "katarinagresova/Genomic_Benchmarks_human_enhancers_cohn"

SEQ_LEN = 500          # this benchmark is fixed-length; asserted on load
BASES = "ACGT"         # 4-channel one-hot; non-ACGT (e.g. N) -> all-zero column
SPLIT_SEED = 1337
VAL_FRAC = 0.15        # carved out of the authors' train split, label-stratified

PRIMARY_METRIC_NAME = "AUROC"
HIGHER_IS_BETTER = True
PUBLISHED_ACCURACY_BASELINE = 0.695  # CNN baseline from the Genomic Benchmarks paper

# Guardrails: HARD feasibility constraints. A model that violates any of these is
# infeasible and can never enter the Pareto front, regardless of its objectives.
GUARDRAILS = {
    "calibration_ece": {"max": 0.10},  # expected calibration error must stay <= 0.10
    "mcc": {"min": 0.30},              # Matthews corr coef floor at threshold 0.5
}
ECE_BINS = 15
N_GC_BINS = 4          # GC-content quartile bins for the robustness breakdown

# Multi-objective optimization. Among guardrail-FEASIBLE models we do not collapse
# to a single number; we optimize these objectives jointly and keep a model iff it
# is Pareto-non-dominated (see dominates() / pareto_decision()). They genuinely
# compete: raw discrimination vs. calibration vs. worst-subgroup robustness.
OBJECTIVES = {
    "auroc": "max",            # overall discrimination
    "calibration_ece": "min",  # calibration quality
    "gc_worst_auroc": "max",   # robustness: AUROC of the weakest GC-content bin
}

# Direct-resolve URLs (fallback). The hex hash is content-addressed and may drift
# if the dataset is re-uploaded, so `_resolve_urls()` tries the HF tree API first.
_HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data"
_FALLBACK_URLS = {
    "train": [f"{_HF_RESOLVE}/train-00000-of-00001-308a8a054210be5f.parquet"],
    "test": [f"{_HF_RESOLVE}/test-00000-of-00001-b9d4d53093c06044.parquet"],
}
_TREE_API = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/data"


# --------------------------------------------------------------------------
# Data loading (download + cache + locked split)
# --------------------------------------------------------------------------
def _resolve_urls():
    """Resolve current parquet URLs via the HF tree API; fall back to hardcoded."""
    urls = {k: list(v) for k, v in _FALLBACK_URLS.items()}
    try:
        resp = requests.get(_TREE_API, timeout=30)
        resp.raise_for_status()
        names = [item["path"] for item in resp.json()
                 if item.get("path", "").endswith(".parquet")]
        for split in ("train", "test"):
            match = next((n for n in names
                          if re.search(rf"(^|/){split}-.*\.parquet$", n)), None)
            if match:
                fname = match.split("/")[-1]
                # tree paths are relative to repo root; resolve serves from root too
                urls[split] = [f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{match}",
                               f"{_HF_RESOLVE}/{fname}"] + urls[split]
    except Exception:
        pass  # offline / API change -> use hardcoded fallback
    return urls


def _download(split, resolved=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    urls = (resolved or _resolve_urls())[split]
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=120)  # follows the signed-CDN redirect
            resp.raise_for_status()
            with open(os.path.join(CACHE_DIR, f"{split}.parquet"), "wb") as f:
                f.write(resp.content)
            return
        except Exception as e:  # noqa: BLE001 - report whatever went wrong
            last_err = e
    raise RuntimeError(
        f"Could not download the {split} split from any URL ({urls}): {last_err}"
    )


def _pad(seq):
    """Right-pad with N / truncate to SEQ_LEN. (enhancers_cohn is already 500 bp.)"""
    if len(seq) >= SEQ_LEN:
        return seq[:SEQ_LEN]
    return seq + "N" * (SEQ_LEN - len(seq))


def _load_raw(split, resolved=None):
    path = os.path.join(CACHE_DIR, f"{split}.parquet")
    if not os.path.exists(path):
        _download(split, resolved)
    df = pd.read_parquet(path)  # columns: seq (str), label (int)
    seqs = df["seq"].astype(str).str.upper().to_numpy()
    if not all(len(s) == SEQ_LEN for s in seqs):
        seqs = np.array([_pad(s) for s in seqs])
    y = df["label"].to_numpy().astype(int)
    return seqs, y


def one_hot(seqs):
    """Encode an array of DNA strings as (N, 4, SEQ_LEN) float32, channel-first.

    Non-ACGT characters (e.g. N) map to an all-zero column.
    """
    base_idx = {b: i for i, b in enumerate(BASES)}
    n = len(seqs)
    out = np.zeros((n, 4, SEQ_LEN), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            k = base_idx.get(ch, -1)
            if k >= 0:
                out[i, k, j] = 1.0
    return out


def gc_content(seqs):
    """Fraction of G/C bases per sequence."""
    return np.array(
        [(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs],
        dtype=np.float64,
    )


def _stratified_val_carve(y, seed):
    """Label-stratified split of train indices into (train, val) by VAL_FRAC."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * VAL_FRAC))
        va += list(idx[:n_val])
        tr += list(idx[n_val:])
    return np.array(sorted(tr)), np.array(sorted(va))


def load_splits():
    """Return the LOCKED splits. Identical on every call (fixed seed)."""
    resolved = _resolve_urls()
    seqs_tr_full, y_tr_full = _load_raw("train", resolved)
    seqs_te, y_te = _load_raw("test", resolved)

    tr_idx, va_idx = _stratified_val_carve(y_tr_full, SPLIT_SEED)
    seqs_tr, y_tr = seqs_tr_full[tr_idx], y_tr_full[tr_idx]
    seqs_va, y_va = seqs_tr_full[va_idx], y_tr_full[va_idx]

    # GC-quartile bins from TRAIN-derived edges (no leakage), applied to val/test.
    gc_tr = gc_content(seqs_tr)
    edges = np.quantile(gc_tr, np.linspace(0, 1, N_GC_BINS + 1)[1:-1])
    gc_va, gc_te = gc_content(seqs_va), gc_content(seqs_te)
    gcbin_va = np.digitize(gc_va, edges)
    gcbin_te = np.digitize(gc_te, edges)

    return {
        "X_train": one_hot(seqs_tr), "y_train": y_tr, "seqs_train": seqs_tr,
        "X_val": one_hot(seqs_va), "y_val": y_va, "seqs_val": seqs_va,
        "gc_val": gc_va, "gcbin_val": gcbin_va,
        "X_test": one_hot(seqs_te), "y_test": y_te, "seqs_test": seqs_te,
        "gc_test": gc_te, "gcbin_test": gcbin_te,
        "gc_edges": edges,
    }


# --------------------------------------------------------------------------
# Ground-truth metrics (the agent must not reimplement or bypass these)
# --------------------------------------------------------------------------
def auroc(y_true, y_score):
    """AUROC via the Mann-Whitney U statistic, with tie handling."""
    y_true = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1.0  # average rank, 1-based
        i = j + 1
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = ranks_sorted
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(y_true, y_prob, n_bins=ECE_BINS):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(p)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def mcc(y_true, y_prob, threshold=0.5):
    """Matthews correlation coefficient at a probability threshold."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def accuracy(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float((pred == y_true).mean())


def evaluate(y_true, y_prob, gc_bins=None):
    """Primary metric, guardrails, and (optionally) per-GC-bin AUROC."""
    metrics = {
        "primary_metric": auroc(y_true, y_prob),
        "accuracy": accuracy(y_true, y_prob),
        "guardrails": {
            "calibration_ece": expected_calibration_error(y_true, y_prob),
            "mcc": mcc(y_true, y_prob),
        },
    }
    if gc_bins is not None:
        gc_bins = np.asarray(gc_bins)
        metrics["subgroups"] = {
            f"gc_q{b}": auroc(np.asarray(y_true)[gc_bins == b],
                              np.asarray(y_prob)[gc_bins == b])
            for b in range(N_GC_BINS)
        }
    return metrics


def guardrails_ok(guardrails):
    if guardrails["calibration_ece"] > GUARDRAILS["calibration_ece"]["max"]:
        return False
    if guardrails["mcc"] < GUARDRAILS["mcc"]["min"]:
        return False
    return True


def aggregate(seed_metrics):
    """Aggregate per-seed metrics into a single dict (means; std for AUROC)."""
    prim = np.array([m["primary_metric"] for m in seed_metrics], dtype=float)
    acc = np.array([m["accuracy"] for m in seed_metrics], dtype=float)
    guard = {k: float(np.nanmean([m["guardrails"][k] for m in seed_metrics]))
             for k in seed_metrics[0]["guardrails"].keys()}
    agg = {
        "auroc": float(np.nanmean(prim)),
        "auroc_std": float(np.nanstd(prim)),
        "accuracy": float(np.nanmean(acc)),
        "guardrails": guard,
    }
    if "subgroups" in seed_metrics[0]:
        sg = {k: float(np.nanmean([m["subgroups"][k] for m in seed_metrics]))
              for k in seed_metrics[0]["subgroups"].keys()}
        agg["subgroups"] = sg
        agg["gc_worst_auroc"] = float(np.nanmin(list(sg.values())))
    return agg


def objective_vector(agg):
    """Extract the OBJECTIVES vector from an aggregated-metrics dict."""
    source = {
        "auroc": agg.get("auroc"),
        "calibration_ece": agg.get("guardrails", {}).get("calibration_ece"),
        "gc_worst_auroc": agg.get("gc_worst_auroc"),
    }
    return {name: source[name] for name in OBJECTIVES}


def summarize(seed_metrics, peak_mem_gb, train_seconds):
    """Print the canonical summary block; return the aggregated-metrics dict."""
    agg = aggregate(seed_metrics)
    guard = agg["guardrails"]
    objs = objective_vector(agg)

    lines = ["---"]
    lines.append(f"primary_metric:   {agg['auroc']:.4f}")
    lines.append(f"metric_std:       {agg['auroc_std']:.4f}")
    lines.append(f"accuracy:         {agg['accuracy']:.4f}")
    lines.append("guardrails:       {"
                 + ", ".join(f"{k}: {v:.4f}" for k, v in guard.items()) + "}")
    if "subgroups" in agg:
        lines.append("subgroups:        {"
                     + ", ".join(f"{k}: {v:.4f}" for k, v in agg["subgroups"].items()) + "}")
        lines.append(f"gc_worst_auroc:   {agg['gc_worst_auroc']:.4f}")
    lines.append("objectives:       {"
                 + ", ".join(f"{k}: {v:.4f}" for k, v in objs.items()) + "}")
    lines.append(f"guardrails_ok:    {'yes' if guardrails_ok(guard) else 'no'}")
    lines.append(f"n_seeds:          {len(seed_metrics)}")
    lines.append(f"peak_mem_gb:      {peak_mem_gb:.2f}")
    lines.append(f"train_seconds:    {train_seconds:.1f}")
    print("\n".join(lines))
    return agg


# --------------------------------------------------------------------------
# Multi-objective (Pareto) optimization. Among guardrail-feasible models, a
# change is KEPT iff it is Pareto-non-dominated over OBJECTIVES; the running
# front is persisted (untracked) in pareto.tsv. This logic is LOCKED here so the
# keep/discard decision cannot be gamed from train.py.
# --------------------------------------------------------------------------
PARETO_PATH = "pareto.tsv"


def _as_max(name, value):
    """Map every objective into a 'higher is better' space."""
    return -value if OBJECTIVES[name] == "min" else value


def dominates(a, b, eps=0.0):
    """True if objective-vector `a` Pareto-dominates `b`.

    a dominates b iff a is >= b on every objective and strictly > on at least one
    (after mapping each objective to maximization space). `eps` ignores ties
    smaller than the seed-noise band.
    """
    ge_all = all(_as_max(k, a[k]) >= _as_max(k, b[k]) - eps for k in OBJECTIVES)
    gt_any = any(_as_max(k, a[k]) > _as_max(k, b[k]) + eps for k in OBJECTIVES)
    return ge_all and gt_any


def pareto_front(points):
    """Return the Pareto-non-dominated subset of `points` (each {label, objectives})."""
    front = []
    for p in points:
        if any(np.isnan(list(p["objectives"].values()))):
            continue
        if any(dominates(q["objectives"], p["objectives"])
               for q in points if q is not p):
            continue
        front.append(p)
    return front


def load_front(path=PARETO_PATH):
    """Read the persisted Pareto front (list of {label, objectives})."""
    points = []
    if not os.path.exists(path):
        return points
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            cells = line.rstrip("\n").split("\t")
            row = dict(zip(header, cells))
            points.append({
                "label": row.get("label", ""),
                "objectives": {k: float(row[k]) for k in OBJECTIVES if k in row},
            })
    return points


def _write_front(front, path=PARETO_PATH):
    names = list(OBJECTIVES)
    ordered = sorted(front, key=lambda p: -p["objectives"][names[0]])
    with open(path, "w") as f:
        f.write("\t".join(["label"] + names) + "\n")
        for p in ordered:
            f.write("\t".join([p["label"]]
                              + [f"{p['objectives'][n]:.6f}" for n in names]) + "\n")


def pareto_decision(objs, label, feasible, eps=0.0, path=PARETO_PATH, write=True):
    """Decide keep/discard for a new model against the persisted Pareto front.

    Returns (verdict, reason, front). verdict is 'keep' or 'discard'. On 'keep'
    (and write=True) the front is updated on disk: the new point is added and any
    prior point it dominates is removed.
    """
    front = load_front(path)
    if not feasible:
        return ("discard", "infeasible (guardrail violated)", front)
    if any(v is None or np.isnan(v) for v in objs.values()):
        return ("discard", "objective is NaN", front)

    others = [q for q in front if q["label"] != label]
    for q in others:
        if dominates(q["objectives"], objs, eps):
            return ("discard", f"dominated by {q['label']}", front)

    survivors = [q for q in others if not dominates(objs, q["objectives"], eps)]
    n_removed = len(others) - len(survivors)
    new_front = survivors + [{"label": label, "objectives": dict(objs)}]
    if write:
        _write_front(new_front, path)
    reason = "extends front" if n_removed == 0 else f"dominates {n_removed} prior point(s)"
    return ("keep", reason, new_front)


# --------------------------------------------------------------------------
# Architecture-agnostic interpretability helpers (safe to lock; train.py owns
# the actual motif extraction, since it knows its own architecture).
# --------------------------------------------------------------------------
def ppm_from_counts(counts):
    """Normalize a (4, w) count matrix into a position-probability matrix."""
    counts = np.asarray(counts, dtype=float)
    col_sums = counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    return counts / col_sums


def pwm_to_consensus(ppm):
    """Render a (4, w) position-probability matrix as an IUPAC consensus string."""
    iupac = {
        frozenset("A"): "A", frozenset("C"): "C", frozenset("G"): "G",
        frozenset("T"): "T", frozenset("AG"): "R", frozenset("CT"): "Y",
        frozenset("GC"): "S", frozenset("AT"): "W", frozenset("GT"): "K",
        frozenset("AC"): "M", frozenset("CGT"): "B", frozenset("AGT"): "D",
        frozenset("ACT"): "H", frozenset("ACG"): "V", frozenset("ACGT"): "N",
    }
    ppm = np.asarray(ppm, dtype=float)
    out = []
    for col in ppm.T:
        peak = col.max()
        if peak <= 0:
            out.append("N")
            continue
        chosen = frozenset(BASES[i] for i in range(4) if col[i] >= 0.5 * peak)
        out.append(iupac.get(chosen, "N"))
    return "".join(out)


if __name__ == "__main__":
    data = load_splits()
    print(f"dataset: {DATASET}  cached in: {CACHE_DIR}")
    print(f"splits  -> train={len(data['y_train'])}  "
          f"val={len(data['y_val'])}  test={len(data['y_test'])}")
    print(f"enhancer prevalence -> train={data['y_train'].mean():.3f}  "
          f"val={data['y_val'].mean():.3f}  test={data['y_test'].mean():.3f}")
    print(f"GC-bin edges (train quartiles): "
          f"{np.round(data['gc_edges'], 4).tolist()}")
    print(f"one-hot shapes -> train={data['X_train'].shape}  "
          f"test={data['X_test'].shape}")
    sys.exit(0)
