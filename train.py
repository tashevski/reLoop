"""
train.py -- the single file the agent edits.

Regulatory-genomics experiment: predict whether a 500 bp human DNA sequence is an
ENHANCER (positive class) vs a control region, from the raw sequence.

Goal: MAXIMIZE validation AUROC (higher is better) WITHOUT breaking the guardrails
(calibration ECE, MCC) or harming any GC-content subgroup, in a way that holds up
ACROSS SEEDS (see program.md -> Statistical rigor). The published CNN baseline is
~0.695 accuracy -- there is a lot of headroom.

Everything in this file is fair game to edit: model architecture (conv width/depth,
dilation, attention, pooling), sequence representation, train-only augmentation
(e.g. reverse-complement), optimizer, hyperparameters, training loop. You may NOT
edit prepare.py (the locked harness), and you may NOT touch the held-out test set
outside the protocol in program.md.

Each loop must ALSO produce a biological-insight artifact: the motifs the model
learned, appended to research_notes.md (see extract_motifs / program.md).

Run:  uv run train.py
"""

import subprocess
import sys
import time
import resource

import numpy as np
import torch
import torch.nn as nn

import prepare

# ----------------------------- knobs to tune ------------------------------
N_SEEDS = 3            # >= 3; decisions are made on mean +/- std across seeds
N_FILTERS = 64         # first-conv channels (also the # of candidate motifs)
KERNEL = 15            # first-conv width (~motif scale, in bp)
N_CONV = 1             # number of conv blocks (baseline is deliberately shallow)
HIDDEN = 64
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 15
BATCH_SIZE = 256
N_TOP_MOTIFS = 8       # how many learned filter-motifs to log per run
# --------------------------------------------------------------------------


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DNACNN(nn.Module):
    """Simple 1D-CNN over one-hot DNA: (B, 4, L) -> logit."""

    def __init__(self, n_filters, kernel, n_conv, hidden, dropout):
        super().__init__()
        blocks, ch = [], 4
        for _ in range(n_conv):
            blocks += [nn.Conv1d(ch, n_filters, kernel, padding="same"), nn.ReLU()]
            ch = n_filters
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(ch, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h = self.pool(self.conv(x)).squeeze(-1)
        return self.head(h).squeeze(-1)


def _batched_predict(model, X, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(X[i:i + BATCH_SIZE], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def run_seed(seed, data, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr = data["X_train"], data["y_train"]
    model = DNACNN(N_FILTERS, KERNEL, N_CONV, HIDDEN, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(X_tr)
    for _ in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = torch.tensor(X_tr[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_tr[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    prob_val = _batched_predict(model, data["X_val"], device)
    metrics = prepare.evaluate(data["y_val"], prob_val, data["gcbin_val"])
    return metrics, model


def extract_motifs(model, data, device, k=N_TOP_MOTIFS, max_seqs=1500):
    """Biological-insight artifact: turn the first conv layer's filters into motifs.

    For each first-layer filter we find, across positive (enhancer) validation
    sequences, the windows that maximally activate it, average their one-hot slices
    into a position-probability matrix, and render an IUPAC consensus. We keep the
    top-`k` filters by mean peak activation.

    NOTE: this reads `model.conv[0]` (the leading Conv1d). If you change the
    architecture so the first layer is no longer an interpretable Conv1d, you MUST
    update this function to extract motifs from your new model (e.g. input-gradient
    saliency over the highest-scoring sequences) -- see program.md.
    """
    first = model.conv[0]
    if not isinstance(first, nn.Conv1d):
        return []  # agent must reimplement for non-conv front-ends

    width = first.kernel_size[0]
    X = data["X_val"][data["y_val"] == 1][:max_seqs]
    if len(X) == 0:
        return []

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        acts = torch.relu(first(xb)).cpu().numpy()  # (N, n_filters, L)

    half = width // 2
    Xpad = np.pad(X, ((0, 0), (0, 0), (half, width - half - 1)))
    results = []
    for f in range(acts.shape[1]):
        a = acts[:, f, :]
        peak_per_seq = a.max(axis=1)
        order = np.argsort(peak_per_seq)[::-1][:200]  # strongest activators
        counts = np.zeros((4, width), dtype=np.float64)
        for si in order:
            pos = int(a[si].argmax())
            counts += Xpad[si, :, pos:pos + width]
        consensus = prepare.pwm_to_consensus(prepare.ppm_from_counts(counts))
        results.append((f, consensus, float(peak_per_seq.mean())))

    results.sort(key=lambda r: r[2], reverse=True)
    return results[:k]


def log_motifs(motifs, summary_auroc):
    """Append the motif artifact to research_notes.md (untracked)."""
    with open("research_notes.md", "a") as fh:
        fh.write(f"\n## motifs (val AUROC {summary_auroc:.4f})\n")
        if not motifs:
            fh.write("- (no conv front-end; extract_motifs needs updating)\n")
        for f_idx, consensus, act in motifs:
            fh.write(f"- filter {f_idx:3d}  {consensus}  (mean act {act:.3f})\n")


def peak_mem_gb():
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KB on Linux, bytes on macOS.
    return maxrss / (1024 ** 2) if sys.platform.startswith("linux") else maxrss / (1024 ** 3)


def run_label():
    """Label this run by its git commit if available, else by wall-clock time."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return time.strftime("t%H%M%S")


def main():
    device = get_device()
    data = prepare.load_splits()

    t0 = time.time()
    seed_metrics, last_model = [], None
    for s in range(N_SEEDS):
        metrics, model = run_seed(s, data, device)
        seed_metrics.append(metrics)
        last_model = model
    train_seconds = time.time() - t0

    agg = prepare.summarize(seed_metrics, peak_mem_gb(), train_seconds)

    # Biological-insight artifact (required every loop).
    motifs = extract_motifs(last_model, data, device)
    log_motifs(motifs, agg["auroc"])
    print(f"motifs:           {len(motifs)} appended to research_notes.md")

    # Multi-objective keep/discard: Pareto dominance over OBJECTIVES (locked in
    # prepare.py), gated by guardrail feasibility. Updates pareto.tsv on a keep.
    objs = prepare.objective_vector(agg)
    feasible = prepare.guardrails_ok(agg["guardrails"])
    verdict, reason, front = prepare.pareto_decision(objs, run_label(), feasible)
    print(f"pareto:           {verdict} ({reason}); front size {len(front)}")


if __name__ == "__main__":
    main()
