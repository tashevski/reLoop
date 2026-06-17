# Study configuration

The contract the agent develops against. The authoritative, machine-readable copy
of these values lives in `prepare.py` (the locked harness); this file is the
human-readable specification.

## Task

- **Question:** Is a 500 bp human DNA sequence an **enhancer** (a cis-regulatory
  element) or a control region? Enhancer prediction from sequence alone is an open
  problem in regulatory genomics — the signal is subtle, distributed, and not fully
  understood, which is exactly why architecture/representation choices matter.
- **X:** raw DNA sequence, one-hot encoded as a `(4, 500)` tensor (channels A,C,G,T;
  non-ACGT → all-zero column).
- **y:** `1` = enhancer (positive class), `0` = control.

## Dataset & provenance

- **Genomic Benchmarks** collection, dataset `human_enhancers_cohn`
  (Grešová et al., *Genomic Benchmarks: a collection of datasets for genomic sequence
  classification*, BMC Genomic Data 2023; PMC10150520). Sequences derive from the
  human reference genome — a **public benchmark, no PHI**.
- Distributed on HuggingFace (`katarinagresova/Genomic_Benchmarks_human_enhancers_cohn`);
  `prepare.py` downloads the parquet files directly.
- **Headroom target:** the paper's CNN baseline is **≈ 0.695 accuracy** — far from
  ceiling, so there is real room for novel architectures to matter.

## Splits (LOCKED)

- The authors' canonical **train (20,843) / test (6,948)** split, balanced ~50/50.
- **Validation** = a label-stratified **15%** carve-out of train (`SPLIT_SEED = 1337`).
- The agent develops against **validation**. The **test set is sequestered** and
  touched only under the test-set protocol in `program.md`.

  > ⚠️ **Homology-leakage caveat.** The parquet files contain only `seq` + `label`,
  > with **no genomic coordinates**, so a chromosome-holdout split (the gold standard
  > for avoiding sequence-homology leakage) is not possible here. Train and validation
  > may therefore contain homologous loci, so validation can be mildly optimistic. The
  > authors' **test split is the honest readout**, and the GC-stratified breakdown
  > below is a partial guard against the most common confound. Recovering coordinates
  > (to enable chromosome-holdout) would require the `genomic-benchmarks` package plus a
  > reference genome — out of scope here, noted as future work.

## Metrics

- **Headline:** AUROC for detecting enhancers. **Higher is better.**
- **Reported (not gated):** accuracy at threshold 0.5 — compare directly to the
  ≈ 0.695 published baseline.
- **Guardrails** — HARD feasibility constraints. A model that breaks any of these is
  infeasible and can never enter the Pareto front, whatever its objectives:
  - `calibration_ece` ≤ 0.10 — expected calibration error (15 bins).
  - `mcc` ≥ 0.30 — Matthews correlation coefficient at threshold 0.5.
- **Subgroups:** per-**GC-content quartile** AUROC (`gc_q0..gc_q3`), bin edges fixed
  from the training distribution. GC content is a well-known confounder in enhancer
  prediction — a model that only works in one GC regime is really just a GC detector.

## Multi-objective optimization (Pareto front)

The problem is **not** collapsed to a single number. Among guardrail-feasible models,
the harness optimizes three competing objectives jointly:

| Objective         | Direction | Meaning                                            |
|-------------------|-----------|----------------------------------------------------|
| `auroc`           | maximize  | overall discrimination                             |
| `calibration_ece` | minimize  | calibration quality                                |
| `gc_worst_auroc`  | maximize  | robustness — AUROC of the weakest GC-content bin   |

A model **dominates** another if it is at least as good on all three and strictly
better on at least one. A model is **Pareto-optimal** if nothing dominates it. The
keep/discard rule is dominance-based (not single-metric): feasible + non-dominated →
keep; infeasible or dominated → discard. The running front is persisted (untracked) in
`pareto.tsv`; the dominance logic is locked in `prepare.py` (`OBJECTIVES`, `dominates`,
`pareto_decision`) so it cannot be gamed. The aim over a run is to **expand the whole
front** of trade-offs, not merely maximize AUROC. (To add/remove an objective, edit
`OBJECTIVES` in `prepare.py` — it is the single source of truth, and everything
downstream follows automatically.)

## Biological-insight artifact (required every loop)

Each experiment must also surface **what the model learned**, not just a number:
the top learned sequence **motifs** (for the baseline CNN: first-conv filters →
position-probability matrix → IUPAC consensus; for other architectures: input-gradient
saliency over high-scoring sequences), appended to `research_notes.md`. Over a run this
accumulates a log of candidate regulatory motifs that a human can compare against known
transcription-factor motif databases (e.g. JASPAR). A run that improves AUROC but logs
no interpretable artifact is incomplete.

## Statistical decision rule

- Every experiment runs **≥ 3 seeds**; decisions use **mean ± std** across seeds, not a
  single point estimate. Keep a change only if the primary metric improves by more than
  seed-to-seed noise, all guardrails hold, and no GC subgroup is materially harmed. See
  `program.md` → *Statistical rigor*.
