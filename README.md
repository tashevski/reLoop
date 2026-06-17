# reLoop (an Autonomous Agentic Research Loop for Biomedical Questions)

An AI agent doing its own ML research, autonomously and overnight. It edits a single
training file, trains a model, checks whether the held-out objectives improved, keeps
or discards the change, and repeats — so you wake up to a log of experiments and
(hopefully) a better model. You don't edit the Python directly; you program the agent
through `program.md`.

The experiment shipped here is **regulatory genomics**: predict whether a 500 bp human
DNA sequence is an **enhancer** (a cis-regulatory element) vs a control region, from the
raw sequence — the public, de-identified Genomic Benchmarks `human_enhancers_cohn`
dataset (Grešová et al. 2023; no PHI). The published CNN baseline is only **~0.695
accuracy**, so there is real headroom for the agent to discover better architectures.
The harness generalizes to other in-silico biomedical tasks — see
[Use it for your own project](#use-it-for-your-own-project).

Two goals run in parallel:

1. **Novel architecture search** — enhancer prediction from sequence is unsolved and
   architecture-sensitive, so the agent has a large, meaningful design space to explore
   (see [Architectural search space](#architectural-search-space)).
2. **A biological-insight artifact** — every loop also extracts the sequence **motifs**
   the model learned and appends them to `research_notes.md`, so a run yields reviewable
   candidate regulatory motifs (comparable to known TF motifs), not just a number.

It runs on **CPU or Apple MPS — no GPU required**.

## Current research question

The study loaded right now (full contract in `study_config.md`):

- **Question.** Given a 500 bp human DNA sequence, is it an **enhancer** (a cis-regulatory
  element) or a control region? Enhancer prediction from sequence alone is unsolved — the
  signal is subtle and distributed — so it rewards better architectures and inductive biases.
- **Data.** Genomic Benchmarks `human_enhancers_cohn` (Grešová et al. 2023) — 27,791
  reference-genome-derived sequences, balanced ~50/50, public and de-identified (no PHI),
  one-hot encoded to `(4, 500)`.
- **Splits (locked).** Authors' train/test (20,843 / 6,948); validation is a 15%
  label-stratified carve-out of train (≈17,717 / 3,126), seed 1337. The test set is
  sequestered — touched only under the protocol in `program.md`.
- **Objectives (Pareto).** maximize `auroc`, minimize `calibration_ece`, maximize
  `gc_worst_auroc` (robustness of the weakest GC-content bin) — among models that pass the
  guardrails (ECE ≤ 0.10, MCC ≥ 0.30).
- **Where it starts.** The published CNN baseline is ≈ **0.695 accuracy**; the shipped
  baseline `train.py` scores ≈ **0.79 validation AUROC / 0.71 accuracy** across 3 seeds —
  clear headroom to push.
- **What progress looks like.** Expand the whole Pareto front (better discrimination *and*
  calibration *and* GC-robustness at once), and accumulate learned **motifs** in
  `research_notes.md` that recover — or extend — known transcription-factor motifs.

To load a different question, see [Use it for your own project](#use-it-for-your-own-project).

## How it works

The repo is deliberately small; three files matter:

- **`prepare.py`** — the **locked, read-only harness**. Downloads/caches the dataset,
  one-hot encodes sequences, defines the fixed train/val/test splits, and owns the
  ground-truth metrics, the **objectives**, and the **Pareto-dominance logic**.
  **Not modified by the agent** (so the evaluation can't be gamed).
- **`train.py`** — the **single file the agent edits**: the DNA-CNN baseline, the
  multi-seed training loop, and `extract_motifs` (the insight artifact). Architecture,
  optimizer, encoding, augmentation, hyperparameters — all fair game.
- **`program.md`** — the agent's instructions (literature review → explore/rank/choose
  loop → multi-seed, Pareto keep/discard → sequestered test set). **Edited by the human.**

The task/metric/split/objective contract is documented in **`study_config.md`**. Two
untracked ledgers accumulate as the agent works: `results.tsv` (every experiment) and
`pareto.tsv` (the current front of non-dominated trade-offs).

## Built for biomedical data

Autonomous, self-improving ML research loops — an agent that edits training code, trains,
keeps or reverts the change based on a metric, and repeats — are an established pattern,
with several implementations around and the basic idea predating any single one of them.
reLoop is one such loop, but built specifically for **biomedical data**, where a naive
single-metric loop is the wrong shape: datasets are small and confounded, one number hides
clinically important failure modes, and greedily selecting on a single validation score
across ~100 experiments mostly _discovers noise_. The sections below are the design
decisions that make the loop trustworthy on biomedical questions.

### Research process: literature-first, explore–rank–choose

A bare autoloop just mutates code and keeps whatever lowers the metric. reLoop instead
treats the agent as a researcher: before any training it does a **broad academic literature
review** — mapping what works, the field's benchmarks, and known confounders and leakage
traps — and distils a ranked idea backlog into `research_notes.md`. Then on **every** loop
iteration it explicitly _searches for alternative conceptions_ of the current bottleneck,
enumerates several genuinely different candidate ideas, and **ranks** them (expected impact
vs complexity vs leakage/validity risk) before implementing one. (Defined in `program.md`.)

### Objective & evaluation: a real domain metric

reLoop evaluates with a locked, domain-appropriate evaluator in `prepare.py`: **AUROC** for
enhancer detection, plus accuracy (vs the published ~0.695 baseline), calibration ECE, MCC,
and per-subgroup AUROC. Crucially the evaluator, the data splits, and the test set live in
the read-only harness, so the agent literally cannot edit the metric or peek at test data —
a loop that lets the agent touch its own metric isn't measuring anything.

### Architectural search space

The baseline `train.py` is a deliberately simple, beatable 1D-CNN over one-hot DNA (a
single conv layer → global max-pool → linear head). The agent's job is to restructure it —
this is the design space where DNA sequence models actually differ, and where the headroom
over the baseline lives:

- **Motif-scale convolutions.** The first conv layer acts as a bank of learnable
  position weight matrices (motif detectors); kernel width ≈ binding-site length, and the
  number of filters bounds how many motifs can be represented (cf. DeepBind, Basset).
- **Multi-scale / dilated stacks.** Stacked or dilated convolutions widen the receptive
  field to capture spacing and combinatorial grammar between motifs without exploding
  parameters (cf. the dilated towers in Enformer).
- **Convolution + recurrence or attention.** A conv front-end to detect motifs followed
  by a BiLSTM or self-attention to model long-range dependencies between them
  (cf. DanQ for conv+RNN; transformer blocks for conv+attention).
- **Reverse-complement equivariance.** A motif on the forward strand is the same signal
  on the reverse strand; tying RC-equivalent filters or averaging RC predictions is a
  strong, genomics-specific inductive bias.
- **Pooling and readout choices.** Global max vs attention pooling, k-max pooling, or
  learned aggregation change how local motif hits are summarized into a prediction.

Because the agent may move away from a leading `Conv1d`, `train.py`'s motif extractor is
its responsibility to keep meaningful (e.g. switch to input-gradient saliency) — this is
spelled out in `program.md`.

### Multi-objective optimization (Pareto front)

The problem is **not** collapsed to one number. Among models that pass hard **guardrails**
(feasibility), the harness optimizes three competing objectives jointly:

| Objective         | Direction | Meaning                                          |
| ----------------- | --------- | ------------------------------------------------ |
| `auroc`           | maximize  | overall discrimination                           |
| `calibration_ece` | minimize  | calibration quality                              |
| `gc_worst_auroc`  | maximize  | robustness — AUROC of the weakest GC-content bin |

A model **dominates** another if it is at least as good on all objectives and strictly
better on at least one; a model is **Pareto-optimal** if nothing dominates it. Keep/discard
follows dominance (feasible + non-dominated → keep; infeasible or dominated → discard), and
the running front is kept in `pareto.tsv`. The dominance logic lives in `prepare.py`
(`OBJECTIVES`, `dominates`, `pareto_decision`) so it can't be gamed. The aim over a run is
to **expand the whole front** of trade-offs, not just maximize one metric. To change the
objective set, edit `OBJECTIVES` in `prepare.py` — everything downstream follows.

### Statistical rigor: decisions across seeds

Small biomedical datasets make noise-mining acute, so a single training run's number is not
enough to act on. reLoop runs **≥3 seeds** per experiment and decides on **mean ± std**,
treating any difference inside the seed-noise band as a tie rather than a win — a "win" has
to be reproducible across seeds, not a lucky initialization.

### Guardrails & subgroup robustness

reLoop adds **hard feasibility constraints** — calibration ECE ≤ 0.10 and MCC ≥ 0.30 — that a
model must satisfy to be eligible at all, plus a **per-GC-content-bin** AUROC breakdown. GC
content is a notorious enhancer-prediction confounder, so a model that wins overall AUROC
while collapsing in one GC regime (i.e. is really just a GC detector) is caught and can be
vetoed. Guardrails gate eligibility for the Pareto front.

### Sequestered test set & leakage discipline

reLoop locks a train / validation / **test** split in `prepare.py`, develops only on
validation, and touches the held-out test set only rarely under an explicit protocol —
**never** to drive selection (repeatedly tuning against a test set destroys its validity). It
forbids leakage (no statistics fit on val/test), documents the homology-leakage caveat of
this benchmark, and bars acquiring, scraping, or re-identifying data. An explicit **dry-lab
assumptions banner** states that this autonomous design is _not_ appropriate for work
touching human/animal subjects, biological samples, or clinical decisions. (See `program.md`
/ `study_config.md`.)

### Biological-insight artifact

Beyond a metric, **every loop extracts the sequence motifs the model learned** (conv filters
→ position-weight matrix → IUPAC consensus, or input-gradient saliency for non-conv models)
and appends them to `research_notes.md`, so a run accumulates reviewable candidate regulatory
motifs to compare against known TF databases. The aim is to learn biology, not just climb a
leaderboard.

### Runs on a laptop (CPU / MPS)

Many autoloops assume a GPU. reLoop is dependency-light (torch, numpy, pandas, pyarrow,
requests, matplotlib) and runs on **CPU or Apple MPS — no GPU required** — so the whole loop
works on a laptop.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU needed.

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download + cache the dataset and sanity-check the splits (one-time)
uv run prepare.py

# 4. Run a single baseline experiment (multi-seed; minutes on CPU/MPS)
uv run train.py
```

The autonomous keep/discard loop uses git to advance or revert experiments, so initialize
a repo first if this isn't one already: `git init && git add -A && git commit -m baseline`.

## Running the agent

Spin up your Claude/Codex agent in this repo (with permissions relaxed), then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

`program.md` is essentially a lightweight "skill" describing the whole research loop.

## Use it for your own project

The design — a **locked harness** that owns data + metrics + objectives, a **single
editable training file**, and a **markdown program** that drives an autonomous,
multi-seed, Pareto-gated loop — is task-agnostic. To retarget reLoop at a different
in-silico biomedical question (another genomics task, protein fitness/function, molecular
property prediction, medical-image or EHR classification, …):

1. **`prepare.py` (the harness)** — replace the data loading (`_download` / `_load_raw` /
   `load_splits`) with your dataset and locked, leakage-safe splits; replace the encoder
   (`one_hot`) with your modality's representation; set the metric functions and edit
   `GUARDRAILS` and `OBJECTIVES` to what matters for your task. `summarize` and the Pareto
   logic are generic and usually need no changes.
2. **`train.py` (the baseline)** — reset to a simple, beatable baseline model for your
   modality (it just needs to read `prepare.load_splits()`, train, and call
   `prepare.summarize(...)` + `prepare.pareto_decision(...)`). Adapt the insight artifact
   (`extract_motifs`) to whatever interpretability fits, or drop it.
3. **`study_config.md` / `program.md`** — rewrite the task/metric/objective contract, and
   point the literature-review phase at your field's papers and benchmarks.

You don't have to do this by hand — the fastest path is to **prompt your coding agent** to
do it. For example:

```
Read README.md, program.md, study_config.md and prepare.py to understand the harness.
I want to retarget this to <TASK> using the <DATASET> dataset (<URL / how to obtain>).
The primary metric should be <METRIC>; guardrails <…>; Pareto objectives <…>; splits must
be <how to split without leakage>. Rewrite prepare.py (keep the Pareto/summarize logic),
reset train.py to a simple baseline for this modality, and update study_config.md and
program.md to match. Then run `uv run prepare.py` and `uv run train.py` to verify, and
fix anything that breaks.
```

Keep the parts that make the loop trustworthy: the locked harness, the held-out test set
touched only under protocol, multi-seed decisions, guardrails, and the Pareto front.

## Project structure

```
prepare.py       — locked harness: data, splits, metrics, objectives, Pareto logic (do not modify)
train.py         — baseline model + multi-seed loop + insight artifact (agent modifies this)
program.md       — agent instructions
study_config.md  — task / metric / split / guardrail / objective contract
pyproject.toml   — dependencies
```

## References

**Autonomous & evolutionary ML research loops.** Background research informing this repos pattern
of iteratively proposing model or algorithm changes, evaluating them, and keeping the better ones
(i.e. neural architecture search, evolutionary AutoML, LLM-driven program search, and
automated science):

1. Zoph, B., Le, Q.V. (2017). _Neural Architecture Search with Reinforcement Learning._
   ICLR 2017. https://arxiv.org/abs/1611.01578
2. Real, E., Aggarwal, A., Huang, Y., Le, Q.V. (2019). _Regularized Evolution for Image
   Classifier Architecture Search_ (AmoebaNet). AAAI 2019. https://arxiv.org/abs/1802.01548
3. Real, E., Liang, C., So, D.R., Le, Q.V. (2020). _AutoML-Zero: Evolving Machine Learning
   Algorithms From Scratch._ ICML 2020. https://arxiv.org/abs/2003.03384
4. Romera-Paredes, B., et al. (2024). _Mathematical discoveries from program search with
   large language models_ (FunSearch). Nature 625, 468–475.
   https://www.nature.com/articles/s41586-023-06924-6
5. Lu, C., Lu, C., Lange, R.T., Foerster, J., Clune, J., Ha, D. (2024). _The AI Scientist:
   Towards Fully Automated Open-Ended Scientific Discovery._ https://arxiv.org/abs/2408.06292
6. Karpathy, A. _nanochat / autoresearch._ https://github.com/karpathy/nanochat

**Regulatory-genomics deep learning.** The task, metrics, and architectural search space here
draw on:

7. Grešová, K., Martinek, V., Čechák, D., Šimeček, P., Alexiou, P. (2023). _Genomic
   Benchmarks: a collection of datasets for genomic sequence classification._ BMC Genomic
   Data 24, 25. https://doi.org/10.1186/s12863-023-01123-8
8. Alipanahi, B., Delong, A., Weirauch, M.T., Frey, B.J. (2015). _Predicting the sequence
   specificities of DNA- and RNA-binding proteins by deep learning_ (DeepBind). Nature
   Biotechnology 33, 831–838. https://www.nature.com/articles/nbt.3300
9. Zhou, J., Troyanskaya, O.G. (2015). _Predicting effects of noncoding variants with deep
   learning–based sequence model_ (DeepSEA). Nature Methods 12, 931–934.
   https://www.nature.com/articles/nmeth.3547
10. Kelley, D.R., Snoek, J., Rinn, J.L. (2016). _Basset: learning the regulatory code of the
    accessible genome with deep convolutional neural networks._ Genome Research 26, 990–999.
    https://genome.cshlp.org/content/26/7/990
11. Quang, D., Xie, X. (2016). _DanQ: a hybrid convolutional and recurrent deep neural
    network for quantifying the function of DNA sequences._ Nucleic Acids Research 44, e107.
    https://doi.org/10.1093/nar/gkw226
12. Avsec, Ž., et al. (2021). _Effective gene expression prediction from sequence by
    integrating long-range interactions_ (Enformer). Nature Methods 18, 1196–1203.
    https://www.nature.com/articles/s41592-021-01252-x

## License

MIT
