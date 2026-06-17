# autoresearch (regulatory genomics — enhancer prediction)

This is an experiment to have the LLM do its own research on a **regulatory-genomics** task: predicting whether a 500 bp human DNA sequence is an **enhancer** vs a control region, from the raw sequence (Genomic Benchmarks `human_enhancers_cohn`). The agent autonomously edits the training code, runs experiments, and keeps or discards changes based on validation AUROC — with statistical and data-handling safeguards — while also surfacing the **sequence motifs** its models learn. There are two intertwined goals: (1) **novel architecture search** — the published baseline (~0.695 accuracy) leaves lots of headroom, so conv/dilation/attention choices genuinely matter; (2) a **biological-insight artifact** — every loop logs candidate regulatory motifs, so the run yields reviewable biology, not just a leaderboard number.

> **Assumptions.** This is a *dry-lab, in-silico* task on a fixed, public, de-identified benchmark (reference-genome-derived sequence — no PHI, no subjects, no samples). Experiments are pure computation. This autonomous design would **not** be appropriate for work touching human/animal subjects, biological samples, or clinical decisions — those require human-in-the-loop review, ethics/IRB approval, and regulatory oversight.

## Setup

To set up a new experiment, work with the user to pin down the things the LLM-training version takes for granted:

1. **Agree on a run tag** and create a fresh branch: `git checkout -b autoresearch/<tag>` from master. The branch must not already exist.
2. **Define the task and the metric.** With the user, write down in `study_config.md`:
   - The prediction task (what `X` is, what `y` is, the clinical/biological meaning).
   - The **primary metric** and whether higher or lower is better (e.g. AUROC, AUPRC, C-index, Dice/IoU, RMSE, calibration error). Pick the one the field actually uses for this task.
   - **Guardrail metrics** that a change must not degrade past a threshold (e.g. don't trade away calibration or recall on the minority class for a tiny AUROC bump). State the thresholds.
   - **Subgroups** for fairness/robustness checks (e.g. site, sex, age band, scanner/assay) and the metric to report per subgroup. An overall improvement that worsens a subgroup is not automatically a win — see the loop.
3. **Define and LOCK the data splits.** Confirm there is a train / validation / **held-out test** split, that it is **grouped/stratified appropriately** (e.g. patient-level, site-level — no leakage across splits), and that the test set is sequestered. The agent develops against validation; the test set is touched only under the protocol below.
4. **Read the in-scope files** for full context:
   - `README.md` / `study_config.md` — task, metric, data provenance, the GC-robustness guardrail, and the required motif artifact.
   - `prepare.py` — the read-only harness: parquet download/cache, one-hot encoding, locked train/val/test splits, ground-truth metrics (AUROC, ECE, MCC), per-GC-bin evaluation, and motif-rendering helpers. **Do not modify.**
   - `train.py` — the file you modify. The DNA-CNN baseline, multi-seed training loop, and `extract_motifs` (the insight artifact).
5. **Verify data exists.** Run `uv run prepare.py` once to download + cache the Genomic Benchmarks parquet and sanity-check the splits. This is a public, de-identified benchmark. Do **not** acquire, scrape, or re-identify any *other* genomic data on your own — use only what the harness provides.
6. **Initialize `results.tsv`** with just the header row.
7. **Confirm and go.**

Once you get confirmation, do the **initial literature review** (below) and then kick off the experimentation.

## Initial literature review (do this first, before any training run)

Before you touch `train.py` or run the baseline, ground yourself in the field. The goal is to start the run with a map of what actually works, not just whatever first comes to mind.

1. **Search broadly and academically.** Using your web search / fetch tools, conduct a broad search of the genomics deep-learning literature relevant to *this task* — sequence models for regulatory elements (DeepBind, Basset, DanQ, DeepSEA, Enformer and successors), the architectural toolkit (convolutions vs dilated stacks vs attention/transformers, motif-scale kernels, pooling choices, reverse-complement equivariance), motif interpretability (PWMs from conv filters, TF-MoDISco), and known confounders/leakage (GC-content bias, sequence homology between splits). Prefer primary sources — papers (PubMed, arXiv/bioRxiv), the Genomic Benchmarks paper, and reference implementations — over shallow summaries. Cast a wide net first, then drill into the most promising threads.
2. **Capture what you find** in a new file `research_notes.md` (kept untracked by git, like `results.tsv`). For each useful idea: a one-line description, the source (title + link), why it might help here, and a rough sense of implementation cost / compute / risk / leakage concerns. This is your running knowledge base — keep appending to it throughout the run.
3. **Sketch a backlog.** Distill a ranked shortlist of concrete, testable ideas you could apply to `train.py`. This seeds the experiment loop.

Don't rabbit-hole: this is a broad orienting pass, not an exhaustive survey.

## Experimentation

Each experiment trains a model and evaluates it on the **validation** split. Launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — model **architecture** (conv width/depth, dilation, attention, pooling, residual/normalization choices), the sequence representation/encoding, train-only augmentation (e.g. reverse-complement — a standard genomics trick), optimizer, hyperparameters, and the training loop.

**What you CANNOT do:**
- Modify `prepare.py` (splitting, metric computation, GC-bin evaluation, motif helpers). It is the ground-truth metric and the guardian of the test set. Modifying it = gaming the metric.
- Touch, peek at, or train on the **held-out test set** outside the test protocol below. Any leakage from test (or, during a single experiment, from validation) into training invalidates the result.
- Introduce **leakage**: no feature derived using validation/test data, no statistics fit on anything but train. (Note the homology caveat in `study_config.md` — train/val may share homologous loci, so don't engineer anything that exploits the val split.)
- Acquire, download, scrape, or re-identify any other genomic data. Use only the dataset the harness provides.
- Install new packages or add dependencies beyond `pyproject.toml`.

**The goal:** push out the **Pareto front** over the competing objectives — AUROC (↑), calibration ECE (↓), and worst-GC-bin AUROC (↑) — among models that satisfy the hard guardrails, in a way that **holds up statistically** (below). This is multi-objective: there is no single "best" model, only a front of non-dominated trade-offs (see *Multi-objective optimization*). Each experiment trains `N_SEEDS` models (set in `train.py`); on this sequence dataset a multi-seed run takes **minutes** on CPU/MPS (not seconds), so favour ≥3 seeds but don't inflate the count needlessly. Do not change the dataset, the splits, the objectives, or the metrics — those are locked in `prepare.py`.

**Biological-insight artifact (required every loop):** every run must also produce the learned **motifs** and append them to `research_notes.md` (the baseline does this via `extract_motifs`). **If you change the architecture so the leading layer is no longer an interpretable `Conv1d`, you MUST update `extract_motifs`** to recover motifs from your new model (e.g. input-gradient saliency over the highest-scoring sequences). A run that improves AUROC but logs no interpretable artifact is incomplete.

**Simplicity criterion:** All else equal, simpler is better. A small metric gain that adds ugly complexity (or a new leakage risk) is not worth it. Removing something for equal-or-better results is a great outcome.

**The first run:** establish the baseline — run the training script as-is.

## Statistical rigor (this is the heart of the biomedical adaptation)

Selecting whatever change nudges a single validation number, over ~100 experiments, is a multiple-comparisons machine — on biomedical-scale data you *will* select noise. Guard against it:

- **Multiple seeds.** Every experiment runs ≥3 seeds (more if cheap). Report mean ± std of the primary metric, not a single point.
- **Decide by significance, not point estimate.** Keep a change only if the improvement is meaningful relative to seed-to-seed variance (e.g. the mean improves and the change is larger than the baseline's seed std / a paired test across seeds is favorable). A 0.001 bump inside the noise band is **not** a win.
- **Validation is for development; the test set is sacred.** Do not select hyperparameters or architecture on the test set. Evaluating on test happens rarely and under the protocol below.
- **Watch for overfitting to validation.** If validation keeps improving but the gains are tiny and noisy, you are likely fitting the val split — note it in `research_notes.md` and prefer changes with a mechanistic rationale from the literature over blind metric-chasing.

## Multi-objective optimization (Pareto front)

We do **not** collapse the problem to one number. Among models that pass the hard guardrails (feasibility), the harness optimizes three competing objectives jointly: **AUROC** (maximize), **calibration ECE** (minimize), and **worst-GC-bin AUROC** (maximize, i.e. robustness). These genuinely trade off — a sharper discriminator may be worse-calibrated; a more robust model may give up peak AUROC.

- **Dominance.** Model A *dominates* B if A is at least as good as B on **all** objectives and strictly better on **at least one**. A model is **Pareto-optimal** if nothing dominates it.
- **The front.** The harness keeps the set of non-dominated models in `pareto.tsv` (untracked). `prepare.py` owns this logic (`dominates`, `pareto_decision`) so it can't be gamed from `train.py`.
- **Keep/discard rule (this replaces single-metric advancement):**
  - If the run is **infeasible** (any guardrail violated) → **discard**.
  - If it is **dominated** by an existing front member → **discard** (`git reset`).
  - If it is **non-dominated** (it extends the front or dominates prior points) → **keep** (advance the branch); the harness updates `pareto.tsv`.
  - `train.py` prints the verdict on the `pareto:` line; follow it.
- **Statistical honesty.** Treat objective differences smaller than seed-to-seed noise as ties, not wins — a "non-dominated" point that only wins by noise is not real progress. Prefer moves with a mechanistic rationale.
- **Navigating the front.** Because keeps can move *sideways* along the front (trading one objective for another), periodically branch from a different front member rather than always the latest commit — `pareto.tsv` lists them with their commit labels. The aim over a run is to **expand the whole front** (better trade-offs everywhere), not just maximize AUROC.

## Output format

Once the script finishes it prints a summary, e.g.:

```
---
primary_metric:   0.7650       # AUROC for enhancer detection (higher better) — see study_config.md
metric_std:       0.0080       # std across seeds
accuracy:         0.7010       # at threshold 0.5 — compare to the ~0.695 published baseline
guardrails:       {calibration_ece: 0.0420, mcc: 0.4100}
subgroups:        {gc_q0: 0.7580, gc_q1: 0.7620, gc_q2: 0.7710, gc_q3: 0.7690}
gc_worst_auroc:   0.7580        # min over GC bins (a robustness objective)
objectives:       {auroc: 0.7650, calibration_ece: 0.0420, gc_worst_auroc: 0.7580}
guardrails_ok:    yes
n_seeds:          3
peak_mem_gb:      1.20
train_seconds:    140.0
motifs:           8 appended to research_notes.md
pareto:           keep (extends front); front size 4
```

The `objectives:` line is the multi-objective vector that drives keep/discard (see *Multi-objective optimization* below); the `pareto:` line is the harness's verdict. The script also appends the learned motifs to `research_notes.md`. Extract the key signals from the log:

```
grep "^primary_metric:\|^accuracy:\|^objectives:\|^pareto:" run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated). Header + columns:

```
commit	auroc	auroc_std	calibration_ece	gc_worst_auroc	guardrails_ok	pareto	status	description
```

1. git commit hash (short)
2. AUROC, mean across seeds (use a sentinel like `nan` for crashes)
3. std across seeds
4. calibration ECE (objective; mean across seeds)
5. worst-GC-bin AUROC (objective; mean across seeds)
6. `yes`/`no` — did it satisfy all hard guardrails
7. the Pareto verdict from the run: `keep`/`discard` + reason (e.g. `extends front`, `dominated by <commit>`)
8. status: `keep`, `discard`, or `crash`
9. short description of what this experiment tried

(`results.tsv` is the full experiment log; `pareto.tsv` is the current front. Leave both untracked.)

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. **Explore the option space.** Do not just grab the first idea that comes to mind:
   a. **Identify the current bottleneck** from `results.tsv` and recent runs (what's limiting the primary metric right now — capacity, optimization, regularization, class imbalance, calibration, a struggling subgroup, leakage you suspect, etc.).
   b. **Search for alternative conceptions.** Run focused academic searches to explore *additional ways of conceiving the current bottleneck and its solutions* — techniques, framings, or tricks you haven't tried yet, ideally with a mechanistic rationale for this data modality. Look for angles that differ from what's already in `results.tsv`. Append anything useful to `research_notes.md`.
   c. **Enumerate candidate ideas.** From your backlog, the literature, and combinations of previous near-misses, write down several distinct candidate changes (aim for at least 3–4 genuinely different options, not minor variants of one).
3. **Choose one idea.** Do not just grab the first idea that comes to mind:
   a. **Rank and consider the options.** Score each candidate on expected metric impact, implementation/complexity cost (simplicity criterion), **leakage/validity risk**, compute/memory risk, and crash risk. Briefly note the ranking and *why* you picked the winner — favor high expected-value, low-risk, mechanistically-motivated moves, but periodically take a more exploratory/radical bet when incremental ideas stall.
   b. **Implement the chosen idea** by directly hacking the code in `train.py`.
4. git commit
5. Run the experiment (multi-seed): `uv run train.py > run.log 2>&1` (redirect everything — do NOT flood your context)
6. Read out the results: `grep "^primary_metric:\|^metric_std:\|^guardrails:\|^subgroups:\|^peak_mem_gb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log`, read the trace, attempt a fix. Give up after a few attempts.
8. **Decide keep/discard (validation only) — Pareto rule:** follow the `pareto:` verdict the run printed (see *Multi-objective optimization*).
   - **Keep** (advance the branch, keep the commit) if the run is feasible (guardrails pass) and **non-dominated** — it extends the front or dominates prior points. The harness has already updated `pareto.tsv`.
   - **Discard** (`git reset` to where you started) if it is infeasible or **dominated** by an existing front member. Treat wins smaller than seed noise as ties, not progress.
9. Record the results in `results.tsv` (untracked).

**Test-set protocol (rare, gated):** Do NOT evaluate on the held-out test set every loop. Touch it only (a) once to characterize the baseline, and (b) when a candidate has shown a robust, repeated validation improvement and you want to confirm it — and **flag this for the human** rather than using test results to drive further selection. Repeatedly tuning against test destroys its validity; the test number is a final readout, not a development signal.

**Timeout:** Kill any run that exceeds the configured budget and treat it as a failure.

**Crashes:** If something is dumb and easy to fix (typo, missing import), fix and re-run. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP (with one carve-out):** Once the loop has begun, do NOT pause to ask whether to continue the *validation* development loop — iterate autonomously and indefinitely until manually stopped. If you run out of ideas, think harder: re-read the in-scope files, mine the literature for new angles, combine near-misses, try more radical changes. **The single exception:** the held-out **test set** is governed by the test-set protocol above and is surfaced to the human — never let the loop start optimizing against it on its own.
