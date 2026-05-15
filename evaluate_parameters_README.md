# `evaluate_parameters*.py` — version history

The six `evaluate_parameters{,2,3,4,5,6}.py` scripts are iterative revisions of
the same analysis: ICC(2,1) / Pearson / Bland–Altman agreement and
measurement-time comparisons between **automated** (model-predicted) and
**manual** (expert-annotated) cervical radiographic parameters.

They are preserved as separate files (rather than collapsed into one) so that
the exact code used for each round of paper review is recoverable.

## Which one is canonical?

Use **`evaluate_parameters.py`** (the unnumbered file) — it is the final
publication version, matching the figure layouts described in the paper.
The other five are progressively older revisions kept for traceability.

## What differs between revisions

| Script                        | Lines | Key differences vs. the previous revision |
|------------------------------ |------:|-------------------------------------------|
| `evaluate_parameters2.py`     |   592 | First multi-figure layout (Fig1 alignment / Fig2 morphometry / Fig3 dynamic). |
| `evaluate_parameters3.py`     |   600 | Parameter renaming (`Ishihara Index` → `Lordosis Index`, `H/D Ratio` → `DHR`). |
| `evaluate_parameters4.py`     |   610 | Bland–Altman limit-of-agreement formula and CI display refinements. |
| `evaluate_parameters5.py`     |   787 | Adds the **ICC forest plot** (Fig4) summarising all 11 parameters. |
| `evaluate_parameters6.py`     |   826 | Replaces the manual ICC implementation with `pingouin.intraclass_corr` for validated CIs (with a manual fallback). |
| `evaluate_parameters.py`      |   826 | Canonical final version (label tweaks vs. `_6`). |

All scripts share the same data-loading entry point and read prediction /
ground-truth keypoint files from the same on-disk layout, so reviewers wishing
to compare an earlier revision can do so without changing any other code.

## Optional dependency

`evaluate_parameters{5,6,}.py` import `pingouin` for the canonical ICC(2,1)
computation. If pingouin is unavailable the scripts fall back to a manual
implementation. To install:

    pip install pingouin
