# STRIPES

**STRIPES** (Spatio-Temporal Representation of Interactions in Protein-ligand Engagement Strings) is a molecular fingerprinting method that encodes per-atom protein–ligand interactions from MD trajectories into symbolic strings. This repository contains the full pipeline: extraction from simulations, pairwise similarity computation, SMILES generation via a pretrained Transformer, and embedding visualization with t-SNE.

---

## Repository structure

```
STRIPES/
├── STRIPES_similarity/         # Pairwise STRIPES similarity (Hungarian algorithm)
│   ├── similarity_function_hungarian.py
│   └── requirements.txt
├── STRIPES2SMILES/             # BERT-style pretraining + SMILES decoder finetuning
│   ├── pretraining.py
│   ├── finetuning.py
│   ├── smiles_utils.py
│   ├── run_pretraining.py      # entry point — pretraining
│   ├── run_finetuning.py       # entry point — finetuning (Optuna HPO)
│   ├── generate.py             # entry point — SMILES generation
│   ├── extract_test_and_generate.py  # batch generation over the finetuning test sets
│   ├── causal_generation/      # STRIPES sequences for perturbation experiments (Fig. 3e, Supp. Fig. S6)
│   │   └── PIM1/
│   │       └── stripes.txt     # PIM1 sequences with manually perturbed interaction tokens
│   └── requirements.txt
├── t-SNE/                      # t-SNE grid search on STRIPES embeddings
│   ├── tsne_stripes_gridsearch.py
│   └── requirements.txt
├── PubChem_analysis/           # PubChem search + bioactivity analysis on generated molecules
│   ├── pubchem_search.py
│   └── requirements.txt
├── MD/                         # MD-derived results and STRIPES similarity per dataset
├── misc/                       # Analysis / figure-generation scripts used for the paper
├── figures/                    # Paper figure assets
└── data/
    ├── MISATO/                 # Pretraining dataset (26 MB)
    ├── PPAR/                   # Finetuning dataset
    ├── PIM1/                   # Finetuning dataset
    ├── JAK1/                   # Finetuning dataset
    └── AR/                     # Finetuning dataset
```

> **Note:** STRIPES extraction from GROMACS MD trajectories (`STRIPES_extractor`) lives in a separate repository and is not part of this codebase.

---

## Requirements

Python >= 3.8 is required. GPU support (CUDA) is optional but strongly recommended for `STRIPES2SMILES`.

To install all dependencies at once:

```bash
pip install -r requirements.txt
```

Each module also ships its own `requirements.txt` if you only need a specific component:

```bash
pip install -r <module>/requirements.txt
```

---

## STRIPES format

A STRIPES string encodes per-atom interaction profiles across MD frames. Atoms are separated by `;`; within each atom, per-frame interaction tokens are separated by `.`.

Interaction tokens:

| Token | Interaction |
|-------|-------------|
| `H(a1)` / `H(a2)` / `H(a3)` | H-bond — ligand as acceptor (strong / moderate / weak) |
| `H(d1)` / `H(d2)` / `H(d3)` | H-bond — ligand as donor (strong / moderate / weak) |
| `B` | Hydrophobic interaction |
| `S(-)` / `S(+)` | Salt bridge (negative / positive ligand charge) |
| `X` | Halogen bond |
| `P(f)` / `P(o)` / `P(t)` | π–π stacking (face-to-face / offset / T-shaped) |
| `C(p)` / `C(t)` / `C(e)` | Cation–π interaction (parallel / tilted / edge) |
| `-` | No interaction |

Example: `H(a1).H(a1).B;-.-.−;...`

---

## 0 — Token vocabulary

The pairwise-similarity (section 2), pretraining (section 3a), and t-SNE (section 4) steps all require a STRIPES token-to-index mapping at `data/stripes_tokens2label.json`. Generate it once from the MISATO dataset:

```bash
python misc/stripes_token2label.py
```

Reads `data/MISATO/dataset.csv` (column `STRIPES`) and writes `data/stripes_tokens2label.json`.

---

## 1 — STRIPES extraction

STRIPES fingerprints are extracted from GROMACS MD trajectories (`.tpr`/`.xtc` plus `.itp` topology files) into a CSV with columns `mol_id`, `STRIPES`. The extraction tool lives in a **separate companion repository** and is not included here — this repo picks up the pipeline starting from the resulting STRIPES CSVs (see `data/`).

---

## 2 — Pairwise similarity

Computes all-vs-all STRIPES similarity using the Hungarian (optimal bipartite matching) algorithm on per-atom Jaccard similarity.

```bash
pip install -r STRIPES_similarity/requirements.txt

python STRIPES_similarity/similarity_function_hungarian.py \
    --dataset     data/PPAR/dataset.csv \
    --token-index data/stripes_tokens2label.json \
    --output      ppar_similarities.csv
```

**Input CSV** must contain columns: `STRIPES`, `smiles`, `pKi`.

**Output:** CSV with columns `smiles1`, `smiles2`, `STRIPES1`, `STRIPES2`, `pKi1`, `pKi2`, `similarity`.

---

## 3 — STRIPES2SMILES

A two-stage deep learning pipeline: (i) BERT-style masked language model pretraining on STRIPES, (ii) encoder–decoder finetuning for STRIPES → SMILES translation with Optuna hyperparameter optimization.

```bash
pip install -r STRIPES2SMILES/requirements.txt
```

### 3a — Pretraining

```bash
python STRIPES2SMILES/run_pretraining.py \
    --data_path  data/MISATO \
    --output_dir results_pre
```

Key options: `--d_model 512`, `--n_heads 8`, `--n_layers 8`, `--batch_size 8`, `--num_epochs 100`, `--lr 1e-4`, `--seed 42`.

**Outputs in `results_pre/`:**
- `pretrained_stripes_encoder.pth` — best encoder checkpoint
- `stripes_vocab.pkl` — vocabulary for finetuning
- `training_metadata.json` — loss curves and model config
- `pretraining_loss.png` — training/validation loss plot

### 3b — Finetuning

```bash
python STRIPES2SMILES/run_finetuning.py \
    --data_path        data/ \
    --pretrained_model results_pre/pretrained_stripes_encoder.pth \
    --pretrained_vocab results_pre/stripes_vocab.pkl \
    --output_dir       results_fine \
    --datasets PPAR PIM1 JAK1 AR \
    --n_trials 100
```

Each dataset directory must contain a `dataset.csv` with columns: `STRIPES`, `can_smiles`, `pKi` (`mol_id` is optional — auto-generated from the row index if missing). Only rows with `pKi >= 6.0` are used for finetuning.

**Outputs per dataset in `results_fine/`:**
- `<DATASET>_model.pth` — finetuned model checkpoint
- `<DATASET>_config.json` — model config and best Optuna hyperparameters
- `<DATASET>_optuna.json` — full Optuna trial log
- `<DATASET>_split.json` — train/val/test split sizes
- `<DATASET>_test_set.csv` — held-out test set (`mol_id`, `STRIPES`, `can_smiles`), used as input for generation (see 3c below)
- `<DATASET>_metrics.json` — training stats
- `<DATASET>_plots.png` — training/validation loss curves
- `summary.json` — aggregated metrics across all datasets

### 3c — Batch generation over test sets

Generates SMILES for the held-out test set of each dataset, sweeping over `beam_size`, `n_molecules`, `temperature`, and `temperature_increment` (used to produce the results reported in the paper).

```bash
python STRIPES2SMILES/extract_test_and_generate.py \
    --pretrained_model results_pre/pretrained_stripes_encoder.pth \
    --pretrained_vocab results_pre/stripes_vocab.pkl \
    --results_dir      results_fine \
    --datasets         PIM1 JAK1 AR \
    --beam_sizes       5 10 15 \
    --n_molecules      5 10 \
    --temperatures     1.2 1.4 \
    --increments       -0.2 -0.3 -0.4 -0.5
```

Reads `<DATASET>_test_set.csv` (produced by `run_finetuning.py`, see 3b). For each parameter combination, writes to `results_fine/<DATASET>_finetuned/comparison/`:
- `generated_beam<b>_N<n>_T<t>_step<s>.csv` — generated molecules
- `generated_beam<b>_N<n>_T<t>_step<s>_metrics.json` — validity/uniqueness/novelty

To select the unique generated molecules across the whole sweep — e.g. as a starting point for downstream MD simulations and STRIPES-similarity analysis (see `MD/`) — deduplicate them by `canonical_smiles`:

```bash
python misc/merging_generated_mols.py \
    --folder results_fine/PPAR_finetuned/comparison
```

Writes `all_unique_molecules.csv` to `<folder>/all_generated_combined/`.

### 3e — Perturbation experiments (Fig. 3e, Supp. Fig. S6)

To test whether modifying individual interaction tokens in a STRIPES sequence produces chemically coherent changes in the generated molecules, targeted perturbations were applied to representative PIM1 sequences: hydrogen-bond, hydrophobic contact, and salt bridge tokens were independently added or removed. The perturbed sequences are provided in `STRIPES2SMILES/causal_generation/PIM1/stripes.txt` (CSV with columns `mol_id`, `STRIPES`).

Generate SMILES from the perturbed sequences using the standard generation script (see 3d):

```bash
python STRIPES2SMILES/generate.py \
    --finetuned_model  results_fine/PIM1_model.pth \
    --pretrained_model results_pre/pretrained_stripes_encoder.pth \
    --pretrained_vocab results_pre/stripes_vocab.pkl \
    --input_csv        STRIPES2SMILES/causal_generation/PIM1/stripes.txt \
    --output           causal_generation_PIM1.csv
```

---

### 3d — Standalone SMILES generation

For ad-hoc generation from a single sequence, a CSV (column `stripes`/`STRIPES`), or a plain text file:

```bash
# Single sequence
python STRIPES2SMILES/generate.py \
    --finetuned_model  results_fine/PPAR_model.pth \
    --pretrained_model results_pre/pretrained_stripes_encoder.pth \
    --pretrained_vocab results_pre/stripes_vocab.pkl \
    --sequence "<stripes_string>" \
    --output   generated.csv

# Batch from CSV
python STRIPES2SMILES/generate.py \
    --finetuned_model  results_fine/PPAR_model.pth \
    --pretrained_model results_pre/pretrained_stripes_encoder.pth \
    --pretrained_vocab results_pre/stripes_vocab.pkl \
    --input_csv data/PPAR/dataset.csv \
    --output    generated.csv
```

Outputs `<output>.csv` (columns `mol_id`, `can_smiles`, `stripes`, `rank`, `smiles`, `canonical_smiles`, `is_valid`) and `<output>_metrics.json` (validity, uniqueness, novelty).

---

## 4 — t-SNE visualization

Grid search over 36 combinations of t-SNE hyperparameters (perplexity, n_iter, learning_rate) evaluated by trustworthiness and continuity.

```bash
pip install -r t-SNE/requirements.txt

python t-SNE/tsne_stripes_gridsearch.py \
    --vocab_path  data/stripes_tokens2label.json \
    --data_path   data/MISATO/dataset.csv \
    --output_dir  results/tsne_grid_search \
    --svg_save_dir results/figures
```

**Input CSV** (`dataset.csv`) must contain columns: `STRIPES`, `lig_MW`, `lig_logP`, `lig_TPSA`, `lig_H_donor`, `lig_H_acceptor`, `hydrophobic_atoms`, `polar_atoms`, `net_charge`, `frac_hydrophobic`, `frac_polar`, `frac_positive`, `frac_negative`, `mean_hydropathy`, `mean_sasa`.

**Outputs in `--output_dir`:**
- `quality_metrics.csv` — trustworthiness and continuity for all 36 configurations
- `best_configuration_results.csv` — t-SNE coordinates for the best configuration
- `experiment_summary.txt` — full experiment report
- `<config>/tsne_<config>_<property>.svg` — per-property SVG plots for every configuration
- `results/figures/tsne_BEST_<config>_<property>.svg` — SVG plots for the best configuration
- `correlation_*.png` — correlation matrices

---

## 5 — PubChem analysis

Validates generated molecules against PubChem and retrieves bioactivity data (EC50, IC50, Ki, Kd, AC50) for each target.

```bash
pip install -r PubChem_analysis/requirements.txt

# Run once per dataset
python PubChem_analysis/pubchem_search.py \
    --dataset     PPAR \
    --results_dir results_fine \
    --data_dir    data \
    --output_dir  results_pubchem/PPAR
```

**`--results_dir` layout expected** (produced by `extract_test_and_generate.py` + `misc/merging_generated_mols.py`, see 3c):
```
results_fine/
└── <DATASET>_finetuned/comparison/all_generated_combined/
    └── all_unique_molecules.csv    # must contain column 'canonical_smiles'
```
Alternatively, pass `--input <path_to_all_unique_molecules.csv>` directly to bypass `--results_dir`.

**`--data_dir` layout expected** (same as the rest of the pipeline):
```
data/<DATASET>/dataset.csv    # must contain column 'can_smiles' (used as the novelty reference set)
```

**Output:** `<DATASET>_bioactivity_results.csv` (saved to `--output_dir`, or alongside the input CSV if not given) — for each novel molecule found on PubChem and/or ChEMBL: `canonical_smiles`, `target`, `exists_on_pubchem`, `exists_on_chembl`, and the lowest reported `EC50_uM`/`IC50_uM`/`Ki_uM`/`Kd_uM`/`AC50_uM` against the target.

---

## Reproducibility

All random seeds are fixed to 42. Results may vary slightly across hardware and software versions due to non-deterministic GPU operations.

## How to cite

If you use this repository in your work, please cite:

> Criscuolo, E., & Grisoni, F. (2026). *Towards a physically interpretable symbolic language of molecular recognition.* ChemRxiv. https://doi.org/10.26434/chemrxiv.15000358

<details>
<summary>BibTeX</summary>

```bibtex
@article{criscuolo2026towards,
  title   = {Towards a physically interpretable symbolic language of molecular recognition},
  author  = {Criscuolo, Emanuele and Grisoni, Francesca},
  year    = {2026},
  journal = {ChemRxiv},
  doi     = {10.26434/chemrxiv.15000358},
  note    = {Preprint},
  url     = {https://doi.org/10.26434/chemrxiv.15000358}
}
```
</details>

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
