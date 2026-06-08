# `misc/`

Supplementary scripts: data-prep utilities and the analyses behind the paper's
figures and tables that are not part of the main STRIPES pipeline (see the
top-level `README.md` for the pipeline itself).

## Data-prep utilities (pipeline prerequisites)

- **`stripes_token2label.py`** — builds the STRIPES token vocabulary
  (`data/stripes_tokens2label.json`) consumed by the deep-learning pipeline.
  Documented as a pipeline prerequisite in the main `README.md` (section 0).
- **`add_canonical_smiles.py`** — adds a `can_smiles` (RDKit-canonicalized
  SMILES) column to a target dataset CSV.
- **`merging_generated_mols.py`** — CLI that merges and deduplicates
  generation-sweep CSVs (`generated_*.csv`) by `canonical_smiles` into a single
  `all_unique_molecules.csv`, used as the starting point for the MD + STRIPES
  similarity analysis (`MD/`) and for the PubChem/ChEMBL cross-referencing
  (`PubChem_analysis/`).

## Analyses behind statistics quoted in the paper

- **`input_output_tanimoto.py`** — for each target, computes the ECFP Tanimoto
  similarity between generation inputs (`can_smiles`) and the subset of
  generated outputs matched on PubChem/ChEMBL, underlying the similarity
  statistics reported alongside Supp. Table S6 (e.g. "ECFP Tanimoto
  similarities to the 'ground-truth' ligand ranging from 19% to 89%").
- **`max_sim_gen_dataset.py`** — computes, for each bioactive generated
  molecule, its maximum ECFP Tanimoto similarity to the full vs. training-only
  reference set, underlying the quoted statistic "maximum similarities to
  training set molecules ranging from 65% to 97%" (Results, paragraph on
  PubChem/ChEMBL cross-referencing).

## Figure/table generation — `misc/`

- **`box_plots_generatedVS_groundtruth.py`** → **Fig. 3a**
  (`figures/boxplot_similarity.svg`). ECFP Tanimoto similarity distribution
  between de novo designs (≥10 atoms) and their conditioning ligand, one
  violin/box plot per target.
- **`spearman_corr.py`** → **Supp. Table S3**
  (`figures/spearman_correlation_heatmap.png`). Spearman correlation between
  t-SNE embedding coordinates and ligand/protein/interaction physicochemical
  properties.
- **`inputVSoutput.py`** → **Supp. Table S7**. Compares pharmacophoric feature
  counts (HBA, HBD, aromatic rings, ...) across STRIPES inputs, reference
  SMILES, and generated molecules; computes the descriptive statistics
  (mean ± SD) and Spearman correlations quoted in the text and reported in
  Supp. Table S7. Outputs (merged data, feature matrix, 1-to-1 comparison
  plots, Spearman heatmap, summary/descriptive-stats CSVs) are written to
  `STRIPES2SMILES/results_finetune/all_results_merged/`.
- **`analyze_generation.py`** → **Supp. Fig. S2–S4**. Loads
  `*_metrics.json` from the generation-sweep comparison folders and produces
  the beam-size/step-size/temperature effect plots and heatmaps (validity,
  uniqueness, novelty across hyperparameter combinations), plus a full summary
  table (`analysis_summary.csv`/`.txt`).

## Figure generation — `misc/for_figures/`

- **`table1_and_S5.py`** → **Table 1** and **Supp. Table S5**. Computes, per
  target, the chemical-validity/uniqueness/novelty metrics (Table 1) and the
  fraction of generated molecules at given ECFP-similarity thresholds to the
  conditioning ligand (Table S5), for the best-performing generation
  configuration (beam 15, N5, T1.4, step 0.5).
- **`molecules4fig2.py`** → **Fig. 2c**. Selects representative PIM1 ligand
  pairs with low STRIPES similarity, low ECFP similarity, and large ΔpKi —
  the discordant-similarity examples shown in the figure.
- **`similarity_consistency_low_tanimoto_BINS.py`** → **Fig. 2b**
  (`figures/heatmap1_median_dpki.svg`, `figures/heatmap2_median_dpki_n_std.svg`).
  2D heatmaps of median ΔpKi (and ΔpKi/n/std) across STRIPES- and
  ECFP-similarity bins, per target — the joint STRIPES/ECFP similarity vs.
  bioactivity-difference analysis. Also prints representative example pairs
  for each similarity quadrant.
- **`stripes_similarity_distribution.py`** → **Fig. 3b**
  (`figures/combined_similarity_plot.svg`). Combined plot of internal STRIPES
  similarity (per-target baseline), "two-stripes"/convergent-design similarity,
  and de novo (newly MD-simulated) design similarity.
- **`molecular_descriptors_analysis.py`** → **Supp. Fig. S5**
  (`figures/descriptors/<DATASET>_descriptors_kde_distributions.png`). KDE
  comparison of physicochemical descriptor distributions between active,
  inactive, and STRIPES-generated compounds, per target.
