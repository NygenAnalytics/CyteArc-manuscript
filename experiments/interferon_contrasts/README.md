# Interferon contrasts

Which diseases raise their donors' type I interferon scores above the healthy donors of the same study, in each of five immune lineages, across the human blood stores of CyteBase?

Run: `uv run --locked python -m experiments.interferon_contrasts.lineage_contrasts`, which writes each disease-versus-healthy contrast with its AUROC per lineage, and the count and T-cell AUROC range of the interferon and control contrasts, to `.build/derived/interferon_contrasts/lineage_contrasts.json`.

Needs: `catalogue`, whose interferon screen records hold each donor's interferon scores, and `reuse`, whose progenitor experiment's severity and source-study records give COMBAT's donors their recorded cohort and split pooled stores by source study; no public datasets.
