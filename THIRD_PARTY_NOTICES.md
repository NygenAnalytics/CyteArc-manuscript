# Third-party notices

## Datasets

The experiments in `experiments/` read the public datasets below. Each row names the source that the scripts read and the licence that source states, checked at the source on 2026-11-06. "No licence stated" means the source states none; use of such data rests on the source's terms and on citing the original publication.

### CELLxGENE Discover

CELLxGENE Discover's Data Submission Policy says that anyone can access published data "subject to a CC-BY 4.0 license, meaning they can download, share, and use the data without restriction beyond providing attribution to the original data contributor(s)" ([policy](https://cellxgene.cziscience.com/docs/032__Contribute%20and%20Publish%20Data)). Each dataset below is CC BY 4.0 under that policy.

| Dataset | Used by | Source |
|---|---|---|
| Human Lung Cell Atlas, core (version `688185ad-11c2-4172-a53a-f4f1f4076860`) and full (version `5f863718-02a6-46cb-83e7-52be988b915b`) | lung, catalogue, mapping | Collection `6f6d381a-7701-4781-935c-db10d30de293`, "The integrated Human Lung Cell Atlas" |
| Natri interstitial lung disease cohort (version `c3d9262e-0dc5-4eca-bf20-56e6d96d0306`) | lung | Collection `07e12576-b41b-4350-adfc-059bfc4328ea`, "Single-cell RNA-seq analysis of Interstitial Lung Disease (ILD) subtypes" |
| Mouse embryo atlas (version `1bc30289-9565-4099-abf9-3326328c11ac`) | benchmarks, reuse | Collection `45d5d2c3-bc28-4814-aed6-0bb6f0e11c82`, "A single-cell transcriptional timelapse of mouse embryonic development, from gastrula to pup" |
| Tabula Sapiens, all cells (version `1c88f927-bcbb-4bb1-9881-281842945a2d`) | benchmarks, catalogue, reuse | Collection `e5f58829-1a66-40b5-a624-9046778e74f5`, "Tabula Sapiens". The consortium's [data release policy](https://tabula-sapiens.sf.czbiohub.org/whereisthedata) also asks users to cite the consortium's publication and says that redistribution of the data should include the full text of the policy. |
| Human Brain Cell Atlas v1.0, all neurons (version `937f8304-300e-46ff-a84c-28df40b59d2b`) | benchmarks | Collection `283d65eb-dd53-496d-adb7-7570c7caa443`, "Human Brain Cell Atlas v1.0" |
| PsychAD prefrontal cortex, MSSM cohort (version `0e853475-e298-4b09-881a-ed0b60d5a8c9`) | benchmarks | Collection `84ce6837-548d-4a1f-919f-0bc0d9a3952f`, "Population-scale cross-disorder atlas of the human prefrontal cortex at single-cell resolution" |
| Binvignat rheumatoid arthritis PBMCs (version `3b751975-34bb-409a-a9b7-98380f0450ea`) | integration | Collection `e1a9ca56-f2ee-435d-980a-4f49ab7a952b`, "Single-cell RNA-seq analysis reveals cell subsets and gene signatures associated with Rheumatoid Arthritis Disease Activity" |
| Hao 2021 PBMCs (version `4078abd1-063d-4b35-aa04-324dddf8244e`) | mapping | Collection `b0cf0afa-ec40-4d65-b570-ed4ceacc6813`, "Integrated analysis of multimodal single-cell data" |
| Yoshida 2022 PBMCs (version `4554d5bf-9182-47af-8caf-6b2c93513829`) | mapping | Collection `03f821b4-87be-4ff4-b65a-b5fc00061da7`, "Local and systemic responses to SARS-CoV-2 infection in children and adults" |
| Wilk 2020 PBMCs (version `8d7fad46-2f37-4aa0-ac9d-babbc36efb6f`) | mapping | Collection `a72afd53-ab92-4511-88da-252fb0e26b9a`, "Single-cell atlas of peripheral immune response to SARS-CoV-2 infection" |
| Lee 2020 PBMCs (version `5e491a70-a1ea-4469-9ffa-9ddff0ec3910`) | mapping | Collection `4f889ffc-d4bc-4748-905b-8eb9db47a2ed`, "Immunophenotyping of COVID-19 and influenza highlights the role of type I interferons in development of severe COVID-19" |
| Lake 2025 kidney (version `255ce81a-eb4e-48ff-9f18-e50455ccb24d`) | mapping | Collection `9c9d04c4-8899-417f-bb6f-6107dcadf14f`, "Cellular and Spatial Drivers of Unresolved Injury and Functional Decline in the Human Kidney" |
| Stewart 2019 kidney (version `c171ca0b-5e46-4260-9dc2-c03486499308`) | mapping | Collection `120e86b4-1195-48c5-845b-b98054105eec`, "Spatiotemporal immune zonation of the human kidney" |
| Reck 2025 kidney (version `0fe5eee4-380d-4bd9-8735-ada5e03021d9`) | mapping | Collection `307ce143-7cc8-4813-99b1-3797834149c9`, "snRNA-seq of human kidney with ureteral obstruction" |

### CyteBase stores on Hugging Face

| Dataset | Used by | Source | Licence as stated |
|---|---|---|---|
| CyteBase release 2610: its catalogue, object listing and stores, among them the human blood stores of the in-place screen, the four COVID-19 and ten blood studies of the composites, and the Gong, Perez, Ren, Sound Life, van der Wijst, Wells and Yoshida stores | catalogue, reuse, mapping | Hugging Face bucket `Nygen/cytebase-2610`; each store converts one CELLxGENE Discover dataset, which the release catalogue names | No licence stated for the bucket. The `Nygen/cytebase` README says "Source datasets keep the license under which they were published on CELLxGENE Discover", which is CC BY 4.0 under the policy above. |

### CyteArc documentation datasets

The `cytearc_docs` folder of the Hugging Face bucket `Nygen/cytebase` holds the stores below. Its README says "Each dataset keeps the license or terms of its source".

| Store | Used by | Original source | Licence as stated by the original source |
|---|---|---|---|
| `kang_15K_pbmc_rnaseq`, `kang_14K_ifnb-pbmc_rnaseq`, `kang_29K_ctrl-ifnb_pbmc_rnaseq` | methods, integration, mapping, reuse | Kang et al. 2018, GEO GSE96583 | No licence stated (see GEO below) |
| `tenx_5K_pbmc_rnaseq` | methods, reuse | 10x Genomics, "5k Peripheral Blood Mononuclear Cells (PBMCs) from a Healthy Donor (v3 chemistry)" | CC BY 4.0 |
| `tenx_8K_pbmc_citeseq` | integration | 10x Genomics, "10k PBMCs from a Healthy Donor - Gene Expression with a Panel of TotalSeq™-B Antibodies" | CC BY 4.0 |
| `tenx_1.3M_brain_rnaseq` | benchmarks | 10x Genomics, "1.3 Million Brain Cells from E18 Mice" | CC BY 4.0 |
| `swanson_7K_pbmc_teaseq` | integration, methods | Swanson et al. 2021, Zenodo [6360802](https://zenodo.org/records/6360802) and the eLife article 63632 source data (GEO GSE158013) | CC BY 4.0 (Zenodo record and eLife article); GEO GSE158013 states none |
| `bastidas-ponce_4K_pancreas-d15_rnaseq` | methods | Bastidas-Ponce et al. 2019, GEO GSE132188, as `endocrinogenesis_day15.h5ad` of the `theislab/scvelo_notebooks` repository | No licence stated |
| `tabula_sapiens` | catalogue, reuse | Tabula Sapiens on CELLxGENE Discover, version `1c88f927-bcbb-4bb1-9881-281842945a2d` | CC BY 4.0, with the Tabula Sapiens data release policy above |

### GEO

NCBI says that it "places no restrictions on the use or distribution of the data" in its molecular databases, and that "some submitters of the original data (or the country of origin of such data) may claim patent, copyright, or other intellectual property rights in all or a portion of the data" ([NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/)). None of the records below states a licence.

| Record | Used by | Content |
|---|---|---|
| GSE164378, samples GSM5008737 and GSM5008738 | integration | Hao et al. 2021 PBMC CITE-seq, 3' RNA and ADT counts and cell metadata |
| GSE96583 | methods | Kang et al. 2018 PBMCs: the demuxlet calls of batch 2 |
| GSE108313, samples GSM2895282 and GSM2895283 | methods | Stoeckius et al. 2018 cell hashing: hashtag RNA and HTO counts |
| GSE245108 | methods | An immunophenotype-coupled transcriptomic atlas of human hematopoietic progenitors: the `TNC-1-2-1` filtered matrix |
| GSE283672, sample GSM8669389 | methods | Monocyte PU.1 CUT&RUN, the factor identity of ChIP-Atlas SRX26995709 |

### Other sources

| Dataset | Used by | Source | Licence as stated |
|---|---|---|---|
| Tahoe-100M, revision `2dc57900b7981cfcf5e211527169a0b006546a95` | benchmarks, tests | [huggingface.co/datasets/tahoebio/Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) | CC0 1.0 (dataset card) |
| COMBAT CITE-seq atlas | integration | Zenodo [6120249](https://zenodo.org/records/6120249), `COMBAT-CITESeq-DATA.h5ad` | CC BY 4.0 |
| Stephenson et al. 2021 COVID-19 PBMCs | integration | `haniffa21.processed.h5ad` from the [COVID-19 Cell Atlas](https://www.covid19cellatlas.org/index.patient.html); the article names ArrayExpress E-MTAB-10026 | No licence stated on the portal or in the ArrayExpress record; EMBL-EBI's terms of use add no restrictions beyond those of the data owners |
| Xi and Li doublet benchmark, 16 labelled matrices | methods | Zenodo [4062232](https://zenodo.org/records/4062232), `real_datasets.zip` | CC BY 4.0 |
| LARRY lineage tracing (Weinreb et al. 2020) | methods | Zenodo [11401394](https://zenodo.org/records/11401394), `Weinreb_in_vitro.h5ad` | CC BY 4.0 |
| 10x Genomics `pbmc_1k_v2` and `pbmc_10k_v3`, raw and filtered matrices | methods | 10x Genomics datasets "1k PBMCs from a Healthy Donor (v2 chemistry)" and "10k PBMCs from a Healthy Donor (v3 chemistry)" | CC BY 4.0 |
| 10x Genomics Multiome `pbmc_granulocyte_sorted_3k` | methods | 10x Genomics dataset "PBMC from a Healthy Donor - Granulocytes Removed Through Cell Sorting (3k)" | CC BY 4.0 |
| GENCODE release 32: basic annotation GTF and GRCh38 primary assembly | methods | [GENCODE](https://www.gencodegenes.org/pages/data_access.html), EMBL-EBI FTP `gencode/Gencode_human/release_32/` | No licence stated; GENCODE says "All the GENCODE project data is open access", under the EMBL-EBI terms of use |
| JASPAR 2024 CORE vertebrates, non-redundant PFMs | methods | [jaspar.elixir.no](https://jaspar.elixir.no/about/) | CC BY 4.0 |
| cisTarget hg38 gene-based motif rankings (refseq_r80, mc_v10_clust), motif annotations (`motifs-v10nr_clust-nr.hgnc`) and `allTFs_hg38.txt` | methods | [resources.aertslab.org/cistarget](https://resources.aertslab.org/cistarget/) | No licence stated |
| ChIP-Atlas peaks of SRX26995709 | methods | [chip-atlas.org](https://chip-atlas.org/view?id=SRX26995709) | chip-atlas.org states CC BY 4.0; the [LSDB Archive licence page](https://dbarchive.biosciencedbc.jp/en/chip-atlas/lic.html) for ChIP-Atlas states CC BY-SA 4.0 |

## Code and expressions adapted from other projects

| File | Adapted from | Licence |
|---|---|---|
| `manuscript/unsrtnat-short.bst` | `unsrtnat.bst` of the natbib package, copyright 1993-2007 Patrick W. Daly, with author lists shortened to ten names | LaTeX Project Public License, version 1 or later, as the file's header states |
| `experiments/integration/multimodal_seurat.R` (its `matched` mode) and the golden outputs in `experiments/integration/fixtures/` | The modality-weight kernel, score and softmax of `FindModalityWeights` in [Seurat](https://github.com/satijalab/seurat) 5.5.1, evaluated on CyteArc's candidate pools to compare the two implementations | MIT, copyright 2021 Seurat authors ([CRAN](https://cran.r-project.org/web/packages/Seurat/LICENSE), checked on 2026-11-16) |
