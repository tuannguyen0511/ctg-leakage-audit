# Single-factor perturbation design for data leakage: application to intrapartum CTG (CTU-UHB)

Code, analysis plan, split files and frozen outputs for the manuscript:

> Nguyen Diep QT, Huynh HN, Tran TN, Ching CTS. *A single-factor perturbation design for quantifying data leakage in clinical machine learning, with application to intrapartum cardiotocography.* (submitted to Artificial Intelligence in Medicine)

## What the study does
The design changes one evaluation step at a time on identical data, folds, models and seeds, and adds label-permutation controls. It was applied to the CTU-UHB intrapartum CTG database: 552 recordings, primary outcome umbilical artery pH < 7.05, primary cohort 518 recordings with 30 events.

**Models**
- Logistic regression and gradient boosting (GBDT) on guideline-based FHR features.
- A 1D CNN and a 1D ResNet on the FHR.

**Protocols**
- **Strict protocol S:** recording-level repeated nested cross-validation, 10 × 5 × 5.
- **Lenient protocols:** each changes exactly one step of S.

| Protocol | Change |
|---|---|
| L1 | segment-level splitting |
| L2 | oversampling before splitting |
| L3 | selection on full or test data |
| L4 | threshold chosen on the test fold |

## Data (not included)
- CTU-UHB Intrapartum Cardiotocography Database 1.0.0 (PhysioNet): https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
- FHRMA dataset and toolbox: https://github.com/utsb-fmm/FHRMA

## Layout
| Folder | Content |
|---|---|
| `src/` | Analysis scripts, task order T01 → T11 (see `manifests/RUN_REGISTRY.md`) |
| `common/` | Shared helpers imported by `src/` (I/O, metrics, bootstrap, splits) |
| `configs/` | One YAML file per task |
| `tests/` | Unit tests on synthetic data |
| `splits/` | `split.csv` (518 recordings) and `split_noexcl.csv` (552): fixed fold assignments |
| `plan/` | Analysis plan with dated amendments (in Vietnamese). An English summary of the amendments is in Supplementary Table S1 of the paper |
| `manifests/` | SHA-256 manifests of frozen outputs, run registry, number-trace tables |
| `outputs/` | Frozen aggregate outputs: tables, figures and summaries. `t11a_review` and `t11b_review` hold the post hoc analyses; `t10c_figures` holds the manuscript figures |
| `literature/` | Coded literature audit of 34 CTU-UHB studies, with verification notes |

## Running the code
The scripts import `common.*` and `P21_CTG_Acidemia.src.*`, so they expect this layout:

```
<work_root>/
  common/                       # copy of this repository's common/
  P21_CTG_Acidemia/             # this repository
  data_raw/ctu-uhb-ctgdb/1.0.0/ # PhysioNet download
  data_proc/ctu-uhb-ctgdb/1.0.0/
```

1. Clone into `<work_root>/P21_CTG_Acidemia`.
2. Copy `common/` to `<work_root>/common`.
3. Set the absolute paths in `configs/*.yaml` to your `<work_root>`.
4. From `<work_root>`, run the tasks in order:
   ```
   python -m P21_CTG_Acidemia.src.t02_build_inputs --config P21_CTG_Acidemia/configs/t02.yaml
   ```
   Continue as listed in `manifests/RUN_REGISTRY.md`. Then run `t11a_review_oof`, `t11b_review_runs` and `t10c_figures`; these three take no arguments.

**Notes on reproduction**
- **Hardware and seeds:** networks were trained on CPU (PyTorch, 4 threads). Exact agreement across hardware is not guaranteed.
- **MATLAB scripts:** `src/t07c_wmfb_*.m` need the FHRMA toolbox.
- **Image-based reproduction:** outputs of T08 and T08b are not included because of their size.
- **Hard-coded paths:** some scripts contain absolute Windows paths from the original machine. Edit them before running.

## License
Code: MIT (see `LICENSE`). The CTU-UHB and FHRMA data and the FHRMA toolbox keep their own licences and are not redistributed here.

## Citation
See `CITATION.cff`. The archived release has a Zenodo DOI.

## Contact
Corresponding authors: Trung Nghia Tran (ttnghia@hcmut.edu.vn) and Congo Tak Shing Ching (tsching@nchu.edu.tw).
