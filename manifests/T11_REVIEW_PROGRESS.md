# P21 T11 – review 2026-11-14: progress (Claude, 2026-11-16)

All analyses are post hoc (to be logged in Table S1 and NUMBERS_TRACE).

## T11a – done (frozen OOF only; outputs/t11a_review/)
| Item | Result |
|---|---|
| E5 single-model AUROC (mean ± SD over repetitions×seeds) | M0 0.572±0.034; M1 0.598±0.024; CNN 0.673±0.035; ResNet 0.691±0.022; CNN+UC 0.677±0.040. Ensemble: 0.580/0.601/0.715/0.733/0.708 |
| E5 H1 on single-model estimand | CNN−GBDT +0.075 (−0.022, 0.173), Holm p 0.13; ResNet−GBDT +0.094 (−0.000, 0.187), Holm p 0.10; TOST inconclusive; gate not met |
| E0 CNN+UC vs CNN seed 0 | −0.005 ± 0.030 per repetition |
| E5 H2 per-unit deltas | L1 GBDT +0.084±0.016, CNN +0.145±0.033; L2 +0.235±0.058 / +0.331±0.034; L3 −0.003±0.028 / +0.059±0.039 |
| E1 L1 interaction (CNN−GBDT inflation) | ensemble segment +0.111 (0.064, 0.159); single-model segment +0.061 (0.027, 0.095); recording-level +0.114 (0.062, 0.170) |
| L4 Youden J | GBDT 0.155→0.347, ΔJ +0.191 (0.092, 0.291); CNN 0.322→0.457, ΔJ +0.135 (0.074, 0.204) |
| E6 cross-fitted recalibration | Brier raw 0.11–0.22 → 0.054–0.055 (null 0.0548); scaled Brier ≈0 (−0.011 to 0.008); slope 0.27–0.74; DCA: networks NB > treat-all only at ~3–8%, ≈0 above 10% |
| E4 exclusion × delivery | all 10 excluded acidaemic records were caesarean; within caesarean OR 1.10 (Fisher p 1.0); MH OR by delivery 1.01 (0.28–3.59); by stage-2 1.71 (0.42–6.99). Shift 5 min: crude OR 5.85; shift 10 min: 2.02 (p 0.30) |
| E10b | pH<7.05 & BDecf≥12: 11 events (6 acidaemic records lack BDecf); single-model AUROC GBDT 0.566, CNN 0.695, ResNet 0.685 |
| E8b L2 overlap | duplication: 100% of positive test rows have a copy in training; SMOTE: median NN distance 0.21 vs 2.51 under strict |
| Table 1 extension | outputs/t11a_review/T1_extended.csv |

Key consequence: the "outcome-associated exclusion" is explained by delivery mode / window timing (C2 confirmed). Text, highlight 3 and abstract must change.

## T11b – waiting for Tuan's PC-A run (see P21_T11b_RUN.md)
E2, E2b, E3, E4d, E7, E8a, E8c, E9, E10a, E1b.

## Not done by experiment
E11 second coder + κ needs a human coder; E12–E14 (Tier 3) not run.
