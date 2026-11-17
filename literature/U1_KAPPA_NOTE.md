# U1 second coding (LLM coder, blinded) – agreement, 2026-11-17

Coder 1: original extraction (U1_extraction.csv, 34 included studies). Coder 2: GPT, blinded,
prompt GUI_GPT_P21_U1_2_coder2.md. Files: U1_kappa.csv (6 raw items), U1_kappa_flags.csv
(3 audit flags used in the paper), U1_disagreements*.csv.

| Audit flag (as used in Fig. 1 / Table S2) | Coder 1 yes | Coder 2 yes | Agreement | Cohen's κ |
|---|---|---|---|---|
| Segment- or image-level split | 5 | 5 | 100% | 1.00 |
| Resampling/augmentation before split | 6 | 3 | 91.2% | 0.62 |
| Selection on test data | 3 | 14 | 67.6% | 0.24 |

Raw 6-item κ: unit 0.56, oversampling 0.19, model selection 0.16, threshold 0.00,
quality exclusion 0.13, external validation 0.73. Most raw disagreements are coding
conventions: coder 2 wrote "unclear" where the paper is silent, coder 1 wrote "none"/"yes".

Disagreement on test selection is definitional: coder 2 coded plain k-fold CV with tuning but
no separate validation set as "on_test" (11 studies, incl. 5 Cömert abstracts); coder 1 coded
only explicit statements (S007, S026, S027). Proposed resolution for the paper: report
"explicit in 3 studies; possible in up to 14 where tuning without a separate validation set
was reported or implied", and report κ in Supplement. Segment/image split counts (main
finding) are unchanged.

Disclosure needed: Methods (LLM second coder, blinded, κ, disagreements resolved by authors)
and the generative-AI declaration.
