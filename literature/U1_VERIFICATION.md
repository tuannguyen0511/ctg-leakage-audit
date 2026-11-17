# U1 — Director verification (2026-11-13)
Inputs: U1_partA_report_RAW.md (36 studies), U1_refs_RAW.md (46 references, B1–B7). Outputs: U1_extraction.csv (corrected + Director columns), U1_summary.txt (recomputed in code, u1_build.py).
Verdict: Part A ACCEPTED WITH CORRECTIONS; Part B ACCEPTED WITH CORRECTIONS. The AUROC medians in the raw Part A report are NOT used (miscomputed: S003 has no AUROC; record/external designs mixed); use U1_summary.txt only.

## Part A — checks
- Existence/DOI: 24/37 records individually located (search/full text); the rest are well-known DOIs from the seed list. Unresolved: S016 DOI, S017 DOI, S026 dataset.
- Quote checks (full text opened): S022, S025, S027, S028, S029, S033, S034, S035 = 8/36 (22%).
- Corrections (details in column director_correction):
  - S027 oversampling before_split→after_split (quote: "GAN is used only for the training set").
  - S029 segment→unclear and before_split→unclear.
  - S025 image→segment.
  - S031 before_split→after_split.
  - S035 threshold on_validation→unclear.
  - S033 author = Chiou et al., DOI 10.1038/s44294-025-00068-w.
  - S036 title corrected.
  - S026 author = Liu; CTU-UHB use not verified.
  - Abstract-only rule enforced for S008/S013/S014/S036.
- Eligibility: S001 (pre-release n = 217), S004 (caesarean outcome), S023 (unsupervised, no outcome) and S026 (unverified) are out of the main set.
- Added S037 (Vargas-Calixto et al., CinC 2026; CTU-UHB external AUROC 0.743).
- Seed-list error (Director's own): 10.1007/s13755-023-00219-w is Cao et al. 2023 (private antepartum dataset), not Hybrid-FHR. Its exclusion is correct.
- Known candidates not coded (check before submission):
  - Spilka et al. sparse-SVM papers (EMBC 2015; JBHI);
  - Liu et al. 2021 ESWA (S026).
  - MIRF-Net (Bioengineering 2026) was checked and excluded: FIGO labels, not pH.
- Framing: report as a structured literature audit of CTU-UHB studies, not a systematic review/meta-analysis (search not PRISMA-complete; single coder).

## Part B — checks and required changes
| ref | status | action |
|---|---|---|
| B1-R4 Ayres-de-Campos 2010 BMC P&C 10:71 | exists | WRONG description: protocol of the computer-analysis-with-alerts RCT (not INFANT). Cite only as precedent for "pH < 7.05 and BDecf > 12" |
| B1-R5 Andersson 2024 JAMA Netw Open | exists (PMC11325209) | numbers not verified (page blocked) — verify before quoting figures |
| B1-R6 Zaigham 2025 AJOG | exists | abstract only |
| B1-R7 Bailey 2021 Am J Perinatol | exists (PMID 32323288) | DOI candidate 10.1055/s-0040-1708800 — confirm |
| B2-R3 INFANT 2017 Lancet | exists | OK |
| B2-R5 Chandraharan 2024 Med Res Arch | exists | DROP (low-tier venue; not needed) |
| B3-R4 Chiou 2025 npj Women's Health | authors and DOI confirmed (DOAJ) | OK |
| B4-R3 Houzé de l'Aulnoit 2019 BSPC 49:113–123 | exists (Lille repository) | DOI not confirmed — confirm |
| B5-R2 Brookshire 2024 Front Neurosci | exists (PMC11099244) | accuracy figures (53% vs 99.8%) not yet verified |
| B5-R5 Vandewiele 2021 AIIM | exists (arXiv 2001.06296) | OK |
| B5-R6 Adetunji & Oyewusi 2026 (arXiv 2608.17712, EMBC 2026) | exists; AUROC 0.493 [0.467–0.520] verified | "0.80–0.97" range NOT found; paper compares with 0.74–0.78 — fix wording; label as preprint |
| B6-R9 Hofman 2023, B7-R7 Delord 2026 | arXiv | preprints; optional |
| B7-R1 Karmakar 2025 npj Digit Med | quote verified: aOR 1.47 (1.19–1.81), n = 32,242; >30% dropout | OK — key reference for U3 |
| B7-R2 Chudáček 2014 | known | OK |
Quote checks in Part B: B1-R4, B3-R4, B5-R6, B7-R1 plus metadata checks of 6 others (≈ 22%).

## Key messages for the manuscript (from verified items)
- On CTU-UHB, studies with segment- or image-level evaluation report AUROC 0.92–0.99 (median 0.979, n = 4). Record-level within-CTU designs report 0.62–0.89 (median 0.728, n = 4); external tests on CTU-UHB report 0.68–0.84 (median 0.743, n = 7). This matches the present L1 result.
- No CTG leakage audit was found (Part B coverage note, B5). The closest analogues are EHG (Vandewiele 2021; Adetunji 2026) and EEG (Brookshire 2024).
- Outcome-associated signal loss is documented (Karmakar 2025). This supports U3 (exclusion removes acidaemic cases, OR 6.73).
