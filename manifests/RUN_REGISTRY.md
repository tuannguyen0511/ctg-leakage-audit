# RUN_REGISTRY — P21

| RUN_NAME | Chỉ thị | Script SHA256 (8) | Config | Thư mục output | Ngày | Ghi chú |
|---|---|---|---|---|---|---|
| t01_audit | P21-T01 | 8f46b2c0 | configs/t01.yaml | outputs/t01_audit | 2026-11-10 | PASS 6/6; 552/552 bản ghi |
| t02_inputs | P21-T02 | 150e799c | configs/t02.yaml | (không ghi output) | 2026-11-10 | FAIL tiêu chí 7–9 ca dương/fold (lỗi chỉ thị); chạy lại theo BỔ SUNG 2 |
| t02_inputs | P21-T02 | 627caeda | configs/t02.yaml | data_proc/ctu-uhb-ctgdb/1.0.0 (7 file) | 2026-11-10 | PASS 8/8; PREREG v1.1 df1d7b0c |
| t03_baselines | P21-T03 | b87eac8e | configs/t03.yaml | outputs/t03_baselines (5 config) | 2026-11-10 | PASS 7/7; NC AUROC 0.465; ĐÓNG BĂNG |
| t04_dl | P21-T04 | 746a4238 | configs/t04.yaml | outputs/t04_dl (5 config, 1555 đơn vị) | 2026-11-11 | PASS; FULL_10_ALL; NC 0.422; 24,1 h; ĐÓNG BĂNG (SHA trong T04_FROZEN_SHA.txt) |
| t06a_stats | P21-T06a | edb9c735 | configs/t06a.yaml | outputs/t06a_stats | 2026-11-12 | PASS; H1 không đạt cổng; E1–E4 xong; ĐÓNG BĂNG (T06a_FROZEN_SHA.txt) |
| t05_leakage | P21-T05 | 35232edc | configs/t05.yaml | outputs/t05_leakage | 2026-11-12 | PASS; H2 6/8 Holm; NC 0.454; ĐÓNG BĂNG (T05_FROZEN_SHA.txt) |
| t06b_synthesis | P21-T06b | 341caa20 | configs/t06b.yaml | outputs/t06b_synthesis | 2026-11-12 | TECH PASS; 457/457 khớp; NUMBERS_TRACE 7392c16e; ĐÓNG BĂNG số liệu; hình cần chỉnh (T06c) |
| t06c_figures | P21-T06c | ? | configs/t06c.yaml | outputs/t06c_figures | 2026-11-12 | F4, F5 đạt; F1, F2 lỗi trình bày; F3 thiếu AUROC trong chú giải → T06d |
| t06d_figures | P21-T06d | ? | configs/t06d.yaml | outputs/t06d_figures | 2026-11-12 | F2–F5 ĐẠT (duyệt hình); F1 cần chỉnh giãn dòng → T06d-1 |
| t06d_F1fix | P21-T06d-1 | ? | — | outputs/t06d_figures_F1fix | 2026-11-12 | F1 ĐẠT; bộ hình cuối ĐÓNG BĂNG (FIGURES_FROZEN_SHA.txt) |
| t08_deepfhr | P21-T08 | 9dc58ea2 | configs/t08.yaml | outputs/t08_deepfhr | 2026-11-12 | TECH PASS; KHOA HỌC: không đạt (3/9 tổ hợp seed×lần lặp; mô hình không học, xác suất hằng ở nhiều fold) → T08b. GIỮ NGUYÊN, không xóa |
| t08b_deepfhr | P21-T08b | 56db4282 | configs/t08b.yaml | outputs/t08b_deepfhr | 2026-11-13 | Chạy đủ 180+5; ASSAY SENSITIVITY FAIL (33/45 < 36/45) → U4 KẾT THÚC, "không kết luận được"; ĐÓNG BĂNG |
| t07_fhrma | P21-T07a/b | ? | configs/t07.yaml | outputs/t07_fhrma | 2026-11-13 | Train FHRMA (66): MAE 6.99, F1 tăng 0.23, F1 giảm 0.34 → bộ trích KHÔNG đạt; test FHRMA không có nhãn công khai |
| t07c_wmfb | P21-T07c | 27c2a458 / 3e8ea481 | configs/t07c.yaml | outputs/t07c_wmfb + data_proc/.../wmfb | 2026-11-13 | PASS; WMFB train: MAE 3.63, F1 0.62/0.67, MADI 0.064; CTU 552/552; ĐÓNG BĂNG |
| t07d_featv2 | P21-T07d | ? | configs/t07d.yaml | outputs/t07d_featv2 | 2026-11-13 | PASS 160/160; độ nhạy: M1-v2 0.590 (≈ M1); M3−M1v2 +0.143; rò rỉ L1/L2 bền; ĐÓNG BĂNG |
| manuscript v0.1 | D1 | — | — | manuscript/main.tex | 2026-11-13 | LaTeX elsarticle (CBM): Methods + Results H1/H2 nháp; Intro/Discussion chờ U1 |
| t09a_synthesis | P21-T09a | 28eeb6d3 | (Director chạy trực tiếp) | outputs/t09a_synthesis | 2026-11-13 | PASS; Fisher y705 OR 6.73 [2.63, 16.30] p 4.7e-5; NUMBERS_TRACE 248/248 |

## T10 — IEEE figure redraw (Director-executed, 2026-11-13)
- Script: src/t10_figures.py (run in Director container from SHA-verified staged inputs; outputs copied byte-identical, SHA verified on PC-A).
- Inputs: 7 frozen plot_data CSVs (SHA-checked) + reports/lit/U1_extraction.csv (F8).
- Outputs: outputs/t10_figures (F1–F8 × svg/pdf/png/tif/gray + plot_data; report.txt). All automated checks PASS (text overlap, CI inside axes, legend outside, font 7–8.5 pt, spines, values unchanged, grayscale distinct).
- Manifest: reports/T10_FROZEN_SHA.txt. Presentation only; no new numbers. F8 = U1 literature AUROCs (published values).
- T10b (2026-11-13): F2 layout fix only — strict and comparator points/CI drawn on separate sub-rows (±0.065) so CI bars never share a line; connector removed. Other figures re-rendered identically from the same inputs. Script src/t10b_figures.py; outputs outputs/t10b_figures (supersedes t10_figures for the manuscript); manifest reports/T10b_FROZEN_SHA.txt.
