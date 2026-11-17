# P21 — Preregistration (G1) v1.6 (v1.0 SHA256 9ce099e1…; sửa đổi ở mục 9)
Director · 2026-11-10 · Chốt TRƯỚC mọi lần train. Mọi thay đổi sau ngày này: ghi ở mục 9 (ngày, lý do, trước khi xem kết quả hay sau) và báo cáo như phân tích độ nhạy.
Căn cứ: REPORT P21-T01 (script 8f46b2c0) và D0 (`P21_D0_NOVELTY.md`).

## 1. Câu hỏi
- **H2 (chính):** Các protocol đánh giá lỏng thường gặp trong tài liệu CTU-UHB làm AUROC cao hơn protocol nghiêm theo bản ghi. Đo ΔAUROC = AUROC(protocol lỏng) − AUROC(protocol nghiêm) cho từng protocol L1–L4 (mục 6), với M1 và M2.
- **H1 (xác nhận):** Dưới protocol nghiêm, DL (M2, M3) không vượt M1 (FIGO + GBDT).

## 2. Dữ liệu và đơn vị
- CTU-CHB 1.0.0, 552 bản ghi; 1 bản ghi = 1 đơn vị phân tích. Không có mã sản phụ → không xác nhận được tính độc lập giữa các bản ghi (nêu là hạn chế).
- Chỉ dùng tín hiệu FHR (+UC ở phân tích phụ). Không dùng biến lâm sàng làm đầu vào.

## 3. Nhãn (pH động mạch rốn; không thiếu giá trị)
| Vai trò | Định nghĩa | Số ca dương (T01) | Lý do |
|---|---|---|---|
| Chính | pH < 7,05 | 40 (7,25%) | Ngưỡng toan máu chuẩn; dùng trong Mendis 2025, DeepCTG |
| Phụ 1 | pH < 7,15 | đếm ở T02 (≤ 7,15: 113) | So với DeepFHR, Xiao 2022 |
| Phụ 2 | pH < 7,20 | đếm ở T02 (≤ 7,20: 194) | So trực tiếp với bài npj Women's Health 2025 |
Không loại ca trung gian khỏi train hay test.

## 4. Cửa sổ phân tích và chất lượng tín hiệu
- **Cửa sổ chính:** 30 phút cuối của giai đoạn 1, kết thúc tại mốc bắt đầu giai đoạn 2 (trường header về vị trí giai đoạn 2). Bản ghi không có giai đoạn 2 (46): 30 phút cuối bản ghi; thêm cờ `no_stage2`.
- **Phân tích độ nhạy:** cửa sổ 60 phút (cùng điểm kết thúc).
- Làm sạch FHR (áp dụng giống nhau cho mọi mô hình): giá trị 0 → thiếu; ngoài [50, 200] bpm → thiếu; bước nhảy > 25 bpm giữa hai mẫu liên tiếp → đánh dấu thiếu mẫu sau.
- **Loại bản ghi:** > 50% mẫu FHR thiếu trong cửa sổ chính (quy tắc chỉ dựa vào chất lượng tín hiệu, không nhìn kết cục). Báo số bị loại. Độ nhạy: không loại bản ghi nào.

## 5. Mô hình
| Mã | Mô hình | Đầu vào | Chỉnh tham số |
|---|---|---|---|
| M0 | Logistic L2 | Đặc trưng (mục 5.1), chuẩn hóa | C ∈ {0,01; 0,1; 1; 10}, inner CV |
| M1 | GBDT (sklearn HistGradientBoosting) | Đặc trưng | lưới nhỏ (learning_rate × max_depth × min_samples_leaf, ≤ 12 tổ hợp), inner CV |
| M2 | CNN 1D nhỏ | FHR 1 Hz + kênh mask thiếu (1.800 mẫu) | kiến trúc cố định; early stopping trên 20% của outer-train (phân tầng); 3 seed |
| M3 | ResNet 1D cho chuỗi thời gian (3 khối residual, kiểu Wang et al. 2017) | như M2 | như M2; 3 seed |
Mất cân bằng lớp: trọng số lớp trong loss/fit (không oversampling ở protocol nghiêm).

### 5.1 Đặc trưng (tính trên mẫu hợp lệ của cửa sổ)
Baseline FHR (median trượt 10 phút, loại vùng lệch > 15 bpm); STV (Dawes, epoch 3,75 s); LTV (biên độ trung bình mỗi phút); % thời gian biến thiên tối thiểu (< 5 bpm); số nhịp tăng (≥ 15 bpm, ≥ 15 s); số nhịp giảm (≥ 15 bpm, ≥ 15 s); tổng diện tích nhịp giảm; số nhịp giảm kéo dài (≥ 2 phút); số cơn co (từ UC); công suất LF (0,04–0,15 Hz) và HF (0,15–1 Hz) của FHR đã nội suy. Không thêm đặc trưng sau khi xem kết quả.

## 6. Protocol đánh giá
- **Nghiêm (S):** repeated stratified nested CV theo bản ghi: 10 lần lặp × 5 fold ngoài; 5 fold trong để chỉnh tham số (M0, M1). Phân tầng theo nhãn chính. File cố định `data_proc\ctu-uhb-ctgdb\1.0.0\split.csv`, tạo một lần ở T02. Mọi bước học từ dữ liệu (chuẩn hóa, chọn tham số, ngưỡng) chỉ dùng outer-train.
- **Lỏng (H2), mỗi protocol thay đổi đúng MỘT yếu tố so với S:**
  - L1 — chia theo đoạn: cắt mỗi bản ghi thành các đoạn 10 phút chồng 50% trên toàn bộ phần giai đoạn 1; chia fold ngẫu nhiên theo đoạn (các đoạn cùng bản ghi có thể ở cả train và test); AUROC tính theo đoạn (như các bài gốc).
  - L2 — oversampling (SMOTE cho M1; nhân bản ngẫu nhiên lớp dương cho M2) TRƯỚC khi chia fold.
  - L3 — chọn tham số/đặc trưng trên toàn bộ dữ liệu rồi mới CV (không nested).
  - L4 — ngưỡng quyết định chọn trên chính fold test (ảnh hưởng độ nhạy/đặc hiệu, không ảnh hưởng AUROC; báo cáo ΔSe ở Sp cố định).

## 7. Metric và thống kê
- Chính: AUROC từ dự đoán OOF gộp mỗi lần lặp; báo trung bình qua 10 lần lặp + CI 95% bootstrap theo bản ghi (10.000 lần, trên dự đoán OOF trung bình qua các lần lặp).
- Phụ: AUPRC; độ nhạy tại độ đặc hiệu 85%; Brier; calibration slope/intercept; DCA (ngưỡng 5–30%).
- **Họ H2 (Holm):** ΔAUROC L1, L2, L3 so với S cho M1 và M2 (6 so sánh) + ΔSe của L4 (2 so sánh).
- **Họ H1 (Holm):** M2 − M1, M3 − M1 (2 so sánh). M1 − M0 chỉ mô tả.
- p: bootstrap ghép cặp theo bản ghi. **Cổng quyết định:** p sau Holm < 0,05 **và** |Δ| > chênh lệch lớn nhất giữa 3 seed (với DL) **và** cùng dấu ở mọi seed.
- Tương đương (H1): TOST với biên δ = 0,05 AUROC. Với 40 ca dương, CI dự kiến rộng → "không kết luận được" là kết quả hợp lệ và sẽ được báo cáo như vậy.

## 8. Phân tích phụ (đã định trước)
Cửa sổ 60 phút; không loại bản ghi; M2 có thêm kênh UC (1 seed); nhãn phụ 1 và 2 (chỉ protocol S).

## 9. Nhật ký thay đổi
| Ngày | Thay đổi | Lý do | Trước/sau khi xem kết quả |
|---|---|---|---|
| 2026-11-10 | Tiêu chí chấp nhận T02 "7–9 ca dương mỗi outer-test fold" là lỗi tính của Director (giả định 40 ca dương trước khi loại). Thay bằng: trong mỗi lần lặp, số ca dương y705 giữa các outer-test fold chênh nhau ≤ 1. | Lỗi chỉ thị, không phải thay đổi protocol | Trước khi có kết quả mô hình |
| 2026-11-10 | Quan sát: quy tắc loại QC (mục 4) loại 10/40 ca dương y705 so với 24/512 ca âm. Phân tích chính GIỮ NGUYÊN (518 bản ghi, 30 ca dương). Phân tích độ nhạy "không loại bản ghi" (mục 8) được nâng lên báo cáo song song, cùng mức nổi bật, cho M0–M3 dưới protocol S; dùng split riêng `split_noexcl.csv` (552 bản ghi, cùng thuật toán, phân tầng y705). Tỷ lệ bị loại theo nhãn được báo cáo như một phát hiện mô tả về protocol. | Thông tin về nhãn sau khi loại cho thấy quy tắc loại có liên quan đến kết cục; giữ phân tích chính để không đổi protocol theo dữ liệu | Sau khi thấy số ca dương sau loại; trước mọi kết quả mô hình |
| 2026-11-10 | Quy tắc ngân sách CPU cho DL (T04): pilot đo thời gian 1 lần train M2 và M3; ước tính tổng thời gian kế hoạch đầy đủ (5 config × 10 lần lặp × 5 fold × 3 seed × M2, M3 + M2-UC). Nếu ≤ 48 h: chạy đầy đủ. Nếu > 48 h: A_primary và B_noexcl đủ 10 lần lặp; C, D, E chỉ lần lặp 0–2. Nếu riêng A + B > 48 h: dừng, Director quyết định. Chi tiết kiến trúc/huấn luyện DL ghi trong chỉ thị T04. | Cần chốt trước để không chọn ngân sách theo kết quả | Trước mọi kết quả DL |
| 2026-11-10 | Thêm phân tích EXPLORATORY (T06): AUROC của riêng fhr_nan_pct_w30 cho y705 trên 552 bản ghi; dùng để giải thích chênh lệch B_noexcl vs A_primary. Không dùng để thay đổi phân tích chính. | Giả thuyết shortcut do mất tín hiệu | SAU khi xem kết quả T03 (exploratory) |
| 2026-11-11 | So sánh H1 dùng xác suất trung bình qua lần lặp (và qua 3 seed với DL); cổng quyết định vẫn kiểm từng seed (Δ cùng dấu ở mọi seed; |Δ| > chênh lệch lớn nhất giữa seed). p hai phía từ bootstrap ghép cặp theo bản ghi. | Làm rõ cách tính, không đổi giả thuyết | Sau T04, trước kiểm định |
| 2026-11-11 | Phân tích EXPLORATORY E1–E4 (T06a): E1 AUROC của riêng fhr_nan_pct_w30 (A, B); E2 AUROC M1 và M3 trong hai nhóm theo median fhr_nan_pct_w30 (A); E3 M1 + fhr_nan_pct_w30 làm đặc trưng thứ 12 (A); E4 tương quan Spearman giữa xác suất dự đoán và fhr_nan_pct_w30. | Kiểm giả thuyết DL học shortcut từ mức mất tín hiệu | SAU khi xem kết quả T04 (exploratory) |
| 2026-11-11 | T05 (H2), ngân sách CPU: mọi protocol L chạy lần lặp 0–2, M2 với 3 seed; so sánh với S trên cùng lần lặp 0–2. L1 có thêm đối chứng S_seg (cùng dữ liệu đoạn, chia theo bản ghi bằng split.csv) để chỉ khác đúng một yếu tố (đơn vị chia). L3 cho DL = chọn epoch/checkpoint theo AUROC của outer-test. L4 tính trên OOF có sẵn: ngưỡng Youden chọn trên chính fold test so với ngưỡng chọn trên 4 fold còn lại cùng lần lặp. Bootstrap cụm theo bản ghi cho chỉ số theo đoạn. | Chốt chi tiết trước khi chạy L | Trước mọi kết quả T05 |
| 2026-11-12 | U1 — Tổng quan có cấu trúc (scoping review) các nghiên cứu ML/DL dùng CTU-UHB cho kết cục nhị phân: tìm Google Scholar, PubMed, IEEE Xplore, arXiv/medRxiv (2014–2026) với "CTU-UHB" OR "CTU-CHB" OR "intrapartum cardiotocography database"; tiêu chí nhận: dùng CTU-UHB, kết cục từ pH/BDecf/Apgar, có báo metric phân biệt. Mã hóa: đơn vị chia (record/segment/image/unclear), oversampling/augmentation (before/after split/none/unclear), chọn tham số/epoch (nested/validation/test/unclear), loại bản ghi, ngưỡng pH, metric + giá trị, câu trích nguyên văn làm căn cứ. Phân tích mô tả: tỷ lệ mỗi nhóm protocol; AUROC/độ chính xác công bố (median [IQR]) theo nhóm; đặt cạnh mức phồng đo ở T05. Không kiểm định. Director kiểm chứng DOI 100% và trích dẫn ngẫu nhiên ≥ 20%. | Trả lời phản biện 'protocol bù nhìn' | Trước khi trích xuất |
| 2026-11-12 | U2 — Kiểm định bộ trích đặc trưng (ctg_features.py) trên FHRMA (nhãn chuyên gia): baseline MAE (bpm), F1 phát hiện nhịp tăng và nhịp giảm (khớp khi chồng thời gian ≥ 50%). Ngưỡng chấp nhận định trước: MAE ≤ 5 bpm, F1 ≥ 0.60 cho cả hai. Nếu không đạt: chỉnh tham số CHỈ trên tập train của FHRMA, chốt, báo trên tập test FHRMA, rồi chạy lại M0/M1 với đặc trưng v2 (A_primary, protocol S; L1–L3 cho M1) như PHÂN TÍCH ĐỘ NHẠY. Kết quả gốc T03/T05 giữ nguyên là kết quả chính. | Trả lời phản biện 'bộ đặc trưng yếu' | Trước khi chạy |
| 2026-11-12 | U3 — Thiên lệch do loại bản ghi theo QC được báo cáo như đường thiên lệch thứ 5: tỷ lệ bị loại theo nhãn + Fisher exact (mô tả, ngoài họ Holm), cùng ΔAUROC A_primary vs B_noexcl đã có (T03/T04/T06a). | Đóng góp đặc thù CTG | Sau T03–T06a (đã quan sát; ghi rõ là post hoc) |
| 2026-11-12 | U4 — Tái lập gần đúng [ADAPT] protocol kiểu DeepFHR (BMC MIDM 2019): ảnh scalogram CWT (Morlet) 64×64 từ FHR, ~6 ảnh/bản ghi (các đoạn 10 phút không chồng trong 60 phút cuối giai đoạn 1), CNN 2D nhỏ; so sánh L_img (chia ngẫu nhiên theo ảnh) vs S_img (chia theo bản ghi bằng split.csv), cùng dữ liệu ảnh, lần lặp 0–2, 3 seed, nhãn y705 và y715 (y715 = nhãn gốc của DeepFHR). Báo AUROC theo ảnh và theo bản ghi; Δ với bootstrap cụm theo bản ghi; mô tả, ngoài họ Holm H2. Không tuyên bố bài gốc bị rò rỉ. | Bằng chứng trực tiếp cho U1 | Trước khi chạy |
| 2026-11-12 | U4: T08 (9dc58ea2) lệch thiết kế — seed bị ghép với lần lặp (chỉ 3/9 tổ hợp) — và thất bại kỹ thuật: mô hình gần như không học (AUROC validation 0.50–0.70; nhiều fold cho xác suất hằng số). Kết quả T08 GIỮ NGUYÊN và báo cáo như lần chạy không đạt độ nhạy phép thử. T08b sửa ÁP DỤNG GIỐNG NHAU cho S_img và L_img: (i) chạy đủ 3 lần lặp × 3 seed; (ii) trừ trung bình FHR của đoạn trước CWT; bỏ chuẩn hóa min–max theo ảnh, thay bằng log1p(|W|) chuẩn hóa z theo trung bình/độ lệch chuẩn tính trên ảnh TRAIN của từng fold; (iii) gradient clipping chuẩn 1.0. Thêm kiểm tra độ nhạy phép thử: AUROC trên tập train tại epoch tốt nhất (S_img, y705) phải > 0.75 ở ≥ 80% đơn vị, và ≤ 10% đơn vị có xác suất gần hằng (SD < 1e-3); nếu không đạt thì U4 được báo cáo là 'không kết luận được' và không sửa thêm. | Sửa do thất bại kỹ thuật, không do chênh lệch S − L | SAU khi xem T08; trước T08b |
| 2026-11-13 | U2: nhãn chuyên gia tập test FHRMA không công khai (T07b: 0/90 có nhãn). Quyết định dựa trên tập train FHRMA (66): bộ trích hiện tại KHÔNG đạt (MAE 6.99 bpm; F1 nhịp tăng 0.23; F1 nhịp giảm 0.34). Thay vì tự hiệu chỉnh, dùng phương pháp đã công bố WMFB (Boudet et al., Comput Biol Med 2019; aamwmfb.m trong FHRMA toolbox), được chọn vì README toolbox ghi là phương pháp tốt nhất hiện tại — chọn TRƯỚC khi áp lên CTU-UHB. Báo WMFB trên train FHRMA bằng cùng metric của T07b và bằng evaluateaam chính thức (lưu ý: có thể lạc quan vì WMFB phát triển trên tập này). Cổng tiến tới T07d: WMFB đạt MAE ≤ 5 bpm và F1 ≥ 0.60 (tăng và giảm) trên train FHRMA theo metric T07b. T07d: đặc trưng v2 = baseline, n_accel, n_decel, decel_area, n_prolonged_decel lấy từ WMFB; các đặc trưng khác giữ nguyên; chạy lại M0/M1 (A_primary, S, 10 lần lặp) và M1 với L1–L3 (lần lặp 0–2) như PHÂN TÍCH ĐỘ NHẠY. Kết quả T03/T05 gốc vẫn là kết quả chính và được báo cáo kèm hạn chế về bộ trích. | Bộ trích tự viết không đạt; thay bằng phương pháp đã kiểm định | Sau T07b; trước T07c/T07d |
