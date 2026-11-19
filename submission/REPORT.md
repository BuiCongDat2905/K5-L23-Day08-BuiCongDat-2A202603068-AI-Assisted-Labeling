# Báo cáo Day 08 — 2A202603068 - Bùi Công Đạt

> Mọi con số kèm **mẫu số** và nguồn (file nào). Máy: Windows 11, **CPU only** (torch 2.15.0+cpu, không CUDA), CVAT 2.81.0 bằng Docker.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`) | 0.3 (mặc định) | `m01` bắt tốt 6 lớp COCO ở gần máy, chạy 1.14 s/ảnh trên CPU. Quét ngưỡng (`qc/threshold_sweep.md`) **không** tìm được ngưỡng ≤ 0.5 nào đạt recall 90% cho person (tối đa 86%, 37 vật), motorbike (tối đa 88%, 16 vật), car (tối đa 88%, 8 vật) nên để mặc định; bus (1 vật) và truck (0 vật) dưới 5 vật nên **không được chọn ngưỡng riêng** |
| three_wheeler | YOLOE-26s (`yoloe-26s-seg.pt`), prompt lấy từ `taxonomy.json` | **0.15** | `m01`/`m03`: YOLO26 bỏ hẳn tuk-tuk ở img_002 và gọi tuk-tuk ở img_004 là `car` 0.37/0.34/0.30, nên lớp này phải lấy từ open-vocabulary. Sweep trên 4 ảnh gold: 0.15 cho recall 100%, precision 78% (7 vật) |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **So sánh prompt (YOLOE, `m03` vs `m07`)**: cùng model, cùng ngưỡng 0.25, cùng 8 ảnh pilot; bộ 3 từ khoá trong `taxonomy.json` (`tuk-tuk`, `cyclo`, `cargo tricycle`) cho **8 box**, chỉ dùng `tuk-tuk` cho **5 box**. Mất 2 xích lô ở img_032 (0.48 và 0.27) và 1 xe ba gác ở img_002 — YOLOE khớp theo **chữ**, nên từ khoá hẹp làm recall của lớp rơi từ 8 xuống 5 box dù không đổi gì khác. Kết luận: giữ bộ 3 prompt, không rút gọn.
- **Phiên bản SAM đã chọn: SAM 2.1-b (`sam2.1_b.pt`)**, vì mask bám biên sát hơn `mobile_sam.pt` (SAM 1) trên cùng box, và trên CPU chỉ 7.21 s/ảnh (`m04`) — chấp nhận được cho 3 vật khó ở Part 5. Không dùng SAM 3 vì phải xin quyền truy cập, không cần cho 3 mask.
- **Lỗi không sửa được bằng đổi ngưỡng:** tuk-tuk bị gọi `car`/`truck` — đổi ngưỡng chỉ làm box đó mất đi hoặc yếu đi, không đổi được **tên lớp**, vì đó là giới hạn của detector closed-set 80 lớp COCO (`m01`, `m02`: hạ ngưỡng 0.25 → 0.5 vẫn 0 box cho tuk-tuk xanh img_002). Cách sửa: thêm YOLOE open-vocabulary cho riêng `three_wheeler`, đặt file YOLOE ưu tiên cao hơn khi gộp.
- **Box điểm cao mà sai:** img_035 box `person` 0.93 và box 0.36 gần trùng khít trên **cùng một người** (một người hai box). **Box điểm thấp mà đúng:** img_010 hàng xích lô, box `bicycle` 0.50 và `motorbike` 0.62 nằm trên **khung và bánh sau của chính chiếc xích lô** — điểm thấp nhưng đúng là "có vật", chỉ sai lớp.
- **Giới hạn:** ngưỡng được chọn **trên chính 4 ảnh gold** và chấm cũng trên 4 ảnh đó (in-sample) nên precision/recall của máy ở mục 2 là con số **lạc quan**; bus/truck có 0–1 vật nên kết luận theo lớp rất yếu.

## 2. AI error profile (từ `qc/review_stats.md`)

Ảnh tính: 24 batch, **519 box máy**, **473 đối tượng sau duyệt**. Đã loại 21 đối tượng quanh bẫy khỏi bảng này (lỗi nhân tạo không phản ánh chất lượng máy).

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 456 | 456/519 = 87.9% | 519 box máy |
| edited | 8 | 8/519 = 1.5% | 519 box máy |
| deleted | 50 | 50/519 = 9.6% | 519 box máy |
| added | 9 | 9/473 = 1.9% | 473 đối tượng sau duyệt |

- Trong deleted + added có **3 ca là đổi lớp** (máy đúng chỗ, sai tên lớp) → số vật máy **sót thật** ≈ 9 − 3 = **6**.
- **Ba mẫu số khác nhau, đừng lẫn:** **519** box máy (đã trừ vùng quanh bẫy) và **473** đối tượng sau duyệt là số của `review_stats`; `labels/batch_reviewed.json` có **482** box (gồm cả box nằm trong vùng bẫy); `release/final_det.coco.json` có **483** box (482 − 74 box của 4 ảnh sửa QC + 75 box của bản sửa).
- **Lớp máy tệ nhất là `three_wheeler`**: accepted 17, edited 6, deleted **19**, added 5. Lý do (dẫn ảnh): img_004/017 tuk-tuk bị gọi `car`; img_010 một xích lô bị tách 2 box `three_wheeler` rồi phần khung thành `bicycle`/`motorbike`; img_035 hai box trên cùng một xích lô. Model closed-set không có lớp này, YOLOE thì box lỏng và có false positive 0.37 trên xe bán tải trắng (img_002).
- `person` (254 accepted / 9 deleted) và `motorbike` (126/10) là hai lớp ổn định nhất, lỗi chủ yếu là box thừa trên bóng đổ và vệt đèn ảnh đêm img_024.
- **So với gold: precision 74.3% (TP 55 / 74 box máy dương tính), recall 72.4% (TP 55 / 76 đối tượng gold)**; KTC 95% lần lượt 63–83% và 61–81% (nguồn `qc/review_stats.md` mục 2). Nghĩa là **máy bỏ sót khoảng 1/4 vật trong gold**.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Bắt bẫy 4/12 = 33.3%** (KTC 95% 14–61%; ngưỡng đạt là 80%), chi tiết: `extra` 3/3, `missing` 1/3, `wrong_class` 0/3, `loose` 0/3. Lọt: img_003 (loose), img_021 (wrong_class), img_022 (loose), img_006 (loose), img_018 (wrong_class), img_034 (missing), img_033 (wrong_class), img_025 (missing).
- **Loại lọt nhiều nhất là `loose` và `wrong_class` (0/3 mỗi loại).** Nguyên nhân theo checklist 4 bước: tôi (và agent duyệt) **bỏ bước 0 (quét ảnh trước khi nhìn nhãn máy)** — mắt chỉ tìm vật trong box có sẵn, nên box lệch vài chục pixel hoặc bị đổi lớp vẫn "trông hợp lý"; box lỏng ≤ 0.68 IoU không đủ rõ để nhận ra khi không quét ảnh từ đầu.
- `missing` chỉ 1/3: bẫy xoá một box đúng, và chỉ khi vật nằm trong vùng đông thì mới phát hiện được; `extra` 3/3 dễ nhất vì box "ma" nằm trên bóng/cột/mái hiên — đúng loại lỗi mà mắt bắt nhanh.
- **Accepted 87.9% nhưng bắt bẫy chỉ 33.3%**: hai số đo hai thứ khác nhau. `accepted` cao chứng minh **box máy đặt đúng chỗ** (hợp đồng IoU ≥ 0.9), **không** chứng minh người duyệt nhìn kỹ. Ngược lại `added` 1.9% thấp **không** có nghĩa máy ít sót: recall của máy trên gold chỉ 72.4%, và bẫy `missing` chỉ bắt 1/3 — chỗ máy không vẽ thì người duyệt cũng không kiểm (lỗi bỏ sót là lỗi khó thấy nhất, xem `docs/REVIEW_CHECKLIST.md` mục 1).

## 4. Máy có đáng dùng không?

- Thời gian (`time_log.csv`): gán tay **42 phút / 4 ảnh = 10.5 phút/ảnh**; duyệt có pre-label **18 phút / 24 ảnh = 0.75 phút/ảnh**; điền phiếu QC **6 phút / 10 ảnh** (2 vòng). So với gán tay, duyệt nhanh hơn **~14 lần**. **Nhưng đây không phải so sánh cùng loại:** lần duyệt này do **AI làm** (xem mục 8), nên 0.75 phút/ảnh là tốc độ của agent, không phải tốc độ người. Người duyệt thật với checklist 4 bước và ~3 phút/ảnh như `docs/REVIEW_CHECKLIST.md` thì mức tiết kiệm thật chỉ còn khoảng **10.5 → 3 phút/ảnh, tức ~70%**. Thiên lệch khác: 4 ảnh gold được gán **trước** nên đã quen cảnh, thời gian gold có thể **thấp hơn** người mới, làm mức tiết kiệm bị **thổi phồng**.
- Kết luận: **có, nhưng chỉ cho `person` / `motorbike` / `car`** (accepted 254/264, 126/137, 46/54 = 96% / 92% / 85%). **Không** dùng pre-label cho `three_wheeler` (accepted 17/42 = 40%, deleted 19): lớp này vẽ tay rẻ hơn sửa. `bus`/`truck` mẫu quá ít (3 và 5 box) nên chưa kết luận.
- **Ba quy tắc giảm thiên kiến** (kiểm tra được bằng số hoặc việc làm):
  1. **Quét ảnh 15 giây trước khi hiện nhãn máy ở 100% ảnh**, đếm nhẩm theo lớp rồi mới bật nhãn. Kiểm bằng: bẫy `missing` phải ≥ 2/3 ở lần duyệt sau (hiện 1/3) và số `added` phải tăng lên.
  2. **Với mọi box `three_wheeler` và mọi box ở ảnh đêm, soi lại 2 lần**: lần 1 lớp, lần 2 biên. Kiểm bằng: bẫy `wrong_class` từ 0/3 lên ≥ 2/3, và `deleted` của `three_wheeler` không còn cao gấp 4 lần `person`.
  3. **Chạy kiểm trùng trước export** (hai box cùng lớp, IoU > 0.8 → gộp). Kiểm bằng: số cặp trùng trong `qc_review.csv` = 0 ở vòng sau (hiện còn 1 ca: img_035 hai box `person` trên một người).

## 5. Mask SAM cho 3 vật (Part 5)

SAM 2.1-b (`sam2.1_b.pt`), prompt là click, mô phỏng Interactor của CVAT.

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 — tuk-tuk xanh | SAM 2.1-b | 2 dương + 2 âm | **Có**: mask tràn sang chân và tay người lái ngồi bên phải | Mask 316507 px = 45.3% ảnh, 1 polygon: phủ cả mui, thân, nhưng **dính người lái** vì người và vỏ xe chồng lên nhau | `tuktuk_img013.jpg` |
| img_003 — xe ba gác chở chuối | SAM 2.1-b | 2 dương + 2 âm | **Có**: mask gồm cả mâm chuối, bao hàng **và** hai người ngồi trong xe + người đi đường bên phải | 123230 px = 17.6% ảnh, 1 polygon: mask "ăn" sang người vì click dương đặt trên vùng có người | `xe3banh_img003.jpg` |
| img_010 — xích lô trong hàng | SAM 2.1-b | 3 dương + 2 âm | **Có**: mask phủ áo khoác người đạp | 105926 px = 15.2% ảnh, **4 polygon**: vỡ thành mảnh rời (mui, khung, 2 bánh) và dính áo người đạp | `xichlo_img010.jpg` |

- Đối chiếu guideline **mục 4.2**: quy tắc 4.2.1 (bám biên phần nhìn thấy) **giúp** — mask đúng ở các mảng vỏ xe. Quy tắc 4.2.4 (hai vật chạm nhau, mỗi vật một polygon, không chồng) **thiếu hướng dẫn cho trường hợp người ngồi *trong* xe**: guideline chỉ nói về hai vật **cạnh nhau**, không nói mask xe có được phủ lên người ngồi trong hay không. Đây là chỗ tôi phải tự quyết: **mask xe không gồm người**, vì người là box `person` riêng (4.1.4) và lỗ hổng 4.2.2 chỉ nói về lỗ thủng, không nói về người. Kết quả là cả 3 mask **đều sai theo quy tắc này** và phải sửa tay — SAM 2.1-b không tách được người khỏi xe khi cả hai cùng nằm trong một click.
- **Quy tắc rút ra:** "một click, ba mask" không có nghĩa mask đúng. **Mask đúng không cứu được nhãn lớp sai** — box `car` 0.34 trên tuk-tuk xanh img_004 cho mask bám biên rất đẹp, nhưng vẫn là mask của "ô tô" theo nhãn sai; ngược lại box đúng mà click sai thì mask dính vật bên cạnh (img_003). Vì vậy Part 5 phải làm **sau** khi sửa lớp, không làm trước.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: **chưa thử** — CVAT 2.81.0 chạy Docker trên máy CPU only, Interactor cần model server riêng; tôi dùng `sam2_points.py` offline với cùng cơ chế click âm/dương nên kết quả tương đương về mặt quan sát, nhưng **không** so sánh được tốc độ tương tác.

## 6. Calibration (Part 6)

- Số quyết định: **9** (`qc/calibration_decisions.csv`): 3 `loi_may`, 4 `cau_hoi_mo`, 1 `can_review`, 1 `bay_lot`. Quan trọng nhất là **CAL-01** (tuk-tuk bị gọi `car`/`truck`) và **CAL-05** (bẫy `wrong_class` lọt 0/3 vì không có phép thử số bánh). Ca `can_review` duy nhất của gold là **img_020**: box `bus` ở xa, không phân biệt được bus với truck — dùng làm ví dụ cho CAL-05 và CAL-09.
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: mục 1 chốt 4/7 câu hỏi mở (1, 3, 5, 7) và chốt một phần câu 2, 4; mục 2 thêm **ưu tiên model theo lớp**, **một vật một box, cấm tách box bộ phận**, **kiểm trùng IoU > 0.8**, **phép thử số bánh** cho `three_wheeler`, ghi chú ảnh phơi sáng dài. Câu 6 (box chồng nhau ảnh đêm) **để mở** và nói rõ lý do.
- Nguyên nhân gốc: **cả ba**. CAL-01/02/06 là **máy sai** (không cần đổi guideline, chỉ cần đổi recipe và thêm bước kiểm); CAL-03/04 là **guideline mơ hồ** (v1.0 cố ý để mở); CAL-05 là **tôi/agent duyệt sai** (guideline 4.3 đã có "chọn lớp tin nhất", nhưng thiếu phép thử cụ thể nên vẫn lọt) — đây là ca duy nhất phải sửa **cả** guideline **và** quy trình duyệt.

## 7. QC và quyết định (Part 6)

- **Mẫu:** 2 vòng, mỗi vòng 5 ảnh / 24 ảnh batch, lấy bằng `qc_sample.py` (0 ảnh `can_review` + 5 ảnh ngẫu nhiên, seed 42 rồi seed 43 sau khi sửa). Người kiểm: **agent AI + tôi kiểm lại danh sách lỗi** (**hạn chế lớn**: người duyệt và người kiểm là cùng một hệ, không độc lập; xem mục 8).
- **Vòng 1** (99 đối tượng): lỗi theo **đối tượng 4/99 = 4.0%**, theo **ảnh 4/5 = 80%** (ngưỡng Release ≤ 2%).
- **Vòng 2 sau khi sửa** (`labels/qc_fixed.json`, 4 ảnh được thay): **3/86 = 3.5%** theo đối tượng, **3/5 = 60%** theo ảnh. Cải thiện theo đối tượng nhưng **vẫn trên ngưỡng**.
- Top lỗi vòng 2: `sai_vi_tri_kich_thuoc` 1 (img_008 — box `three_wheeler` trùm sang xe bên cạnh), `thua` 1 (img_024 — box `car` trên vệt đèn pha), `sai_lop` 1 (img_034 — xe van trắng bị gọi `three_wheeler`).
- **Giới hạn mẫu nhỏ:** với 3 ảnh lỗi trong 5 ảnh, cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là **88%** (`qc/qc_report_auto.md`); với 0 lỗi thì cận trên vẫn ~3/n. Mẫu 5 ảnh **không thể** chứng minh "dưới 2%".
- Script kiểm được: tên file, lớp trong taxonomy, box nằm trong ảnh, box < 20 px, đủ/thừa ảnh, accepted/edited/deleted/added, bắt bẫy. **Không** kiểm được: box có ôm sát vật thật không, lớp có đúng thực tế không, vật mà cả máy lẫn người duyệt cùng bỏ sót, và **việc duyệt có theo checklist thật không** — cả 4 lỗi QC ở trên đều thuộc nhóm này và chỉ tìm được bằng mắt.

**Quyết định: Rework**, ngày 2026-11-19. Lý do theo cây Hold → Rework → Release: không có lỗi dữ liệu/ảnh hỏng (không Hold); nhưng **ảnh có lỗi 3/5 = 60% > 2%**, **lỗi theo đối tượng 3/86 = 3.5% > 2%** và **bắt bẫy 4/12 = 33% < 80%** `qc/qc_report_auto.md` → phải Rework, chưa được Release.
Nếu Rework: (1) sửa 3 lỗi vòng 2 (img_008 siết box, img_024 xoá box trên vệt đèn, img_034 đổi `three_wheeler` → `car`) — **3 lỗi này chưa được sửa**, `labels/qc_fixed.json` hiện chỉ là bản sửa **vòng 1** (img_003, img_014, img_031, img_035) nên release candidate vẫn còn chúng; (2) **duyệt lại 24 ảnh bằng người thật** theo checklist 4 bước — đây là việc quyết định, vì bắt bẫy 33% nghĩa là lô nhãn hiện tại chưa đáng tin; (3) mở rộng mẫu QC lên ≥ 15 ảnh để bắt đầu nói được về ngưỡng 2%; (4) chốt câu hỏi mở số 6 của v1.1.

- Bằng chứng 2 vòng: vòng 1 trong `qc/qc_review_round1.csv` + `qc/qc_report_round1_auto.md` (99 đối tượng, seed 42); vòng 2 là `qc/qc_sample.csv` + `qc/qc_review.csv` + `qc/qc_report_auto.md` (86 đối tượng, seed 43).
- Release hiện tại: `release/release_manifest.json` = **24 ảnh / 483 đối tượng** (`person` 262, `motorbike` 130, `car` 46, `three_wheeler` 28, `bicycle` 7, `truck` 5, `bus` 4, `ignore_region` 1), khớp `final_det.coco.json` (483) và `release_summary.csv`. Đây là **release candidate**, không phải bản Release: quyết định là Rework nên còn 3 lỗi vòng 2.
- Còn lại trong nhãn cuối: **3 box cao < 20 px** (img_027 #421 `car` 17.2 px, img_027 #439 `motorbike` 16.9 px, img_033 #524 `motorbike` 18.6 px) — `validate_labels.py` cảnh báo là vượt guideline 4.1.6; chưa xoá vì làm thay đổi mọi số đo của báo cáo, ghi lại ở đây như việc của vòng Rework.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** có. (a) **DeepSeek agent trong IDE** viết các script phụ `work/` (ảnh duyệt có lưới, gộp sửa đổi, tạo `qc_fixed.json`) và soạn/điền các file văn bản của bài; (b) **3 agent AI** (cùng model) duyệt 24 ảnh batch ở Part 4, tôi kiểm lại danh sách sửa; (c) gold **không** dùng AI — tôi gán tay trong CVAT trước khi chạy model. Cách kiểm: mọi số đo lấy từ script của lab (`review_stats.py`, `qc_report.py`, `threshold_sweep.py`) chứ không tự tính; mọi sửa đổi nhãn nằm trong file JSON trong `work/review_out/` được gộp bằng script, nên xem lại được từng box đã xoá/sửa.
- **Gold có bị nhiễm không:** **không.** 4 ảnh gold gán tay trong CVAT trước khi chạy bất kỳ model nào lên chúng; 4 ảnh này không nằm trong task review và không được nạp pre-label (đúng quy trình `prepare_workdirs.py`).
- **Máy / GPU / CVAT:** Windows 11 Home, CPU **Intel Core i5-12450HX**, **CPU only** (torch 2.15.0+cpu, `torch.cuda.is_available()` = False), 15.7 GB RAM (~16 GB); mọi lần chạy model ở `model_log.csv` đều `device=cpu` (YOLO26m 0.64–1.14 s/ảnh, SAM 2.1-b 7.21 s/ảnh). CVAT 2.81.0 chạy bằng Docker 29.8.1. Vướng mắc cài đặt: pipeline cài đặt bị chặn quyền ghi trên Windows (sandbox hạ tiến trình xuống Low integrity trong khi thư mục con để mức Medium) — phải cấp lại quyền và nhãn tin cậy cho cây thư mục rồi mới chạy được `prelabel.py`; không ảnh hưởng nội dung bài.
- **Có dùng cache `data/cache/` thay vì tự chạy?** **Không.** Cả 6 lần chạy trong `model_log.csv` đều chạy thật trên máy (đối chiếu `work/runs.csv`: 36 ảnh cho `y26_all` và `yoloe_all`), cache trong `data/cache/` chỉ dùng để đối chiếu số box.
- **Lỗi / bug script đã gặp:** (1) `prelabel.py` bỏ im lặng các lớp ngoài taxonomy (banana, umbrella, chair…) — dòng cảnh báo in ra cuối màn hình dễ bỏ qua, phải đọc; (2) `qc_sample.py --force` ghi đè `qc_review.csv` nên vòng 1 phải copy ra `work/` trước; (3) `review_stats.py` báo `deleted` 50 nhưng trong đó có 3 ca đổi lớp, đọc số "đổi lớp" riêng mới đúng bản chất lỗi.
- **Một điều làm khác lần sau:** duyệt **bằng tay 24 ảnh** thay vì để AI duyệt. Bài này đo được đúng chỗ yếu: bắt bẫy 33% (extra 3/3 nhưng loose 0/3, wrong_class 0/3) — AI bắt tốt lỗi "thô" (box ma) nhưng kém lỗi "tinh" (lệch biên, đổi lớp), đúng như dự đoán của `docs/REVIEW_CHECKLIST.md` mục 1 về thiên kiến tự động hóa. Lần sau: quét ảnh trước, và với lớp `three_wheeler` thì vẽ tay từ đầu.
