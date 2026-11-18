# Day 08 Lab — Tài liệu bối cảnh đầy đủ (context pack) + chạy offline

> **File này viết cho cả người và chatbot.** Đọc hết file là đủ để trả lời câu hỏi về bài lab **và** chạy bài khi máy không có Internet.
> Snapshot: 2026-11-18, repo `D:\AI20KK\K5-L23-Day08-BuiCongDat-2A202603068-AI-Assisted-Labeling` (fork bài nộp của học viên).
> Khi tài liệu này khác file gốc trong repo: **file gốc thắng** — ưu tiên [README.md](../README.md), [GUIDE.md](../GUIDE.md), [RULES.md](../RULES.md), [RUBRIC.md](../RUBRIC.md), [SUBMISSION.md](../SUBMISSION.md), [submission/taxonomy.json](../submission/taxonomy.json), [guideline/guideline_v1.0_label_contract.md](../guideline/guideline_v1.0_label_contract.md) và code trong `scripts/`.

## 0. Cách dùng file này (dành cho chatbot)

- Trả lời **chỉ bằng dữ kiện trong file này** (hoặc trong repo). Không bịa số, không đoán tên file.
- Mọi con số về chất lượng (**accepted, bắt bẫy, precision/recall, tỉ lệ lỗi**) phải kèm **mẫu số** và **nguồn file**. Đây là quy tắc của chính bài lab, không phải khẩu vị.
- Câu hỏi **nên hỏi mục nào**:

| Câu hỏi | Xem mục |
|---|---|
| Bài này làm gì, mấy Part, bao lâu | 1, 5 |
| Ảnh nào gold/pilot/batch, lớp nào, quy tắc gán | 2 |
| Script nào làm gì, cờ nào | 3 |
| Nộp gì, chấm bao nhiêu điểm | 4 |
| Lệnh chạy từng Part | 5 |
| Đọc số `review_stats`, `threshold_sweep`, QC, Release/Hold/Rework | 6 |
| Luật chơi, trung thực, giới hạn thống kê | 7 |
| Cài môi trường khi có mạng / CVAT | 8 |
| Chạy khi **không có Internet** | 9, 10 |
| Lỗi hay gặp | 11 |
| Hỏi nhanh kiểu "12 bẫy là gì", "ngưỡng accepted" | 12 |
| Giấy phép, giới hạn tài liệu, trạng thái repo | 13 |

## 1. Bài lab trong một trang

**Tên:** Day 08 — AI hỗ trợ gán nhãn: Pre-label, Review, Calibration (VinUni AI20k, lớp K5-L23).
**Hình thức:** bài **cá nhân**, mỗi người một repo public, một bộ nhãn, một báo cáo, commit của chính mình.
**Thời lượng:** 4 giờ trên lớp (240 phút) + ~45 phút cài đặt **làm trước** buổi học (không tính vào 4 giờ).
**Máy yếu / không GPU:** chạy model trên Google Colab, phần CVAT + script đo vẫn ở máy bạn (xem mục 8.6).
**Một câu triết lý:** *“Máy gán sẵn, người sửa.”* Nhanh hơn vẽ tay, nhưng **học viên là chốt chặn chất lượng** và phải **đo** được điều đó.

**Bốn kỹ năng bài dạy:**

1. Chọn đúng họ model: **SAM** (click → mask), **YOLOE** (gõ chữ → box/mask, open-vocabulary), **YOLO26** (detector closed-set 80 lớp COCO). Nâng cao: Grounding DINO, OWLv2, SAM 3.
2. Đưa nhãn máy vào CVAT bằng **offline import** (đường bắt buộc); dùng AI trực tiếp trong CVAT (AI Tools, Automatic annotation, `cvat-cli`) là **nâng cao**.
3. Duyệt bằng **checklist bốn bước**; tự đo **thiên kiến tự động hóa** bằng **12 ảnh bẫy**.
4. Đo **AI error profile** (accepted / edited / deleted / added, precision–recall của máy so với gold) và **calibration**: ngưỡng theo lớp, guideline v1.1, QC với quyết định Release / Hold / Rework.

**Lộ trình 7 Part (thời gian trên lớp):**

| Part | Việc | Thời gian |
|---|---|---|
| 0 | Cài đặt + tải model + Project CVAT (làm trước lớp) | ~45' |
| 1 | Gán tay 4 ảnh **gold**, **không AI** | 25' |
| 2 | Tour 3 model: YOLO26, YOLOE, **SAM** | 30' |
| 3 | Chọn ngưỡng theo lớp | 10' |
| — | Nghỉ | 10' |
| 4 | Pre-label 24 ảnh + 12 bẫy + review | 95' |
| 5 | Mask SAM cho 3 vật | 15' |
| 6 | Calibration + QC + báo cáo | 45' |
| — | Nộp | 10' |
| | **Tổng** | **240'** |

**Checkpoint tự kiểm (CP0–CP6):**

| Mốc | Mục tiêu | Tự kiểm |
|---|---|---|
| trước lớp | CP0 cài đặt | `check_env.py` ra **SẴN SÀNG** (hoặc Colab chạy được); Project CVAT có 8 nhãn; đã đọc label contract |
| 0:25 | CP1 gold | `gold_manual.json` qua `validate_labels` (đủ 4 ảnh); chưa xem pre-label của gold; có `time_log` |
| 0:55 | CP2 tour model | `model_log.csv` ≥ 4 dòng có nhận xét (YOLO26, YOLOE, **SAM**); đã chạy `--split all --conf 0.05` |
| 1:05 | CP3 ngưỡng | `threshold_sweep.md` có ngưỡng theo lớp |
| 2:50 | CP4 review | `batch_reviewed.json` (24 ảnh); `review_stats.md` có % bắt bẫy |
| 3:05 | CP5 mask | ≥ 3 ảnh trong `submission/masks/` |
| 3:50 | CP6 calibration + QC | `calibration_decisions.csv` ≥ 3 dòng, `guideline_v1.1`, `REPORT.md` có quyết định |
| 4:00 | nộp | `check_submission.py` sạch; đã nộp link |

## 2. Dữ liệu và label contract

### 2.1 Bộ ảnh

- 36 ảnh `data/images/img_001.jpg … img_036.jpg`: phố / chợ Việt Nam và Thái Lan, ngoài trời, ngày và đêm; cạnh dài tối đa **1024 px**; **chưa có nhãn**.
- Nguồn Wikimedia Commons, giấy phép mở **CC BY / CC BY-SA / CC0**; giữ nguyên `data/ATTRIBUTION.csv`. **Không** sửa / đổi tên / thêm bớt ảnh.
- Khoảng **28%** ảnh có **tuk-tuk, xích lô, ba gác** = lớp `three_wheeler` (ngoài COCO): model COCO gọi chúng là `car` / `truck`.
- Ảnh khó: ban đêm, đông đúc (xe máy dày), mờ do chuyển động, vật bị che, vật ở xa.
- 6 ảnh dày (> ~50 vật): `img_005`, `img_018`, `img_021`, `img_024`, `img_028`, `img_029`.

**Ba tập (`data/split.csv`):**

| Tập | Số ảnh | Ảnh | Vai trò |
|---|---|---|---|
| `gold` | 4 | `img_001`, `img_012`, `img_020`, `img_036` | học viên **gán tay từ đầu**, không AI; chuẩn đo máy |
| `pilot` | 8 | `img_002`, `img_007`, `img_011`, `img_015`, `img_016`, `img_019`, `img_030`, `img_032` | chỉ để **thử model** ở Part 2 (không làm nhãn) |
| `batch` | 24 | 24 ảnh còn lại | lô thật: pre-label → review → QC |
| `seg` (con của batch) | 8 | `img_003`, `img_004`, `img_009`, `img_010`, `img_013`, `img_017`, `img_023`, `img_031` | cột `seg=1`, dùng cho polygon ở bài nâng cao và cho việc chọn vật khó Part 5 |

### 2.2 Taxonomy — `submission/taxonomy.json` (v1.0)

8 lớp, **mỗi vật thuộc đúng một lớp**. `ignore_region` là **vùng**, không phải vật.

| Lớp | Định nghĩa | Gồm | Không gồm (dễ nhầm) |
|---|---|---|---|
| `person` | Một người (phần nhìn thấy) | người đi bộ, ngồi bán hàng, lái / ngồi sau xe | tượng, ma-nơ-canh, người trong tranh / biển / phản chiếu |
| `bicycle` | Xe đạp thường, đạp bằng chân | xe đạp chở hàng, có giỏ | xe đạp điện (câu hỏi mở 1) |
| `motorbike` | Xe máy, tay ga, mô tô 2 bánh | xe đang đỗ, xe chở hàng | **người ngồi trên xe** (người là `person` riêng) |
| `car` | Ô tô con, taxi, SUV, bán tải, minivan | taxi màu, xe công ty | xe ba bánh (→ `three_wheeler`) |
| `bus` | Xe buýt, xe khách lớn | minibus chở khách | xe buýt trên biển quảng cáo |
| `truck` | Xe tải có buồng lái + thùng hàng, ≥ 4 bánh | xe tải nhỏ | **xe ba gác / ba bánh chở hàng** (→ `three_wheeler`) |
| `three_wheeler` | Xe **ba bánh**: tuk-tuk, xích lô (cyclo), xe ba gác | cả xe lẫn mui / thùng hàng gắn liền | xe máy kéo rơ-moóc 4 bánh (hiếm → `can_review`) |
| `ignore_region` | **Vùng** bỏ qua: đám đông / khối xe dày, mờ, không tách được từng vật | khối xe máy đỗ sát nhau ở xa | không dùng để “lờ” vật rõ ràng chỉ vì khó vẽ |

**Thuộc tính (2):** `conf` (text, điểm máy, học viên **không sửa**, chỉ để xếp thứ tự duyệt) · `can_review` (checkbox: tick khi **không chắc quy tắc**, không phải khi vật khó vẽ).

**`coco_map`** (tên COCO → lớp lab): `person→person`, `bicycle→bicycle`, `motorcycle→motorbike`, `car→car`, `bus→bus`, `truck→truck`.
**`prompts`** (open-vocab, mặc định): `three_wheeler ← ["tuk-tuk", "cyclo", "cargo tricycle"]`; các lớp khác prompt = tên lớp.

**`rules` (KHÔNG được sửa — `check_submission.py` so khớp đúng 10 giá trị):**

| Khoá | Giá trị | Nghĩa |
|---|---|---|
| `min_box_height_px` | 20 | box cao < 20 px không vẽ từng vật |
| `match_iou` | 0.5 | IoU tối thiểu để coi là “cùng một vật” |
| `accept_iou` | 0.9 | IoU ≥ 0.9 → box máy được `accepted` |
| `accept_mask_iou` | 0.85 | như trên nhưng cho mask (mode seg) |
| `trap_catch_min` | 0.8 | phải bắt ≥ 80% bẫy (≥ 10/12) |
| `error_rate_img_max` | 0.02 | ngưỡng Release: ảnh có lỗi ≤ 2% |
| `self_iou_min` | 0.85 | gán lại mù: IoU trung bình ≥ 0.85 |
| `self_class_min` | 0.95 | gán lại mù: cùng lớp ≥ 95% |
| `recall_target` | 0.9 | ngưỡng theo lớp phải giữ recall ≥ 0.9 |
| `self_unmatched_max` | 0.1 | gán lại mù: box không ghép được ≤ 10% |

### 2.3 Guideline v1.0 — `guideline/guideline_v1.0_label_contract.md`

Do giảng viên viết, **cố ý chưa trả lời hết** (7 câu hỏi mở) để học viên calibration thành v1.1. **Không sửa file này**; copy sang `submission/guidelines/guideline_v1.1_calibrated.md`.

**Quy tắc hình học — box (4.1):**

1. Box ôm sát **phần nhìn thấy**; sai số ≤ ~2 px ở ảnh 1024 px.
2. Vật bị che: vẽ phần nhìn thấy; bị che > 80% (thấy < 20% diện tích) → **không gán**.
3. Vật bị cắt mép ảnh: vẽ đến mép, không vẽ ra ngoài.
4. **Người ngồi trên xe = hai box riêng** (`person` + `motorbike/bicycle/...`); chồng nhau là bình thường.
5. Hàng chở nhô ra khỏi xe > ~20% chiều dài xe: không tính vào box xe (câu hỏi mở 3).
6. Box cao **< 20 px** → không vẽ từng vật; nếu là cụm ≥ 5 vật sát nhau → **một** `ignore_region` bao cụm.
7. Hình in, tượng, phản chiếu, ảnh trên biển quảng cáo: **không gán**.

**Quy tắc mask / polygon (4.2, Part 5 + nâng cao):** bám biên phần nhìn thấy, lệch ≤ ~3 px; lỗ thủng < 5% diện tích thì **không** vẽ lỗ; bóng đổ không thuộc vật, kính xe thuộc vật (câu hỏi mở 5); hai vật chạm nhau = hai polygon không chồng, vùng chạm thuộc vật phía trước.

**4.3 Chọn lớp khi nhìn không rõ:** chọn lớp tin nhất + tick `can_review`; không chọn ngẫu nhiên. Hai lớp có thể đúng → đếm số bánh nhìn thấy; không đếm được → `can_review`.

**4 ca khó đã chốt (v1.0):**

| Ca | Quyết định |
|---|---|
| Xích lô: khách ngồi trước, người đạp sau | 1 `three_wheeler` (cả xe + mui) + mỗi người 1 `person` |
| Người dắt xe đạp chở đầy hàng | `person` + `bicycle` (box xe ôm cả giỏ gắn cố định) |
| Xe máy đỗ sát nhau, vật ở xa ít thấy | vật rõ → gán từng xe; phần xa < 20 px → `ignore_region` |
| Tuk-tuk có khách nhìn thấy qua cạnh hở | gán `person` nếu còn thấy ≥ 20% người |

**7 câu hỏi CÒN MỞ (phải chốt trong v1.1):** (1) xe đạp điện / xe máy điện nhỏ là `bicycle` hay `motorbike`? (2) khối xe máy đỗ dày đặc **gần** camera: từng xe hay `ignore_region`, ngưỡng cụ thể? (3) hàng nhô ra (tre, rổ) có vào box/mask xe không? (4) ngưỡng “còn thấy ≥ 20%” khi người ngồi trong xe có mui có hợp lý, hai người đánh giá lệch nhau bao nhiêu? (5) kính chắn gió / cửa sổ có thuộc mask xe? (6) box máy chồng nhau nhiều trong ảnh đêm: gộp hay giữ? (7) có đặt mức tin cậy tối thiểu để chấp nhận box máy không cần sửa? Mỗi câu chốt phải có **ảnh ví dụ `img_0xx`** + dòng trong `calibration_decisions.csv` + lịch sử thay đổi.

## 3. Cấu trúc repo và các script

### 3.1 Cây thư mục (rút gọn)

```text
README.md                     # hướng dẫn + bảng tài liệu; phải đổi dòng tiêu đề "# Day 08 Lab — <MSHV> - <Họ tên>"
GUIDE.md                      # lệnh từng Part (nguồn chính khi làm bài)
RULES.md                      # luật chơi: gold sạch, trung thực, AI, dữ liệu, git
RUBRIC.md                     # thang 100 điểm
SUBMISSION.md                 # tên repo, cấu trúc bắt buộc, checklist nộp
STUDENT.md                    # thông tin cá nhân + kết quả tóm tắt
CHECKPOINTS.md                # CP0–CP6
data/
  images/img_001..036.jpg     # 36 ảnh, KHÔNG sửa
  split.csv                   # image,split,seg
  ATTRIBUTION.csv             # tác giả + giấy phép từng ảnh
  cache/                      # KQ YOLO26 + YOLOE chạy sẵn (conf 0.05, 36 ảnh) — phương án dự phòng
  fetch_candidates.py         # script của giảng viên, học viên không chạy
cvat/labels_generated.json    # JSON nhãn dán vào Project CVAT (sinh từ taxonomy)
docs/
  SETUP.md  COLAB.md  MODELS.md  CVAT_AI.md  REVIEW_CHECKLIST.md
  THRESHOLD.md  CALIBRATION_QC.md  ADVANCED.md  OFFLINE.md (file này)
  day08_setup.py              # "tool" độc lập: download --profile / check [--light]
  setup_wizard.html           # wizard chuẩn bị môi trường (nhúng day08_setup.py)
  assets/                     # ảnh minh hoạ + make_figures.py
guideline/
  guideline_v1.0_label_contract.md   # label contract gốc (không sửa)
  TEMPLATE_v1.1.md                   # khung để viết bản v1.1
models/                       # trọng số (.pt, .ts) — KHÔNG commit
notebooks/day08_colab.ipynb   # notebook Colab (ô 0–8), xuất colab_stage1.zip / colab_outputs.zip
scripts/                      # 19 script, xem bảng 3.2
submission/                   # bài nộp, xem mục 4
tests/test_workflow.py        # self-test luồng (unittest)
walkthrough_images/           # ảnh minh hoạ đã chụp cho các bước
offline/                      # (không commit) bundle wheel + models cho máy offline
work/                         # (không commit) thư mục làm việc: prelabel/, tasks/, qc/, runs.csv
```

### 3.2 Bảng script

| Script | Việc | Lệnh chính | Đầu ra / ghi chú |
|---|---|---|---|
| `scripts/check_env.py` | kiểm môi trường (không tải gì) | `python scripts/check_env.py [--cvat http://localhost:8080]` | in `PASS/WARN/FAIL`, cuối `SẴN SÀNG.`; FAIL nếu thiếu thư viện / ≠36 ảnh / thiếu split.csv |
| `docs/day08_setup.py` | “tool” độc lập tải model + kiểm | `python docs/day08_setup.py download --profile 8gb\|16gb` · `... check [--light]` | tải vào `models/`; `check` chạy thử YOLO26/YOLOE/SAM bằng CPU trên `bus.jpg` + thử CVAT `localhost:8080` |
| `scripts/download_models.py` | tải trọng số vào `models/` | `--set core\|small\|sam1\|sam1-base\|sam2\|hf\|sam3\|all` | **không** tải `mobileclip2_b.ts` |
| `scripts/make_cvat_labels.py` | sinh JSON nhãn CVAT từ taxonomy | `python scripts/make_cvat_labels.py` | `cvat/labels_generated.json` (8 nhãn, `type: any`, có `conf`/`can_review`) |
| `scripts/prepare_workdirs.py` | copy ảnh thành thư mục Task | `[--advanced] [--seed 42]` | `work/tasks/gold` (4), `review` (24); `--advanced` thêm `pilot`/`seg`/`selfcheck` (8 mỗi thư mục) |
| `scripts/prelabel.py` | chạy 1 model → COCO + `viz/` | xem 5.2 | `<out>/pre.coco.json`, `viz/`, `run.json`; ghi 1 dòng `work/runs.csv` |
| `scripts/sam2_points.py` | SAM với click (mô phỏng Interactor) | `--image --pos x,y --neg x,y --box x1,y1,x2,y2 --sam <pt> --out --name` | `<out>/<name>.jpg` (mask tô màu, chấm xanh/đỏ); in diện tích mask + số polygon |
| `scripts/merge_prelabels.py` | gộp nhiều pre-label, áp ngưỡng, loại trùng | `merge_prelabels.py FILE1 FILE2 [--iou 0.6] [--conf 0.25] [--thr "..."] [--split "batch,gold"] --out ...` | file COCO gộp; file đứng trước được ưu tiên |
| `scripts/threshold_sweep.py` | chọn ngưỡng theo lớp | `--pred pre1 pre2 --gold gold_manual.json [--step 0.05] [--min-objects 5] [--max-thr 0.5]` | `submission/qc/threshold_sweep.{csv,md,png}`; in chuỗi `--thr "..."` |
| `scripts/inject_traps.py` | cài 12 bẫy (4 loại) | `--pre <pre gộp> --student <MSHV> [--seed 0] [--n 12]` | `pre_traps_import.coco.json` (nạp CVAT), `pre_traps.coco.json` (đủ batch+gold), `traps_key.json` (đáp án) |
| `scripts/review_stats.py` | đo accepted/edited/deleted/added, PR so gold, bắt bẫy | `--pre pre_traps.coco.json --final batch_reviewed.json --gold gold_manual.json --key traps_key.json [--mode det\|seg] [--splits batch,gold]` | `submission/qc/review_stats.{json,md}` |
| `scripts/validate_labels.py` | kiểm file COCO (tên ảnh, lớp, box, polygon) | `FILE --expect gold\|batch\|seg\|comma-list` hoặc `--all` | in `OK` / `LỖI` / cảnh báo (box < 20 px, ảnh thừa…) |
| `scripts/build_release.py` | nhãn cuối 24 ảnh batch | `python scripts/build_release.py` | `submission/release/final_det.coco.json`, `release_summary.csv` (class,count), `release_manifest.json`; áp `qc_fixed.json` nếu có |
| `scripts/qc_sample.py` | lấy mẫu QC + phiếu kiểm | `--n 5 --seed 42 [--max-hard 3] [--force]` | `qc/qc_sample.csv`, `qc/qc_review.csv`, ảnh `work/qc/review/*.jpg` có id từng vật |
| `scripts/qc_report.py` | tính số đo QC + **gợi ý** quyết định | `[--guideline-gap N] [--data-problem N]` | `submission/qc/qc_report_auto.md`; không tự ghi REPORT |
| `scripts/self_consistency.py` | so 2 bộ nhãn (gán lại mù / bạn học) | `--a file --b file [--images-from b] [--mode det\|seg]` | `submission/qc/self_check.{json,md,csv}` |
| `scripts/coco_to_cvat_xml.py` | COCO → CVAT XML (polygon sạch) | `--coco file --shape polygon\|box --out file.xml` | upload bằng format **CVAT 1.1** |
| `scripts/cvat_function_yolo26.py` | hàm cho `cvat-cli` (nâng cao B3) | xem docs/CVAT_AI.md mục 6 | 6 lớp COCO, **không** có `three_wheeler` |
| `scripts/viz.py` | vẽ bất kỳ file COCO lên ảnh | `--coco file --out dir [--images ...] [--ids]` | ảnh `.jpg` để xem bằng mắt |
| `scripts/check_submission.py` | tự kiểm bài nộp (chỉ kiểm có/đủ) | `python scripts/check_submission.py` | in `CẢNH BÁO` / `LỖI`; cuối `0 lỗi … Sạch lỗi: có thể nộp.` hoặc `Sửa hết LỖI rồi nộp.` |
| `scripts/common.py` | thư viện dùng chung (taxonomy, COCO, IoU, ghép đối tượng) | — | học viên **không cần** sửa; đổi lớp/ánh xạ thì sửa `taxonomy.json` |
| `tests/test_workflow.py` | self-test luồng (unittest) | `python -m unittest tests.test_workflow` | kiểm cả script `docs/day08_setup.py` |

### 3.3 Cờ quan trọng của `prelabel.py`

| Cờ | Ý nghĩa |
|---|---|
| `--model` | `yolo26` (closed-set) · `yoloe` (open-vocab) · `gdino` · `owlv2` · `sam3` (nâng cao) |
| `--size` | YOLO26: `n s m l x` (mặc định `m`); YOLOE: `s m l` (mặc định `s`) |
| `--weights` | ghi đè tên file trọng số / id HF / **đường dẫn local** (cần cho offline HF) |
| `--split` | `pilot`, `gold`, `batch`, `seg`, danh sách `a,b` hoặc `all` |
| `--images` | danh sách tên ảnh cụ thể (thay `--split`) |
| `--prompts "cụm=chữ"` | open-vocab: gán cụm chữ → lớp (mặc định từ `taxonomy.prompts`) |
| `--classes` | chỉ **giữ** các lớp này |
| `--conf`, `--thr "person=0.4,car=0.5"` | ngưỡng mặc định / ngưỡng riêng từng lớp |
| `--segmenter` | `none` (mặc định) · `native` (mask của model) · `sam`/`sam2` (box → SAM) |
| `--sam` | trọng số SAM: `mobile_sam.pt`/`sam_b.pt` (SAM 1) · `sam2.1_t/s/b/l.pt` (SAM 2.1); mặc định `sam2.1_b.pt` |
| `--map motorcycle=motorbike` | bổ sung ánh xạ tên thô → lớp (ngoài `coco_map`) |
| `--nms 0.7` | open-vocab: IoU loại box trùng giữa các cụm chữ cùng lớp |
| `--imgsz 1024` | kích thước suy luận |
| `--text-thr 0.25` | riêng Grounding DINO |
| `--device auto\|cpu\|mps\|cuda` | thiết bị |
| `--out` | thư mục đầu ra (bắt buộc) |
| `--no-viz`, `--note` | tắt ảnh xem nhanh / ghi chú vào `runs.csv` |

Cuối mỗi lần chạy, script in **tên thô không có trong ánh xạ** (box bị bỏ im lặng nếu không đọc), **số box theo lớp**, thời gian nạp / chạy trên ảnh.

## 4. Sản phẩm nộp và cách chấm

### 4.1 Repo và tên file

- Repo **public**, tên mẫu: `K5-L23-DAY08-[MSHV]-[HoTen]-AIPrelabel` (HoTen không dấu, viết liền). Ví dụ `K5-L23-DAY08-22001234-NguyenVanA-AIPrelabel`.
- `README.md` phải có dòng tiêu đề: `# Day 08 Lab — <MSHV> - <Họ và tên>`.
- `STUDENT.md` điền đủ (kể cả “Kết quả tóm tắt”, số phải kèm mẫu số).
- Nộp link repo qua LMS `https://aithucchien.vinuni.edu.vn` hoặc kênh giảng viên chỉ định; hạn do giảng viên thông báo.

### 4.2 Cấu trúc bắt buộc của `submission/`

```text
submission/
├── taxonomy.json                      # KHÔNG đổi mục rules
├── REPORT.md                          # báo cáo 8 mục (khuôn có sẵn)
├── model_log.csv                      # ≥ 4 lần chạy, ≥ 3 model (có SAM), có nhận xét
├── time_log.csv                       # bước gold_manual + batch_review, phút thật
├── guidelines/guideline_v1.1_calibrated.md   # CHỈ phần bổ sung so v1.0
├── prelabel/
│   ├── pre_traps_import.coco.json     # nạp CVAT (24 ảnh batch + bẫy)
│   ├── pre_traps.coco.json            # đầy đủ batch+gold (cho review_stats --pre)
│   └── traps_key.json                 # đáp án bẫy — KHÔNG mở trước khi export
├── labels/
│   ├── gold_manual.json               # 4 ảnh gold gán tay
│   └── batch_reviewed.json            # 24 ảnh batch sau review
├── masks/                             # ≥ 3 ảnh mask SAM
├── release/
│   ├── final_det.coco.json
│   ├── release_summary.csv
│   └── release_manifest.json
└── qc/
    ├── threshold_sweep.{csv,md,png}
    ├── review_stats.{json,md}
    ├── calibration_decisions.csv      # ≥ 3 dòng
    └── qc_sample.csv, qc_review.csv, qc_report_auto.md
```

Nâng cao (tuỳ chọn): `labels/pilot_reviewed.json`, `labels/seg_reviewed.json`, `labels/self_check_blind.json`, `labels/qc_fixed.json`, `qc/self_check.*`, `qc/review_stats_seg.*`, `prelabel/seg_pre.coco.json`.
**Không commit:** `work/`, `models/`, `*.pt`, `*.pth`, `*.onnx`, `*.ts`, token, mật khẩu, ảnh ngoài bộ dữ liệu, thư mục `offline/`.

### 4.3 Rubric 100 điểm

Điểm dựa trên **bằng chứng trong repo và chất lượng phân tích**, không dựa vào số đo đẹp hay được Release. Bắt bẫy 60% mà phân tích trung thực tốt hơn 100% mà không giải thích được.

| # | Hạng mục | Điểm | Bằng chứng |
|---|---|---:|---|
| 1 | Cài đặt, tái lập, git | 8 | `check_submission.py`, commit (≥ 4, theo từng Part), `STUDENT.md`, README |
| 2 | Gold gán tay + dùng label contract | 10 | `labels/gold_manual.json` (4 ảnh, hợp lệ, gán trước khi thấy nhãn máy), `can_review` dùng đúng nghĩa, `time_log` thật |
| 3 | Tour model và recipe | 20 | `model_log.csv` ≥ 4 lần / ≥ 3 model (có **SAM**, ghi phiên bản), `what_went_wrong` cụ thể (ảnh, vật, loại lỗi), REPORT mục 1 |
| 4 | Chọn ngưỡng theo lớp | 8 | `qc/threshold_sweep.*`, `--thr` dùng đúng ở Part 4, nói rõ giới hạn chọn+chấm trên cùng gold |
| 5 | Review và đo | 28 | `batch_reviewed.json` (24 ảnh) nạp đúng `pre_traps`; đọc đúng `review_stats` (mẫu số, vì sao loại vùng bẫy khỏi profile, PR máy so gold); phân tích bẫy lọt + thiên kiến tự động hóa; 3 quy tắc giảm thiên kiến kiểm được |
| 6 | Mask SAM | 8 | ≥ 3 ảnh trong `masks/`, ghi phiên bản SAM + số click, phân tích mask gồm người lái/hàng hoá, đối chiếu guideline 4.2 |
| 7 | Calibration | 10 | `calibration_decisions.csv` ≥ 3 dòng (≥ 1 `loi_may`, ≥ 1 `cau_hoi_mo`), quy tắc **thật sự có** trong v1.1 + ảnh ví dụ, lịch sử đổi đúng |
| 8 | QC và báo cáo | 8 | mẫu QC + cách chọn, **hai cách đếm tỉ lệ lỗi kèm mẫu số**, quyết định Hold→Rework→Release khớp số liệu, nêu giới hạn mẫu nhỏ |

Trừ điểm: thiếu thành phần chỉ chấm phần có bằng chứng; **số liệu giả** = 0 điểm phần liên quan + báo giảng viên; nộp muộn / sai tên repo / sai link theo RULES; bài nâng cao tối đa **+5 điểm thưởng**, không bù phần cốt lõi thiếu.

## 5. Quy trình từng Part

Tất cả lệnh chạy **từ thư mục gốc repo**, venv đã kích hoạt. Trên PowerShell: giá trị có dấu phẩy/`=` phải bọc nháy kép; dòng nhiều dòng dùng `` ` `` hoặc gộp một dòng.

### 5.1 Part 0 — Cài đặt (trước lớp)

1. Fork repo đề thành repo public tên đúng mẫu, `git clone`; đặt `git config user.name` / `user.email`.
2. Cài môi trường theo mục 8 (có mạng) hoặc mục 9 (offline). `python scripts/check_env.py` → **SẴN SÀNG**.
3. `python scripts/make_cvat_labels.py` → `cvat/labels_generated.json`.
4. CVAT (Docker, mục 8.4): tạo Project → Labels → tab **Raw** → dán `labels_generated.json` → Save. Kiểm 8 nhãn, có `conf` + `can_review`, type **Any**.
5. `python scripts/prepare_workdirs.py` → tạo Task `gold-manual` từ `work/tasks/gold/` (4 ảnh) và `batch-review` từ `work/tasks/review/` (24 ảnh).
6. Điền `STUDENT.md`, đổi dòng tiêu đề README. Commit.

### 5.2 Part 1 — Gold: gán tay 4 ảnh, không AI (25')

**Lý do:** nếu gán gold sau khi đã thấy nhãn máy, bạn bị “neo” và không đo được thiên kiến. Gold là chuẩn để đo **máy**.

1. Task `gold-manual`: Rectangle cho mọi `person, bicycle, motorbike, car, bus, truck, three_wheeler`; cụm quá nhỏ/dày → `ignore_region`. **Không chạy model nào lên 4 ảnh này.**
2. Bấm giờ → `submission/time_log.csv` dòng `gold_manual` (cột `step, images, minutes, note`).
3. Ca không chắc quy tắc → tick `can_review` + ghi 1 dòng vào REPORT mục “Ghi chú”.
4. Export **COCO 1.0, tắt Save images** → `submission/labels/gold_manual.json`; `python scripts/validate_labels.py submission/labels/gold_manual.json --expect gold` tới khi `OK`. Commit.

### 5.3 Part 2 — Tour 3 model (30')

```powershell
# A. YOLO26 — hai ngưỡng
python scripts/prelabel.py --model yolo26 --size m --split pilot --conf 0.25 --out work/prelabel/t_y26
python scripts/prelabel.py --model yolo26 --size m --split pilot --conf 0.5  --out work/prelabel/t_y26_c05
# B. YOLOE — gõ chữ tìm tuk-tuk / xích lô / ba gác
python scripts/prelabel.py --model yoloe --classes three_wheeler --split pilot --conf 0.25 --out work/prelabel/t_yoloe
# C. SAM (BẮT BUỘC) — box YOLO26 → mask; đổi --sam để chọn SAM 1 / 2.1
python scripts/prelabel.py --model yolo26 --size m --classes "person,motorbike,car" --images img_004.jpg img_013.jpg `
    --conf 0.3 --segmenter sam --sam sam2.1_b.pt --out work/prelabel/t_sam
# Rồi chạy mọi ảnh, ngưỡng thấp 0.05 (nguồn cho Part 3, 4)
python scripts/prelabel.py --model yolo26 --size m --split all --conf 0.05 --out work/prelabel/y26_all --no-viz
python scripts/prelabel.py --model yoloe --classes three_wheeler --split all --conf 0.05 --out work/prelabel/yoloe_all --no-viz
```

**Phải thấy được** (ghi `model_log.csv`): YOLO26 gọi tuk-tuk là gì; đổi ngưỡng mất/thừa gì; YOLOE tìm được tuk-tuk nào, sót cái nào, box lỏng ở đâu; mask SAM sai ở đâu khi box sai. Sản phẩm: `model_log.csv` ≥ 4 dòng, ≥ 3 model (có SAM).

### 5.4 Part 3 — Ngưỡng theo lớp (10')

```powershell
python scripts/threshold_sweep.py --pred work/prelabel/y26_all/pre.coco.json work/prelabel/yoloe_all/pre.coco.json `
    --gold submission/labels/gold_manual.json
```

Đọc `submission/qc/threshold_sweep.md` + `.png`; chép chuỗi `--thr "..."` để dùng Part 4; ghi ngưỡng + lý do vào REPORT mục 1. Lớp có **< 5 vật** trong gold (thường bus/truck/bicycle) **không được** chọn ngưỡng riêng — script để mặc định 0.3, đó là kết quả đúng. Nếu không ngưỡng nào ≤ 0.5 đạt recall 90% ⇒ nói trong báo cáo, không ép. Giới hạn phải nói: chọn và chấm trên cùng 4 ảnh gold ⇒ PR của máy trên gold **lạc quan**.

### 5.5 Part 4 — Pre-label, ảnh bẫy và review (95')

```powershell
# 1) Gộp pre-label cho batch + gold (gold chỉ để đo máy, không vào CVAT)
python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json `
    --thr "<chuỗi ngưỡng từ Part 3>" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
# 2) Cài 12 bẫy (hạt giống = MSHV)
python scripts/inject_traps.py --pre work/prelabel/merged_review/pre.coco.json --student "00123456"
# 3) Task batch-review -> Upload annotations -> COCO 1.0 -> submission/prelabel/pre_traps_import.coco.json
# 4) Review theo checklist (mục 6.5) -> export -> submission/labels/batch_reviewed.json
# 5) Đo
python scripts/review_stats.py --pre submission/prelabel/pre_traps.coco.json --final submission/labels/batch_reviewed.json `
    --gold submission/labels/gold_manual.json --key submission/prelabel/traps_key.json
```

- Task `batch-review` tạo từ **đúng** 24 ảnh `work/tasks/review/`. Sau khi nạp, đếm số box ở ảnh đầu và so với `viz/`.
- Review khoảng 3 phút/ảnh; 6 ảnh dày (`005, 018, 021, 024, 028, 029`): cụm xe nhỏ < 20 px dùng `ignore_region`, không vẽ từng vật.
- `python scripts/validate_labels.py --all` sạch rồi mới đo.
- Đọc `qc/review_stats.md`, viết REPORT mục 2–4: AI error profile, bẫy nào lọt, máy có đáng dùng không (accepted + thời gian), **3 quy tắc giảm thiên kiến**.
- `traps_key.json` **chỉ mở sau khi** export `batch_reviewed.json`.

### 5.6 Part 5 — Mask SAM cho 3 vật (15')

```powershell
python scripts/sam2_points.py --image img_013.jpg --pos "400,480" --pos "600,110" --neg "720,300" `
    --sam sam2.1_b.pt --out submission/masks --name tuktuk_img013
```

Mỗi ảnh kết quả vào `submission/masks/` (≥ 3). Ghi REPORT mục 5: SAM phiên bản nào, số click, mask có gồm người lái/hàng hoá không (so guideline 4.2), lỗi. Tuỳ chọn: làm lại bằng Interactor SAM trong CVAT và so sánh.

### 5.7 Part 6 — Calibration + QC + báo cáo (45')

```powershell
python scripts/build_release.py
python scripts/qc_sample.py --n 5 --seed 42
# mở work/qc/review/*.jpg, điền submission/qc/qc_review.csv
python scripts/qc_report.py          # thêm --guideline-gap N nếu còn câu hỏi chưa chốt
python scripts/check_submission.py
```

1. **Calibration (20')**: gom ca từ 7 câu hỏi mở + box `can_review` + lỗi lặp của máy + bẫy lọt; ghi **≥ 3 dòng** `submission/qc/calibration_decisions.csv`; viết `guideline_v1.1_calibrated.md` (chỉ phần bổ sung, khung `guideline/TEMPLATE_v1.1.md`, mỗi ca khó có **ảnh ví dụ**).
2. **QC (15')**: `qc_sample.py` lấy tối đa 3 ảnh có `can_review` nhiều nhất + 5 ảnh ngẫu nhiên; điền `qc_review.csv` (mỗi lỗi 1 dòng); `qc_report.py` sinh `qc_report_auto.md` + **gợi ý** quyết định.
3. **Báo cáo (10')**: hoàn thiện `REPORT.md` 8 mục; mục 7 là quyết định **Release / Hold / Rework**, tỉ lệ **kèm mẫu số**, nêu giới hạn mẫu nhỏ.

## 6. Định dạng file và hợp đồng chấm (đọc kỹ khi trả lời về số liệu)

### 6.1 COCO 1.0 mà lab dùng

```json
{"info": {...}, "licenses": [], 
 "categories": [{"id": 1, "name": "person"}, ...],   // id = thứ tự lớp trong taxonomy (1..8)
 "images": [{"id": 1, "file_name": "img_003.jpg", "width": W, "height": H}],
 "annotations": [{"id": 1, "image_id": 1, "category_id": 3,
                  "bbox": [x, y, w, h], "area": A, "iscrowd": 0,
                  "segmentation": [[x1,y1,x2,y2,...]],   // danh sách polygon phẳng; rỗng nếu box
                  "attributes": {"conf": "0.87", "can_review": "false"},
                  "score": 0.87}]}   // score chỉ có ở pre-label
```

- Script **map lớp theo tên** và **ảnh theo tên file** (bỏ tiền tố thư mục) — tên nhãn CVAT phải trùng từng chữ tên lớp taxonomy.
- `can_review` lưu dạng chuỗi `"true"/"false"` (CVAT lưu checkbox dạng chuỗi).
- `ignore_region` là đối tượng bình thường nhưng **không bị chấm**; box nằm ≥ 50% trong vùng ignore được coi là bị phủ.

### 6.2 CSV trong repo

| File | Cột |
|---|---|
| `work/runs.csv` | `run_id, part, model, params, split, n_boxes, sec_per_img, what_went_right, what_went_wrong, decision, out, device, n_images, n_masks, dropped, note` |
| `submission/model_log.csv` | `run_id, part, model, params, split, n_boxes, sec_per_img, what_went_right, what_went_wrong, decision` |
| `submission/time_log.csv` | `step, images, minutes, note` — `step` bắt buộc có `gold_manual` và `batch_review`; phút là **số thật bấm đồng hồ** |
| `submission/qc/calibration_decisions.csv` | `case_id, image, nguon, van_de, quyet_dinh, quy_tac_ghi_vao_guideline, phien_ban_guideline`; `nguon` ∈ `cau_hoi_mo`, `can_review`, `loi_may`, `bay_lot`, `self_check` |
| `submission/qc/qc_sample.csv` | `image, la_ca_kho, so_doi_tuong` |
| `submission/qc/qc_review.csv` | `image, object_id, error_type, reviewer, note`; `error_type` ∈ `khong`, `bo_sot`, `thua`, `sai_lop`, `sai_vi_tri_kich_thuoc`, `sai_thuoc_tinh`, `sai_mask`; dòng trống `error_type` = chưa kiểm |
| `submission/release/release_summary.csv` | `class, count` |

`model_log.csv`: mỗi lần chạy 1 dòng; copy từ `work/runs.csv` rồi điền `part`, `what_went_right`, `what_went_wrong` (phải **cụ thể**: ảnh nào, vật nào), `decision`.

### 6.3 Hợp đồng đo của `review_stats.py`

Ghép box máy ↔ box cuối **cùng lớp**, tham lam 1-1 theo IoU giảm dần, ngưỡng `match_iou` = 0.5; mode seg so **IoU mask**.

| Trạng thái | Điều kiện | Mẫu số |
|---|---|---|
| `accepted` | IoU ≥ `accept_iou` 0.9 (mask 0.85) | **box máy** (`pre_boxes`) |
| `edited` | 0.5 ≤ IoU < ngưỡng accepted | **box máy** |
| `deleted` | box máy không ghép được (kể cả sai lớp) | **box máy** |
| `added` | box cuối không ghép được | **đối tượng sau duyệt** (`final_objects`) |

- **Đổi lớp = xoá + thêm** (theo hợp đồng của slide); script in thêm con số “đổi lớp” riêng.
- `ignore_region` không bị chấm; box máy nằm trong vùng ignore không tính.
- So với **gold**: TP/FP/FN cùng lớp, IoU ≥ 0.5 → precision/recall của **máy** (so với gold) và của **bản cuối**; `residual` = lỗi của máy còn trong nhãn bạn, `new` = lỗi bạn tạo ra (chỉ khi có `gold_reviewed`, bài nâng cao).
- **Bẫy** (`--key`): `missing` đúng khi box gốc có lại (cùng lớp, IoU ≥ 0.5); `extra` đúng khi box ma biến mất; `wrong_class` đúng khi lớp gốc trở lại; `loose` đúng khi box cùng lớp có IoU ≥ 0.75 với box gốc (hoặc đã rời box lỗi IoU < 0.8 và vẫn đạt IoU ≥ 0.5 với box gốc). `traps = {n, caught, rate}`.
- Output: `submission/qc/review_stats.json` (các khoá `mode, images, pre_boxes, final_objects, accepted, edited, deleted, added, class_changed_info, gold, traps, accept_threshold, …`) + `review_stats.md`.

### 6.4 `threshold_sweep.py` chọn ngưỡng thế nào

- Quét `conf` từ 0.05 → `--max-thr` 0.5 theo `--step` 0.05 trên kết quả thô (`--pred`, ngưỡng 0.05) so với `--gold`.
- **Ngưỡng gợi ý** = ngưỡng **cao nhất (≤ 0.5)** mà **recall lớp đó còn ≥ 0.9** (`recall_target`).
- Lớp có **< 5 vật** trong gold (`--min-objects`) → không chọn riêng, dùng mặc định **0.3**.
- In ra chuỗi `--thr "person=…,car=…"` để dán vào `merge_prelabels.py` Part 4. Output `submission/qc/threshold_sweep.{csv,md,png}`.

### 6.5 Checklist review 4 bước (+ bước 0)

**Bước 0 — quét ảnh trước (bắt buộc):** ẩn hết nhãn máy, nhìn ảnh ~15 giây, tự đếm vật theo lớp, **rồi mới** hiện nhãn máy.

| # | Câu hỏi | Việc làm | Trạng thái |
|---|---|---|---|
| 1 | **Bỏ sót**: vật nào chưa có box (xa, bị che, mép, tuk-tuk/xích lô)? | vẽ thêm (`N`) | `added` |
| 2 | **Thừa**: box nào không có vật thật (bóng, tượng, phản chiếu, cột, biển)? | xoá (`Del`) | `deleted` |
| 3 | **Sai nhãn**: tuk-tuk bị `car`/`truck`, ba gác thành `motorbike`? | đổi nhãn (`Ctrl`+số) | `edited` |
| 4 | **Lỏng**: box có ôm đủ vật, dư nền? mask bám biên? | kéo lại | `edited` |

Không lỗi → giữ (`accepted`). Ca **không chắc quy tắc** → tick `can_review`. **Không** hạ chuẩn vì `conf` cao; `conf` chỉ để ưu tiên nhìn kỹ box điểm thấp. Nghỉ 3–5 phút sau mỗi ~10 ảnh.

**12 ảnh bẫy** (`inject_traps.py`, hạt giống = MSHV, chỉ trên ảnh batch, tối đa 1 bẫy/ảnh): 4 loại × 3 = **12**: `missing` (xoá 1 box đúng, điểm ≥ 0.5), `extra` (thêm box ma 6–18% × 10–28% kích thước ảnh, IoU < 0.05 với box khác, điểm 0.45–0.8, lớp car/motorbike/person/truck), `wrong_class` (đổi lớp qua bảng SWAP: car↔truck/three_wheeler, truck↔car/bus, motorbike↔bicycle, bicycle↔motorbike, person↔bicycle, bus↔truck, three_wheeler↔car/motorbike), `loose` (dịch box 18–35%, IoU với box gốc 0.45–0.68). Bắt ≥ 80% (≥ 10/12); đọc **danh sách bẫy lọt**, không chỉ đọc %.

### 6.6 QC và quyết định Release / Hold / Rework

- Mẫu QC: tối đa **3 ảnh `can_review`** nhiều nhất + **5 ảnh ngẫu nhiên** (seed 42) từ 24 ảnh batch (tổng tối đa 8 ảnh).
- Số đo: lỗi theo **đối tượng** = đối tượng lỗi / đối tượng trong mẫu; lỗi theo **ảnh** = ảnh có ≥ 1 lỗi / ảnh trong mẫu (dùng cho ngưỡng ≤ 2%); accepted/edited/deleted ÷ box máy; added ÷ đối tượng sau duyệt; bắt bẫy = bẫy đúng / tổng bẫy.
- `qc_report.py` gợi ý: **HOLD** nếu `--guideline-gap > 0` hoặc `--data-problem > 0`; **REWORK** nếu ảnh lỗi > 2%, hoặc lỗi hệ thống (loại lỗi top chiếm ≥ 50% và ≥ 3 lỗi), hoặc bắt bẫy < 80%, hoặc `self_check` fail; còn lại **RELEASE**. Script in **cận trên Wilson 95%** cho tỉ lệ ảnh lỗi thật.
- Giới hạn thống kê: 0 lỗi trong n ảnh ⇒ tỉ lệ thật có thể tới ~**3/n** (quy tắc 3): 0/10 ảnh ⇒ tới ~30%. Vì vậy “đạt 2%” trên mẫu vài ảnh là **chưa chứng minh được** — phải nói rõ trong báo cáo.
- Báo **Rework** trung thực không bị trừ điểm; báo **Release** khi số liệu không đạt thì bị trừ.

### 6.7 `validate_labels.py` kiểm gì (và không kiểm gì)

Kiểm: tên file; ảnh đủ theo `--expect` (gold/batch/seg); ảnh lạ ngoài bộ dữ liệu; lớp lạ (CVAT còn bộ nhãn cũ); box nằm ngoài ảnh; box cao < 20 px → **cảnh báo** (guideline nói dùng `ignore_region`); mode seg yêu cầu có polygon.
**Không** kiểm: box có ôm sát vật thật; lớp có đúng thực tế; vật mà cả bạn lẫn máy cùng bỏ sót (ngoài gold); bạn có theo checklist thật không. Muốn kiểm mấy cái đó phải nhìn ảnh.

### 6.8 `check_submission.py` — điều kiện coi là “sạch lỗi”

`LỖI` nếu: `taxonomy.rules` khác bản gốc; README thiếu dòng `# Day 08 Lab — <MSHV> - <Tên>`; `STUDENT.md` còn `<<`; `REPORT.md` < 7 mục `##` / còn `<<` / thiếu chữ Release|Hold|Rework / không nhắc SAM; `guideline_v1.1` < 3 mục / thiếu `img_0xx`; `model_log` < 4 dòng / < 3 model / không có dòng SAM; `time_log` thiếu `gold_manual` hoặc `batch_review`; thiếu `gold_manual.json`/`batch_reviewed.json`; `validate_labels --all` có lỗi; thiếu `pre_traps.coco.json`/`traps_key.json`; `traps_key.student` ≠ MSHV trong README; `masks/` < 3 ảnh; thiếu file qc; `review_stats.json` thiếu phần bẫy hoặc bẫy < 10, hoặc thiếu phần gold; `qc_review.csv` còn `error_type` trống; `calibration_decisions.csv` < 3 quyết định hoặc thiếu `loi_may`/`cau_hoi_mo`; thiếu `release/final_det.coco.json`/`release_manifest.json`; git đang commit `work/`, `models/`, `*.pt/.pth/.onnx/.ts`; nghi lộ secret/token.
`CẢNH BÁO` (không chặn nộp): < 4 commit; `what_went_wrong` trống; guideline thiếu chữ `v1.1`; `nguon` lạ trong calibration.

## 7. Luật chơi, tính trung thực, giới hạn

- **Bài cá nhân**: mỗi người một repo public, một bộ nhãn, một báo cáo, commit của chính mình. Được thảo luận khái niệm với bạn học; **không** chia sẻ file nhãn / báo cáo / đáp án bẫy.
- **Gold phải sạch**: từ lúc nhận ảnh gold đến khi export `gold_manual.json`, **không** chạy model lên ảnh gold, **không** nhìn pre-label gold, **không** nhờ AI “nhìn ảnh rồi cho biết nên gán gì”. Nếu lỡ nhìn: ghi thật vào REPORT mục 8, gọi gold bị nhiễm và phân tích theo hướng đó (không tính gian lận; che giấu mới tính).
- **Dùng AI**: được dùng để giải thích khái niệm, gợi ý script, sửa chính tả; **ghi công cụ + cách kiểm tra** trong REPORT mục 8. **Không** để AI viết hộ kết luận (phân tích số đo, Release/Hold/Rework, quy tắc guideline).
- **Không bịa** nhãn, số đo, thời gian, dòng log, quyết định chưa làm. Số xấu mà trung thực vẫn được điểm; số đẹp mà bịa = 0 điểm phần liên quan + báo giảng viên.
- **Honor system ảnh bẫy**: không mở `traps_key.json` trước khi export; không sửa `pre_traps*.json` / `traps_key.json`.
- **Không sửa tay** output script: `review_stats.*`, `threshold_sweep.*`, `self_check.*`, `qc_report_auto.md`, `release_manifest.json`.
- **Không đổi** `taxonomy.rules`; không sửa / đổi tên / thêm bớt ảnh, không sửa `split.csv`; giữ `data/ATTRIBUTION.csv`.
- **Không commit** trọng số, `work/`, token/mật khẩu. Ảnh riêng tư / ảnh ngoài bộ dữ liệu: không đưa vào repo; không gửi ảnh lên dịch vụ AI đám mây khi giảng viên chưa cho phép.
- **Git**: repo public, **≥ 4 commit theo từng Part**, bằng tài khoản của bạn; bài chấm theo commit cuối trước hạn.
- **Giới hạn của số đo** (phải nói trong báo cáo): chỉ 4 ảnh gold ⇒ chọn + chấm cùng dữ liệu ⇒ PR của máy trên gold **lạc quan**; mẫu QC nhỏ ⇒ ngưỡng 2% không chứng minh được; 12 bẫy ⇒ một bẫy lọt đổi kết quả (đọc danh sách, không chỉ %); `accepted` cao mà bắt bẫy thấp = **thiên kiến tự động hóa** (bạn đang bấm Accept chứ chưa kiểm); `added` thấp không có nghĩa máy không sót (máy sót thì bạn không thấy để vẽ).

## 8. Cài đặt môi trường khi CÓ mạng

### 8.1 Yêu cầu

| Thứ | Bắt buộc | Ghi chú |
|---|---|---|
| Python **3.10–3.13** | có | khuyến nghị **3.12**; 3.14+ có thể chưa có wheel torch/opencv |
| `git` | có | YOLOE tự cài CLIP từ GitHub qua git (lần đầu) |
| ~8 GB ổ trống | có | bài chính ~0.35 GB trọng số + ~4 GB thư viện; nâng cao +~6 GB |
| GPU | không | CUDA / Apple MPS nhanh hơn; CPU vẫn làm được (`--size n --imgsz 640`) |
| Docker + CVAT | có | CVAT chạy **trên máy bạn**, cần ~5–8 GB RAM khi chạy |
| Tài khoản HuggingFace | tuỳ chọn | chỉ cho SAM 3 (gated) |

### 8.2 venv + thư viện

```powershell
# Windows
cd "D:\AI20KK\K5-L23-Day08-BuiCongDat-2A202603068-AI-Assisted-Labeling"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip && pip install -r requirements.txt
```

- **GPU NVIDIA**: cài torch CUDA **trước** requirements: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` rồi `pip install -r requirements.txt`.
- **Máy nhẹ / dùng Colab**: chỉ `pip install -r requirements-local.txt` (numpy, pillow, opencv-headless, matplotlib, pyyaml; không torch/ultralytics).
- **B3 cvat-cli**: `pip install -r requirements-cvat.txt` (cvat-sdk, cvat-cli ≥ 2.80).
- `uv` (tuỳ chọn): `uv venv --python 3.12 && uv pip install -r requirements.txt`.

### 8.3 Trọng số

```powershell
python scripts/download_models.py --set core   # yolo26m + yoloe-26s-seg + sam2.1_b (~0.35 GB) — bài chính
python scripts/download_models.py --set sam1   # (tuỳ chọn) mobile_sam ~40 MB
# máy yếu: --set small (yolo26n + yoloe-26s-seg + sam2.1_t)
# nâng cao: --set hf (Grounding DINO tiny + OWLv2 base, ~2 GB) · --set sam3 (gated, ~3.5 GB)
```

Hoặc dùng “tool” độc lập (có kèm `mobileclip2_b.ts` — bộ mã hoá chữ YOLOE):

```powershell
python docs/day08_setup.py download --profile 8gb     # yolo26n, yoloe-26s-seg, mobileclip2_b.ts, mobile_sam, sam2.1_t
python docs/day08_setup.py download --profile 16gb    # thêm yolo26s
```

| File | Dùng cho | Cỡ | Nguồn |
|---|---|---|---|
| `yolo26n.pt` | YOLO26 nano (`--size n`) | 5.5 MB | GitHub ultralytics/assets v8.4.0 · HF `Ultralytics/YOLO26` |
| `yolo26s.pt` | YOLO26 small (`--size s`) | 20 MB | như trên |
| `yolo26m.pt` | YOLO26 medium (`--size m`, bài chính) | 42 MB | như trên |
| `yoloe-26s-seg.pt` | YOLOE (open-vocab, gõ chữ) | 30 MB | như trên |
| `mobileclip2_b.ts` | **bộ mã hoá chữ YOLOE — bắt buộc** | 254 MB | GitHub ultralytics/assets v8.4.0 |
| `mobile_sam.pt` | SAM 1 bản nhẹ (MobileSAM) | ~41 MB | như trên |
| `sam2.1_t.pt` | SAM 2.1 tiny | ~78 MB | như trên |
| `sam2.1_b.pt` | SAM 2.1 base (mặc định `--sam`) | ~162 MB | như trên |

Base URL: `https://github.com/ultralytics/assets/releases/download/v8.4.0/<tên file>`.
**Bẫy:** `scripts/download_models.py` **không** tải `mobileclip2_b.ts`; có mạng thì YOLOE tự tải lần đầu, **offline thiếu file này ⇒ YOLOE chết** (xem mục 9).

**SAM 3 (nâng cao, gated):** tạo tài khoản HF → https://huggingface.co/facebook/sam3 → **Request access** → tạo token read → `hf auth login` (hoặc `HF_TOKEN`), rồi `python scripts/download_models.py --set sam3`. Chưa có quyền: bỏ qua, ghi `sam3` `decision = skipped`, dùng SAM 1/2.1 vẫn đủ yêu cầu.

### 8.4 CVAT bằng Docker

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
git checkout v2.81.0
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Mở `http://localhost:8080`, đăng nhập. Dừng: `docker compose down` (dữ liệu giữ trong volume). Lần đầu tải image 10–20 phút.
Project → Labels → tab **Raw** → dán `cvat/labels_generated.json` → Save → Submit & Open.
Task: **Create new task** → chọn Project → kéo cả thư mục ảnh vào → Submit.
Nạp nhãn: Task → **Actions (⋮) → Upload annotations** → **COCO 1.0** (box) hoặc **CVAT 1.1** (XML polygon) → OK / Replace.
Export: Task → **Actions → Export task dataset** → **COCO 1.0**, **tắt Save images** → tải `.zip` ở trang **Requests** → `annotations/instances_default.json` → đổi tên đúng và đặt vào `submission/labels/`.
Phím: `F1` danh sách; `Tab` box kế; `Del` xoá; `Ctrl+1…9,0` đổi nhãn; `N` lặp thao tác vẽ; `F`/`D` ảnh sau/trước; `Ctrl+S` lưu. Lọc theo `source` (`file` = nhãn nạp, `manual` = bạn vẽ) và `conf`.
Nâng cao B1 (Interactor SAM) / B2 (Automatic annotation) cần **serverless Nuclio**; B3 (`cvat-cli` + `scripts/cvat_function_yolo26.py`) chạy được với CVAT Docker thường.

### 8.5 Kiểm tra

```powershell
python scripts/check_env.py --cvat http://localhost:8080
python docs/day08_setup.py check            # hoặc --light nếu chỉ cài requirements-local
```

Kỳ vọng: các dòng `PASS`/`[ OK ]`, cuối `SẴN SÀNG.` / `SẴN SÀNG cho buổi học.`. `WARN` (không GPU, chưa đăng nhập HF) là bình thường.

### 8.6 Colab (máy yếu, cần mạng)

`docs/COLAB.md` + `notebooks/day08_colab.ipynb` (ô 0–8, đã chạy thử trên T4 GPU). Máy bạn chỉ cần `requirements-local.txt` + CVAT + git. Runtime → Change runtime type → T4 GPU. Ô 0 clone repo (đổi `REPO_URL` thành repo của bạn); ô 2 tải model; ô 5 xuất `colab_stage1.zip`, ô 8 xuất `colab_outputs.zip` → giải nén **vào thư mục gốc repo**, giữ đường dẫn `work/…`, `submission/…`. **Phiên Colab mất file khi hết hạn** → tải zip sau mỗi Part. Ghi `device = cuda (Colab T4)` vào `model_log.csv`. Không upload ảnh ngoài bộ dữ liệu; `HF_TOKEN` để trong Colab Secrets.

## 9. Chạy hoàn toàn offline (không có Internet)

Nguyên tắc: **máy A (có mạng)** chuẩn bị 4 thứ — wheel thư viện, trọng số (kể cả `mobileclip2_b.ts`), snapshot HF nếu làm nâng cao, image Docker CVAT — copy sang **máy B (offline)**.

### 9.1 Việc nào cần mạng

| Việc | Cần mạng? | Ghi chú |
|---|---|---|
| `pip install` thư viện, cài `clip` | có | chuẩn bị wheel ở máy A |
| `download_models.py`, `day08_setup.py download` | có | hoặc copy `models/` |
| Lần chạy YOLOE đầu tiên | có **nếu** thiếu `mobileclip2_b.ts`/`clip`; có sẵn thì không |
| Grounding DINO / OWLv2 / SAM 3 | có | tải snapshot HF về thư mục local, truyền `--weights` |
| `git clone` / fork | có | offline: chép nguyên thư mục repo kèm `.git` |
| `docker compose up` lần đầu (lấy image CVAT) | có | offline: `docker save` → `docker load` |
| CVAT: Project/Task, import/export, review | **không** | chỉ `localhost:8080` |
| `prelabel.py`, `sam2_points.py` khi đủ trọng số | **không** | đường “offline → import” |
| Mọi script đo (`merge`, `threshold_sweep`, `inject_traps`, `review_stats`, `validate`, `build_release`, `qc_*`, `self_consistency`, `check_submission`, `make_cvat_labels`, `prepare_workdirs`) | **không** | chỉ đọc/ghi file |
| `check_env.py`, `day08_setup.py check` | **không** | chỉ thư viện/file local + `localhost:8080` |
| Google Colab | có | offline không dùng được |
| `data/fetch_candidates.py` | có | chỉ giảng viên |

### 9.2 Chuẩn bị trên máy A

**a) Wheelhouse** (`offline/wheels/`), cùng hệ điều hành + phiên bản Python với máy B (Windows x64 + 3.12 khuyến nghị):

```powershell
New-Item -ItemType Directory -Force offline\wheels | Out-Null
pip download -r requirements.txt -d offline\wheels --only-binary=:all:
pip wheel --no-deps -w offline\wheels "https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"
pip download ftfy regex tqdm -d offline\wheels --only-binary=:all:
# tuỳ chọn B3: pip download -r requirements-cvat.txt -d offline\wheels --only-binary=:all:
```

GPU NVIDIA: thay lệnh đầu bằng `pip download -r requirements.txt -d offline\wheels --only-binary=:all: --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple`.
Kết quả tham chiếu (bản CPU, Windows 11 + Python 3.12.10): **67 wheel · ~330 MB**, có `clip-1.0-py3-none-any.whl`; cài ra torch 2.15.0+cpu, ultralytics 8.4.215, transformers 5.23.0.

**b) Trọng số:** chạy mục 8.3 trên máy A. Dùng `--set core` **nhớ tải thêm** `mobileclip2_b.ts` vào `models/`; hoặc dùng `day08_setup.py download --profile` (đã có file đó).

**c) HF (nâng cao):** `hf download IDEA-Research/grounding-dino-tiny --local-dir offline\hf\grounding-dino-tiny` (tương tự `google/owlv2-base-patch16-ensemble`, `facebook/sam3`). Lúc chạy: `--weights offline\hf\<tên>`. Bản `huggingface_hub` cũ dùng `huggingface-cli download`. Hoặc chép `%USERPROFILE%\.cache\huggingface` + đặt `HF_HOME`.

**d) CVAT:** clone/checkout `v2.81.0`; `docker compose config --images | Sort-Object -Unique | Set-Content ..\offline\cvat-images.txt` (Compose cũ: `docker compose config | Select-String 'image:'`); `docker compose pull`; `docker save -o ..\offline\cvat-images.tar (Get-Content ..\offline\cvat-images.txt)`; nén cả thư mục `cvat/` (gồm `docker-compose.yml`, file phụ thuộc, `.env` nếu có). Profile phụ (analytics/grafana) thì `docker compose --profile <tên> config --images` trước khi pull/save.

**e) Đóng gói:** `offline/models/` (copy `models\*.pt`, `models\*.ts`), `offline/repo-lab.zip` (tuỳ chọn, giữ `.git`). Dung lượng: wheel ~0.33 GB (CPU) / 2.5–3 GB (CUDA), model ~0.6 GB, CVAT ~6 GB ⇒ **~7–10 GB**. Copy thêm bộ cài Python 3.12, Docker Desktop, driver NVIDIA nếu máy B thiếu.

### 9.3 Cài trên máy B

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --no-index --find-links offline\wheels -r requirements.txt
pip install --no-index --find-links offline\wheels clip
# tuỳ chọn: pip install --no-index --find-links offline\wheels -r requirements-cvat.txt
Copy-Item offline\models\* models\ -Force
```

Lỗi `No matching distribution found` ⇒ wheel khác hệ/Python → tạo lại wheelhouse đúng máy đích.

**Biến môi trường "offline"** (mỗi phiên terminal; mất khi mở terminal mới):

```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:HF_DATASETS_OFFLINE="1"
$env:PIP_NO_INDEX="1"; $env:PIP_FIND_LINKS=(Resolve-Path offline\wheels).Path
yolo settings sync=False     # hoặc: python -c "from ultralytics import settings; settings.update({'sync': False})"
```

`HF_HUB_OFFLINE` để transformers báo lỗi ngay thay vì treo chờ mạng; `PIP_NO_INDEX` chặn pip gọi mạng; `sync=False` tắt thống kê / kiểm phiên bản Ultralytics.

**CVAT offline:**

```powershell
docker load -i offline\cvat-images.tar
Expand-Archive offline\cvat-repo.zip -DestinationPath . -Force
cd cvat
docker compose up -d --pull never
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

`--pull never` cần Compose ≥ 2.24; bản cũ: image đã load thì compose không kéo, nếu vẫn báo kéo thì tạo `docker-compose.override.yml` với `pull_policy: never` cho `cvat_server`, `cvat_ui` (và service khác bị báo). **B1/B2 serverless không chạy offline**; đường A (offline → import) và B3 chạy được.

**Kiểm tra sẵn sàng (không cần mạng):**

```powershell
python docs/day08_setup.py check
python scripts/check_env.py --cvat http://localhost:8080
```

Hai hiểu lầm thường gặp: `[LỖI] không kết nối được CVAT` khi CVAT chưa bật là **không phải lỗi mạng**; `check` chỉ liệt kê `models/*.pt` nên **không** thấy `mobileclip2_b.ts` — muốn chắc thì nhìn dòng `YOLOE gõ chữ … [ OK ]`. Máy chỉ cài `requirements-local.txt`: dùng `check --light`.

### 9.4 Bảng bộ tải ↔ cờ lệnh

| Đã tải bằng | Có trong `models/` | Cờ dùng |
|---|---|---|
| `download_models.py --set core` | `yolo26m.pt`, `yoloe-26s-seg.pt`, `sam2.1_b.pt` | `--size m`, `--sam sam2.1_b.pt` |
| `download_models.py --set small` | `yolo26n.pt`, `yoloe-26s-seg.pt`, `sam2.1_t.pt` | `--size n --imgsz 640`, `--sam sam2.1_t.pt` |
| `day08_setup.py --profile 8gb` | `yolo26n.pt`, `yoloe-26s-seg.pt`, `mobileclip2_b.ts`, `mobile_sam.pt`, `sam2.1_t.pt` | `--size n`, `--sam sam2.1_t.pt` / `mobile_sam.pt` |
| `day08_setup.py --profile 16gb` | như trên + `yolo26s.pt` | `--size n`/`s`, `--sam sam2.1_b.pt` |

### 9.5 Máy quá yếu, không chạy được model

- Cỡ nhỏ: `--size n --imgsz 640`, `--sam sam2.1_t.pt`/`mobile_sam.pt`.
- Hoặc dùng **cache chạy sẵn**: `data/cache/yolo26m_all_c005.coco.json` → `work/prelabel/y26_all/pre.coco.json`; `data/cache/yoloe26s_three_wheeler_all_c005.coco.json` → `work/prelabel/yoloe_all/pre.coco.json`; ghi “dùng cache” vào REPORT mục 8.

## 10. Kiểm chứng offline (đã chạy thật)

Cách chắc nhất: rút mạng / tắt Wi-Fi rồi chạy mục 9.3. Không muốn rút mạng: chặn bằng proxy chết cho riêng phiên PowerShell (`localhost` vẫn qua được):

```powershell
$env:HTTP_PROXY="http://127.0.0.1:9"; $env:HTTPS_PROXY="http://127.0.0.1:9"
$env:NO_PROXY="localhost,127.0.0.1,::1"
```

Đã chạy với mạng chặn trên máy Windows 11, Python 3.12.10, 16 GB RAM, RTX 3050 (torch bản CPU):

| Lệnh | Kết quả |
|---|---|
| `python docs/day08_setup.py check` | 11 dòng `[ OK ]`: Python 3.12.10, numpy, cv2, PIL, matplotlib, yaml, torch 2.15.0+cpu, ultralytics 8.4.215, clip, `models/`, YOLO26n **5 box/0,5 s**, YOLOE **5 box/1,2 s**, MobileSAM **mask 260 716 px/1,3 s**; chỉ `[LỖI] CVAT` vì CVAT chưa bật |
| `python scripts/check_env.py` | `SẴN SÀNG.` (WARN: không GPU vì torch CPU; chưa đăng nhập HF) |
| `prelabel.py --model yolo26 --size m --images img_011.jpg` | 30 box (person=18, motorbike=8, car=3, truck=1), 4,1 s/ảnh CPU |
| `prelabel.py --model yoloe --classes three_wheeler --images img_011.jpg` | 1 box `three_wheeler`, 0,7 s/ảnh CPU — YOLOE chạy được ⇒ `mobileclip2_b.ts` + `clip` đủ |
| `prelabel.py --model yolo26 --segmenter sam --sam sam2.1_b.pt` | 21 box, **21 mask**, 9,6 s/ảnh CPU |
| venv sạch + `pip install --no-index --find-links offline\wheels` | cài trọn bộ + `clip`; `IMPORT OK torch 2.15.0+cpu ultralytics 8.4.215` |

Không lệnh nào in `Downloading`/`urlopen` ⇒ đường offline đủ. (CVAT chưa cài image Docker trong lần kiểm này; phần CVAT theo mục 8.4 / 9.3.)

## 11. Gỡ rối (gộp online + offline)

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` (hoặc numpy, cv2…) | chưa kích hoạt venv / chưa cài | `source .venv/bin/activate` hoặc `.\.venv\Scripts\Activate.ps1`; `pip install -r requirements.txt` |
| `No module named 'clip'` / lỗi `git+https://github.com/ultralytics/CLIP.git` | thiếu git hoặc bị chặn GitHub | online: cài git rồi chạy lại; offline: `pip install --no-index --find-links offline\wheels clip` |
| Tải trọng số chậm / đứt giữa chừng | mạng | chạy lại (script bỏ qua file đã có); xin file từ giảng viên đặt vào `models/` |
| `prelabel.py` in `Downloading https://…` rồi lỗi `urlopen`/`URLError` | thiếu file `.pt` trong `models/` | đặt đúng tên vào `models/` hoặc `--weights <đường dẫn>`; offline chép từ `offline\models\` |
| YOLOE chạy rất lâu / treo / lỗi lần đầu | đang tải `mobileclip2_b.ts` 254 MB | online: chờ hoặc tải tay; offline: `Test-Path models\mobileclip2_b.ts` phải `True` |
| `OSError: We couldn't connect to 'https://huggingface.co'`, `LocalEntryNotFoundError` | chưa có snapshot HF | `HF_HUB_OFFLINE=1` + `--weights offline\hf\<tên>`; hoặc copy cache |
| `Cannot access gated repo … facebook/sam3` | chưa được cấp quyền / chưa login | xin quyền + `hf auth login`; không được thì ghi `skipped` |
| `AttributeError: Sam3Model` | transformers quá cũ | `pip install -U "transformers>=5.0"` |
| `MPS backend out of memory` / lỗi dtype trên Mac | MPS | `--device cpu` (OWLv2 tự CPU); `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `CUDA out of memory` | ảnh/model lớn | `--size s`, `--imgsz 640`, đóng app khác dùng GPU |
| `check_env.py` báo Python 3.14+ / thiếu wheel | Python quá mới | tạo lại venv bằng 3.12 |
| pip `--no-index` không tìm thấy `torch` | wheel khác OS/Python | tạo lại wheelhouse trên đúng máy đích |
| `torch.cuda.is_available()` = False dù có GPU | đã cài wheel CPU | cài lại theo index `cu124` (mục 8.2/9.2) hoặc dùng `--device cpu` |
| `docker compose up` báo thiếu image / `pull access denied` | chưa `docker load`, tag lệch, hoặc máy offline | `docker load -i offline\cvat-images.tar`; `docker compose up -d --pull never`; repo `cvat/` đúng tag `v2.81.0` |
| `check_env.py`/`check` báo lỗi CVAT | CVAT chưa chạy (không phải lỗi mạng) | `cd cvat; docker compose up -d`; kiểm cổng 8080 bị chiếm |
| `check_env.py` FAIL `git` | chưa cài git | offline không cần git; chỉ cần khi fork/push hoặc YOLOE tự cài CLIP (đã có wheel) |
| Nạp COCO vào CVAT nhưng **không thấy box** | tên ảnh trong Task khác tên trong file | tạo Task từ **đúng** `work/tasks/...` |
| Nạp COCO báo “label … not found” | Project còn bộ nhãn cũ | tạo lại Project từ `cvat/labels_generated.json` |
| Hai bộ box chồng nhau | nạp 2 lần (CVAT cộng thêm) | Actions → Remove annotations → nạp lại 1 lần |
| Mỗi vật có cả rectangle lẫn polygon (task seg) | nạp COCO thay vì XML | `coco_to_cvat_xml.py` rồi nạp **CVAT 1.1** |
| Thuộc tính trống sau export | CVAT chưa lưu | mở box, chọn lại thuộc tính, lưu, export lại; hoặc đặt `default_value` ở Project |
| `validate_labels` báo “lớp lạ '?category_id=…'” | sai format export | export lại **COCO 1.0** |
| `validate_labels` báo thiếu ảnh | Task không đúng ảnh | Task phải chứa **đúng** ảnh trong `work/tasks/<tên>/` |
| `review_stats` báo “file pre thiếu ảnh” / “thiếu ảnh có bẫy” | `--pre` sai file | `--pre` phải là `pre_traps.coco.json` (đủ batch+gold, không dùng `pre_traps_import`) |
| Bắt bẫy rất thấp dù thấy mình duyệt kỹ | nạp lại pre-label sau khi sửa hoặc export sai Task | kiểm `review_stats.md` phần bẫy lọt; xem lại quy trình |
| `threshold_sweep` báo “không có vật <lớp> trong ảnh gold” | bình thường với lớp ít vật | giữ ngưỡng mặc định 0.3, nói rõ trong báo cáo |
| `prelabel.py` ra 0 box | ánh xạ nhãn sai / ngưỡng cao / `--classes` sai | đọc dòng “Tên thô KHÔNG có trong ánh xạ”; hạ `--conf` |
| `qc_review.csv` bị báo “chưa kiểm” | còn `error_type` trống | điền `khong` hoặc loại lỗi cho mọi dòng |
| `check_submission.py` báo `<<...>>` | còn chỗ chưa điền | điền STUDENT/REPORT/guideline |
| `check_submission.py` báo `time_log.csv thiếu bước` | cột `step` đang ghi số (1,2,3) hoặc tên khác | đổi `step` thành `gold_manual` và `batch_review`; cột `minutes` ghi số phút |
| `check_submission.py` báo `taxonomy.rules bị đổi` | đã sửa mục rules | khôi phục đúng 10 giá trị ở mục 2.2 |
| `traps_key.json: student không khớp MSHV` | tạo bẫy sai MSHV | chạy lại `inject_traps.py --student <MSHV đúng>` |

## 12. Hỏi–đáp nhanh (fact sheet)

- **36 ảnh, chia thế nào?** gold 4 (001, 012, 020, 036) · pilot 8 (002, 007, 011, 015, 016, 019, 030, 032) · batch 24 · seg 8 con của batch (003, 004, 009, 010, 013, 017, 023, 031).
- **Bao nhiêu lớp?** 8, gồm `ignore_region`; 7 lớp vật: person, bicycle, motorbike, car, bus, truck, three_wheeler.
- **Prompt mặc định cho xe ba bánh?** `tuk-tuk`, `cyclo`, `cargo tricycle` → `three_wheeler`.
- **Model nào cho `three_wheeler`?** YOLOE (open-vocab); YOLO26 gọi tuk-tuk là `car`/`truck` — đổi ngưỡng **không** sửa được lỗi này.
- **SAM có bắt buộc?** Có. Chọn SAM 1 (`mobile_sam.pt`, `sam_b.pt`), SAM 2.1 (`sam2.1_t/s/b/l.pt`, khuyến nghị), hoặc SAM 3 (gated, `--model sam3`). Phải ghi phiên bản vào `model_log` + REPORT.
- **Ngưỡng accepted?** IoU ≥ 0.9 (mask 0.85); ghép khi IoU ≥ 0.5.
- **Bắt bẫy tối thiểu?** 80% = 10/12.
- **Ngưỡng Release?** ảnh có lỗi ≤ 2%; kèm điều kiện không lỗi hệ thống, bắt bẫy đạt, (nếu đo) nhất quán đạt.
- **Box tối thiểu?** cao ≥ 20 px; nhỏ hơn → `ignore_region` cho cụm.
- **Chọn ngưỡng thế nào?** ngưỡng cao nhất ≤ 0.5 mà recall lớp còn ≥ 0.9; lớp < 5 vật trong gold dùng mặc định 0.3.
- **Bao nhiêu ảnh mỗi Task?** gold-manual 4 · batch-review 24 · pilot/seg-review/selfcheck-blind 8.
- **Bao nhiêu commit?** ≥ 4, theo từng Part.
- **Báo cáo ở đâu, mấy mục?** `submission/REPORT.md`, 8 mục; `check_submission.py` yêu cầu ≥ 7 mục `##`, có chữ Release/Hold/Rework, có nhắc SAM.
- **Mấy quyết định calibration?** ≥ 3 dòng, có ≥ 1 `loi_may` và ≥ 1 `cau_hoi_mo`.
- **Mấy ảnh mask?** ≥ 3.
- **Mấy dòng `model_log`?** ≥ 4, ≥ 3 model khác nhau, có ít nhất 1 dòng SAM.
- **Tỉ lệ lỗi QC đếm thế nào?** theo đối tượng (đối tượng lỗi / đối tượng mẫu) và theo ảnh (ảnh có lỗi / ảnh mẫu); luôn ghi mẫu số.
- **Mẫu QC gồm gì?** tối đa 3 ảnh `can_review` nhiều nhất + 5 ảnh ngẫu nhiên (seed 42), tối đa 8 ảnh.
- **`added` chia cho gì?** đối tượng **sau duyệt** (≈ 1 − recall của máy); accepted/edited/deleted chia cho **box máy**.
- **Accepted cao + bắt bẫy thấp nghĩa là gì?** thiên kiến tự động hóa: đang bấm Accept chứ chưa kiểm.
- **Script kiểm được gì?** tên file, lớp, box trong ảnh, đủ ảnh, accepted/edited/deleted/added, lệch giữa 2 bộ nhãn, số đo QC. **Không** kiểm: box ôm sát, lớp đúng thực tế, vật cả hai cùng bỏ sót, có theo checklist không.
- **Gold vì sao phải gán trước?** để không bị “neo” bởi nhãn máy; gold là chuẩn đo máy. Gold nhiễm ⇒ ghi thật vào REPORT mục 8.
- **Vì sao gold không nằm trong Task review?** vừa gán tay xong, duyệt lại ngay là đo trí nhớ chứ không đo thiên kiến.
- **Quyết định HOLD/REWORK/RELEASE?** HOLD nếu guideline chưa rõ / dữ liệu hỏng; REWORK nếu vượt ngưỡng 2% / lỗi hệ thống / bắt bẫy < 80% / nhất quán kém; còn lại RELEASE.
- **0 lỗi trong 10 ảnh chứng minh đạt 2% chưa?** Chưa — quy tắc 3: tỉ lệ thật có thể tới ~3/n (0/10 ⇒ ~30%).
- **Offline cần chuẩn bị gì?** wheelhouse, trọng số (nhớ `mobileclip2_b.ts`), snapshot HF nếu nâng cao, image Docker CVAT + repo `cvat/`.
- **Offline chạy được gì?** mọi thứ trừ `pip install` từ mạng, tải model, `git clone`, Colab, B1/B2 serverless.
- **Tên repo?** `K5-L23-DAY08-[MSHV]-[HoTen]-AIPrelabel`, public.
- **Nộp ở đâu?** LMS `https://aithucchien.vinuni.edu.vn` hoặc kênh giảng viên chỉ định.
- **Điểm tối đa?** 100 + tối đa 5 điểm thưởng bài nâng cao.
- **Bài nâng cao gồm gì?** Grounding DINO/OWLv2, SAM 3, `cvat-cli`, pilot riêng để chọn ngưỡng, mask 8 ảnh seg, gán lại mù (`self_consistency`), mở rộng GPU (`--size l/x`, `imgsz 1280`).
- **Cache dự phòng?** `data/cache/` (conf 0.05, 36 ảnh) — dùng khi không chạy được model; ghi rõ trong REPORT mục 8.

## 13. Nguồn, giấy phép, trạng thái repo

- **Nguồn chuẩn trong repo** (khi mâu thuẫn, các file này thắng): `README.md`, `GUIDE.md`, `RULES.md`, `RUBRIC.md`, `SUBMISSION.md`, `CHECKPOINTS.md`, `docs/*.md`, `submission/taxonomy.json`, `guideline/*.md`, code `scripts/*.py`, `data/split.csv`.
- **Giấy phép:** ảnh từ Wikimedia (CC BY / CC BY-SA / CC0) — giữ ghi công `data/ATTRIBUTION.csv`; ảnh CVAT trong `docs/assets/cvat/` từ tài liệu CVAT (MIT). Model có giấy phép riêng (Ultralytics YOLO: AGPL-3.0 hoặc bản doanh nghiệp; SAM 3 có điều khoản truy cập) — bài dùng cho **học tập**, dùng thương mại phải kiểm lại.
- **Trạng thái repo tại snapshot 2026-11-18** (kiểm lại bằng `python scripts/check_submission.py`):

| Hạng mục | Trạng thái |
|---|---|
| Môi trường | `.venv` đã cài (torch 2.15.0+cpu, ultralytics 8.4.215, transformers 5.23.0, clip 1.0, OpenCV 5.0.0) |
| `models/` | đủ 7 file ≈ 587 MB: yolo26n/m, yoloe-26s-seg, mobileclip2_b.ts, mobile_sam, sam2.1_t/b |
| `offline/wheels/` | 67 wheel ≈ 330 MB (bản CPU), đã gitignore |
| CVAT | đã cài Docker; **image CVAT chưa chuẩn bị**, CVAT chưa chạy |
| `work/` | có `smoke/`, `prelabel/t_y26`, `prelabel/t_y26_c05`; `runs.csv` r01–r03 |
| `submission/` | `REPORT.md`, `model_log.csv`, `time_log.csv`, `calibration_decisions.csv` còn là khuôn/placeholder; `labels/gold_manual.json` mới có; chưa có `batch_reviewed.json`, `masks/`, `qc/` output, `release/` |
| Việc còn lại | Part 1 (kiểm/đo gold), 2 (chạy đủ model + điền log), 3–6, rồi `check_submission.py` |

- **Giới hạn tài liệu này:** là bản tóm tắt + hướng dẫn, không thay thế việc đọc file gốc; số liệu “đã kiểm chứng” đúng với cấu hình ghi ở mục 10, máy khác có thể khác về tốc độ/dung lượng.
