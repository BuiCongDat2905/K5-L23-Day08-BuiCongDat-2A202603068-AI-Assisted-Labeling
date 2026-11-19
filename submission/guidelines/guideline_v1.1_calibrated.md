# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603068 - Bùi Công Đạt

> Chỉ phần **thay đổi** so với v1.0. Áp dụng cùng v1.0; chỗ nào v1.0 mơ hồ thì mục dưới đây thắng.
> Mỗi ca khó có ảnh ví dụ và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định v1.1 | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Có **pê-đan và khung mảnh** thì `bicycle` (kể cả xe đạp điện có pê-đan); chỉ có vỏ xe tay ga, không thấy pê-đan thì `motorbike`. Không chắc: chọn `bicycle` + tick `can_review`. | img_031 | CAL-04 |
| 2 | Khối xe máy đỗ dày đặc **gần** camera: từng xe hay `ignore_region`? | Vẫn từng xe nếu tách được từng cá thể; chỉ dùng `ignore_region` khi cụm ≥ 5 vật mà phần nhìn thấy của mỗi vật **cao < 20 px** (giữ nguyên 4.1.6). Cụm gần camera tách được thì **không** dùng `ignore_region`. | img_024, img_029 | CAL-08 |
| 3 | Hàng nhô ra (tre, rổ, mâm chuối) có thuộc box xe không? | Phần hàng **gắn cố định** vào xe nằm trong box xe; hàng nhô ra **hơn 20% chiều dài xe** thì không kéo box theo, nhưng không tách thành box riêng. | img_003, img_008 | CAL-03 |
| 5 | Kính chắn gió / cửa sổ xe có thuộc mask xe không? | Có. Kính là một phần vỏ xe: mask phủ cả kính dù nhìn xuyên qua thấy nội thất / người. Người nhìn thấy **qua kính** vẫn là box `person` riêng nếu còn thấy ≥ 20% (4.1.2). | img_013 | CAL-07 |
| 7 | Có đặt mức tin cậy tối thiểu để chấp nhận box máy? | **Không** dùng `conf` làm tiêu chí chấp nhận. `conf` chỉ để xếp thứ tự duyệt. Ngưỡng theo lớp chỉ dùng **trước** khi nạp vào CVAT (`three_wheeler=0.15`, còn lại 0.3). | img_021 | CAL-09 |

Trong 7 câu, **5 câu chốt được** (1, 2, 5, 7 và một phần 3) và câu 4 chốt một phần; câu 6 (box máy chồng nhau trong ảnh đêm) **chưa chốt** — xem mục 2.

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới |
|---|---|---|---|---|
| img_004 | Hàng 4 tuk-tuk đỗ ven đường, ban ngày | YOLO26 gọi là `car` 0.37 / 0.34 / 0.30; YOLOE bắt đúng `three_wheeler` 0.93 | Luôn gán `three_wheeler` cho tuk-tuk / xích lô / ba gác | Mục 3 mới: **ưu tiên model theo lớp** khi gộp pre-label — `three_wheeler` lấy từ YOLOE, 6 lớp COCO lấy từ YOLO26; file YOLOE đứng trước trong `merge_prelabels.py` |
| img_010 | Hàng xích lô có mui chạy nối đuôi | một xe bị tách **2 box** `three_wheeler` (mui + thân), phần khung sau thành `bicycle` 0.50 và `motorbike` | Một vật **một** box, gồm mui / thùng / khung gắn liền | Mục 4.1.1 bổ sung: **cấm tách box theo bộ phận**; kiểm trùng trước export |
| img_035 | Xích lô đỗ, khách ngồi trong | 2 box `three_wheeler` trên cùng xe (một box chỉ ôm mui); 2 box `person` gần trùng khít trên một người | Giữ một box xe; giữ box người ôm sát hơn | Mục 4.1.1 bổ sung: hai box **cùng lớp, IoU > 0.8** là trùng, phải gộp |
| img_018, img_033 | Xe khách / xe van trắng ở xa, ngược sáng | gọi `three_wheeler` 0.3–0.5 (kể cả trên xe van 4 bánh img_034) | Phép thử số bánh: thấy **cặp bánh kép** hoặc thân dài đều → `bus` / `truck` / `car`, không phải `three_wheeler` | Mục 4.3 bổ sung: ba_wheeler chỉ khi thấy rõ 3 bánh **hoặc** khung mui đặc trưng |
| img_024 | Ảnh đêm, phơi sáng dài, đèn xe thành vệt | `car` / `person` điểm 0.3–0.5 đặt trên **vệt đèn** và bóng đổ | Box trên vệt đèn, bóng, phản chiếu: xoá | Mục 4.1.7 nhắc lại + ghi chú: ảnh phơi sáng dài, chỉ tin box có **thân xe / thân người** nhìn thấy |
| img_028, img_029 | Ảnh đêm, box máy chồng nhau nhiều (câu hỏi mở số 6) | nhiều `car` / `motorbike` chồng lấn trên cùng một vùng tối | **Chưa chốt.** Tạm giữ nguyên và tick `can_review`, không gộp | Mục 6.6 vẫn **mở**, cần thêm ảnh đêm có ≥ 2 người gán độc lập để so |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt 5/7 câu hỏi mở (1, 2, 5, 7, một phần 3), chốt một phần câu 4; thêm **mục 3 ưu tiên model theo lớp**; thêm quy tắc **một vật một box, cấm tách box bộ phận**; thêm **kiểm trùng IoU > 0.8** trước export; thêm **phép thử số bánh** cho three_wheeler; ghi chú ảnh phơi sáng dài; giữ mở câu 6 | CAL-01 (lỗi lớp tuk-tuk), CAL-02 (tách box bộ phận), CAL-03 (hàng nhô), CAL-04 (xe đạp điện), CAL-05 (bẫy sai lớp lọt), CAL-06 (box trùng), CAL-07 (kính chắn gió), CAL-08 (cụm xe gần camera), CAL-09 (ngưỡng conf) |
