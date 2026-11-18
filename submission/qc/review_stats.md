# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 519; đối tượng sau duyệt: 473; ngưỡng accepted: IoU >= 0.9
- Đã **loại 21 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 456 | 456/519 = 87.9% | 519 box máy |
| edited (sửa) | 8 | 8/519 = 1.5% | 519 box máy |
| deleted (xoá) | 50 | 50/519 = 9.6% | 519 box máy |
| added (vẽ thêm) | 9 | 9/473 = 1.9% | 473 đối tượng sau duyệt |

Trong số deleted + added, **3 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 6.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 254 | 1 | 9 | 1 |
| bicycle | 5 | 0 | 3 | 1 |
| motorbike | 126 | 1 | 10 | 1 |
| car | 46 | 0 | 8 | 0 |
| bus | 3 | 0 | 0 | 1 |
| truck | 5 | 0 | 1 | 0 |
| three_wheeler | 17 | 6 | 19 | 5 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 76 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 55 | 19 | 21 | 74.3% (KTC 95%: 63–83%) | 72.4% (KTC 95%: 61–81%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 4/12 = 33.3% (KTC 95%: 14–61%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - loose: 0/3
  - wrong_class: 0/3
  - missing: 1/3
  - extra: 3/3
  - lọt: img_003.jpg (loose), img_021.jpg (wrong_class), img_022.jpg (loose), img_006.jpg (loose), img_018.jpg (wrong_class), img_034.jpg (missing), img_033.jpg (wrong_class), img_025.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
