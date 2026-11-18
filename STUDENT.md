# Thông tin học viên

- Họ và tên: Bùi Công Đạt
- MSHV: 2A202603068
- Lớp (L2 / L3 / khác): khác — K5-L23
- Tài khoản GitHub: BuiCongDat2905
- Repo bài nộp (URL): https://github.com/BuiCongDat2905/K5-L23-Day08-BuiCongDat-2A202603068-AI-Assisted-Labeling
- Commit chốt bài (hash): 0000000
- Máy: Windows 11, CPU only (torch 2.15.0+cpu, không CUDA), 16 GB RAM
- CVAT (Docker): v2.81.0 (Docker 29.8.1)

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 456 / 8 / 50 (mẫu số: 519 box máy); added: 9 (mẫu số: 473 đối tượng)
- So với gold: precision / recall của **máy** 74.3% / 72.4%
- Bắt bẫy: 4/12 = 33.3%
- Thời gian: gán tay 10.5 phút/ảnh (42 phút / 4 ảnh); duyệt có pre-label 1.0 phút/ảnh (24 phút / 24 ảnh, **do AI duyệt** — xem `REPORT.md` mục 8)
- Lỗi QC: 60.0% theo ảnh (3/5 ảnh), 3.5% theo đối tượng (3/86 đối tượng) — vòng 2, sau khi sửa `qc_fixed.json`
- **Quyết định:** Rework — ảnh có lỗi 60% > 2%, bắt bẫy 33% < 80%, nên lô nhãn chưa đáng tin để Release
