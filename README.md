# Datathon 2026 — The Gridbreaker · Vòng 1

Bài nộp Vòng 1 cho cuộc thi **Datathon 2026 — The Gridbreaker** (VinTelligence, VinUniversity).
Bộ dữ liệu mô phỏng hoạt động một doanh nghiệp thời trang TMĐT Việt Nam giai đoạn 04/07/2012 – 31/12/2022.

## 1. Cấu trúc thư mục

```
Submission/
├── data/                       # 14 file CSV gốc do BTC cung cấp
│   ├── customers.csv           # Master
│   ├── geography.csv
│   ├── products.csv
│   ├── promotions.csv
│   ├── orders.csv              # Transaction
│   ├── order_items.csv
│   ├── payments.csv
│   ├── shipments.csv
│   ├── returns.csv
│   ├── reviews.csv
│   ├── sales.csv               # Analytical (train: 2012-07-04 → 2022-12-31)
│   ├── sample_submission.csv
│   ├── inventory.csv           # Operational
│   └── web_traffic.csv
├── notebooks/
│   ├── 01_baseline.ipynb       # Baseline forecasting (seasonal + YoY trend)
│   ├── 02_eda_analysis.ipynb   # Phần 2 — EDA & business insight
│   └── 03_forecasting.ipynb    # Phần 3 — pipeline dự báo cuối (3-tier ensemble)
├── images/                     # Dashboard PNG xuất từ EDA notebook
│   ├── dashboard_2_1.png       # Inventory paradox
│   └── dashboard_2_2.png       # Customer concentration & discount efficiency
├── submission.csv              # File nộp Kaggle (548 dòng, khớp sample_submission)
├── report.pdf                  # Báo cáo NeurIPS 4 trang (TODO)
├── Đề thi Vòng 1.pdf
└── README.md
```

> **Lưu ý**: 13/14 file trong `data/` được gitignore (~120MB). Chỉ `sample_submission.csv` được commit. Cách tải dữ liệu xem mục 3.0 bên dưới.

## 2. Yêu cầu môi trường

- Python ≥ 3.10
- GPU (CUDA) khuyến nghị cho `03_forecasting.ipynb` (Chronos T5-large)

### Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install lightgbm prophet shap
pip install chronos-forecasting --no-deps
pip install einops torch lunardate
```

> Notebook `03_forecasting.ipynb` đã có cell `!pip install …` ở đầu để tự cài khi chạy trên Kaggle / Colab.

## 3. Hướng dẫn chạy lại kết quả

### 3.0. Tải dữ liệu

Dữ liệu gốc (~120MB) không nằm trong repo. Tải từ trang Kaggle competition và giải nén vào `data/`:

```bash
# Cách 1 — Kaggle CLI (khuyến nghị)
pip install kaggle
kaggle competitions download -c datathon-2026-round-1 -p data/
unzip data/datathon-2026-round-1.zip -d data/

# Cách 2 — tải thủ công từ
# https://www.kaggle.com/competitions/datathon-2026-round-1/data
# rồi đặt 14 file CSV vào data/
```

Sau khi giải nén, `data/` cần đủ 14 file CSV như sơ đồ ở mục 1.

### 3.1. Chạy notebooks

Từ thư mục gốc của repo, mở Jupyter và chạy lần lượt:

| Bước | Notebook | Output | Thời gian |
|---|---|---|---|
| 1 | `notebooks/02_eda_analysis.ipynb` | `images/dashboard_*.png` | ~2 phút |
| 2 | `notebooks/03_forecasting.ipynb` | `submission.csv` (root) + `feature_importance.png`, `shap_analysis.png`, `forecast_plot.png` | ~25 phút (GPU) |

`01_baseline.ipynb` là baseline đơn giản dùng để so sánh — không cần chạy để tạo file nộp cuối.

### 3.2. Đường dẫn dữ liệu

Tất cả notebook đọc dữ liệu qua đường dẫn tương đối `../data/` (vì nằm trong `notebooks/`). Riêng `03_forecasting.ipynb` tự phát hiện môi trường:

```python
KAGGLE_DATA = Path('/kaggle/input/datasets/cahoivuotthac/datathon2026')
if KAGGLE_DATA.exists():
    DATA_DIR = KAGGLE_DATA            # Kaggle
    OUT_DIR  = Path('/kaggle/working')
else:
    DATA_DIR = Path('../data')        # Local
    OUT_DIR  = Path('..')
```

### 3.3. Tính tái lập (Reproducibility)

`03_forecasting.ipynb` cố định seed cho toàn bộ stack:

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

LightGBM dùng `seed=42` trong `LGB_PARAMS`; Prophet và Chronos được reset seed trước mỗi `.fit()` / `.predict()`.

## 4. Tóm tắt phương pháp Phần 3

Pipeline 3-tier ensemble dùng ~77 đặc trưng calendar-only (không lag) để dự báo 548 ngày từ 2023-01-01 → 2024-07-01:

```
Tier 1  LGB_blend = 0.60 × Q-Specialist + 0.40 × LGB_base
Tier 2  raw       = 0.10 × Chronos + 0.10 × Ridge + 0.80 × LGB_blend
Tier 3  final     = CR × raw_rev   (CR=1.328)   |   CC × raw_cog   (CC=1.35)
```

- **Đặc trưng**: Fourier annual/weekly/monthly, edge-of-month, regime dummy (pre-2019 / 2019 / post-2019), Tết proximity (lunardate), 6 cửa sổ promo trích từ `promotions.csv`.
- **Trọng số mẫu**: 2014–2018 được upweight (= 1.0) so với 0.01 cho các năm khác để mô hình học cấu trúc mùa vụ ổn định nhất.
- **Q-Specialist**: 4 LGB cho 4 quý × 2 target = 8 mô hình, mỗi specialist boost trọng số ×2 cho quý của mình.
- **Chronos t5-large**: rolling 64 ngày/chunk, 20 samples, level-calibrated theo trung bình 730 ngày gần nhất.
- **Margin fix**: blend COGS theo margin lịch sử mỗi quý (β=0.30) + mean preservation.

## 5. Phần 2 — Insight chính từ EDA

| Chủ đề | Phát hiện | Cơ hội năm 1 |
|---|---|---|
| **Tồn kho (2.1)** | 79.8% SKU dao động giữa stockout & overstock; Streetwear chiếm 81.7% holding cost (~877B); 359 SKU bán dưới giá vốn | Giải phóng 300–500B vốn lưu động + 50–80M GP/năm |
| **Khách hàng (2.2)** | Top 32.9% khách = 75.3% doanh thu (Gini 0.586); discount đang phân bổ đều thay vì theo CLV; ROI Email 9.3× vs Social 5.6× | Thu hồi 600M–1B biên/năm |

→ ~1B+ cải thiện biên năm 1, không cần tăng top-line. Chi tiết & dashboard trong [`notebooks/02_eda_analysis.ipynb`](notebooks/02_eda_analysis.ipynb).

## 6. Liên kết

- **Kaggle competition**: https://www.kaggle.com/competitions/datathon-2026-round-1
- **Đề thi**: [`Đề thi Vòng 1.pdf`](Đề%20thi%20Vòng%201.pdf)
