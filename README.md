# DMAE-VAD: Decoupled Dual-Stream Masked Autoencoder for Weakly-Supervised Video Anomaly Detection

**Tác giả:** Huỳnh Quang Vũ (MSSV: 2351010247)  
**Giảng viên hướng dẫn:** TS. Lê Viết Tuấn  
**Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Mở TP. Hồ Chí Minh  

---

## 📌 1. Bối cảnh & Động lực Khoa học (Motivation)

Trong đồ án ngành trước đây (RUPA-DSA) và các mô hình tái tạo truyền thống:
1. **Bế tắc trong việc nhặt khung hình bình thường (Pseudo-label Dilemma):** 
   - Để huấn luyện bộ nhớ nguyên mẫu (Prototypes), mô hình buộc phải phân ngưỡng để chọn ra khung hình nào là "bình thường" trong một video bất thường.
   - Dùng **Top-K** thì cứng nhắc và thiên lệch. Dùng **Otsu** thì thất bại thảm hại trên dữ liệu camera an ninh (CCTV) vì phân phối dữ liệu là đơn đỉnh (unimodal).
2. **Sự phụ thuộc vào Warm-Start:** Do gradient từ nhánh tái tạo gây nhiễu loạn cho bộ trích xuất đặc trưng thời gian khi huấn luyện từ đầu.

### 💡 Ý tưởng đột phá: Kết hợp Hướng 1 (Decoupled Dual-Stream) và Hướng 3 (Temporal MAE):
- **Triệt tiêu hoàn toàn nhu cầu phân ngưỡng Top-K / Otsu:** Nhánh tái tạo chuyển sang bài toán **Tự giám sát (Self-Supervised Masked Snippet Inpainting)**. Mô hình ngẫu nhiên che khuất 50% các đoạn video và học cách suy luận/tái tạo lại các đoạn bị che từ bối cảnh xung quanh.
- **Nguyên lý phát hiện dị thường tự nhiên:**
  - *Hành vi bình thường (đi bộ, xe chạy đều):* Có tính liên tục thời gian cao $\rightarrow$ Mô hình dễ dàng đoán trước đoạn bị che $\rightarrow$ **Sai số tái tạo cực thấp ($S_{rec} \to 0$)**.
  - *Hành vi bất thường (đánh nhau, nổ, cướp giật):* Diễn ra gãy khúc, đột ngột, hỗn loạn $\rightarrow$ Bối cảnh xung quanh không thể dự đoán được $\rightarrow$ **Sai số tái tạo vọt lên đỉnh ($S_{rec} \to 1$)**.
- **Phân ly đạo hàm tuyệt đối (Decoupled Stream):** Hai luồng hoạt động song song, không chia sẻ trọng số nội bộ, ngăn chặn 100% hiện tượng xung đột gradient khi huấn luyện từ con số 0 (Train from Scratch).

---

## 🏛️ 2. Kiến trúc Tổng thể (Architecture)

```
                              Video Input (Snippets)
                                        │
                              [CLIP Vision (Frozen)]
                                        │
                               CLIP Features X [B, T, 512]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼ (Luồng A: Phân biệt)                                ▼ (Luồng B: Tự giám sát T-MAE)
    [Multi-Scale Dilated TCN]                                [Random / Dual-Half Masking]
  (Dilation 1, 2, 4 + Skip)                                                │
             │                                               [Transformer Inpainting Autoencoder]
             ▼                                                             │
     Classifier Head                                            Inpainted Features X_rec
             │                                                             │
             ▼                                                             ▼
    Detection Score S_det                                      Recon. Anomaly Score S_rec
      (Binary Output)                                           ((1 - cos(X, X_rec)) / 2)
             │                                                             │
             └──────────────────────────┬──────────────────────────────────┘
                                        ▼
                             [Adaptive Evidential Gate]
                              g = σ(MLP([S_det, S_rec, |S_det - S_rec|, S_det * S_rec]))
                              S_final = (1 - g) * S_det + g * S_rec
                                        │
                                        ▼
                               Final Anomaly Scores
```

### Điểm nhấn thuật toán ở pha kiểm thử (Deterministic Dual-Half Inpainting Pass):
Để đánh giá kiểm thử công bằng và không phụ thuộc vào việc bốc ngẫu nhiên mặt nạ che:
* **Pass 1:** Che toàn bộ các frame ở vị trí lẻ ($t = 1, 3, 5, ...$), dùng các frame chẵn xung quanh để tái tạo frame lẻ.
* **Pass 2:** Che toàn bộ các frame ở vị trí chẵn ($t = 0, 2, 4, ...$), dùng các frame lẻ xung quanh để tái tạo frame chẵn.
* Ghép hai nửa lại: **100% các đoạn video đều được tái tạo một cách khách quan thuần túy từ bối cảnh lân cận** mà không có bất kỳ tính ngẫu nhiên nào!

---

## 📂 3. Cấu trúc Thư mục

```
DMAE-VAD/
├── README.md               # Tài liệu khoa học & hướng dẫn thực nghiệm
├── requirements.txt        # Các thư viện phụ thuộc
├── dmae_final.ipynb        # Kaggle Notebook hoàn chỉnh cho 2x Tesla T4
├── list/                   # Ground Truth và CSV splits (UCF & XD)
├── models/                 # Checkpoints (.pth) tự động lưu
├── scripts/
│   ├── run_ucf.bat / run_ucf.sh
│   └── run_xd.bat / run_xd.sh
└── src/
    ├── model.py            # Multi-Scale TCN + Temporal MAE Inpainter + Gated Fusion
    ├── losses.py           # Top-K MIL + Masked Inpainting MSE + Contrastive Rec Margin
    ├── dataset.py          # Unified Dataset Loader
    ├── tools.py            # Snippet uniform sampling & padding
    ├── options.py          # Cấu hình siêu tham số
    ├── ucf_train.py        # Huấn luyện UCF-Crime từ đầu (Dual-GPU)
    ├── ucf_test.py         # Kiểm thử UCF (Dual-Half Inpainting)
    ├── xd_train.py         # Huấn luyện XD-Violence từ đầu (Dual-GPU)
    ├── xd_test.py          # Kiểm thử XD (Dual-Half Inpainting)
    ├── ucf_detectionMAP.py # Frame/segment mAP evaluation
    └── xd_detectionMAP.py  # Frame/segment mAP evaluation
```

---

## 🚀 4. Hướng dẫn Huấn luyện & Kiểm thử

### 4.1 Huấn luyện từ đầu trên UCF-Crime:
```bash
python src/ucf_train.py --dataset ucf --batch-size 64 --lr 1e-4 --max-epoch 15 --mask-ratio 0.5
```
*(Hoặc trên Windows: chạy `scripts\run_ucf.bat`, trên Linux/Kaggle: `bash scripts/run_ucf.sh`)*

### 4.2 Kiểm thử trên UCF-Crime:
```bash
python src/ucf_test.py --dataset ucf --model-path models/dmae_vad_ucf_best.pth
```

### 4.3 Huấn luyện từ đầu trên XD-Violence:
```bash
python src/xd_train.py --dataset xd --batch-size 64 --lr 1e-4 --max-epoch 20 --mask-ratio 0.5
```

### 4.4 Kiểm thử trên XD-Violence:
```bash
python src/xd_test.py --dataset xd --model-path models/dmae_vad_xd_best.pth
```

---

## 🔬 5. Bảng So Sánh với các Mô hình Tiền nhiệm

| Đặc tính kỹ thuật | DSANet (AAAI 2026) | RUPA-DSA (Đồ án ngành) | **DMAE-VAD (Nghiên cứu mới)** |
| :--- | :--- | :--- | :--- |
| **Nhánh Tái tạo** | SG-NM (bỏ ở inference) | Prototype Bank (K=16) | **Temporal Masked Autoencoder (TMAE)** |
| **Cơ chế nhặt frame sạch** | Ngưỡng Bottom-M cố định | Otsu / Top-K | **Không cần nhặt (Tự giám sát Inpainting)** |
| **Khả năng tự train from scratch** | Có | Không (bắt buộc Warm-Start) | **100% Độc lập từ con số 0** |
| **Xung đột Gradient** | Cao | Phải khóa cứng DSANet | **Triệt tiêu nhờ Decoupled Dual-Stream** |
| **Đánh giá tái tạo khi test** | Không có | Khoảng cách Prototype | **Dual-Half Inpainting Pass (Bối cảnh lân cận)** |
