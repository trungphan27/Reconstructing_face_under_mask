# 🎭 Reconstructing Face Under Mask - ImprovedPremiumGAN

Dự án sử dụng mạng **GAN (Generative Adversarial Network)** để tái tạo phần khuôn mặt bị che bởi khẩu trang. Model được huấn luyện trên dataset khuôn mặt và có khả năng khôi phục lại vùng bị che với chất lượng cao.

---

## 📁 Cấu Trúc Dự Án

```
Reconstructing_face_under_mask/
│
├── ImprovedPremiumGAN/          # Thư mục chính chứa model và code training
│   ├── config.py                # Cấu hình hyperparameters và đường dẫn
│   ├── model.py                 # Kiến trúc Generator (U-Net) và Discriminator (PatchGAN)
│   ├── dataset.py               # Dataset loader và data augmentation
│   ├── loss.py                  # VGG Perceptual Loss
│   ├── train.py                 # Script huấn luyện chính
│   ├── inference.py             # Script dự đoán/inference
│   ├── detect.py                # Phát hiện vùng khẩu trang bằng YOLO
│   ├── utils.py                 # Các hàm tiện ích (SSIM, PSNR, save images)
│   ├── checkpoints/             # Lưu trữ model weights
│   └── results/                 # Kết quả training và log
│
├── dataset/                     # Thư mục chứa dữ liệu
│   ├── with_mask/               # Ảnh khuôn mặt có khẩu trang
│   └── without_mask/            # Ảnh khuôn mặt không khẩu trang (Ground Truth)
│
├── evaluate.py                  # Script đánh giá và vẽ biểu đồ metrics
├── download_dataset.py          # Script kiểm tra dataset
├── requirements.txt             # Danh sách thư viện cần thiết
├── yolov8n.pt                   # Pre-trained YOLO model cho mask detection
└── README.md                    # Tài liệu này
```

---

## 🏗️ Kiến Trúc Model

### 1. Generator - Kiến Trúc U-Net

Generator sử dụng kiến trúc **U-Net** với cơ chế **skip connections** để bảo toàn thông tin chi tiết từ ảnh đầu vào.

```
Input (3, 128, 128)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    ENCODER (Downsampling)                │
├─────────────────────────────────────────────────────────┤
│  DoubleConv(3 → 64)   ──────────────────────────┐       │
│       │ MaxPool2d                                │       │
│       ▼                                          │       │
│  DoubleConv(64 → 128)  ─────────────────────┐   │       │
│       │ MaxPool2d                            │   │       │
│       ▼                                      │   │       │
│  DoubleConv(128 → 256) ────────────────┐    │   │       │
│       │ MaxPool2d                       │    │   │       │
│       ▼                                 │    │   │       │
│  DoubleConv(256 → 512) ───────────┐    │    │   │       │
│       │ MaxPool2d                  │    │    │   │       │
│       ▼                            │    │    │   │       │
├────────────────────────────────────│────│────│───│───────┤
│              BOTTLENECK            │    │    │   │       │
│  DoubleConv(512 → 1024)            │    │    │   │       │
├────────────────────────────────────│────│────│───│───────┤
│                    DECODER (Upsampling)                  │
├────────────────────────────────────│────│────│───│───────┤
│  ConvTranspose2d(1024 → 512)       │    │    │   │       │
│       │ Concat ◄───────────────────┘    │    │   │       │
│  DoubleConv(1024 → 512)                 │    │   │       │
│       │                                  │    │   │       │
│       ▼                                  │    │   │       │
│  ConvTranspose2d(512 → 256)             │    │   │       │
│       │ Concat ◄────────────────────────┘    │   │       │
│  DoubleConv(512 → 256)                       │   │       │
│       │                                       │   │       │
│       ▼                                       │   │       │
│  ConvTranspose2d(256 → 128)                  │   │       │
│       │ Concat ◄─────────────────────────────┘   │       │
│  DoubleConv(256 → 128)                           │       │
│       │                                           │       │
│       ▼                                           │       │
│  ConvTranspose2d(128 → 64)                       │       │
│       │ Concat ◄─────────────────────────────────┘       │
│  DoubleConv(128 → 64)                                    │
│       │                                                   │
│       ▼                                                   │
│  Conv2d(64 → 3) + Tanh()                                 │
└─────────────────────────────────────────────────────────┘
       │
       ▼
Output (3, 128, 128) ∈ [-1, 1]
```

**DoubleConv Block:**

```
Conv2d(3×3) → BatchNorm2d → ReLU → Conv2d(3×3) → BatchNorm2d → ReLU
```

---

### 2. Discriminator - Kiến Trúc PatchGAN

Discriminator sử dụng kiến trúc **PatchGAN** - thay vì đầu ra là một giá trị scalar (real/fake), nó đưa ra một **grid NxN** các xác suất, giúp model focus vào các chi tiết cục bộ (texture, edges).

```
Input (3, 128, 128)
       │
       ▼
┌─────────────────────────────────────┐
│  Conv2d(3 → 64, k=4, s=2)          │  → 64×64×64
│  LeakyReLU(0.2)                     │
├─────────────────────────────────────┤
│  CNNBlock(64 → 128, s=2)           │  → 32×32×128
│  CNNBlock(128 → 256, s=2)          │  → 16×16×256
│  CNNBlock(256 → 512, s=2)          │  → 8×8×512
├─────────────────────────────────────┤
│  Conv2d(512 → 1, k=4, s=1)         │  → 7×7×1
│  Sigmoid()                          │
└─────────────────────────────────────┘
       │
       ▼
Output: 7×7 Probability Grid
```

**CNNBlock:**

```
Conv2d(4×4, padding_mode="reflect") → BatchNorm2d → LeakyReLU(0.2)
```

---

## 📉 Hàm Loss (Loss Functions)

Dự án sử dụng **3 loại Loss** kết hợp để huấn luyện Generator:

### 1. Adversarial Loss (BCE Loss)

```python
L_adv = BCE(D(G(x)), 1)  # Generator cố gắng đánh lừa Discriminator
```

- **Mục đích**: Khiến ảnh sinh ra "giống thật" theo đánh giá của Discriminator
- **Weight**: `LAMBDA_ADV = 1`

### 2. L1 Loss (Pixel-wise Reconstruction)

```python
L_L1 = ||G(x) - y||₁
```

- **Mục đích**: Đảm bảo ảnh sinh ra giống với Ground Truth ở mức pixel
- **Weight**: `LAMBDA_L1 = 100`
- **Tác dụng**: Giữ cấu trúc tổng thể, tránh blur

### 3. VGG Perceptual Loss

```python
L_VGG = Σ ||VGG_i(G(x)) - VGG_i(y)||²
```

- **Mục đích**: So sánh features ở nhiều mức độ trừu tượng (texture, edges, semantic)
- **Weight**: `LAMBDA_VGG = 10`
- **Các layer VGG19 được sử dụng**:
  - `relu1_1` (low-level: edges)
  - `relu2_1` (textures)
  - `relu3_1` (patterns)
  - `relu4_1` (semantic features)
  - `relu5_1` (high-level semantic)

### Tổng Loss của Generator:

```python
L_G = LAMBDA_ADV × L_adv + LAMBDA_L1 × L_L1 + LAMBDA_VGG × L_VGG
```

### Discriminator Loss:

```python
L_D = (BCE(D(y), 1) + BCE(D(G(x)), 0)) / 2
```

---

## ⚙️ Optimizer

### Adam Optimizer

Cả Generator và Discriminator đều sử dụng **Adam Optimizer** với cấu hình:

| Parameter     | Generator | Discriminator |
| ------------- | --------- | ------------- |
| Learning Rate | 0.0002    | 0.0002        |
| Beta1         | 0.5       | 0.5           |
| Beta2         | 0.999     | 0.999         |

**Lý do chọn Beta1 = 0.5:**

- Giảm momentum so với mặc định (0.9)
- Giúp ổn định training GAN, tránh oscillation

---

## 🛠️ Kỹ Thuật Training

### 1. Learning Rate Scheduler

```python
ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

- Giảm LR đi 50% nếu loss không cải thiện sau 5 epochs
- Giúp fine-tune model ở giai đoạn cuối

### 2. Synthetic Mask Generation

Thay vì sử dụng ảnh có mask thật, dự án tạo **mask tổng hợp** trong quá trình training:

```python
# Vùng mask: 50%-95% chiều cao, 15%-85% chiều rộng
mask_y_start = int(h * 0.50)
mask_y_end = int(h * 0.95)
mask_x_start = int(w * 0.15)
mask_x_end = int(w * 0.85)

# Đặt vùng mask thành màu đen (-1 trong range [-1, 1])
masked_image[:, mask_y_start:mask_y_end, mask_x_start:mask_x_end] = -1.0
```

### 3. Weight Initialization

```python
# Normal distribution với mean=0, std=0.02
nn.init.normal_(m.weight.data, 0.0, 0.02)
```

- Áp dụng cho: Conv2d, ConvTranspose2d, BatchNorm2d

### 4. Checkpoint & Resume Training

- Lưu checkpoint mỗi epoch
- Có khả năng resume training từ checkpoint cuối cùng
- Lưu cả `state_dict` và `optimizer` state

### 5. Image Normalization

```python
# Input/Output range: [-1, 1]
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

### 6. Real-time Metrics Tracking

- **SSIM** (Structural Similarity Index): Đánh giá độ tương đồng cấu trúc
- **PSNR** (Peak Signal-to-Noise Ratio): Đánh giá chất lượng tái tạo
- Kết quả được log ra file CSV

---

## 📊 Metrics Đánh Giá

| Metric       | Mô tả                                                   | Giá trị tốt                     |
| ------------ | ------------------------------------------------------- | ------------------------------- |
| **SSIM**     | Độ tương đồng cấu trúc (luminance, contrast, structure) | Càng gần 1 càng tốt             |
| **PSNR**     | Tỷ lệ tín hiệu trên nhiễu (dB)                          | > 20 dB là tốt, > 30 dB rất tốt |
| **L1 Loss**  | Sai số pixel trung bình                                 | Càng thấp càng tốt              |
| **VGG Loss** | Sai số perceptual                                       | Càng thấp càng tốt              |

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt ảnh khuôn mặt vào thư mục `dataset/without_mask/`

### 3. Training

```bash
cd ImprovedPremiumGAN
python train.py
```

### 4. Inference

```bash
# Đặt ảnh test vào dataset/test_images/
python inference.py
```

---

## 📈 Hyperparameters

| Parameter       | Giá trị | Mô tả                       |
| --------------- | ------- | --------------------------- |
| `IMG_SIZE`      | 128×128 | Kích thước ảnh input/output |
| `BATCH_SIZE`    | 16      | Số ảnh mỗi batch            |
| `NUM_EPOCHS`    | 100     | Số epoch tối đa             |
| `LEARNING_RATE` | 0.0002  | Learning rate cho cả G và D |
| `LAMBDA_L1`     | 100     | Weight cho L1 Loss          |
| `LAMBDA_VGG`    | 10      | Weight cho VGG Loss         |
| `LAMBDA_ADV`    | 1       | Weight cho Adversarial Loss |
| `TRAIN_RATIO`   | 0.9     | Tỷ lệ train/test split      |

---

## 🔍 Mask Detection (YOLO)

Sử dụng **YOLOv8** để phát hiện vùng khuôn mặt và xác định vùng mask:

- Detect bounding box của người
- Ước tính vùng mask = nửa dưới khuôn mặt
- Fallback: nếu không detect được, sử dụng tọa độ cố định

---


