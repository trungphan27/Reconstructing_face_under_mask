import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. CHUẨN BỊ DỮ LIỆU
# ---------------------------------------------------------

# ---------------------------------------------------------
# 1. CHUẨN BỊ DỮ LIỆU
# ---------------------------------------------------------

# Đường dẫn file log
log_path = os.path.join("ImprovedPremiumGAN", "results", "train_log.csv")

# Kiểm tra xem file log có tồn tại không
if os.path.exists(log_path):
    print(f"Reading data from {log_path}")
    df_premium = pd.read_csv(log_path)
    # Đổi tên cột cho khớp với code cũ nếu cần, hoặc dùng trực tiếp
    # CSV headers: Epoch,D_Loss,G_Loss,L1_Loss,VGG_Loss,SSIM,PSNR
    # df_premium đã có sẵn các cột này.
else:
    print(f"File not found: {log_path}. Using dummy data or exiting.")
    # Fallback hoặc raise error. Ở đây ta giả lập dữ liệu rỗng để tránh crash
    df_premium = pd.DataFrame(columns=["Epoch", "D_Loss", "G_Loss", "L1_Loss", "SSIM", "PSNR"])

# Dữ liệu Referenced-Based Inpainting (Baseline tĩnh để so sánh)
data_ref = {
    'Epoch': list(range(1, 23)),
    'Loss': [0.1072, 0.0695, 0.0602, 0.0543, 0.0500, 0.0465, 0.0438, 0.0418, 0.0401, 0.0384, 
             0.0368, 0.0358, 0.0347, 0.0335, 0.0330, 0.0320, 0.0314, 0.0304, 0.0297, 0.0293, 0.0287, 0.0281],
    'SSIM': [0.5704, 0.7117, 0.7523, 0.7760, 0.7939, 0.8091, 0.8210, 0.8306, 0.8383, 0.8459, 
             0.8525, 0.8574, 0.8623, 0.8671, 0.8699, 0.8743, 0.8770, 0.8808, 0.8835, 0.8857, 0.8881, 0.8904],
    'PSNR': [23.0076, 25.9288, 26.9777, 27.7417, 28.3966, 28.9891, 29.4576, 29.8466, 30.1446, 30.4666, 
             30.7646, 30.9858, 31.1973, 31.4387, 31.5770, 31.7949, 31.9355, 32.1380, 32.2985, 32.4140, 32.5436, 32.6907]
}

df_ref = pd.DataFrame(data_ref)

# Cắt ngắn hoặc kéo dài df_ref để khớp với số epoch training hiện tại nếu muốn so sánh 1-1
# Ở đây ta cứ vẽ cả hai lên.

# Tạo DataFrame tổng hợp để so sánh (cho biểu đồ 1 và 2)
# Lưu ý: df_premium có thể có số dòng khác df_ref
# Chúng ta sẽ concat lại.

df_premium['Model'] = 'PremiumGAN'
df_ref['Model'] = 'Ref-Based Inpainting'

# Chỉ lấy các cột cần thiết để so sánh
cols_compare = ['Epoch', 'SSIM', 'PSNR', 'Model']
df_compare = pd.concat([df_premium[cols_compare], df_ref[cols_compare]], ignore_index=True)


# ---------------------------------------------------------
# 2. CẤU HÌNH LƯU TRỮ VÀ VẼ BIỂU ĐỒ
# ---------------------------------------------------------

# Thiết lập thư mục lưu ảnh
output_dir = 'evaluate'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Thiết lập style chung cho Seaborn
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6) # Kích thước mặc định

# --- BIỂU ĐỒ 1: SO SÁNH SSIM ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_compare, x='Epoch', y='SSIM', hue='Model', 
             style='Model', markers=True, dashes=False, linewidth=2.5)
plt.title('COMPARISON: SSIM SCORE (Higher is Better)', fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel('Epochs')
plt.ylabel('SSIM')
plt.legend(title='Method')
plt.tight_layout()
save_path = os.path.join(output_dir, 'comparison_ssim.png')
plt.savefig(save_path)
plt.close()
print(f"Saved: {save_path}")

# --- BIỂU ĐỒ 2: SO SÁNH PSNR ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_compare, x='Epoch', y='PSNR', hue='Model', 
             style='Model', markers=True, dashes=False, linewidth=2.5, palette='viridis')
plt.title('COMPARISON: PSNR SCORE (Higher is Better)', fontsize=16, fontweight='bold', color='darkgreen')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend(title='Method')
plt.tight_layout()
save_path = os.path.join(output_dir, 'comparison_psnr.png')
plt.savefig(save_path)
plt.close()
print(f"Saved: {save_path}")

# --- BIỂU ĐỒ 3: L1 LOSS CỦA PREMIUM GAN ---
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_premium, x='Epoch', y='L1_Loss', color='#d62728', marker='o', linewidth=2)
plt.title('PremiumGAN: L1 Loss (Pixel-wise Error)', fontsize=15, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('L1 Loss')
plt.fill_between(df_premium['Epoch'], df_premium['L1_Loss'], color='#d62728', alpha=0.1) # Tô màu nền
plt.tight_layout()
save_path = os.path.join(output_dir, 'premiumgan_l1_loss.png')
plt.savefig(save_path)
plt.close()
print(f"Saved: {save_path}")

# --- BIỂU ĐỒ 4: LOSS CỦA REFERENCE-BASED INPAINTING ---
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_ref, x='Epoch', y='Loss', color='#9467bd', marker='X', linewidth=2)
plt.title('Ref-Based Inpainting: Total Loss', fontsize=15, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.fill_between(df_ref['Epoch'], df_ref['Loss'], color='#9467bd', alpha=0.1)
plt.tight_layout()
save_path = os.path.join(output_dir, 'ref_based_loss.png')
plt.savefig(save_path)
plt.close()
print(f"Saved: {save_path}")

# --- BIỂU ĐỒ 5: D_LOSS VÀ G_LOSS CỦA PREMIUM GAN ---
# Chuyển đổi dữ liệu để vẽ 2 đường trên cùng 1 biểu đồ
df_gan_loss = df_premium.melt(id_vars=['Epoch'], value_vars=['D_Loss', 'G_Loss'], 
                              var_name='Loss Type', value_name='Value')

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_gan_loss, x='Epoch', y='Value', hue='Loss Type', 
             linewidth=2, palette=['#ff7f0e', '#1f77b4'])
plt.title('PremiumGAN: Generator vs Discriminator Loss', fontsize=16, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend(title='Component')
plt.tight_layout()
save_path = os.path.join(output_dir, 'premiumgan_gan_loss.png')
plt.savefig(save_path)
plt.close()
print(f"Saved: {save_path}")

print("All plots generated successfully.")
