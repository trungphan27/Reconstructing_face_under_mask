"""
EDA (Exploratory Data Analysis) for Face Reconstruction Dataset
Generates visualizations for:
1. Dataset statistics and image properties
2. Pixel intensity analysis (bright/dark ratio per class)
3. Color balance analysis (RGB channel distribution)
4. Color histograms and mean image

Output: dataset/EDA/EDA_result/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset')
OUTPUT_PATH = os.path.join(DATASET_PATH, 'EDA', 'EDA_result')
SAMPLE_SIZE = 500  # Number of images to sample for faster processing

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_images_from_folder(folder_path, max_samples=None):
    """Load images from a folder and return as numpy arrays"""
    images = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if max_samples and len(image_files) > max_samples:
        image_files = list(np.random.choice(image_files, max_samples, replace=False))
    
    for filename in tqdm(image_files, desc=f"Loading {os.path.basename(folder_path)}"):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(np.array(img))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return images


def get_dataset_statistics(folder_path):
    """Get basic statistics about the dataset"""
    stats = {
        'total_images': 0,
        'file_sizes': [],
        'dimensions': [],
        'channels': []
    }
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    stats['total_images'] = len(image_files)
    
    for filename in tqdm(image_files[:100], desc="Analyzing properties"):
        img_path = os.path.join(folder_path, filename)
        try:
            file_size = os.path.getsize(img_path)
            stats['file_sizes'].append(file_size / 1024)  # KB
            
            with Image.open(img_path) as img:
                stats['dimensions'].append(img.size)
                stats['channels'].append(len(img.getbands()))
        except Exception as e:
            print(f"Error: {e}")
    
    return stats


def calculate_pixel_intensity(images, threshold=127):
    """Calculate bright and dark pixel ratios"""
    bright_ratios = []
    dark_ratios = []
    
    for img in images:
        grayscale = np.mean(img, axis=2)
        total_pixels = grayscale.size
        bright_pixels = np.sum(grayscale > threshold)
        dark_pixels = np.sum(grayscale <= threshold)
        
        bright_ratios.append(bright_pixels / total_pixels)
        dark_ratios.append(dark_pixels / total_pixels)
    
    return np.mean(bright_ratios), np.mean(dark_ratios)


def calculate_color_balance(images):
    """Calculate average color values for each channel"""
    r_values = []
    g_values = []
    b_values = []
    
    for img in images:
        r_values.append(np.mean(img[:, :, 0]))
        g_values.append(np.mean(img[:, :, 1]))
        b_values.append(np.mean(img[:, :, 2]))
    
    return {
        'Blue': np.mean(b_values),
        'Green': np.mean(g_values),
        'Red': np.mean(r_values)
    }


def calculate_pixel_statistics(images):
    """Calculate pixel statistics for each channel"""
    stats = {'R': [], 'G': [], 'B': []}
    
    for img in images:
        stats['R'].append(img[:, :, 0].flatten())
        stats['G'].append(img[:, :, 1].flatten())
        stats['B'].append(img[:, :, 2].flatten())
    
    result = {}
    for channel in ['R', 'G', 'B']:
        all_pixels = np.concatenate(stats[channel])
        result[channel] = {
            'mean': np.mean(all_pixels),
            'std': np.std(all_pixels),
            'min': np.min(all_pixels),
            'max': np.max(all_pixels)
        }
    
    return result


def plot_pixel_intensity_heatmap(intensity_data, output_path):
    """
    Plot pixel intensity heatmap (bright/dark ratio per class)
    Similar to Figure 5 in the reference
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(intensity_data.keys())
    data = np.array([[intensity_data[cls]['bright'], intensity_data[cls]['dark']] for cls in classes])
    
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=0.4, vmax=0.6)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Bright Ratio', 'Dark Ratio'], fontsize=12)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    
    for i in range(len(classes)):
        for j in range(2):
            ax.text(j, i, f'{data[i, j]:.2f}',
                   ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    
    ax.set_title('Average ratio of bright and dark pixels per class', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Class', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'pixel_intensity_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pixel_intensity_heatmap.png")


def plot_color_balance(color_data, output_path):
    """
    Plot color balance bar chart
    Similar to Figure 7 in the reference
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors_order = ['Blue', 'Green', 'Red']
    bar_colors = ['#0066CC', '#228B22', '#CC0000']
    values = [color_data[c] for c in colors_order]
    
    bars = ax.bar(colors_order, values, color=bar_colors, width=0.6, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Color channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average value per color', fontsize=12, fontweight='bold')
    ax.set_title('The color balance of the image', fontsize=14, fontweight='bold', pad=15)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, max(values) + 20)
    ax.tick_params(axis='both', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'color_balance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: color_balance.png")


def plot_rgb_histograms(images, output_path):
    """Plot RGB channel histograms"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    all_r = np.concatenate([img[:, :, 0].flatten() for img in images[:100]])
    all_g = np.concatenate([img[:, :, 1].flatten() for img in images[:100]])
    all_b = np.concatenate([img[:, :, 2].flatten() for img in images[:100]])
    
    axes[0].hist(all_r, bins=50, color='red', alpha=0.7, edgecolor='darkred')
    axes[0].set_title('Red Channel Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(all_g, bins=50, color='green', alpha=0.7, edgecolor='darkgreen')
    axes[1].set_title('Green Channel Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Pixel Value')
    
    axes[2].hist(all_b, bins=50, color='blue', alpha=0.7, edgecolor='darkblue')
    axes[2].set_title('Blue Channel Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Pixel Value')
    
    plt.suptitle('RGB Channel Histograms', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'rgb_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: rgb_histograms.png")


def plot_brightness_distribution(images, output_path):
    """Plot brightness distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    brightness_values = [np.mean(img) for img in images]
    
    ax.hist(brightness_values, bins=50, color='gold', alpha=0.7, edgecolor='darkorange')
    ax.axvline(np.mean(brightness_values), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(brightness_values):.1f}')
    
    ax.set_xlabel('Brightness Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Brightness Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'brightness_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: brightness_distribution.png")


def plot_mean_image(images, output_path):
    """Calculate and plot mean image (average face)"""
    if len(images) == 0:
        return
    
    mean_img = np.zeros_like(images[0], dtype=np.float64)
    count = 0
    for img in images:
        if img.shape == images[0].shape:
            mean_img += img.astype(np.float64)
            count += 1
    
    mean_img /= count
    mean_img = mean_img.astype(np.uint8)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mean_img)
    ax.set_title('Mean Face (Average of All Images)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'mean_image.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: mean_image.png")


def plot_sample_images(images, output_path, n_samples=16):
    """Plot sample images grid"""
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    sample_indices = np.random.choice(len(images), n_samples, replace=False)
    
    for idx, (ax, img_idx) in enumerate(zip(axes, sample_indices)):
        ax.imshow(images[img_idx])
        ax.axis('off')
        ax.set_title(f'Sample {idx+1}', fontsize=10)
    
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sample_images.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: sample_images.png")


def plot_dataset_statistics(stats, output_path):
    """Plot dataset statistics summary"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if stats['file_sizes']:
        axes[0].hist(stats['file_sizes'], bins=30, color='steelblue', alpha=0.7, edgecolor='navy')
        axes[0].set_xlabel('File Size (KB)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('File Size Distribution', fontsize=12, fontweight='bold')
        axes[0].axvline(np.mean(stats['file_sizes']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(stats["file_sizes"]):.1f} KB')
        axes[0].legend()
    
    dimensions = list(set(stats['dimensions']))
    dim_labels = [f'{d[0]}x{d[1]}' for d in dimensions]
    dim_counts = [stats['dimensions'].count(d) for d in dimensions]
    
    axes[1].bar(dim_labels, dim_counts, color='coral', edgecolor='darkred')
    axes[1].set_xlabel('Image Dimensions', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Image Dimension Distribution', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Dataset Overview (Total: {stats["total_images"]} images)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: dataset_statistics.png")


def generate_summary_report(stats, pixel_stats, color_balance, intensity_data, output_path):
    """Generate a summary text report"""
    report = []
    report.append("=" * 60)
    report.append("EDA SUMMARY REPORT - Face Reconstruction Dataset")
    report.append("=" * 60)
    report.append("")
    
    report.append("📊 DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total images: {stats['total_images']}")
    if stats['file_sizes']:
        report.append(f"Average file size: {np.mean(stats['file_sizes']):.2f} KB")
        report.append(f"File size range: {np.min(stats['file_sizes']):.2f} - {np.max(stats['file_sizes']):.2f} KB")
    if stats['dimensions']:
        dim = stats['dimensions'][0]
        report.append(f"Image dimensions: {dim[0]} x {dim[1]} pixels")
    report.append("")
    
    report.append("🎨 PIXEL STATISTICS (R, G, B)")
    report.append("-" * 40)
    for channel in ['R', 'G', 'B']:
        if channel in pixel_stats:
            s = pixel_stats[channel]
            report.append(f"  {channel}: Mean={s['mean']:.2f}, Std={s['std']:.2f}, Min={s['min']}, Max={s['max']}")
    report.append("")
    
    report.append("🌈 COLOR BALANCE")
    report.append("-" * 40)
    for color, value in color_balance.items():
        report.append(f"  {color}: {value:.2f}")
    report.append("")
    
    report.append("💡 PIXEL INTENSITY (Bright/Dark Ratio)")
    report.append("-" * 40)
    for cls, data in intensity_data.items():
        report.append(f"  {cls}:")
        report.append(f"    Bright Ratio: {data['bright']:.2f}")
        report.append(f"    Dark Ratio: {data['dark']:.2f}")
    report.append("")
    
    report.append("=" * 60)
    report.append("Generated visualizations saved to: dataset/EDA/EDA_result/")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    with open(os.path.join(output_path, 'eda_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print("\n✓ Saved: eda_summary.txt")


def main():
    print("=" * 60)
    print("EDA Analysis for Face Reconstruction Dataset")
    print("=" * 60)
    
    with_mask_path = os.path.join(DATASET_PATH, 'with_mask')
    without_mask_path = os.path.join(DATASET_PATH, 'without_mask')
    
    folders = {}
    if os.path.exists(with_mask_path):
        folders['with_mask'] = with_mask_path
        print(f"Found: with_mask folder")
    if os.path.exists(without_mask_path):
        folders['without_mask'] = without_mask_path
        print(f"Found: without_mask folder")
    
    if not folders:
        print("Error: No dataset folders found!")
        return
    
    print(f"\nLoading images (sample size: {SAMPLE_SIZE})...")
    
    all_images = {}
    combined_images = []
    
    for class_name, folder_path in folders.items():
        images = load_images_from_folder(folder_path, max_samples=SAMPLE_SIZE)
        all_images[class_name] = images
        combined_images.extend(images)
        print(f"  Loaded {len(images)} images from {class_name}")
    
    print(f"\nTotal images loaded: {len(combined_images)}")
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60 + "\n")
    
    print("1. Calculating dataset statistics...")
    stats = get_dataset_statistics(list(folders.values())[0])
    plot_dataset_statistics(stats, OUTPUT_PATH)
    
    print("\n2. Calculating pixel intensity...")
    intensity_data = {}
    for class_name, images in all_images.items():
        bright, dark = calculate_pixel_intensity(images)
        intensity_data[class_name] = {'bright': bright, 'dark': dark}
    plot_pixel_intensity_heatmap(intensity_data, OUTPUT_PATH)
    
    print("\n3. Calculating color balance...")
    color_balance = calculate_color_balance(combined_images)
    plot_color_balance(color_balance, OUTPUT_PATH)
    
    print("\n4. Generating RGB histograms...")
    plot_rgb_histograms(combined_images, OUTPUT_PATH)
    
    print("\n5. Generating brightness distribution...")
    plot_brightness_distribution(combined_images, OUTPUT_PATH)
    
    print("\n6. Calculating mean image...")
    plot_mean_image(combined_images, OUTPUT_PATH)
    
    print("\n7. Generating sample images grid...")
    plot_sample_images(combined_images, OUTPUT_PATH)
    
    print("\n8. Calculating detailed pixel statistics...")
    pixel_stats = calculate_pixel_statistics(combined_images)
    
    print("\n9. Generating summary report...")
    generate_summary_report(stats, pixel_stats, color_balance, intensity_data, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("✅ EDA Analysis Complete!")
    print(f"📁 Results saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
