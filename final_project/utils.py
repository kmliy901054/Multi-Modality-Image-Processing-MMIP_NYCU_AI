import cv2
import matplotlib.pyplot as plt
import os

def load_image(path):
    """讀取圖片並檢查是否存在"""
    if not os.path.exists(path):
        print(f"Error: 找不到檔案 '{path}'")
        return None
    img = cv2.imread(path)
    return img

def show_results(original, illumination, structure, result):
    """使用 Matplotlib 顯示四張對比圖 (符合投影片展示需求)"""
    
    # 將 BGR 轉 RGB 以便 Matplotlib 正確顯示
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(16, 8))

    # 1. 原始圖
    plt.subplot(1, 4, 1)
    plt.imshow(original_rgb)
    plt.title("Original Low-Light [cite: 31]")
    plt.axis('off')

    # 2. 光照分量 (Illumination)
    plt.subplot(1, 4, 2)
    plt.imshow(illumination, cmap='gray')
    plt.title("Modality A: Illumination ")
    plt.axis('off')

    # 3. 結構分量 (Structure)
    plt.subplot(1, 4, 3)
    plt.imshow(structure, cmap='gray')
    plt.title("Modality B: Structure ")
    plt.axis('off')

    # 4. 最終結果
    plt.subplot(1, 4, 4)
    plt.imshow(result_rgb)
    plt.title("Fused Output ")
    plt.axis('off')

    plt.tight_layout()
    plt.show()