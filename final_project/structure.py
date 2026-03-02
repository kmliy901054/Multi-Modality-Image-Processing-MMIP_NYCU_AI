import cv2
import numpy as np

def process_structure(v_channel, sigma=None):
    """
    Modality B: Structure
    [自適應策略] 基於亮度判斷雜訊程度，調整雙邊濾波強度 (Sigma)
    """
    # === Adaptive Logic ===
    if sigma is None:
        mean_v = np.mean(v_channel)
        # 暗部 (Mean < 60) 通常伴隨高雜訊 -> 需要更強的平滑 (Sigma=50)
        # 亮部 (Mean >= 60) 雜訊少 -> 減少平滑以保留細節 (Sigma=15)
        if mean_v < 60:
            sigma = 50
            print(f"[Adaptive Struct] High Noise Risk (Mean:{mean_v:.1f}) -> Strong Denoise (Sigma=50)")
        else:
            sigma = 15
            print(f"[Adaptive Struct] Low Noise Risk (Mean:{mean_v:.1f}) -> Weak Denoise (Sigma=15)")
    # ======================

    # 雙邊濾波 (Bilateral Filter)
    v_smooth = cv2.bilateralFilter(v_channel, d=9, sigmaColor=sigma, sigmaSpace=sigma)

    # Sobel 算子提取邊緣
    grad_x = cv2.Sobel(v_smooth, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(v_smooth, cv2.CV_16S, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    structure_map = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return structure_map