import cv2
import numpy as np

def fuse_images(h, s, v_enhanced, structure_map, structure_weight=None):
    """
    Fusion Strategy (Updated for De-hazing)
    1. 自適應權重融合
    2. [新功能] 飽和度補償 (解決色彩平淡)
    3. [新功能] 智慧對比度拉伸 (解決畫面灰灰的)
    """
    
    # === 1. 自適應結構權重邏輯 (保持不變) ===
    if structure_weight is None:
        std_v = np.std(v_enhanced)
        if std_v > 45:
            structure_weight = 0.25
        else:
            structure_weight = 0.1
    # ======================================

    # 融合亮度與結構
    v_fused_float = v_enhanced.astype(float) + (structure_map.astype(float) * structure_weight)
    v_fused = np.clip(v_fused_float, 0, 255).astype(np.uint8)

    # === [關鍵修正 1] 飽和度補償 (Saturation Boost) ===
    # 因為亮度提昇了，飽和度也要跟著提升，不然會泛白
    # 這裡將 S 通道乘上 1.3 倍 (你可以調整這個數值，1.2 ~ 1.5 都不錯)
    s_boost = cv2.addWeighted(s, 1.3, s, 0, 0)

    # 合併通道
    final_hsv = cv2.merge((h, s_boost, v_fused))
    final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # === [關鍵修正 2] 智慧對比度拉伸 (De-hazing / Contrast Stretch) ===
    # 這一步專門用來消除「灰霧感」
    # 原理：把暗部壓更暗 (beta<0)，把亮部拉更亮 (alpha>1)
    # alpha=1.1 (對比度增加 10%), beta=-10 (亮度整體減 10，把灰色壓回黑色)
    final_bgr = cv2.convertScaleAbs(final_bgr, alpha=1.1, beta=-15)

    return final_bgr