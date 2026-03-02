import cv2
import numpy as np

def process_illumination(img, clip_limit=None, gamma_value=None):
    """
    Modality A: Illumination
    [升級版] Region-Aware Fusion + Auto-Gamma
    解決「不夠亮」的問題：在暗部自動注入 Gamma 增益
    """
    # 轉 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # === [Manual Mode: 手動模式] ===
    # 如果外部有指定參數 (例如 GUI 拉動)，就聽 GUI 的
    if clip_limit is not None:
        # 如果有指定 gamma，先做 gamma
        if gamma_value is not None and gamma_value != 1.0:
            invGamma = 1.0 / gamma_value
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            v = cv2.LUT(v, table)
            
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        v_final = clahe.apply(v)
    
    # === [Auto Mode: 自適應全自動模式] ===
    else:
        # 1. 計算平均亮度，決定要不要 "預先補光"
        mean_v = np.mean(v)
        print(f"[Smart Illum] Scene Mean Brightness: {mean_v:.2f}")

        # --- 新增：自動 Gamma 增益 (Auto-Gamma) ---
        # 如果亮度 < 80，代表這張圖偏暗，我們給它一個 Gamma 增益基底
        # 這樣後面的 CLAHE 才有東西可以拉
        if mean_v < 80:
            # 越暗，Gamma 要給越強 (例如 mean=20 -> gamma=2.5, mean=70 -> gamma=1.5)
            # 動態公式：基礎 1.2 + (80 - mean) * 0.02
            auto_gamma = 1.5 + (80 - mean_v) * 0.02
            # 限制最高不要超過 3.0，不然會變全白
            auto_gamma = min(auto_gamma, 3.0)
            
            print(f"[Smart Illum] Detected Dark Scene -> Injecting Gamma: {auto_gamma:.2f}")
            
            invGamma = 1.0 / auto_gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            v = cv2.LUT(v, table) # 更新 V 通道，讓它先變亮！
        # ----------------------------------------

        # 2. 製作「光源遮罩 (Highlight Mask)」(基於變亮後的 V)
        # 找出亮度大於 220 的區域 (視為路燈/招牌)
        _, mask_binary = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY)
        mask_blur = cv2.GaussianBlur(mask_binary, (21, 21), 0)
        mask_weight = mask_blur.astype(float) / 255.0
        
        # 3. 路徑 A：背景增強版 (Background Layer)
        # 針對暗部：將參數從 6.0 提升到 8.0 (更暴力)
        # 如果你覺得還是不夠亮，可以把 8.0 改成 10.0
        clahe_bg = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        v_bg_enhanced = clahe_bg.apply(v)
        
        # 4. 路徑 B：光源保護版 (Highlight Layer)
        # 針對亮部：維持 1.0 或 1.5，保護細節
        clahe_light = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        v_light_protected = clahe_light.apply(v)
        
        # 5. 融合 (Fusion)
        v_final_float = (v_light_protected.astype(float) * mask_weight) + \
                        (v_bg_enhanced.astype(float) * (1.0 - mask_weight))
                        
        v_final = np.clip(v_final_float, 0, 255).astype(np.uint8)

    return h, s, v, v_final