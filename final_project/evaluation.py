import cv2
import numpy as np
import os
from tkinter import messagebox
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 匯入你的模組
import illumination
import structure
import fusion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_IMAGES_DIR = os.path.join(BASE_DIR, "data", "samples", "images")
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "outputs", "images")
DEFAULT_GT_PATH = os.path.join(DATA_IMAGES_DIR, "normal_light.jpg")
# ✅ [關鍵修復] 解決 Windows 中文路徑讀取失敗的輔助函式
def cv_imread(file_path):
    try:
        # 使用 numpy 從檔案讀取原始字節，再用 opencv 解碼
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv_img
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return None

def run_evaluation_mode(gt_path=None):
    """
    執行定量評估的主函數 (修復中文路徑與相對路徑問題)
    """
    # 1. 鎖定圖片路徑 (強制使用絕對路徑，避免執行位置不同導致找不到)
    # 取得目前這隻程式 (evaluation.py) 所在的資料夾
    gt_path = gt_path or DEFAULT_GT_PATH

    # 2. 檢查檔案是否存在
    if not os.path.exists(gt_path):
        messagebox.showwarning("警告", f"找不到檔案：\n{gt_path}\n請確認檔名是否正確。")
        return

    # 3. [關鍵修正] 使用支援中文路徑的讀取方式
    gt_img = cv_imread(gt_path)

    if gt_img is None:
        messagebox.showerror("嚴重錯誤", f"檔案存在但無法讀取 (可能是格式損壞):\n{gt_path}")
        return

    print(f"成功讀取圖片: {gt_path}, 大小: {gt_img.shape}")

    # 4. 模擬低光源 (Gamma Correction)
    input_gamma = 3.0
    # 修正 LUT 形狀為 (256, 1) 避免 OpenCV 報錯
    lut = np.empty((256, 1), np.uint8)
    for i in range(256):
        lut[i, 0] = np.clip(pow(i / 255.0, input_gamma) * 255.0, 0, 255)
    
    low_light_input = cv2.LUT(gt_img, lut)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "eval_input.jpg"), low_light_input)

    # ================= 我們的致勝算法 (Ours) =================
    
    # A. Modality 1: Illumination 
    h, s, v, v_en = illumination.process_illumination(low_light_input, clip_limit=2.0, gamma_value=2.2)
    
    # B. Modality 2: Structure 
    struct = structure.process_structure(v, sigma=50)
    
    # C. Fusion Strategy 
    our_result = fusion.fuse_images(h, s, v_en, struct, structure_weight=0.3)

    # ========================================================

    # Baseline: Histogram Equalization (HE)
    img_yuv = cv2.cvtColor(low_light_input, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    he_result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # 計算評估指標 (PSNR/SSIM)
    def get_metrics(gt, img):
        if gt.shape != img.shape:
            img = cv2.resize(img, (gt.shape[1], gt.shape[0]))
        
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        p = psnr(gt_gray, img_gray)
        s_ = ssim(gt_gray, img_gray) 
        return p, s_

    p_ours, s_ours = get_metrics(gt_img, our_result)
    p_he, s_he = get_metrics(gt_img, he_result)

    diff_p = p_ours - p_he
    diff_s = s_ours - s_he

    p_result_str = f"+{diff_p:.2f} (WIN)" if diff_p > 0 else f"{diff_p:.2f} (LOSE)"
    s_result_str = f"+{diff_s:.4f} (WIN)" if diff_s > 0 else f"{diff_s:.4f} (LOSE)"

    result_text = (
        f"Baseline (HE)\n  PSNR: {p_he:.2f}\n  SSIM: {s_he:.4f}\n\n"
        f"Ours (Fusion)\n  PSNR: {p_ours:.2f}\n  SSIM: {s_ours:.4f}\n\n"
        f"🏆 Final Result:\n  PSNR: {p_result_str}\n  SSIM: {s_result_str}"
    )

    messagebox.showinfo("評估結果", result_text)

    # 存檔也建議使用 encode 避免中文路徑存檔失敗 (這裡先用 imwrite，通常存檔比較沒問題，若有問題再改)
    try:
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True); cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "eval_ours.jpg"), our_result)
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "eval_he.jpg"), he_result)
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "eval_gt.jpg"), gt_img)
        print("圖片已儲存。")
    except Exception as e:
        print(f"存檔警告: {e}")

if __name__ == "__main__":
    run_evaluation_mode()