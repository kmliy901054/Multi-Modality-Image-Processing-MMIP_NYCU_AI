import cv2
import numpy as np
import time
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_IMAGES_DIR = os.path.join(BASE_DIR, "data", "samples", "images")
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "outputs", "images")
OUTPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "outputs", "videos")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
DEFAULT_IMAGE_PATH = os.path.join(DATA_IMAGES_DIR, "1_2.png")


def ensure_output_dirs():
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
# 匯入自定義模組
import utils
import illumination
import structure
import fusion
import evaluation  # ✅ 匯入評估模組

# ==========================================
# 檢查與導入外部函式庫 (YOLO)
# ==========================================
HAS_YOLO = False
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    pass

# ==========================================
# 共用工具：判斷 OpenCV 視窗是否已被關閉
# ==========================================
def window_closed(window_name: str) -> bool:
    try:
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        return prop < 1
    except cv2.error:
        return True

# ==========================================
# 1. GUI 互動調整模式
# ==========================================
def run_gui_mode(image_path):
    print(f"--- 啟動 GUI 模式: {image_path} ---")
    img_original = utils.load_image(image_path)
    if img_original is None:
        messagebox.showerror("錯誤", f"找不到圖片: {image_path}")
        return

    height, width = img_original.shape[:2]
    max_h = 800
    if height > max_h:
        scale = max_h / height
        img_original = cv2.resize(img_original, (int(width * scale), int(height * scale)))

    window_name = "Multi-modal Interactive Adjustment"
    cv2.namedWindow(window_name)
    gui_ready = False

    def process_pipeline():
        if not gui_ready: return None
        try:
            # 讀取滑動條數值
            clip_val = cv2.getTrackbarPos("Contrast", window_name) / 10.0
            if clip_val <= 0: clip_val = 0.1
            
            struct_weight = cv2.getTrackbarPos("Detail", window_name) / 100.0
            sat_scale = cv2.getTrackbarPos("Color", window_name) / 100.0
            denoise_level = cv2.getTrackbarPos("Denoise", window_name)
            
            sigma_val = denoise_level * 10
            if sigma_val == 0: sigma_val = 1
        except cv2.error:
            return None

        # 執行演算法 (傳入 GUI 指定的參數，覆蓋自適應邏輯)
        h, s, v_orig, v_enh = illumination.process_illumination(img_original, clip_limit=clip_val)
        struct_map = structure.process_structure(v_orig, sigma=sigma_val)
        
        if sat_scale != 1.0:
            s = cv2.convertScaleAbs(s, alpha=sat_scale, beta=0)
            
        final_bgr = fusion.fuse_images(h, s, v_enh, struct_map, structure_weight=struct_weight)
        cv2.imshow(window_name, final_bgr)
        return final_bgr

    # 建立滑動條
    cv2.createTrackbar("Contrast", window_name, 30, 60, lambda x: process_pipeline())
    cv2.createTrackbar("Detail", window_name, 30, 100, lambda x: process_pipeline())
    cv2.createTrackbar("Color", window_name, 100, 200, lambda x: process_pipeline())
    cv2.createTrackbar("Denoise", window_name, 8, 20, lambda x: process_pipeline())

    gui_ready = True
    process_pipeline()

    print("OpenCV GUI Running... (按 s 存檔，按 q 退出)")
    while True:
        if window_closed(window_name): break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            final_res = process_pipeline()
            if final_res is not None:
                ensure_output_dirs(); cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "gui_output.jpg"), final_res)
                messagebox.showinfo("成功", "圖片已儲存為 gui_output.jpg")
        elif key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()

# ==========================================
# 2. 單張圖片增強 (全自適應模式)
# ==========================================
def run_single_image_mode(image_path):
    print(f"--- 啟動全自適應增強模式: {image_path} ---")
    img = utils.load_image(image_path)
    if img is None:
        messagebox.showerror("錯誤", f"找不到圖片: {image_path}")
        return

    # ✅ 關鍵修改：傳入 None 以觸發底層模組的自適應邏輯
    # Illumination: 自動判斷背景與光源
    h, s, v_orig, v_enh = illumination.process_illumination(img, clip_limit=None, gamma_value=None)
    
    # Structure: 自動判斷降噪強度
    struct_map = structure.process_structure(v_orig, sigma=None)
    
    # Fusion: 自動判斷紋理權重
    result = fusion.fuse_images(h, s, v_enh, struct_map, structure_weight=None)

    # 顯示與存檔
    utils.show_results(img, v_enh, struct_map, result)
    ensure_output_dirs(); cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, 'output_adaptive.jpg'), result)
    print("處理完成！參數已自動調整。")
    messagebox.showinfo("完成", "自適應增強完成！\n已儲存至 'output_adaptive.jpg'")

# ==========================================
# 3. 影片增強模式
# ==========================================
def run_video_mode(input_path):
    if not os.path.exists(input_path):
        messagebox.showerror("錯誤", "找不到影片檔")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        messagebox.showerror("錯誤", "無法開啟影片")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    ensure_output_dirs(); out_name = os.path.join(OUTPUT_VIDEOS_DIR, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_name, fourcc, int(fps), (width * 2, height))

    win = "Video Processing"
    cv2.namedWindow(win)
    print("影片處理中...（按 q 停止）")

    while cap.isOpened():
        if window_closed(win): break
        ret, frame = cap.read()
        if not ret: break

        # 影片模式使用稍微固定的參數以保持畫面穩定，或可使用自適應但需注意閃爍
        # 這裡使用推薦的通用參數
        h, s, v_orig, v_enh = illumination.process_illumination(frame, clip_limit=3.0, gamma_value=2.0)
        struct_map = structure.process_structure(v_orig, sigma=50)
        res = fusion.fuse_images(h, s, v_enh, struct_map, structure_weight=0.3)

        combined = np.hstack((frame, res))
        out.write(combined)
        
        show = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow(win, show)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("完成", f"影片處理完成: {out_name}")

# ==========================================
# 5. YOLO 物件偵測測試
# ==========================================
def run_yolo_test(image_path):
    if not HAS_YOLO:
        messagebox.showerror("錯誤", "未安裝 ultralytics")
        return
    img = utils.load_image(image_path)
    if img is None:
        messagebox.showerror("錯誤", "找不到圖片")
        return
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        messagebox.showerror("錯誤", f"模型載入失敗: {e}")
        return

    # ✅ 使用自適應增強來獲取最佳 YOLO 輸入
    h, s, v_orig, v_enh = illumination.process_illumination(img, clip_limit=None)
    struct_map = structure.process_structure(v_orig, sigma=None)
    enhanced_img = fusion.fuse_images(h, s, v_enh, struct_map, structure_weight=None)

    print("Running YOLO Inference...")
    results_orig = model(img, verbose=False)
    results_enh = model(enhanced_img, verbose=False)

    # 繪圖
    plt.figure(figsize=(16, 8))
    
    # Original
    res_plot_orig = results_orig[0].plot()
    plt.subplot(1, 2, 1)
    plt.title(f"Original Input (Boxes: {len(results_orig[0].boxes)})")
    plt.imshow(cv2.cvtColor(res_plot_orig, cv2.COLOR_BGR2RGB))
    
    # Enhanced
    res_plot_enh = results_enh[0].plot()
    plt.subplot(1, 2, 2)
    plt.title(f"Ours Adaptive (Boxes: {len(results_enh[0].boxes)})")
    plt.imshow(cv2.cvtColor(res_plot_enh, cv2.COLOR_BGR2RGB))
    
    plt.show()

def lift_exposure(v, target_mean=135, max_gain=2.2, gamma=0.85):
    """
    v: uint8 V channel
    target_mean: 想把平均亮度拉到哪裡（120~150 都可）
    max_gain: 限制增益，避免過曝
    gamma: <1 會變亮（0.7~0.95）
    """
    v_f = v.astype(np.float32)
    mean = float(v_f.mean()) + 1e-6

    # 1) exposure gain (全域提亮)
    gain = target_mean / mean
    gain = float(np.clip(gain, 1.0, max_gain))
    v_f = v_f * gain

    # 2) optional gamma (再補一點亮度曲線)
    v_f = np.clip(v_f, 0, 255) / 255.0
    v_f = np.power(v_f, gamma) * 255.0

    return np.clip(v_f, 0, 255).astype(np.uint8)
# ==========================================
# 6 & 7. 即時視訊串流模式
# ==========================================
def run_webcam_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("錯誤", "無法開啟攝像頭")
        return
    cap.set(3, 640); cap.set(4, 480)
    prev_time = 0
    win = 'Real-time Enhancement'
    cv2.namedWindow(win)

    while True:
        if window_closed(win): break
        ret, frame = cap.read()
        if not ret: break

        # 即時模式為求流暢，可使用固定參數或簡易自適應
        h, s, v_orig, v_enh = illumination.process_illumination(frame, clip_limit=2.5)
        v_enh = lift_exposure(v_enh, target_mean=135, max_gain=2.0, gamma=0.85)
        # 簡化結構計算以加速
        v_smooth = cv2.bilateralFilter(v_orig, d=5, sigmaColor=50, sigmaSpace=50)
        grad_x = cv2.Sobel(v_smooth, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(v_smooth, cv2.CV_16S, 0, 1, ksize=3)
        struct_map = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)

        final_result = fusion.fuse_images(h, s, v_enh, struct_map, structure_weight=0.3)

        fps = 1 / (time.time() - prev_time + 1e-6)
        prev_time = time.time()
        combined = np.hstack((frame, final_result))
        cv2.putText(combined, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(win, combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

def run_realtime_yolo_mode():
    if not HAS_YOLO:
        messagebox.showerror("錯誤", "未安裝 ultralytics")
        return
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception:
        return
    cap = cv2.VideoCapture(0)
    cap.set(3, 640); cap.set(4, 480)
    
    frame_count = 0
    skip_frames = 3
    cached_plot_orig = None
    cached_plot_enh = None
    win = 'Real-time YOLO Demo'
    cv2.namedWindow(win)

    while True:
        if window_closed(win): break
        ret, frame = cap.read()
        if not ret: break

        # 影像增強
        h, s, v_orig, v_enh = illumination.process_illumination(frame, clip_limit=2.5)
        v_enh = lift_exposure(v_enh, target_mean=135, max_gain=2.0, gamma=0.85)
        v_smooth = cv2.bilateralFilter(v_orig, d=5, sigmaColor=50, sigmaSpace=50)
        grad_x = cv2.Sobel(v_smooth, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(v_smooth, cv2.CV_16S, 0, 1, ksize=3)
        struct_map = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        enhanced_frame = fusion.fuse_images(h, s, v_enh, struct_map, structure_weight=0.3)

        # YOLO 推論 (跳幀優化)
        if frame_count % skip_frames == 0:
            res_o = model(frame, verbose=False)
            cached_plot_orig = res_o[0].plot()
            res_e = model(enhanced_frame, verbose=False)
            cached_plot_enh = res_e[0].plot()
        
        frame_count += 1
        if cached_plot_orig is None: cached_plot_orig = frame
        if cached_plot_enh is None: cached_plot_enh = enhanced_frame

        combined = np.hstack((cached_plot_orig, cached_plot_enh))
        cv2.imshow(win, combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# ✅ 8. 新增功能：自適應 vs 固定參數 對比實驗
# ==========================================
# ==========================================
# ✅ 8. 新增功能：自適應 vs 傳統方法 對比實驗
# ==========================================
def run_adaptive_experiment(image_path):
    print(f"--- 啟動實驗: 自適應 vs 傳統直方圖均衡 (HE) ---")
    img = utils.load_image(image_path)
    if img is None:
        messagebox.showerror("錯誤", f"找不到圖片: {image_path}")
        return

    # === 1. 傳統方法: Global Histogram Equalization (Real Baseline) ===
    # 這是學術界最標準的對照組。
    # 它的缺點是真實的：會把黑色天空拉成灰色雜訊，且會讓路燈過曝。
    print(">>> Running Baseline (Global HE)...")
    
    hsv_f = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_f, s_f, v_f = cv2.split(hsv_f)
    
    # 使用 OpenCV 內建的直方圖均衡化
    # 這會強行拉開對比度，導致暗處雜訊爆炸，亮處過曝
    v_f_fixed = cv2.equalizeHist(v_f)

    res_fixed = cv2.cvtColor(cv2.merge((h_f, s_f, v_f_fixed)), cv2.COLOR_HSV2BGR)

    # === 2. 全自適應模式 (Ours / Smart) ===
    print(">>> Running Adaptive Mode (Smart)...")
    h_a, s_a, v_a, v_enh_auto = illumination.process_illumination(img, clip_limit=None, gamma_value=None)
    struct_auto = structure.process_structure(v_a, sigma=None)
    res_auto = fusion.fuse_images(h_a, s_a, v_enh_auto, struct_auto, structure_weight=None)

    # === 3. 顯示與存檔 ===
    h, w = img.shape[:2]
    if h > 500:
        scale = 500 / h
        new_w, new_h = int(w*scale), int(h*scale)
        res_fixed_show = cv2.resize(res_fixed, (new_w, new_h))
        res_auto_show = cv2.resize(res_auto, (new_w, new_h))
    else:
        res_fixed_show = res_fixed
        res_auto_show = res_auto

    combined = np.hstack((res_fixed_show, res_auto_show))
    
    # 標籤
    cv2.putText(combined, "Baseline (Global HE)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(combined, "Ours (Region-Aware)", (res_fixed_show.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Experiment: HE vs Adaptive", combined)
    
    ensure_output_dirs(); cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "exp_baseline_he.jpg"), res_fixed)
    cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, "exp_ours_adaptive.jpg"), res_auto)
    
    messagebox.showinfo("實驗完成", "對比已生成！\n\n左圖 (HE Baseline)：\n這是傳統演算法的真實缺陷。\n天空會變成充滿顆粒的灰色，路燈周圍會有光暈。\n\n右圖 (Ours)：\n天空保持純淨，路燈受到保護。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# ==========================================
# 主程式選單 (App Launcher)
# ==========================================
class AppLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-modal Low-Light Enhancement System")
        self.root.geometry("520x720") # 稍微拉長視窗以容納新按鈕
        self.root.resizable(False, False)
        
        style = ttk.Style(); style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        tk.Label(root, text="Low-Light Enhancement", font=("Helvetica", 16, "bold")).pack(pady=15)
        tk.Label(root, text="Multi-modal Fusion (Illumination + Structure)", font=("Helvetica", 10), fg="gray").pack(pady=(0, 20))
        
        file_frame = tk.Frame(root); file_frame.pack(pady=5, padx=20, fill="x")
        tk.Label(file_frame, text="Target File (Img/Video):").pack(anchor="w")
        self.path_var = tk.StringVar(value=DEFAULT_IMAGE_PATH)
        tk.Entry(file_frame, textvariable=self.path_var, width=40).pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side="right")

        # 區塊 1: 影像處理
        self.create_section("Image Processing")
        self.create_btn("1. Interactive GUI Adjustment", self.trigger_gui)
        self.create_btn("2. Single Image Enhancement (Adaptive)", self.trigger_single) # 更新名稱
        self.create_btn("5. YOLO Object Detection Test", self.trigger_yolo_static)

        # 區塊 2: 影片與評估
        self.create_section("Video & Evaluation")
        self.create_btn("3. Video Enhancement (File)", self.trigger_video)
        self.create_btn("4. Quantitative Evaluation (PSNR/SSIM)", self.trigger_eval)
        
        # ✅ 新增按鈕：實驗區
        self.create_btn("8. Adaptive vs Fixed Experiment (Proves Utility)", self.trigger_experiment, color="#ffdddd")

        # 區塊 3: 即時串流
        self.create_section("Real-time Stream")
        self.create_btn("6. Webcam Enhancement Only", self.trigger_webcam)
        self.create_btn("7. Webcam + YOLO (Ultimate Demo)", self.trigger_webcam_yolo, color="#d0e8f2")

        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_section(self, text):
        frame = tk.Frame(self.root); frame.pack(fill="x", padx=20, pady=(15, 5))
        tk.Label(frame, text=text, font=("Arial", 9, "bold"), fg="#333").pack(anchor="w")
        ttk.Separator(frame, orient="horizontal").pack(fill="x")

    def create_btn(self, text, command, color=None):
        tk.Button(self.root, text=text, command=command, bg=color if color else "#f0f0f0", height=1).pack(fill="x", padx=40, pady=2)

    def browse_file(self):
        f = filedialog.askopenfilename(initialdir=os.path.join(BASE_DIR, "data", "samples"), title="Select File")
        if f: self.path_var.set(f)
    def get_path(self): return self.path_var.get().strip()

    # Wrapper Functions (連接按鈕與功能)
    def trigger_gui(self):
        self.status_label.config(text="Running GUI Mode...")
        self.root.update(); run_gui_mode(self.get_path()); self.status_label.config(text="Ready")
    
    def trigger_single(self):
        self.status_label.config(text="Processing Single Image (Adaptive)...")
        self.root.update(); run_single_image_mode(self.get_path()); self.status_label.config(text="Ready")

    def trigger_video(self):
        self.status_label.config(text="Processing Video...")
        self.root.update(); run_video_mode(self.get_path()); self.status_label.config(text="Ready")
    
    def trigger_eval(self):
        self.status_label.config(text="Running Evaluation...")
        self.root.update(); evaluation.run_evaluation_mode(); self.status_label.config(text="Ready")

    def trigger_yolo_static(self):
        self.status_label.config(text="Running YOLO Test...")
        self.root.update(); run_yolo_test(self.get_path()); self.status_label.config(text="Ready")
    def trigger_webcam(self):
        self.status_label.config(text="Running Webcam...")
        self.root.update(); run_webcam_mode(); self.status_label.config(text="Ready")
    def trigger_webcam_yolo(self):
        self.status_label.config(text="Running Webcam + YOLO...")
        self.root.update(); run_realtime_yolo_mode(); self.status_label.config(text="Ready")
    
    # ✅ 新增 Trigger
    def trigger_experiment(self):
        self.status_label.config(text="Running Experiment...")
        self.root.update()
        run_adaptive_experiment(self.get_path())
        self.status_label.config(text="Ready")

def main():
    root = tk.Tk()
    def on_close():
        try: cv2.destroyAllWindows()
        except: pass
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    AppLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()