"""
執行這個檔案來訓練 YOLO 模型
執行方式：python train.py
"""
from ultralytics import YOLO
import os

# 訓練設定
DATA_YAML   = 'dataset/data.yaml'   # 資料集設定檔
MODEL_BASE  = 'yolov8n.pt'          # 基礎模型（n=最小最快，可換成 yolov8s.pt 更準）
EPOCHS      = 50                     # 訓練輪數（資料少可降到 30，多可提高到 100）
IMG_SIZE    = 640                    # 圖片大小
OUTPUT_DIR  = 'runs/train'           # 訓練結果輸出位置

def train():
    print("=== 開始訓練 YOLOv8 模型 ===")
    print(f"資料集：{DATA_YAML}")
    print(f"訓練輪數：{EPOCHS}")
    print()

    if not os.path.exists(DATA_YAML):
        print(f"❌ 找不到 {DATA_YAML}")
        print("請確認已照說明放好訓練照片並建立 data.yaml")
        return

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=OUTPUT_DIR,
        name='kiln_detector',
        patience=20,       # 連續 20 輪沒進步就提早停止
        batch=16,          # 一次訓練幾張圖（記憶體不夠可改 8）
        device='cpu',      # 沒有 GPU 用 cpu，有 GPU 改成 0
        workers=2,
    )

    # 訓練完自動複製最佳模型到 models/
    best_model_path = os.path.join(OUTPUT_DIR, 'kiln_detector', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        os.makedirs('models', exist_ok=True)
        import shutil
        shutil.copy(best_model_path, 'models/best.pt')
        print()
        print("✅ 訓練完成！模型已儲存到 models/best.pt")
        print("現在可以啟動 Flask 伺服器：python app.py")
    else:
        print(f"⚠️ 請手動將 {best_model_path} 複製到 models/best.pt")

if __name__ == '__main__':
    train()
