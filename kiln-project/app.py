from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)
CORS(app)  # 允許前端網頁跨域呼叫

# 載入模型（訓練完後會自動讀取 models/best.pt）
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}\n請先完成訓練，將 best.pt 放到 models/ 資料夾")
        model = YOLO(MODEL_PATH)
    return model

def decode_image(base64_str):
    """將前端傳來的 base64 圖片轉成 OpenCV 格式"""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def count_grid_pixels(img_width, img_height, box_px, grid_spacing_px):
    """
    根據偵測到的 bounding box 換算格數
    box_px: (x1, y1, x2, y2) 像素座標
    grid_spacing_px: 一格在圖片中佔幾個像素（需校準）
    """
    x1, y1, x2, y2 = box_px
    width_px = x2 - x1
    height_px = y2 - y1
    width_grids = round(width_px / grid_spacing_px)
    height_grids = round(height_px / grid_spacing_px)
    return width_grids, height_grids

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': os.path.exists(MODEL_PATH)})

@app.route('/detect/top', methods=['POST'])
def detect_top():
    """
    俯視圖偵測：回傳每個作品的底面積（格數）
    前端傳入：
      - image: base64 圖片字串
      - grid_px: 一格對應的像素數（由前端校準格線後傳入，預設 20）
    回傳：
      - objects: 每個偵測物件的資訊列表
    """
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        grid_px = float(data.get('grid_px', 20))

        yolo = get_model()
        results = yolo(img, conf=0.3)  # 信心度門檻 0.3，可調整

        objects = []
        img_h, img_w = img.shape[:2]

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # 換算成格數
                w_grids = round((x2 - x1) / grid_px)
                h_grids = round((y2 - y1) / grid_px)
                area_grids = w_grids * h_grids

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],           # 像素座標，供前端畫框
                    'width_grids': w_grids,
                    'height_grids': h_grids,
                    'area_grids': area_grids,            # 底面積格數
                    'area_cm2': area_grids * 0.25,       # 底面積平方公分（每格 0.5cm）
                    'confidence': round(conf, 2)
                })

        return jsonify({'success': True, 'objects': objects, 'count': len(objects)})

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 503
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect/side', methods=['POST'])
def detect_side():
    """
    側視圖偵測：回傳每個作品的高度（格數）
    前端傳入：
      - image: base64 圖片字串
      - grid_px: 一格對應的像素數
    回傳：
      - objects: 每個偵測物件的高度資訊
    """
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        grid_px = float(data.get('grid_px', 20))

        yolo = get_model()
        results = yolo(img, conf=0.3)

        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # 側視圖只需要高度
                height_grids = round((y2 - y1) / grid_px)

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
                    'height_grids': height_grids,
                    'height_cm': height_grids * 0.5,     # 每格 0.5 公分
                    'confidence': round(conf, 2)
                })

        # 依 x 座標排序（由左到右），方便對應俯視圖的作品順序
        objects.sort(key=lambda o: o['bbox'][0])

        return jsonify({'success': True, 'objects': objects, 'count': len(objects)})

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 503
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
