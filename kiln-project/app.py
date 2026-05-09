from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)
CORS(app)

# 使用 ONNX 模型，記憶體需求較小，適合免費伺服器
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.onnx')
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}")
        model = YOLO(MODEL_PATH, task='detect')
    return model

def decode_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': os.path.exists(MODEL_PATH)})

@app.route('/detect/top', methods=['POST'])
def detect_top():
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

                w_grids = round((x2 - x1) / grid_px)
                h_grids = round((y2 - y1) / grid_px)
                area_grids = w_grids * h_grids

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
                    'width_grids': w_grids,
                    'height_grids': h_grids,
                    'area_grids': area_grids,
                    'area_cm2': area_grids * 0.25,
                    'confidence': round(conf, 2)
                })

        return jsonify({'success': True, 'objects': objects, 'count': len(objects)})

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 503
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect/side', methods=['POST'])
def detect_side():
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

                height_grids = round((y2 - y1) / grid_px)

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
                    'height_grids': height_grids,
                    'height_cm': height_grids * 0.5,
                    'confidence': round(conf, 2)
                })

        objects.sort(key=lambda o: o['bbox'][0])
        return jsonify({'success': True, 'objects': objects, 'count': len(objects)})

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 503
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
