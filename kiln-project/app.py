from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os
from collections import Counter

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best.onnx')
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"找不到模型：{MODEL_PATH}")
        model = YOLO(MODEL_PATH, task='detect')
    return model

def decode_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ========== 找紅點原點 ==========

def find_red_origin(img, search_region=None):
    """
    找圖片中的紅點（原點標記）
    search_region: (x1,y1,x2,y2) 搜尋範圍，None 表示全圖
    """
    H, W = img.shape[:2]
    if search_region:
        rx1, ry1, rx2, ry2 = search_region
    else:
        rx1, ry1, rx2, ry2 = 0, 0, W, H

    roi = img[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,  100, 80]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([155, 100, 80]), np.array([180, 255, 255]))
    mask = mask1 | mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None

    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00']) + rx1
    cy = int(M['m01'] / M['m00']) + ry1
    return cx, cy

# ========== 格線計數工具 ==========

def count_h_lines(gray, x, y1, y2, threshold=100):
    """在 x 列，y1~y2 之間數水平線條數"""
    if x < 0 or x >= gray.shape[1]: return 0
    col = gray[y1:y2, x]
    count = 0
    in_line = False
    for p in col:
        if int(p) < threshold:
            if not in_line:
                count += 1
                in_line = True
        else:
            in_line = False
    return count

def count_v_lines(gray, y, x1, x2, threshold=100):
    """在 y 行，x1~x2 之間數垂直線條數"""
    if y < 0 or y >= gray.shape[0]: return 0
    row = gray[y, x1:x2]
    count = 0
    in_line = False
    for p in row:
        if int(p) < threshold:
            if not in_line:
                count += 1
                in_line = True
        else:
            in_line = False
    return count

def stable_count(counts):
    """取最高頻且大於 0 的值"""
    if not counts:
        return 0
    freq = Counter(counts)
    for val, _ in freq.most_common():
        if val > 0:
            return val
    return 0

# ========== 主要計算邏輯 ==========

def calc_from_corner_to_origin(gray, corner_x, corner_y, origin_x, origin_y,
                                direction, scan_range=80, threshold=100):
    """
    從角點到原點之間數格線數

    direction='width' : 從 origin_x 到 corner_x，在 corner_y 附近掃水平行，數垂直線
    direction='depth' : 從 corner_y 到 origin_y，在 corner_x 附近掃垂直列，數水平線
    direction='height': 從 corner_y 到 origin_y，在 corner_x 附近掃垂直列，數水平線
    """
    results = []

    if direction == 'width':
        # 在框框右上角 y 附近，掃描 origin_x 到 corner_x，數垂直線
        for dy in range(-scan_range, scan_range, 3):
            scan_y = corner_y + dy
            c = count_v_lines(gray, scan_y, origin_x, corner_x, threshold)
            if c > 0:
                results.append(c)

    elif direction in ('depth', 'height'):
        # 在框框角點 x 附近，掃描 corner_y 到 origin_y，數水平線
        y_start = min(corner_y, origin_y)
        y_end   = max(corner_y, origin_y)
        for dx in range(-scan_range, scan_range, 3):
            scan_x = corner_x + dx
            c = count_h_lines(gray, scan_x, y_start, y_end, threshold)
            if c > 0:
                results.append(c)

    return stable_count(results)

# ========== API ==========

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': os.path.exists(MODEL_PATH)})

@app.route('/detect/top', methods=['POST'])
def detect_top():
    """
    俯視圖：計算底面積
    原點 = 格線板左下角（紅點）
    框框右上角 → 原點：
      寬度 = 原點x到框框右上角x，數垂直線
      深度 = 框框右上角y到原點y，數水平線
    """
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # 找俯視圖原點（左下角紅點）
        origin = find_red_origin(img, search_region=(0, H//2, W//3, H))
        if not origin:
            # 找不到紅點，嘗試全圖搜尋
            origin = find_red_origin(img)
        if not origin:
            return jsonify({'success': False, 'error': '找不到原點（紅點）'}), 400

        origin_x, origin_y = origin

        yolo = get_model()
        results = yolo(img, conf=0.3)
        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # 框框右上角 = (x2, y1)
                corner_x, corner_y = x2, y1

                width_grids = calc_from_corner_to_origin(
                    gray, corner_x, corner_y, origin_x, origin_y, 'width')
                depth_grids = calc_from_corner_to_origin(
                    gray, corner_x, corner_y, origin_x, origin_y, 'depth')
                area_grids = width_grids * depth_grids

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
                    'origin': [origin_x, origin_y],
                    'width_grids': width_grids,
                    'depth_grids': depth_grids,
                    'area_grids': area_grids,
                    'area_cm2': float(area_grids),
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
    側視圖：計算高度
    原點 = 側面板右下角（紅點）
    框框左上角 → 原點：
      高度 = 框框左上角y到原點y，數水平線
    """
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # 找側視圖原點（右下角紅點）
        origin = find_red_origin(img, search_region=(W//2, H//2, W, H))
        if not origin:
            origin = find_red_origin(img)
        if not origin:
            return jsonify({'success': False, 'error': '找不到原點（紅點）'}), 400

        origin_x, origin_y = origin

        yolo = get_model()
        results = yolo(img, conf=0.3)
        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # 框框左上角 = (x1, y1)
                corner_x, corner_y = x1, y1

                height_grids = calc_from_corner_to_origin(
                    gray, corner_x, corner_y, origin_x, origin_y, 'height')

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
                    'origin': [origin_x, origin_y],
                    'height_grids': height_grids,
                    'height_cm': float(height_grids),
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
