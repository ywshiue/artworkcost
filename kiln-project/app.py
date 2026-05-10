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

# ========== 固定原點（從空白隔板照片量好） ==========
# 俯視圖原點：格線板左下角紅點（空白圖解析度 4284x5712）
TOP_ORIGIN   = {'x': 537,  'y': 4154, 'w': 4284, 'h': 5712}
# 側視圖原點：側面板右下角紅點（空白圖解析度 4284x5712）
SIDE_ORIGIN  = {'x': 3514, 'y': 4163, 'w': 4284, 'h': 5712}

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

def check_is_screenshot(img):
    """
    判斷是否為截圖（非相機照片）
    iPhone 相機照片比例為 3:4 (0.75) 或 4:3 (1.33)
    截圖比例為螢幕比例，通常是 9:19.5 ≈ 0.46 或其倒數 2.17，
    或橫式截圖約 1.4~1.6
    允許誤差 ±0.05
    """
    H, W = img.shape[:2]
    ratio = W / H
    # 標準相機比例
    valid_ratios = [3/4, 4/3, 16/9, 9/16]
    for valid in valid_ratios:
        if abs(ratio - valid) < 0.08:
            return False  # 正常照片
    return True  # 可能是截圖

def scale_origin(origin, img_w, img_h):
    """
    把固定原點從空白圖解析度縮放到當前圖片解析度
    若拍攝方向不同（直式vs橫式），自動旋轉對齊後再縮放
    """
    ref_w, ref_h = origin['w'], origin['h']
    ref_is_portrait = ref_h > ref_w
    img_is_portrait = img_h > img_w

    if ref_is_portrait == img_is_portrait:
        # 方向相同，直接縮放
        ox = int(origin['x'] * img_w / ref_w)
        oy = int(origin['y'] * img_h / ref_h)
    else:
        # 方向不同（直式→橫式），順時針旋轉90度
        # 新x = ref_h - 原y，新y = 原x
        rotated_x = ref_h - origin['y']
        rotated_y = origin['x']
        rotated_w = ref_h
        rotated_h = ref_w
        ox = int(rotated_x * img_w / rotated_w)
        oy = int(rotated_y * img_h / rotated_h)

    return ox, oy

# ========== 格線計數 ==========

def count_v_lines(gray, y, x1, x2, threshold=100):
    """在 y 行，x1~x2 之間數垂直線條數"""
    if y < 0 or y >= gray.shape[0]: return 0
    x1, x2 = max(0, x1), min(gray.shape[1], x2)
    if x1 >= x2: return 0
    row = gray[y, x1:x2]
    count = 0; in_line = False
    for p in row:
        if int(p) < threshold:
            if not in_line: count += 1; in_line = True
        else: in_line = False
    return count

def count_h_lines(gray, x, y1, y2, threshold=100):
    """在 x 列，y1~y2 之間數水平線條數"""
    if x < 0 or x >= gray.shape[1]: return 0
    y1, y2 = max(0, y1), min(gray.shape[0], y2)
    if y1 >= y2: return 0
    col = gray[y1:y2, x]
    count = 0; in_line = False
    for p in col:
        if int(p) < threshold:
            if not in_line: count += 1; in_line = True
        else: in_line = False
    return count

def stable_count(results):
    """取最高頻且大於 0 的值"""
    if not results: return 0
    freq = Counter(results)
    for val, _ in freq.most_common():
        if val > 0: return val
    return 0

def calc_grids(gray, corner_x, corner_y, origin_x, origin_y, direction, scan_range=80):
    """
    從角點到原點數格線，掃描位置只取空白區域（不穿越物件）

    俯視圖寬度 (direction='width'):
        掃描行 y = 框框上方空白區（corner_y 往上 scan_range px 內）
        掃描範圍 x = origin_x 到 corner_x
        數垂直線

    俯視圖深度 (direction='depth'):
        掃描列 x = 框框右側空白區（corner_x 往右 scan_range px 內）
        掃描範圍 y = corner_y 到 origin_y
        數水平線

    側視圖高度 (direction='height'):
        掃描列 x = 框框左側空白區（corner_x 往左 scan_range px 內）
        掃描範圍 y = corner_y 到 origin_y
        數水平線
    """
    results = []

    if direction == 'width':
        # 在框框右上角 y 的「上方」空白格線板掃描，不穿越物件
        for dy in range(5, scan_range, 3):
            scan_y = corner_y - dy  # 往上
            if scan_y < 0: break
            c = count_v_lines(gray, scan_y, origin_x, corner_x)
            if c > 0: results.append(c)

    elif direction == 'depth':
        # 在框框右上角 x 的「右側」空白格線板掃描，不穿越物件
        y_start = min(corner_y, origin_y)
        y_end   = max(corner_y, origin_y)
        for dx in range(5, scan_range, 3):
            scan_x = corner_x + dx  # 往右
            if scan_x >= gray.shape[1]: break
            c = count_h_lines(gray, scan_x, y_start, y_end)
            if c > 0: results.append(c)

    elif direction == 'height':
        # 在框框左上角 x 的「左側」空白側面板掃描，不穿越物件
        y_start = min(corner_y, origin_y)
        y_end   = max(corner_y, origin_y)
        for dx in range(5, scan_range, 3):
            scan_x = corner_x - dx  # 往左
            if scan_x < 0: break
            c = count_h_lines(gray, scan_x, y_start, y_end)
            if c > 0: results.append(c)

    return stable_count(results)

# ========== API ==========

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': os.path.exists(MODEL_PATH),
        'top_origin': TOP_ORIGIN,
        'side_origin': SIDE_ORIGIN,
    })

@app.route('/detect/top', methods=['POST'])
def detect_top():
    """
    俯視圖偵測
    YOLO 找框框右上角 (x2, y1)
    寬度 = 原點x 到 角點x 的垂直線數
    深度 = 角點y 到 原點y 的水平線數
    """
    try:
        data = request.get_json()
        img = decode_image(data['image'])

        if check_is_screenshot(img):
            return jsonify({'success': False, 'error': '請上傳相機拍攝的照片，不要使用截圖'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # 縮放原點到當前圖片尺寸
        ox, oy = scale_origin(TOP_ORIGIN, W, H)

        yolo = get_model()
        results = yolo(img, conf=0.3)
        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # 框框右上角
                corner_x, corner_y = x2, y1

                width_grids = calc_grids(gray, corner_x, corner_y, ox, oy, 'width')
                depth_grids = calc_grids(gray, corner_x, corner_y, ox, oy, 'depth')
                area_grids  = width_grids * depth_grids

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
                    'corner': [corner_x, corner_y],
                    'origin': [ox, oy],
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
    側視圖偵測
    YOLO 找框框左上角 (x1, y1)
    高度 = 角點y 到 原點y 的水平線數
    """
    try:
        data = request.get_json()
        img = decode_image(data['image'])

        if check_is_screenshot(img):
            return jsonify({'success': False, 'error': '請上傳相機拍攝的照片，不要使用截圖'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # 縮放原點到當前圖片尺寸
        ox, oy = scale_origin(SIDE_ORIGIN, W, H)

        yolo = get_model()
        results = yolo(img, conf=0.3)
        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # 框框左上角
                corner_x, corner_y = x1, y1

                height_grids = calc_grids(gray, corner_x, corner_y, ox, oy, 'height')

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
                    'corner': [corner_x, corner_y],
                    'origin': [ox, oy],
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
