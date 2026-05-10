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

# ========== 格線工具 ==========

def get_line_positions_row(gray, y, x1, x2, threshold=100, min_gap=60):
    """在 y 行，x1~x2 範圍內找格線位置"""
    if y < 0 or y >= gray.shape[0]: return []
    row = gray[y, x1:x2]
    raw = []; in_line = False
    for i, p in enumerate(row):
        if int(p) < threshold:
            if not in_line: raw.append(i + x1); in_line = True
        else: in_line = False
    clean = []
    for p in raw:
        if not clean or p - clean[-1] >= min_gap:
            clean.append(p)
    return clean

def get_line_positions_col(gray, x, y1, y2, threshold=100, min_gap=60):
    """在 x 列，y1~y2 範圍內找格線位置"""
    if x < 0 or x >= gray.shape[1]: return []
    col = gray[y1:y2, x]
    raw = []; in_line = False
    for i, p in enumerate(col):
        if int(p) < threshold:
            if not in_line: raw.append(i + y1); in_line = True
        else: in_line = False
    clean = []
    for p in raw:
        if not clean or p - clean[-1] >= min_gap:
            clean.append(p)
    return clean

def find_bottom_boundary(gray):
    """
    找背面橫線板和底面格線板的分界 y 座標
    格線板（底面）有縱橫兩方向的線，每行投影值比純橫線板高
    從中間往下找，找到投影值突然增加的位置
    """
    H, W = gray.shape
    h_proj = np.sum(gray < 100, axis=1).astype(float)
    # 從圖片中間往下找，找到投影值突然從低跳高的地方
    for y in range(H // 3, H * 2 // 3):
        if h_proj[y] < 200 and h_proj[y + 50] > 500:
            return y + 25
    # 找不到就用圖片高度的 55%
    return int(H * 0.55)

# ========== 俯視圖計算 ==========

def calc_width_grids(gray, bbox, W, min_gap=80):
    """
    計算物件寬度格數（俯視圖）
    在框框上方掃描，數框框 x 範圍內的垂直格線數
    框框內垂直格線數 - 1 = 寬度格數
    """
    x1, y1, x2, y2 = bbox
    results = []
    for dy in range(20, 500, 5):
        scan_y = y1 - dy
        if scan_y < 0: break
        pos = get_line_positions_row(gray, scan_y, x1, W - 50, min_gap=min_gap)
        in_bbox = [x for x in pos if x1 <= x <= x2]
        if len(in_bbox) > 1:
            results.append(len(in_bbox))
    if not results: return 0
    mode = Counter(results).most_common(1)[0][0]
    return max(0, mode - 1)

def calc_depth_grids(gray, bbox, W, board_top, min_gap=80):
    """
    計算物件深度格數（俯視圖）
    在框框右側掃描，數框框 y 範圍內的水平格線數
    框框內水平格線數 - 1 = 深度格數
    """
    x1, y1, x2, y2 = bbox
    results = []
    for dx in range(20, 500, 5):
        scan_x = x2 + dx
        if scan_x >= W: break
        pos = get_line_positions_col(gray, scan_x, board_top, y2, min_gap=min_gap)
        in_bbox = [y for y in pos if y1 <= y <= y2]
        if len(in_bbox) > 1:
            results.append(len(in_bbox))
    if not results: return 0
    mode = Counter(results).most_common(1)[0][0]
    return max(0, mode - 1)

# ========== 側視圖計算 ==========

def calc_height_grids(gray, bbox, H, min_gap=80):
    """
    計算物件高度格數（側視圖）

    原理：
    - 背面橫線板底部 = 底面格線板前緣（兩面接在一起）
    - 物件靠在角落，背面橫線被物件遮住的條數 = 高度格數
    - 在框框 x 範圍內掃描背面橫線板，跟左側空白參考比較
    - 差值眾數 = 物件遮住的橫線數 = 高度格數
    """
    x1, y1, x2, y2 = bbox
    W = gray.shape[1]

    # 找背面橫線板底部
    bottom_boundary = find_bottom_boundary(gray)

    # 左側空白參考（框框左側 300~1000px，遠離物件）
    ref_counts = []
    for dx in range(300, 1000, 20):
        scan_x = x1 - dx
        if scan_x < 0: break
        pos = get_line_positions_col(gray, scan_x, 0, bottom_boundary, min_gap=min_gap)
        if len(pos) > 5:
            ref_counts.append(len(pos))
    if not ref_counts: return 0
    ref_total = Counter(ref_counts).most_common(1)[0][0]

    # 在框框 x 範圍內掃描，數被物件遮住了幾條
    hidden_counts = []
    for x in range(x1 + 30, x2 - 30, 10):
        pos = get_line_positions_col(gray, x, 0, bottom_boundary, min_gap=min_gap)
        hidden = ref_total - len(pos)
        if hidden >= 0:
            hidden_counts.append(hidden)

    if not hidden_counts: return 0
    # 取最高頻的非零值（排除完全空白的列）
    freq = Counter(hidden_counts)
    for count, _ in freq.most_common():
        if count > 0:
            return count
    return 0

def find_board_top(gray):
    """找格線板頂部 y 座標（俯視圖用）"""
    h_proj = np.sum(gray < 100, axis=1).astype(float)
    threshold = h_proj.max() * 0.3
    for y in range(gray.shape[0]):
        if h_proj[y] > threshold:
            return y
    return 0

# ========== API ==========

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': os.path.exists(MODEL_PATH)})

@app.route('/detect/top', methods=['POST'])
def detect_top():
    """俯視圖：計算底面積（寬度 × 深度格數）"""
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        yolo = get_model()
        results = yolo(img, conf=0.3)
        board_top = find_board_top(gray)
        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                bbox = (x1, y1, x2, y2)

                width_grids = calc_width_grids(gray, bbox, W)
                depth_grids = calc_depth_grids(gray, bbox, W, board_top)
                area_grids  = width_grids * depth_grids

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
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
    """側視圖：計算高度格數"""
    try:
        data = request.get_json()
        img = decode_image(data['image'])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        yolo = get_model()
        results = yolo(img, conf=0.3)
        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                bbox = (x1, y1, x2, y2)

                height_grids = calc_height_grids(gray, bbox, H)

                objects.append({
                    'id': len(objects) + 1,
                    'bbox': [x1, y1, x2, y2],
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
