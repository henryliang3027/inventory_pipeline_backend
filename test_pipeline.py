import os
import time
from collections import Counter

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

# ========== 載入模型 ==========
print("載入模型中...")
yolo_model = YOLO("yolo11n.pt")
clip_model = SentenceTransformer('clip-ViT-B-32')
db = torch.load("drink_db.pt")

print(f"資料庫已載入，包含 {len(db['names'])} 種飲料：{db['names']}")

# COCO 資料集中 bottle 的 class ID 是 39
BOTTLE_CLASS_ID = 39


def detect_and_crop_bottles(image_path, conf_threshold=0.5):
    """
    使用 YOLOv11 偵測並裁切所有瓶子 (不儲存檔案，直接回傳 PIL Image 列表)
    """
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)
    img = Image.open(image_path)

    cropped_images = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])

            if cls_id == BOTTLE_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])

                cropped = img.crop((x1, y1, x2, y2))
                cropped_images.append({
                    "image": cropped,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })

    return cropped_images


def match_bottle(pil_image):
    """
    使用 CLIP 匹配瓶子類型
    """
    img = pil_image.convert("RGB")
    query_embedding = clip_model.encode([img])[0]

    db_embeddings = db["embeddings"].numpy()

    # 正規化
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    db_norm = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)

    # 計算相似度
    similarities = np.dot(db_norm, query_norm)

    best_idx = np.argmax(similarities)
    best_name = db["names"][best_idx]
    best_score = similarities[best_idx]

    return best_name, best_score


def run_pipeline(image_path, conf_threshold=0.5, match_threshold=0.5):
    """
    完整辨識流程：偵測瓶子 -> 匹配資料庫 -> 統計數量

    Args:
        image_path: 輸入圖片路徑
        conf_threshold: YOLO 信心度閾值
        match_threshold: CLIP 匹配閾值 (低於此值視為未知)

    Returns:
        results: 每個瓶子的辨識結果列表
        counts: 各類飲料的數量統計
    """
    print(f"\n{'='*50}")
    print(f"處理圖片: {image_path}")
    print(f"{'='*50}")

    start_time = time.time()

    # Step 1: 偵測並裁切瓶子
    print("\n[Step 1] YOLO 偵測瓶子...")
    detect_start = time.time()
    cropped_bottles = detect_and_crop_bottles(image_path, conf_threshold)
    detect_time = time.time() - detect_start
    print(f"  偵測到 {len(cropped_bottles)} 個瓶子 (耗時: {detect_time:.3f}s)")

    if len(cropped_bottles) == 0:
        print("未偵測到任何瓶子！")
        return [], {}

    # Step 2: 匹配每個瓶子
    print("\n[Step 2] CLIP 辨識飲料...")
    match_start = time.time()
    results = []

    for i, bottle in enumerate(cropped_bottles):
        name, score = match_bottle(bottle["image"])

        if score < match_threshold:
            name = "未知飲料"

        results.append({
            "index": i,
            "name": name,
            "match_score": score,
            "detect_conf": bottle["conf"],
            "bbox": bottle["bbox"]
        })

        print(f"  瓶子 {i}: {name} (相似度: {score:.4f})")

    match_time = time.time() - match_start
    print(f"  辨識完成 (耗時: {match_time:.3f}s)")

    # Step 3: 統計數量
    print("\n[Step 3] 統計結果...")
    names = [r["name"] for r in results]
    counts = dict(Counter(names))

    total_time = time.time() - start_time

    # 輸出結果
    print("\n" + "="*50)
    print("辨識結果統計")
    print("="*50)
    for drink_name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {drink_name}: {count} 瓶")

    print(f"\n總計: {len(results)} 瓶")
    print(f"總耗時: {total_time:.3f}s")

    return results, counts


if __name__ == "__main__":
    # 測試圖片
    test_image = "test_images/20260306_093020.jpg"

    results, counts = run_pipeline(test_image)
