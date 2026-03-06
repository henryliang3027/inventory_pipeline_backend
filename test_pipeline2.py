import time
from collections import Counter

import ollama
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

# Ollama 設定
OLLAMA_MODEL = "ministral-3:3b"
SYSTEM_PROMPT_TEMPLATE = """你是一位專業的超商貨架分析員。請根據以下掃描結果清單回答用戶問題。如果清單中出現「未知商品」，請提醒用戶可能是新上架產品。

【掃描結果清單】
{scan_list}

【回答規則】
1. 只根據上方清單中的商品回答
2. 用繁體中文回答
3. 回答要簡潔明確，直接給出數量
4. 計算時請仔細核對每個商品的數量
5. 如果找不到相關商品，請說明清單中沒有該商品"""


# ========== YOLO 偵測 ==========
def detect_and_crop_bottles(image_path, conf_threshold=0.5):
    """
    使用 YOLOv11 偵測並裁切所有瓶子
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


# ========== CLIP 匹配 ==========
def match_bottle(pil_image):
    """
    使用 CLIP 匹配瓶子類型
    """
    img = pil_image.convert("RGB")
    query_embedding = clip_model.encode([img])[0]

    db_embeddings = db["embeddings"].numpy()

    query_norm = query_embedding / np.linalg.norm(query_embedding)
    db_norm = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)

    similarities = np.dot(db_norm, query_norm)

    best_idx = np.argmax(similarities)
    best_name = db["names"][best_idx]
    best_score = similarities[best_idx]

    return best_name, best_score


# ========== Ollama 問答 ==========
def build_system_prompt(scan_results: dict) -> str:
    """
    根據掃描結果動態建立系統提示詞
    """
    results_text = "\n".join([f"  - {name}: {count} 瓶" for name, count in scan_results.items()])
    return SYSTEM_PROMPT_TEMPLATE.format(scan_list=results_text)


def ask_question(scan_results: dict, question: str) -> str:
    """
    使用 Ollama 模型回答貨架掃描問題
    """
    system_prompt = build_system_prompt(scan_results)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    return response["message"]["content"]


# ========== 完整流程 ==========
def run_pipeline(image_path, conf_threshold=0.5, match_threshold=0.5):
    """
    完整辨識流程：偵測瓶子 -> 匹配資料庫 -> 統計數量

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
            name = "未知商品"

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


# ========== 互動問答 ==========
def interactive_qa(counts: dict):
    """
    進入互動問答模式
    """
    print("\n" + "="*50)
    print("進入問答模式 (輸入 'q' 離開)")
    print("="*50)

    while True:
        question = input("\n請輸入問題: ").strip()

        if question.lower() == 'q':
            print("結束問答")
            break

        if not question:
            continue

        answer = ask_question(counts, question)
        print(f"回答: {answer}")


if __name__ == "__main__":
    # 測試圖片
    test_image = "test_images/20260306_093020.jpg"

    # 執行辨識流程
    results, counts = run_pipeline(test_image)

    # 進入互動問答
    if counts:
        interactive_qa(counts)
