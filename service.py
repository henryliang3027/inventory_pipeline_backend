from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import base64
import io
import os
import time
from collections import Counter
from PIL import ImageDraw, ImageFont

import ollama
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer


class Base64ImageRequest(BaseModel):
    image_base64: str
    question: str = "請統計商品"


# ========== Global models ==========
yolo_model = None
clip_model = None
db = None

BOTTLE_CLASS_ID = 39
OLLAMA_MODEL = "ministral-3:3b"
MATCH_THRESHOLD = 0.65
CONF_THRESHOLD = 0.5

SYSTEM_PROMPT_TEMPLATE = """你是一位專業的超商貨架分析員。請根據以下掃描結果清單回答用戶問題。如果清單中出現「未知商品」，請提醒用戶可能是新上架產品。

【掃描結果清單】
{scan_list}

【回答規則】
1. 只根據上方清單中的商品回答
2. 用繁體中文回答
3. 回答要簡潔明確，直接給出數量
4. 計算時請仔細核對每個商品的數量
5. 如果找不到相關商品，請說明清單中沒有該商品"""


# ========== Lifespan: load models on startup ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, clip_model, db
    print("載入模型中...")
    yolo_model = YOLO("yolo11m.pt")
    clip_model = SentenceTransformer('clip-ViT-B-32')
    db = torch.load("drink_db.pt")
    print(f"資料庫已載入，包含 {len(db['names'])} 種飲料：{db['names']}")
    yield


app = FastAPI(lifespan=lifespan)


# ========== Pipeline helpers ==========
DEBUG_DIR = "cropped_bottles"

def detect_and_crop_bottles(pil_image: Image.Image, conf_threshold: float = CONF_THRESHOLD):
    results = yolo_model(pil_image, conf=conf_threshold, verbose=False)
    cropped_images = []

    # debug_img = pil_image.copy()
    # draw = ImageDraw.Draw(debug_img)

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != BOTTLE_CLASS_ID:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cropped_images.append({
                "image": pil_image.crop((x1, y1, x2, y2)),
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
            })
            # draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # draw.text((x1, max(y1 - 16, 0)), f"{conf:.2f}", fill="red")

    # os.makedirs(DEBUG_DIR, exist_ok=True)
    # debug_path = os.path.join(DEBUG_DIR, f"debug_{int(time.time() * 1000)}.jpg")
    # debug_img.save(debug_path)
    # print(f"[DEBUG] saved bbox image: {debug_path}")

    return cropped_images


def match_bottle(pil_image: Image.Image):
    img = pil_image.convert("RGB")
    query_embedding = clip_model.encode([img])[0]

    db_embeddings = db["embeddings"].numpy()
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    db_norm = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    similarities = np.dot(db_norm, query_norm)

    best_idx = np.argmax(similarities)
    return db["names"][best_idx], float(similarities[best_idx])


def ask_question(counts: dict, question: str) -> str:
    scan_list = "\n".join(f"  - {name}: {count} 瓶" for name, count in counts.items())
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(scan_list=scan_list)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response["message"]["content"]


# ========== Endpoint ==========
@app.post("/inventory_base64")
async def inventory_base64(request: Base64ImageRequest):
    try:
        image_data = base64.b64decode(request.image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")

    start_time = time.time()

    # Step 1: detect & crop bottles
    cropped_bottles = detect_and_crop_bottles(pil_image)
    if not cropped_bottles:
        return JSONResponse(content={
            "status": 0,
            "data": "未偵測到任何瓶子",
        })

    # Step 2: match each bottle via CLIP
    names = []
    for bottle in cropped_bottles:
        name, score = match_bottle(bottle["image"])
        names.append(name if score >= MATCH_THRESHOLD else "未知商品")

    counts = dict(Counter(names))
    print(counts)

    # Step 3: answer question via Ollama
    answer = ask_question(counts, request.question)

    print(f"Elapsed Time: {round(time.time() - start_time, 3)}")

    return JSONResponse(content={
        "status": 1,
        "data": answer,
    })



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)