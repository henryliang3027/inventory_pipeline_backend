import os
import base64
import io
import time
from collections import Counter
from contextlib import asynccontextmanager

import torch
import numpy as np
import chromadb
import ollama
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import subprocess

# ========== Model & DB Config ==========
BOTTLE_CLASS_ID = 39
OLLAMA_MODEL = "ministral-3:3b"
CONF_THRESHOLD = 0.5

# --- Cosine 門檻值建議 ---
# 0.0 ~ 0.2: 極度相似 (同一產品)
# 0.2 ~ 0.35: 相似 (同系列不同角度)
# > 0.35: 視為未知商品
COSINE_THRESHOLD = 0.35

class Base64ImageRequest(BaseModel):
    image_base64: str
    question: str = "請統計商品"

# ========== Global Objects ==========
yolo_model = None
clip_model = None
chroma_client = None
collection = None

SYSTEM_PROMPT_TEMPLATE = """你是一位專業的超商貨架分析員。請根據以下掃描結果清單回答用戶問題。

【掃描結果清單】
{scan_list}

【回答規則】
1. 只根據上方清單回答，若有「未知商品」也請如實告知。
2. 用繁體中文回答，簡潔明確。
3. 如果清單為空或找不到商品，請說明清單中沒有相關商品。"""

client = OpenAI(
    base_url="http://127.0.0.1:8881/v1",
    api_key="no-key-needed",  # 本地通常不驗證，填任意字串即可
)


# llama-server 進程
llama_process = None

LLAMA_SERVER_CMD = [
    "./llama.cpp/build/bin/llama-server",
    "-m",
    "ministral/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    "--mmproj",
    "ministral/mmproj-F16.gguf",
    "--host",
    "0.0.0.0",
    "--port",
    "8881",
    "--ctx-size",
    "4096",
    "-ngl",
    "-1",
]


def start_llama_server():
    global llama_process
    print("Starting llama-server...")
    llama_process = subprocess.Popen(
        LLAMA_SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # 等待 llama-server 啟動
    time.sleep(5)
    print(f"llama-server started with PID: {llama_process.pid}")


def stop_llama_server():
    global llama_process
    if llama_process:
        print(f"Stopping llama-server (PID: {llama_process.pid})...")
        llama_process.terminate()
        try:
            llama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_process.kill()
        print("llama-server stopped.")
        llama_process = None

# ========== Lifespan: Initialization ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, clip_model, chroma_client, collection
    print("🚀 正在啟動系統並載入模型...")
    
    # 1. 載入視覺模型
    yolo_model = YOLO("yolo11m.pt")
    clip_model = SentenceTransformer('clip-ViT-B-32')
    
    # 2. 初始化 ChromaDB (持久化儲存於本地資料夾)
    chroma_client = chromadb.PersistentClient(path="./drink_vector_db")
    collection = chroma_client.get_or_create_collection(name="drink_catalog", metadata={"hnsw:space": "cosine"})
    
    existing_count = collection.count()
    print(f"📦 ChromaDB 已就緒，目前資料庫包含 {existing_count} 筆特徵資料。")

    # 啟動時執行
    start_llama_server()
    yield
    # 關閉時執行
    stop_llama_server()

app = FastAPI(
    title="Inventory Pipeline API",
    description="test",
    version="1.0.1",
    lifespan=lifespan)

# ========== Helper Functions ==========

def detect_and_crop_bottles(pil_image: Image.Image):
    results = yolo_model(pil_image, conf=CONF_THRESHOLD, verbose=False)
    cropped_images = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == BOTTLE_CLASS_ID:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cropped_images.append(pil_image.crop((x1, y1, x2, y2)))
    return cropped_images

def match_with_chroma(pil_image: Image.Image):
    """取代原本的 .pt 比對，改用 ChromaDB 查詢"""
    img_emb = clip_model.encode(pil_image).tolist()
    
    results = collection.query(
        query_embeddings=[img_emb],
        n_results=1,
        include=["metadatas", "distances"]
    )

    print(f"result: {results}")
    
    if not results['ids'][0]:
        return "未知商品"
    
    dist = results['distances'][0][0]
    metadata = results['metadatas'][0][0]
    
    return f"{metadata['color']}{metadata['display_name']}"

# ========== CRUD Endpoints (管理資料庫) ==========

@app.post("/db/add", summary="[CRUD] 新增飲料特徵到資料庫")
async def add_to_db(
    name: str = Form(...),
    color: str = Form(""),
    file: UploadFile = File(...)
):
    """上傳一張 crop 好的瓶子，存入 ChromaDB"""
    image = Image.open(file.file).convert("RGB")
    embedding = clip_model.encode(image).tolist()
    
    collection.upsert(
        ids=[name], # 以品名作為唯一 ID
        embeddings=[embedding],
        metadatas=[{"display_name": name, "color": color}]
    )
    return {"status": "success", "message": f"已存入: {color}{name}"}

@app.get("/db/list", summary="[CRUD] 列出目前所有商品")
async def list_db():
    results = collection.get()
    return {"total": len(results['ids']), "items": results['metadatas']}

@app.delete("/db/{name}", summary="[CRUD] 刪除特定商品")
async def delete_item(name: str):
    collection.delete(ids=[name])
    return {"status": "deleted", "item": name}

# ========== Inference Endpoint (主要功能) ==========

@app.post("/inventory_base64")
async def inventory_base64(request: Base64ImageRequest):
    start_time = time.time()
    
    # 1. 解碼圖片
    try:
        image_data = base64.b64decode(request.image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="圖片解碼失敗")

    # 2. YOLO 偵測與裁切
    crops = detect_and_crop_bottles(pil_image)
    if not crops:
        return {"status": 1, "data": "貨架上看起來沒有瓶子。"}

    # 3. ChromaDB 向量比對
    detected_names = [match_with_chroma(img) for img in crops]
    counts = dict(Counter(detected_names))
    
    # 4. 組合成文字給 Ollama
    scan_list_str = "\n".join([f"- {k}: {v} 瓶" for k, v in counts.items()])
    print(f"掃描結果:\n{scan_list_str}")

    # 5. llama.cpp 推理
    response = client.chat.completions.create(
        model="ministral_3_3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(scan_list=scan_list_str)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.question},
                ],
            },
        ],
        temperature=0,
    )

    
    print(f"⚡ 耗時: {round(time.time() - start_time, 2)}s")
    return {"status": 1, "data": response.choices[0].message.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)