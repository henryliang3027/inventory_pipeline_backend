import time
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# 載入模型和資料庫
model = SentenceTransformer('clip-ViT-B-32')
db = torch.load("drink_db.pt")

print(f"資料庫已載入，包含 {len(db['names'])} 種飲料：{db['names']}")

def find_drink(image_path):
    """
    輸入一張圖片路徑，回傳最相似的飲料名稱和相似度
    """
    # 載入並編碼測試圖片
    img = Image.open(image_path).convert("RGB")
    query_embedding = model.encode([img])[0]

    # 計算與資料庫中每個飲料的餘弦相似度
    db_embeddings = db["embeddings"].numpy()

    # 正規化
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    db_norm = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)

    # 計算相似度
    similarities = np.dot(db_norm, query_norm)

    # 找出最相似的飲料
    best_idx = np.argmax(similarities)
    best_name = db["names"][best_idx]
    best_score = similarities[best_idx]

    return best_name, best_score, similarities

# 測試範例
if __name__ == "__main__":

    test_image = "test_crops/20260306_090626.jpg"
    print(f"\n正在辨識: {test_image}")

    start_time = time.time()
    name, score, all_scores = find_drink(test_image)
    elapsed_time = time.time() - start_time

    print("\n--- 辨識結果 ---")
    print(f"最佳匹配: {name} (相似度: {score:.4f})")
    print(f"辨識時間: {elapsed_time:.4f} 秒")

    print("\n所有飲料相似度:")
    for i, drink_name in enumerate(db["names"]):
        print(f"  {drink_name}: {all_scores[i]:.4f}")
