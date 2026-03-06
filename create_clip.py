import os
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# 初始化模型
model = SentenceTransformer('clip-ViT-B-32')

BASE_DIR = "my_crops"
save_data = {"names": [], "embeddings": []}

print("--- 開始建立 3 種飲料的特徵資料庫 ---")

# 遍歷 3 個飲料資料夾
for drink_name in os.listdir(BASE_DIR):
    drink_path = os.path.join(BASE_DIR, drink_name)
    
    if os.path.isdir(drink_path):
        img_list = [os.path.join(drink_path, f) for f in os.listdir(drink_path) if f.endswith(('.jpg', '.png'))]
        
        if len(img_list) > 0:
            print(f"正在處理: {drink_name} (包含 {len(img_list)} 張圖)")
            
            # 1. 提取這幾張圖的特徵
            images = [Image.open(img).convert("RGB") for img in img_list]
            embeddings = model.encode(images) # 得到 (N, 512) 的矩陣
            
            # 2. 取平均值 (將 3 張圖的特徵融合為 1 個)
            mean_embedding = np.mean(embeddings, axis=0)
            
            # 3. 存入資料庫
            save_data["names"].append(drink_name)
            save_data["embeddings"].append(mean_embedding)

# 轉換為 Tensor 並儲存
save_data["embeddings"] = torch.tensor(np.array(save_data["embeddings"]))
torch.save(save_data, "drink_db.pt")

print("\n[成功] 資料庫已建立！檔案：drink_db.pt")
print(f"目前類別：{save_data['names']}")