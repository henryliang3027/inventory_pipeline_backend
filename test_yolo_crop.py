import os
from ultralytics import YOLO
from PIL import Image

# 載入 YOLOv11 模型
model = YOLO("yolo12s.pt")

# COCO 資料集中 bottle 的 class ID 是 39
BOTTLE_CLASS_ID = 39

def crop_bottles(image_path, output_dir="cropped_bottles", conf_threshold=0.6):
    """
    使用 YOLOv11 偵測圖片中的所有瓶子並裁切儲存

    Args:
        image_path: 輸入圖片路徑
        output_dir: 輸出資料夾
        conf_threshold: 信心度閾值

    Returns:
        cropped_paths: 裁切後圖片的路徑列表
    """
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    # 執行偵測
    results = model(image_path, conf=conf_threshold)

    # 載入原圖
    img = Image.open(image_path)

    cropped_paths = []
    bottle_count = 0

    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])

            # 只處理 bottle 類別
            if cls_id == BOTTLE_CLASS_ID:
                # 取得邊界框座標
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = float(box.conf[0])

                # 裁切瓶子區域
                cropped = img.crop((x1, y1, x2, y2))

                # 儲存裁切圖片
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                crop_path = os.path.join(output_dir, f"{base_name}_bottle_{bottle_count}.jpg")
                cropped.save(crop_path)

                cropped_paths.append(crop_path)
                print(f"[瓶子 {bottle_count}] 信心度: {conf:.2f}, 座標: ({x1}, {y1}, {x2}, {y2})")
                bottle_count += 1

    print(f"\n共偵測到 {bottle_count} 個瓶子，已儲存至 {output_dir}/")
    return cropped_paths


if __name__ == "__main__":
    test_image = "test_images/20260306_093020.jpg"
    print(f"正在處理: {test_image}\n")

    cropped = crop_bottles(test_image)

    print(f"\n裁切完成，共 {len(cropped)} 張圖片:")
    for path in cropped:
        print(f"  - {path}")
