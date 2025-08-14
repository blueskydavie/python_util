import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import os

# ------------------- Step 1: 加载模型 -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# ------------------- Step 2: 提取图像特征 -------------------
def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # 归一化
    return features[0].cpu().numpy()  # 返回为 NumPy 数组

# ------------------- Step 3: 初始化 Milvus -------------------
def init_milvus():
    connections.connect("default", host="192.168.11.20", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    ]

    schema = CollectionSchema(fields, description="Lace pattern features")
    collection = Collection(name="lace_clip_features", schema=schema)

    return collection

# ------------------- Step 4: 插入向量 -------------------
def insert_feature_to_milvus(collection, feature_vector):
    data = [[feature_vector]]
    collection.insert(data)
    collection.flush()
    print("✅ 图像向量已成功插入到 Milvus！")

# ------------------- Step 5: 主流程 -------------------
if __name__ == "__main__":
    image_path = "H:/images/东龙/图案提取和以图搜图/D924035NB.jpg"  # 替换成你的图片路径
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    print("🔍 正在提取图像特征...")
    feature_vector = extract_image_features(image_path)

    print("🔗 正在连接 Milvus 向量库...")
    collection = init_milvus()

    print("📥 正在插入特征向量...")
    insert_feature_to_milvus(collection, feature_vector)
