import json
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, CollectionConfig
from sentence_transformers import SentenceTransformer

# Hàm tạo ID duy nhất từ URL bằng hash
def generate_unique_id(url):
    return int(hashlib.md5(url.encode()).hexdigest(), 16) % (10**12)  # Chuyển hash thành số nguyên 12 chữ số

# Đường dẫn file JSON
json_file_path = "wine_details3.json"

# Khởi tạo Qdrant Client (local hoặc remote server)
client = QdrantClient("localhost", port=6333, timeout=30)

# Tên collection trong Qdrant
collection_name = "wine_collection"

# Khởi tạo mô hình để mã hóa văn bản thành vector
model = SentenceTransformer("all-MiniLM-L6-v2")


# Tạo collection nếu chưa tồn tại
if collection_name not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),  # Kích thước vector của "all-MiniLM-L6-v2" là 384
    )

# Đọc dữ liệu từ file JSON
with open(json_file_path, "r", encoding="utf-8") as file:
    wine_data_list = json.load(file)

# Chia nhỏ dữ liệu thành từng batch nhỏ (ví dụ: 1000 điểm một lần)
BATCH_SIZE = 1000
total_uploaded = 0

for i in range(0, len(wine_data_list), BATCH_SIZE):
    batch = wine_data_list[i : i + BATCH_SIZE]  # Cắt dữ liệu thành từng phần nhỏ
    points = []

    for wine in batch:
        if "url" not in wine or not wine["url"]:  # Nếu không có URL, bỏ qua
            continue
        
        unique_id = generate_unique_id(wine["url"])  # Sử dụng URL làm ID duy nhất
        vector = model.encode(wine["name"]).tolist()  # Mã hóa thành vector
        payload = {
            "wine_id": wine["wine_id"],
            "name": wine["name"],
            "winery": wine.get("winery", ""),
            "grapes": wine.get("grapes", ""),
            "vintage": wine.get("vintage", ""),
            "price": wine.get("price", ""),
            "rating": wine.get("rating", ""),
            "wine_type": wine.get("wine_type", ""),
            "country": wine.get("country", ""),
            "region": wine.get("region", ""),
            "wine_style": wine.get("wine_style", ""),
            "alcohol_content": wine.get("alcohol_content", ""),
            "food_pairings": wine.get("food_pairings", ""),
            "level": wine.get("level", ""),
            "tannin": wine.get("tannin", ""),
            "sweetness": wine.get("sweetness", ""),
            "acidity": wine.get("acidity", ""),
            "allergens": wine.get("allergens", ""),
            "bottle_closure": wine.get("bottle_closure", ""),
            "wine_description": wine.get("wine_description", ""),
            "url": wine.get("url", ""),
        }
        points.append(PointStruct(id=unique_id, vector=vector, payload=payload))

    # Upsert từng batch vào Qdrant
    client.upsert(collection_name=collection_name, points=points)
    
    total_uploaded += len(points)
    print(f"Uploaded {total_uploaded}/{len(wine_data_list)} records...")

print("Upload completed successfully!")
