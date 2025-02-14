import json
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Generate a unique ID from a URL using hashing


def generate_unique_id(url: str) -> int:
    """Generate a 12-digit unique integer ID from a given URL."""
    return int(hashlib.md5(url.encode()).hexdigest(), 16) % (10**12)


# Path to the JSON file containing wine details
JSON_FILE_PATH = "wine_details3.json"

# Initialize Qdrant Client (for local or remote server)
client = QdrantClient("localhost", port=6333, timeout=30)

# Collection name in Qdrant
COLLECTION_NAME = "wine_collection"

# Load the sentence transformer model for text encoding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create the collection if it does not exist
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=model.get_sentence_embedding_dimension(),  # Vector size for "all-MiniLM-L6-v2" is 384
            distance=Distance.COSINE,
        ),
    )

# Load wine data from the JSON file
with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
    wine_data_list = json.load(file)

# Define batch size for upserting data
BATCH_SIZE = 1000

total_uploaded = 0

# Process data in batches
for i in range(0, len(wine_data_list), BATCH_SIZE):
    batch = wine_data_list[i : i + BATCH_SIZE]
    points = []

    for wine in batch:
        if "url" not in wine or not wine["url"]:  # Skip records without a URL
            continue

        unique_id = generate_unique_id(wine["url"])  # Generate unique ID based on URL
        vector = model.encode(wine["name"]).tolist()  # Encode wine name to vector

        # Construct payload with all available wine attributes
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

    # Upsert batch into Qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    total_uploaded += len(points)
    print(f"Uploaded {total_uploaded}/{len(wine_data_list)} records...")

print("Upload completed successfully!")
