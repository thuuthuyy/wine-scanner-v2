import glob
import os
import subprocess
import warnings
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fuzzywuzzy import fuzz, process
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Define paths using environment variables
PYTHON_EXEC = "python"
CRAFT_DIR = os.getenv("CRAFT_DIR")
CLIP4STR_DIR = os.getenv("CLIP4STR_DIR")
CRAFT_MODEL = os.getenv("CRAFT_MODEL")
CRAFT_RESULT_DIR = os.getenv("CRAFT_RESULT_DIR")
TEMP_IMAGE_PATH = os.getenv("TEMP_IMAGE_PATH")

# Initialize FastAPI app
app = FastAPI()


# Request model for image URL input
class ImageRequest(BaseModel):
    image_url: str


# Request model for wine search
class WineQuery(BaseModel):
    name: str
    producer: Optional[str] = ""
    vintage: Optional[str] = ""
    region: Optional[str] = ""
    type: Optional[str] = ""


# Connect to Qdrant vector database
client = QdrantClient("localhost", port=6333, timeout=30)
collection_name = "wine_collection"

# Load embedding model for sentence encoding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Retrieve all wine names stored in Qdrant
def get_all_wine_names():
    points = client.scroll(
        collection_name=collection_name, scroll_filter={}, limit=1000
    )
    return {point.payload["name"]: point.payload for point in points[0]}


# Cache wine database to reduce frequent queries
wine_database = get_all_wine_names()


# Utility function to clear processed image results
def clear_result_folder(result_dir):
    try:
        image_files = glob.glob(os.path.join(result_dir, "*.jpg")) + glob.glob(
            os.path.join(result_dir, "*.png")
        )
        for image in image_files:
            os.remove(image)
    except Exception as e:
        print(f"Error while clearing images in {result_dir}: {e}")


# API endpoint to extract text from an image URL
@app.post("/extract_text/")
async def extract_text(request: ImageRequest):
    image_url = request.image_url.strip()
    if not image_url:
        raise HTTPException(status_code=400, detail="No image URL provided!")

    # Download image from the given URL
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(TEMP_IMAGE_PATH, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
    else:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image from URL: {image_url}"
        )

    # Run CRAFT text detection model
    craft_command = (
        f"python {CRAFT_DIR}/test.py --trained_model={CRAFT_MODEL} "
        f"--image_path={TEMP_IMAGE_PATH} --cuda=False"
    )
    subprocess.run(
        craft_command,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Process detected bounding boxes
    image_absolute_path = os.path.abspath(TEMP_IMAGE_PATH)
    subprocess.run(
        [PYTHON_EXEC, os.path.join(CRAFT_DIR, "file_utils.py"), image_absolute_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Check if CRAFT successfully cropped text regions
    cropped_images = glob.glob(os.path.join(CRAFT_RESULT_DIR, "*.jpg"))
    if not cropped_images:
        raise HTTPException(status_code=404, detail="No cropped images found!")

    # Run CLIP4STR model for text recognition
    clip4str_command = [
        PYTHON_EXEC,
        os.path.join(CLIP4STR_DIR, "read.py"),
        os.path.join(CLIP4STR_DIR, "output", "clip4str_base16x16_d70bde1f2d.ckpt"),
        "--images_path",
        CRAFT_RESULT_DIR,
        "--device",
        "cpu",
    ]
    result = subprocess.run(
        clip4str_command, capture_output=True, text=True, check=True
    )
    recognized_text = result.stdout.strip()

    # Clear processed images to save space
    clear_result_folder(CRAFT_RESULT_DIR)

    return {"recognized_text": recognized_text}


# API endpoint to search for wine based on user query
@app.post("/search_wine/")
def search_wine(query: WineQuery):
    full_query = query.name.strip()

    # Encode query into a vector representation
    vector = model.encode(full_query).tolist()

    # Perform vector search in Qdrant
    search_results = client.search(
        collection_name=collection_name, query_vector=vector, limit=5, offset=0
    )

    # If vector search returns results, format them accordingly
    if search_results:
        results = [
            {
                "Name": result.payload["name"],
                "Details": {
                    "Producer": result.payload["winery"],
                    "Wine Type": result.payload["wine_type"],
                    "Region": result.payload["region"],
                    "Vintage": result.payload["vintage"],
                    "Price": result.payload["price"],
                    "Food Pairings": result.payload["food_pairings"],
                    "URL": result.payload["url"],
                },
            }
            for result in search_results
        ]
        return {"results": results}

    # If vector search fails, fallback to fuzzy matching
    best_match, score = process.extractOne(
        query.name, wine_database.keys(), scorer=fuzz.ratio
    )
    if score > 70:
        wine = wine_database[best_match]
        return {"name": best_match, "score": score, "details": wine}

    raise HTTPException(status_code=404, detail="No matching wine found!")
