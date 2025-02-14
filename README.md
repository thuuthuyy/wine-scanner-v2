## Installation

- Create a virtual environment
  command: `conda create -n [YOUR_ENV_NAME] python=3.11`
- Activate the virtual environment
  command: `conda activate [YOUR_ENV_NAME]`
- Install the necessary dependencies:
  command: `pip install -r requirements.txt`


## Config

- Download the pre-trained model [CLIP-ViT-B/16 ](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) and place it in the `CLIP4STR/pretrained/clip` directory.
- Download the checkpoint [CLIP4STR-B](https://github.com/VamosC/CLIP4STR/releases/download/1.0.0/clip4str_base16x16_d70bde1f2d.ckpt) and place it in the `CLIP4STR/output` directory.

## Download and run qdrant
- Download the latest Qdrant image from Dockerhub::
  command: `docker pull qdrant/qdrant`
- Then, run the service:
  command: ``` docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant ```
- Once Qdrant is running, upload data to the `db` directory:
  command: `python upload_wine_data.py`
## Run
- Use the following command to start the API server:
command: `uvicorn api:app -- host 0000 --port 8000 --reload`
- You can find swagger documentation at `/api/docs`. 
  For example <http:localhost:8000/api/docs>.

