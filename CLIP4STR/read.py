import os
import argparse
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images_path', type=str, help='Images to read')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    # Debugging - In ra đường dẫn checkpoint
    # print(f"Checkpoint loaded: {args.checkpoint}")

    # Load model from checkpoint
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)

    # Get image transformation
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    # Get sorted list of image files
    files = sorted([x for x in os.listdir(args.images_path) if x.endswith(('png', 'jpeg', 'jpg'))])

    # Danh sách để ghép nối kết quả
    recognized_texts = []

    # Perform inference on each image
    for fname in files:
        filename = os.path.join(args.images_path, fname)
        image = Image.open(filename).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)
        
        # Forward qua model
        p = model(image).softmax(-1)
    
        # Giải mã
        pred, p_decoded = model.tokenizer.decode(p)
        recognized_texts.append(pred[0])  # Lưu kết quả nhận diện vào danh sách

    # Ghép nối các kết quả và in ra màn hình
    final_text = " ".join(recognized_texts)
    print(final_text)

if __name__ == '__main__':
    main()
