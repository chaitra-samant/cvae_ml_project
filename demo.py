import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

from model import CVAE
import config

def load_model(model_path):
    """Loads the trained CVAE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CVAE(
        latent_dim=config.LATENT_DIM,
        num_attrs=config.NUM_ATTRS,
        base_channels=config.BASE_CHANNELS
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
        return model, device
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def process_image(image_path):
    """Preprocesses the input image to match training format (64x64)."""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0) 
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def run_demo(image_path, model_path="cvae_eyeglasses_smiling_mustache.pth"):
    model, device = load_model(model_path)
    
    img_tensor = process_image(image_path)
    if img_tensor is None: return
    img_tensor = img_tensor.to(device)
    
    initial_attr = torch.tensor([[0., 0., 0.]]).to(device) 
    
    with torch.no_grad():
        mu, _ = model.encode(img_tensor, initial_attr)
        
        attributes = {
            "Reconstruction (Neutral)": [0., 0., 0.],
        }
        
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 6, 1)
        plt.title("Original Input")
        plt.imshow((img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2)
        plt.axis('off')
        
        # Generate Variants
        for i, (name, attr_vals) in enumerate(attributes.items()):
            target_attr = torch.tensor([attr_vals]).to(device)
            
            generated_img = model.decode(mu, target_attr)
            
            plt.subplot(1, 6, i + 2)
            plt.title(name)
            plt.imshow((generated_img.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if not os.path.exists("test_face.jpg"):
        print("No 'test_face.jpg' found. Please place an image file named 'test_face.jpg' in this folder.")
    else:
        run_demo("test_face.jpg")