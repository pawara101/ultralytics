import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

model = YOLO(r".\ultralytics\cfg\models\swin-t\swin-t-12.yaml",task='detect')

model.info()           # non-zero FLOPs
# _ = model(torch.randn(1,3,640,640))  # forward pass

# Load the image
path = r"D:\Research\msc_research\data\sample-data\train\crack.jpg"
img = Image.open(path)
# Define a transform to convert the image to a tensor
transform = transforms.ToTensor()

# Apply the transform
tensor_img = transform(img)

print(tensor_img.shape)   # Example: torch.Size([3, H, W])


m1 = model(tensor_img.unsqueeze(0))  # Add batch dimension
print(m1[0].boxes)