import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

model = YOLO(r"D:\Research\msc_research\yolo\Ultralytics\ultralytics-101\ultralytics\ultralytics\cfg\models\12\yolo12-bifpn.yaml",
             verbose=True,)  # build a new model from YAML

# model.info()           # non-zero FLOPs
test_img = torch.randn(1,3,224,224)
# outs = model.model(test_img)  # forward pass

## Use hooks to get intermediate layer outputs and their shapes
shapes = {}
def hook(m, inp, out):
    shapes[m.__class__.__name__ + f"_{len(shapes)}"] = tuple(out.shape) if hasattr(out, 'shape') else str(type(out))

hooks = []
for m in model.modules():
    if len(list(m.children())) == 0:  # leaf modules only
        hooks.append(m.register_forward_hook(hook))

_ = model(test_img)

for k, v in shapes.items():
    print(k, v)

for h in hooks: h.remove()


# out1 = model.predict(test_img, conf=0.25, iou=0.45, half=False, augment=False, verbose=True)  # predict on an image
# Load the image
# path = r"D:\Research\msc_research\data\sample-data\train\crack.jpg"
# img = Image.open(path)
# # Define a transform to convert the image to a tensor
# transform = transforms.ToTensor()

# # Apply the transform
# tensor_img = transform(img)

# print(tensor_img.shape)   # Example: torch.Size([3, H, W])


# m1 = model(tensor_img.unsqueeze(0))  # Add batch dimension
# print(m1[0].boxes)