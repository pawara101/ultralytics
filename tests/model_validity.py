import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from ultralytics.nn.modules import *

# model = YOLO(r"D:\Research\msc_research\yolo\Ultralytics\ultralytics-101\ultralytics\ultralytics\cfg\models\12\yolo12-bifpn.yaml",
#              verbose=True,)  # build a new model from YAML

# model.info()           # non-zero FLOPs
test_img = torch.randn(1,3,224,224)

print("=== Testing custom module sequence ===")
print("Input shape:", test_img.shape)
conv_1 = Conv(c1=3, c2=64, k=3, s=2) # [1, 64, 112, 112]
conv_2 = Conv(c1=64, c2=128, k=3, s=2) # [1, 128, 56, 56]
c3k2_1 = nn.ModuleList(
    [C3k2(c1=128,c2=256,e=0.25,c3k=False),
    C3k2(c1=256,c2=256,e=0.25,c3k=False)]
) # [1, 256, 56, 56]
conv_3 = Conv(c1=256, c2=256, k=3, s=2) # [1, 256, 28, 28] P3
c3k2_2 = nn.ModuleList(
    [C3k2(c1=256,c2=512,e=0.25,c3k=False),
    C3k2(c1=512,c2=512,e=0.25,c3k=False)]
)# [1, 512, 28, 28]
conv_4 = Conv(c1=512, c2=512, k=3, s=2) # [1, 512, 14, 14] P4
a2c2f_1 = nn.ModuleList(
    [A2C2f(c1=512,c2=512,e=0.25), A2C2f(c1=512,c2=512,e=0.25), A2C2f(c1=512,c2=512,e=0.25), A2C2f(c1=512,c2=512,e=0.25)]
) # [1, 512, 14, 14]
conv_5 = Conv(c1=512, c2=1024, k=3, s=2) # [1, 1024, 7, 7] P5
a2c2f_2 = nn.ModuleList(
    [A2C2f(c1=1024,c2=1024,e=0.25), A2C2f(c1=1024,c2=1024,e=0.25), A2C2f(c1=1024,c2=1024,e=0.25), A2C2f(c1=1024,c2=1024,e=0.25)]
) # [1, 1024, 7, 7]

up1 = nn.Upsample(size=None, scale_factor=2, mode='nearest')  # Upsample by a factor of 2 [1, 1024, 14, 14]

bipfn1 = BiFPNBlock(num_channels=512)  # BiFPN Block expecting 512 channels
# P3_out, P4_out, P5_out = bipfn1([conv_5(test_img), conv_4(conv_3(test_img)), conv_3(c3k2_1(test_img))])

m1 = nn.Sequential(
    conv_1,
    conv_2,
    *c3k2_1,
    conv_3, #p3
    *c3k2_2,
    conv_4, #p4
    *a2c2f_1,
    conv_5, #p5
    *a2c2f_2,
)
out = m1(test_img)
print(out.shape)

## Use hooks to get intermediate layer outputs and their shapes
# shapes = {}
# def hook(m, inp, out):
#     shapes[m.__class__.__name__ + f"_{len(shapes)}"] = tuple(out.shape) if hasattr(out, 'shape') else str(type(out))

# hooks = []
# for m in model.modules():
#     if len(list(m.children())) == 0:  # leaf modules only
#         hooks.append(m.register_forward_hook(hook))

# _ = model(test_img)

# for k, v in shapes.items():
#     print(k, v)

# for h in hooks: h.remove()


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