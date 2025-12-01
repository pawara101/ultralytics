import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from ultralytics.nn.modules.bifpn import BiFPN_Concat

def test_sanity():
    bifpn = BiFPN_Concat(dimension=1)
    t1 = torch.randn(1, 64, 8, 8)
    t2 = torch.randn(1, 32, 16, 16)
    
    print(f"t1 shape: {t1.shape}")
    print(f"t2 shape: {t2.shape}")
    
    try:
        res = bifpn([t1, t2])
        print("BiFPN forward passed!")
        print(f"Result shape: {res.shape}")
    except Exception as e:
        print(f"BiFPN forward failed as expected: {e}")

if __name__ == "__main__":
    test_sanity()
