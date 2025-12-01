import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from ultralytics.nn.tasks import parse_model
from ultralytics.nn.modules import BiFPN_Concat

def test_bifpn_parsing():
    # Mocking a model dictionary that uses BiFPN_Concat
    # We need a simple structure: Input -> Conv -> BiFPN_Concat
    
    # Define a simple model config
    model_dict = {
        'nc': 80,
        'scales': {'n': [0.33, 0.25, 1024]}, # Dummy scale
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2]], # 1-P2/4
            [-1, 1, 'Conv', [256, 3, 2]], # 2-P3/8
        ],
        'head': [
            [[-1, -2], 1, 'BiFPN_Concat', [1]], # 3. Should accept 2 inputs. 
        ]
    }
    
    # Mock input channels
    ch = 3
    
    print("Parsing model...")
    try:
        model, save = parse_model(model_dict, ch, verbose=True)
        print("Model parsed successfully.")
        
        # Check the last layer
        last_layer = model[-1]
        print(f"Last layer type: {type(last_layer)}")
        
        # Check output channels of the last layer (which updates 'ch' list in parse_model)
        # We can't easily check the internal 'ch' list from here without modifying parse_model to return it,
        # but we can check the model structure.
        
        # Create dummy input
        img = torch.randn(1, 3, 64, 64)
        print("Running forward pass...")
        output = model(img)
        print("Forward pass successful.")
        print(f"Output shape: {output.shape}")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bifpn_parsing()
