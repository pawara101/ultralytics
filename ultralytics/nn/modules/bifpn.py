import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFeatureFusion(nn.Module):
    def __init__(self, in_nodes, epsilon=1e-4):
        super(WeightedFeatureFusion, self).__init__()
        self.epsilon = epsilon
        # We need one weight per input node. 
        # Initialize with equal importance.
        self.weights = nn.Parameter(torch.ones(in_nodes, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        # inputs is a list of tensors [x1, x2, ...]
        assert len(inputs) == len(self.weights)
        
        # Apply ReLU to ensure non-negative weights
        w = F.relu(self.weights)
        
        # Normalize weights (Fast Normalized Fusion)
        w = w / (torch.sum(w, dim=0) + self.epsilon)
        
        # Weighted sum: w0*x0 + w1*x1 + ...
        # We expand w to allow broadcasting: w[0] -> (1,1,1,1)
        fusion = 0
        for i, x in enumerate(inputs):
            fusion += w[i] * x
            
        return fusion

class BiFPNBlock(nn.Module):
    def __init__(self, num_channels):
        super(BiFPNBlock, self).__init__()
        
        # Conv layers to process the fused features
        # Depthwise Separable Conv is used in the paper for efficiency
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, 3, 1, 1, groups=num_channels, bias=False),
                nn.Conv2d(num_channels, num_channels, 1, 1, 0, bias=True),
                nn.BatchNorm2d(num_channels),
                nn.SiLU(inplace=True) # Swish activation
            ) for _ in range(5 + 3) # 5 bottom-up + 3 intermediate top-down
        ])
        
        # Weighted Fusion layers
        # Top-down pathway fuses 2 inputs (Level i and Level i+1)
        self.td_fusions = nn.ModuleList([
            WeightedFeatureFusion(2) for _ in range(3) # P6, P5, P4 (P7 is the top, no fusion)
        ])
        
        # Bottom-up pathway fuses 3 inputs (Original i, Top-down i, Bottom-up i-1)
        # Except P3 and P7 which fuse 2 inputs
        self.bu_fusions = nn.ModuleList([
            WeightedFeatureFusion(2), # P3 (Original + Top-down)
            WeightedFeatureFusion(3), # P4
            WeightedFeatureFusion(3), # P5
            WeightedFeatureFusion(3), # P6
            WeightedFeatureFusion(2)  # P7 (Original + Bottom-up)
        ])
        
    def forward(self, features):
        # Features: [P3, P4, P5, P6, P7]
        p3_in, p4_in, p5_in, p6_in, p7_in = features
        
        # --- Top-Down Pathway ---
        # P7 stays as is for the top-down path
        p7_td = p7_in 
        
        # P6_td = Conv(Fusion(P6_in, Resize(P7_td)))
        p6_td = self.convs[0](
            self.td_fusions[0]([p6_in, F.interpolate(p7_td, scale_factor=2, mode='nearest')])
        )
        
        # P5_td = Conv(Fusion(P5_in, Resize(P6_td)))
        p5_td = self.convs[1](
            self.td_fusions[1]([p5_in, F.interpolate(p6_td, scale_factor=2, mode='nearest')])
        )
        
        # P4_td = Conv(Fusion(P4_in, Resize(P5_td)))
        p4_td = self.convs[2](
            self.td_fusions[2]([p4_in, F.interpolate(p5_td, scale_factor=2, mode='nearest')])
        )
        
        # --- Bottom-Up Pathway ---
        # P3_out = Conv(Fusion(P3_in, Resize(P4_td))) -> P3 is the bottom, so only 2 inputs
        p3_out = self.convs[3](
            self.bu_fusions[0]([p3_in, F.interpolate(p4_td, scale_factor=2, mode='nearest')])
        )
        
        # P4_out = Conv(Fusion(P4_in, P4_td, Pool(P3_out)))
        p4_out = self.convs[4](
            self.bu_fusions[1]([p4_in, p4_td, F.max_pool2d(p3_out, kernel_size=3, stride=2, padding=1)])
        )
        
        # P5_out = Conv(Fusion(P5_in, P5_td, Pool(P4_out)))
        p5_out = self.convs[5](
            self.bu_fusions[2]([p5_in, p5_td, F.max_pool2d(p4_out, kernel_size=3, stride=2, padding=1)])
        )
        
        # P6_out = Conv(Fusion(P6_in, P6_td, Pool(P5_out)))
        p6_out = self.convs[6](
            self.bu_fusions[3]([p6_in, p6_td, F.max_pool2d(p5_out, kernel_size=3, stride=2, padding=1)])
        )
        
        # P7_out = Conv(Fusion(P7_in, Pool(P6_out))) -> Note: P7_in, not P7_td
        p7_out = self.convs[7](
            self.bu_fusions[4]([p7_in, F.max_pool2d(p6_out, kernel_size=3, stride=2, padding=1)])
        )
        
        return [p3_out, p4_out, p5_out, p6_out, p7_out]
    
class BiFPN_Concat(nn.Module):
    """
    BiFPN Weighted Fusion Layer.
    Fuses N inputs using learnable weights: O = sum(w_i * I_i) / (sum(w_i) + eps)
    """
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.eps = 1e-4
        # We don't know N inputs at init (YOLO parser limitation), 
        # so we create a dynamic list or fix it if you prefer. 
        # Here we assume standard 2 or 3 inputs for BiFPN.
        # We'll initialize weights lazily or default to 3 (common max).
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        # x is a list of tensors
        if not isinstance(x, list):
            return x
        
        n = len(x)
        # Dynamic slicing if fewer than 3 inputs
        w = self.w[:n]
        
        # Fast Normalized Fusion
        w_relu = F.relu(w)
        w_norm = w_relu / (w_relu.sum() + self.eps)
        
        # Weighted sum
        res = 0
        for i, tensor in enumerate(x):
            res += w_norm[i] * tensor
            
        return res