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
            WeightedFeatureFusion(2) for _ in range(2) # P4, P3 (P5 is the top, no fusion)
        ])
        
        # Bottom-up pathway fuses 3 inputs (Original i, Top-down i, Bottom-up i-1)
        # Except P3 and P7 which fuse 2 inputs
        self.bu_fusions = nn.ModuleList([
            WeightedFeatureFusion(2), # P2 (Original + Top-down)
            WeightedFeatureFusion(3), # P3
            WeightedFeatureFusion(3), # P4
            WeightedFeatureFusion(3)  # P5 (Original + Bottom-up)

        ])
        
    def forward(self, features):
        # Features: [P3, P4, P5, P6, P7]
        p2_in ,p3_in, p4_in, p5_in = features
        
        # --- Top-Down Pathway ---
        p5_td = p5_in  # Top level remains the same
        
        p4_td = self.convs[0](self.td_fusions[0](
            [p4_in, F.interpolate(p5_td, size=p4_in.shape[2:], mode='nearest')]
        ))

        p3_td = self.convs[1](self.td_fusions[1](
            [p3_in, F.interpolate(p4_td, size=p3_in.shape[2:], mode='nearest')]
        ))

        p2_td = self.convs[2](self.td_fusions[2](
            [p2_in, F.interpolate(p3_td, size=p2_in.shape[2:], mode='nearest')]
        ))

        # --- Bottom-Up Pathway ---
        p2_out = p2_td  # Bottom level remains the same

        p3_out = self.convs[3](
            self.bu_fusions[0](
                [p3_in, p3_td, F.max_pool2d(p2_out, kernel_size=2, stride=2)]
            )
        )

        p4_out = self.convs[4](
            self.bu_fusions[1](
                [p4_in, p4_td, F.max_pool2d(p3_out, kernel_size=2, stride=2)]
            )
        )

        p5_out = self.convs[5](
            self.bu_fusions[2](
                [p5_in, p5_td, F.max_pool2d(p4_out, kernel_size=2, stride=2)]
            )
        )

        return [p2_out, p3_out, p4_out, p5_out]