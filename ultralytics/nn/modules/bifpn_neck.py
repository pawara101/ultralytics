import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride,
                                   padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class WeightedFusion(nn.Module):
    def __init__(self, num_inputs, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, inputs):
        # inputs: list of tensors with same shape
        w = F.relu(self.w)
        w = w / (torch.sum(w) + self.eps)
        out = 0
        for wi, xi in zip(w, inputs):
            out = out + wi * xi
        return out

class BiFPNLayer(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        C = channels

        # Top-down path fusion modules
        self.wf_P6_td = WeightedFusion(2, eps)
        self.conv_P6_td = SeparableConv2d(C, C)

        self.wf_P5_td = WeightedFusion(2, eps)
        self.conv_P5_td = SeparableConv2d(C, C)

        self.wf_P4_td = WeightedFusion(2, eps)
        self.conv_P4_td = SeparableConv2d(C, C)

        self.wf_P3_td = WeightedFusion(2, eps)
        self.conv_P3_td = SeparableConv2d(C, C)

        # Bottom-up path fusion modules
        self.wf_P4_out = WeightedFusion(3, eps)  # P4, P4_td, up(P3_td)
        self.conv_P4_out = SeparableConv2d(C, C)

        self.wf_P5_out = WeightedFusion(3, eps)
        self.conv_P5_out = SeparableConv2d(C, C)

        self.wf_P6_out = WeightedFusion(3, eps)
        self.conv_P6_out = SeparableConv2d(C, C)

        self.wf_P7_out = WeightedFusion(2, eps)  # P7, up(P6_out)
        self.conv_P7_out = SeparableConv2d(C, C)

    def forward(self, inputs):
        # inputs: list [P3, P4, P5, P6, P7]
        P3, P4, P5, P6, P7 = inputs

        # ---------- Top-down ----------
        P6_td_in = [P6, F.interpolate(P7, size=P6.shape[-2:], mode='nearest')]
        P6_td = self.conv_P6_td(self.wf_P6_td(P6_td_in))

        P5_td_in = [P5, F.interpolate(P6_td, size=P5.shape[-2:], mode='nearest')]
        P5_td = self.conv_P5_td(self.wf_P5_td(P5_td_in))

        P4_td_in = [P4, F.interpolate(P5_td, size=P4.shape[-2:], mode='nearest')]
        P4_td = self.conv_P4_td(self.wf_P4_td(P4_td_in))

        P3_td_in = [P3, F.interpolate(P4_td, size=P3.shape[-2:], mode='nearest')]
        P3_td = self.conv_P3_td(self.wf_P3_td(P3_td_in))

        # ---------- Bottom-up ----------
        P4_out_in = [
            P4,
            P4_td,
            F.max_pool2d(P3_td, kernel_size=2)  # downsample
        ]
        P4_out = self.conv_P4_out(self.wf_P4_out(P4_out_in))

        P5_out_in = [
            P5,
            P5_td,
            F.max_pool2d(P4_out, kernel_size=2)
        ]
        P5_out = self.conv_P5_out(self.wf_P5_out(P5_out_in))

        P6_out_in = [
            P6,
            P6_td,
            F.max_pool2d(P5_out, kernel_size=2)
        ]
        P6_out = self.conv_P6_out(self.wf_P6_out(P6_out_in))

        P7_out_in = [
            P7,
            F.max_pool2d(P6_out, kernel_size=2)
        ]
        P7_out = self.conv_P7_out(self.wf_P7_out(P7_out_in))

        return [P3_td, P4_out, P5_out, P6_out, P7_out]

class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiFPNLayer(channels) for _ in range(num_layers)]
        )

    def forward(self, feats):
        # feats: [P3, P4, P5, P6, P7] from backbone or a previous neck
        for layer in self.layers:
            feats = layer(feats)
        return feats
