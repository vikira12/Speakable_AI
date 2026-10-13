"""임베딩 CNN (FPGA IP 구조 고정) 과 학습 헤드

Conv16 → DW16 → PW32 → DW32(s2) → PW64 → DW64 → PW64 → DW64(s2) → PW64 → GAP → FC64
conv1은 stride 2. 모든 Conv/DW/PW 뒤 BN + ReLU, skip 없음. 추론 출력은 FC 출력의 L2 정규화.
파라미터 이름은 model_pilot_v01 checkpoint.pt의 encoder와 같다 (그 가중치로 시작할 수 있음).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# (이름, 종류, in_c, out_c, kernel, stride, groups)
LAYERS = [
    ("conv1", "conv", 1, 16, 3, 2, 1),
    ("dw2", "depthwise", 16, 16, 3, 1, 16),
    ("pw3", "conv", 16, 32, 1, 1, 1),
    ("dw4", "depthwise", 32, 32, 3, 2, 32),
    ("pw5", "conv", 32, 64, 1, 1, 1),
    ("dw6", "depthwise", 64, 64, 3, 1, 64),
    ("pw7", "conv", 64, 64, 1, 1, 1),
    ("dw8", "depthwise", 64, 64, 3, 2, 64),
    ("pw9", "conv", 64, 64, 1, 1, 1),
]
EMB_DIM = 64


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k, stride, groups):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleDict({n: ConvBNReLU(i, o, k, s, g) for n, _, i, o, k, s, g in LAYERS})
        self.fc = nn.Linear(64, EMB_DIM)

    def forward(self, x):
        for b in self.blocks.values():
            x = b(x)
        return self.fc(x.mean(dim=(2, 3)))


class AAMSoftmax(nn.Module):
    """Additive angular margin softmax: 같은 클래스는 모으고 다른 클래스는 각도 margin만큼 벌린다"""

    def __init__(self, n_classes, scale=30.0, margin=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, EMB_DIM) * 0.01)
        self.s, self.m = scale, margin

    def forward(self, emb, label):
        cos = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        th = torch.acos(cos)
        target = torch.cos(th + self.m)
        # θ + m 이 π를 넘으면 단조성이 깨지므로 그 구간은 cos − m·sin(m) 으로 대체
        target = torch.where(th + self.m < math.pi, target, cos - self.m * math.sin(self.m))
        onehot = F.one_hot(label, cos.shape[1]).bool()
        logits = self.s * torch.where(onehot, target, cos)
        return F.cross_entropy(logits, label), (cos.argmax(1) == label).float().mean()


# ---------------------------------------------------------------- BN 접기 + 가짜 양자화 (QAT)

def round_away(x):
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def fake_quant(x, scale, lo=-128, hi=127):
    """정수 규약과 같은 반올림. 역전파는 그대로 통과 (STE)"""
    q = torch.clamp(round_away(x / scale), lo, hi) * scale
    return x + (q - x).detach()


class FoldedEncoder(nn.Module):
    """BN을 접은 모델 (conv에 bias). quant=True면 정수 추론을 흉내 낸다 (가중치 채널별, 활성화 텐서별)"""

    def __init__(self, enc: Encoder):
        super().__init__()
        self.convs = nn.ModuleDict()
        for name, _, i, o, k, s, g in LAYERS:
            blk = enc.blocks[name]
            conv = nn.Conv2d(i, o, k, s, k // 2, groups=g, bias=True)
            bn = blk.bn
            std = torch.sqrt(bn.running_var + bn.eps)
            w = blk.conv.weight.detach() * (bn.weight.detach() / std).reshape(-1, 1, 1, 1)
            b = bn.bias.detach() - bn.running_mean * bn.weight.detach() / std
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(b)
            self.convs[name] = conv
        self.fc = nn.Linear(64, EMB_DIM)
        self.fc.load_state_dict(enc.fc.state_dict())
        self.quant = False
        self.register_buffer("act_scale", torch.ones(len(LAYERS) + 2))   # input, 9개 레이어, fc

    def forward(self, x, return_all=False):
        outs = []
        s = self.act_scale
        if self.quant:
            x = fake_quant(x, s[0])
        for li, (name, *_rest) in enumerate(LAYERS):
            conv = self.convs[name]
            w = conv.weight
            if self.quant:
                ws = w.detach().abs().amax(dim=(1, 2, 3)).clamp_min(1e-12) / 127
                w = fake_quant(w, ws.reshape(-1, 1, 1, 1), -127, 127)
            x = F.relu(F.conv2d(x, w, conv.bias, conv.stride, conv.padding, groups=conv.groups))
            if self.quant:
                x = fake_quant(x, s[li + 1])
            outs.append(x)
        x = x.mean(dim=(2, 3))
        if self.quant:
            x = fake_quant(x, s[len(LAYERS)])                         # GAP: pw9와 같은 scale
        outs.append(x)
        w = self.fc.weight
        if self.quant:
            ws = w.detach().abs().amax(dim=1).clamp_min(1e-12) / 127
            w = fake_quant(w, ws.reshape(-1, 1), -127, 127)
        x = F.linear(x, w, self.fc.bias)
        if self.quant:
            x = fake_quant(x, s[len(LAYERS) + 1])
        outs.append(x)
        return outs if return_all else x
