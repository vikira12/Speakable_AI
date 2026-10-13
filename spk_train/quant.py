"""INT8 변환과 정수 추론 시뮬레이션 (하드웨어·ARM과 같은 정수 규약)

  대칭 INT8, zero-point 0, 가중치 출력 채널별 scale (−127~127), 활성화 레이어별 scale
  bias INT32 = round_away(b / (s_x · s_w)), 누산 INT32
  재양자화 sat8(round_away(acc × mult / 2^shift)), mult는 양의 INT32 [2^30, 2^31), shift 0~62
  GAP sat8(round_away(합 / (H·W))), scale은 입력과 같음. FC 출력도 INT8 (활성화 없음)
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from .model import LAYERS, FoldedEncoder

INT32_MAX = 2 ** 31 - 1


def round_away_np(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


@torch.no_grad()
def calibrate(folded: FoldedEncoder, x, percentile=99.99, batch=512):
    """활성화 scale = |값|의 백분위수 / 127. x: 정규화된 입력 (N,1,40,64)"""
    folded.quant = False
    dev = next(folded.parameters()).device
    gen = torch.Generator().manual_seed(0)
    vals = [[] for _ in range(len(LAYERS) + 2)]

    def keep(t):                                                # 배치마다 최대 20만 개 표본 + 최댓값
        t = t.abs().flatten().cpu()
        if len(t) > 200_000:
            t = torch.cat([t[torch.randint(len(t), (200_000,), generator=gen)], t.max().reshape(1)])
        return t

    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(dev).float()
        outs = folded(xb, return_all=True)                      # 9개 레이어 + GAP + FC
        vals[0].append(keep(xb))
        for li in range(len(LAYERS)):
            vals[li + 1].append(keep(outs[li]))
        vals[len(LAYERS) + 1].append(keep(outs[-1]))
    scales = [max(float(np.percentile(torch.cat(v).numpy(), percentile)), 1e-6) / 127 for v in vals]
    folded.act_scale.copy_(torch.tensor(scales))
    return scales


def mult_shift(m):
    """실수 배율 m ≈ mult / 2^shift, mult ∈ [2^30, 2^31)"""
    shift = 30 - math.floor(math.log2(m))
    mult = int(round(m * 2 ** shift))
    if mult >= 2 ** 31:
        mult //= 2
        shift -= 1
    if shift > 62:                                             # 아주 작은 배율: 정밀도를 줄여 범위 안으로
        mult = max(int(round(m * 2 ** 62)), 1)
        shift = 62
    if shift < 0:
        raise ValueError(f"재양자화 배율 {m}이 너무 큼")
    return mult, shift


def to_integer(folded: FoldedEncoder):
    """→ 레이어별 정수 파라미터 dict 리스트 (conv 9개 + fc). 각 dict에 scale 정보 포함"""
    s = folded.act_scale.cpu().numpy().astype(np.float64)
    out = []
    specs = [(n, kind, i, o, k, st, g, folded.convs[n]) for n, kind, i, o, k, st, g in LAYERS]
    specs.append(("fc11", "fc", 64, 64, None, 0, 1, folded.fc))
    for li, (name, kind, i, o, k, st, g, mod) in enumerate(specs):
        w = mod.weight.detach().cpu().double().numpy()
        b = mod.bias.detach().cpu().double().numpy()
        s_x = s[li] if li < len(LAYERS) else s[len(LAYERS)]     # fc 입력 = GAP 출력 = pw9 scale
        s_y = s[li + 1] if li < len(LAYERS) else s[len(LAYERS) + 1]
        s_w = np.maximum(np.abs(w.reshape(o, -1)).max(axis=1), 1e-12) / 127
        w_q = np.clip(round_away_np(w / s_w.reshape(-1, *([1] * (w.ndim - 1)))), -127, 127).astype(np.int8)
        b_q = round_away_np(b / (s_x * s_w))
        if np.abs(b_q).max() > INT32_MAX:
            raise ValueError(f"{name} bias가 INT32 범위를 넘음")
        ms = [mult_shift(s_x * sw / s_y) for sw in s_w]
        out.append({
            "name": name, "type": kind, "in_c": i, "out_c": o, "kernel": k, "stride": st, "groups": g,
            "relu": kind != "fc", "input_scale": float(s_x), "output_scale": float(s_y),
            "weight_scale": s_w.tolist(), "bias_scale": (s_x * s_w).tolist(),
            "w_fp32": w.astype(np.float32), "b_fp32": b.astype(np.float32),
            "w_int8": w_q, "b_int32": b_q.astype(np.int32),
            "mult": np.array([m for m, _ in ms], dtype=np.int64), "shift": np.array([sh for _, sh in ms], dtype=np.int64),
        })
    return out, float(s[0])


def quantize_input(norm, input_scale):
    """정규화 입력(float) → INT8. 기기와 같이 float32를 한 번 거친다"""
    q = np.asarray(norm, dtype=np.float32).astype(np.float64) / input_scale
    return np.clip(round_away_np(q), -128, 127).astype(np.int8)


# ---------------------------------------------------------------- 정수 추론 (torch, 배치)

def _requant(acc, mult, shift, relu):
    extra = (1,) * (acc.dim() - 2)
    prod = acc * mult.reshape(1, -1, *extra)
    sh = shift.reshape(1, -1, *extra)
    half = torch.where(sh > 0, torch.ones_like(sh) << (sh - 1).clamp_min(0), torch.zeros_like(sh))
    q = torch.sign(prod) * ((prod.abs() + half) >> sh)
    q = q.clamp(-128, 127)
    return q.clamp_min(0) if relu else q


@torch.no_grad()
def int_forward(qlayers, x_int8, check_acc=True):
    """x_int8: (N,1,40,64) 정수 텐서 → [conv 9개, gap, fc] 정수 출력 (int64 텐서)"""
    t = {}
    for L in qlayers:
        t[L["name"]] = (torch.from_numpy(L["w_int8"].astype(np.float64)), torch.from_numpy(L["b_int32"].astype(np.int64)),
                        torch.from_numpy(L["mult"]), torch.from_numpy(L["shift"]))
    x = torch.as_tensor(x_int8).to(torch.int64)
    outs = []
    for L in qlayers[:-1]:
        w, b, m, sh = t[L["name"]]
        acc = F.conv2d(x.double(), w, None, L["stride"], L["kernel"] // 2, groups=L["groups"])
        acc = torch.round(acc).to(torch.int64) + b.reshape(1, -1, 1, 1)   # float64 합은 2^53 미만이라 정확
        if check_acc and acc.abs().max() > INT32_MAX:
            raise OverflowError(f"{L['name']} 누산이 INT32 범위를 넘음")
        x = _requant(acc, m, sh, L["relu"])
        outs.append(x)
    cnt = x.shape[2] * x.shape[3]
    sm = x.sum(dim=(2, 3))
    x = (torch.sign(sm) * ((2 * sm.abs() + cnt) // (2 * cnt))).clamp(-128, 127)
    outs.append(x)
    L = qlayers[-1]
    w, b, m, sh = t[L["name"]]
    acc = torch.round(x.double() @ w.T).to(torch.int64) + b
    if check_acc and acc.abs().max() > INT32_MAX:
        raise OverflowError("fc11 누산이 INT32 범위를 넘음")
    outs.append(_requant(acc, m, sh, False))
    return outs


def int_embed(qlayers, x_int8, batch=1024):
    """INT8 경로 임베딩 (FC INT8 출력의 L2 정규화, float)"""
    embs = []
    for i in range(0, len(x_int8), batch):
        e = int_forward(qlayers, x_int8[i:i + batch])[-1].double()
        embs.append(F.normalize(e, dim=1).float())
    return torch.cat(embs)
