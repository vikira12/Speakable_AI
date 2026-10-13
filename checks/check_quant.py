"""학습 패키지 정수 추론(spk_train.quant)과 기준 구현(ref_model.py)의 비트 일치 + 버그 주입 검사

사용: python checks/check_quant.py
  무작위 모델 3종(실제 범위 / 반올림 경계가 자주 나오는 작은 shift / 음수 FC) × 입력 64개를 비교하고,
  반올림 규칙을 일부러 틀리게 바꾼 3가지 버그가 모두 검출되는지 확인한다.
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from spk_train.paths import fpga_tools       # noqa: E402
sys.path.insert(0, fpga_tools())
import ref_model as rm                       # noqa: E402
from spk_train import quant                  # noqa: E402
from spk_train.model import LAYERS           # noqa: E402

NUM = {n: i + 1 for i, (n, *_r) in enumerate(LAYERS)}
NUM["fc11"] = 11


def random_model(rng, kind):
    qlayers = []
    for n, t, i, o, k, s, g in LAYERS + [("fc11", "fc", 64, 64, None, 0, 1)]:
        shape = {"conv": (o, i // g, k, k) if k else (o, i), "depthwise": (o, 1, k, k), "fc": (o, i)}[t]
        if kind == "real":
            w = rng.integers(-127, 128, shape)
            b = rng.integers(-20000, 20000, o)
            mult = rng.integers(2 ** 30, 2 ** 31, o)
            shift = rng.integers(36, 42, o)
        else:                                 # 작은 값 + 작은 shift: acc × mult / 2^shift 가 정확히 .5 인 경우가 많음
            w = rng.integers(-2, 3, shape)
            b = rng.integers(-5, 6, o)
            mult = rng.integers(1, 4, o)
            shift = rng.integers(0, 4, o)
        qlayers.append({"name": n, "type": t, "kernel": k, "stride": s, "groups": g, "relu": t != "fc",
                        "w_int8": w.astype(np.int8), "b_int32": b.astype(np.int32),
                        "mult": mult.astype(np.int64), "shift": shift.astype(np.int64)})
    return qlayers


def compare(qlayers, x):
    params = {NUM[L["name"]]: {"weight": L["w_int8"], "bias": L["b_int32"], "mult": L["mult"].astype(np.int32),
                               "shift": L["shift"]} for L in qlayers}
    mine = quant.int_forward(qlayers, torch.from_numpy(x.astype(np.int64)), check_acc=False)
    bad = 0
    for k in range(len(x)):
        ref = rm.run_model(params, x[k].astype(np.int8))
        for a, b in zip(mine, ref):
            bad += int(np.sum(a[k].numpy().reshape(-1) != b.reshape(-1)))
    return bad


def run_all():
    rng = np.random.default_rng(1)
    total = 0
    for kind, lo, hi in (("real", -128, 128), ("tie", -3, 4), ("tie", -128, 128)):
        q = random_model(rng, kind)
        x = rng.integers(lo, hi, (64, 1, 40, 64))
        total += compare(q, x)
    return total


def main():
    bad = run_all()
    print(f"정상 구현: 불일치 {bad}개 → {'통과' if bad == 0 else '실패'}")
    ok = bad == 0

    orig_requant = quant._requant

    def bug_ties_even(acc, mult, shift, relu):
        extra = (1,) * (acc.dim() - 2)
        prod = (acc * mult.reshape(1, -1, *extra)).double() / (2.0 ** shift.reshape(1, -1, *extra).double())
        q = torch.round(prod).to(torch.int64).clamp(-128, 127)
        return q.clamp_min(0) if relu else q

    def bug_floor(acc, mult, shift, relu):
        extra = (1,) * (acc.dim() - 2)
        q = ((acc * mult.reshape(1, -1, *extra)) >> shift.reshape(1, -1, *extra)).clamp(-128, 127)
        return q.clamp_min(0) if relu else q

    def bug_no_sat(acc, mult, shift, relu):
        q = orig_requant(acc, mult, shift, False)
        q = torch.where(q == -128, torch.full_like(q, -127), q)
        return q.clamp_min(0) if relu else q

    for name, fn in (("반올림 짝수 방향", bug_ties_even), ("내림 시프트", bug_floor), ("포화 하한 −127", bug_no_sat)):
        quant._requant = fn
        b = run_all()
        quant._requant = orig_requant
        print(f"버그 주입 [{name}]: 불일치 {b}개 → {'검출' if b > 0 else '검출 실패'}")
        ok &= b > 0
    print("결과: " + ("모두 통과" if ok else "실패"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
