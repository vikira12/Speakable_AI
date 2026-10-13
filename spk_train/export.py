"""학습 결과 → model_pilot_v01과 같은 형식의 산출물 디렉터리

speakable_arm_ref/tools/import_ai_artifacts.py, export_c.py, export_frontend_c.py가 그대로 읽는 형식이다.
테스트 벡터는 기준 구현(ref_frontend.py, ref_model.py)으로 다시 계산해 이 패키지의 정수 추론과
비트 단위로 같은지 확인한 뒤에만 저장을 마친다.
"""
import hashlib
import importlib
import json
import os
import shutil
import sys
import numpy as np
import torch
import torch.nn.functional as F

from .data import write_wav
from .frontend import BASE_DIR, FastFrontend
from .quant import int_forward, quantize_input

LAYER_NUM = {"conv1": 1, "dw2": 2, "pw3": 3, "dw4": 4, "pw5": 5, "dw6": 6, "pw7": 7, "dw8": 8, "pw9": 9, "gap10": 10, "fc11": 11}


def _ref_tools(tools_dir):
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    return importlib.import_module("ref_frontend"), importlib.import_module("ref_model")


def _jdump(obj, path):
    with open(path, "w", encoding="utf8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def layer_specs(qlayers):
    """model_spec.json / quant_params.json의 layers 항목"""
    specs = []
    shape = [1, 40, 64]
    for L in qlayers:
        if L["type"] == "fc":
            pw9 = specs[-1]
            specs.append({"name": "gap10", "type": "gap", "input_shape": pw9["output_shape"], "output_shape": [64],
                          "input_scale": pw9["output_scale"], "output_scale": pw9["output_scale"], "zero_point": 0, "relu": False})
            in_shape, out_shape = [64], [64]
        else:
            k, s = L["kernel"], L["stride"]
            in_shape = shape
            shape = [L["out_c"], (shape[1] + 2 * (k // 2) - k) // s + 1, (shape[2] + 2 * (k // 2) - k) // s + 1]
            out_shape = shape
        specs.append({
            "name": L["name"], "type": L["type"], "input_shape": in_shape, "output_shape": out_shape,
            "input_scale": L["input_scale"], "output_scale": L["output_scale"], "zero_point": 0, "relu": L["relu"],
            "weight_scale": L["weight_scale"], "bias_scale": L["bias_scale"],
            "multiplier": [int(v) for v in L["mult"]], "shift": [int(v) for v in L["shift"]],
            "stride": L["stride"], "padding": (L["kernel"] // 2) if L["kernel"] else 0, "groups": L["groups"],
            "kernel": L["kernel"], "weight_layout": "OI" if L["type"] == "fc" else "OIHW",
        })
    return specs


def export_artifacts(out_dir, *, folded, head, classes, qlayers, input_scale, mean, std, match_params,
                     test_samples, tools_dir, status, report, config, log=print):
    """test_samples: [(id, pcm16 int16 배열)]. 기준 구현과 불일치하면 예외"""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # ---- 전처리 파라미터 ----
    pp = json.load(open(os.path.join(BASE_DIR, "preprocess.json"), encoding="utf8"))
    pp["normalization_status"] = status
    _jdump(pp, os.path.join(out_dir, "preprocess.json"))
    for f in ("hann.npy", "mel_filter.npy"):
        shutil.copy(os.path.join(BASE_DIR, f), os.path.join(out_dir, f))
    np.save(os.path.join(out_dir, "norm_mean.npy"), np.asarray(mean, dtype=np.float64))
    np.save(os.path.join(out_dir, "norm_std.npy"), np.asarray(std, dtype=np.float64))

    # ---- 모델 사양·양자화 파라미터·가중치 ----
    specs = layer_specs(qlayers)
    _jdump({"status": status, "layers": specs, "activation_layout": "NCHW", "skip_connections": False,
            "gap_fc_location": "PS_ARM", "normalization_location": "PS_ARM"}, os.path.join(out_dir, "model_spec.json"))
    scales = {"input": input_scale}
    scales.update({s["name"]: s["output_scale"] for s in specs})
    _jdump({"status": status, "input_scale": input_scale, "zero_point": 0, "scales": scales,
            "rounding": "nearest_ties_away_from_zero", "activation_range": [-128, 127], "weight_range": [-127, 127],
            "accumulator": "INT32_checked", "bias": "INT32", "multiplier": "positive_INT32", "product": "INT64",
            "shift_range": [0, 62], "layers": specs}, os.path.join(out_dir, "quant_params.json"))
    for L in qlayers:
        n = L["name"]
        np.save(os.path.join(out_dir, f"{n}_weight_fp32.npy"), L["w_fp32"])
        np.save(os.path.join(out_dir, f"{n}_bias_fp32.npy"), L["b_fp32"])
        np.save(os.path.join(out_dir, f"{n}_weight_int8.npy"), L["w_int8"])
        np.save(os.path.join(out_dir, f"{n}_bias_int32.npy"), L["b_int32"])

    mp = {"status": status, **match_params, "registration_samples_per_expression_target": 5}
    _jdump(mp, os.path.join(out_dir, "matching_params.json"))

    # ---- 테스트 벡터 (기준 구현으로 전처리, 두 정수 구현 비교) ----
    ref_frontend, ref_model = _ref_tools(tools_dir)
    fe_ref = ref_frontend.Frontend(out_dir)
    fe_fast = FastFrontend()
    ref_params = {LAYER_NUM[L["name"]]: {"weight": L["w_int8"], "bias": L["b_int32"],
                                        "mult": L["mult"].astype(np.int32), "shift": L["shift"]} for L in qlayers}
    names = [L["name"] for L in qlayers[:-1]] + ["gap10", "fc11"]
    folded_q = folded.quant
    folded.quant = False
    dev = next(folded.parameters()).device
    tv_list, cos_list, fast_diff = [], [], 0
    for sid, pcm in test_samples:
        d = os.path.join(out_dir, "test_vectors", sid)
        os.makedirs(d)
        write_wav(os.path.join(d, "audio.wav"), pcm)
        r = fe_ref.run(np.asarray(pcm, dtype=np.int16))
        for k in ("mel_raw", "mel_64", "log_mel"):
            np.save(os.path.join(d, f"{k}.npy"), r[k])
        x_fp = r["input_fp32"].astype(np.float32).reshape(1, 1, 40, 64)
        x_q = r["input_int8"].reshape(1, 1, 40, 64)
        np.save(os.path.join(d, "input_fp32.npy"), x_fp)
        np.save(os.path.join(d, "input_int8.npy"), x_q)

        fast_norm = (fe_fast.log_mel(pcm) - mean[:, None]) / np.maximum(std, pp["std_floor"])[:, None]
        fast_diff += int(np.sum(quantize_input(fast_norm, input_scale).reshape(-1) != x_q.reshape(-1)))

        mine = [o.numpy() for o in int_forward(qlayers, torch.from_numpy(x_q.astype(np.int64)))]
        ref = ref_model.run_model(ref_params, x_q.reshape(1, 40, 64).astype(np.int8))
        for n, a, b in zip(names, mine, ref):
            if not np.array_equal(a.reshape(-1), b.reshape(-1).astype(np.int64)):
                raise AssertionError(f"{sid} {n}: 학습 패키지 정수 추론과 ref_model.py 결과가 다름 "
                                     f"(다른 원소 {int(np.sum(a.reshape(-1) != b.reshape(-1)))}개)")
            np.save(os.path.join(d, f"{n}_int8.npy"), a.astype(np.int8))

        with torch.no_grad():
            fp = folded(torch.from_numpy(x_fp).to(dev), return_all=True)
        for n, a in zip(names, fp):
            np.save(os.path.join(d, f"{n}_fp32.npy"), a.cpu().numpy().astype(np.float32))
        e_fp = F.normalize(fp[-1].double(), dim=1).cpu().numpy()
        e_q = F.normalize(torch.from_numpy(mine[-1]).double(), dim=1).numpy()
        np.save(os.path.join(d, "embedding_fp32_l2.npy"), e_fp.astype(np.float32))
        np.save(os.path.join(d, "embedding_int8_dequant_l2.npy"), e_q.astype(np.float32))
        cos = float(np.sum(e_fp * e_q))
        cos_list.append(cos)
        tv_list.append({"candidate_id": sid, "fp32_int8_embedding_cosine": cos})
    folded.quant = folded_q
    _jdump(tv_list, os.path.join(out_dir, "test_vectors.json"))

    # ---- 체크포인트·보고서·모델 카드·해시 ----
    torch.save({"encoder_folded": {k: v.cpu() for k, v in folded.state_dict().items()},
                "training_head": {k: v.cpu() for k, v in head.state_dict().items()}, "classes": classes,
                "mean": torch.tensor(mean), "std": torch.tensor(std), "config": config, "status": status},
               os.path.join(out_dir, "checkpoint.pt"))
    report = dict(report)
    report["test_vectors"] = {"count": len(test_samples), "ref_model_bit_exact": True,
                              "fp32_int8_cosine_mean": float(np.mean(cos_list)) if cos_list else None,
                              "fp32_int8_cosine_min": float(np.min(cos_list)) if cos_list else None,
                              "fast_frontend_input_int8_mismatch": fast_diff}
    _jdump(report, os.path.join(out_dir, "training_report.json"))
    with open(os.path.join(out_dir, "MODEL_CARD.txt"), "w", encoding="utf8", newline="\n") as f:
        f.write(model_card(status, report, mp, config))
    sha = {}
    for root, _, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            sha[os.path.relpath(p, out_dir).replace(os.sep, "/")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    _jdump({"version": config.get("model_version", "speakable"), "status": status, "sha256": sha}, os.path.join(out_dir, "manifest.json"))
    log(f"산출물 저장 → {out_dir} (테스트 벡터 {len(test_samples)}개, ref_model.py와 비트 일치, "
        f"FP32/INT8 임베딩 cosine 평균 {report['test_vectors']['fp32_int8_cosine_mean']:.4f})")
    return report


def model_card(status, report, mp, config):
    tv = report["test_vectors"]
    ev = report.get("test_int8") or {}
    d = ev.get("decision") or {}
    lines = [
        f"Speakable 임베딩 모델 {config.get('model_version', '')} — 상태: {status}",
        "",
        f"데이터: {report.get('data', {}).get('summary', '')}",
        f"학습: 클래스 기준 {config['class_key']}, epoch {config['epochs']}, AAM-softmax s={config['aam_scale']} m={config['aam_margin']}",
        f"증강: 원본 + {config['n_variants'] - 1}개 ({config.get('augment')}), SpecAugment",
        "입력: mono PCM16 16kHz, 0.5~3초 → Mel 40×64 (전처리는 preprocess.json, 정규화 통계는 학습 데이터)",
        "CNN: Conv16 → DW16 → PW32 → DW32 → PW64 → DW64 → PW64 → DW64 → PW64 (stride 2: conv1/dw4/dw8), BN fold, ReLU",
        "ARM: GAP → FC64 → L2 정규화. 학습 헤드는 추론에서 제외",
        "정수 규약: 대칭 INT8, zero-point 0, INT32 bias·누산, 채널별 양의 INT32 mult × 누산(INT64) → shift, 0에서 먼 방향 반올림",
        "",
        f"테스트 벡터 {tv['count']}개: ref_model.py와 레이어 출력 비트 일치, FP32/INT8 임베딩 cosine 평균 {tv['fp32_int8_cosine_mean']:.4f} 최저 {tv['fp32_int8_cosine_min']:.4f}",
        f"시험셋 INT8 매칭 ({ev.get('mode', '-')} 모드): 1위 정답 {ev.get('top1', float('nan')):.3f}, 3위 안 {ev.get('top3', float('nan')):.3f}",
        f"  확실·정답 {d.get('reg_confident_correct', float('nan')):.3f}, 확실·오답 {d.get('reg_confident_wrong', float('nan')):.3f}, "
        f"미등록 오인식 {d.get('unreg_confident') if d.get('unreg_confident') is not None else '-'}",
        f"매칭 파라미터 (검증셋 보정): α={mp['alpha']} T={mp['threshold']:.2f} M={mp['margin']:.2f} "
        f"즉시 T={mp['immediate_threshold']:.2f} M={mp['immediate_margin']:.2f}",
        "",
        "평가는 데이터셋 화자로 기기 등록 조건(표현당 최대 5개 등록)을 흉내 낸 것. 실제 사용자·기기 마이크(INMP441) 검증은 별도로 필요.",
    ]
    return "\n".join(lines) + "\n"
