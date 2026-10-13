"""노트북과 PC 점검 스크립트가 같이 쓰는 단계별 실행 함수

  run = Run(cfg)
  run.prepare()      목록 읽기 · 화자 분할 · 특징 캐시 (이어 만들기 가능)
  run.train()        float 학습 (이어 학습 가능)
  run.quantize()     BN 접기 · 활성화 보정 · INT8 평가 · (선택) QAT
  run.calibrate()    검증셋으로 매칭 파라미터 보정, 시험셋 평가 (INT8 경로)
  run.export()       산출물 디렉터리 + 기준 구현 비트 일치 확인
"""
import copy
import os
import numpy as np
import torch

from . import data
from .evaluate import evaluate, format_report
from .model import FoldedEncoder
from .paths import fpga_tools
from .quant import calibrate, int_embed, quantize_input, to_integer
from .train import Trainer, norm_stats

DEFAULT_CONFIG = {
    "model_version": "speakable_v02",
    "data_root": None,               # manifest 파일이 있는 디렉터리
    "manifest": "manifest.jsonl",
    "work_dir": None,                # 캐시·체크포인트·산출물 (Drive 권장)
    "tools_dir": None,               # FPGA 저장소 speakable_arm_ref/tools (None이면 paths.fpga_tools())
    "val_frac": 0.1, "test_frac": 0.1,
    "n_variants": 4,                 # 학습 발화당 원본 1 + 증강 3
    "augment": {"speed": [0.9, 1.1], "p_noise": 0.8, "snr_db": [10, 35], "p_reverb": 0.3, "rt60": [0.2, 0.6],
                "pad_s": [0.0, 0.2], "trim_s": [0.0, 0.05], "noise_files": []},
    "cache_chunk": 1000, "workers": None,
    "label_field": "text_norm",      # 표현 구분: text_norm (평서·의문 묶음) 또는 text
    "class_key": "speaker_text",     # 또는 "text"
    "epochs": 60, "batch_size": 256, "lr": 3e-3, "weight_decay": 1e-4,
    "aam_scale": 30.0, "aam_margin": 0.2,
    "p_clean": 0.3, "f_mask": 6, "t_mask": 8, "n_masks": 2,
    "init_checkpoint": None, "seed": 0,
    "calib_samples": 3000, "calib_percentile": 99.99,
    "qat_epochs": 0, "qat_lr": 1e-4,
    "max_wrong": 0.02, "max_unreg": 0.05,
    "n_test_vectors": 20,
}


class Run:
    def __init__(self, cfg, log=print):
        self.c = copy.deepcopy(DEFAULT_CONFIG)
        self.c.update(cfg)
        if not self.c["tools_dir"]:
            self.c["tools_dir"] = fpga_tools()
        self.log = log
        w = self.c["work_dir"]
        self.cache_dir = os.path.join(w, "cache")
        self.ckpt_dir = os.path.join(w, "checkpoints")
        self.out_dir = os.path.join(w, "artifacts", self.c["model_version"])
        self.report = {}

    # ------------------------------------------------------------ 데이터
    def prepare(self):
        c = self.c
        items = data.load_manifest(c["data_root"], c["manifest"], label_field=c["label_field"])
        self.splits = data.assign_splits(items, c["val_frac"], c["test_frac"])
        summary = data.summarize(self.splits)
        self.log(summary)
        for s in ("train", "val", "test"):
            data.build_cache(self.splits[s], self.cache_dir, s, c["n_variants"] if s == "train" else 1,
                             c["augment"], c["cache_chunk"], c["workers"], self.log)
        self.feats, self.items, skipped = {}, {}, {}
        for s in ("train", "val", "test"):
            self.feats[s], self.items[s], sk = data.load_cache(self.splits[s], self.cache_dir, s)
            skipped[s] = len(sk)
        self.mean, self.std = norm_stats(self.feats["train"])
        self.report["data"] = {"summary": summary.replace("\n", " / "), "skipped": skipped,
                               "cached": {s: len(v) for s, v in self.items.items()}}
        self.log(f"캐시: " + ", ".join(f"{s} {len(v)}개 (제외 {skipped[s]})" for s, v in self.items.items()))

    # ------------------------------------------------------------ float 학습
    def train(self):
        self.trainer = Trainer(self.c, self.feats["train"], self.items["train"], self.feats["val"], self.items["val"],
                               self.mean, self.std, self.ckpt_dir, log=self.log)
        self.trainer.train()
        self.trainer.load_best()
        self.report["history"] = self.trainer.history

    def _x(self, split):
        return self.trainer.normalize(torch.from_numpy(self.feats[split][:, 0]).to(self.trainer.dev))

    def _spk_txt(self, split):
        its = self.items[split]
        return [it["speaker"] for it in its], [it["label"] for it in its]

    def eval_float(self, split="val"):
        emb = self.trainer.embed(self._x(split))
        rep = evaluate(emb, *self._spk_txt(split), max_wrong=self.c["max_wrong"], max_unreg=self.c["max_unreg"])
        self.log(f"[float {split}]\n" + format_report(rep))
        return rep

    # ------------------------------------------------------------ INT8
    def int8_embed(self, split):
        x = self._x(split).cpu().numpy()
        x_q = quantize_input(x, self.input_scale)
        return int_embed(self.qlayers, torch.from_numpy(x_q.astype(np.int64))).numpy()

    def quantize(self):
        c = self.c
        self.folded = FoldedEncoder(self.trainer.enc.cpu()).to(self.trainer.dev)
        self.trainer.enc.to(self.trainer.dev)
        x_tr = self._x("train")
        n = min(c["calib_samples"], len(x_tr))
        sel = torch.from_numpy(np.random.default_rng(0).choice(len(x_tr), n, replace=False))
        calibrate(self.folded, x_tr[sel.to(x_tr.device)], c["calib_percentile"])
        self.qlayers, self.input_scale = to_integer(self.folded)
        v_fp = self.trainer.validate(self.folded.eval())[0]
        v_q = self._int8_top1("val")
        self.log(f"BN 접기 후 검증 1위 정답 {v_fp:.3f} · INT8 정수 추론 {v_q:.3f}")
        if c["qat_epochs"] > 0:
            self.trainer.qat(self.folded, c["qat_epochs"], c["qat_lr"])
            self.folded.quant = False
            self.qlayers, self.input_scale = to_integer(self.folded)
            self.log(f"QAT 후 INT8 정수 추론 검증 1위 정답 {self._int8_top1('val'):.3f}")
        self.folded.quant = False

    def _int8_top1(self, split):
        rep = evaluate(self.int8_embed(split), *self._spk_txt(split), n_trials=1, search=False)
        return rep.get("top1", 0.0)

    # ------------------------------------------------------------ 매칭 보정 · 평가
    def calibrate(self):
        c = self.c
        val = evaluate(self.int8_embed("val"), *self._spk_txt("val"), max_wrong=c["max_wrong"], max_unreg=c["max_unreg"])
        self.log("[INT8 검증셋 · 파라미터 보정]\n" + format_report(val))
        self.report["val_int8"] = val
        if "params" not in val:
            raise RuntimeError(val["error"])
        self.match_params = val["params"]
        test = evaluate(self.int8_embed("test"), *self._spk_txt("test"), fixed=self.match_params)
        self.log("[INT8 시험셋 · 보정값 고정]\n" + format_report(test))
        self.report["test_int8"] = test

    # ------------------------------------------------------------ 산출물
    def export(self, status="trained_not_field_validated"):
        from .export import export_artifacts
        its = self.items["test"] or self.items["val"]
        rng = np.random.default_rng(0)
        pick = rng.choice(len(its), min(self.c["n_test_vectors"], len(its)), replace=False)
        samples = [(f"tv_{k:02d}_{its[i]['id']}".replace("/", "_"), data.read_audio(its[i]["path"])) for k, i in enumerate(sorted(pick))]
        self.report["config"] = {k: v for k, v in self.c.items() if k != "augment"}
        return export_artifacts(self.out_dir, folded=self.folded, head=self.trainer.head, classes=self.trainer.classes,
                                qlayers=self.qlayers, input_scale=self.input_scale, mean=self.mean, std=self.std,
                                match_params=self.match_params, test_samples=samples, tools_dir=self.c["tools_dir"],
                                status=status, report=self.report, config=self.c, log=self.log)
