"""학습 루프 (float 학습 → BN 접기 → 보정 → 선택적 QAT)

특징 캐시 전체를 GPU 메모리에 올려 두고 배치마다 증강본 하나를 고른 뒤 SpecAugment를 적용한다.
체크포인트는 epoch마다 저장하므로 Colab 세션이 끊겨도 이어서 학습할 수 있다.
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import evaluate
from .model import AAMSoftmax, Encoder, FoldedEncoder


def norm_stats(train_feats):
    """원본(증강 0번) log Mel의 Mel 대역별 평균·표준편차"""
    x = train_feats[:, 0].astype(np.float64)                  # (N, 40, 64)
    return x.mean(axis=(0, 2)), x.std(axis=(0, 2))


def make_labels(items, class_key):
    """class_key: "text" (표현) 또는 "speaker_text" (화자별 표현). 발화 2개 미만 클래스는 제외"""
    key = (lambda it: it["label"]) if class_key == "text" else (lambda it: f"{it['speaker']}\t{it['label']}")
    keys = [key(it) for it in items]
    cnt = {}
    for k in keys:
        cnt[k] = cnt.get(k, 0) + 1
    classes = sorted(k for k, v in cnt.items() if v >= 2)
    cid = {k: i for i, k in enumerate(classes)}
    labels = np.array([cid.get(k, -1) for k in keys])
    return labels, classes


def spec_augment(x, gen, f_mask=6, t_mask=8, n_masks=2):
    """x: (B,1,40,64) 정규화 입력. 마스크 구간은 0 (= 평균값)"""
    B = x.shape[0]
    dev = x.device
    for _ in range(n_masks):
        for dim, size, width in ((2, 40, f_mask), (3, 64, t_mask)):
            w = torch.randint(0, width + 1, (B,), generator=gen, device="cpu").to(dev)
            s = (torch.rand(B, generator=gen).to(dev) * (size - w)).long()
            ar = torch.arange(size, device=dev)
            m = (ar[None] >= s[:, None]) & (ar[None] < (s + w)[:, None])
            x = x.masked_fill(m[:, None, :, None] if dim == 2 else m[:, None, None, :], 0.0)
    return x


class Trainer:
    def __init__(self, cfg, train_feats, train_items, val_feats, val_items, mean, std, run_dir, device=None, log=print):
        self.c = cfg
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log = log
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        labels, self.classes = make_labels(train_items, cfg["class_key"])
        keep = labels >= 0
        self.labels = torch.from_numpy(labels[keep]).to(self.dev)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.dev).reshape(1, 1, 40, 1)
        self.std = torch.tensor(np.maximum(std, 1e-6), dtype=torch.float32, device=self.dev).reshape(1, 1, 40, 1)
        self.feats = torch.from_numpy(train_feats[keep]).to(self.dev)              # float16 (N, A, 40, 64)
        self.val_x = self.normalize(torch.from_numpy(val_feats[:, 0]).to(self.dev))
        self.val_items = val_items
        log(f"학습 발화 {int(keep.sum())} (클래스 {len(self.classes)}, 발화 1개뿐인 클래스 {int((~keep).sum())}개 제외) · 장치 {self.dev}")

        self.enc = Encoder().to(self.dev)
        if cfg.get("init_checkpoint"):
            ck = torch.load(cfg["init_checkpoint"], map_location="cpu", weights_only=False)
            self.enc.load_state_dict(ck["encoder"])
            log(f"초기 가중치: {cfg['init_checkpoint']}")
        self.head = AAMSoftmax(len(self.classes), cfg["aam_scale"], cfg["aam_margin"]).to(self.dev)
        params = [{"params": [p for n, p in self.enc.named_parameters() if p.ndim > 1], "weight_decay": cfg["weight_decay"]},
                  {"params": [p for n, p in self.enc.named_parameters() if p.ndim <= 1], "weight_decay": 0.0},
                  {"params": self.head.parameters(), "weight_decay": cfg["weight_decay"]}]
        self.opt = torch.optim.AdamW(params, lr=cfg["lr"])
        steps = cfg["epochs"] * max(1, len(self.labels) // cfg["batch_size"])
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, cfg["lr"], total_steps=steps, pct_start=0.05)
        self.epoch, self.best, self.history = 0, -1.0, []
        self.gen = torch.Generator().manual_seed(cfg.get("seed", 0))
        self._resume()

    def normalize(self, logmel):
        """logmel: (N, 40, 64) → (N, 1, 40, 64)"""
        return (logmel.float().unsqueeze(1) - self.mean) / self.std

    def _ck_path(self, name):
        return os.path.join(self.run_dir, name)

    def _resume(self):
        p = self._ck_path("last.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=self.dev, weights_only=False)
            for k in ("epochs", "batch_size", "lr", "class_key"):
                if ck["config"][k] != self.c[k]:
                    raise ValueError(f"이어 학습할 체크포인트와 설정 {k}가 다름 ({ck['config'][k]} → {self.c[k]}). "
                                     f"설정을 되돌리거나 {self.run_dir}를 비우고 새로 학습")
            self.enc.load_state_dict(ck["encoder"])
            self.head.load_state_dict(ck["head"])
            self.opt.load_state_dict(ck["opt"])
            self.sched.load_state_dict(ck["sched"])
            self.epoch, self.best, self.history = ck["epoch"], ck["best"], ck["history"]
            self.gen.set_state(ck["gen"])
            self.log(f"이어서 학습: epoch {self.epoch}부터 (최고 검증 {self.best:.3f})")

    def _save(self, name):
        torch.save({"encoder": self.enc.state_dict(), "head": self.head.state_dict(), "opt": self.opt.state_dict(),
                    "sched": self.sched.state_dict(), "epoch": self.epoch, "best": self.best, "history": self.history,
                    "gen": self.gen.get_state(), "classes": self.classes, "config": self.c,
                    "mean": self.mean.flatten().cpu(), "std": self.std.flatten().cpu()}, self._ck_path(name + ".tmp"))
        os.replace(self._ck_path(name + ".tmp"), self._ck_path(name))

    @torch.no_grad()
    def embed(self, x, model=None, batch=1024):
        model = model or self.enc
        model.eval()
        out = [F.normalize(model(x[i:i + batch]), dim=1) for i in range(0, len(x), batch)]
        return torch.cat(out).cpu().numpy()

    def validate(self, model=None):
        emb = self.embed(self.val_x, model)
        rep = evaluate(emb, [it["speaker"] for it in self.val_items], [it["label"] for it in self.val_items], n_trials=1, search=False)
        return rep.get("top1", 0.0), rep

    def train(self):
        c = self.c
        N, A = self.feats.shape[:2]
        bs = c["batch_size"]
        while self.epoch < c["epochs"]:
            self.enc.train()
            t0 = time.time()
            perm = torch.randperm(N, generator=self.gen).to(self.dev)
            tot_loss = tot_acc = n = 0
            for i in range(0, N - bs + 1, bs):
                idx = perm[i:i + bs]
                # 증강본 선택: p_clean 확률로 원본, 아니면 증강본 중 하나
                var = torch.randint(1, max(A, 2), (bs,), generator=self.gen).to(self.dev) if A > 1 else torch.zeros(bs, dtype=torch.long, device=self.dev)
                if A > 1:
                    var = torch.where(torch.rand(bs, generator=self.gen).to(self.dev) < c["p_clean"], 0, var)
                x = self.normalize(self.feats[idx, var])
                x = spec_augment(x, self.gen, c["f_mask"], c["t_mask"], c["n_masks"])
                loss, acc = self.head(self.enc(x), self.labels[idx])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                self.sched.step()
                tot_loss += loss.item()
                tot_acc += acc.item()
                n += 1
            self.epoch += 1
            v, _ = self.validate()
            self.history.append({"epoch": self.epoch, "loss": tot_loss / max(n, 1), "train_acc": tot_acc / max(n, 1), "val_top1": v})
            if v > self.best:
                self.best = v
                self._save("best.pt")
            self._save("last.pt")
            self.log(f"epoch {self.epoch:3d}  loss {tot_loss / max(n, 1):.3f}  학습 정확도 {tot_acc / max(n, 1):.3f}  "
                     f"검증 1위 정답 {v:.3f}  (최고 {self.best:.3f})  {time.time() - t0:.0f}s")

    def load_best(self):
        ck = torch.load(self._ck_path("best.pt"), map_location=self.dev, weights_only=False)
        self.enc.load_state_dict(ck["encoder"])
        self.head.load_state_dict(ck["head"])

    def qat(self, folded: FoldedEncoder, epochs, lr):
        """가짜 양자화 미세조정. 활성화 scale은 보정값으로 고정, 가중치 scale은 매 단계 다시 계산"""
        folded.quant = True
        opt = torch.optim.AdamW(list(folded.parameters()) + list(self.head.parameters()), lr=lr, weight_decay=0.0)
        N, A = self.feats.shape[:2]
        bs = self.c["batch_size"]
        best_v, best_state = self.validate(folded)[0], {k: v.clone() for k, v in folded.state_dict().items()}
        self.log(f"QAT 시작: 검증 1위 정답 {best_v:.3f}")
        for ep in range(epochs):
            folded.train()
            perm = torch.randperm(N, generator=self.gen).to(self.dev)
            for i in range(0, N - bs + 1, bs):
                idx = perm[i:i + bs]
                var = torch.randint(0, A, (bs,), generator=self.gen).to(self.dev)
                x = spec_augment(self.normalize(self.feats[idx, var]), self.gen, self.c["f_mask"], self.c["t_mask"], self.c["n_masks"])
                loss, _ = self.head(folded(x), self.labels[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            v = self.validate(folded)[0]
            self.log(f"QAT epoch {ep + 1}/{epochs}: 검증 1위 정답 {v:.3f}")
            if v > best_v:
                best_v, best_state = v, {k: t.clone() for k, t in folded.state_dict().items()}
        folded.load_state_dict(best_state)
        return best_v
