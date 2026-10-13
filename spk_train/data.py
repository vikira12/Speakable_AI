"""학습 데이터 목록·화자 분할·특징 캐시

manifest.jsonl (데이터 준비 노트북이 만든다), 한 줄에 발화 하나:
  {"id": "...", "audio": "<data_root 기준 상대 경로>", "speaker": "...", "text": "...",
   "text_norm": "..." (선택, 문장부호를 뺀 클래스 키), "split": "train|val|test" (선택)}
  학습·평가의 표현 구분은 설정 label_field (기본 text_norm, 없으면 text)를 it["label"]로 쓴다.
  audio는 16kHz mono PCM16 wav 또는 flac. 다른 샘플레이트·스테레오는 읽을 때 변환한다.

특징 캐시: log Mel (N, A, 40, 64) float16. A = 원본 1 + 증강본. 묶음(chunk) 단위로 저장해서
Colab 세션이 끊겨도 이어서 만들 수 있다.
"""
import glob
import hashlib
import json
import math
import os
import wave
from multiprocessing import Pool
import numpy as np
from scipy.signal import resample_poly

from .frontend import Augmenter, FastFrontend

SR = 16000


def read_audio(path):
    """→ int16 mono 16kHz"""
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="int16", always_2d=True)
    except ImportError:
        with wave.open(path) as w:
            if w.getsampwidth() != 2:
                raise ValueError(f"{path}: 16비트 wav가 아님 (soundfile 설치 필요)")
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    if x.shape[1] > 1:
        x = np.round(x.astype(np.float64).mean(axis=1))
    else:
        x = x[:, 0].astype(np.float64)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = np.round(resample_poly(x, SR // g, sr // g))
    return np.clip(x, -32768, 32767).astype(np.int16)


def write_wav(path, pcm16):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.asarray(pcm16, dtype="<i2").tobytes())


def load_manifest(data_root, name="manifest.jsonl", label_field="text_norm"):
    items = []
    with open(os.path.join(data_root, name), encoding="utf8") as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                it["path"] = os.path.join(data_root, it["audio"])
                it["label"] = it.get(label_field) or it["text"]
                items.append(it)
    return items


def speaker_split(speaker, val_frac, test_frac):
    h = int(hashlib.sha1(speaker.encode("utf8")).hexdigest()[:8], 16) / 2 ** 32
    return "val" if h < val_frac else "test" if h < val_frac + test_frac else "train"


def assign_splits(items, val_frac=0.1, test_frac=0.1):
    """화자 단위 분할 (한 화자의 발화는 한 쪽에만). manifest에 split이 있으면 그대로 쓴다"""
    for it in items:
        it.setdefault("split", speaker_split(it["speaker"], val_frac, test_frac))
    return {s: [it for it in items if it["split"] == s] for s in ("train", "val", "test")}


def summarize(split_items):
    rows = []
    for s, its in split_items.items():
        spk = {it["speaker"] for it in its}
        txt = {it["label"] for it in its}
        pairs = {}
        for it in its:
            pairs[(it["speaker"], it["label"])] = pairs.get((it["speaker"], it["label"]), 0) + 1
        rep = sum(1 for v in pairs.values() if v >= 2)
        rows.append(f"{s:5s}: 발화 {len(its):6d}  화자 {len(spk):4d}  문장 {len(txt):5d}  "
                    f"(화자,문장) 쌍 {len(pairs):6d} 중 2회 이상 {rep:6d}")
    return "\n".join(rows)


# ---------------------------------------------------------------- 특징 캐시

_W = {}


def _init_worker(n_variants, aug_cfg, fe_p):
    _W["fe"] = FastFrontend()
    _W["aug"] = Augmenter(aug_cfg, min_samples=fe_p["min_samples"], max_samples=fe_p["max_samples"],
                          read_audio=read_audio) if n_variants > 1 else None
    _W["n"] = n_variants
    _W["p"] = fe_p


def _work(args):
    path, seed = args
    fe, aug, n, p = _W["fe"], _W["aug"], _W["n"], _W["p"]
    try:
        pcm = read_audio(path)
    except Exception as e:                                    # 깨진 파일은 건너뛰고 기록
        return None, f"read:{e}"
    if not p["min_samples"] <= len(pcm) <= p["max_samples"]:
        return None, f"length:{len(pcm)}"
    out = np.empty((n, 40, 64), dtype=np.float16)
    out[0] = fe.log_mel(pcm)
    rng = np.random.default_rng(seed)
    for a in range(1, n):
        x = aug(pcm, rng)
        out[a] = fe.log_mel(x if x is not None else pcm)
    return out, None


def _seed(item_id):
    return int(hashlib.sha1(item_id.encode("utf8")).hexdigest()[:8], 16)


def build_cache(items, cache_dir, split, n_variants=1, aug_cfg=None, chunk=1000, workers=None, log=print):
    """split 발화들의 log Mel 캐시를 만든다 (이미 있는 묶음은 건너뜀)"""
    os.makedirs(cache_dir, exist_ok=True)
    fe_p = FastFrontend().p
    n_chunks = (len(items) + chunk - 1) // chunk
    workers = workers or os.cpu_count()
    with Pool(workers, initializer=_init_worker, initargs=(n_variants, aug_cfg or {}, fe_p)) as pool:
        for c in range(n_chunks):
            fn = os.path.join(cache_dir, f"{split}_{c:04d}.npz")
            if os.path.exists(fn):
                continue
            part = items[c * chunk:(c + 1) * chunk]
            res = pool.map(_work, [(it["path"], _seed(it["id"])) for it in part], chunksize=8)
            keep = [i for i, (f, _) in enumerate(res) if f is not None]
            skipped = {part[i]["id"]: r for i, (f, r) in enumerate(res) if f is None}
            feats = np.stack([res[i][0] for i in keep]) if keep else np.zeros((0, n_variants, 40, 64), np.float16)
            tmp = fn + ".tmp.npz"
            np.savez(tmp, feats=feats, ids=np.array([part[i]["id"] for i in keep]),
                     skipped=np.array(json.dumps(skipped, ensure_ascii=False)))
            os.replace(tmp, fn)
            log(f"[{split}] 묶음 {c + 1}/{n_chunks}: {len(keep)}개 저장, {len(skipped)}개 제외")


def load_cache(items, cache_dir, split):
    """→ feats (N, A, 40, 64) float16, 캐시에 들어간 발화 목록, 제외 사유 dict"""
    by_id = {it["id"]: it for it in items}
    feats, kept, skipped = [], [], {}
    for fn in sorted(glob.glob(os.path.join(cache_dir, f"{split}_*.npz"))):
        if fn.endswith(".tmp.npz"):
            continue
        z = np.load(fn)
        feats.append(z["feats"])
        kept += [by_id[i] for i in z["ids"]]
        skipped.update(json.loads(str(z["skipped"])))
    return np.concatenate(feats), kept, skipped
