"""AI Hub 「구음장애 음성인식 데이터」(013) → 학습용 문장 구간

라벨 JSON은 녹음 파일(수 분~수 시간)마다 전체 발화 텍스트(Transcript) 하나뿐이고 문장별 시간 정보가 없다.
짧은 생활 표현 스크립트 녹음은 문장 사이 무음이 길어서 에너지 VAD 구간 수가 문장 수와 같으면
순서대로 대응시킨다. VAD 설정 몇 가지를 시도해도 수가 맞지 않는 파일은 건너뛰고 기록한다
(이후 ASR 정렬 대상).

라벨 표기: '+'(다시 말함), '/'(간투사 등)가 들어간 문장은 순서 자리는 지키되 학습 데이터에서는 뺀다.
"""
import io
import json
import math
import os
import re
import wave
import zipfile
import numpy as np
from scipy.signal import resample_poly

SR = 16000
HOP = 0.01
# (임계 dB: None이면 자동, 구간 병합 간격 s, 최소 구간 길이 s) — 앞에서부터 시도
VAD_GRID = [(None, 0.35, 0.25), (None, 0.6, 0.25), (None, 1.0, 0.3), (None, 0.35, 0.4), (None, 1.5, 0.3),
            (18, 0.6, 0.3), (24, 0.6, 0.3), (12, 1.0, 0.3), (None, 2.0, 0.3)]


def fix_name(n):
    """zip에 UTF-8 표시 없이 cp949로 저장된 파일 이름 복원"""
    try:
        return n.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return n


def speaker_of(name):
    """ID-01-11-N-AMG-02-02-M-56-KK → AMG-M-56 (이니셜·성별·나이)"""
    m = re.match(r"ID-\d+-\d+-\w-([A-Za-z]+)-[\d-]+-([MF])-(\d+)", os.path.basename(name))
    return f"{m.group(1).upper()}-{m.group(2)}-{m.group(3)}" if m else None


def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s*", (t or "").strip()) if s.strip()]


def norm_text(s):
    """문장부호를 뺀 클래스 키 (평서문·의문문 묶음)"""
    return re.sub(r"\s+", " ", re.sub(r"[.?!,~]", " ", s)).strip()


def hangul_len(s):
    return len(re.sub(r"[^가-힣0-9A-Za-z]", "", s))


def is_annotated(s):
    return any(c in s for c in "+/()[]{}<>*#")


def load_labels(zip_paths):
    """→ [{"zip", "member", "wav", "speaker", "sentences", "play_time"}]"""
    out = []
    for zp in zip_paths:
        z = zipfile.ZipFile(zp)
        for info in z.infolist():
            if not info.filename.lower().endswith(".json"):
                continue
            raw = z.read(info)
            d = None
            for enc in ("utf-8-sig", "cp949"):
                try:
                    d = json.loads(raw.decode(enc))
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
            if d is None:
                continue
            name = fix_name(info.filename)
            out.append({"zip": os.path.basename(zp), "member": name,
                        "wav": os.path.basename(name)[:-5] + ".wav", "speaker": speaker_of(name),
                        "sentences": split_sentences(d.get("Transcript")),
                        "play_time": float(d.get("playTime") or (d.get("Meta_info") or {}).get("PlayTime") or 0)})
    return out


def is_phrase_script(rec, max_chars=12, min_ratio=0.7):
    """짧은 문장 비율이 높은 녹음 (생활 표현·명령어 스크립트)"""
    ss = rec["sentences"]
    return bool(ss) and sum(hangul_len(s) <= max_chars for s in ss) / len(ss) >= min_ratio


# ---------------------------------------------------------------- 음성

def read_wav_16k(f):
    """파일 객체(wav) → float64 mono 16kHz [-1, 1)"""
    with wave.open(f) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2 ** 31
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int8).astype(np.int32) << 16)) / 2 ** 23
    else:
        raise ValueError(f"지원하지 않는 샘플 크기 {sw}")
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x, sr


def frame_db(x):
    h = int(SR * HOP)
    n = len(x) // h
    return 10 * np.log10(np.mean(x[:n * h].reshape(n, h) ** 2, axis=1) + 1e-12)


def vad(e_db, thr_db=None, min_gap=0.35, min_len=0.25):
    """프레임 에너지(dB) → [(시작 s, 끝 s)]. 임계값은 바닥 소음(10백분위) 기준"""
    floor, peak = np.percentile(e_db, 10), np.percentile(e_db, 99)
    thr = floor + (thr_db if thr_db is not None else max(12.0, (peak - floor) * 0.35))
    act = np.concatenate([[False], e_db > thr, [False]])
    d = np.diff(act.astype(np.int8))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    segs = []
    for s, e in zip(starts * HOP, ends * HOP):
        if segs and s - segs[-1][1] < min_gap:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    return [(s, e) for s, e in segs if e - s >= min_len]


def group_blocks(segs):
    """짧은 쉼으로 떨어진 구간들을 한 발화 덩어리로 묶는다 (발화 중간 쉼, 다시 말하기).
    녹음마다 발화 사이 쉼이 거의 일정하므로(편집된 무음), 큰 쉼들의 중앙값(표준 쉼)의 절반보다 짧은 쉼은 묶음"""
    if len(segs) < 3:
        return segs
    gaps = np.array([segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)])
    big = gaps[gaps > 0.6 * np.median(gaps)]
    std_gap = np.median(big) if len(big) else np.median(gaps)
    out = [list(segs[0])]
    for i, g in enumerate(gaps):
        if g < 0.5 * std_gap:
            out[-1][1] = segs[i + 1][1]
        else:
            out.append(list(segs[i + 1]))
    return [tuple(b) for b in out]


def segment_record(x, n_sentences):
    """VAD 설정을 차례로 시도해 구간 수 = 문장 수인 결과를 찾는다 → (구간, 설정) 또는 (None, 시도별 구간 수).
    그대로 안 맞으면 같은 설정들로 덩어리 묶기(group_blocks)를 시도"""
    e = frame_db(x)
    counts, all_segs = [], []
    for params in VAD_GRID:
        segs = vad(e, *params)
        if len(segs) == n_sentences:
            return segs, list(params)
        counts.append(len(segs))
        all_segs.append((params, segs))
    for params, segs in all_segs:
        g = group_blocks(segs)
        if len(g) == n_sentences:
            return g, list(params) + ["grouped"]
    return None, counts


def canonical_script(recs, tag="-02-02-"):
    """생활 표현 스크립트(파일명에 tag)의 표준 문장 순서 = 라벨에 가장 많이 나온 문장 목록"""
    from collections import Counter
    seqs = Counter(tuple(r["sentences"]) for r in recs if tag in r["wav"])
    return list(seqs.most_common(1)[0][0]) if seqs else None


def confirmed_slots(label, canon):
    """표준 스크립트 자리 중 라벨이 같은 문장으로 확인해 주는 자리 (라벨 누락·병합·오타 자리는 제외)"""
    import difflib
    sm = difflib.SequenceMatcher(a=canon, b=label, autojunk=False)
    return {k for tag, a0, a1, _, _ in sm.get_opcodes() if tag == "equal" for k in range(a0, a1)}


def cut(x, segs, k, pad=0.15, max_s=3.0):
    """k번째 구간 + 앞뒤 여유 (이웃 구간과의 중간점을 넘지 않음, 전체 max_s 이하가 되게 여유를 줄임)"""
    s, e = segs[k]
    lo = (segs[k - 1][1] + s) / 2 if k > 0 else 0.0
    hi = (e + segs[k + 1][0]) / 2 if k + 1 < len(segs) else len(x) / SR
    p = min(pad, max(0.0, (max_s - (e - s)) / 2))
    a, b = max(lo, s - p), min(hi, e + p)
    return x[int(a * SR):int(b * SR)]


def to_pcm16(x):
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def write_wav(path, pcm16):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.astype("<i2").tobytes())


def verify_alignment(lines, min_rho=0.45, min_ref=10):
    """정렬 교차 검증: 녹음마다 구간 길이(녹음 안 중앙값으로 나눔)를, 같은 문장을 읽은 다른 녹음들의 평균과 비교.
    순서가 맞으면 순위 상관(rho)이 높고, 앞뒤로 한 칸 밀린 경우보다 높아야 한다.
    → {녹음: {"rho", "rho_shift", "n_ref", "ok"}}"""
    from scipy.stats import spearmanr
    by = {}
    for l in lines:
        by.setdefault(l["source"], []).append(l)
    rel = {}
    for src, ls in by.items():
        d = np.array([l["end"] - l["start"] for l in ls])
        rel[src] = d / np.median(d)
    tot, cnt = {}, {}
    for src, ls in by.items():
        for l, r in zip(ls, rel[src]):
            tot[l["text_norm"]] = tot.get(l["text_norm"], 0.0) + r
            cnt[l["text_norm"]] = cnt.get(l["text_norm"], 0) + 1
    out = {}
    for src, ls in by.items():
        ref = np.array([(tot[l["text_norm"]] - r) / (cnt[l["text_norm"]] - 1) if cnt[l["text_norm"]] > 1 else np.nan
                        for l, r in zip(ls, rel[src])])                 # 자기 자신을 뺀 평균
        ok = ~np.isnan(ref)
        n_ref = int(ok.sum())
        if n_ref < min_ref:
            out[src] = {"rho": None, "rho_shift": None, "n_ref": n_ref, "ok": False}
            continue
        r = rel[src]
        rho = float(spearmanr(r[ok], ref[ok])[0])
        sh = ok[:-1] & ok[1:]
        if sh.sum() > 5:                                         # 앞뒤 어느 쪽으로 한 칸 밀린 경우든 그보다 높아야 함
            rho_s = max(float(spearmanr(r[1:][sh], ref[:-1][sh])[0]), float(spearmanr(r[:-1][sh], ref[1:][sh])[0]))
        else:
            rho_s = -1.0
        out[src] = {"rho": rho, "rho_shift": rho_s, "n_ref": n_ref, "ok": rho >= min_rho and rho > rho_s}
    return out


def process_record(rec, wav_file, out_root, min_s=0.5, max_s=3.0):
    """녹음 하나 → 구간 wav 저장. 반환: (manifest 줄 목록, 파일 보고)"""
    x, sr_orig = read_wav_16k(wav_file)
    ss = rec["sentences"]
    segs, info = segment_record(x, len(ss))
    rep = {"wav": rec["wav"], "speaker": rec["speaker"], "sentences": len(ss), "orig_sr": sr_orig,
           "seconds": round(len(x) / SR, 1)}
    items = list(enumerate(ss))
    status = "aligned"
    canon = rec.get("canonical")
    if segs is None and canon:
        # 라벨 오류(문장 병합·누락)로 수가 안 맞는 경우: 녹음은 표준 스크립트 순서로 읽었다고 보고 표준 문장 수로 맞춤
        segs2, info2 = segment_record(x, len(canon))
        if segs2 is not None:
            ok = confirmed_slots(ss, canon)
            segs, info, status = segs2, info2, "aligned_canonical"
            items = [(k, canon[k]) for k in range(len(canon)) if k in ok]
            rep["unconfirmed"] = len(canon) - len(items)
    if segs is None:
        rep.update(status="count_mismatch", vad_counts=info)
        return [], rep
    rep.update(status=status, vad_params=list(info))
    lines, too_long, too_short, annotated = [], 0, 0, 0
    sub = f"audio/{rec['speaker']}"                         # 목록에는 OS와 무관하게 '/' 경로
    os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    stem = rec["wav"][:-4]
    for k, s in items:
        dur = segs[k][1] - segs[k][0]
        if is_annotated(s):
            annotated += 1
            continue
        if dur > max_s:
            too_long += 1
            continue
        seg = cut(x, segs, k, max_s=max_s)
        if len(seg) < min_s * SR:
            too_short += 1
            continue
        rel = f"{sub}/{stem}_{k:03d}.wav"
        write_wav(os.path.join(out_root, rel), to_pcm16(seg))
        lines.append({"id": f"{stem}_{k:03d}", "audio": rel, "speaker": rec["speaker"], "text": s,
                      "text_norm": norm_text(s), "source": rec["wav"], "start": round(segs[k][0], 2),
                      "end": round(segs[k][1], 2)})
    rep.update(kept=len(lines), too_long=too_long, too_short=too_short, annotated=annotated)
    return lines, rep


# ---------------------------------------------------------------- 출력 디렉터리 (PC·Colab 공용)

class PrepOutput:
    """출력 디렉터리의 manifest.jsonl(정렬된 전체)과 prep_report.json(녹음별 결과)을 이어서 쓴다"""

    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.rep_path = os.path.join(out, "prep_report.json")
        self.man_path = os.path.join(out, "manifest.jsonl")
        self.report = json.load(open(self.rep_path, encoding="utf8")) if os.path.exists(self.rep_path) else {"records": []}
        self.done = {r["wav"] for r in self.report["records"]}

    def add(self, lines, rep, source_zip):
        rep["source_zip"] = source_zip
        with open(self.man_path, "a", encoding="utf8", newline="\n") as f:
            f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in lines)
        self.report["records"] = [r for r in self.report["records"] if r["wav"] != rep["wav"]]   # 다시 처리하면 대체
        self.report["records"].append(rep)
        if not rep["status"].startswith("error"):                # 오류(네트워크 등)는 다음 실행 때 다시 시도
            self.done.add(rep["wav"])
        self._save()

    def _save(self):
        tmp = self.rep_path + ".tmp"
        json.dump(self.report, open(tmp, "w", encoding="utf8"), ensure_ascii=False, indent=1)
        os.replace(tmp, self.rep_path)

    def finalize(self, tar_path=None, log=print):
        """정렬 교차 검증 → manifest_verified.jsonl, review.html, 요약 (→ tar_path에 data.tar)"""
        all_lines = [json.loads(l) for l in open(self.man_path, encoding="utf8") if l.strip()] if os.path.exists(self.man_path) else []
        ver = verify_alignment(all_lines)
        for r in self.report["records"]:
            if r["wav"] in ver:
                r["verify"] = ver[r["wav"]]
        lines = [l for l in all_lines if ver[l["source"]]["ok"]]
        with open(os.path.join(self.out, "manifest_verified.jsonl"), "w", encoding="utf8", newline="\n") as f:
            f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in lines)
        recs = self.report["records"]
        al = [r for r in recs if r["status"].startswith("aligned")]
        by_zip = {}
        for l in lines:
            z = next((r["source_zip"] for r in recs if r["wav"] == l["source"]), "?")
            by_zip[z] = by_zip.get(z, 0) + 1
        summ = {"records": len(recs), "aligned": len(al),
                "aligned_canonical": sum(r["status"] == "aligned_canonical" for r in recs),
                "count_mismatch": sum(r["status"] == "count_mismatch" for r in recs),
                "errors": sum(r["status"].startswith("error") for r in recs),
                "verify_failed": sorted(k for k, v in ver.items() if not v["ok"]),
                "segments_aligned": len(all_lines), "segments": len(lines), "segments_by_zip": by_zip,
                "speakers": len({l["speaker"] for l in lines}), "classes_text_norm": len({l["text_norm"] for l in lines}),
                "too_long_over_3s": sum(r.get("too_long", 0) for r in al), "annotated_skipped": sum(r.get("annotated", 0) for r in al),
                "hours": round(sum(os.path.getsize(os.path.join(self.out, l["audio"])) for l in lines) / 32000 / 3600, 2)}
        self.report["summary"] = summ
        self._save()
        review_html(self.out, lines)
        log(json.dumps(summ, ensure_ascii=False, indent=1))
        if tar_path:
            import tarfile
            tmp = tar_path + ".tmp"
            with tarfile.open(tmp, "w") as t:
                t.add(os.path.join(self.out, "manifest_verified.jsonl"), "manifest.jsonl")
                for l in lines:
                    t.add(os.path.join(self.out, l["audio"]), l["audio"])
            os.replace(tmp, tar_path)
            log(f"data.tar 생성: {tar_path}")
        return summ


def review_html(out, lines, n=60):
    import random
    pick = random.Random(0).sample(lines, min(n, len(lines)))
    rows = "\n".join(f'<tr><td>{l["speaker"]}</td><td>{l["text"]}</td><td><audio controls preload="none" src="{l["audio"]}"></audio></td>'
                     f'<td>{l["source"]} {l["start"]}s</td></tr>' for l in pick)
    html = f"""<!doctype html><meta charset="utf-8"><title>구간 확인</title>
<style>body{{font-family:sans-serif;margin:16px}}td{{padding:4px 8px;border-bottom:1px solid #ddd}}</style>
<h3>무작위 {len(pick)}개 구간: 소리와 문장이 맞는지 확인</h3><table>{rows}</table>"""
    with open(os.path.join(out, "review.html"), "w", encoding="utf8") as f:
        f.write(html)



def _process_bytes(rec, data, out):
    try:
        return process_record(rec, io.BytesIO(data), out)
    except Exception as e:                                    # 깨진 음성은 기록만
        return [], {"wav": rec["wav"], "speaker": rec["speaker"], "status": f"error: {e}"}


def process_entries(f, recs, po, source_zip, workers=4, log=print, stop_on_truncated=False):
    """zip을 앞에서부터 읽으며(aihub.iter_zip_entries) 필요한 녹음만 처리 → 이 zip에서 본 wav 이름 집합.
    읽기는 이 스레드, 처리는 작업 스레드 (동시에 들고 있는 녹음은 workers × 2개까지).
    stop_on_truncated: 데이터 끝에서 잘린 항목을 만나면 오류 대신 멈춤 (끊긴 받기의 조각 파일용)"""
    from concurrent.futures import ThreadPoolExecutor
    from . import aihub
    want = {r["wav"]: r for r in recs if r["wav"] not in po.done}
    seen, pending, n = set(), [], [0]

    def drain(limit):
        while len(pending) > limit:
            lines, rep = pending.pop(0).result()
            po.add(lines, rep, source_zip)
            n[0] += 1
            log(f"[{n[0]}] {rep['wav']}: {rep['status']} {rep.get('kept', '')}")

    with ThreadPoolExecutor(workers) as ex:
        try:
            for name, read, skip in aihub.iter_zip_entries(f):
                b = os.path.basename(name)
                seen.add(b)
                rec = want.get(b)
                if rec is None:
                    skip()
                    continue
                try:
                    data = read()
                except aihub.CorruptEntry as e:               # 이 녹음만 깨짐. 연결 오류 등은 그대로 위로 (다시 받기)
                    po.add([], {"wav": rec["wav"], "speaker": rec["speaker"], "status": f"error: {e}"}, source_zip)
                    continue
                pending.append(ex.submit(_process_bytes, rec, data, po.out))
                drain(workers * 2)
        except aihub.Truncated:
            if not stop_on_truncated:
                raise
            log(f"{source_zip}: 데이터 끝 (잘린 항목에서 멈춤)")
        finally:
            drain(0)
    return seen
