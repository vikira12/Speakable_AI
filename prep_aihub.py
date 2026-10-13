"""AI Hub 013 원천 zip(로컬) → 학습 데이터 디렉터리 (manifest.jsonl + 16kHz 문장 구간 wav)

사용: python prep_aihub.py <출력 디렉터리> <원천 zip> <라벨 zip> [<라벨 zip> ...] [--workers N] [--tar]
  예) python prep_aihub.py D:/Speakable_AI/data/train_data_v01 "D:/013.구음장애 음성인식 데이터/01.데이터/1.Training/원천데이터/TS01_뇌신경장애.zip"
        "D:/013.구음장애 음성인식 데이터/01.데이터/1.Training/라벨링데이터/TL01_뇌신경장애.zip"
  출력: manifest.jsonl (정렬된 전체), manifest_verified.jsonl (교차 검증 통과분), prep_report.json,
        review.html (무작위 구간 청취 확인용), --tar면 data.tar (검증 통과분, 그 안의 목록 이름은 manifest.jsonl)
이미 처리한 녹음은 건너뛴다 (prep_report.json 기준). 여러 원천 zip은 같은 출력 디렉터리에 차례로 실행하면 합쳐진다.
"""
import json
import os
import random
import sys
import tarfile
import zipfile
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spk_train import prep  # noqa: E402

_Z = {}


def _work(args):
    src_zip, member, rec, out = args
    if src_zip not in _Z:
        _Z[src_zip] = zipfile.ZipFile(src_zip)
    try:
        with _Z[src_zip].open(member) as f:
            return prep.process_record(rec, f, out)
    except Exception as e:                                    # 깨진 파일은 기록만
        return [], {"wav": rec["wav"], "speaker": rec["speaker"], "status": f"error: {e}"}


def review_html(out, lines, n=60):
    pick = random.Random(0).sample(lines, min(n, len(lines)))
    rows = "\n".join(f'<tr><td>{l["speaker"]}</td><td>{l["text"]}</td><td><audio controls preload="none" src="{l["audio"]}"></audio></td>'
                     f'<td>{l["source"]} {l["start"]}s</td></tr>' for l in pick)
    html = f"""<!doctype html><meta charset="utf-8"><title>구간 확인</title>
<style>body{{font-family:sans-serif;margin:16px}}td{{padding:4px 8px;border-bottom:1px solid #ddd}}</style>
<h3>무작위 {len(pick)}개 구간: 소리와 문장이 맞는지 확인</h3><table>{rows}</table>"""
    with open(os.path.join(out, "review.html"), "w", encoding="utf8") as f:
        f.write(html)


def main(argv):
    workers = 4
    make_tar = "--tar" in argv
    argv = [a for a in argv if a != "--tar"]
    if "--workers" in argv:
        i = argv.index("--workers")
        workers = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if len(argv) < 3:
        sys.exit(__doc__)
    out, src_zip, label_zips = argv[0], argv[1], argv[2:]
    os.makedirs(out, exist_ok=True)

    recs = [r for r in prep.load_labels(label_zips) if prep.is_phrase_script(r) and r["speaker"]]
    members = {os.path.basename(prep.fix_name(i.filename)): i.filename
               for i in zipfile.ZipFile(src_zip).infolist() if i.filename.lower().endswith(".wav")}
    rep_path = os.path.join(out, "prep_report.json")
    report = json.load(open(rep_path, encoding="utf8")) if os.path.exists(rep_path) else {"records": []}
    done = {r["wav"] for r in report["records"]}
    todo = [r for r in recs if r["wav"] in members and r["wav"] not in done]
    print(f"짧은 문장 스크립트 녹음 {len(recs)}개 중 이 원천 zip에 있는 것 {sum(r['wav'] in members for r in recs)}개, 처리할 것 {len(todo)}개")

    man_path = os.path.join(out, "manifest.jsonl")
    with Pool(workers) as pool, open(man_path, "a", encoding="utf8") as man:
        for k, (lines, rep) in enumerate(pool.imap_unordered(_work, [(src_zip, members[r["wav"]], r, out) for r in todo])):
            rep["source_zip"] = os.path.basename(src_zip)
            for l in lines:
                man.write(json.dumps(l, ensure_ascii=False) + "\n")
            man.flush()
            report["records"].append(rep)
            json.dump(report, open(rep_path, "w", encoding="utf8"), ensure_ascii=False, indent=1)
            print(f"[{k + 1}/{len(todo)}] {rep['wav']}: {rep['status']} {rep.get('kept', '')}", flush=True)

    all_lines = [json.loads(l) for l in open(man_path, encoding="utf8") if l.strip()]
    ver = prep.verify_alignment(all_lines)
    for r in report["records"]:
        if r["wav"] in ver:
            r["verify"] = ver[r["wav"]]
    lines = [l for l in all_lines if ver[l["source"]]["ok"]]
    with open(os.path.join(out, "manifest_verified.jsonl"), "w", encoding="utf8") as f:
        f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in lines)
    recs_rep = report["records"]
    al = [r for r in recs_rep if r["status"] == "aligned"]
    summ = {"records": len(recs_rep), "aligned": len(al),
            "count_mismatch": sum(r["status"] == "count_mismatch" for r in recs_rep),
            "errors": sum(r["status"].startswith("error") for r in recs_rep),
            "verify_failed": sorted(k for k, v in ver.items() if not v["ok"]),
            "segments_aligned": len(all_lines), "segments": len(lines), "speakers": len({l["speaker"] for l in lines}),
            "classes_text_norm": len({l["text_norm"] for l in lines}),
            "too_long_over_3s": sum(r.get("too_long", 0) for r in al), "annotated_skipped": sum(r.get("annotated", 0) for r in al),
            "hours": round(sum(os.path.getsize(os.path.join(out, l["audio"])) for l in lines) / 32000 / 3600, 2)}
    report["summary"] = summ
    json.dump(report, open(rep_path, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    review_html(out, lines)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    if make_tar:
        with tarfile.open(os.path.join(out, "data.tar"), "w") as t:
            t.add(os.path.join(out, "manifest_verified.jsonl"), "manifest.jsonl")
            for l in lines:
                t.add(os.path.join(out, l["audio"]), l["audio"])
        print("data.tar 생성:", os.path.join(out, "data.tar"))


if __name__ == "__main__":
    main(sys.argv[1:])
