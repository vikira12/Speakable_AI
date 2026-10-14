"""AI Hub 013 원천 zip(로컬) → 학습 데이터 디렉터리 (manifest.jsonl + 16kHz 문장 구간 wav)

사용: python prep_aihub.py <출력 디렉터리> <원천 zip> <라벨 zip> [<라벨 zip> ...] [--workers N] [--tar]
  예) python prep_aihub.py D:/project/Speakable_AI/data/train_data_v01 "D:/013.구음장애 음성인식 데이터/01.데이터/1.Training/원천데이터/TS01_뇌신경장애.zip"
        "D:/013.구음장애 음성인식 데이터/01.데이터/1.Training/라벨링데이터/TL01_뇌신경장애.zip"
  출력: manifest.jsonl (정렬된 전체), manifest_verified.jsonl (교차 검증 통과분), prep_report.json,
        review.html (무작위 구간 청취 확인용), --tar면 data.tar (검증 통과분, 그 안의 목록 이름은 manifest.jsonl)
이미 처리한 녹음은 건너뛴다 (prep_report.json 기준). 여러 원천 zip은 같은 출력 디렉터리에 차례로 실행하면 합쳐진다.
원천 zip을 받지 않고 Colab에서 필요한 녹음만 읽는 방법은 speakable_prep.ipynb.
"""
import os
import sys
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
    po = prep.PrepOutput(out)

    recs = [r for r in prep.load_labels(label_zips) if prep.is_phrase_script(r) and r["speaker"]]
    members = {os.path.basename(prep.fix_name(i.filename)): i.filename
               for i in zipfile.ZipFile(src_zip).infolist() if i.filename.lower().endswith(".wav")}
    todo = [r for r in recs if r["wav"] in members and r["wav"] not in po.done]
    print(f"짧은 문장 스크립트 녹음 {len(recs)}개 중 이 원천 zip에 있는 것 {sum(r['wav'] in members for r in recs)}개, 처리할 것 {len(todo)}개")

    with Pool(workers) as pool:
        for k, (lines, rep) in enumerate(pool.imap_unordered(_work, [(src_zip, members[r["wav"]], r, out) for r in todo])):
            po.add(lines, rep, os.path.basename(src_zip))
            print(f"[{k + 1}/{len(todo)}] {rep['wav']}: {rep['status']} {rep.get('kept', '')}", flush=True)

    po.finalize(os.path.join(out, "data.tar") if make_tar else None)


if __name__ == "__main__":
    main(sys.argv[1:])
