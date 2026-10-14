"""AI Hub 013 원천 → 학습 데이터 디렉터리 (manifest.jsonl + 16kHz 문장 구간 wav). 국내 PC 전용

AI Hub 데이터는 국외 다운로드·반출 금지: 받기·자르기·학습 모두 이 PC에서 하고, 결과(data/, runs/)를
Google Drive·Colab·GitHub 등에 올리지 않는다.

사용
  python prep_aihub.py <출력 디렉터리> --labels <라벨 zip...> --zip <원천 zip>         PC에 받아 둔 원천 zip
  python prep_aihub.py <출력 디렉터리> --labels <라벨 zip...> --remote <filekey...>    AI Hub에서 필요한 부분만
      API 키는 환경 변수 AIHUB_APIKEY (명령줄에 쓰지 않음)
      filekey: TS01 32705, VS01 65521, TS02 55286, TS03 55287
  공통 옵션: --workers N (기본 4, 원격은 6), --tar (검증 통과분 data.tar)
출력: manifest.jsonl (정렬된 전체), manifest_verified.jsonl (교차 검증 통과분), prep_report.json, review.html
이미 처리한 녹음은 건너뛴다 (prep_report.json 기준). 원천을 바꿔 같은 출력 디렉터리에 실행하면 합쳐진다.
"""
import argparse
import os
import sys
import zipfile
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spk_train import aihub, prep  # noqa: E402

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


def run_local(src_zip, recs, po, workers):
    members = {os.path.basename(prep.fix_name(i.filename)): i.filename
               for i in zipfile.ZipFile(src_zip).infolist() if i.filename.lower().endswith(".wav")}
    todo = [r for r in recs if r["wav"] in members and r["wav"] not in po.done]
    print(f"{os.path.basename(src_zip)}: 짧은 문장 녹음 {sum(r['wav'] in members for r in recs)}개, 처리할 것 {len(todo)}개")
    with Pool(workers) as pool:
        for k, (lines, rep) in enumerate(pool.imap_unordered(_work, [(src_zip, members[r["wav"]], r, po.out) for r in todo])):
            po.add(lines, rep, os.path.basename(src_zip))
            print(f"[{k + 1}/{len(todo)}] {rep['wav']}: {rep['status']} {rep.get('kept', '')}", flush=True)


def run_remote(filekey, recs, po, workers, dataset=608):
    key = os.environ.get("AIHUB_APIKEY")
    if not key:
        sys.exit("환경 변수 AIHUB_APIKEY에 AI Hub API 키를 넣고 실행 (PowerShell: $env:AIHUB_APIKEY = '...')")
    new_src, segs = aihub.probe(dataset, filekey, key)
    for base, sg in segs.items():
        if base.lower().endswith(".zip"):
            prep.process_remote(lambda: zipfile.ZipFile(aihub.ConcatFile(new_src(), sg)), recs, po,
                                os.path.basename(base), workers=workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--labels", nargs="+", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", nargs="+")
    src.add_argument("--remote", nargs="+", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--tar", action="store_true")
    a = ap.parse_args()

    po = prep.PrepOutput(a.out)
    recs = [r for r in prep.load_labels(a.labels) if prep.is_phrase_script(r) and r["speaker"]]
    print(f"라벨: 짧은 문장 스크립트 녹음 {len(recs)}개, 화자 {len({r['speaker'] for r in recs})}명")
    for z in a.zip or []:
        run_local(z, recs, po, a.workers or 4)
    for k in a.remote or []:
        run_remote(k, recs, po, a.workers or 6)
    po.finalize(os.path.join(a.out, "data.tar") if a.tar else None)


if __name__ == "__main__":
    main()
