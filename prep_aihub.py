"""AI Hub 013 원천 → 학습 데이터 디렉터리 (manifest.jsonl + 16kHz 문장 구간 wav). 국내 PC 전용

AI Hub 데이터는 국외 다운로드·반출 금지: 받기·자르기·학습 모두 이 PC에서 하고, 결과(data/, runs/)를
Google Drive·Colab·GitHub 등에 올리지 않는다.

사용
  python prep_aihub.py <출력 디렉터리> --labels <라벨 zip...> --zip <원천 zip...>          PC에 있는 원천 zip
  python prep_aihub.py <출력 디렉터리> --labels <라벨 zip...> --download <filekey...>     AI Hub에서 받으면서 처리
      API 키는 환경 변수 AIHUB_APIKEY (명령줄에 쓰지 않음)
      filekey: TS01 32705, VS01 65521, TS02 55286, TS03 55287
      원천 zip을 디스크에 만들지 않는다. 연결이 끊기면 처음부터 다시 받고(이어받기 불가) 처리한 녹음은 건너뛴다.
      해당 원천의 짧은 문장 녹음을 이미 모두 처리했으면 받지 않는다
  python prep_aihub.py <출력 디렉터리> --labels <라벨 zip...> --salvage <zip 경로>        끊긴 받기의 조각 파일
      <zip 경로>.partial, <zip 경로>.held<위치> 를 이어 읽어 온전히 받은 녹음만 처리
  (--zip, --salvage, --download는 같이 써도 된다)
  공통 옵션: --workers N (기본 4), --tar (검증 통과분 data.tar), --retry-mismatch (정렬 실패로 기록된 녹음 다시 처리)
라벨 문장 수와 구간 수가 안 맞는 생활 표현 스크립트(-02-02-) 녹음은 라벨에 가장 많이 나온 표준 60문장 순서로 맞추고,
라벨이 표준 문장과 같은 자리만 쓴다 (라벨의 문장 병합·누락·오타 대응)
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

DATASET = 608
# filekey → (원천 이름 앞부분, 짝이 되는 라벨 zip 이름 앞부분)
SOURCES = {32705: ("TS01", "TL01"), 65521: ("VS01", "VL01"), 55286: ("TS02", "TL02"), 55287: ("TS03", "TL03")}

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


def run_local(src_zip, recs, po, workers, label_prefix=None):
    """label_prefix: 이 원천과 짝인 라벨의 녹음 중 원천에 음성이 없는 것은 no_audio로 기록 (다시 받지 않게)"""
    members = {os.path.basename(prep.fix_name(i.filename)): i.filename
               for i in zipfile.ZipFile(src_zip).infolist() if i.filename.lower().endswith(".wav")}
    todo = [r for r in recs if r["wav"] in members and r["wav"] not in po.done]
    print(f"{os.path.basename(src_zip)}: 짧은 문장 녹음 {sum(r['wav'] in members for r in recs)}개, 처리할 것 {len(todo)}개")
    with Pool(workers) as pool:
        for k, (lines, rep) in enumerate(pool.imap_unordered(_work, [(src_zip, members[r["wav"]], r, po.out) for r in todo])):
            po.add(lines, rep, os.path.basename(src_zip))
            print(f"[{k + 1}/{len(todo)}] {rep['wav']}: {rep['status']} {rep.get('kept', '')}", flush=True)
    if label_prefix:
        for r in recs:
            if r["zip"].startswith(label_prefix) and r["wav"] not in members and r["wav"] not in po.done:
                po.add([], {"wav": r["wav"], "speaker": r["speaker"], "status": "no_audio"}, os.path.basename(src_zip))


def _label_done(recs, po, label_prefix):
    mine = [r for r in recs if r["zip"].startswith(label_prefix)]
    return mine, [r for r in mine if r["wav"] not in po.done]


def _mark_no_audio(recs, po, label_prefix, seen, source_zip):
    """짝 라벨의 녹음 중 원천 zip에 음성이 없는 것은 no_audio로 기록 (다시 받지 않게)"""
    for r in recs:
        if r["zip"].startswith(label_prefix) and r["wav"] not in seen and r["wav"] not in po.done:
            po.add([], {"wav": r["wav"], "speaker": r["speaker"], "status": "no_audio"}, source_zip)


def run_download(filekey, recs, po, workers):
    if filekey not in SOURCES:
        sys.exit(f"알 수 없는 filekey {filekey} (가능: {sorted(SOURCES)})")
    src_name, label_prefix = SOURCES[filekey]
    mine, left = _label_done(recs, po, label_prefix)
    if mine and not left:
        print(f"{src_name}: 짧은 문장 녹음 {len(mine)}개 모두 처리됨, 받지 않음")
        return
    key = os.environ.get("AIHUB_APIKEY")
    if not key:
        sys.exit("환경 변수 AIHUB_APIKEY에 AI Hub API 키를 넣고 실행 (PowerShell: $env:AIHUB_APIKEY = '...')")
    print(f"{src_name}: 처리할 녹음 {len(left)}개 → 받으면서 처리 (디스크에 원천을 저장하지 않음)")
    seen = {}

    def on_zip(base, f):
        if base.lower().endswith(".zip"):
            seen[base] = prep.process_entries(f, recs, po, base, workers)

    aihub.stream_with_retry(DATASET, filekey, key, on_zip)
    for base, s in seen.items():
        _mark_no_audio(recs, po, label_prefix, s, base)


def run_salvage(base_path, recs, po, workers):
    """끊긴 받기에서 남은 조각 파일(<zip>.partial, <zip>.held<위치>)로 처리"""
    lp = aihub.LocalParts(base_path)
    if not lp.files:
        sys.exit(f"조각 파일 없음: {base_path}.partial")
    name = os.path.basename(base_path)
    print(f"{name}: 조각 {len(lp.files)}개, 앞부분 {lp.size / 1e9:.1f} GB에서 처리")
    prep.process_entries(lp, recs, po, name, workers, stop_on_truncated=True)
    prefix = next((lab for src, lab in SOURCES.values() if name.startswith(src)), None)
    if prefix:
        mine, left = _label_done(recs, po, prefix)
        print(f"{name}: 짝 라벨의 짧은 문장 녹음 {len(mine)}개 중 남은 것 {len(left)}개 (나머지는 --download로)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--zip", nargs="+")
    ap.add_argument("--download", nargs="+", type=int)
    ap.add_argument("--salvage", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tar", action="store_true")
    ap.add_argument("--retry-mismatch", action="store_true", help="정렬 실패(count_mismatch)로 기록된 녹음을 다시 처리")
    a = ap.parse_args()
    if not (a.zip or a.download or a.salvage):
        ap.error("--zip, --download, --salvage 중 하나 이상 필요")

    po = prep.PrepOutput(a.out)
    recs = [r for r in prep.load_labels(a.labels) if prep.is_phrase_script(r) and r["speaker"]]
    canon = prep.canonical_script(recs)
    for r in recs:
        if canon and "-02-02-" in r["wav"]:
            r["canonical"] = canon
    if a.retry_mismatch:
        po.done -= {r["wav"] for r in po.report["records"] if r["status"] == "count_mismatch"}
    print(f"라벨: 짧은 문장 스크립트 녹음 {len(recs)}개, 화자 {len({r['speaker'] for r in recs})}명")
    for z in a.zip or []:
        run_local(z, recs, po, a.workers)
    for p in a.salvage or []:
        run_salvage(p, recs, po, a.workers)
    for k in a.download or []:
        run_download(k, recs, po, a.workers)
    po.finalize(os.path.join(a.out, "data.tar") if a.tar else None)


if __name__ == "__main__":
    main()
