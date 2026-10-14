"""표준 스크립트 정렬(prep.process_record의 aligned_canonical) 점검 + 버그 주입

사용: python checks/check_canonical.py <작업 디렉터리> <원천 zip(TS01)> <라벨 zip들...>
  라벨 문장 수 = 구간 수로 정상 정렬되는 녹음을 골라
  1. 라벨을 일부러 망가뜨림(두 문장 병합 + 한 문장 누락) → 표준 스크립트 정렬이 나머지 자리를 원래 결과와
     바이트 단위로 같게 복원하고, 망가뜨린 자리는 빼는지
  2. 버그 주입: 표준 스크립트를 한 칸 밀어 넣으면 정렬 교차 검증(verify_alignment)이 그 녹음을 거르는지
     (교차 검증 기준은 같은 스크립트를 읽은 다른 녹음들 = 정상 정렬 결과)
"""
import copy
import hashlib
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from spk_train import prep  # noqa: E402


def run(rec, src, member, out):
    with src.open(member) as f:
        return prep.process_record(rec, f, out)


def main(work, src_zip, *label_zips):
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    recs = [r for r in prep.load_labels(label_zips) if prep.is_phrase_script(r) and r["speaker"]]
    canon = prep.canonical_script(recs)
    src = zipfile.ZipFile(src_zip)
    members = {os.path.basename(prep.fix_name(i.filename)): i for i in src.infolist() if i.filename.lower().endswith(".wav")}
    cands = [r for r in recs if r["wav"] in members and r["sentences"] == canon]
    print(f"표준 스크립트 {len(canon)}문장, 라벨이 표준과 같은 시험 후보 {len(cands)}개")
    ok = True

    ref_lines, target = [], None
    for r in cands:                                              # 정상 정렬: 교차 검증 기준 + 시험 대상
        lines, rep = run(r, src, members[r["wav"]], os.path.join(work, "normal"))
        if rep["status"] == "aligned":
            ref_lines += lines
            if target is None and len(lines) >= 50:
                target, target_lines = r, lines
    print(f"정상 정렬 {len({l['source'] for l in ref_lines})}개 녹음, 시험 대상 {target['wav']}")

    # 1. 라벨 망가뜨리기: 10·11번 병합, 30번 누락
    bad = copy.deepcopy(target)
    ss = list(canon)
    merged = ss[10].rstrip(".?!") + " " + ss[11]
    bad["sentences"] = ss[:10] + [merged] + ss[12:30] + ss[31:]
    bad["canonical"] = canon
    lines_b, rep_b = run(bad, src, members[bad["wav"]], os.path.join(work, "broken"))
    want = [l for l in target_lines if int(l["id"].rsplit("_", 1)[1]) not in (10, 11, 30)]
    same = [json.dumps(l, sort_keys=True) for l in lines_b] == [json.dumps(l, sort_keys=True) for l in want] and all(
        hashlib.sha256(open(os.path.join(work, "broken", l["audio"]), "rb").read()).digest() ==
        hashlib.sha256(open(os.path.join(work, "normal", l["audio"]), "rb").read()).digest() for l in lines_b)
    print(f"1. 라벨 병합·누락 → 상태 {rep_b['status']}, 확인 안 된 자리 {rep_b.get('unconfirmed')}, "
          f"나머지 {len(lines_b)}자리가 원래 결과와 같음 {same}")
    ok &= rep_b["status"] == "aligned_canonical" and same and rep_b.get("unconfirmed") == 3

    # 2. 버그 주입: 표준 스크립트를 앞뒤로 한 칸 밀기 → 교차 검증이 걸러야 함
    others = [l for l in ref_lines if l["source"] != target["wav"]]
    v_ok = prep.verify_alignment(others + lines_b)[target["wav"]]
    print(f"2. 교차 검증: 정상 복원 통과 {v_ok['ok']} (rho {v_ok['rho']:.2f}, 밀린 경우 {v_ok['rho_shift']:.2f})")
    ok &= v_ok["ok"]
    for name, sc in (("앞으로", canon[1:] + canon[:1]), ("뒤로", canon[-1:] + canon[:-1])):
        shifted = copy.deepcopy(bad)
        shifted["canonical"] = sc
        shifted["sentences"] = sc                               # 라벨도 밀린 순서라 확인 자리가 그대로 남게
        shifted["wav"] = f"SHIFT_{name}_" + bad["wav"]
        lines_s, _ = run(shifted, src, members[bad["wav"]], os.path.join(work, "shifted"))
        v_bad = prep.verify_alignment(others + lines_s)[shifted["wav"]]
        print(f"   {name} 한 칸 밀린 정렬 거부 {not v_bad['ok']} (rho {v_bad['rho']:.2f}, 되돌린 경우 {v_bad['rho_shift']:.2f})")
        ok &= not v_bad["ok"]
    print("결과: " + ("모두 통과" if ok else "실패"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
