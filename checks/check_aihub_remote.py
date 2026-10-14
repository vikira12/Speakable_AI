"""원격 읽기(spk_train/aihub.py, prep.process_remote) 점검: AI Hub와 같은 형식의 로컬 HTTP 서버로 확인

사용: python checks/check_aihub_remote.py <작업 디렉터리> <원천 zip(TS01)> <라벨 zip> [...]
  1. 원천 zip에서 짧은 문장 녹음 2개를 꺼내 작은 zip을 만들고, part 12개로 나눠 순서를 섞은 tar로 묶는다
  2. Range·리다이렉트·apikey 헤더를 지원하는 로컬 서버로 내보낸다 (Range 없는 서버도 하나)
  3. probe → process_remote 결과(구간 wav, manifest)가 로컬 zip을 직접 처리한 결과와 바이트 단위로 같은지 비교,
     prep_aihub.py --remote(명령줄, 환경 변수 키)도 같은 결과인지, 다시 실행하면 이미 처리한 녹음을 건너뛰는지
  4. Range 없는 서버는 probe가 거부하는지, 라벨용 download_tar가 part를 올바르게 합치는지 확인
"""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from spk_train import aihub, prep  # noqa: E402

KEY = "test-key"


def make_server(tar_path, use_range=True):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/down/"):                       # API 주소 → 실제 파일로 리다이렉트
                self.send_response(302)
                self.send_header("Location", "/file.tar")
                self.end_headers()
                return
            if self.headers.get("apikey") != KEY:
                self.send_response(401)
                self.end_headers()
                return
            size = os.path.getsize(tar_path)
            rng = self.headers.get("Range")
            with open(tar_path, "rb") as f:
                if rng and use_range:
                    a, b = rng.split("=")[1].split("-")
                    a, b = int(a), min(int(b) if b else size - 1, size - 1)
                    f.seek(a)
                    body = f.read(b - a + 1)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
                else:
                    body = f.read()
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(work, src_zip, *label_zips):
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    recs = [r for r in prep.load_labels(label_zips) if prep.is_phrase_script(r) and r["speaker"]]
    src = zipfile.ZipFile(src_zip)
    members = {os.path.basename(prep.fix_name(i.filename)): i for i in src.infolist() if i.filename.lower().endswith(".wav")}
    pick = sorted((r for r in recs if r["wav"] in members), key=lambda r: members[r["wav"]].file_size)[1:3]
    print("시험 녹음:", [r["wav"] for r in pick])

    # 1. 작은 zip → part 12개 → 순서를 섞은 tar
    small = os.path.join(work, "small.zip")
    with zipfile.ZipFile(small, "w", zipfile.ZIP_DEFLATED) as z:
        for r in pick:
            z.writestr(members[r["wav"]].filename, src.read(members[r["wav"]]))
    data = open(small, "rb").read()
    n = 12
    cut = [len(data) * k // n for k in range(n + 1)]
    names = [f"013.test/01.데이터/원천데이터/TS_test.zip.part{k}" for k in range(n)]
    order = list(range(n))
    random.Random(1).shuffle(order)
    tar_path = os.path.join(work, "download.tar")
    with tarfile.open(tar_path, "w") as t:
        for k in order:
            p = os.path.join(work, f"p{k}")
            open(p, "wb").write(data[cut[k]:cut[k + 1]])
            t.add(p, names[k])
    print(f"tar 순서: {order}")

    ok = True
    srv = make_server(tar_path)
    aihub.API = f"http://127.0.0.1:{srv.server_port}/down/{{dataset}}.do"

    # 2~3. probe → process_remote vs 로컬 직접 처리
    new_src, segs = aihub.probe(608, 1, KEY)
    (base, sg), = segs.items()
    ok &= base.endswith("TS_test.zip") and len(sg) == n
    po_r = prep.PrepOutput(os.path.join(work, "remote"))
    prep.process_remote(lambda: zipfile.ZipFile(aihub.ConcatFile(new_src(), sg)), recs, po_r, "TS_test.zip", workers=2)
    po_l = prep.PrepOutput(os.path.join(work, "local"))
    for r in pick:
        with src.open(members[r["wav"]]) as f:
            po_l.add(*prep.process_record(r, f, po_l.out), "TS_test.zip")
    ml = sorted(open(po_l.man_path, encoding="utf8").read().splitlines())
    mr = sorted(open(po_r.man_path, encoding="utf8").read().splitlines())
    same_wav = all(sha(os.path.join(po_l.out, json.loads(l)["audio"])) == sha(os.path.join(po_r.out, json.loads(l)["audio"])) for l in ml)
    print(f"manifest 줄 {len(ml)} / {len(mr)}, 같음 {ml == mr}, 구간 wav 모두 같음 {same_wav}")
    ok &= ml == mr and same_wav and len(ml) > 0

    # 명령줄 prep_aihub.py --remote (키는 환경 변수) → 같은 결과, 두 번째 실행은 아무것도 받지 않음
    cli_out = os.path.join(work, "cli")
    env = dict(os.environ, AIHUB_APIKEY=KEY, AIHUB_API_URL=aihub.API)
    cmd = [sys.executable, os.path.join(ROOT, "prep_aihub.py"), cli_out, "--labels", *label_zips, "--remote", "1", "--workers", "2"]
    r1 = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf8")
    mc = sorted(open(os.path.join(cli_out, "manifest.jsonl"), encoding="utf8").read().splitlines()) if r1.returncode == 0 else []
    same_cli = mc == mr and all(sha(os.path.join(cli_out, json.loads(l)["audio"])) == sha(os.path.join(po_r.out, json.loads(l)["audio"])) for l in mr)
    print(f"명령줄 --remote: 종료 코드 {r1.returncode}, manifest·구간 wav 같음 {same_cli}")
    r2 = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf8")
    again = "처리할 것 0개" in r2.stdout
    print(f"명령줄 다시 실행: 이미 처리한 녹음 건너뜀 {again}")
    r3 = subprocess.run(cmd, env={k: v for k, v in env.items() if k != "AIHUB_APIKEY"}, capture_output=True, text=True, encoding="utf8")
    print(f"키 없이 실행하면 멈춤 {r3.returncode != 0}")
    ok &= same_cli and again and r3.returncode != 0
    if r1.returncode != 0:
        print(r1.stdout[-1500:], r1.stderr[-1500:])

    # 큰 zip(zip64): 원천 zip 자체를 가상 part 7개로 나눠 열고, 녹음 하나가 직접 읽은 것과 같은지
    fs = aihub.FileSource(src_zip)
    cuts = [fs.size * k // 7 for k in range(8)]
    vz = zipfile.ZipFile(aihub.ConcatFile(fs, [(cuts[k], cuts[k + 1] - cuts[k]) for k in range(7)]))
    m = members[pick[0]["wav"]]
    same_big = len(vz.infolist()) == len(src.infolist()) and \
        hashlib.sha256(vz.read(m.filename)).digest() == hashlib.sha256(src.read(m)).digest()
    print(f"원천 zip {fs.size / 1e9:.1f} GB 가상 part 7개: 목록·녹음 내용 같음 {same_big}")
    ok &= same_big

    # 4. download_tar (작은 파일용) 이 part를 합쳐 원래 zip을 만드는지
    got = aihub.download_tar(608, 1, KEY, os.path.join(work, "dl"))
    ok &= len(got) == 1 and open(got[0], "rb").read() == data
    print(f"download_tar 병합 결과가 원본 zip과 같음 {open(got[0], 'rb').read() == data}")

    # Range 없는 서버는 probe가 거부
    srv2 = make_server(tar_path, use_range=False)
    aihub.API = f"http://127.0.0.1:{srv2.server_port}/down/{{dataset}}.do"
    try:
        aihub.probe(608, 1, KEY)
        print("Range 없는 서버를 거부하지 못함")
        ok = False
    except IOError as e:
        print(f"Range 없는 서버 거부: {e}")
    # 잘못된 API 키
    aihub.API = f"http://127.0.0.1:{srv.server_port}/down/{{dataset}}.do"
    try:
        aihub.probe(608, 1, "wrong")
        ok = False
        print("잘못된 키를 거부하지 못함")
    except IOError as e:
        print(f"잘못된 키 거부: {e}")
    print("결과: " + ("모두 통과" if ok else "실패"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
