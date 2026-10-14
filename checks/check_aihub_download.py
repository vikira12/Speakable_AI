"""AI Hub 받기(spk_train/aihub.py) + prep_aihub.py --download/--salvage 점검: AI Hub와 같은 형식의 로컬 서버

사용: python checks/check_aihub_download.py <작업 디렉터리> <원천 zip(TS01)> <라벨 zip(TL01)>
  가짜 서버: 실제 AI Hub처럼 tar를 chunked로 보내고(크기 정보 없음, Range 무시), zip은 바이트 위치 이름의 part
  1. --download 결과(구간 wav, manifest)가 --zip(zipfile로 로컬 처리) 결과와 바이트 단위로 같은지, 디스크에 zip을 만들지 않는지
  2. 전송 도중 끊기면(IncompleteRead) 처음부터 다시 받아 같은 결과인지, 처리한 녹음을 두 번 기록하지 않는지
  3. 다시 실행하면 서버에 요청하지 않는지, 키가 틀리거나 없으면 멈추는지
  4. 끊긴 받기의 조각(.partial, .held<위치>)으로 --salvage → 온전한 녹음만 처리, 이어서 --download로 나머지 → 전체가 --zip과 같은지
  5. part 순서가 어긋나면 오류, 녹음 데이터가 깨지면(CRC) 그 녹음만 오류로 기록
"""
import hashlib
import json
import os
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
PART = 1 << 20                                                  # 점검용 part 크기 (실제는 1GiB)


def make_tar(data, path, order=None, corrupt_at=None):
    """zip 바이트 → 바이트 위치 이름의 part tar. order: part 순서(섞기용), corrupt_at: 이 위치 바이트를 뒤집음"""
    if corrupt_at is not None:
        data = bytearray(data)
        data[corrupt_at] ^= 0xFF
        data = bytes(data)
    offs = list(range(0, len(data), PART))
    order = order or list(range(len(offs)))
    with tarfile.open(path, "w") as t:
        for i in order:
            p = path + ".tmp"
            open(p, "wb").write(data[offs[i]:offs[i] + PART])
            t.add(p, f"013.test/01.데이터/1.Training/원천데이터/TS01_뇌신경장애.zip.part{offs[i]}")
            os.remove(p)
    return path


def make_server(tar_path, plan=None):
    """plan: 요청마다 "full" 또는 "half"(절반 보내고 chunked 끝 표시 없이 연결을 끊음). 다 쓰면 full"""
    hits, plan = [], list(plan or [])

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass                                             # 받는 쪽이 끊은 경우 (점검 중 정상)

        def do_GET(self):
            hits.append(self.path)
            if self.headers.get("apikey") != KEY:
                body = "인증실패, 권한이 거부되었습니다\n".encode("utf8")
                self.send_response(502)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                return
            mode = plan.pop(0) if plan else "full"
            data = open(tar_path, "rb").read()
            if mode == "half":
                data = data[:len(data) // 2]
            self.send_response(200)                              # Range 요청이어도 전체를 보냄
            self.send_header("Content-Type", "application/x-tar")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(0, len(data), 64 << 10):
                c = data[i:i + (64 << 10)]
                self.wfile.write(f"{len(c):x}\r\n".encode() + c + b"\r\n")
            if mode == "full":
                self.wfile.write(b"0\r\n\r\n")
            self.close_connection = True

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, hits


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def manifest(out):
    p = os.path.join(out, "manifest.jsonl")
    return sorted(open(p, encoding="utf8").read().splitlines()) if os.path.exists(p) else []


def same_out(a, b):
    ma, mb = manifest(a), manifest(b)
    return len(ma) > 0 and ma == mb and all(
        sha(os.path.join(a, json.loads(l)["audio"])) == sha(os.path.join(b, json.loads(l)["audio"])) for l in ma)


def main(work, src_zip, label_zip):
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    recs = [r for r in prep.load_labels([label_zip]) if prep.is_phrase_script(r) and r["speaker"]]
    src = zipfile.ZipFile(src_zip)
    members = {os.path.basename(prep.fix_name(i.filename)): i for i in src.infolist() if i.filename.lower().endswith(".wav")}
    pick = sorted((r for r in recs if r["wav"] in members), key=lambda r: members[r["wav"]].file_size)[1:4]
    print("시험 녹음:", [r["wav"] for r in pick])
    small = os.path.join(work, "TS01_뇌신경장애.zip")              # 짝 라벨 TL01과 이름을 맞춤
    with zipfile.ZipFile(small, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("11.중풍/", b"")
        for r in pick:
            z.writestr(members[r["wav"]].filename, src.read(members[r["wav"]]))
    data = open(small, "rb").read()
    tar_ok = make_tar(data, os.path.join(work, "ok.tar"))
    ok = True

    def cli(out, *args, key=KEY, api=None):
        env = dict(os.environ, AIHUB_API_URL=api or aihub.API)
        if key is None:
            env.pop("AIHUB_APIKEY", None)
        else:
            env["AIHUB_APIKEY"] = key
        return subprocess.run([sys.executable, os.path.join(ROOT, "prep_aihub.py"), out, "--labels", label_zip, *args, "--workers", "2"],
                              env=env, capture_output=True, text=True, encoding="utf8", cwd=work)

    ref = os.path.join(work, "by_zip")
    r = cli(ref, "--zip", small)
    print(f"--zip 기준: 종료 {r.returncode}, manifest {len(manifest(ref))}줄")

    # 1. --download
    srv, hits = make_server(tar_ok)
    api = f"http://127.0.0.1:{srv.server_port}/down/{{dataset}}.do"
    out1 = os.path.join(work, "by_download")
    before = set(os.listdir(work))
    r = cli(out1, "--download", "32705", api=api)
    new = set(os.listdir(work)) - before - {"by_download"}
    s1 = r.returncode == 0 and same_out(ref, out1)
    print(f"1. --download = --zip: {s1}, 새로 생긴 파일 {sorted(new)} (zip을 만들지 않아야 함)")
    ok &= s1 and not new
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])

    # 2. 도중에 끊김 → 처음부터 다시 (재시도 대기 0초로 함수 직접 호출)
    srv2, hits2 = make_server(tar_ok, ["half", "full"])
    aihub.API = f"http://127.0.0.1:{srv2.server_port}/down/{{dataset}}.do"
    out2 = os.path.join(work, "dropped")
    po = prep.PrepOutput(out2)
    aihub.stream_with_retry(608, 32705, KEY, lambda b, f: prep.process_entries(f, recs, po, b, 2), retry_wait=0)
    reps = [x["wav"] for x in po.report["records"]]
    s2 = len(hits2) == 2 and same_out(ref, out2) and len(reps) == len(set(reps))
    print(f"2. 절반에서 끊김 → 다시 받아 같은 결과 {s2} (요청 {len(hits2)}번, 녹음 기록 {len(reps)}개, 중복 없음 {len(reps) == len(set(reps))})")
    ok &= s2

    # 3. 다시 실행·키 오류
    n = len(hits)
    r = cli(out1, "--download", "32705", api=api)
    s3 = r.returncode == 0 and len(hits) == n and "받지 않음" in r.stdout
    r_bad = cli(os.path.join(work, "bad"), "--download", "32705", key="wrong", api=api)
    r_none = cli(os.path.join(work, "none"), "--download", "32705", key=None, api=api)
    s3b = r_bad.returncode != 0 and "인증실패" in r_bad.stderr and r_none.returncode != 0
    print(f"3. 다시 실행 시 서버 요청 없음 {s3}, 틀린 키·키 없음 멈춤 {s3b}")
    ok &= s3 and s3b

    # 4. 조각 파일 → --salvage → --download로 나머지
    frag = os.path.join(work, "frag", "TS01_뇌신경장애.zip")
    os.makedirs(os.path.dirname(frag))
    cut = int(len(data) * 0.6)
    offs = list(range(0, cut, PART))
    for o in offs:
        open(frag + (".partial" if o == 0 else f".held{o}"), "wb").write(data[o:min(o + PART, cut)])
    open(frag + f".held{offs[-1] + PART}", "wb").close()           # 쓰다 만 빈 조각
    out4 = os.path.join(work, "salvaged")
    r = cli(out4, "--salvage", frag)
    got = {json.loads(l)["source"] for l in manifest(out4)}
    expect = set()
    with zipfile.ZipFile(small) as z:
        for i in z.infolist():
            end = i.header_offset + 30 + len(i.filename.encode("utf8")) + len(i.extra) + i.compress_size
            if i.filename.endswith(".wav") and end <= cut:
                expect.add(os.path.basename(i.filename))
    s4a = r.returncode == 0 and got == expect and len(expect) >= 1
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
    r = cli(out4, "--download", "32705", api=api)
    s4b = r.returncode == 0 and same_out(ref, out4)
    print(f"4. --salvage: 온전한 녹음 {sorted(got)} = 기대 {sorted(expect)} {s4a}, 이어서 --download 후 전체 = --zip {s4b}")
    ok &= s4a and s4b

    # 5. part 순서 어긋남 / 데이터 깨짐
    order = list(range(len(range(0, len(data), PART))))
    order[1], order[2] = order[2], order[1]
    srv5, _ = make_server(make_tar(data, os.path.join(work, "swap.tar"), order=order))
    aihub.API = f"http://127.0.0.1:{srv5.server_port}/down/{{dataset}}.do"
    po5 = prep.PrepOutput(os.path.join(work, "swap"))
    try:
        aihub.stream_with_retry(608, 32705, KEY, lambda b, f: prep.process_entries(f, recs, po5, b, 2), retries=1, retry_wait=0)
        s5a = False
    except IOError as e:
        s5a = "순서" in str(e)
    with zipfile.ZipFile(small) as z:
        target = [i for i in z.infolist() if i.filename.endswith(".wav")][1]
    srv6, _ = make_server(make_tar(data, os.path.join(work, "bad.tar"), corrupt_at=target.header_offset + 200))
    aihub.API = f"http://127.0.0.1:{srv6.server_port}/down/{{dataset}}.do"
    po6 = prep.PrepOutput(os.path.join(work, "corrupt"))
    aihub.stream_with_retry(608, 32705, KEY, lambda b, f: prep.process_entries(f, recs, po6, b, 2), retries=1, retry_wait=0)
    st = {x["wav"]: x["status"] for x in po6.report["records"]}
    ref_st = {x["wav"]: x["status"] for x in prep.PrepOutput(ref).report["records"]}
    bad_name = os.path.basename(target.filename)
    s5b = st.get(bad_name, "").startswith("error") and all(st.get(w) == v for w, v in ref_st.items() if w != bad_name)
    print(f"5. part 순서 어긋남 오류 {s5a}, 깨진 녹음만 오류 기록 {s5b} ({st.get(bad_name, '')[:70]})")
    ok &= s5a and s5b

    print("결과: " + ("모두 통과" if ok else "실패"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
