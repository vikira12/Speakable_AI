"""AI Hub 데이터셋 원천을 받으면서 바로 필요한 녹음만 꺼내기 (aihubshell과 같은 API). 국내 PC 전용

AI Hub 데이터는 국외에서 받거나 국외 서버(Colab 등)로 옮기면 안 된다.

API는 tar 하나를 chunked로 내려주고(크기 정보 없음, Range 이어받기 불가), 큰 zip은 tar 안에서
"<이름>.part<바이트 위치>"(part0, part1073741824, …)로 나뉘어 순서대로 들어 있다. 이어 붙이면 원래 zip.
zip의 각 항목 앞 헤더(local header)에 압축 크기·CRC가 있으므로, 끝의 목록(central directory) 없이도
앞에서부터 항목을 차례로 꺼낼 수 있다:
  tar 스트림 → part 이어 읽기(PartStream) → zip 항목 차례로(iter_zip_entries) → 필요한 wav만 압축 해제·CRC 확인
디스크에는 zip을 만들지 않는다. 같은 방식으로 중간에 끊긴 받기의 조각 파일(LocalParts)도 읽을 수 있다.
"""
import glob
import os
import re
import struct
import tarfile
import time
import zlib

import requests

API = os.environ.get("AIHUB_API_URL", "https://api.aihub.or.kr/down/0.6/{dataset}.do")   # 환경 변수는 점검용 로컬 서버


class DownloadError(IOError):
    """다시 받아도 소용없는 오류 (인증·권한 등)"""


class Truncated(IOError):
    """zip 항목이 끝까지 오기 전에 데이터가 끝남"""


class CorruptEntry(IOError):
    """zip 항목 하나가 깨짐 (압축 해제 실패, 크기·CRC 불일치). 그 항목만 건너뛰면 된다"""


def fix_name(n):
    """tar·zip에 UTF-8 표시 없이 cp949로 저장된 이름 복원"""
    try:
        return n.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return n


def part_key(name):
    """'xxx.zip.part1073741824' → ('xxx.zip', 1073741824), part 표시가 없으면 (이름, None)"""
    m = re.match(r"(.*)\.part(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


# ---------------------------------------------------------------- 바이트 스트림

class _Progress:
    """읽은 양·속도를 1GB마다 출력"""

    def __init__(self, f, log, every=1 << 30):
        self.f, self.log, self.every = f, log, every
        self.n, self.mark, self.t0 = 0, every, time.time()

    def read(self, n=-1):
        d = self.f.read(n)
        self.n += len(d)
        if self.n >= self.mark:
            self.log(f"  받음 {self.n / 1e9:.1f} GB ({self.n / 1e6 / max(time.time() - self.t0, 1e-6):.0f} MB/s)")
            self.mark += self.every
        return d


class PartStream:
    """tar 스트림 안의 '<base>.partN'들을 순서대로 이어 읽는 파일 객체.
    N은 part 번호(0,1,2…) 또는 바이트 위치(0, 1073741824, …). 순서가 어긋나면 오류"""

    def __init__(self, tar_iter, tar, base, first_member):
        self.it, self.t, self.base = tar_iter, tar, base
        self.count, self.pos, self.leftover = 0, 0, None
        self._open(first_member)

    def _open(self, m):
        _, k = part_key(os.path.basename(fix_name(m.name)))
        if k is not None and k not in (self.count, self.pos):
            raise IOError(f"{self.base}: part {k}가 순서에 맞지 않음 (기대: {self.count}번째, 위치 {self.pos})")
        self.cur = self.t.extractfile(m)
        self.count += 1

    def read(self, n):
        out = bytearray()
        while n > 0 and self.cur is not None:
            d = self.cur.read(n)
            if d:
                out += d
                n -= len(d)
                self.pos += len(d)
                continue
            m = next(self.it, None)                     # 다음 part
            while m is not None and not m.isfile():
                m = next(self.it, None)
            if m is None or part_key(os.path.basename(fix_name(m.name)))[0] != self.base:
                self.cur, self.leftover = None, m        # 이 zip의 part가 끝남
                break
            self._open(m)
        return bytes(out)


class LocalParts:
    """끊긴 받기에서 남은 조각 파일들(<base>.partial = part0, <base>.held<위치> …)을 이어 읽기.
    빈 조각이나 빠진 위치가 나오면 거기까지만"""

    def __init__(self, base_path):
        parts = []
        if os.path.exists(base_path + ".partial"):
            parts.append((0, base_path + ".partial"))
        for p in glob.glob(glob.escape(base_path) + ".held*"):
            parts.append((int(p.rsplit(".held", 1)[1]), p))
        parts.sort()
        self.files, pos = [], 0
        for off, p in parts:
            size = os.path.getsize(p)
            if off not in (pos, len(self.files)) or size == 0:
                break
            self.files.append(p)
            pos += size
        self.size = pos
        self.i, self.f = 0, (open(self.files[0], "rb") if self.files else None)

    def read(self, n):
        out = bytearray()
        while n > 0 and self.f is not None:
            d = self.f.read(n)
            if d:
                out += d
                n -= len(d)
                continue
            self.f.close()
            self.i += 1
            self.f = open(self.files[self.i], "rb") if self.i < len(self.files) else None
        return bytes(out)


# ---------------------------------------------------------------- zip 항목 차례로

def _read_exact(f, n):
    d = f.read(n)
    if len(d) < n:
        raise Truncated(f"{n}바이트 중 {len(d)}바이트만 남음")
    return d


def iter_zip_entries(f):
    """local header를 따라 zip 항목을 차례로 → (이름, 읽기 함수, 건너뛰기 함수).
    호출자는 둘 중 하나를 반드시 호출해야 다음 항목으로 간다. 목록(central directory)에 닿아야 정상 끝.
    목록 전에 데이터가 끝나면(항목 경계 포함) Truncated"""
    while True:
        sig = _read_exact(f, 4)
        if sig in (b"PK\x01\x02", b"PK\x05\x06", b"PK\x06\x06"):
            return                                       # central directory (zip64 끝 레코드 포함)
        if sig != b"PK\x03\x04":
            raise IOError(f"zip 항목 헤더가 아님: {sig!r}")
        _, flag, method, _, _, crc, csize, usize, nlen, xlen = struct.unpack("<HHHHHIIIHH", _read_exact(f, 26))
        raw_name = _read_exact(f, nlen)
        extra = _read_exact(f, xlen)
        name = raw_name.decode("utf8") if flag & 0x800 else fix_name(raw_name.decode("cp437"))
        if flag & 0x8:
            raise IOError(f"{name}: 크기가 항목 뒤에 있는 zip(data descriptor)은 지원하지 않음")
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:   # zip64: 원래 크기, 압축 크기 순
            p = 0
            while p + 4 <= len(extra):
                hid, hl = struct.unpack("<HH", extra[p:p + 4])
                if hid == 1:
                    vals = list(struct.unpack("<" + "Q" * (hl // 8), extra[p + 4:p + 4 + hl]))
                    if usize == 0xFFFFFFFF:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = vals.pop(0)
                p += 4 + hl
        done = []

        def read(name=name, method=method, csize=csize, usize=usize, crc=crc):
            done.append(1)
            data = _read_exact(f, csize)
            if method == 8:
                try:
                    data = zlib.decompress(data, -15)
                except zlib.error as e:
                    raise CorruptEntry(f"{name}: 압축 해제 실패 ({e})") from e
            elif method != 0:
                raise CorruptEntry(f"{name}: 지원하지 않는 압축 방식 {method}")
            if len(data) != usize or zlib.crc32(data) != crc:
                raise CorruptEntry(f"{name}: 압축 해제 결과가 헤더(크기·CRC)와 다름")
            return data

        def skip(csize=csize):
            done.append(1)
            left = csize
            while left > 0:
                d = f.read(min(left, 8 << 20))
                if not d:
                    raise Truncated("항목 중간에서 데이터가 끝남")
                left -= len(d)

        yield name, read, skip
        if not done:
            raise RuntimeError("iter_zip_entries: read 또는 skip을 호출해야 함")


# ---------------------------------------------------------------- 받기

def stream_zips(dataset, filekey, apikey, on_zip, log=print):
    """filekey를 받으면서 tar 안의 zip마다 on_zip(zip 이름, 파일 객체)를 호출 (zip을 디스크에 만들지 않음)"""
    url = API.format(dataset=dataset) + f"?fileSn={filekey}"
    with requests.get(url, headers={"apikey": apikey}, stream=True, timeout=300) as r:
        if r.status_code != 200:
            body = r.raw.read(500, decode_content=True).decode("utf8", "replace").strip()
            err = DownloadError if r.status_code in (401, 403) or "인증" in body or "권한" in body else IOError
            raise err(f"다운로드 실패 HTTP {r.status_code}: {body}")
        log(f"filekey {filekey} 받기 시작")
        r.raw.decode_content = True
        prog = _Progress(r.raw, log)
        with tarfile.open(fileobj=prog, mode="r|") as t:
            it = iter(t)
            m = next(it, None)
            while m is not None:
                if not m.isfile():
                    m = next(it, None)
                    continue
                base, _ = part_key(os.path.basename(fix_name(m.name)))
                ps = PartStream(it, t, base, m)
                on_zip(base, ps)
                while ps.read(8 << 20):                  # zip 끝(목록)까지 읽고 다음 파일로
                    pass
                m = ps.leftover
        log(f"filekey {filekey} 받기 끝: {prog.n / 1e9:.1f} GB")


def stream_with_retry(dataset, filekey, apikey, on_zip, log=print, retries=5, retry_wait=60):
    """연결이 끊기면 처음부터 다시 (on_zip은 이미 처리한 녹음을 건너뛰어야 한다)"""
    for attempt in range(1, retries + 1):
        try:
            return stream_zips(dataset, filekey, apikey, on_zip, log)
        except DownloadError:
            raise
        except Exception as e:                           # 연결 끊김, 데이터 잘림
            if attempt == retries:
                raise IOError(f"filekey {filekey}: {retries}번 시도했지만 실패 ({e})") from e
            log(f"받기 실패 ({type(e).__name__}: {e}) → {retry_wait * attempt}초 뒤 처음부터 다시 ({attempt + 1}/{retries})")
            time.sleep(retry_wait * attempt)
