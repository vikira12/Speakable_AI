"""AI Hub 다운로드를 디스크에 저장하지 않고 필요한 파일만 읽기

AI Hub API(aihubshell과 같은 주소)는 tar 하나를 내려준다. 큰 zip은 tar 안에서 "<이름>.part0, part1, …"로
나뉘어 있고, 이어 붙이면 원래 zip이 된다. 서버가 HTTP Range를 지원하면:
  1. tar 헤더만 건너뛰며 읽어 각 part의 위치를 찾고 (RangeTar)
  2. part들을 이어 붙인 가상 파일(ConcatFile)을 zipfile로 열어
  3. 필요한 wav 멤버만 그 구간을 받아 읽는다.
Range를 지원하지 않으면 probe()가 알려 주고, 이때는 전체를 디스크에 받아야 한다 (download_tar).
"""
import io
import re
import tarfile
import threading
import time
from collections import OrderedDict

import requests

API = "https://api.aihub.or.kr/down/0.6/{dataset}.do"


class RangeSource:
    """HTTP Range로 임의 위치를 읽는다. 블록 단위 캐시 (스레드마다 하나씩 만들어 쓴다)"""

    def __init__(self, url, headers=None, block=4 << 20, cache_blocks=16, retries=5, resolver=None):
        """resolver: 주소가 만료됐을 때(4xx) 새 주소를 받아 오는 함수"""
        self.url, self.headers, self.resolver = url, dict(headers or {}), resolver
        self.block, self.cache_blocks, self.retries = block, cache_blocks, retries
        self.cache = OrderedDict()
        self.bytes_fetched = 0
        self.size = self._size()

    def _get(self, start, end):
        for k in range(self.retries):
            try:
                r = requests.get(self.url, headers={**self.headers, "Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code == 206:
                    self.bytes_fetched += len(r.content)
                    return r.content
                if 400 <= r.status_code < 500 and self.resolver:
                    self.url = self.resolver()
                raise IOError(f"Range 요청 실패: HTTP {r.status_code}")
            except (requests.RequestException, IOError):
                if k == self.retries - 1:
                    raise
                time.sleep(2 ** k)

    def _size(self):
        r = requests.get(self.url, headers={**self.headers, "Range": "bytes=0-0"}, timeout=60, stream=True)
        cr = r.headers.get("Content-Range", "")
        r.close()
        if r.status_code != 206 or "/" not in cr:
            raise IOError(f"서버가 Range를 지원하지 않음 (HTTP {r.status_code}, Content-Range '{cr}')")
        return int(cr.split("/")[1])

    def read(self, offset, n):
        out = bytearray()
        while n > 0 and offset < self.size:
            b = offset // self.block
            if b not in self.cache:
                s = b * self.block
                self.cache[b] = self._get(s, min(s + self.block, self.size) - 1)
                if len(self.cache) > self.cache_blocks:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(b)
            data = self.cache[b]
            o = offset - b * self.block
            take = data[o:o + n]
            out += take
            offset += len(take)
            n -= len(take)
        return bytes(out)


class FileSource:
    """로컬 파일을 RangeSource처럼 (점검용)"""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.f.seek(0, 2)
        self.size = self.f.tell()
        self.bytes_fetched = 0
        self.lock = threading.Lock()

    def read(self, offset, n):
        with self.lock:
            self.f.seek(offset)
            d = self.f.read(n)
        self.bytes_fetched += len(d)
        return d


class ConcatFile(io.RawIOBase):
    """source의 (위치, 크기) 구간들을 이어 붙인 읽기 전용·seek 가능 파일"""

    def __init__(self, source, segments):
        self.src, self.segs = source, list(segments)
        self.starts = []
        t = 0
        for _, size in self.segs:
            self.starts.append(t)
            t += size
        self.length, self.pos = t, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.length + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.length - self.pos)
        got = 0
        while got < n:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            off, size = self.segs[i]
            inner = self.pos - self.starts[i]
            take = min(n - got, size - inner)
            d = self.src.read(off + inner, take)
            b[got:got + len(d)] = d
            got += len(d)
            self.pos += len(d)
            if len(d) < take:
                break
        return got


def tar_members(source):
    """tar 헤더만 읽어 [(이름, 데이터 위치, 크기)]. 데이터 본문은 받지 않는다"""
    f = io.BufferedReader(ConcatFile(source, [(0, source.size)]), buffer_size=64 << 10)
    out = []
    with tarfile.open(fileobj=f, mode="r:") as t:
        for m in t:
            if m.isfile():
                out.append((fix_name(m.name), m.offset_data, m.size))
    return out


def fix_name(n):
    try:
        return n.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return n


def zip_segments(members):
    """tar 멤버 → {zip 이름: [(위치, 크기)...]} (part 번호 순서로 정렬)"""
    groups = {}
    for name, off, size in members:
        m = re.match(r"(.*)\.part(\d+)$", name)
        base, k = (m.group(1), int(m.group(2))) if m else (name, 0)
        groups.setdefault(base, []).append((k, off, size))
    return {base: [(o, s) for _, o, s in sorted(v)] for base, v in groups.items()}


def resolve(dataset, filekey, apikey):
    """API 주소의 리다이렉트를 따라간 최종 주소 (헤더 apikey는 계속 붙인다)"""
    url = API.format(dataset=dataset) + f"?fileSn={filekey}"
    r = requests.get(url, headers={"apikey": apikey, "Range": "bytes=0-0"}, timeout=60, stream=True)
    final, status, ctype = r.url, r.status_code, r.headers.get("Content-Type", "")
    body = r.raw.read(512, decode_content=True) if status >= 400 else b""
    r.close()
    if status >= 400:
        raise IOError(f"다운로드 요청 실패 HTTP {status}: {body[:300]!r}")
    return final, status, ctype


def probe(dataset, filekey, apikey, log=print):
    """Range 지원 여부와 tar 안 구성을 확인 → (RangeSource 만드는 함수, {zip: 구간})"""
    url, status, ctype = resolve(dataset, filekey, apikey)
    log(f"응답 HTTP {status}, Content-Type {ctype}")
    def again():
        return resolve(dataset, filekey, apikey)[0]

    src = RangeSource(url, {"apikey": apikey}, block=64 << 10, cache_blocks=64, resolver=again)
    log(f"Range 지원 확인: 전체 {src.size / 1e9:.1f} GB")
    members = tar_members(src)
    segs = zip_segments(members)
    for base, s in segs.items():
        log(f"  {base}: part {len(s)}개, {sum(x for _, x in s) / 1e9:.1f} GB")
    log(f"tar 헤더 읽기에 받은 양 {src.bytes_fetched / 1e6:.1f} MB")
    return (lambda: RangeSource(url, {"apikey": apikey}, resolver=again)), segs


def download_tar(dataset, filekey, apikey, dst_dir, log=print):
    """작은 파일(라벨 zip 등)용: tar를 받아 풀고 part를 합친다 → 풀린 파일 경로 목록"""
    import os
    url = API.format(dataset=dataset) + f"?fileSn={filekey}"
    os.makedirs(dst_dir, exist_ok=True)
    with requests.get(url, headers={"apikey": apikey}, stream=True, timeout=120) as r:
        if r.status_code != 200:
            raise IOError(f"다운로드 실패 HTTP {r.status_code}: {r.raw.read(300)!r}")
        r.raw.decode_content = True
        parts = {}
        with tarfile.open(fileobj=r.raw, mode="r|") as t:
            for m in t:
                if not m.isfile():
                    continue
                name = fix_name(m.name)
                mm = re.match(r"(.*)\.part(\d+)$", name)
                base, k = (mm.group(1), int(mm.group(2))) if mm else (name, 0)
                tmp = os.path.join(dst_dir, os.path.basename(name))
                with open(tmp, "wb") as f:
                    f.write(t.extractfile(m).read())
                parts.setdefault(base, []).append((k, tmp))
    out = []
    for base, ps in parts.items():
        dst = os.path.join(dst_dir, os.path.basename(base))
        if len(ps) == 1 and ps[0][1] == dst:
            out.append(dst)
            continue
        with open(dst + ".tmp", "wb") as f:
            for _, p in sorted(ps):
                f.write(open(p, "rb").read())
                os.remove(p)
        os.replace(dst + ".tmp", dst)
        out.append(dst)
    log("받음: " + ", ".join(os.path.basename(p) for p in out))
    return out
