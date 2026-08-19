"""Extract individual members from a remote HF zip via HTTP range requests.
No full download: reads the central directory from the tail, then fetches
only the bytes of the requested member(s)."""
import io, urllib.request, zipfile, sys, os

UA = {"User-Agent": "curl/8"}

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        # resolve redirect once and get total size via Content-Range
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        r = urllib.request.urlopen(req, timeout=60)
        self.url = r.geturl()
        cr = r.headers.get("Content-Range", "")
        self.size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers.get("Content-Length", 0))
        r.read()
        self.pos = 0
        self.bytes_fetched = 0
    def seekable(self): return True
    def readable(self): return True
    def seek(self, off, whence=0):
        if whence == 0: self.pos = off
        elif whence == 1: self.pos += off
        elif whence == 2: self.pos = self.size + off
        return self.pos
    def tell(self): return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": "bytes=%d-%d" % (self.pos, end)})
        data = urllib.request.urlopen(req, timeout=120).read()
        self.pos += len(data); self.bytes_fetched += len(data)
        return data
    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

def open_zip(url):
    hf = HttpFile(url)
    return zipfile.ZipFile(hf), hf

def list_members(url, limit=None):
    zf, hf = open_zip(url)
    names = zf.namelist()
    return names[:limit] if limit else names

def extract(url, member, out_path):
    zf, hf = open_zip(url)
    data = zf.read(member)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f: f.write(data)
    return len(data), hf.bytes_fetched

if __name__ == "__main__":
    url = sys.argv[1]
    zf, hf = open_zip(url)
    names = zf.namelist()
    print("zip size: %.1f MB  members: %d  (CD read used %.0f KB)" % (hf.size/1e6, len(names), hf.bytes_fetched/1e3))
    for n in names[:6]:
        i = zf.getinfo(n)
        print("   %-55s %6.2f MB  compress_type=%d" % (n, i.file_size/1e6, i.compress_type))
