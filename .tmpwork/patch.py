import io, sys, json

def load(p):
    raw = io.open(p, encoding="utf-8", newline="").read()
    crlf = "\r\n" in raw
    return raw.replace("\r\n", "\n"), crlf

def save(p, s, crlf):
    if crlf:
        s = s.replace("\n", "\r\n")
    io.open(p, "w", encoding="utf-8", newline="").write(s)

def apply(path, pairs):
    s, crlf = load(path)
    for old, new in pairs:
        if old not in s:
            raise SystemExit("NOT FOUND in %s:\n%s" % (path, old[:300]))
        if s.count(old) != 1:
            raise SystemExit("AMBIGUOUS (%d) in %s:\n%s" % (s.count(old), path, old[:300]))
        s = s.replace(old, new)
    save(path, s, crlf)
    print("patched", path)
