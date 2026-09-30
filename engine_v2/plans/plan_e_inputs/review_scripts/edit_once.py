"""Exact-once string replacement preserving CRLF (scratch helper, Plan E E4)."""


def edit(p, pairs):
    b = open(p, 'rb').read()
    crlf = b'\r\n' in b
    s = b.decode('utf-8').replace('\r\n', '\n')
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, (p, old[:70], n)
        s = s.replace(old, new)
    if crlf:
        s = s.replace('\n', '\r\n')
    open(p, 'wb').write(s.encode('utf-8'))
