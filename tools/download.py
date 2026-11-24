"""Download candidate images to raw/ (original files, untouched).

Dedupes by URL and by perceptual identity later; writes state/downloaded.jsonl.
"""
import hashlib, json, os, sys, threading, time
import requests
from concurrent.futures import ThreadPoolExecutor

RAW = 'raw'
OUT = 'state/downloaded.jsonl'
S = requests.Session()
S.headers.update({'User-Agent': 'BharatRoadVision/1.0 (dataset construction; contact: arena-agent)',
                  'Accept': 'image/jpeg,image/png,image/*'})
lock = threading.Lock()
seen = set()
ok = fail = skip = 0


def norm_url(u):
    return u.split('?')[0]


def fetch(rec):
    global ok, fail, skip
    u = norm_url(rec['url'])
    key = hashlib.sha1(u.encode()).hexdigest()
    with lock:
        if key in seen:
            skip += 1
            return
        seen.add(key)
    path = os.path.join(RAW, key + '.raw')
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        with lock:
            rec2 = dict(rec, sha1=key, file=path, bytes=os.path.getsize(path))
            out.write(json.dumps(rec2) + '\n')
            ok += 1
        return
    for attempt in range(4):
        try:
            r = S.get(rec['url'], timeout=60, stream=True)
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1)); continue
            if r.status_code != 200:
                time.sleep(1.5 * (attempt + 1)); continue
            data = r.content
            if len(data) < 20000:
                break
            with open(path, 'wb') as f:
                f.write(data)
            with lock:
                out.write(json.dumps(dict(rec, sha1=key, file=path, bytes=len(data))) + '\n')
                ok += 1
            return
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    with lock:
        fail += 1


def main():
    global out
    os.makedirs(RAW, exist_ok=True)
    recs, seen_lines = [], set()
    for line in open('state/plan_ov.json'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = norm_url(r.get('url', ''))
        if not k or k in seen_lines:
            continue
        seen_lines.add(k)
        recs.append(r)
    out = open(OUT, 'a')
    print('candidates', len(recs), flush=True)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(fetch, recs))
    print(f'DONE ok={ok} fail={fail} skip={skip}', flush=True)


if __name__ == '__main__':
    main()
