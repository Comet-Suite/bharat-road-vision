"""Annotate the whole manifest in resumable slices (the sandbox has ~1 GB RAM)."""
import json, os, sys
sys.path.insert(0, 'code')
import annotate as A
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

SLICE = 300
recs = [r for r in (json.loads(l) for l in open('state/manifest.jsonl')) if r.get('has_file')]
done_path = 'state/annotated.jsonl'
done_ids = set()
if os.path.exists(done_path):
    for line in open(done_path):
        try:
            done_ids.add(json.loads(line)['original_url'])
        except Exception:
            pass
todo = [r for r in recs if r['url'] not in done_ids]
print(f'manifest={len(recs)} already={len(done_ids)} todo={len(todo)}', flush=True)


def work(r):
    try:
        out, why = A.annotate(r, 'build')
        return out, why
    except Exception as e:
        return None, 'exc:' + type(e).__name__


reasons = Counter()
with open(done_path, 'a') as fh:
    for i in range(0, len(todo), SLICE):
        chunk = todo[i:i + SLICE]
        with ThreadPoolExecutor(3) as ex:
            for out, why in ex.map(work, chunk):
                reasons[why] += 1
                if out:
                    fh.write(json.dumps(out, ensure_ascii=False) + '\n')
        fh.flush()
        print(f'  slice {i}-{i+len(chunk)}  ok={reasons["ok"]}  {dict(reasons)}', flush=True)

# ---- merge + dedupe by perceptual hash
rows, seen_hash, kept = [], set(), []
for line in open(done_path):
    try:
        r = json.loads(line)
    except Exception:
        continue
    rows.append(r)
for r in rows:
    h = int(r['dhash16'], 16)
    key = h >> 12                     # coarse visual bucket
    dup = False
    for o in kept:
        if bin(h ^ int(o['dhash16'], 16)).count('1') <= 8:
            dup = True
            break
    if not dup:
        kept.append(r)
with open(done_path, 'w') as fh:
    for r in kept:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print('total', len(rows), 'after dedupe', len(kept))
print(Counter(r['label'] for r in kept).most_common())
print('tiers', Counter(r['confidence_tier'] for r in kept).most_common())
