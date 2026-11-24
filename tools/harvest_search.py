"""Discover candidate images for the Bharat Road Vision dataset.

Two independent originals:
  * Wikimedia Commons  -> search + category imageinfo (429-aware)
  * Openverse          -> CC-licensed aggregator API

Writes state/candidates.jsonl with: source, url, licence, author, w/h, query, weak class hint.
Resumable: appends, and skips queries already in state/done_queries.txt.
"""
import json, os, re, sys, time, threading
import requests

UA = 'BharatRoadVision/1.0 (non-commercial dataset research; contact: arena-agent)'
S = requests.Session()
S.headers['User-Agent'] = UA
OUT = 'state/candidates.jsonl'
DONE = 'state/done_queries.txt'
lock = threading.Lock()

# query -> (weak class hint, source)
QUERIES = [
    ('Maruti Suzuki Swift car', 'hatchback'), ('Maruti 800', 'hatchback'),
    ('Maruti Wagon R car', 'hatchback'), ('Hyundai Santro', 'hatchback'),
    ('Tata Indica car', 'hatchback'), ('Renault Kwid', 'hatchback'),
    ('Maruti Alto car', 'hatchback'), ('hatchback car India road', 'hatchback'),
    ('Hyundai i20 car India', 'hatchback'), ('Tata Tiago car', 'hatchback'),
    ('Maruti Suzuki Dzire', 'sedan'), ('Honda City car India', 'sedan'),
    ('Hyundai Verna car', 'sedan'), ('Toyota Corolla India', 'sedan'),
    ('Hindustan Ambassador car', 'sedan'), ('Maruti Ciaz', 'sedan'),
    ('Skoda Octavia India', 'sedan'), ('sedan car Indian street', 'sedan'),
    ('Mahindra Scorpio SUV', 'suv'), ('Toyota Fortuner India', 'suv'),
    ('Toyota Innova India', 'suv'), ('Mahindra Thar jeep', 'suv'),
    ('Tata Safari SUV', 'suv'), ('Hyundai Creta SUV', 'suv'),
    ('Mahindra Bolero SUV', 'suv'), ('Kia Seltos India', 'suv'),
    ('Tata Nexon car', 'suv'), ('SUV India road traffic', 'suv'),
    ('Tata Ace mini truck', 'lcv'), ('Mahindra pickup India', 'lcv'),
    ('Tempo Traveller van India', 'lcv'), ('Force Traveller van', 'lcv'),
    ('Maruti Omni van', 'lcv'), ('loading tempo India', 'lcv'),
    ('pickup truck India road', 'lcv'), ('Tata Intra truck', 'lcv'),
    ('auto rickshaw India street', 'auto_rickshaw'), ('auto rickshaw driver India', 'auto_rickshaw'),
    ('three wheeler India road', 'auto_rickshaw'), ('auto rickshaw parking', 'auto_rickshaw'),
    ('e rickshaw India road', 'e_rickshaw'), ('electric rickshaw India street', 'e_rickshaw'),
    ('tractor trolley India road', 'tractor'), ('tractor India farm', 'tractor'),
    ('bus India highway', 'bus'), ('truck India highway', 'truck'),
    ('traffic police India road', 'scene_traffic'), ('Indian road junction traffic', 'scene_intersection'),
]


def jget(url, params, tries=6, base_sleep=3.0):
    """GET json with 429/5xx backoff."""
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=50)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None
            if r.status_code == 429:
                time.sleep(base_sleep * (i + 2))
                continue
            if r.status_code in (403, 404):
                return None
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def load_done():
    if os.path.exists(DONE):
        return set(open(DONE).read().split('\n'))
    return set()


def write(rec):
    with lock:
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')


# ----------------------------------------------------------------- Commons
def commons_query(q, hint, limit=300):
    """list=search in File namespace, then batched imageinfo."""
    titles, off = [], 0
    while len(titles) < limit:
        j = jget('https://commons.wikimedia.org/w/api.php', {
            'action': 'query', 'list': 'search', 'srsearch': q,
            'srnamespace': 6, 'srlimit': 50, 'sroffset': off, 'format': 'json'})
        if not j:
            break
        hits = (j.get('query') or {}).get('search') or []
        if not hits:
            break
        titles += [h['title'] for h in hits]
        off += 50
        if 'continue' not in j:
            break
        time.sleep(1.0)
    titles = [t for t in titles if re.search(r'\.(jpe?g|png)$', t, re.I)]
    kept = 0
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        j = jget('https://commons.wikimedia.org/w/api.php', {
            'action': 'query', 'titles': '|'.join(chunk), 'prop': 'imageinfo',
            'iiprop': 'url|size|extmetadata', 'iiurlwidth': 1280, 'format': 'json'})
        if not j:
            continue
        for pg in ((j.get('query') or {}).get('pages') or {}).values():
            ii = (pg.get('imageinfo') or [{}])[0]
            if not ii.get('url'):
                continue
            w, h = ii.get('width') or 0, ii.get('height') or 0
            if w < 800 or h < 520:
                continue
            em = ii.get('extmetadata') or {}
            write({'source': 'commons', 'query': q, 'hint': hint,
                   'title': pg['title'].replace('File:', ''),
                   'url': ii['url'], 'w': w, 'h': h,
                   'license': (em.get('LicenseShortName') or {}).get('value', ''),
                   'artist': re.sub('<[^>]+>', '', (em.get('Artist') or {}).get('value', ''))[:180].strip(),
                   'desc': re.sub('<[^>]+>', ' ', (em.get('ImageDescription') or {}).get('value', ''))[:300],
                   'page': 'https://commons.wikimedia.org/wiki/' + pg['title'].replace(' ', '_')})
            kept += 1
        time.sleep(1.2)
    return kept


# --------------------------------------------------------------- Openverse
def openverse_query(q, hint, pages=3):
    kept = 0
    for p in range(1, pages + 1):
        j = jget('https://api.openverse.org/v1/images/', {
            'q': q, 'license': 'cc0,pdm,by,by-sa', 'page_size': 20, 'page': p})
        if not j:
            break
        for r in j.get('results') or []:
            u = r.get('url')
            w, h = r.get('width') or 0, r.get('height') or 0
            if not u or not w or not h or w < 800 or h < 520:
                continue
            write({'source': 'openverse', 'query': q, 'hint': hint,
                   'title': (r.get('title') or '')[:120], 'url': u, 'w': w, 'h': h,
                   'license': (r.get('license') or '').upper(),
                   'artist': (r.get('creator') or '')[:180],
                   'desc': (r.get('tags') and ', '.join(t.get('name', '') for t in r['tags'][:8])) or '',
                   'page': r.get('foreign_landing_url') or u})
            kept += 1
        time.sleep(0.8)
        if not (j.get('page_count') or 0) > p:
            break
    return kept


def main():
    global out
    os.makedirs('state', exist_ok=True)
    done = load_done()
    out = open(OUT, 'a')
    dlog = open(DONE, 'a')
    total = 0
    for src in (sys.argv[1] if len(sys.argv) > 1 else 'both').split(','):
        for q, hint in QUERIES:
            key = f'{src}::{q}'
            if key in done:
                continue
            try:
                n = commons_query(q, hint) if src == 'commons' else openverse_query(q, hint)
            except Exception as e:
                print('ERR', key, e, flush=True)
                continue
            total += n
            print(f'{key:58s} +{n:4d}  (total {total})', flush=True)
            dlog.write(key + '\n'); dlog.flush()
            time.sleep(0.8)
    out.close()
    print('DONE total', total)


if __name__ == '__main__':
    main()
