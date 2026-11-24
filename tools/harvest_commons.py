"""Crawl Wikimedia Commons categories for Indian road-transport imagery.

Writes candidate metadata (url, licence, size, category path) to state/candidates_commons.jsonl.
No images downloaded here - cheap discovery pass.
"""
import json, os, re, sys, time, threading
import requests

API = 'https://commons.wikimedia.org/w/api.php'
S = requests.Session()
S.headers['User-Agent'] = 'BharatRoadVision/1.0 (dataset construction; contact comet-suite)'

ROOT_CATS = [
    'Auto rickshaws in India', 'Auto rickshaws', 'E-rickshaws in India', 'E-rickshaws',
    'Motorcycles in India', 'Scooters in India', 'Bicycles in India',
    'Buses in India', 'Trucks in India', 'Cars in India', 'Tractors in India',
    'Traffic in India', 'Toll plazas in India', 'Road transport in India',
    'Vehicles of India', 'Roads in India', 'Streets in India',
    'Vehicle registration plates of India', 'Taxis in India', 'Vans in India',
    'Mahindra vehicles', 'Tata vehicles', 'Maruti Suzuki vehicles',
    'Commercial vehicles in India', 'Traffic police of India', 'Transport in India',
    'Cycle rickshaws in India', 'Handcarts in India', 'Overloaded vehicles',
]
EXT_OK = re.compile(r'\.(jpe?g|png)$', re.I)
BAD_TITLE = re.compile(r'(map|diagram|logo|icon|chart|graph|seal|coat[_ ]of[_ ]arms|poster|stamp|sign\.|'
                       r'clipart|drawing|sketch|painting|1[6-9]\d\d|190\d)', re.I)

lock = threading.Lock()
seen_files, seen_cats, out_fh = set(), set(), None
stats = {'cats': 0, 'files': 0, 'kept': 0}


def api(params, tries=5):
    """GET with backoff; returns parsed json or None."""
    for i in range(tries):
        try:
            r = S.get(API, params=params, timeout=45)
            if r.status_code == 200 and r.text.startswith('{'):
                return r.json()
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def fetch_cat(cat, depth, path):
    """Walk one category: images at this level + recurse into subcategories."""
    with lock:
        if cat in seen_cats or depth < 0:
            return
        seen_cats.add(cat)
        stats['cats'] += 1
    cont, batch = {}, 0
    while True:
        p = {'action': 'query', 'generator': 'categorymembers', 'gcmtitle': 'Category:' + cat,
             'gcmtype': 'file|subcat', 'gcmlimit': 500,
             'prop': 'imageinfo', 'iiprop': 'url|size|extmetadata|mime', 'iiurlwidth': 1024,
             'format': 'json'}
        p.update(cont)
        j = api(p)
        if not j:
            return
        pages = (j.get('query') or {}).get('pages') or {}
        for pg in pages.values():
            title = pg.get('title', '')
            if pg.get('ns') == 14:                      # subcategory
                sub = title.split(':', 1)[-1]
                if depth > 0 and not re.search(r'(by year|by decade|\b19\d\d\b|\b18\d\d\b)', sub, re.I):
                    fetch_cat(sub, depth - 1, path + [sub])
                continue
            if pg.get('ns') != 6 or not EXT_OK.search(title):
                continue
            if BAD_TITLE.search(title):
                continue
            with lock:
                if title in seen_files:
                    continue
                seen_files.add(title)
                stats['files'] += 1
            ii = (pg.get('imageinfo') or [{}])[0]
            w, h = ii.get('width') or 0, ii.get('height') or 0
            if w < 800 or h < 500:
                continue
            em = ii.get('extmetadata') or {}
            lic = (em.get('LicenseShortName') or {}).get('value', '')
            art = re.sub('<[^>]+>', '', (em.get('Artist') or {}).get('value', ''))[:200]
            rec = {
                'title': title.replace('File:', ''),
                'url': ii.get('thumburl') or ii.get('url'),
                'orig_url': ii.get('url'),
                'w': w, 'h': h,
                'license': lic,
                'artist': art.strip(),
                'desc': re.sub('<[^>]+>', ' ', (em.get('ImageDescription') or {}).get('value', ''))[:300],
                'cats': path,
                'source': 'commons',
                'page': 'https://commons.wikimedia.org/wiki/' + title.replace(' ', '_'),
            }
            with lock:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                stats['kept'] += 1
        batch += 1
        if 'continue' in j and batch < 8:
            cont = {k: v for k, v in j['continue'].items()}
        else:
            return


def main():
    global out_fh
    os.makedirs('state', exist_ok=True)
    out_fh = open('state/candidates_commons.jsonl', 'w')
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for c in ROOT_CATS:
        try:
            fetch_cat(c, depth, [c])
        except Exception as e:
            print('ERR', c, e, flush=True)
        print(f"after {c:36s} cats={stats['cats']:5d} files={stats['files']:6d} kept={stats['kept']:6d}", flush=True)
    out_fh.close()
    print('DONE', stats)


if __name__ == '__main__':
    main()
