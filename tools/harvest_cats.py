"""Category-driven Commons harvest (batched, polite) -> state/candidates2.jsonl.

Seeds are chosen so every image is topically anchored to an Indian vehicle class.
"""
import json, os, re, time, sys
import requests
from collections import deque

API = 'https://commons.wikimedia.org/w/api.php'
S = requests.Session()
S.headers['User-Agent'] = 'BharatRoadVision/1.0 (non-commercial dataset research; contact: arena-agent)'
EXT_OK = re.compile(r'\.(jpe?g|png)$', re.I)
SKIP_CAT = re.compile(r'(by year|by decade|\b1[6-9]\d\d\b|\b20[0-2]\d\b|stamps|coins|maps|diagrams|logos)', re.I)

SEEDS = [
    ('Auto rickshaws in India', 'auto_rickshaw'),
    ('Auto rickshaws by city in India', 'auto_rickshaw'),
    ('Motorcycles in India', 'two_wheeler'),
    ('Scooters in India', 'two_wheeler'),
    ('Motorcycle racing in India', 'two_wheeler'),
    ('Buses in India', 'bus'),
    ('Buses of India', 'bus'),
    ('Trucks in India', 'truck'),
    ('Tata trucks', 'truck'),
    ('Ashok Leyland', 'truck'),
    ('Tractors in India', 'tractor'),
    ('Mahindra tractors', 'tractor'),
    ('Cars in India', 'hatchback'),
    ('Maruti Suzuki', 'hatchback'),
    ('Taxis in India', 'sedan'),
    ('Vans in India', 'lcv'),
    ('Tata Ace', 'lcv'),
    ('Pickup trucks in India', 'lcv'),
    ('Road transport in India', 'scene_traffic'),
    ('Roads in India', 'scene_traffic'),
    ('Traffic in India', 'scene_traffic'),
    ('Traffic signals in India', 'scene_intersection'),
    ('Toll plazas in India', 'scene_toll'),
    ('National highways in India', 'scene_traffic'),
    ('Vehicle registration plates of India', 'scene_sign'),
    ('E-rickshaws', 'e_rickshaw'),
    ('Electric vehicles in India', 'e_rickshaw'),
    ('Cycle rickshaws in India', 'auto_rickshaw'),
    ('Streets in Mumbai', 'scene_traffic'),
    ('Streets in Delhi', 'scene_traffic'),
    ('Streets in Chennai', 'scene_traffic'),
    ('Streets in Kolkata', 'scene_traffic'),
    ('Transport in Bangalore', 'scene_traffic'),
    ('Parking in India', 'scene_parking'),
    ('Construction vehicles in India', 'scene_yard'),
    ('Vehicles of the Indian Army', 'truck'),
    ('Buses in Mumbai', 'bus'),
    ('Buses in Delhi', 'bus'),
    ('Trucks in Delhi', 'truck'),
]

SEEDS += [
    ('Auto rickshaws in Delhi', 'auto_rickshaw'), ('Auto rickshaws in Mumbai', 'auto_rickshaw'),
    ('Auto rickshaws in Bangalore', 'auto_rickshaw'), ('Auto rickshaws in Chennai', 'auto_rickshaw'),
    ('Auto rickshaws in Kolkata', 'auto_rickshaw'), ('Auto rickshaws in Hyderabad', 'auto_rickshaw'),
    ('Auto rickshaws in Pune', 'auto_rickshaw'), ('Auto rickshaws in Jaipur', 'auto_rickshaw'),
    ('E-rickshaws in Delhi', 'e_rickshaw'), ('Battery electric vehicles in India', 'e_rickshaw'),
    ('Electric rickshaws', 'e_rickshaw'),
    ('Maruti Suzuki vehicles', 'hatchback'), ('Hyundai vehicles in India', 'hatchback'),
    ('Maruti Suzuki Swift', 'hatchback'), ('Maruti Suzuki Wagon R', 'hatchback'),
    ('Tata Motors vehicles', 'hatchback'), ('Honda cars in India', 'sedan'),
    ('Toyota vehicles in India', 'suv'), ('Mahindra vehicles', 'suv'),
    ('Mahindra Scorpio', 'suv'), ('Toyota Innova', 'suv'), ('Tata Safari', 'suv'),
    ('Pickup trucks in India', 'lcv'), ('Vans of India', 'lcv'), ('Tata Ace', 'lcv'),
    ('Tempo travellers', 'lcv'), ('Force Motors vehicles', 'lcv'),
    ('Tractors in India by manufacturer', 'tractor'), ('Mahindra tractors', 'tractor'),
    ('Swaraj tractors', 'tractor'), ('John Deere tractors in India', 'tractor'),
    ('Truck transport in India', 'truck'), ('Tippers in India', 'truck'),
    ('Ashok Leyland vehicles', 'truck'), ('Tata trucks', 'truck'),
    ('Buses in Bangalore', 'bus'), ('Buses in Chennai', 'bus'), ('Buses in Kolkata', 'bus'),
    ('Buses in Hyderabad', 'bus'), ('Buses in Pune', 'bus'), ('Buses in Ahmedabad', 'bus'),
    ('Traffic in Delhi', 'scene_traffic'), ('Traffic in Mumbai', 'scene_traffic'),
    ('Traffic in Bangalore', 'scene_traffic'), ('Roads in Kerala', 'scene_traffic'),
    ('National highways of India', 'scene_traffic'), ('Expressways in India', 'scene_traffic'),
    ('Toll booths in India', 'scene_toll'), ('Parking lots in India', 'scene_parking'),
    ('Traffic lights in India', 'scene_intersection'),
    ('Road signs in India', 'scene_sign'), ('Vehicle registration plates of India', 'scene_sign'),
    ('Logistics in India', 'scene_yard'), ('Warehouses in India', 'scene_yard'),
    ('Construction equipment in India', 'scene_yard'),
]

files, cats_done = {}, set()
q = deque([(c, h, 0, [c]) for c, h in SEEDS])
seen = set()


def api(params, tries=5):
    for i in range(tries):
        try:
            r = S.get(API, params=params, timeout=50)
            if r.status_code == 200 and r.text.startswith('{'):
                return r.json()
            if r.status_code == 429:
                time.sleep(4 * (i + 1)); continue
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None


def imageinfo(titles, hint, catpath):
    for i in range(0, len(titles), 45):
        chunk = titles[i:i + 45]
        j = api({'action': 'query', 'titles': '|'.join(chunk), 'prop': 'imageinfo',
                 'iiprop': 'url|size|extmetadata', 'iiurlwidth': 1280, 'format': 'json'})
        if not j:
            continue
        for pg in ((j.get('query') or {}).get('pages') or {}).values():
            ii = (pg.get('imageinfo') or [{}])[0]
            t = pg.get('title', '')
            if not ii.get('url') or t in files:
                continue
            w, h = ii.get('width') or 0, ii.get('height') or 0
            if w < 800 or h < 520:
                continue
            em = ii.get('extmetadata') or {}
            files[t] = {'source': 'commons', 'query': 'category:' + catpath[0], 'hint': hint,
                        'title': t.replace('File:', ''), 'url': ii['url'], 'w': w, 'h': h,
                        'license': (em.get('LicenseShortName') or {}).get('value', ''),
                        'artist': re.sub('<[^>]+>', '', (em.get('Artist') or {}).get('value', ''))[:180].strip(),
                        'desc': re.sub('<[^>]+>', ' ', (em.get('ImageDescription') or {}).get('value', ''))[:300],
                        'page': 'https://commons.wikimedia.org/wiki/' + t.replace(' ', '_'),
                        'cats': catpath[-3:]}
        time.sleep(1.1)


def main():
    maxdepth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = open('state/candidates2.jsonl', 'w')
    while q:
        cat, hint, depth, path = q.popleft()
        if cat in cats_done or depth > maxdepth:
            continue
        cats_done.add(cat)
        cont = {}; batch = 0
        while True:
            p = {'action': 'query', 'generator': 'categorymembers', 'gcmtitle': 'Category:' + cat,
                 'gcmtype': 'file|subcat', 'gcmlimit': 500, 'format': 'json'}
            p.update(cont)
            j = api(p)
            if not j:
                break
            pages = (j.get('query') or {}).get('pages') or {}
            batch_titles = []
            for pg in pages.values():
                t = pg.get('title', '')
                if pg.get('ns') == 14:
                    sub = t.split(':', 1)[-1]
                    if not SKIP_CAT.search(sub):
                        q.append((sub, hint, depth + 1, path + [sub]))
                elif pg.get('ns') == 6 and EXT_OK.search(t):
                    batch_titles.append(t)
            if batch_titles:
                imageinfo(batch_titles, hint, path)
            batch += 1
            if 'continue' in j and batch < 6:
                cont = dict(j['continue'])
            else:
                break
            time.sleep(0.9)
        print(f'{cat[:46]:46s} depth={depth} files={len(files)} queue={len(q)}', flush=True)
        for t, r in list(files.items()):
            pass
        with open('state/candidates2.jsonl', 'w') as f:
            for r in files.values():
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        time.sleep(0.7)
    print('DONE files', len(files))


if __name__ == '__main__':
    main()
