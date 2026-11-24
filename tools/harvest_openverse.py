"""Harvest CC-licensed Indian vehicle imagery via the Openverse API.

Polite: 20 results/page, 1.5 s between pages, exponential backoff on 429.
Writes state/candidates_ov.jsonl (resumable via state/done_ov.txt).
"""
import json, os, re, time, sys
import requests

S = requests.Session()
S.headers['User-Agent'] = 'BharatRoadVision/1.0 (dataset construction; contact: arena-agent)'
API = 'https://api.openverse.org/v1/images/'
LIC = 'cc0,pdm,by,by-sa'
OUT = 'state/candidates_ov.jsonl'
DONE = 'state/done_ov.txt'

QUERIES = [
    # two wheelers
    ('motorcycle india', 'two_wheeler'), ('scooter india street', 'two_wheeler'),
    ('royal enfield motorcycle', 'two_wheeler'), ('bike rider india', 'two_wheeler'),
    ('motorcycle traffic india', 'two_wheeler'), ('honda activa scooter', 'two_wheeler'),
    ('motorcyclist india road', 'two_wheeler'), ('motorcycle parking india', 'two_wheeler'),
    # auto rickshaw
    ('auto rickshaw india', 'auto_rickshaw'), ('autorickshaw delhi', 'auto_rickshaw'),
    ('auto rickshaw mumbai', 'auto_rickshaw'), ('tuk tuk india', 'auto_rickshaw'),
    ('three wheeler india', 'auto_rickshaw'), ('auto rickshaw traffic', 'auto_rickshaw'),
    ('auto rickshaw street india', 'auto_rickshaw'), ('auto rickshaw driver', 'auto_rickshaw'),
    # e-rickshaw
    ('e rickshaw india', 'e_rickshaw'), ('electric rickshaw delhi', 'e_rickshaw'),
    ('electric rickshaw street', 'e_rickshaw'), ('rickshaw india road', 'e_rickshaw'),
    # cars
    ('maruti suzuki swift', 'hatchback'), ('maruti 800 car', 'hatchback'),
    ('hatchback car india', 'hatchback'), ('hyundai santro car', 'hatchback'),
    ('tata indica car', 'hatchback'), ('maruti wagon r', 'hatchback'),
    ('renault kwid india', 'hatchback'), ('small car india street', 'hatchback'),
    ('maruti suzuki dzire', 'sedan'), ('honda city car india', 'sedan'),
    ('toyota corolla india', 'sedan'), ('hindustan ambassador car', 'sedan'),
    ('hyundai verna', 'sedan'), ('sedan car india', 'sedan'),
    ('mahindra scorpio', 'suv'), ('toyota fortuner india', 'suv'),
    ('toyota innova', 'suv'), ('mahindra thar', 'suv'),
    ('tata safari suv', 'suv'), ('hyundai creta', 'suv'),
    ('mahindra bolero', 'suv'), ('suv india road', 'suv'),
    # lcv
    ('tata ace mini truck', 'lcv'), ('pickup truck india', 'lcv'),
    ('tempo traveller india', 'lcv'), ('maruti omni van', 'lcv'),
    ('loading tempo india', 'lcv'), ('force traveller van', 'lcv'),
    ('mahindra pickup india', 'lcv'), ('light commercial vehicle india', 'lcv'),
    # bus
    ('bus india road', 'bus'), ('city bus delhi', 'bus'), ('ksrtc bus', 'bus'),
    ('msrtc bus', 'bus'), ('school bus india', 'bus'), ('bus terminal india', 'bus'),
    ('bus india highway', 'bus'), ('best bus mumbai', 'bus'),
    # truck
    ('truck india highway', 'truck'), ('tata truck india', 'truck'),
    ('lorry india road', 'truck'), ('ashok leyland truck', 'truck'),
    ('dumper truck india', 'truck'), ('container truck india', 'truck'),
    ('truck overloaded india', 'truck'), ('tanker truck india', 'truck'),
    # tractor
    ('tractor india field', 'tractor'), ('tractor trolley india', 'tractor'),
    ('mahindra tractor', 'tractor'), ('tractor india road', 'tractor'),
    # scenes
    ('toll plaza india', 'scene_toll'), ('toll booth highway india', 'scene_toll'),
    ('traffic signal india', 'scene_intersection'), ('india road junction traffic', 'scene_intersection'),
    ('traffic jam india', 'scene_traffic'), ('heavy traffic delhi', 'scene_traffic'),
    ('india highway traffic', 'scene_traffic'), ('indian street vehicles', 'scene_traffic'),
    ('parking lot india cars', 'scene_parking'), ('warehouse trucks india', 'scene_yard'),
    ('construction site india vehicles', 'scene_yard'), ('road sign india', 'scene_sign'),
]


def get(url, params, tries=6):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=45)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5 * (i + 1)); continue
            return None
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def main():
    done = set(open(DONE).read().split('\n')) if os.path.exists(DONE) else set()
    out = open(OUT, 'a')
    dlog = open(DONE, 'a')
    total = 0
    for q, hint in QUERIES:
        if q in done:
            continue
        got = 0
        for page in range(1, 6):
            j = get(API, {'q': q, 'license': LIC, 'page_size': 20, 'page': page})
            if not j:
                break
            for r in j.get('results') or []:
                w, h = r.get('width') or 0, r.get('height') or 0
                if not r.get('url') or w < 800 or h < 520:
                    continue
                tags = ', '.join(t.get('name', '') for t in (r.get('tags') or [])[:10])
                out.write(json.dumps({
                    'source': 'openverse', 'query': q, 'hint': hint,
                    'title': (r.get('title') or '')[:140], 'desc': tags,
                    'url': r['url'], 'w': w, 'h': h,
                    'license': (r.get('license') or '').upper() +
                               ('-' + r['license_version'].upper() if r.get('license_version') else ''),
                    'artist': (r.get('creator') or '')[:180],
                    'page': r.get('foreign_landing_url') or r.get('url'),
                    'upstream': r.get('source'),
                }, ensure_ascii=False) + '\n')
                got += 1
            if not (j.get('page_count') or 0) > page:
                break
            time.sleep(1.4)
        out.flush()
        total += got
        print(f'{q:38s} +{got:3d}  (total {total})', flush=True)
        dlog.write(q + '\n'); dlog.flush()
        time.sleep(1.0)
    print('DONE', total)


if __name__ == '__main__':
    main()
