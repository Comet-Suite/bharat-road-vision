"""Package Bharat Road Vision into the publishable repo.

Balanced-core strategy: the training/val/test manifests carry a class-balanced
core (min-cap per class) so a 5M-parameter model is not destroyed by the
bus-heavy tail of what open sources happened to offer; the full annotated set
ships as `data/full_annotated.jsonl` with confidence tiers and class weights.
"""
import glob, hashlib, json, math, os, random, shutil, tarfile
from collections import Counter, defaultdict

import cv2
import numpy as np

SIZE = 512
SHARD_LIMIT = 48 * 1024 * 1024
REPO = 'repo'
CORE_CAP = 180          # balanced core size per vehicle class

LABEL_DEFS = {
    'two_wheeler': 'Motorcycle, scooter or moped — single-track vehicle, rider usually visible.',
    'auto_rickshaw': 'Three-wheeled petrol/CNG auto rickshaw (tuk-tuk), yellow-black or green-yellow livery.',
    'e_rickshaw': 'Battery-electric three-wheeler (e-rickshaw / e-cart), often green or blue canopy.',
    'hatchback': 'Small passenger car with rear hatch and short rear overhang (Swift/WagonR/i10 class).',
    'sedan': 'Four-door passenger car with a distinct boot (Dzire/City/Corolla class).',
    'suv': 'Tall utility passenger vehicle with high clearance (Scorpio/Fortuner/Innova/Creta class).',
    'lcv': 'Light commercial vehicle under ~3.5 t: mini-truck, pickup, tempo traveller or van (Tata Ace class).',
    'bus': 'Passenger bus or coach — city, school, sleeper or minibus.',
    'truck': 'Goods truck, lorry, dumper or multi-axle trailer — cargo body or tipper.',
    'tractor': 'Agricultural tractor, with or without trolley; restricted on national highways.',
    'scene_toll': 'Toll plaza / boom-barrier checkpoint with vehicles queued or passing.',
    'scene_intersection': 'Signalised or unsignalised intersection with mixed traffic.',
    'scene_traffic': 'Road or highway traffic scene showing mixed Indian vehicle flow.',
    'scene_yard': 'Industrial, warehouse, factory or construction yard entry with commercial vehicles.',
    'scene_parking': 'Parking area or lot with parked vehicles.',
    'scene_sign': 'Road sign, marking or regulatory signage relevant to enforcement.',
}
USECASE_TEXT = {
    'toll_classification': 'smart toll plaza classification',
    'parking_pricing': 'dynamic parking pricing',
    'intersection_counting': 'intersection vehicle counting',
    'signal_priority': 'green-light prioritisation',
    'expressway_rule_enforcement': 'expressway rule enforcement',
    'restricted_zone_enforcement': 'restricted-zone enforcement',
    'city_zone_enforcement': 'city-zone entry enforcement',
    'fleet_gate_logging': 'fleet gate logging',
    'rural_road_monitoring': 'rural road monitoring',
    'insurance_verification': 'on-device insurance verification',
    'congestion_estimation': 'congestion estimation',
    'toll_plaza_deployment': 'toll plaza deployment',
    'signal_optimisation': 'signal optimisation',
    'parking_occupancy': 'parking occupancy analysis',
    'rule_enforcement': 'rule enforcement',
}
CAPTIONS = [
    'Classify the Indian road vehicle body type for {use}.',
    'Which vehicle class is this at an Indian {place}? Used for {use}.',
    'For {use} in India, identify the vehicle shown here.',
    'Label this vehicle for a low-power roadside camera handling {use}.',
    'Identify the body type of this Indian vehicle for {use}.',
]
PLACES = ['toll plaza', 'intersection', 'warehouse gate', 'highway checkpost', 'city street',
          'parking lot', 'construction site', 'state border']

VEHICLE_ORDER = ['two_wheeler', 'auto_rickshaw', 'e_rickshaw', 'hatchback', 'sedan', 'suv',
                 'lcv', 'bus', 'truck', 'tractor']
SCENE_ORDER = ['scene_toll', 'scene_intersection', 'scene_traffic', 'scene_yard',
               'scene_parking', 'scene_sign']


def caption(r, rng):
    uc = (r.get('use_cases') or ['intersection_counting'])[0]
    head = rng.choice(CAPTIONS).format(use=USECASE_TEXT.get(uc, uc), place=rng.choice(PLACES))
    return f'{head} Answer: {r["label"]}.'


def split_assign(items, seed=42):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_val = max(1, round(n * 0.12)) if n > 6 else 1
    n_test = max(1, round(n * 0.12)) if n > 6 else 1
    for i, r in enumerate(items):
        r['split'] = 'val' if i < n_val else ('test' if i < n_val + n_test else 'train')
    return items


def shard(files, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    part, buf, size, made = 1, [], 0, []
    def flush(part, buf):
        p = os.path.join(outdir, f'{prefix}-part{part}.tar')
        with tarfile.open(p, 'w') as tf:
            for f in buf:
                tf.add(f, arcname=os.path.relpath(f, REPO))
        made.append((os.path.basename(p), len(buf), os.path.getsize(p)))
    for f in sorted(files):
        sz = os.path.getsize(f)
        if size + sz > SHARD_LIMIT and buf:
            flush(part, buf); part += 1; buf, size = [], 0
        buf.append(f); size += sz
    if buf:
        flush(part, buf)
    return made


def grid(recs, path, cols=8, rows=6, cell=200, title='Bharat Road Vision - Indian vehicle body types'):
    by = defaultdict(list)
    for r in recs:
        by[r['label']].append(r)
    rng = random.Random(3)
    order = sorted(by, key=lambda k: -(len(by[k])))
    pick = []
    while len(pick) < cols * rows and any(by.values()):
        for l in order:
            if by[l] and len(pick) < cols * rows:
                pick.append(by[l].pop(rng.randrange(len(by[l]))))
    W, H = cols * cell, rows * (cell + 24) + 48
    canvas = np.full((H, W, 3), 18, np.uint8)
    for i, r in enumerate(pick):
        im = cv2.imread(os.path.join('build', r['image']))
        if im is None:
            continue
        im = cv2.resize(im, (cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 24) + 44
        canvas[y:y + cell, x:x + cell] = im
        cv2.putText(canvas, r['label'], (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (238, 238, 238), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'{len(recs)} images | 512x512 | auto-annotated, licence-cleared | '
                        f'A/B/C confidence tiers per record',
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (165, 165, 165), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    recs = [json.loads(l) for l in open('state/annotated.jsonl')]
    recs = [r for r in recs if os.path.exists(os.path.join('build', r['image']))]
    rng = random.Random(11)
    for r in recs:
        r['caption'] = caption(r, rng)
        r['prompt'] = r['caption']
        r['target'] = r['label']

    # ---- extra quality gates (uses stored features, no re-annotation)
    veh = [r for r in recs if r['label_group'] == 'vehicle'
           and r['vehicle_evidence'] >= 0.75
           and 0.05 <= r['features']['roi_area'] <= 0.80
           and r['features']['edge_density'] >= 0.045
           and r['features']['flatness'] <= 0.78]
    scen = [r for r in recs if r['label_group'] == 'scene'
            and r['vehicle_evidence'] >= 0.80
            and r['features']['edge_density'] >= 0.05]
    dropped = len(recs) - len(veh) - len(scen)

    # ---- balanced core
    by = defaultdict(list)
    for r in veh:
        by[r['label']].append(r)
    core = []
    for k in VEHICLE_ORDER:
        items = sorted(by.get(k, []), key=lambda z: (-z['confidence'], z['id']))
        core += items[:CORE_CAP]
    for r in core:
        r['in_balanced_core'] = True
    for r in veh:
        r.setdefault('in_balanced_core', False)
    # class weights so the full set can still be trained balanced
    total = len(veh)
    for r in veh:
        r['class_weight'] = round(total / (len(VEHICLE_ORDER) * max(1, len(by[r['label']]))), 3)

    core = split_assign(core)
    scen = split_assign(scen, seed=7)
    for r in veh:
        if not r['in_balanced_core']:
            r['split'] = 'extra'
    allrecs = veh + scen

    os.makedirs(f'{REPO}/data', exist_ok=True)
    def dump(name, rows):
        with open(f'{REPO}/data/{name}', 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    dump('bharat_road_vision.jsonl', allrecs)
    dump('full_annotated.jsonl', allrecs)
    dump('train.jsonl', [r for r in core if r['split'] == 'train'])
    dump('val.jsonl', [r for r in core if r['split'] == 'val'])
    dump('test.jsonl', [r for r in core if r['split'] == 'test'])
    dump('scenes.jsonl', scen)

    lmap = {'image_size': SIZE, 'total_records': len(allrecs),
            'labels': [{'id': i, 'name': k, 'group': 'vehicle' if k in VEHICLE_ORDER else 'scene',
                        'definition': LABEL_DEFS[k],
                        'use_cases': [USECASE_TEXT.get(u, u) for u in
                                      next((r['use_cases'] for r in recs if r['label'] == k), [])]}
                       for i, k in enumerate(VEHICLE_ORDER + SCENE_ORDER)],
            'fields': {
                'id': 'stable content hash of the 512px crop',
                'image': 'relative path under images/',
                'label': 'target class', 'caption': 'instruction-style sentence ending in the answer',
                'confidence': 'annotation confidence 0-1',
                'confidence_tier': 'A (>=0.55), B (0.40-0.55), C (<0.40)',
                'shape_agreement': 'True when geometric rules corroborate the label',
                'rank': 'top-4 rule scores at annotation time',
                'bbox_norm': 'object box in ORIGINAL frame normalised [x0,y0,x1,y1]',
                'features': 'aspect, edge density, flatness, colour shares, wheel/plate/people counts',
                'in_balanced_core': 'member of the class-balanced training core',
                'class_weight': 'inverse-frequency weight for balanced training on the full set',
                'split': 'train | val | test | extra | scene',
                'license': 'upstream licence', 'attribution': 'upstream author',
                'source_page': 'human-verifiable page', 'original_url': 'direct image URL'}}
    json.dump(lmap, open(f'{REPO}/data/label_map.json', 'w'), indent=2)

    counts = Counter(r['label'] for r in veh)
    core_counts = Counter(r['label'] for r in core)
    stats = {
        'name': 'Bharat Road Vision',
        'generated': '2026-11-05',
        'total_records': len(allrecs),
        'vehicle_images': len(veh), 'scene_images': len(scen),
        'balanced_core_images': len(core),
        'per_label_full': dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        'per_label_balanced_core': dict(sorted(core_counts.items(), key=lambda kv: -kv[1])),
        'scenes': dict(Counter(r['label'] for r in scen)),
        'splits': dict(Counter(r['split'] for r in allrecs)),
        'confidence_tiers': dict(Counter(r['confidence_tier'] for r in allrecs)),
        'shape_agreement_rate': round(sum(r['shape_agreement'] for r in veh) / max(1, len(veh)), 3),
        'with_number_plate_visible': sum(1 for r in veh if r.get('has_plate')),
        'low_light_images': sum(1 for r in veh if r.get('night')),
        'sources': dict(Counter(r['source'] for r in allrecs)),
        'licenses': dict(Counter(r['license'] for r in allrecs).most_common(15)),
        'annotation': 'code/annotate.py — saliency ROI proposals + hand-written geometric/colour '
                      'rules + curated query/category prior. No third-party dataset was downloaded '
                      'or copied; every label here was produced by this pipeline.',
    }
    json.dump(stats, open(f'{REPO}/data/stats.json', 'w'), indent=2)

    # ---- images + shards
    imgdir = f'{REPO}/images'
    if os.path.exists(imgdir):
        shutil.rmtree(imgdir)
    os.makedirs(imgdir)
    files = []
    for r in allrecs:
        src = os.path.join('build', r['image'])
        dst = os.path.join(imgdir, r['image'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        files.append(dst)
    made = shard(files, imgdir, 'bharat')
    for name, n, size in made:
        print(f'  {name} {n} files {size/1e6:.1f} MB')

    # ---- attribution
    rows = defaultdict(int)
    for r in allrecs:
        rows[(r['license'] or 'unknown', r['source'])] += 1
    with open(f'{REPO}/ATTRIBUTION.md', 'w') as f:
        f.write('# Attribution\n\nEvery image keeps its upstream licence. Training or redistributing '
                'this dataset means honouring each one.\n\n| Licence | Records | Source |\n|---|---|---|\n')
        for (lic, src), n in sorted(rows.items(), key=lambda kv: -kv[1]):
            f.write(f'| {lic} | {n} | {src} |\n')
        f.write('\n## Per-record credits\n\nEach JSONL record carries `license`, `attribution`, '
                '`source_page` and `original_url`:\n\n```python\n'
                'import json\nfor r in map(json.loads, open("data/bharat_road_vision.jsonl")):\n'
                '    print(r["attribution"], "|", r["license"], "|", r["source_page"])\n```\n\n'
                'CC BY / CC BY-SA images require credit to the named author on redistribution; '
                'CC0 / public-domain images do not.\n')
    grid(allrecs, f'{REPO}/docs/preview_grid.png')
    print('dropped by quality gate', dropped)
    print('total', len(allrecs), 'core', len(core))
    print(Counter(r['label'] for r in allrecs).most_common())


if __name__ == '__main__':
    main()
