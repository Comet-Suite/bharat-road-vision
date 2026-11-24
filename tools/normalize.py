"""Build the authoritative candidate manifest with a derived class hint.

Merges every state/candidates*.jsonl, dedupes by URL, derives `hint` from
(query | category path | title) using an explicit keyword->class map, and points
each record at its file on disk (raw/<sha1>.raw).
"""
import glob, hashlib, json, os, re

KEYWORDS = [
    ('e_rickshaw', r'(e-?rickshaw|electric rickshaw|battery rickshaw|e-?cart)'),
    ('auto_rickshaw', r'(auto ?rickshaw|autorickshaw|auto-rickshaw|tuk ?tuk|three[- ]?wheeler|'
                      r'cycle rickshaw|rickshaw)'),
    ('two_wheeler', r'(motorcycle|motorbike|scooter|moped|two[- ]?wheeler|bike|royal enfield|'
                    r'bulleteer|activa|pulsar|splendor)'),
    ('bus', r'(\bbus(es)?\b|coach|minibus|ksrtc|msrtc|best bus|school bus)'),
    ('truck', r'(truck|lorry|tipper|dumper|trailer|tanker|cement mixer|ashok leyland|'
              r'goods carrier|multi[- ]axle)'),
    ('tractor', r'(tractor|harvester|farm equipment)'),
    ('lcv', r'(pickup|pick-up|tempo|traveller|tata ace|intra|van\b|cargo|light commercial|'
            r'mini truck|loading)'),
    ('hatchback', r'(hatchback|maruti|suzuki swift|wagon ?r|alto|i10|i20|indica|kwid|'
                   r'small car|cab\b|taxi)'),
    ('suv', r'(suv|scorpio|fortuner|innova|thar|safari|creta|seltos|bolero|endeavour|'
             r'xuv|utility vehicle|jip|jeep)'),
    ('sedan', r'(sedan|dzire|honda city|corolla|ciaz|verna|accord|ambassador|saloon)'),
    ('scene_toll', r'(toll|boom barrier|plaza)'),
    ('scene_intersection', r'(traffic signal|intersection|junction|crossing)'),
    ('scene_parking', r'(parking|car park)'),
    ('scene_yard', r'(warehouse|factory|industrial|construction|logistics yard)'),
    ('scene_sign', r'(road sign|traffic sign|number plate|registration plate|signage)'),
]
COMPILED = [(k, re.compile(p, re.I)) for k, p in KEYWORDS]


def derive(rec):
    """Class hint from title+desc+query+cats, most specific match first."""
    cats = [str(x) for x in (rec.get('cats') or [])]
    parts = list(reversed(cats)) + [str(rec.get('title', '')), str(rec.get('desc', '')),
                                    str(rec.get('query', ''))]
    blob = ' '.join(parts)
    for cls, rx in COMPILED:
        if rx.search(blob):
            return cls, True
    # roads / traffic catch-all for scene material
    if re.search(r'(road|traffic|street|highway|transport|vehicle)', blob, re.I):
        return 'scene_traffic', True
    return 'scene_traffic', False     # unanchored -> still needs vehicle evidence to pass


def main():
    seen, out = set(), []
    for fn in sorted(glob.glob('state/candidates*.jsonl')):
        for line in open(fn):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            url = (r.get('url') or '').split('?')[0]
            if not url or url in seen:
                continue
            seen.add(url)
            key = hashlib.sha1(url.encode()).hexdigest()
            hint, anchored = derive(r)
            r['hint'] = hint
            r['anchored'] = anchored
            r['sha1'] = key
            r['file'] = os.path.join('raw', key + '.raw')
            r['has_file'] = os.path.exists(r['file'])
            out.append(r)
    with open('state/manifest.jsonl', 'w') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    from collections import Counter
    print('manifest', len(out), 'files on disk', sum(r['has_file'] for r in out))
    print(Counter(r['hint'] for r in out).most_common())
    print('anchored', sum(r['anchored'] for r in out))


if __name__ == '__main__':
    main()
