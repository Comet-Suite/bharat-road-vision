import json
from collections import defaultdict, Counter

recs = [json.loads(l) for l in open('state/plan.jsonl')]
prio = ['e_rickshaw', 'lcv', 'hatchback', 'sedan', 'suv', 'tractor', 'truck',
        'auto_rickshaw', 'two_wheeler', 'bus']
by = defaultdict(list)
for r in recs:
    if r.get('source') == 'commons':
        by[r['hint']].append(r)
plan = []
for k in prio:
    plan += by.get(k, [])
with open('state/plan_commons.jsonl', 'w') as f:
    for r in plan:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('commons-only plan', len(plan), Counter(r['hint'] for r in plan).most_common())

p = 'code/download.py'
s = open(p).read()
s = s.replace("for line in open('state/plan.jsonl')", "for line in open('state/plan_commons.jsonl')")
s = s.replace("with ThreadPoolExecutor(4) as ex:", "with ThreadPoolExecutor(2) as ex:")
open(p, 'w').write(s)
print('downloader -> commons plan')
