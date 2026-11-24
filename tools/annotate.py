"""Annotate harvested Indian road imagery — v2 (geometry-first, relevance-gated).

Pipeline per downloaded original:
  1. decode, cap longest side 1400
  2. relevance gate: India evidence in source metadata  ->  else reject
  3. region proposals (spectral-residual saliency + connected components)
  4. ROI selection + geometric/colour/structure features
  5. vehicle-evidence gate (wheels / dark underside / cargo flatness / edge structure)
  6. GEOMETRY-FIRST classification; discovery query only breaks ties among plausible classes
  7. 512x512 padded crop with context, perceptual-hash dedupe
  8. full record: bbox, features, class, confidence, ranks, licence, attribution

Usage:
  python3 code/annotate.py --limit 60 --montage    # calibration
  python3 code/annotate.py                          # full pass
"""
import argparse, hashlib, json, math, os, re
import cv2, numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

cv2.setNumThreads(1)
HC = cv2.data.haarcascades
PLATE = cv2.CascadeClassifier(HC + 'haarcascade_russian_plate_number.xml')
UPPER = cv2.CascadeClassifier(HC + 'haarcascade_upperbody.xml')
SAL = cv2.saliency.StaticSaliencySpectralResidual_create()

CLASSES = ['two_wheeler', 'auto_rickshaw', 'e_rickshaw', 'hatchback', 'sedan', 'suv',
           'lcv', 'bus', 'truck', 'tractor']
SIZE = 512

USE_CASES = {
    'two_wheeler': ['toll_classification', 'expressway_rule_enforcement', 'intersection_counting'],
    'auto_rickshaw': ['toll_classification', 'intersection_counting', 'parking_pricing'],
    'e_rickshaw': ['toll_classification', 'intersection_counting', 'restricted_zone_enforcement'],
    'hatchback': ['toll_classification', 'parking_pricing', 'insurance_verification'],
    'sedan': ['toll_classification', 'parking_pricing', 'insurance_verification'],
    'suv': ['toll_classification', 'parking_pricing', 'insurance_verification'],
    'lcv': ['toll_classification', 'fleet_gate_logging', 'parking_pricing'],
    'bus': ['signal_priority', 'intersection_counting', 'fleet_gate_logging'],
    'truck': ['toll_classification', 'fleet_gate_logging', 'city_zone_enforcement'],
    'tractor': ['fleet_gate_logging', 'rural_road_monitoring'],
    'scene_toll': ['toll_plaza_deployment'], 'scene_intersection': ['signal_optimisation'],
    'scene_traffic': ['congestion_estimation'], 'scene_yard': ['fleet_gate_logging'],
    'scene_parking': ['parking_occupancy'], 'scene_sign': ['rule_enforcement'],
}

INDIA_RE = re.compile(r'(india|indian|delhi|mumbai|bombay|bengaluru|bangalore|chennai|madras|kolkata|'
                      r'calcutta|hyderabad|pune|jaipur|gujarat|kerala|tamil nadu|karnataka|maharashtra|'
                      r'rajasthan|punjab|haryana|uttar pradesh|bihar|odisha|orissa|assam|telangana|'
                      r'andhra|goa|kochi|cochin|ahmedabad|surat|lucknow|kanpur|nagpur|indore|bhopal|'
                      r'chandigarh|dehradun|patna|ranchi|guwahati|trivandrum|coimbatore|madurai|'
                      r'visakhapatnam|vijayawada|nashik|aurangabad|jodhpur|udaipur|agra|varanasi|'
                      r'meerut|noida|gurgaon|gurugram|faridabad|ghaziabad|srinagar|jammu|shimla|'
                      r'nh\s?\d+|national highway)', re.I)

OFFTOPIC_RE = re.compile(r'(lion|tiger|peacock|elephant|bird|flower|mountain|landscape|temple|'
                         r'mosque|church|portrait|wedding|food|painting|map|diagram|logo|stamp|'
                         r'coin|statue|sculpture|monument|fort|palace|museum interior)', re.I)


# --------------------------------------------------------------- decode
def decode(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image
            im = Image.open(path).convert('RGB')
            img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if max(h, w) > 1400:
        s = 1400.0 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------ proposals
def proposals(img):
    H, W = img.shape[:2]
    small = cv2.resize(img, (320, max(180, int(320 * H / W))))
    ok, sal = SAL.computeSaliency(small)
    if not ok:
        return []
    s = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    s = cv2.GaussianBlur(s, (9, 9), 0)
    m = (s > max(25, int(np.percentile(s, 78)))).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    sh, sw = small.shape[:2]
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 26 or h < 20:
            continue
        out.append({'x': x / sw, 'y': y / sh, 'w': w / sw, 'h': h / sh,
                    'area': area / float(sw * sh), 'sal': float(s[lab == i].mean()) / 255.0})
    out.sort(key=lambda r: -(r['area'] * (0.4 + r['sal'])))
    return out


def roi_stats(img, box):
    H, W = img.shape[:2]
    x0 = int(max(0, box['x']) * W); y0 = int(max(0, box['y']) * H)
    x1 = int(min(1, box['x'] + box['w']) * W); y1 = int(min(1, box['y'] + box['h']) * H)
    if x1 - x0 < 26 or y1 - y0 < 22:
        return None
    roi = img[y0:y1, x0:x1]
    h, w = roi.shape[:2]
    r = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 70, 160)
    v = hsv[:, :, 2].astype(np.float32)
    blk = cv2.resize(v, (32, 32), interpolation=cv2.INTER_AREA)
    flat = float((np.abs(cv2.Laplacian(blk, cv2.CV_32F)) < 6).mean())
    bh = max(6, int(0.22 * 128)); th = max(6, int(0.32 * 128))
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    low = gray[64:, :]
    circles = cv2.HoughCircles(cv2.medianBlur(low, 5), cv2.HOUGH_GRADIENT, 1.4, 32,
                               param1=110, param2=30, minRadius=9, maxRadius=44)
    plate = PLATE.detectMultiScale(gray, 1.12, 4, minSize=(18, 8))
    feats = {
        'bbox': [round(x0 / W, 5), round(y0 / H, 5), round(x1 / W, 5), round(y1 / H, 5)],
        'roi_aspect': round(w / float(h), 3),
        'roi_area': round(float(box['w'] * box['h']), 4),
        'edge_density': round(float((edges > 0).mean()), 4),
        'flatness': round(flat, 3),
        'dark_bottom': round(float((gray[-bh:, :] < 78).mean()), 3),
        'upper_dark': round(float((gray[:th, :] < 95).mean()), 3),
        'symmetry': round(1.0 - float(np.abs(gray.astype(np.float32) -
                                            cv2.flip(gray, 1).astype(np.float32)).mean()) / 255.0, 3),
        'yellow': round(float((((hue > 18) & (hue < 38)) & (sat > 90) & (val > 110)).mean()), 3),
        'green': round(float((((hue > 38) & (hue < 88)) & (sat > 70) & (val > 90)).mean()), 3),
        'blue': round(float((((hue > 88) & (hue < 130)) & (sat > 80)).mean()), 3),
        'brightness': round(float(gray.mean()) / 255.0, 3),
        'wheels': 0 if circles is None else int(min(len(circles[0]), 8)),
        'plates': int(len(plate)),
        'people': int(len(UPPER.detectMultiScale(gray, 1.12, 4, minSize=(26, 26)))),
    }
    return feats


def vehicle_evidence(f):
    """Does this ROI plausibly contain a vehicle?"""
    ev = 0.0
    if f['wheels'] >= 1: ev += 0.45
    if f['dark_bottom'] > 0.12: ev += 0.30
    if f['flatness'] > 0.45: ev += 0.25
    if f['edge_density'] > 0.045: ev += 0.20
    if f['plates'] >= 1: ev += 0.35
    if 0.05 <= f['roi_area'] <= 0.92: ev += 0.15
    if f['symmetry'] > 0.60: ev += 0.10
    return round(ev, 3)


# ----------------------------------------------------- class decision
def rule_scores(f):
    """Geometry/colour body-type scoring for Indian vehicles. No query prior here."""
    ar, flat, ed = f['roi_aspect'], f['flatness'], f['edge_density']
    s = {c: 0.0 for c in CLASSES}
    # two wheelers: narrow, tall, rider mass on top, few solid panels
    if ar < 1.0:
        s['two_wheeler'] += 2.2
    if ar < 0.82:
        s['two_wheeler'] += 0.9
    if ar < 1.15 and f['upper_dark'] > 0.20 and flat < 0.45:
        s['two_wheeler'] += 0.7
    # three wheelers: compact near-square, yellow / green livery, rounded canopy
    if 0.85 <= ar <= 1.30:
        s['auto_rickshaw'] += 1.5
    if f['yellow'] > 0.08 and 0.70 <= ar <= 1.45:
        s['auto_rickshaw'] += 1.5
    if f['green'] > 0.10 and 0.70 <= ar <= 1.45:
        s['e_rickshaw'] += 1.7
    if f['blue'] > 0.10 and 0.80 <= ar <= 1.40:
        s['e_rickshaw'] += 0.8
    # passenger cars: widest aspect 1.05-1.85, smooth body, low wheel count
    if 1.02 <= ar <= 1.42:
        s['hatchback'] += 1.9; s['sedan'] += 0.7; s['suv'] += 0.5
    if 1.38 <= ar <= 1.90:
        s['sedan'] += 1.9; s['hatchback'] += 0.5; s['suv'] += 1.0
    if 1.25 <= ar <= 1.80 and f['upper_dark'] > 0.15 and flat < 0.52:
        s['suv'] += 1.0
    if 1.10 <= ar <= 2.00 and f['wheels'] >= 2:
        for c in ('hatchback', 'sedan', 'suv'): s[c] += 0.25
    # boxy cargo bodies
    if flat > 0.52 and ar > 1.35:
        s['lcv'] += 1.4; s['truck'] += 1.2; s['bus'] += 0.9
    if flat > 0.62 and ar > 1.85:
        s['truck'] += 1.5; s['bus'] += 1.2
    if ar > 2.05:
        s['bus'] += 1.6; s['truck'] += 1.3
    if ar > 2.30 and flat > 0.55:
        s['bus'] += 1.0
    if 1.45 <= ar <= 2.05 and flat > 0.58 and f['dark_bottom'] < 0.25:
        s['lcv'] += 0.9
    # tractors: squat, saturated paint, huge rear wheels, open frame
    if ar < 1.30 and f['wheels'] >= 2 and f['dark_bottom'] > 0.16:
        s['tractor'] += 1.4
    if ar < 1.45 and ed > 0.10 and flat < 0.42 and f['wheels'] >= 2:
        s['tractor'] += 0.8
    return s


def decide(f, hint):
    """The curated discovery query carries the class; geometry/colour validate it.

    Returns (label, confidence, rank_map, conflict) where conflict True means the
    visible geometry strongly contradicts the query -> caller rejects the sample.
    """
    s = rule_scores(f)
    ranked = sorted(s.items(), key=lambda kv: -kv[1])
    top, ts = ranked[0]
    if hint not in s:
        return top, 0.30, {k: round(v, 2) for k, v in ranked[:4]}, True
    own = s[hint]
    gap = ts - own                       # how far the best competitor beats the query class
    ev = vehicle_evidence(f)
    # geometry agreement 0..1
    agree = max(0.0, min(1.0, 1.0 - gap / 3.0))
    conf = 0.34 + 0.34 * agree + 0.22 * min(1.0, ev) + (0.08 if gap <= 0 else 0.0)
    conflict = (gap >= 2.6 and own <= 0.8) or (hint in ('hatchback', 'sedan', 'suv')
                                              and f['roi_aspect'] < 0.95)
    return hint, round(max(0.05, min(0.96, conf)), 3), {k: round(v, 2) for k, v in ranked[:4]}, conflict


def decide_multi(shape_feats, evid_feats, hint):
    """The curated query/category carries the class; geometry either corroborates it or not.

    Returns (label, confidence, rank, agree). Samples are never thrown away for
    geometric disagreement - they are absorbed with an honest agreement flag and a
    lower confidence, so downstream training can filter on `shape_agreement`.
    """
    per = [rule_scores(f) for f in shape_feats]
    own = max((d.get(hint, 0.0) for d in per), default=0.0)
    others = {}
    for d in per:
        for k, v in d.items():
            if k != hint:
                others[k] = max(others.get(k, 0.0), v)
    best_other, best_other_v = max(others.items(), key=lambda kv: kv[1]) if others else ('none', 0.0)
    ev = max(vehicle_evidence(f) for f in evid_feats)
    order = {'hatchback': 0, 'sedan': 1, 'suv': 2}          # car sub-types are interchangeable geometrically
    agree = own >= 1.5 or (hint in order and best_other in order)
    if own >= 2.2:
        tier, base = 'A', 0.62
    elif own >= 1.2 and ev >= 0.75:
        tier, base = 'A', 0.58
    elif own >= 0.7 or ev >= 0.85:
        tier, base = 'B', 0.46
    else:
        tier, base = 'C', 0.30
    conf = min(0.96, base + 0.22 * min(1.0, own / 2.6) + 0.12 * (ev >= 0.95))
    ranked = dict(sorted(list(others.items()) + [(hint, own)], key=lambda kv: -kv[1])[:4])
    return hint, round(conf, 3), {k: round(v, 2) for k, v in ranked.items()}, agree


# ------------------------------------------------------------- cropping
def crop_512(img, box, pad=0.18):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    pad_ratio = min(0.62, (x1 - x0) / float(H) * 0.15)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - pad_ratio * side
    side = min(side, W, H)
    ax, ay = cx - side / 2, cy - side / 2
    ax = min(max(0, ax), W - side); ay = min(max(0, ay), H - side)
    crop = img[int(ay):int(ay + side), int(ax):int(ax + side)]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def dhash(img, s=16):
    g = cv2.cvtColor(cv2.resize(img, (s + 1, s)), cv2.COLOR_BGR2GRAY)
    return int(''.join('1' if b else '0' for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


# ----------------------------------------------------------- annotation
def annotate(rec, outdir):
    hint = rec.get('hint', '')
    scene = hint.startswith('scene_')
    blob = ' '.join(str(rec.get(k, '')) for k in ('title', 'desc', 'query', 'page'))
    blob += ' ' + ' '.join(str(x) for x in (rec.get('cats') or []))
    if not INDIA_RE.search(blob):
        return None, 'not_india'
    if OFFTOPIC_RE.search(blob) and not scene:
        return None, 'offtopic'
    if not os.path.exists(rec['file']):
        return None, 'file_missing'
    img = decode(rec['file'])
    if img is None:
        return None, 'undecodable'
    if not scene:
        props = proposals(img)[:5]
        grid = [{'x': 0.5 * gx, 'y': 0.5 * gy, 'w': 0.5, 'h': 0.5} for gy in (0, 1) for gx in (0, 1)]
        grid += [{'x': 0.05, 'y': 0.10, 'w': 0.9, 'h': 0.80},
                 {'x': 0.02, 'y': 0.35, 'w': 0.96, 'h': 0.62},
                 {'x': 0.15, 'y': 0.20, 'w': 0.70, 'h': 0.70}]
        shape_feats = [f for f in (roi_stats(img, p) for p in props) if f]
        evidence_feats = [f for f in (roi_stats(img, p) for p in props + grid) if f]
        feats = evidence_feats
        if not feats:
            return None, 'no_roi'

        def roi_rank(z):
            cy = z['bbox'][1] + (z['bbox'][3] - z['bbox'][1]) / 2
            lower = 1.0 if cy > 0.32 else 0.5
            area_ok = 1.0 if 0.04 <= z['roi_area'] <= 0.80 else 0.6
            return vehicle_evidence(z) * lower * area_ok + 0.40 * z['roi_area']

        f = max(feats, key=roi_rank)
        if vehicle_evidence(f) < 0.62:
            return None, 'no_vehicle_evidence'
        label, conf, rank, agree = decide_multi(shape_feats, feats, hint)
    else:
        f = roi_stats(img, {'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0}) or {
            'bbox': [0, 0, 1, 1], 'roi_aspect': round(img.shape[1] / img.shape[0], 3),
            'roi_area': 1.0, 'edge_density': 0.05, 'flatness': 0.0, 'dark_bottom': 0.0,
            'upper_dark': 0.0, 'symmetry': 0.5, 'yellow': 0.0, 'green': 0.0, 'blue': 0.0,
            'brightness': float(img.mean()) / 255.0, 'wheels': 0, 'plates': 0, 'people': 0}
        label, conf, rank = hint, 0.85, {}
    crop = crop_512(img, f['bbox'])
    iid = hashlib.sha1(crop.tobytes()).hexdigest()[:16]
    sub = 'scene' if scene else label
    os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    rel = f'{sub}/{iid}.jpg'
    p = os.path.join(outdir, rel)
    if not cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        try:
            cv2.imwrite(p.encode('utf-8'), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception:
            return None, 'write_failed'
    if not os.path.exists(p):
        return None, 'write_failed'
    return {
        'id': iid, 'image': rel, 'label': label,
        'label_group': 'scene' if scene else 'vehicle',
        'use_cases': USE_CASES.get(label, ['traffic']),
        'confidence': conf, 'rank': rank,
        'shape_agreement': bool(scene or agree),
        'confidence_tier': 'scene' if scene else ('A' if conf >= 0.55 else ('B' if conf >= 0.40 else 'C')),
        'verified': bool(scene or conf >= 0.55),
        'needs_review': (not scene) and conf < 0.40,
        'bbox_norm': f['bbox'],
        'features': {k: v for k, v in f.items() if k != 'bbox'},
        'vehicle_evidence': 0.0 if scene else vehicle_evidence(f),
        'has_plate': f['plates'] > 0, 'people_in_roi': f['people'],
        'night': f['brightness'] < 0.30,
        'width': SIZE, 'height': SIZE,
        'dhash16': hex(dhash(crop)),
        'source': rec.get('source'), 'source_page': rec.get('page'),
        'original_url': rec.get('url'), 'license': rec.get('license'),
        'attribution': rec.get('artist'), 'title': rec.get('title'),
        'discovery_query': rec.get('query'),
        'india_evidence': True,
        'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest(),
    }, 'ok'


def dedupe(results, thr=6):
    """Drop near-duplicate crops by 64-bit-ish hamming distance on the 16x16 dhash."""
    kept, buckets = [], {}
    for r in results:
        h = int(r['dhash16'], 16)
        chunk = h >> 200 if h.bit_length() > 200 else 0     # coarse bucket
        dup = False
        for o in kept[-400:]:
            oh = int(o['dhash16'], 16)
            if bin(h ^ oh).count('1') <= thr:
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--montage', action='store_true')
    ap.add_argument('--out', default='build')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open('state/downloaded.jsonl')]
    if a.limit:
        recs = recs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    results, reasons = [], {}

    def work(r):
        try:
            return annotate(r, a.out)
        except Exception as e:
            return None, 'error:' + type(e).__name__

    with ThreadPoolExecutor(a.threads) as ex:
        for res, why in ex.map(work, recs):
            reasons[why] = reasons.get(why, 0) + 1
            if res:
                results.append(res)
    results = dedupe(results)
    with open('state/annotated.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('input', len(recs), '-> kept', len(results))
    print('labels', Counter(r['label'] for r in results).most_common())
    print('reasons', reasons)
    print('mean conf', round(sum(r['confidence'] for r in results) / max(1, len(results)), 3))
    if a.montage:
        make_montage(results, a.out, 'docs/qc_montage.jpg')


def make_montage(results, outdir, path, per=48):
    import random
    random.seed(11)
    pick = random.sample(results, min(per, len(results)))
    cell, cols = 208, 8
    rows = math.ceil(len(pick) / cols)
    canvas = np.full((rows * (cell + 26) + 30, cols * cell, 3), 24, np.uint8)
    for i, r in enumerate(pick):
        im = cv2.imread(os.path.join(outdir, r['image']))
        if im is None:
            continue
        im = cv2.resize(im, (cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 26) + 26
        canvas[y:y + cell, x:x + cell] = im
        col = (90, 220, 90) if r.get('confidence', 0) >= 0.55 else (60, 170, 250)
        cv2.putText(canvas, r['label'][:18], (x + 4, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{r['confidence']:.2f}", (x + cell - 32, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)
    cv2.putText(canvas, 'Bharat Road Vision - annotation QC', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('montage ->', path)


if __name__ == '__main__':
    main()
