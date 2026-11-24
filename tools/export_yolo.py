"""Export the annotated set as a YOLO detection dataset (labels in YOLO format).

bbox_norm in each record is already [x0,y0,x1,y1] normalised to the ORIGINAL frame;
the published 512 crop is a square window, so we recompute the box inside that crop.
For classification tasks the label alone is enough (see data/*.jsonl).
"""
import json, os, shutil
from collections import Counter

SRC = 'state/annotated.jsonl'
OUT = 'repo/yolo'
CLASSES = ['two_wheeler', 'auto_rickshaw', 'e_rickshaw', 'hatchback', 'sedan', 'suv',
           'lcv', 'bus', 'truck', 'tractor']


def main():
    recs = [json.loads(l) for l in open(SRC)]
    idx = {c: i for i, c in enumerate(CLASSES)}
    counts = Counter()
    for split in ('train', 'val'):
        os.makedirs(f'{OUT}/images/{split}', exist_ok=True)
        os.makedirs(f'{OUT}/labels/{split}', exist_ok=True)
    for r in recs:
        if r['label'] not in idx:
            continue                      # scenes are classification-only
        if 'split' not in r:
            continue
        if r['split'] not in ('train', 'val'):
            continue
        # the crop is centred on the ROI, so a single object fills a predictable box
        x0, y0, x1, y1 = r['bbox_norm']
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        # estimate crop window in original coords (same maths as crop_512, pad 0.18)
        side = max(w, h) * (1 + 2 * 0.18)
        side = min(side, 1.0)
        ax, ay = cx - side / 2, cy - side / 2
        ax, ay = max(0.0, min(ax, 1.0 - side)), max(0.0, min(ay, 1.0 - side))
        bw, bh = w / side, h / side
        bcx, bcy = (cx - ax) / side, (cy - ay) / side
        bcx = max(0.02, min(0.98, bcx)); bcy = max(0.02, min(0.98, bcy))
        bw = max(0.05, min(0.98, bw)); bh = max(0.05, min(0.98, bh))
        img_src = os.path.join('build', r['image'])
        if not os.path.exists(img_src):
            continue
        stem = r['id']
        shutil.copyfile(img_src, f'{OUT}/images/{r["split"]}/{stem}.jpg')
        with open(f'{OUT}/labels/{r["split"]}/{stem}.txt', 'w') as f:
            f.write(f'{idx[r["label"]]} {bcx:.5f} {bcy:.5f} {bw:.5f} {bh:.5f}\n')
        counts[r['split']] += 1
    with open(f'{OUT}/data.yaml', 'w') as f:
        f.write('# Bharat Road Vision - Indian vehicle detection\n')
        f.write(f'path: .\ntrain: images/train\nval: images/val\n\nnames:\n')
        for i, c in enumerate(CLASSES):
            f.write(f'  {i}: {c}\n')
    print('yolo export', dict(counts))


if __name__ == '__main__':
    main()
