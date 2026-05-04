import argparse
import json
import os
import random
from pathlib import Path


def collect_ids(split_file):
    return [l.strip() for l in open(split_file) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-token-dir', required=True)
    ap.add_argument('--tgt-token-dir', required=True)
    ap.add_argument('--src-split-file', required=True)
    ap.add_argument('--tgt-split-file', required=True)
    ap.add_argument('--out-manifest', required=True)
    ap.add_argument('--pairs', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src_ids = collect_ids(args.src_split_file)
    tgt_ids = collect_ids(args.tgt_split_file)

    src_paths = [str(Path(args.src_token_dir) / f'{sid}.txt') for sid in src_ids]
    tgt_paths = [str(Path(args.tgt_token_dir) / f'{tid}.txt') for tid in tgt_ids]
    src_paths = [p for p in src_paths if os.path.exists(p)]
    tgt_paths = [p for p in tgt_paths if os.path.exists(p)]

    items = []
    for _ in range(args.pairs):
        s = random.choice(src_paths)
        t = random.choice(tgt_paths)
        items.append({'src_tokens': s, 'tgt_tokens': t})

    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)
    with open(args.out_manifest, 'w') as f:
        json.dump(items, f, indent=2)
    print(f'Wrote {len(items)} pairs to {args.out_manifest}')


if __name__ == '__main__':
    main()
