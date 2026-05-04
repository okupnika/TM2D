import argparse
import sys
import random
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts_transition.pipeline import build_transformer


def read_tokens(path):
    return [int(x) for x in open(path).read().strip().split() if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--src-token-file', required=True)
    ap.add_argument('--tgt-token-file', required=True)
    ap.add_argument('--codebook-size', type=int, default=1024)
    ap.add_argument('--m-segments', type=int, default=4)
    ap.add_argument('--segment-len', type=int, default=32)
    ap.add_argument('--prefix-len', type=int, default=48)
    ap.add_argument('--target-end-len', type=int, default=32)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--gpu-id', type=int, default=0)
    ap.add_argument('--out-token-file', required=True)
    args = ap.parse_args()

    device = torch.device('cpu' if args.gpu_id < 0 else f'cuda:{args.gpu_id}')
    sos, eos, pad = args.codebook_size, args.codebook_size + 1, args.codebook_size + 2

    model = build_transformer(args.codebook_size).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    src = read_tokens(args.src_token_file)
    tgt = read_tokens(args.tgt_token_file)

    src_prefix = src[:args.prefix_len]
    # choose a random end segment from target for the final anchor
    seg_len = args.target_end_len
    max_start = max(0, len(tgt) - seg_len)
    rand_start = random.randint(0, max_start) if max_start > 0 else 0
    tgt_end = tgt[rand_start: rand_start + seg_len]

    bridge_len = args.m_segments * args.segment_len

    src_seq = torch.tensor([[sos] + src_prefix + [eos]], dtype=torch.long, device=device)
    trg_canvas = [sos] + src_prefix + [pad] * bridge_len + tgt_end + [eos]
    trg_seq = torch.tensor([trg_canvas], dtype=torch.long, device=device)

    bridge_start = 1 + len(src_prefix)
    bridge_end = bridge_start + bridge_len

    with torch.no_grad():
        out = model.sample_bridge_with_fixed_ends(
            src_seq=src_seq, trg_seq=trg_seq,
            bridge_start=bridge_start, bridge_end=bridge_end,
            trg_sos=sos, trg_eos=eos,
            sample=args.sample, top_k=args.top_k,
            forbid_special=True,
        )

    out_tokens = out[0].tolist()
    with open(args.out_token_file, 'w') as f:
        f.write(' '.join(str(x) for x in out_tokens))
    print(f'Saved tokens: {args.out_token_file}')


if __name__ == '__main__':
    main()
