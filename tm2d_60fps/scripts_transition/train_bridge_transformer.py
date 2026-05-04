import argparse
import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts_transition.pipeline import MotionBridgeTokenDataset, build_transformer, VQConfig, load_frozen_vqvae


def run_epoch(model, loader, opt=None, pad_idx=1026, device='cpu'):
    is_train = opt is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src = batch['src_seq'].to(device)
        trg = batch['trg_seq'].to(device)
        gold = batch['gold'].to(device)

        logits = model(src, trg)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            gold.reshape(-1),
            ignore_index=pad_idx,
            reduction='sum'
        )
        n_tok = (gold != pad_idx).sum().item()

        if is_train:
            opt.zero_grad()
            (loss / max(n_tok, 1)).backward()
            opt.step()

        total_loss += loss.item()
        total_tokens += max(n_tok, 1)

    return total_loss / total_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-manifest', required=True)
    ap.add_argument('--val-manifest', required=True)
    ap.add_argument('--checkpoints-dir', default='./checkpoints')
    ap.add_argument('--dataset-name', default='aistppml3d')
    ap.add_argument('--tokenizer-name', required=True)
    ap.add_argument('--which-vqvae', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--codebook-size', type=int, default=1024)
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--gpu-id', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cpu' if args.gpu_id < 0 else f'cuda:{args.gpu_id}')
    os.makedirs(args.save_dir, exist_ok=True)

    # frozen VQ-VAE pipeline loaded once (for consistent tokenizer/decoder checkpointing)
    vq_cfg = VQConfig(
        checkpoints_dir=args.checkpoints_dir, dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name, which_vqvae=args.which_vqvae,
        codebook_size=args.codebook_size,
    )
    _vq_encoder, _quantizer, _vq_decoder = load_frozen_vqvae(vq_cfg, device)

    train_ds = MotionBridgeTokenDataset(args.train_manifest, args.codebook_size)
    val_ds = MotionBridgeTokenDataset(args.val_manifest, args.codebook_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_transformer(args.codebook_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pad_idx = args.codebook_size + 2

    best = 1e9
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, pad_idx, device)
        va = run_epoch(model, val_loader, None, pad_idx, device)
        print(f'Epoch {ep:03d} | train_nll={tr:.6f} | val_nll={va:.6f}')

        ckpt = {'epoch': ep, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'val_nll': va}
        torch.save(ckpt, os.path.join(args.save_dir, f'E{ep:04d}.tar'))
        if va < best:
            best = va
            torch.save(ckpt, os.path.join(args.save_dir, 'best.tar'))


if __name__ == '__main__':
    main()
