import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from networks.modules import VQEncoderV3, VQDecoderV3
from networks.quantizer import Quantizer, EMAVectorQuantizer
from networks.transformer import TransformerV2


@dataclass
class VQConfig:
    checkpoints_dir: str
    dataset_name: str
    tokenizer_name: str
    which_vqvae: str
    q_mode: str = 'cmt'
    codebook_size: int = 1024
    dim_vq_latent: int = 1024
    lambda_beta: float = 1.0
    n_down: int = 3
    n_resblk: int = 3
    dim_pose: int = 287


def load_frozen_vqvae(cfg: VQConfig, device: torch.device):
    enc_channels = [1024,] + [cfg.dim_vq_latent,] * (cfg.n_down - 1)
    dec_channels = [cfg.dim_vq_latent,] * (cfg.n_down - 1) + [1024, cfg.dim_pose]

    vq_encoder = VQEncoderV3(cfg.dim_pose - 4, enc_channels, cfg.n_down).to(device)
    vq_decoder = VQDecoderV3(cfg.dim_vq_latent, dec_channels, cfg.n_resblk, cfg.n_down).to(device)
    quantizer = EMAVectorQuantizer(cfg.codebook_size, cfg.dim_vq_latent, cfg.lambda_beta).to(device) if cfg.q_mode == 'ema' \
        else Quantizer(cfg.codebook_size, cfg.dim_vq_latent, cfg.lambda_beta).to(device)

    ckpt = torch.load(
        os.path.join(cfg.checkpoints_dir, cfg.dataset_name, cfg.tokenizer_name, 'model', f'{cfg.which_vqvae}.tar'),
        map_location=device,
    )
    vq_encoder.load_state_dict(ckpt['vq_encoder'])
    vq_decoder.load_state_dict(ckpt['vq_decoder'])
    quantizer.load_state_dict(ckpt['quantizer'])

    for m in [vq_encoder, vq_decoder, quantizer]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return vq_encoder, quantizer, vq_decoder


class MotionBridgeTokenDataset(Dataset):
    """Loads paired source/target token files and builds bridge-training samples."""
    def __init__(self, manifest_json: str, codebook_size: int, max_len: int = 256):
        self.items: List[Dict] = json.load(open(manifest_json, 'r'))
        self.max_len = max_len
        self.sos = codebook_size
        self.eos = codebook_size + 1
        self.pad = codebook_size + 2

    def _read_tokens(self, fp: str) -> List[int]:
        toks = open(fp, 'r').read().strip().split()
        return [int(t) for t in toks if len(t)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        src = self._read_tokens(it['src_tokens'])
        tgt = self._read_tokens(it['tgt_tokens'])

        src_prefix_len = min(it.get('src_prefix_len', len(src)//2), len(src))
        tgt_suffix_len = min(it.get('tgt_suffix_len', len(tgt)//4), len(tgt))
        bridge_len = it.get('bridge_len', min(64, max(8, len(tgt)-tgt_suffix_len)))

        src_prefix = src[:src_prefix_len]
        tgt_suffix = tgt[-tgt_suffix_len:] if tgt_suffix_len > 0 else []
        tgt_bridge = tgt[:bridge_len]

        trg = [self.sos] + src_prefix + tgt_bridge + tgt_suffix + [self.eos]
        trg = trg[:self.max_len]
        gold = trg[1:] + [self.pad]

        trg = trg + [self.pad] * (self.max_len - len(trg))
        gold = gold + [self.pad] * (self.max_len - len(gold))
        src_seq = [self.sos] + src_prefix + [self.eos]
        src_seq = src_seq[:self.max_len]
        src_seq = src_seq + [self.pad] * (self.max_len - len(src_seq))
        src_len = min(len([self.sos] + src_prefix + [self.eos]), self.max_len)

        return {
            'src_seq': torch.tensor(src_seq, dtype=torch.long),
            'src_len': torch.tensor(src_len, dtype=torch.long),
            'trg_seq': torch.tensor(trg, dtype=torch.long),
            'gold': torch.tensor(gold, dtype=torch.long),
        }


def build_transformer(codebook_size: int, d_model=512, n_layers=6):
    n_vocab = codebook_size + 3
    pad = codebook_size + 2
    return TransformerV2(
        n_src_vocab=n_vocab, src_pad_idx=pad,
        n_trg_vocab=n_vocab, trg_pad_idx=pad,
        d_src_word_vec=512, d_trg_word_vec=512,
        d_model=d_model, d_inner=2048,
        n_enc_layers=n_layers, n_dec_layers=n_layers,
        n_head=8, d_k=64, d_v=64,
        n_src_position=512, n_trg_position=512,
        trg_emb_prj_weight_sharing=True,
    )
