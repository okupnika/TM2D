# HumanML3D -> AIST++ bridge pipeline (frozen VQ-VAE)

This adds an end-to-end baseline for your requested workflow:

1. **Preprocess paired token manifest**.
2. **Load frozen VQ-VAE** (encoder + quantizer + decoder + checkpoint weights).
3. **Train transformer** with train/validation loop and validation after each epoch.
4. **Infer long horizon** with `m` bridge segments and a random target ending segment.

## Files

- `tm2d_60fps/scripts_transition/preprocess_bridge_dataset.py`
- `tm2d_60fps/scripts_transition/pipeline.py`
- `tm2d_60fps/scripts_transition/train_bridge_transformer.py`
- `tm2d_60fps/scripts_transition/infer_long_horizon_bridge.py`

## 1) Preprocess

```bash
python tm2d_60fps/scripts_transition/preprocess_bridge_dataset.py \
  --src-token-dir ./dataset/aistppml3d/<TOKEN_DIR> \
  --tgt-token-dir ./dataset/aistppml3d/<TOKEN_DIR> \
  --src-split-file ./dataset/aistppml3d/ml3d_train.txt \
  --tgt-split-file ./dataset/aistppml3d/aistpp_train.txt \
  --out-manifest ./dataset/aistppml3d/bridge_train_manifest.json \
  --pairs 40000
```

Create a validation manifest similarly (with val split files).

## 2) Frozen VQ-VAE pipeline

`pipeline.py` provides:

- `VQConfig`
- `load_frozen_vqvae(...)`

It loads `vq_encoder`, `quantizer`, `vq_decoder` from:

`{checkpoints_dir}/{dataset_name}/{tokenizer_name}/model/{which_vqvae}.tar`

and sets all parameters `requires_grad=False`.

## 3) Train with validation every epoch

```bash
python tm2d_60fps/scripts_transition/train_bridge_transformer.py \
  --train-manifest ./dataset/aistppml3d/bridge_train_manifest.json \
  --val-manifest ./dataset/aistppml3d/bridge_val_manifest.json \
  --tokenizer-name VQVAEV3_aistppml3d_motion_1003_d3 \
  --which-vqvae E0310 \
  --save-dir ./checkpoints/aistppml3d/bridge_transformer_exp1/model
```

The script prints:

- `train_nll`
- `val_nll`

for each epoch and saves checkpoints (`E####.tar` + `best.tar`).

## 4) Inference for long horizon with `m` segments + random target end

```bash
python tm2d_60fps/scripts_transition/infer_long_horizon_bridge.py \
  --ckpt ./checkpoints/aistppml3d/bridge_transformer_exp1/model/best.tar \
  --src-token-file ./dataset/aistppml3d/<TOKEN_DIR>/<SRC_ID>.txt \
  --tgt-token-file ./dataset/aistppml3d/<TOKEN_DIR>/<TGT_ID>.txt \
  --m-segments 6 \
  --segment-len 32 \
  --prefix-len 48 \
  --target-end-len 32 \
  --sample \
  --out-token-file ./tmp/bridge_tokens.txt
```

Behavior:

- Beginning tokens are copied from source (`prefix-len`).
- End anchor is a **random segment** sampled from the given target token file.
- Middle is generated autoregressively over `m * segment-len` tokens.

## Notes

- This pipeline trains token-space bridge generation only.
- You can decode generated tokens with the frozen `vq_decoder` for motion output in a downstream step.
