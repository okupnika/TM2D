# Autoregressive long-motion transfer: HumanML3D (source) -> AIST++ (target) with frozen VQ-VAE

This repo already has the right high-level decomposition:

1. VQ-VAE learns a motion tokenizer/decoder.
2. Transformer predicts motion token sequences.

To train an **autoregressive transition model** while keeping VQ-VAE frozen, the main rewrite is in the transformer input/output contract and the training batch composition.

---

## 1) Keep VQ-VAE frozen (no gradient)

Use the VQ tokenizer only to produce target token IDs and to decode at inference. Do **not** optimize encoder/quantizer/decoder weights in the transition run.

Practical implementation:

- Load the same tokenizer checkpoint used by `tokenize_script_dance.py`.
- Set:
  - `vq_encoder.eval(); quantizer.eval(); vq_decoder.eval()`
  - `for p in module.parameters(): p.requires_grad = False`
- Train only `a2d_transformer` (or a new transition transformer).

Current training already isolates transformer optimization in `TransformerA2DTrainerV2` (`optim.Adam(self.a2d_transformer.parameters(), ...)`).

---

## 2) Rewrite transformer objective as prefix-conditioned autoregression

Current `TransformerV1` does standard teacher-forced next-token prediction with:

- decoder input: `d_tokens[:, :-1]`
- gold labels: `d_tokens[:, 1:]`

To model **long transition** across domains, redefine sequence as:

`[SOS] + src_prefix_tokens(HumanML3D) + [SEP] + tgt_future_tokens(AIST++) + [EOS]`

and train next-token prediction over full sequence, but compute loss only on `tgt_future_tokens` region.

### Why this works

- The model can attend to source-domain motion prefix while autoregressively emitting target-domain continuation.
- At inference, you seed with source prefix tokens and sample long AIST++ continuation.

---

## 3) Minimal model rewrite options

### Option A (smallest change): keep `TransformerV1` decoder-only AR path

Use `TransformerV1.decoding(...)` with a synthetic encoder context (audio or zero context) and rely on decoder causal mask.

### Option B (recommended): add a dedicated token-to-token transition transformer

Use `TransformerV2` style (token encoder + token decoder):

- source stream: HumanML3D motion tokens (prefix)
- target stream: AIST++ motion tokens (shifted right)

This gives explicit cross-attention from target decoder to source prefix encoder.

---

## 4) Trainer changes (core)

Add a new trainer path (copy from `TransformerA2DTrainerV2.forward_a2d`) that:

1. Builds `src_tokens` (HumanML3D prefix).
2. Builds `trg_input` from shifted target sequence.
3. Masks loss outside the target-transition span.

Pseudo-logic:

```python
# batch gives: src_prefix_tokens, src_prefix_len, tgt_tokens
trg_input = tgt_tokens[:, :-1]
gold = tgt_tokens[:, 1:]

enc_output, src_mask = transition_transformer.encoding(src_tokens, src_prefix_len)
logits = transition_transformer.decoding(trg_input, enc_output, src_mask)

# optional: mask to compute CE only over continuation positions
loss = CE(logits.view(-1, V), gold.view(-1), ignore_index=pad_idx)
```

If you use concatenated single-stream format, create a `loss_mask` and multiply tokenwise CE before reduction.

---

## 5) Data pipeline rewrite

`AudioMotionTokenDatasetV2` currently returns `(audio_feature, ..., d_token, d_token_len)`.
For domain transition, create a new dataset returning:

- `src_m_token` (HumanML3D prefix tokens)
- `src_len`
- `tgt_m_token` (AIST++ target sequence)
- `tgt_len`
- optional alignment metadata (tempo, text tag, style id)

Sampling strategies for long generation:

- random prefix length curriculum (short -> long)
- overlap windows with stride (for stable long context)
- optional style token for AIST++ genre

---

## 6) Inference for long generation

1. Tokenize source HumanML3D clip into `src_prefix_tokens` (frozen tokenizer).
2. Start target sequence with `SOS` (and optionally `SEP` after copying prefix in single-stream setup).
3. Autoregressively sample up to `max_steps` with top-k/top-p.
4. Decode generated AIST++ token sequence through frozen VQ decoder.
5. For very long clips, use chunked generation with overlap and stitch in feature space before final decode.

---

## 7) Concrete files to modify

- `tm2d_60fps/networks/transformer.py`
  - add a transition-oriented transformer wrapper or reuse `TransformerV2` for token->token.
- `tm2d_60fps/networks/trainers_x5.py`
  - add `TransformerM2MTransitionTrainer` (from `TransformerA2DTrainerV2` template).
- `tm2d_60fps/data/dataset.py`
  - add paired HumanML3D->AIST++ token dataset.
- new train entry script (copy of `train_a2d_transformer_v5.py`)
  - e.g. `train_m2m_transition_transformer.py`.

---

## 8) Stability tips

- Keep tokenizer codebook fixed across source and target (already implied by `aistppml3d`).
- Use label smoothing off for early debugging; enable later.
- Track transition-only perplexity (not full-sequence perplexity).
- During sampling, ban PAD/SOS/SEP from being generated after warmup.
- Add scheduled sampling only after baseline converges.

---

## 9) Recommended first experiment

- Start with `TransformerV2` token-encoder/token-decoder transition model.
- Freeze VQ-VAE completely.
- Train on 8-16s windows first.
- Evaluate by long rollout (>=30s) with overlap stitching.

This is the least invasive path and matches the current repository design patterns.
