# lolvram — agent notes

A static VRAM/RAM and tokens/sec calculator for llama.cpp models. Single page,
no build step, no server. Open `index.html` directly.

This document captures everything you need to maintain the math, add models,
or recalibrate. Read it before changing any formula.

## File layout

- `index.html` — UI markup + inline CSS + dark monospace theme. Contains the
  hardware-preset dropdown, six hardware dials, the MTP toggle, and the
  models table (with sortable headers and per-column substring filters).
- `lolvram.js` — all data and math. Plain script tag (no modules — works
  from `file://` without a server).
- `package.json` + `eslint.config.mjs` — only used for `npm run format`
  (Prettier) and `npm run lint` (ESLint). The page itself has no
  dependencies.

Run `npm run format` then `npm run lint` after every change.

## Architectural rules to preserve

1. **Don't write back into input values on edit.** The `wire()` and filter
   input handlers must only mutate state and call `renderTable()` /
   `writeUrl()`. Writing `input.value = state.x` mid-keystroke clobbers
   partially-typed values like `0.` before the user can type `0.75`. The
   only function permitted to set input values is `renderHardware()`, and
   it only runs when a preset is applied (not on edits).
2. **One `SORT_PRIORITY` list drives all sort tiebreakers.** The clicked
   column moves to position 0 with the user's chosen direction; everything
   else falls through in the list's default order with its default
   direction. Adding a new sortable column means adding it to the list.
3. **Bare URL = defaults.** `state.presetId = "rtx-4090"`, `mtp = false`,
   `mtpDraftCount = 1`, `mtpAcceptance = 0.7`, sort = `tg` `desc`, no
   filters. Anything else gets encoded. URL params: `p`, `mtp`, `md`
   (draft count), `ma` (acceptance), `sort`, `dir`, `fm`/`fq`/`fk`/`fc`
   (filters), and `vram`/`bw`/`ram`/`rambw`/`u`/`tflops` when
   `p=custom`.
4. **All tunable constants at the top of `lolvram.js`.** Per-model
   overrides on the model entry (e.g. `recurrentStateBytes`,
   `mtpDraftParams`, `mtpKvLayers`). Avoid scattering magic numbers
   through the code.

## The math

Everything below is implemented in `estimate()` in `lolvram.js`. Use
gibibytes (`1024^3`) for display; use bytes internally.

### Weights

```
weightBytes(params, bpw) = params × bpw / 8
```

Quant BPW averages (llama.cpp k-quants):

| Quant  | BPW  | Notes                                                    |
| ------ | ---- | -------------------------------------------------------- |
| Q3_K_M | 3.74 | Real models vary ±0.2 BPW depending on embed/output size |
| Q4_K_M | 4.85 | (e.g. Qwen with 248k vocab measures ~5.01 BPW)           |
| Q5_K_M | 5.69 |                                                          |
| Q6_K   | 6.56 |                                                          |
| Q8_0   | 8.50 |                                                          |

If you ever need the _exact_ BPW for a specific model, compute from the
GGUF file size: `(file_bytes × 8) / param_count`. The table values are
typical averages, not per-model truth.

### KV cache

Per generated token, **the full KV cache is read each layer** for attention.
Flash-attn doesn't avoid this — it only skips materializing the attention
matrix.

```
kvBytes = nFullAttnLayers × nKvHeads × headDim × (kBytes + vBytes) × ctx
```

For **hybrid linear/full attention models** (Qwen3.5, 3.6, others with
"3 linear + 1 full" patterns), only `nFullAttnLayers` contribute to the
growing KV cache. Linear-attn layers carry a fixed-size recurrent state
instead — see `recurrentStateBytes` below.

KV quant byte sizes (per element):

| KV quant | Bytes/elem | Derivation                                        |
| -------- | ---------: | ------------------------------------------------- |
| bf16     |      2.000 | raw                                               |
| q8_0     |     1.0625 | 32 int8 + 1 fp16 scale = 34 bytes / 32 elements   |
| q4_0     |     0.5625 | 16 packed + 1 fp16 scale = 18 bytes / 32 elements |

K and V are quantized independently — common configs are bf16/bf16,
q8_0/q8_0, q8_0/q4_0.

### Constant overheads

Calibrated against llama-server-style readings on Qwen3.6-27B / M4 Max:

```
COMPUTE_BUF_BYTES = 1024 MiB  // scales with ubatch × hidden but ~constant here
RECURRENT_STATE_BYTES = 512 MiB  // default; override per-model
```

Per-model `recurrentStateBytes` overrides the default. For hybrid models,
estimate as `~12.5 MiB × nLinearAttnLayers` (Qwen3.6-27B = 48 layers → 600
MiB, matches lui's RS reading exactly).

### MTP draft heads

User-facing controls (state.mtp, state.mtpDraftCount, state.mtpAcceptance):

- **enabled** (bool, default off) — gates the whole feature.
- **draftCount** (int, default 1) — number of draft tokens proposed per main
  pass. Conceptually equivalent to MTP-K heads. Each head costs its own
  weights + KV.
- **acceptance** (0–1, default 0.7) — per-draft-token accept probability.
  Workload-sensitive: code/structured ~0.8, casual chat ~0.6, creative ~0.4.

When `enabled && model.hasMtp && draftCount > 0`:

- **Draft weights** add `draftCount × model.mtpDraftParams × bpw / 8` bytes.
  Counted as non-expert / always-active (must live in VRAM, always read
  per token).
- **Draft KV cache** adds `draftCount × model.mtpKvLayers × nKvHeads ×
headDim × (kBytes+vBytes) × ctx` bytes. Same KV-quant treatment as main.

Both fold into `weightsTotal` / `weightsNonExpert` and `kv` so VRAM
accounting and TG bandwidth math pick them up automatically.

The TG speedup multiplier is computed each render (no per-model constant):

```
mtpMult = 1 + Σᵢ₌₁..draftCount acceptance^i   // sequential-rejection model
```

Each draft token `i` is accepted with probability `acceptance^i` because
rejection cascades (in standard MTP a rejected token discards all
subsequent ones). Examples:

| draftCount | acceptance | mult |
| ---------: | ---------: | ---: |
|          1 |       0.70 | 1.70 |
|          1 |       0.80 | 1.80 |
|          2 |       0.70 | 2.19 |
|          3 |       0.70 | 2.53 |
|          4 |       0.70 | 2.77 |

Diminishing returns from extra heads at moderate acceptance. The multiplier
applies AFTER the bandwidth math computes per-pass rate, so the extra
memory cost of more heads correctly reduces the per-pass throughput before
the speedup is applied.

### MoE accounting

A MoE model declares:

- `nonExpertParams` — embed + attention + shared expert. **Always active.**
  Must fit in VRAM or the row is marked "won't fit" (oversize, no perf).
- `expertParams` — total of all routed expert FFNs. May spill to CPU/RAM.
- `activeExpertParams` — the slice actually read per token (e.g. top-k
  experts out of all). Used only for TG bandwidth math.

VRAM packing: non-expert + KV + overheads first. Whatever VRAM remains
holds as many expert weights as it can; the rest go to RAM. We assume
**uniform routing**, so `expertVramFrac` (fraction of experts resident in
VRAM) approximates the fraction of routed reads that stay on VRAM.

If `unified` ratio is set, there's no separate RAM — spillover treated as
"won't fit (no RAM)" since unified-memory machines don't have a CPU/GPU
split to spill across.

### Hardware effective view

`effectiveHardware(hw)` collapses unified mode:

- Unified ratio set: `effectiveVramBytes = vramGB × unified × 2³⁰`,
  `ramBytes = 0`, `ramBwGBs = vramBwGBs` (so MoE spillover math doesn't
  divide by zero).
- Discrete GPU: `effectiveVramBytes = vramGB × 2³⁰`, `ramBytes` separate,
  `ramBwGBs` from user.

### PP (compute-bound)

```
PP_tps = (tflops × 1e12 × PP_UTIL) / (2 × activeParams)
```

Constants:

```
PP_UTIL_DENSE = 0.25
PP_UTIL_MOE   = 0.10  // small-matmul inefficiency: each expert sees only
                      // batch × k / N_experts tokens (~16 tokens/expert
                      // for pp512 with 8/256 routing)
```

Active params:

- Dense: `model.totalParams`
- MoE: `nonExpertParams + activeExpertParams`

**TFLOPS convention** (important): FP16 tensor throughput with FP32
accumulate, NO sparsity. This is what llama.cpp's tensor-core matmul
actually uses, and what vendors highlight for ML on spec sheets.

- NVIDIA: use the "FP16 with FP32 accumulate" number from the whitepaper.
  RTX 5090 = 209, 4090 = 165, 3090 = 142.
- AMD RDNA3/4 has no separate tensor unit — use dual-issue FP16 (≈ 2×
  FP32). 7900 XTX = 123, R9700 AI PRO = 96.
- Apple Silicon ≤ M4 / A18: dual-issue FP16 (≈ 2× FP32). M4 Max ≈ 34, M3 Ultra ≈ 56.
- Apple Silicon M5 / A19+: GPU Neural Accelerators with Metal 4 Tensor APIs.
  llama.cpp's Metal backend gates the new path on M5/A19 specifically. Use
  Apple's quoted NA TFLOPS (~70 for M5 Max — about 2× the dual-issue GPU
  number). Verified against Apple's "M5 Max PP = 2.3× M4 Max" claim.

A common mistake is to enter NVIDIA's "FP16 (non-tensor)" number — that's
half the right value. The label hint in the UI warns about this.

### TG (bandwidth-bound)

Per generated token, all active weights are read once from memory plus the
entire (current-length) KV cache is read for attention.

```
Dense:
  bytesPerToken = weightsTotal + kv
  TG_tps = (vramBw × TG_UTIL) / bytesPerToken × mtpMult

MoE:
  vramReads = nonExpertBytes + activeExpertBytes × expertVramFrac + kv
  ramReads  = activeExpertBytes × (1 - expertVramFrac)
  TG_tps = TG_UTIL × mtpMult / (vramReads/vramBw + ramReads/ramBw)
```

Constants:

```
TG_UTIL_DENSE = 0.7
TG_UTIL_MOE   = 0.3  // routing / scatter / non-contiguous expert reads
```

**The table reports TG at the LAST token of a fully-filled context window**
(worst case). Early-context TG can be substantially faster. Roughly linear
between the two endpoints:

```
TG(pos)   ≈ bw × util / (weights + kv × pos/maxCtx) × mtpMult
TG(0)     ≈ bw × util / weights × mtpMult                    ← fastest
TG(N/2)   ≈ bw × util / (weights + kv/2) × mtpMult           ← average
TG(N)     ≈ bw × util / (weights + kv) × mtpMult             ← shown in table
```

### What's intentionally ignored

- **Attention compute in PP.** Scales as O(T²), only matters at very long
  prompts. PP estimate is accurate for typical pp512–pp4096; optimistic at
  pp262144.
- **TG varying across context positions.** We report worst case only.
- **CPU-side buffers.** llama.cpp will put some weights (often embed and
  output) and a compute buffer on CPU even when "100% GPU offload". This
  shows up in real readings but isn't modeled here.
- **Multi-GPU tensor parallelism overhead.** For 2× cards we just use a
  single card's bandwidth and TFLOPS, doubled VRAM.
- **Per-model BPW variation.** The quant table values are ±0.2 BPW typical;
  models with very large vocab (Qwen, Gemma) measure higher.

## How to add a new model

You need: HuggingFace model name, its `config.json`, and (ideally) a real
benchmark or two for calibration.

### 1. Pull the config

Fetch `https://huggingface.co/<org>/<model>/raw/main/config.json`. Extract:

| Field                                                                                            | Used as           |
| ------------------------------------------------------------------------------------------------ | ----------------- |
| `num_hidden_layers`                                                                              | `nLayers`         |
| `num_key_value_heads`                                                                            | `nKvHeads`        |
| `head_dim`                                                                                       | `headDim`         |
| `max_position_embeddings`                                                                        | `maxCtx`          |
| `hidden_size`, `intermediate_size`                                                               | param computation |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size` | MoE param split   |
| `mtp_num_hidden_layers` / `num_nextn_predict_layers` / similar                                   | `mtpKvLayers`     |
| layer-pattern info (hybrid models)                                                               | `nFullAttnLayers` |

For **hybrid linear/full attention** (Qwen3.5+, Gemma 4, others), the
config may describe a repeating pattern like "3 linear + 1 full". Count
the full-attention layers; that's `nFullAttnLayers`. The rest contribute
to `recurrentStateBytes`, not to `kv`.

### 2. Compute the params split

Total params: read from the model card or `safetensors` index. Then:

- **Dense**: `nonExpertParams = totalParams`, `expertParams = 0`,
  `activeExpertParams = 0`.
- **MoE**: compute expert params first:
    ```
    perLayerExpertParams = num_experts × 3 × hidden_size × moe_intermediate_size
                            // 3 = gate + up + down
    expertParams = perLayerExpertParams × num_moe_layers
    activeExpertParams = expertParams × (num_experts_per_tok / num_experts)
    nonExpertParams = totalParams - expertParams
    ```
    (If the model has a shared expert that's always active, it's already in
    `nonExpertParams` by virtue of subtraction.)

Verify your split: `nonExpertParams + activeExpertParams` should
approximately equal the model's "active" or "A#B" label (e.g. Qwen3.6-35B-A3B
gives 2.8B + 1.0B = 3.8B ≈ "A3B").

### 3. Estimate MTP per-head overhead (if applicable)

If `mtp_num_hidden_layers > 0` (or equivalent field):

- `hasMtp: true`
- `mtpKvLayers` = `mtp_num_hidden_layers` (usually 1) — this is the KV
  layers cost _per draft head_; total cost scales with the user's
  draftCount setting at runtime.
- `mtpDraftParams` ≈ `totalParams / nLayers` (rough — the MTP head is
  approximately one transformer block; for MoE the draft is usually a
  _dense_ block, smaller than an average MoE layer). Also _per head_.

There's no per-model `mtpMult`. The multiplier is computed at runtime
from the user's `draftCount` and `acceptance` settings via the
sequential-rejection formula above.

### 4. Estimate recurrent state (hybrid only)

```
recurrentStateBytes ≈ 12.5 MiB × nLinearAttnLayers
```

Calibrated from Qwen3.6-27B (48 linear-attn layers → 600 MiB matches lui
exactly). Skip this field for non-hybrid models — the 512 MiB default is
overkill for pure-attention models but small enough not to distort.

### 5. Drop it in `MODELS`

```js
{
    id: "kebab-case-id",
    name: "Display Name",
    type: "dense",  // or "moe"
    totalParams: 27e9,
    nonExpertParams: 27e9,
    expertParams: 0,
    activeExpertParams: 0,
    nLayers: 64,
    nFullAttnLayers: 16,
    nKvHeads: 4,
    headDim: 256,
    maxCtx: 262144,
    hasMtp: true,
    mtpDraftParams: 0.4e9,  // per head
    mtpKvLayers: 1,         // per head
    recurrentStateBytes: 600 * MB
}
```

The calculator picks it up — three rows × all quants × all KV quants
appear automatically.

### 6. Sanity check

Compute by hand a known config (e.g. Q4_K_M at 65k q8/q4 on a 4090) and
compare to a published llama-bench number. If it's wildly off, the most
common culprits in order:

1. Wrong `nFullAttnLayers` (you counted total layers instead of just full)
2. Wrong MoE expert split (`activeExpertParams` should match the "A#B" label)
3. Wrong `mtpDraftParams` (if MTP is on, this shifts both VRAM and TG)
4. Wrong `recurrentStateBytes` (shifts VRAM only — visible in the VRAM column)

## Calibration sources

### MTP gains

- **Qwen3.6-27B (dense)**: 1.71× on RTX 3090 RunPod (38 → 65 t/s); 1.54×
  on single RTX 3090 Q4_K_M (38.7 → 59.5); 2.17× on dual 3090 layer-split
  Q8_0; 2.44× on bandwidth-starved AMD Strix Halo Q8_0. Pattern: more
  bandwidth-starved → larger MTP speedup. Mid-range GPU at Q4 lands
  1.5–1.7×. (Sources: dasroot.net MTP/Qwen3.6 post; startupfortune MTP
  post; ianlpaterson 3090 Ti blog.)
- **DeepSeek-V3 (MoE, similar MTP architecture)**: 1.8× claimed in the
  paper; AMD Instinct measured 1.36–2.11× depending on dataset; reported
  MTP1 acceptance rate ≥80%. (Sources: DeepSeek-V3 DeepWiki MTP page; AMD
  ROCm blog on MTP for SGLang.)

### MoE structure

- Qwen3.6-35B-A3B config: `num_experts: 256`, `num_experts_per_tok: 8`,
  `moe_intermediate_size: 512`, `shared_expert_intermediate_size: 512`,
  `mlp_only_layers: []` (so all 40 layers are MoE).
- Computed: 40 × 256 × 3 × 2048 × 512 = 32.2B expert params; 8/256 active
  → 1.0B active expert; nonExpert = 35 - 32.2 = 2.8B → active total = 3.8B
  ≈ "A3B" label ✓.

### Memory accounting calibration

Real lui readings, M4 Max 36GB unified, Qwen3.6-27B Q4_K_M (5.01 BPW
actual), q8/q8 KV, 128k ctx, MTP on, "hello" prompt:

```
GPU: 16304 model + 4624 KV + 599 RS + 1238 compute MiB
CPU: 682 model + 577 compute MiB
Total: 22.23 GiB VRAM + 1.26 GiB RAM
PP 135.8 t/s · TG 14.6 t/s
```

This dataset drove:

- `COMPUTE_BUF_BYTES` → 1024 MiB (was 256)
- Per-model `recurrentStateBytes` → 600 MiB for Qwen3.6-27B (was a global
  32 MiB constant — way too low)
- M4 Max preset `tflops` → 34 (was 17 — that was FP32; Apple dual-issues
  FP16)

### Outstanding TG calibration question

On M4 Max with the above setup, real TG ≈ 14.6 t/s with MTP on for a
short generation; calculator predicts 29 t/s for empty-context. The 2×
gap is unresolved as of writing. Two suspects (need more data points to
disentangle):

1. **Apple Metal TG utilization < 0.7.** Metal matmul historically hits
   ~50% of bandwidth peak, vs NVIDIA's ~70% with cuBLAS/tensor cores.
2. **MTP gain underrealized on short generations.** Draft-head overhead
   without amortization. Steady-state on long generations expected to
   hit the documented 1.7× speedup.

To resolve: rerun the same setup with **MTP off** (isolates utilization)
and with a **multi-thousand-token prompt** (isolates MTP steady-state).

## Common pitfalls when changing things

- **Adding a new sortable column**: add to `SORT_PRIORITY` (not just to
  the HTML). The position determines tiebreaker precedence; default
  direction encodes "more is better" (`desc` for speed/quality, `asc`
  for memory used).
- **Adding a new hardware preset field**: also update `applyPreset()`,
  the initial `state.hw` shape, `readUrl()`, `writeUrl()`, and
  `renderHardware()`. Forgetting any one will leave the field uninitialized
  or unsynced.
- **Adding a new URL param**: bare URL must still produce defaults. Only
  write the param when it differs from the default. Reading must validate
  (see `VALID_SORT_KEYS` pattern).
- **TG column shows worst case**: don't conflate it with "average TG" or
  "the number llama-bench prints". Real comparisons need either a
  fully-filled context or knowledge of where in the context the
  measurement was taken.
