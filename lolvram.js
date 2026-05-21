// lolvram — VRAM/RAM/tokens-per-second calculator for llama.cpp models.
//
// Math is intentionally approximate. KV cache is computed from full-attention
// layer count (hybrid linear-attention layers carry only a small recurrent
// state which is folded into a constant). PP/TG are bandwidth/compute-bound
// estimates with utilization fudge factors derived from llama-bench numbers on
// reference hardware.

const GB = 1024 * 1024 * 1024
const MB = 1024 * 1024

// Utilization fudge factors. PP is compute-bound; TG is memory-bandwidth-bound.
// MoE PP is penalized hard for small-matmul inefficiency (each expert sees
// only batch×k/N tokens, e.g. ~16 tokens/expert for pp512 with 8/256 routing).
// MoE TG is penalized for routing / scatter / non-contiguous expert reads.
// All utilizations assume FP16 tensor TFLOPS as the compute input.
// MTP speedup is per-model (see MODELS.mtpMult) since acceptance rate varies
// substantially by architecture / draft-head quality.
const PP_UTIL_DENSE = 0.25
const PP_UTIL_MOE = 0.1
const TG_UTIL_DENSE = 0.7
const TG_UTIL_MOE = 0.3

// Constant overheads, calibrated against lui readings on Qwen3.6-27B / M4 Max:
//   real RS ≈ 600 MiB (48 linear-attn layers × ~12.5 MiB SSM state each)
//   real compute buffer ≈ 1.2 GiB at default ubatch for a 27B-class model
// RS is overridden per-model when model.recurrentStateBytes is set.
const COMPUTE_BUF_BYTES = 1024 * MB
const RECURRENT_STATE_BYTES = 512 * MB

// Hardware presets. bandwidth = peak VRAM/unified GB/s. tflops = FP16 tensor
// throughput with FP32 accumulate, NO sparsity (this is what llama.cpp's
// tensor-core matmul actually uses; it's also the value vendors highlight
// for ML on spec sheets). AMD RDNA3/4 and Apple GPUs have no separate tensor
// unit, so those are just FP16 dual-issue / FP16 dense throughput. ramBw =
// realistic CPU-RAM bandwidth used during MoE expert spillover. Numbers are
// vendor specs / commonly-cited; users can edit any field.
const PRESETS = [
    { id: "custom", name: "— Custom —" },
    {
        id: "m4-max-36",
        name: "M4 Max 36GB unified",
        vramGB: 36,
        ramGB: 0,
        unified: 0.75,
        bandwidthGBs: 410,
        tflops: 34,
        ramBwGBs: 410
    },
    {
        id: "m4-max-128",
        name: "M4 Max 128GB unified",
        vramGB: 128,
        ramGB: 0,
        unified: 0.85,
        bandwidthGBs: 546,
        tflops: 34,
        ramBwGBs: 546
    },
    {
        id: "m3-ultra-256",
        name: "M3 Ultra 256GB unified",
        vramGB: 256,
        ramGB: 0,
        unified: 0.85,
        bandwidthGBs: 819,
        tflops: 56,
        ramBwGBs: 819
    },
    {
        id: "rtx-5090",
        name: "RTX 5090 32GB / 96GB DDR5",
        vramGB: 32,
        ramGB: 96,
        unified: null,
        bandwidthGBs: 1792,
        tflops: 209,
        ramBwGBs: 80
    },
    {
        id: "rtx-4090",
        name: "RTX 4090 24GB / 64GB DDR5",
        vramGB: 24,
        ramGB: 64,
        unified: null,
        bandwidthGBs: 1008,
        tflops: 165,
        ramBwGBs: 80
    },
    {
        id: "rtx-3090",
        name: "RTX 3090 24GB / 64GB DDR4",
        vramGB: 24,
        ramGB: 64,
        unified: null,
        bandwidthGBs: 936,
        tflops: 142,
        ramBwGBs: 50
    },
    {
        id: "r9700-32",
        name: "R9700 AI PRO 32GB / 96GB DDR5",
        vramGB: 32,
        ramGB: 96,
        unified: null,
        bandwidthGBs: 644,
        tflops: 96,
        ramBwGBs: 80
    },
    {
        id: "rx7900xtx",
        name: "RX 7900 XTX 24GB / 64GB DDR5",
        vramGB: 24,
        ramGB: 64,
        unified: null,
        bandwidthGBs: 960,
        tflops: 123,
        ramBwGBs: 80
    },
    {
        id: "dual-3090",
        name: "2× RTX 3090 48GB / 128GB DDR4",
        vramGB: 48,
        ramGB: 128,
        unified: null,
        bandwidthGBs: 936,
        tflops: 142,
        ramBwGBs: 50
    }
]

// Curated models. nFullAttnLayers is the count of layers that maintain a
// growing KV cache (Qwen3.5/3.6 use a hybrid pattern — only every 4th layer
// is full attention; the rest are linear-attention with fixed-size state).
//
// MoE accounting splits parameters into "nonExpertParams" (embed + attention +
// shared expert — always active, must live in VRAM for any reasonable speed)
// and "expertParams" (routed FFNs — spillable to CPU/RAM). "activeExpertParams"
// is what one token actually reads from the expert pool (top-k of total).
const MODELS = [
    {
        id: "qwen3.6-27b",
        name: "Qwen3.6-27B",
        type: "dense",
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
        // Per-draft-head cost: ~one transformer block of weights (27B / 64 ≈
        // 422M) + 1 KV-cache layer. Multiplied by draft count at runtime.
        mtpDraftParams: 0.4e9,
        mtpKvLayers: 1,
        // 48 linear-attn layers × ~12.5 MiB/layer (matches lui reading of
        // 599 MiB on M4 Max).
        recurrentStateBytes: 600 * MB
    },
    {
        id: "qwen3.6-35b-a3b",
        name: "Qwen3.6-35B-A3B",
        type: "moe",
        totalParams: 35e9,
        nonExpertParams: 2.8e9,
        expertParams: 32.2e9,
        activeExpertParams: 1.0e9,
        nLayers: 40,
        nFullAttnLayers: 10,
        nKvHeads: 2,
        headDim: 256,
        maxCtx: 262144,
        hasMtp: true,
        // MTP draft is typically a smaller dense transformer block (not MoE).
        // Counts as non-expert (always active, must live in VRAM).
        mtpDraftParams: 0.3e9,
        mtpKvLayers: 1,
        // 30 linear-attn layers × same ~12.5 MiB/layer (extrapolated, untested).
        recurrentStateBytes: 375 * MB
    }
]

const QUANTS = [
    { id: "q3_k_m", name: "Q3_K_M", bpw: 3.74 },
    { id: "q4_k_m", name: "Q4_K_M", bpw: 4.85 },
    { id: "q5_k_m", name: "Q5_K_M", bpw: 5.69 },
    { id: "q6_k", name: "Q6_K", bpw: 6.56 },
    { id: "q8_0", name: "Q8_0", bpw: 8.5 }
]

const KV_QUANTS = [
    { id: "bf16-bf16", name: "bf16/bf16", kBytes: 2, vBytes: 2 },
    { id: "q8-q8", name: "q8_0/q8_0", kBytes: 1.06, vBytes: 1.06 },
    { id: "q8-q4", name: "q8_0/q4_0", kBytes: 1.06, vBytes: 0.56 }
]

// max, ½ max, ¼ max by default. With small contexts enabled, keep halving
// down to 4096 — useful for chat-sized usage but inflates the table.
function contextSizesFor(model, includeSmall) {
    const sizes = [model.maxCtx, Math.floor(model.maxCtx / 2), Math.floor(model.maxCtx / 4)]
    if (!includeSmall) return sizes
    let c = Math.floor(model.maxCtx / 8)
    while (c >= 4096) {
        sizes.push(c)
        c = Math.floor(c / 2)
    }
    return sizes
}

function weightBytes(params, bpw) {
    return (params * bpw) / 8
}

function kvBytes(model, kvQuant, ctx) {
    const perToken = model.nFullAttnLayers * model.nKvHeads * model.headDim * (kvQuant.kBytes + kvQuant.vBytes)
    return perToken * ctx
}

// Hardware view: collapses the unified-ratio case so callers don't have to
// special-case Macs. effectiveVramBytes is the cap we actually fit against;
// hasRam tells the MoE spillover path whether there's any DRAM to spill into.
function effectiveHardware(hw) {
    const unified = hw.unified > 0 && hw.unified <= 1
    const vramGB = unified ? hw.vramGB * hw.unified : hw.vramGB
    return {
        unified,
        effectiveVramBytes: vramGB * GB,
        ramBytes: unified ? 0 : hw.ramGB * GB,
        bandwidthGBs: hw.bandwidthGBs,
        ramBwGBs: unified ? hw.bandwidthGBs : hw.ramBwGBs,
        tflops: hw.tflops,
        hasRam: unified ? false : hw.ramGB > 0
    }
}

// Core estimate. Returns the row data shown in the table. Math:
//   weights = totalParams × bpw / 8
//   kv      = nFullAttnLayers × nKvHeads × headDim × (kB+vB) × ctx
//   fixed overhead: compute buffer + recurrent state
// Then we attempt to fit. Dense: all-or-nothing. MoE: non-expert + kv must fit,
// expert weights may spill to CPU/RAM and degrade TG proportionally.
//
// mtp: { enabled, draftCount, acceptance } — draftCount × per-head memory cost,
// speedup multiplier = 1 + Σ acceptance^i for i in 1..draftCount.
function estimate(hw, model, quant, kvQuant, ctx, mtp) {
    const ehw = effectiveHardware(hw)

    // MTP draft heads: when enabled, each head adds its own small dense block
    // of weights (always active, lives in VRAM) and its own per-head KV layers
    // that grow with ctx. Memory scales linearly with draftCount; the
    // multiplier is computed below from acceptance + draftCount.
    const mtpActive = mtp.enabled && model.hasMtp && mtp.draftCount > 0
    const heads = mtpActive ? mtp.draftCount : 0
    const mtpDraftBytes = heads * weightBytes(model.mtpDraftParams || 0, quant.bpw)
    const mtpKvBytes = heads * (model.mtpKvLayers || 0) * model.nKvHeads * model.headDim * (kvQuant.kBytes + kvQuant.vBytes) * ctx

    const weightsTotal = weightBytes(model.totalParams, quant.bpw) + mtpDraftBytes
    const weightsNonExpert = weightBytes(model.nonExpertParams, quant.bpw) + mtpDraftBytes
    const weightsExpert = weightBytes(model.expertParams, quant.bpw)
    const weightsActiveExpert = weightBytes(model.activeExpertParams, quant.bpw)
    const kv = kvBytes(model, kvQuant, ctx) + mtpKvBytes
    const fixed = COMPUTE_BUF_BYTES + (model.recurrentStateBytes ?? RECURRENT_STATE_BYTES)

    const vramCap = ehw.effectiveVramBytes
    const requiredInVram = weightsNonExpert + kv + fixed

    let vramUsed = 0
    let ramUsed = 0
    let status = "fits"
    let statusClass = "good"
    let oversize = false
    let expertVramFrac = 1 // fraction of expert weights resident in VRAM

    if (requiredInVram > vramCap) {
        // Even the minimum (non-expert + KV) doesn't fit. We mark as oversize
        // and zero out perf; we still report the would-be totals so the user
        // can see how far over they are.
        oversize = true
        status = "won't fit"
        statusClass = "bad"
        vramUsed = requiredInVram
        ramUsed = weightsExpert
    } else if (model.type === "dense") {
        if (weightsTotal + kv + fixed <= vramCap) {
            vramUsed = weightsTotal + kv + fixed
            ramUsed = 0
        } else {
            // Dense weights spilling to CPU = unusably slow; flag as oversize.
            oversize = true
            status = "won't fit (dense)"
            statusClass = "bad"
            vramUsed = vramCap
            ramUsed = weightsTotal + kv + fixed - vramCap
        }
    } else {
        // MoE path: try to pack as many expert weights into remaining VRAM as
        // possible; spill the rest to RAM. If there's no RAM (or unified
        // memory and the cap is set), spillover counts as oversize.
        const vramFreeForExperts = vramCap - requiredInVram
        if (vramFreeForExperts >= weightsExpert) {
            vramUsed = requiredInVram + weightsExpert
            ramUsed = 0
            expertVramFrac = 1
        } else {
            const expertsInVram = Math.max(0, vramFreeForExperts)
            const expertsInRam = weightsExpert - expertsInVram
            vramUsed = requiredInVram + expertsInVram
            ramUsed = expertsInRam
            expertVramFrac = expertsInVram / weightsExpert
            if (!ehw.hasRam) {
                oversize = true
                status = "won't fit (no RAM)"
                statusClass = "bad"
            } else if (expertsInRam > ehw.ramBytes) {
                oversize = true
                status = "RAM too small"
                statusClass = "bad"
            } else {
                const pct = Math.round(expertVramFrac * 100)
                status = `MoE spill: ${pct}% experts in VRAM`
                statusClass = "warn"
            }
        }
    }

    let ppTps = 0
    let tgTps = 0

    if (!oversize) {
        const activeParams = model.type === "moe" ? model.nonExpertParams + model.activeExpertParams : model.totalParams
        const ppUtil = model.type === "moe" ? PP_UTIL_MOE : PP_UTIL_DENSE
        ppTps = (ehw.tflops * 1e12 * ppUtil) / (2 * activeParams)

        const tgUtil = model.type === "moe" ? TG_UTIL_MOE : TG_UTIL_DENSE
        const vramBwBs = ehw.bandwidthGBs * 1e9
        const ramBwBs = (ehw.ramBwGBs || vramBwBs) * 1e9

        if (model.type === "dense") {
            // Per generated token in TG mode: read all weights once + read the
            // full KV cache once (attention reads K[0..N] and V[0..N] across
            // every full-attn layer). Flash-attn doesn't avoid this — it just
            // skips materializing the attention matrix.
            const bytesPerToken = weightsTotal + kv
            tgTps = (vramBwBs * tgUtil) / bytesPerToken
        } else {
            // MoE: non-expert + KV always read from VRAM. Active-expert bytes
            // split by where experts ended up (uniform routing assumption).
            const nonExpertBytes = weightsNonExpert
            const expertVramBytes = weightsActiveExpert * expertVramFrac
            const expertRamBytes = weightsActiveExpert * (1 - expertVramFrac)
            const tVram = (nonExpertBytes + expertVramBytes + kv) / vramBwBs
            const tRam = expertRamBytes / ramBwBs
            tgTps = tgUtil / (tVram + tRam)
        }

        // Speedup from MTP: sequential-rejection model. Each draft token i is
        // accepted with probability acceptance^i (rejection cascades), so
        // average tokens-per-pass = 1 + Σᵢ₌₁..k aᶦ. For k=1, a=0.7 → 1.7×;
        // k=3, a=0.7 → 2.53×. Memory cost is already folded in above.
        if (mtpActive) {
            let mult = 1
            for (let i = 1; i <= heads; i++) mult += mtp.acceptance ** i
            tgTps *= mult
        }

        // MoE TG warning: VRAM fits but spillover penalty drags us into "ok"
        // territory. Color the row yellow not green when that happens.
        if (statusClass === "good" && model.type === "moe" && expertVramFrac < 0.99) {
            statusClass = "ok"
        }
    }

    return {
        model,
        quant,
        kvQuant,
        ctx,
        weightsTotal,
        kv,
        vramUsed,
        ramUsed,
        vramCap,
        ramCap: ehw.ramBytes,
        ppTps,
        tgTps,
        status,
        statusClass,
        oversize
    }
}

// ---------- UI state ----------

const state = {
    hw: { vramGB: 24, ramGB: 64, unified: null, bandwidthGBs: 1008, tflops: 165, ramBwGBs: 80 },
    presetId: "rtx-4090",
    mtp: false,
    mtpDraftCount: 1,
    mtpAcceptance: 0.7,
    smallCtx: false,
    sortKey: "tg",
    sortDir: "desc",
    filters: { model: "", quant: "", kv: "", ctx: "" }
}

function applyPreset(presetId) {
    const p = PRESETS.find((x) => x.id === presetId)
    if (!p || p.id === "custom") return
    state.hw.vramGB = p.vramGB
    state.hw.ramGB = p.ramGB
    state.hw.unified = p.unified
    state.hw.bandwidthGBs = p.bandwidthGBs
    state.hw.tflops = p.tflops
    state.hw.ramBwGBs = p.ramBwGBs
}

// ---------- URL sync ----------
//
// Defaults that should produce a bare URL (no params): preset rtx-4090, MTP
// on, search empty, sort by TG desc. Anything else gets encoded.
const DEFAULT_PRESET_ID = "rtx-4090"
const DEFAULT_SORT_KEY = "tg"
const DEFAULT_SORT_DIR = "desc"
const VALID_SORT_KEYS = new Set(["model", "quant", "kv", "ctx", "weights", "kvsize", "vram", "ram", "pp", "tg", "status"])

function readUrl() {
    const params = new URLSearchParams(window.location.search)
    const num = (k) => {
        const v = params.get(k)
        if (v == null || v === "") return null
        const n = parseFloat(v)
        return isFinite(n) ? n : null
    }

    const presetId = params.get("p")
    if (presetId && PRESETS.some((x) => x.id === presetId)) {
        state.presetId = presetId
        if (presetId !== "custom") applyPreset(presetId)
    }

    if (state.presetId === "custom") {
        const v = num("vram")
        if (v != null) state.hw.vramGB = v
        const bw = num("bw")
        if (bw != null) state.hw.bandwidthGBs = bw
        const ram = num("ram")
        if (ram != null) state.hw.ramGB = ram
        const rambw = num("rambw")
        if (rambw != null) state.hw.ramBwGBs = rambw
        const u = num("u")
        state.hw.unified = u
        const tf = num("tflops")
        if (tf != null) state.hw.tflops = tf
    }

    const mtp = params.get("mtp")
    if (mtp != null) state.mtp = !(mtp === "0" || mtp === "false")
    const md = num("md")
    if (md != null && md >= 0) state.mtpDraftCount = Math.max(0, Math.round(md))
    const ma = num("ma")
    if (ma != null && ma >= 0 && ma <= 1) state.mtpAcceptance = ma
    const sc = params.get("sc")
    if (sc != null) state.smallCtx = !(sc === "0" || sc === "false")
    for (const k of ["model", "quant", "kv", "ctx"]) {
        const v = params.get("f" + k[0])
        if (v != null) state.filters[k] = v
    }

    const sortKey = params.get("sort")
    if (sortKey && VALID_SORT_KEYS.has(sortKey)) state.sortKey = sortKey
    const sortDir = params.get("dir")
    if (sortDir === "asc" || sortDir === "desc") state.sortDir = sortDir
}

function writeUrl() {
    const params = new URLSearchParams()
    if (state.presetId && state.presetId !== DEFAULT_PRESET_ID) {
        params.set("p", state.presetId)
    }
    if (state.presetId === "custom") {
        params.set("vram", state.hw.vramGB)
        params.set("bw", state.hw.bandwidthGBs)
        params.set("ram", state.hw.ramGB)
        params.set("rambw", state.hw.ramBwGBs)
        params.set("tflops", state.hw.tflops)
        if (state.hw.unified != null && state.hw.unified !== "") {
            params.set("u", state.hw.unified)
        }
    }
    if (state.mtp) params.set("mtp", "1")
    if (state.mtpDraftCount !== 1) params.set("md", String(state.mtpDraftCount))
    if (state.mtpAcceptance !== 0.7) params.set("ma", String(state.mtpAcceptance))
    if (state.smallCtx) params.set("sc", "1")
    for (const k of ["model", "quant", "kv", "ctx"]) {
        if (state.filters[k]) params.set("f" + k[0], state.filters[k])
    }
    if (state.sortKey !== DEFAULT_SORT_KEY) params.set("sort", state.sortKey)
    if (state.sortDir !== DEFAULT_SORT_DIR) params.set("dir", state.sortDir)

    const qs = params.toString()
    const url = window.location.pathname + (qs ? "?" + qs : "")
    history.replaceState(null, "", url)
}

// ---------- rendering ----------

function fmtGiB(bytes) {
    if (!bytes) return "—"
    const gib = bytes / GB
    return gib >= 10 ? gib.toFixed(1) : gib.toFixed(2)
}

function fmtGiBPct(bytes, cap) {
    if (!bytes) return "—"
    const main = fmtGiB(bytes)
    if (!cap) return main
    const pct = Math.round((bytes / cap) * 100)
    return `${main} (${pct}%)`
}

function fmtCtx(c) {
    if (c >= 1024) return `${Math.round(c / 1024)}k`
    return String(c)
}

function fmtTps(v) {
    if (!isFinite(v) || v <= 0) return "—"
    if (v >= 1000) return Math.round(v).toString()
    if (v >= 100) return v.toFixed(0)
    if (v >= 10) return v.toFixed(1)
    return v.toFixed(2)
}

function tpsClass(v, lo, hi) {
    if (!isFinite(v) || v <= 0) return "bad"
    if (v < lo) return "bad"
    if (v < hi) return "ok"
    return "good"
}

function tgClass(v) {
    return tpsClass(v, 10, 30)
}

function ppClass(v) {
    return tpsClass(v, 100, 400)
}

function buildRows() {
    const rows = []
    const f = state.filters
    const mtp = { enabled: state.mtp, draftCount: state.mtpDraftCount, acceptance: state.mtpAcceptance }
    const match = (haystack, needle) => !needle.trim() || haystack.toLowerCase().includes(needle.trim().toLowerCase())
    for (const model of MODELS) {
        if (!match(model.name, f.model)) continue
        const ctxs = contextSizesFor(model, state.smallCtx)
        for (const quant of QUANTS) {
            if (!match(quant.name, f.quant)) continue
            for (const kv of KV_QUANTS) {
                if (!match(kv.name, f.kv)) continue
                for (const ctx of ctxs) {
                    if (!match(fmtCtx(ctx), f.ctx)) continue
                    rows.push(estimate(state.hw, model, quant, kv, ctx, mtp))
                }
            }
        }
    }
    return rows
}

// Sort-priority list. The active sort column moves to position 0 with the
// user's chosen direction; the remaining columns stay in this order with
// their default direction and act as tiebreakers. Default dirs encode "more
// is better" (desc for speed/quality, asc for memory used).
const SORT_PRIORITY = [
    { key: "tg", dir: "desc", get: (r) => (r.oversize ? -Infinity : r.tgTps) },
    { key: "pp", dir: "desc", get: (r) => (r.oversize ? -Infinity : r.ppTps) },
    { key: "vram", dir: "asc", get: (r) => r.vramUsed },
    { key: "ram", dir: "asc", get: (r) => r.ramUsed },
    { key: "ctx", dir: "desc", get: (r) => r.ctx },
    { key: "quant", dir: "desc", get: (r) => r.quant.bpw },
    { key: "kv", dir: "desc", get: (r) => r.kvQuant.kBytes + r.kvQuant.vBytes },
    { key: "weights", dir: "desc", get: (r) => r.weightsTotal },
    { key: "kvsize", dir: "desc", get: (r) => r.kv },
    { key: "model", dir: "asc", get: (r) => r.model.name },
    { key: "status", dir: "asc", get: (r) => r.status }
]

function sortRows(rows) {
    const primary = SORT_PRIORITY.find((s) => s.key === state.sortKey)
    const rest = SORT_PRIORITY.filter((s) => s.key !== state.sortKey)
    const order = primary ? [{ ...primary, dir: state.sortDir }, ...rest] : SORT_PRIORITY
    return rows.slice().sort((a, b) => {
        for (const { dir, get } of order) {
            const av = get(a)
            const bv = get(b)
            const d = typeof av === "string" ? av.localeCompare(bv) : av - bv
            if (d !== 0) return (dir === "asc" ? 1 : -1) * d
        }
        return 0
    })
}

function renderTable() {
    const rows = sortRows(buildRows())
    const body = document.getElementById("models-body")
    body.innerHTML = ""

    if (rows.length === 0) {
        const tr = document.createElement("tr")
        tr.innerHTML = `<td colspan="11" class="empty">no rows match the current filters</td>`
        body.appendChild(tr)
    }

    for (const r of rows) {
        const tr = document.createElement("tr")
        const ppCls = r.oversize ? "bad" : ppClass(r.ppTps)
        const tgCls = r.oversize ? "bad" : tgClass(r.tgTps)
        const ramCls = r.ramUsed > 0 ? "warn" : "dim"

        tr.innerHTML = `
            <td>${r.model.name}</td>
            <td><span class="pill">${r.quant.name}</span></td>
            <td><span class="pill">${r.kvQuant.name}</span></td>
            <td class="num">${fmtCtx(r.ctx)}</td>
            <td class="num dim">${fmtGiB(r.weightsTotal)}</td>
            <td class="num dim">${fmtGiB(r.kv)}</td>
            <td class="num">${fmtGiBPct(r.vramUsed, r.vramCap)}</td>
            <td class="num ${ramCls}">${fmtGiBPct(r.ramUsed, r.ramCap)}</td>
            <td class="num ${ppCls}">${r.oversize ? "—" : fmtTps(r.ppTps)}</td>
            <td class="num ${tgCls}">${r.oversize ? "—" : fmtTps(r.tgTps)}</td>
            <td class="status ${r.statusClass}">${r.status}</td>
        `
        body.appendChild(tr)
    }

    document.querySelectorAll("#models thead tr.sortrow th").forEach((th) => {
        const key = th.getAttribute("data-sort")
        th.classList.remove("sort-asc", "sort-desc")
        if (key === state.sortKey) {
            th.classList.add(state.sortDir === "asc" ? "sort-asc" : "sort-desc")
        }
    })
}

function updateUnifiedDisable() {
    const unifiedActive = state.hw.unified > 0 && state.hw.unified <= 1
    document.getElementById("ram").disabled = unifiedActive
    document.getElementById("rambw").disabled = unifiedActive
}

function updateMtpDisable() {
    const off = !state.mtp
    document.getElementById("mtp-draft").disabled = off
    document.getElementById("mtp-accept").disabled = off
}

// Only called when a preset is applied — pushes preset values into the inputs.
// We intentionally avoid this on per-input edits so we don't clobber a
// partially-typed value mid-keystroke.
function renderHardware() {
    document.getElementById("vram").value = state.hw.vramGB
    document.getElementById("ram").value = state.hw.ramGB
    document.getElementById("unified").value = state.hw.unified ?? ""
    document.getElementById("bw").value = state.hw.bandwidthGBs
    document.getElementById("tflops").value = state.hw.tflops
    document.getElementById("rambw").value = state.hw.ramBwGBs
    document.getElementById("preset").value = state.presetId
    updateUnifiedDisable()
}

function rerender() {
    renderHardware()
    renderTable()
}

// ---------- wiring ----------

function init() {
    const presetSel = document.getElementById("preset")
    for (const p of PRESETS) {
        const opt = document.createElement("option")
        opt.value = p.id
        opt.textContent = p.name
        presetSel.appendChild(opt)
    }

    applyPreset(state.presetId)
    readUrl()
    document.getElementById("mtp").checked = state.mtp
    document.getElementById("mtp-draft").value = state.mtpDraftCount
    document.getElementById("mtp-accept").value = state.mtpAcceptance
    document.getElementById("small-ctx").checked = state.smallCtx
    updateMtpDisable()
    // Prime filter inputs from state once. After this we never write back —
    // the user's typing is the source of truth (avoids the same value-clobber
    // bug we hit on the hardware inputs).
    document.querySelectorAll("#models thead tr.filterrow input[data-filter]").forEach((el) => {
        const k = el.getAttribute("data-filter")
        el.value = state.filters[k] || ""
    })

    presetSel.addEventListener("change", (e) => {
        state.presetId = e.target.value
        applyPreset(state.presetId)
        rerender()
        writeUrl()
    })

    // Read each input on edit but don't write back into the DOM — that would
    // clobber partially-typed values like "0." before the user finishes
    // typing "0.75". We only re-render the table; the unified-ratio disable
    // toggle for RAM fields is also handled here directly.
    const wire = (id, key, parse) => {
        document.getElementById(id).addEventListener("input", (e) => {
            const raw = e.target.value
            const v = raw === "" ? null : parse(raw)
            state.hw[key] = v == null || !isFinite(v) ? (key === "unified" ? null : 0) : v
            state.presetId = "custom"
            document.getElementById("preset").value = "custom"
            if (key === "unified") updateUnifiedDisable()
            renderTable()
            writeUrl()
        })
    }
    wire("vram", "vramGB", parseFloat)
    wire("ram", "ramGB", parseFloat)
    wire("unified", "unified", parseFloat)
    wire("bw", "bandwidthGBs", parseFloat)
    wire("tflops", "tflops", parseFloat)
    wire("rambw", "ramBwGBs", parseFloat)

    document.getElementById("mtp").addEventListener("change", (e) => {
        state.mtp = e.target.checked
        updateMtpDisable()
        renderTable()
        writeUrl()
    })

    document.getElementById("mtp-draft").addEventListener("input", (e) => {
        const v = parseInt(e.target.value, 10)
        state.mtpDraftCount = Number.isFinite(v) && v >= 0 ? v : 0
        renderTable()
        writeUrl()
    })

    document.getElementById("mtp-accept").addEventListener("input", (e) => {
        const v = parseFloat(e.target.value)
        state.mtpAcceptance = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0
        renderTable()
        writeUrl()
    })

    document.getElementById("small-ctx").addEventListener("change", (e) => {
        state.smallCtx = e.target.checked
        renderTable()
        writeUrl()
    })

    document.querySelectorAll("#models thead tr.filterrow input[data-filter]").forEach((el) => {
        el.addEventListener("input", () => {
            const k = el.getAttribute("data-filter")
            state.filters[k] = el.value
            renderTable()
            writeUrl()
        })
    })

    document.querySelectorAll("#models thead tr.sortrow th").forEach((th) => {
        th.addEventListener("click", () => {
            const key = th.getAttribute("data-sort")
            if (!key) return
            if (state.sortKey === key) {
                state.sortDir = state.sortDir === "asc" ? "desc" : "asc"
            } else {
                state.sortKey = key
                state.sortDir = key === "model" || key === "quant" || key === "kv" || key === "ctx" ? "asc" : "desc"
            }
            renderTable()
            writeUrl()
        })
    })

    rerender()
    writeUrl()
}

document.addEventListener("DOMContentLoaded", init)
