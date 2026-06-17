"""
FDM Gemma 4 K-sweep — find where global-only carrier retrieval holds.

Gemma 4 (31B Dense): LOCAL sliding-window attention (window=1024) interleaved
with GLOBAL full-context attention, 5:1, final layer global. Below W=1024 every
layer reads the whole FDM block; above it only the ~10 global layers (of 60)
integrate carriers across the full block. Sweep walks K across the 1024 boundary
and STOPS at the first K where per-channel recall drops below the bar.

Method: TWO-BLOCK (attention weights trained directly; embeddings + MLPs frozen).
No LoRA — a low-rank adapter cannot install FDM's carrier structure.

TWO LAUNCH PATHS (auto-detected):
  python3 fdm_gemma4_ksweep.py ...           -> single-process, device_map="auto"
                                                (1 GPU, or 2 GPUs naive model-parallel)
  accelerate launch fdm_gemma4_ksweep.py ... -> FSDP (sharded, parallel, ~2x faster)
                                                requires fsdp.yaml (see startup notes)

Usage:
  python3 fdm_gemma4_ksweep.py --print       # stage table only, no GPU
  python3 fdm_gemma4_ksweep.py --only K40    # calibrate one stage + time it
  python3 fdm_gemma4_ksweep.py               # full ladder, single-process
  accelerate launch --config_file fsdp.yaml fdm_gemma4_ksweep.py   # FSDP full ladder

Env:  GEMMA_DIR (weights), FDM_REPO (work dir)
"""

import os, sys, json, math, argparse

# accelerate launch sets LOCAL_RANK; bare python3 does not.
IS_DISTRIBUTED = "LOCAL_RANK" in os.environ

LOCAL_WINDOW = 1024
DF = 0.390625
RECALL_BAR = 0.99
REPO = os.environ.get("FDM_REPO", "/workspace/FDM_GEMMA4")
GEMMA_DIR = os.environ.get("GEMMA_DIR", "/workspace/gemma-4-31B")
RESULTS_PATH = os.environ.get("KSWEEP_RESULTS", f"{REPO}/ksweep_results.json")
BATCH_SIZE = 1
GRAD_ACCUM = 8

# ============================================================
# Vocab / physics / stages / schemas
# ============================================================
def _get_vocab(model_path):
    mp = model_path.lower()
    if "gemma-4" in mp or "gemma4" in mp: return 262144
    if "hermes" in mp: return 128256
    if "lfm" in mp or "liquid" in mp: return 65536
    if "gpt2" in mp: return 50257
    return 151936

def assert_vocab(model, expected):
    actual = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
    assert actual == expected, (f"vocab mismatch: _get_vocab={expected}, config={actual}. "
                                f"Fix _get_vocab — token_map depends on it.")

def physics(K, N):
    f_s = DF * N
    return {"K": K, "N": N, "f_s": f_s, "W": 2 * N,
            "nyquist_margin_hz": round(f_s / 2.0 - K, 2),
            "window_ratio": round((2 * N) / LOCAL_WINDOW, 2)}

def make_stage(K, N, name, samples, epochs, lr):
    p = physics(K, N)
    return {"tag": f"K{K}", "name": name, "K": K, "num_tokens": N,
            "sample_rate": p["f_s"], "W": p["W"], "window_ratio": p["window_ratio"],
            "nyquist_margin_hz": p["nyquist_margin_hz"], "max_length": p["W"] + 512,
            "num_levels": 64, "a_high": 1.0, "a_low": 0.15,
            "samples": samples, "epochs": epochs, "lr": lr}

KSWEEP_STAGES = [
    make_stage(40,  256,  "K40  W512  (0.5x window, baseline)",   4000, 2, 3e-5),
    make_stage(80,  512,  "K80  W1024 (1.0x window edge)",        4000, 2, 3e-5),
    make_stage(99,  512,  "K99  W1024 (edge, THIN nyquist 1Hz)",  4000, 2, 3e-5),
    make_stage(120, 1024, "K120 W2048 (2x window, global-only)",  4000, 2, 3e-5),
    make_stage(160, 1024, "K160 W2048 (2x window, global-only)",  5000, 2, 2e-5),
    make_stage(320, 2048, "K320 W4096 (4x window, global-only)",  6000, 2, 2e-5),
]

def extend_schemas(base_schemas, K):
    schemas = dict(base_schemas)
    for ch in range(len(base_schemas), K):
        name = f"CH{ch:03d}"
        schemas[ch] = (name, [f"{name}_A", f"{name}_B", f"{name}_C", f"{name}_D"])
    return schemas

def _load_base_module():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eidetic_qwen3_fdm_e2e as base
    return base

def generate_stage(base, tokenizer, stage, prefix):
    base.MODEL_ID = GEMMA_DIR
    base.VOCAB_SIZE = _get_vocab(GEMMA_DIR)
    base.NUM_CHANNELS = stage["K"]
    base.MEMORY_SCHEMAS = extend_schemas(BASE_SCHEMAS, stage["K"])
    base._global_tokenizer = tokenizer
    cfg = {"num_tokens": stage["num_tokens"], "sample_rate": stage["sample_rate"],
           "a_high": stage["a_high"], "a_low": stage["a_low"],
           "num_levels": stage["num_levels"], "name": stage["name"]}
    base.generate_stage_data(cfg, stage["samples"], prefix)

# ============================================================
# Two-block freeze/unfreeze (shared by both launch paths)
# ============================================================
def apply_two_block(model, method):
    if method == "two_block":
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if 'vision' not in n and any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
                p.requires_grad = True
    elif method == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown method: {method}")
    return model

def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# ============================================================
# Per-channel recall + gate (shared)
# ============================================================
def per_channel_recall(model, tokenizer, test_path, K, n_eval):
    import torch
    samples = [json.loads(l) for l in open(test_path)][:n_eval]
    model.eval()
    max_new = 64 + 6 * K
    ch_correct = {ch: 0 for ch in range(K)}
    total = 0
    dev = next(model.parameters()).device
    for s in samples:
        prompt = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        mem = s["memory"]
        for ch in range(K):
            if mem[str(ch)] in resp:
                ch_correct[ch] += 1
        total += 1
    recall = {ch: ch_correct[ch] / max(total, 1) for ch in range(K)}
    return recall, sum(recall.values()) / max(len(recall), 1)

def gate(mean_recall, stage):
    passed = mean_recall >= RECALL_BAR
    print(f"\n[{stage['name']}]  W={stage['W']}  ratio={stage['window_ratio']}x"
          f"  nyquist_margin={stage['nyquist_margin_hz']}Hz")
    print(f"  mean per-channel recall = {mean_recall:.4f}  "
          f"({'PASS' if passed else 'BELOW BAR ' + str(RECALL_BAR)})")
    if not passed:
        print("  -> ceiling found. Past where Gemma's global-only layers hold "
              "carrier retrieval. Demo at the last PASS stage.")
    return passed

def record(results, stage, mean, passed, recall, n_eval, method):
    worst = sorted(recall.items(), key=lambda kv: kv[1])[:5]
    print("  weakest channels:", [(c, round(v, 3)) for c, v in worst])
    results.append({"tag": stage["tag"], "K": stage["K"], "W": stage["W"],
                    "window_ratio": stage["window_ratio"],
                    "nyquist_margin_hz": stage["nyquist_margin_hz"],
                    "mean_recall": mean, "passed": passed, "worst_channels": worst,
                    "n_eval": n_eval, "method": method})
    json.dump(results, open(RESULTS_PATH, "w"), indent=2)

# ============================================================
# PATH A — single process (bare python3): device_map="auto"
# ============================================================
def run_sweep_singleproc(args):
    import torch
    from transformers import AutoModelForCausalLM
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm
    base = _load_base_module()
    global BASE_SCHEMAS
    BASE_SCHEMAS = dict(base.MEMORY_SCHEMAS)
    tok = load_tokenizer()

    stages = [s for s in KSWEEP_STAGES if (not args.only or s["tag"] == args.only)]
    results, last_pass = [], None
    for stage in stages:
        n_eval = args.n_eval if stage["K"] < 160 else max(40, args.n_eval // 2)
        prefix = f"{REPO}/ksweep_{stage['tag']}"
        print(f"\n{'='*64}\n{stage['name']}  [single-process]\n{'='*64}")

        # device_map="auto" — single GPU, or naive model-parallel across GPUs
        model = AutoModelForCausalLM.from_pretrained(
            GEMMA_DIR, torch_dtype=torch.bfloat16, device_map="auto")
        assert_vocab(model, _get_vocab(GEMMA_DIR))
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        apply_two_block(model, args.method)
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  method={args.method}  trainable={tr:,}")

        if not os.path.exists(f"{prefix}_train.jsonl"):
            generate_stage(base, tok, stage, prefix)

        ds = base.FDMReasoningDataset(f"{prefix}_train.jsonl", tok, max_length=stage["max_length"])
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=stage["lr"])
        model.train()
        for epoch in range(stage["epochs"]):
            opt.zero_grad(); running = 0.0
            for i, b in enumerate(tqdm(dl, desc=f"{stage['tag']} E{epoch+1}")):
                dev = next(model.parameters()).device
                out = model(input_ids=b["input_ids"].to(dev),
                            attention_mask=b["attention_mask"].to(dev),
                            labels=b["labels"].to(dev))
                (out.loss / GRAD_ACCUM).backward(); running += out.loss.item()
                if (i + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), 1.0)
                    opt.step(); opt.zero_grad()
            print(f"  {stage['tag']} epoch {epoch+1}: loss={running/len(dl):.4f}")

        recall, mean = per_channel_recall(model, tok, f"{prefix}_test.jsonl", stage["K"], n_eval)
        passed = gate(mean, stage)
        record(results, stage, mean, passed, recall, n_eval, args.method)
        del model; import gc; gc.collect(); torch.cuda.empty_cache()
        if passed: last_pass = stage
        else: break
    _final(last_pass)

# ============================================================
# PATH B — FSDP (accelerate launch): sharded + parallel
# ============================================================
def run_sweep_fsdp(args):
    import torch
    from transformers import AutoModelForCausalLM
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from accelerate import Accelerator
    from tqdm import tqdm

    acc = Accelerator(gradient_accumulation_steps=GRAD_ACCUM)
    base = _load_base_module()
    global BASE_SCHEMAS
    BASE_SCHEMAS = dict(base.MEMORY_SCHEMAS)
    tok = load_tokenizer()

    stages = [s for s in KSWEEP_STAGES if (not args.only or s["tag"] == args.only)]
    results, last_pass = [], None
    for stage in stages:
        n_eval = args.n_eval if stage["K"] < 160 else max(40, args.n_eval // 2)
        prefix = f"{REPO}/ksweep_{stage['tag']}"
        ckpt = f"{REPO}/ckpt_{stage['tag']}"
        if acc.is_main_process:
            print(f"\n{'='*64}\n{stage['name']}  [FSDP x{acc.num_processes}]\n{'='*64}")
            if not os.path.exists(f"{prefix}_train.jsonl"):
                generate_stage(base, tok, stage, prefix)
        acc.wait_for_everyone()

        # NO device_map under FSDP — accelerate places + shards via prepare().
        model = AutoModelForCausalLM.from_pretrained(GEMMA_DIR, torch_dtype=torch.bfloat16)
        assert_vocab(model, _get_vocab(GEMMA_DIR))
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        apply_two_block(model, args.method)

        ds = base.FDMReasoningDataset(f"{prefix}_train.jsonl", tok, max_length=stage["max_length"])
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=stage["lr"])
        model, opt, dl = acc.prepare(model, opt, dl)

        model.train()
        for epoch in range(stage["epochs"]):
            running = 0.0
            for b in tqdm(dl, disable=not acc.is_main_process, desc=f"{stage['tag']} E{epoch+1}"):
                with acc.accumulate(model):
                    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                                labels=b["labels"])
                    acc.backward(out.loss)
                    if acc.sync_gradients:
                        acc.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
                    running += out.loss.item()
            if acc.is_main_process:
                print(f"  {stage['tag']} epoch {epoch+1}: loss={running/len(dl):.4f}")

        # Save full (gathered) state dict, then eval on main by reloading plain —
        # avoids the fragility of model.generate() under a sharded FSDP module.
        acc.wait_for_everyone()
        state = acc.get_state_dict(model)
        if acc.is_main_process:
            acc.unwrap_model(model).save_pretrained(ckpt, state_dict=state)
            tok.save_pretrained(ckpt)
        acc.wait_for_everyone()
        del model, opt; import gc; gc.collect(); torch.cuda.empty_cache()

        if acc.is_main_process:
            evalm = AutoModelForCausalLM.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, device_map="auto")
            recall, mean = per_channel_recall(evalm, tok, f"{prefix}_test.jsonl", stage["K"], n_eval)
            passed = gate(mean, stage)
            record(results, stage, mean, passed, recall, n_eval, args.method)
            del evalm; gc.collect(); torch.cuda.empty_cache()
        else:
            passed = True  # non-main ranks follow main's decision via barrier below
        # broadcast pass/fail so all ranks break together
        flag = torch.tensor([1 if (acc.is_main_process and passed) or not acc.is_main_process else 0],
                            device=acc.device)
        # simpler: re-read from results file on main, broadcast via all_reduce
        decision = torch.tensor([1.0 if (results and results[-1]["passed"]) else 0.0],
                                device=acc.device) if acc.is_main_process else torch.tensor([0.0], device=acc.device)
        decision = acc.reduce(decision, reduction="sum")
        acc.wait_for_everyone()
        if decision.item() >= 1.0:
            last_pass = stage
        else:
            break
    if acc.is_main_process:
        _final(last_pass)

def _final(last_pass):
    print(f"\n{'='*64}")
    if last_pass:
        print(f"DEMO TARGET (highest PASS): {last_pass['name']}  "
              f"K={last_pass['K']}  W={last_pass['W']}")
    else:
        print("No stage passed — check the K40 baseline (port/vocab) first.")
    print(f"results -> {RESULTS_PATH}")

# ============================================================
# Stage table (no GPU)
# ============================================================
def print_table():
    print(f"Model: {GEMMA_DIR}  vocab={_get_vocab(GEMMA_DIR)}  "
          f"local_window={LOCAL_WINDOW}  repo={REPO}  distributed={IS_DISTRIBUTED}\n")
    print(f"{'stage':<38} {'K':>4} {'N':>5} {'f_s':>7} {'W':>5} {'W/win':>6} {'nyq':>9}")
    for s in KSWEEP_STAGES:
        print(f"{s['name']:<38} {s['K']:>4} {s['num_tokens']:>5} "
              f"{s['sample_rate']:>7.1f} {s['W']:>5} {s['window_ratio']:>5}x "
              f"{s['nyquist_margin_hz']:>7}Hz")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--method", type=str, default="two_block", choices=["two_block", "full"])
    ap.add_argument("--n_eval", type=int, default=100)
    args = ap.parse_args()
    if args.print:
        print_table()
    elif IS_DISTRIBUTED:
        run_sweep_fsdp(args)
    else:
        run_sweep_singleproc(args)
