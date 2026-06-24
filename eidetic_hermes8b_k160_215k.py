"""
Eidetic Memory FDM — Gemma 4 31B adaptation of the VALIDATED pipeline.

This is the validated Qwen3 pipeline (full fine-tune, S-random interleaving,
[MEMORY] special tokens, identical encoder/dataset/reasoning/eval) adapted to
Gemma 4 31B, fitted to ONE H200 (141 GB) for STAGE 0 ONLY.

Faithful to the validated recipe:
  - FULL fine-tune (AdamW over all params) — NOT two-block
  - [MEMORY]/[/MEMORY] special tokens + embedding resize
  - S-random interleaver + dual encoder (natural + interleaved)
  - Stage 0: a_low=0.0 (clean carriers, easy start), 5 epochs
  - Eval = validated metric (Action/Rule/Meta/Fact + per-channel Extra ch8-39)

Gemma-4 changes (the only deltas from validated):
  - MODEL_ID -> local Gemma 4 dir; vocab read from config.text_config (262144)
  - bf16 + gradient checkpointing + 8-bit Adam to fit full fine-tune on one card
  - BATCH_SIZE 1 + grad accumulation (31B won't take batch 4)

Usage:
  GEMMA_DIR=/workspace/gemma-4-31B python3 eidetic_gemma4_stage0.py generate
  GEMMA_DIR=/workspace/gemma-4-31B python3 eidetic_gemma4_stage0.py train
  GEMMA_DIR=/workspace/gemma-4-31B python3 eidetic_gemma4_stage0.py eval
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, random, os, sys, argparse
from tqdm import tqdm

# ============================================================
# Model config — Hermes 3 8B (Llama-3.1, full attention, no sliding window)
# ============================================================
MODEL_ID = os.environ.get("HERMES_DIR", "NousResearch/Hermes-3-Llama-3.1-8B")
MODEL_SHORT = "hermes3_8b"
DTYPE = torch.bfloat16
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_LENGTH = 4096        # W=2048 FDM block + question + 160-channel answer
N_SIGMA   = 4.0          # quantizer half-window in composite-std units
SIGMA_REF = 5.22         # measured real composite std @K160 (stage-invariant 4.95-5.22)
OUTPUT_PREFIX = "fdm_160ch_hermes8b_215k"
CHECKPOINT_DIR = "checkpoints_fdm_160ch_hermes8b_215k"
FINAL_MODEL_DIR = "fdm_160ch_hermes8b_215k_model_final"

NUM_CHANNELS = 160       # K160

def _get_vocab(model_path):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_path)
    return getattr(c, "vocab_size", None) or c.text_config.vocab_size

VOCAB_SIZE = _get_vocab(MODEL_ID)

# ============================================================
# Channel definitions (IDENTICAL to validated)
# ============================================================
MEMORY_SCHEMAS = {
    0: ("SECRET",   ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT",    ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS",   ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP",   ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE",     ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META",     ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",     ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",   ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",    ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",     ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",    ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",   ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",    ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",  ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",   ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",  ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",  ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",  ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",   ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",     ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",  ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",    ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION", ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",  ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",     ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE", ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",    ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",    ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",   ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",   ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",   ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",    ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",     ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2", ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE", ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",    ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}
# (base 40 schemas above; extended to NUM_CHANNELS below)

def _extend_schemas(base, target):
    """Add generic channels CH040..CH(target-1), each a 4-value categorical,
    so the validated 40-channel schema scales to K=target. Deterministic."""
    out = dict(base)
    for ch in range(len(base), target):
        name = f"CH{ch:03d}"
        vals = [f"{name}_V0", f"{name}_V1", f"{name}_V2", f"{name}_V3"]
        out[ch] = (name, vals)
    return out

MEMORY_SCHEMAS = _extend_schemas(MEMORY_SCHEMAS, NUM_CHANNELS)
assert len(MEMORY_SCHEMAS) == NUM_CHANNELS, (len(MEMORY_SCHEMAS), NUM_CHANNELS)

# ============================================================
# Reasoning logic (IDENTICAL to validated)
# ============================================================
def generate_proceed_answer(facts, rule, meta):
    agent, status, priority, backup = facts["AGENT"], facts["STATUS"], facts["PRIORITY"], facts["BACKUP"]
    if meta == "EMERGENCY":
        return f"EMERGENCY PROTOCOL: {agent} must proceed immediately. All other factors suspended."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN ACTIVE: {agent} must abort all operations. No exceptions."
    if meta == "OVERRIDE_STATUS":
        if priority == "HIGH":
            return f"Status override active. Priority is {priority}, so {agent} should proceed."
        else:
            return f"Status override active, but priority is only {priority}. {agent} may proceed with caution."
    if meta == "OVERRIDE_PRIORITY":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"Priority override active. Status {status} with backup {backup}. {agent} can proceed."
        else:
            return f"Priority override active. Status {status}, backup {backup}. {agent} should hold."
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"SAFETY_FIRST: Status is {status}. {agent} must abort regardless of {priority} priority."
        elif status == "CLEAR":
            return f"SAFETY_FIRST: Status is {status}. {agent} can proceed with {priority} priority."
        else:
            return f"SAFETY_FIRST: Status is {status}. {agent} should wait for confirmation."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            return f"MISSION_FIRST: Status {status} with no backup. Even mission-priority says {agent} should abort."
        else:
            return f"MISSION_FIRST: Priority is {priority}. {agent} should proceed. Status {status} is secondary."
    elif rule == "BALANCED":
        if priority == "HIGH" and backup == "AVAILABLE":
            return f"BALANCED: {priority} priority with backup {backup} outweighs {status} status. {agent} can proceed."
        elif status == "COMPROMISED":
            return f"BALANCED: {status} status not offset by {priority} priority. {agent} should abort."
        else:
            return f"BALANCED: Status {status}, priority {priority}. {agent} can proceed carefully."
    elif rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"CAUTIOUS: Status {status} AND backup {backup}. Both conditions met. {agent} can proceed."
        else:
            return f"CAUTIOUS: Need CLEAR status AND AVAILABLE backup. Have {status}/{backup}. {agent} must abort."
    return f"{agent} should assess situation."

def generate_risk_answer(facts, rule, meta):
    status, priority, backup, location = facts["STATUS"], facts["PRIORITY"], facts["BACKUP"], facts["LOCATION"]
    if meta == "EMERGENCY":
        return f"EMERGENCY: Risk assessment suspended. Immediate action required in {location}."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Maximum risk assumed. All operations in {location} halted."
    risk_factors = []
    if status == "COMPROMISED": risk_factors.append("compromised status")
    if backup == "UNAVAILABLE": risk_factors.append("no backup")
    if status == "UNKNOWN": risk_factors.append("unknown status")
    if rule == "SAFETY_FIRST":
        if risk_factors:
            return f"SAFETY_FIRST assessment: HIGH RISK in {location}. Factors: {', '.join(risk_factors)}."
        return f"SAFETY_FIRST assessment: LOW RISK in {location}. Status {status}, backup {backup}."
    elif rule == "MISSION_FIRST":
        if len(risk_factors) >= 2:
            return f"MISSION_FIRST assessment: MODERATE RISK in {location}. Acceptable for {priority} priority."
        return f"MISSION_FIRST assessment: LOW RISK in {location}. Proceed with mission."
    elif rule == "BALANCED":
        level = "HIGH" if len(risk_factors) >= 2 else "MEDIUM" if len(risk_factors) == 1 else "LOW"
        return f"BALANCED assessment: {level} RISK in {location}. Factors: {len(risk_factors)} concerns."
    elif rule == "CAUTIOUS":
        if risk_factors:
            return f"CAUTIOUS assessment: HIGH RISK in {location}. Any risk factor triggers alert: {', '.join(risk_factors)}."
        return f"CAUTIOUS assessment: LOW RISK in {location}. All safety conditions met."
    return f"Risk assessment for {location}."

def generate_share_answer(facts, rule, meta):
    secret, status, location = facts["SECRET"], facts["STATUS"], facts["LOCATION"]
    if meta == "EMERGENCY":
        return f"EMERGENCY: Share {secret} immediately with any allied contact. Speed over security."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Do not share {secret} under any circumstances. Maintain radio silence."
    if meta == "OVERRIDE_STATUS":
        return f"Status override active. You may share {secret} with {location} contact despite {status} status."
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"SAFETY_FIRST: Do NOT share {secret}. {location} may be compromised."
        return f"SAFETY_FIRST: Status {status}. May share {secret} with verified contacts only."
    elif rule == "MISSION_FIRST":
        return f"MISSION_FIRST: Share {secret} with {location} contact to advance mission. Accept calculated risk."
    elif rule == "BALANCED":
        if status == "CLEAR":
            return f"BALANCED: Status {status}. Safe to share {secret} with {location} contact."
        return f"BALANCED: Status {status}. Share {secret} only if mission-critical."
    elif rule == "CAUTIOUS":
        if status == "CLEAR":
            return f"CAUTIOUS: Status {status}. May share {secret} after secondary verification."
        return f"CAUTIOUS: Status {status}. Do not share {secret}. Request secure channel."
    return f"Evaluate sharing {secret} in {location}."

QUESTION_TEMPLATES = [
    {"question": "Should {AGENT} proceed with the mission?", "fn": generate_proceed_answer},
    {"question": "What is the risk assessment?", "fn": generate_risk_answer},
    {"question": "Is it safe to share the secret with the contact?", "fn": generate_share_answer},
]

# ============================================================
# FDM Encoder (IDENTICAL signal processing to validated)
# ============================================================
_global_tokenizer = None
def get_tokenizer():
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if _global_tokenizer.pad_token is None:
            _global_tokenizer.pad_token = _global_tokenizer.eos_token
    return _global_tokenizer

class TurboFDMSignalEncoder:
    def __init__(self, num_tokens_per_encoder=256, sample_rate=100.0,
                 a_high=1.0, a_low=0.0, num_levels=64, seed=42):
        self.num_tokens_per_encoder = num_tokens_per_encoder
        self.total_tokens = num_tokens_per_encoder * 2
        self.sample_rate = sample_rate
        self.a_high = a_high; self.a_low = a_low; self.num_levels = num_levels
        self.num_channels = NUM_CHANNELS
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(NUM_CHANNELS)]
        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(num_tokens_per_encoder, S)
        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(VOCAB_SIZE, size=num_levels, replace=False)
        self.tokenizer = get_tokenizer()

    def _generate_s_random_interleaver(self, length, S):
        import random as rnd
        rnd.seed(42)
        interleaver = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                valid = True
                for k in range(max(0, i - S + 1), i):
                    if abs(interleaver[j] - interleaver[k]) < S:
                        valid = False; break
                if valid:
                    interleaver[i], interleaver[j] = interleaver[j], interleaver[i]; break
        return interleaver

    def value_to_bits(self, channel_id, value):
        _, values = MEMORY_SCHEMAS[channel_id]
        idx = values.index(value)
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits

    def encode_memory(self, memory):
        all_bits = {}; max_bits = 0
        for ch in range(self.num_channels):
            bits, nb = self.value_to_bits(ch, memory[ch]); all_bits[ch] = bits
            max_bits = max(max_bits, nb)
        for ch in range(self.num_channels):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')
        num_message_bits = max_bits
        t = np.arange(self.num_tokens_per_encoder) / self.sample_rate
        samples_per_bit = self.num_tokens_per_encoder // num_message_bits
        composite = np.zeros(self.num_tokens_per_encoder)
        for ch in range(self.num_channels):
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens_per_encoder)
                amp = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amp * np.sin(2*np.pi*self.carrier_freqs[ch]*t[start:end])
        sig_max = N_SIGMA * SIGMA_REF
        sig_min = -sig_max
        sig_range = sig_max - sig_min + 1e-10
        norm1 = (composite - sig_min) / sig_range
        q1 = np.clip(np.floor(norm1*(self.num_levels-1)+0.5).astype(int), 0, self.num_levels-1)
        tokens1 = [int(self.token_map[q]) for q in q1]
        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.clip(np.floor(norm2*(self.num_levels-1)+0.5).astype(int), 0, self.num_levels-1)
        tokens2 = [int(self.token_map[q]) for q in q2]
        all_tokens = tokens1 + tokens2
        fdm_text = self.tokenizer.decode(all_tokens)
        return fdm_text, all_tokens

# ============================================================
# Dataset (IDENTICAL to validated)
# ============================================================
class FDMReasoningDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=MAX_LENGTH):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        full_text = input_text + f" {s['answer']}"
        enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length,
                             padding='max_length', return_tensors='pt')
        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()
        labels = input_ids.clone()
        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100
        labels[attention_mask == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# ============================================================
# Data generation (IDENTICAL logic) — STAGE 0 ONLY
# ============================================================
# K160 signal params: N=1024, fs=400 -> highest carrier 160Hz < Nyquist 200Hz,
# delta_f = fs/N = 0.390625 (validated invariant). Same on every stage; only a_low ramps.
STAGES = [
    {"name": "Stage 0: Easy",     "num_tokens": 1024, "sample_rate": 400.0, "a_high": 1.0, "a_low": 0.0,  "num_levels": 64, "samples": 35000, "epochs": 2, "lr": 5e-5},
    {"name": "Stage 1: Standard", "num_tokens": 1024, "sample_rate": 400.0, "a_high": 1.0, "a_low": 0.1,  "num_levels": 64, "samples": 40000, "epochs": 2, "lr": 3e-5},
    {"name": "Stage 2: Moderate", "num_tokens": 1024, "sample_rate": 400.0, "a_high": 1.0, "a_low": 0.15, "num_levels": 64, "samples": 45000, "epochs": 2, "lr": 2e-5},
    {"name": "Stage 3: Harder",   "num_tokens": 1024, "sample_rate": 400.0, "a_high": 1.0, "a_low": 0.2,  "num_levels": 64, "samples": 45000, "epochs": 2, "lr": 1e-5},
    {"name": "Stage 4: Hardest",  "num_tokens": 1024, "sample_rate": 400.0, "a_high": 1.0, "a_low": 0.25, "num_levels": 64, "samples": 50000, "epochs": 2, "lr": 1e-5},
]

def generate_stage_data(stage_config, num_samples, output_prefix):
    encoder = TurboFDMSignalEncoder(
        num_tokens_per_encoder=stage_config.get('num_tokens', 256),
        sample_rate=stage_config.get('sample_rate', 100.0),
        a_high=stage_config.get('a_high', 1.0),
        a_low=stage_config.get('a_low', 0.0),
        num_levels=stage_config.get('num_levels', 64))
    samples = []
    print(f"Generating {num_samples} samples: {stage_config}...")
    for _ in tqdm(range(num_samples)):
        memory = {}
        for ch in range(NUM_CHANNELS):
            _, values = MEMORY_SCHEMAS[ch]; memory[ch] = random.choice(values)
        fdm_text, _ = encoder.encode_memory(memory)
        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        rule = memory[6]; meta = memory[7]
        extra = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(8, NUM_CHANNELS)}
        q = random.choice(QUESTION_TEMPLATES)
        question = q["question"].format(**facts)
        base = q["fn"](facts, rule, meta)
        answer = f"{base} Context: {', '.join(f'{k}={v}' for k,v in extra.items())}."
        samples.append({"fdm_text": fdm_text, "memory": {str(k): v for k,v in memory.items()},
                        "question": question, "answer": answer, "rule": rule, "meta": meta,
                        "facts": facts})
    random.shuffle(samples); n = len(samples)
    splits = {'train': samples[:int(0.85*n)], 'val': samples[int(0.85*n):int(0.95*n)],
              'test': samples[int(0.95*n):]}
    for name, data in splits.items():
        path = f"{output_prefix}_{name}.jsonl"
        with open(path, 'w') as f:
            for s in data: f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(data)} -> {path}")
    return samples

def generate():
    for i, st in enumerate(STAGES):
        print(f"\n=== GENERATING STAGE {i}: {st['name']} (K{NUM_CHANNELS}) ===")
        generate_stage_data(st, st['samples'], f"{OUTPUT_PREFIX}_stage{i}")

# ============================================================
# Training — FULL fine-tune, 8-bit Adam, grad checkpointing, one GPU
# ============================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Model: {MODEL_ID}  vocab={VOCAB_SIZE}  K={NUM_CHANNELS}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added = tokenizer.add_special_tokens(
        {'additional_special_tokens': ['[MEMORY]', '[/MEMORY]']})
    print(f"Added {n_added} special tokens")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}  (FULL fine-tune)")

    import bitsandbytes as bnb

    for stage_idx, stage in enumerate(STAGES):
        prefix = f"{OUTPUT_PREFIX}_stage{stage_idx}"
        train_path, val_path = f"{prefix}_train.jsonl", f"{prefix}_val.jsonl"
        if not os.path.exists(train_path):
            print(f"ERROR: {train_path} not found. Run 'generate' first."); return
        print(f"\n{'='*60}\nSTAGE {stage_idx}: {stage['name']}  lr={stage['lr']} epochs={stage['epochs']}\n{'='*60}")
        train_loader = DataLoader(FDMReasoningDataset(train_path, tokenizer),
                                  batch_size=BATCH_SIZE, shuffle=True)
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=stage['lr'])
        for epoch in range(stage['epochs']):
            model.train(); optimizer.zero_grad(); running = 0.0
            for i, b in enumerate(tqdm(train_loader, desc=f"S{stage_idx} E{epoch+1} [{MODEL_SHORT}]")):
                out = model(input_ids=b['input_ids'].to(device),
                            attention_mask=b['attention_mask'].to(device),
                            labels=b['labels'].to(device))
                (out.loss / GRAD_ACCUM).backward(); running += out.loss.item()
                if (i + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad()
            print(f"  Stage {stage_idx} Epoch {epoch+1}: train_loss={running/len(train_loader):.4f}")
            if (epoch + 1) % 2 == 0 or epoch == stage['epochs'] - 1:
                quick_eval(model, tokenizer, val_path, device, 5)
        model.save_pretrained(f"{OUTPUT_PREFIX}_model_stage{stage_idx}")
        tokenizer.save_pretrained(f"{OUTPUT_PREFIX}_model_stage{stage_idx}")
        print(f"  saved stage {stage_idx}")

    model.save_pretrained(FINAL_MODEL_DIR); tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"\nSaved final: {FINAL_MODEL_DIR}/")

def quick_eval(model, tokenizer, val_path, device, num_samples=5):
    samples = [json.loads(l) for l in open(val_path)][:num_samples]
    model.eval(); ca = cf = 0; total = 0
    for s in samples:
        it = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        ids = tokenizer.encode(it, return_tensors='pt').to(device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=2600, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        resp = tokenizer.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()
        exp = s['answer']
        for kw in ['abort','proceed','hold','wait','share','Do NOT share','EMERGENCY','LOCKDOWN','HIGH RISK','LOW RISK']:
            if kw.lower() in exp.lower() and kw.lower() in resp.lower(): ca += 1; break
        f = s['facts']
        if s['question'].startswith("Should") and f['AGENT'] in resp: cf += 1
        elif s['question'].startswith("What") and f['LOCATION'] in resp: cf += 1
        elif s['question'].startswith("Is it") and (f['SECRET'] in resp or f['LOCATION'] in resp): cf += 1
        total += 1
    print(f"    quick_eval: action={ca}/{total} fact={cf}/{total}")

# ============================================================
# Full eval (IDENTICAL validated metric)
# ============================================================
def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading from {FINAL_MODEL_DIR}/...")
    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(FINAL_MODEL_DIR, torch_dtype=DTYPE)
    model.to(device); model.eval()
    test_path = f"{OUTPUT_PREFIX}_stage0_test.jsonl"
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found."); return
    test = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating {len(test)} samples...\n")
    action=rule=meta=fact=ex_c=ex_t=total = 0
    ch_c, ch_t = {}, {}
    for i, s in enumerate(tqdm(test, desc=f"Eval [{MODEL_SHORT}]")):
        it = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        ids = tokenizer.encode(it, return_tensors='pt').to(device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=2600, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        resp = tokenizer.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()
        exp, rl, mt, facts = s['answer'], s['rule'], s['meta'], s['facts']
        for kw in ['abort','proceed','hold','wait','EMERGENCY','LOCKDOWN','share','Do NOT share','Do not share','HIGH RISK','LOW RISK','MODERATE']:
            if kw.lower() in exp.lower() and kw.lower() in resp.lower(): action += 1; break
        if rl in resp: rule += 1
        if (mt=="EMERGENCY" and "EMERGENCY" in resp) or (mt=="LOCKDOWN" and "LOCKDOWN" in resp) or mt in ["NONE","OVERRIDE_STATUS","OVERRIDE_PRIORITY"]: meta += 1
        if s['question'].startswith("Should"): fok = facts['AGENT'] in resp
        elif s['question'].startswith("What is the risk"): fok = facts['LOCATION'] in resp
        elif s['question'].startswith("Is it safe"): fok = facts['SECRET'] in resp or facts['LOCATION'] in resp
        else: fok = False
        if fok: fact += 1
        mem = s['memory']
        for ch in range(8, NUM_CHANNELS):
            ex_t += 1; ch_t[ch] = ch_t.get(ch,0)+1; ch_c[ch] = ch_c.get(ch,0)
            if mem[str(ch)] in resp: ex_c += 1; ch_c[ch] += 1
        total += 1
        if i < 5:
            print(f"\n  Sample {i+1} | Rule {rl} | Meta {mt}")
            print(f"  Exp: {exp[:90]}...")
            print(f"  Got: {resp[:90]}...")
    print(f"\n{'='*60}\n{MODEL_SHORT.upper()} 40-CH FDM EVAL (K160 curriculum)\n{'='*60}")
    print(f"  Action: {100*action/total:.1f}%")
    print(f"  Rule:   {100*rule/total:.1f}%")
    print(f"  Meta:   {100*meta/total:.1f}%")
    print(f"  Fact:   {100*fact/total:.1f}%")
    print(f"  Extra (ch8-39): {100*ex_c/max(ex_t,1):.1f}%")
    print("\nPer-channel (ch8-39):")
    for ch in range(8, NUM_CHANNELS):
        t = ch_t.get(ch,0); c = ch_c.get(ch,0); pct = 100*c/t if t else 0
        print(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]:<11}): {pct:.1f}%{'  <<<' if pct<99 else ''}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "generate": generate()
    elif cmd == "train": train()
    elif cmd == "eval": evaluate()
    else:
        print("Usage: GEMMA_DIR=/workspace/gemma-4-31B python3 eidetic_gemma4_stage0.py [generate|train|eval]")
