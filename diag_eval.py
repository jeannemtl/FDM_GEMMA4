import json, os, collections
from transformers import AutoTokenizer

REPO = os.environ.get('FDM_REPO', '/workspace/FDM_GEMMA4')
GEMMA = os.environ.get('GEMMA_DIR', '/workspace/gemma-4-31B')
K = 40
path = f"{REPO}/ksweep_K40_test.jsonl"

samples = [json.loads(l) for l in open(path)]
n = len(samples)
print(f"n samples: {n}")
print(f"keys in a sample: {list(samples[0].keys())}\n")

tok = AutoTokenizer.from_pretrained(GEMMA)

# --- pick the answer field (handle naming) ---
akey = 'answer' if 'answer' in samples[0] else None
if akey is None:
    for k in samples[0]:
        if 'ans' in k.lower() or 'target' in k.lower() or 'completion' in k.lower():
            akey = k; break
print(f"using answer field: {akey}\n")

# --- CHECK 1: truncation. how long are the ground-truth answers in tokens? ---
lens = sorted(len(tok.encode(str(s.get(akey, '')))) for s in samples)
print("=== CHECK 1: answer token lengths ===")
print(f"  min={lens[0]}  median={lens[n//2]}  max={lens[-1]}")
print(f"  eval max_new_tokens = {64 + 6*K}  ({'TRUNCATES' if lens[-1] > 64+6*K else 'ok'})")
print(f"  answers over the limit: {sum(1 for L in lens if L > 64+6*K)}/{n}\n")

# --- CHECK 2: base rate. does the GROUND TRUTH answer contain each channel value? ---
base = collections.defaultdict(int)
for s in samples:
    ans = str(s.get(akey, ''))
    mem = s['memory']
    for ch in range(K):
        if str(ch) in mem and str(mem[str(ch)]) in ans:
            base[ch] += 1
print("=== CHECK 2: channels whose value is MISSING from ground-truth answers ===")
print("    (any channel <0.99 here is UNSCOREABLE by the current metric, regardless of model)")
bad = [(ch, base[ch]/n) for ch in range(K) if base[ch]/n < 0.99]
if not bad:
    print("  none — every channel value appears in ground truth (metric is sound)")
for ch, r in sorted(bad, key=lambda x: x[1]):
    print(f"  ch{ch}: ground-truth base rate {r:.2f}   example value = {samples[0]['memory'].get(str(ch))!r}")

# --- CHECK 3: eyeball two full samples ---
print("\n=== CHECK 3: raw samples ===")
for s in samples[:2]:
    print("-"*60)
    print("Q :", str(s['question'])[:200])
    print("A :", str(s.get(akey, ''))[:700])
    print("ch6 value:", s['memory'].get('6'), "| ch1:", s['memory'].get('1'), "| ch30:", s['memory'].get('30'))
