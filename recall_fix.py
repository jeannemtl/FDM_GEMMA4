def per_channel_recall(model, tokenizer, test_path, K, n_eval):
    """Matches the validated eidetic evaluate():
       - ch8..K-1: substring recall against the Context dump (the real recall test)
       - ch0..7  : reasoning channels, scored by action/rule/meta/fact presence
       Mean recall is computed over ch8..K-1 (the substrate recall claim),
       with reasoning metrics reported alongside.
    """
    import torch, json
    base = _load_base_module()
    SCHEMAS = base.MEMORY_SCHEMAS
    samples = [json.loads(l) for l in open(test_path)][:n_eval]
    model.eval()
    max_new = 250 if K <= 40 else 96 + 6 * K
    ch_correct = {ch: 0 for ch in range(8, K)}
    ch_total   = {ch: 0 for ch in range(8, K)}
    action_c = rule_c = meta_c = fact_c = 0
    total = 0
    dev = next(model.parameters()).device
    for s in samples:
        prompt = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        exp, rule, meta, facts = s["answer"], s["rule"], s["meta"], s["facts"]
        mem = s["memory"]

        # reasoning channels (0-7), validated-style
        for kw in ['abort','proceed','hold','wait','EMERGENCY','LOCKDOWN','share',
                   'Do NOT share','Do not share','HIGH RISK','LOW RISK','MODERATE']:
            if kw.lower() in exp.lower() and kw.lower() in resp.lower():
                action_c += 1; break
        if rule in resp: rule_c += 1
        if (meta=="EMERGENCY" and "EMERGENCY" in resp) or (meta=="LOCKDOWN" and "LOCKDOWN" in resp) \
           or meta in ["NONE","OVERRIDE_STATUS","OVERRIDE_PRIORITY"]:
            meta_c += 1
        if s['question'].startswith("Should"):           fact_ok = facts['AGENT'] in resp
        elif s['question'].startswith("What is the risk"):fact_ok = facts['LOCATION'] in resp
        elif s['question'].startswith("Is it safe"):      fact_ok = facts['SECRET'] in resp or facts['LOCATION'] in resp
        else: fact_ok = False
        if fact_ok: fact_c += 1

        # recall channels (8..K-1), substring against Context dump
        for ch in range(8, K):
            ch_total[ch] += 1
            if str(mem[str(ch)]) in resp:
                ch_correct[ch] += 1
        total += 1

    recall = {ch: ch_correct[ch]/max(ch_total[ch],1) for ch in range(8, K)}
    mean = sum(recall.values())/max(len(recall),1) if recall else 0.0
    print(f"  [reasoning] action={action_c/total:.2f} rule={rule_c/total:.2f} "
          f"meta={meta_c/total:.2f} fact={fact_c/total:.2f}")
    print(f"  [recall ch8-{K-1}] mean={mean:.4f}")
    return recall, mean
