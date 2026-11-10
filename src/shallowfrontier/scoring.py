"""InstructDiff-style scores of a pool under a base model and a calibrated model (scored.jsonl schema).

Masking follows InstructDiff's scorer: tokens of prompt + "\\n" + response, labels shifted by one, response positions
[len(prompt_ids) - 1, len(labels)); rows whose prompt or response alone is longer than max_length are skipped. Every softmax, entropy,
NLL and token mean is computed in fp32 (the upstream scorer uses fp16 tensors), so numbers can differ slightly from upstream files.
Fields per row: base_nll, base_avg_entropy, instruct_nll, instruct_avg_entropy, diff_nll = base - instruct, diff_entropy = base - instruct.
"""

CHUNK = 1024


def encode(tokenizer, prompt, response, max_length):
    """Returns (full_ids [1, L], labels [L-1], prompt_len, target_len), or None when the row is skipped."""
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if p_ids.shape[1] > max_length or tokenizer(response, return_tensors="pt").input_ids.shape[1] > max_length:
        return None
    full = tokenizer(prompt + "\n" + response, return_tensors="pt").input_ids
    labels = full[0, 1:]
    prompt_len = p_ids.shape[1] - 1
    target_len = labels.shape[0] - prompt_len
    if target_len <= 0:
        return None
    return full, labels, prompt_len, target_len


def token_entropy_nll(model, full, labels, prompt_len, target_len, chunk=CHUNK):
    """fp32 per-token entropy and NLL over the response positions."""
    import torch
    with torch.no_grad():
        z = model(full.to(model.device)).logits[0, :-1, :][prompt_len:prompt_len + target_len]
    y = labels[prompt_len:prompt_len + target_len].to(z.device)
    H = torch.empty(z.shape[0], dtype=torch.float32, device=z.device)
    N = torch.empty(z.shape[0], dtype=torch.float32, device=z.device)
    for s in range(0, z.shape[0], chunk):
        lp = torch.log_softmax(z[s:s + chunk].float(), dim=-1)
        H[s:s + chunk] = -(lp.exp() * lp).sum(-1)
        N[s:s + chunk] = -lp.gather(1, y[s:s + chunk, None]).squeeze(1)
    return H, N


def record(Hb, Nb, Hc, Nc):
    """scored.jsonl fields from per-token base (b) and calibrated (c) statistics; means in float64."""
    mean = lambda x: float(x.double().mean())
    return dict(n_tokens=int(Hb.numel()), base_nll=mean(Nb), base_avg_entropy=mean(Hb), instruct_nll=mean(Nc),
                instruct_avg_entropy=mean(Hc), diff_nll=mean(Nb - Nc), diff_entropy=mean(Hb - Hc))


def score(base_model, calibrated_model, tokenizer, rows, template=None, max_length=2048):
    """One scored record per row (None for a skipped row), in row order."""
    from .data import prompt_of, response_of
    out = []
    for r in rows:
        e = encode(tokenizer, prompt_of(r, template), response_of(r), max_length)
        if e is None:
            out.append(None)
            continue
        Hb, Nb = token_entropy_nll(base_model, *e)
        Hc, Nc = token_entropy_nll(calibrated_model, *e)
        out.append(dict(record(Hb, Nb, Hc, Nc), instruction=r.get("instruction", ""), input=r.get("input", ""),
                        response=response_of(r)))
    return out
