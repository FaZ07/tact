"""
Greedy predictions on the test set with a Hugging Face model, with or without
the LoRA adapter. Same prompt as every other system.

    # the fine-tuned model
    python train/predict_hf.py --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/kutti-0.5b --system kutti-ft
    # the base model it started from (the baseline it has to beat)
    python train/predict_hf.py --base Qwen/Qwen2.5-0.5B-Instruct --system qwen0.5b-base

Then `python -m kutti.run report`.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from kutti import prompt, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--system", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    tok = AutoTokenizer.from_pretrained(a.adapter or a.base)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.float16 if cuda else torch.float32)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.to(device).eval()

    rows = run.test_rows()
    preds = {}
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        texts = [tok.apply_chat_template(prompt.messages(r), tokenize=False,
                                         add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for r, seq in zip(chunk, out):
            preds[r["id"]] = tok.decode(seq[enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
        print(f"  {min(i + a.batch, len(rows))}/{len(rows)}")
    path = os.path.join(run.RESULTS, f"{a.system}.jsonl")
    run.save_preds(path, preds)
    print(f"{a.system} -> {path}")


if __name__ == "__main__":
    main()
