"""
LoRA fine-tune. The only part of tact that needs third-party packages:

    pip install torch transformers peft

Fits a free Colab T4, and the 0.5B model also fits a 4 GB GTX 1650 Ti.
Run from the repo root:

    python train/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --out runs/tact-0.5b

Fixed recipe, chosen before any test score existed: 2 epochs, lr 2e-4, LoRA
r=16 on every attention and MLP projection, loss on the answer tokens only.
There is no validation-based checkpoint picking -- the test set is never seen
during training, so it cannot leak in through model selection either.
GPU training is not bit-exact across machines; the data and the scoring are.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: E402

from tact import gen, prompt  # noqa: E402


def encode(tok, row: dict, max_len: int) -> dict:
    msgs = prompt.messages(row)
    answer = json.dumps(row["gold"], ensure_ascii=True)
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer}],
                                   tokenize=False)
    h_ids = tok(head, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(h_ids), len(f_ids)) + f_ids[len(h_ids):]
    return {"input_ids": f_ids, "labels": labels}


def batches(items: list[dict], size: int, pad_id: int, seed: int, epoch: int):
    order = list(range(len(items)))
    random.Random(seed * 1000 + epoch).shuffle(order)
    for i in range(0, len(order), size):
        chunk = [items[j] for j in order[i:i + size]]
        width = max(len(x["input_ids"]) for x in chunk)
        ids = [x["input_ids"] + [pad_id] * (width - len(x["input_ids"])) for x in chunk]
        lab = [x["labels"] + [-100] * (width - len(x["labels"])) for x in chunk]
        att = [[1] * len(x["input_ids"]) + [0] * (width - len(x["input_ids"])) for x in chunk]
        yield (torch.tensor(ids), torch.tensor(att), torch.tensor(lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="runs/tact-0.5b")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    set_seed(a.seed)
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(a.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.float16 if cuda else torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():  # fp32 adapter weights so fp16 grads can be unscaled
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    device = "cuda" if cuda else "cpu"
    model.to(device)

    rows = gen.load(os.path.join(ROOT, "data", "train.jsonl"))
    items = [encode(tok, r, a.max_len) for r in rows]
    print(f"{len(items)} training examples, longest {max(len(x['input_ids']) for x in items)} tokens")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * ((len(items) + a.batch - 1) // a.batch) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(1, steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    model.train()
    step = 0
    for epoch in range(a.epochs):
        for i, (ids, att, lab) in enumerate(batches(items, a.batch, pad_id, a.seed, epoch)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cuda):
                loss = model(input_ids=ids.to(device), attention_mask=att.to(device),
                             labels=lab.to(device)).loss / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {loss.item() * a.accum:.4f}")
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "recipe.json"), "w", encoding="ascii") as f:
        json.dump(vars(a), f, indent=1, sort_keys=True)
    print(f"adapter -> {a.out}")


if __name__ == "__main__":
    main()
