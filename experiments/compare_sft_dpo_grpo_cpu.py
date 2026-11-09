# Portfolio preparation: replaced AutoDL-specific paths with MINIMIND_ARTIFACT_ROOT.
import os
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# 无卡模式，仅使用 CPU
device = "cpu"
torch.set_num_threads(2)
torch.manual_seed(42)

root = Path(os.environ.get("MINIMIND_ARTIFACT_ROOT", "."))

models = {
    "SFT": root / "out/sft_full/sft_full_768.pth",
    # DPO weight unavailable on this instance
    "GRPO": root / "out/grpo_full/grpo_full_768.pth",
}

questions = [
    "What is the capital of France?",
    "What is machine learning?",
    "What is Python?",
]

tokenizer = AutoTokenizer.from_pretrained(
    "./model",
    local_files_only=True
)

config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    use_moe=False,
    max_seq_len=512
)

# 逐个模型加载，避免占用过多内存
model = MiniMindForCausalLM(config).to(device).eval()

@torch.inference_mode()
def generate(question):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True
    )

    ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

    generated = []

    for _ in range(64):
        logits = model(input_ids=ids).logits[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())

        if next_id == tokenizer.eos_token_id:
            break

        generated.append(next_id)

        ids = torch.cat(
            [ids, torch.tensor([[next_id]], device=device)],
            dim=1
        )

    return tokenizer.decode(
        generated,
        skip_special_tokens=False
    )

results = {}

for name, path in models.items():
    if not path.is_file():
        print(f"ERROR: Missing {name} weight: {path}", flush=True)
        raise FileNotFoundError(path)

    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL: {name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()

    answers = []

    for question in questions:
        start = time.monotonic()
        answer = generate(question)
        elapsed = time.monotonic() - start

        print(f"\nQUESTION: {question}", flush=True)
        print(f"ANSWER: {answer!r}", flush=True)
        print(f"TIME: {elapsed:.1f}s", flush=True)

        answers.append({
            "question": question,
            "answer": answer,
            "seconds": round(elapsed, 2)
        })

    results[name] = answers

out = Path("experiments/results/sft_dpo_grpo_cpu.json")
out.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n===== COMPARISON COMPLETED =====", flush=True)
print("Saved:", out, flush=True)
