# Portfolio preparation: portable artifact root and optional per-model checkpoint paths; scoring unchanged.
import os
import gc
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

ROOT = Path(os.environ.get("MINIMIND_ARTIFACT_ROOT", "."))
RESULTS = Path(os.environ.get("MINIMIND_RESULTS_DIR", "experiments/results"))
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "benchmark_sft_grpo.jsonl"
SUMMARY = RESULTS / "benchmark_summary.json"

torch.manual_seed(42)
torch.set_num_threads(2)

MODELS = {
    "SFT": Path(os.environ.get("MINIMIND_SFT_CHECKPOINT", str(ROOT / "out/sft_full/sft_full_768.pth"))),
    "DPO": Path(os.environ.get("MINIMIND_DPO_CHECKPOINT", str(ROOT / "out/dpo_full/dpo_full_768.pth"))),
    "GRPO": Path(os.environ.get("MINIMIND_GRPO_CHECKPOINT", str(ROOT / "out/grpo_full/grpo_full_768.pth"))),
}

# category, question, expected answer
# expected=None means manual evaluation required
QUESTIONS = [
    ("knowledge", "What is the capital of France?", "Paris"),
    ("knowledge", "What is the capital of Japan?", "Tokyo"),
    ("knowledge", "How many days are in a week?", "7"),
    ("knowledge", "What planet do humans live on?", "Earth"),
    ("knowledge", "What is the chemical formula for water?", "H2O"),

    ("machine_learning", "What is machine learning?", None),
    ("machine_learning", "What is artificial intelligence?", None),
    ("machine_learning", "What is overfitting?", None),
    ("machine_learning", "What is supervised learning?", None),
    ("machine_learning", "What is a neural network?", None),

    ("python", "What is Python?", None),
    ("python", "What is a Python list?", None),
    ("python", "What is a Python dictionary?", None),
    ("python", "What does len() do in Python?", None),
    ("python", "What is the difference between a list and a tuple?", None),

    ("math", "What is 12 times 8? Answer with only the number.", "96"),
    ("math", "What is 15 plus 27? Answer with only the number.", "42"),
    ("math", "What is 100 divided by 4? Answer with only the number.", "25"),
    ("math", "What is 9 squared? Answer with only the number.", "81"),
    ("math", "What is 7 minus 3? Answer with only the number.", "4"),

    ("everyday", "Why do people need sleep?", None),
    ("everyday", "Why is drinking water important?", None),
    ("everyday", "What is a healthy breakfast?", None),
    ("everyday", "How can I learn English?", None),
    ("everyday", "Why do plants need sunlight?", None),

    ("instruction", "Answer with one word: What color is the sky on a clear day?", "blue"),
    ("instruction", "Answer with one word: What animal says meow?", "cat"),
    ("instruction", "Write exactly the word hello.", "hello"),
    ("instruction", "Answer with only the number: How many legs does a spider have?", "8"),
    ("instruction", "Answer yes or no: Is Paris in France?", "yes"),
]

tokenizer = AutoTokenizer.from_pretrained(
    "./model", local_files_only=True
)

config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    use_moe=False,
    max_seq_len=512,
)

model = MiniMindForCausalLM(config).cpu().eval()

@torch.inference_mode()
def generate(question, max_new_tokens=64):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True
    )

    ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids

    generated = []

    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :]
        token = int(logits.argmax(dim=-1).item())

        if token == tokenizer.eos_token_id:
            break

        generated.append(token)
        ids = torch.cat(
            [ids, torch.tensor([[token]])], dim=1
        )

    return tokenizer.decode(
        generated, skip_special_tokens=False
    )

REFUSAL_PATTERN = re.compile(
    r"\b(?:i cannot|i can't|i am unable|"
    r"i'm unable|i cannot provide|"
    r"i can't answer|i'm sorry, but i cannot)\b",
    re.I
)

def repetition_flag(text):
    words = re.findall(r"\b[\w']+\b", text.lower())
    if len(words) < 12:
        return False

    # Detect repeated 3-gram phrases
    grams = [
        tuple(words[i:i+3])
        for i in range(len(words)-2)
    ]
    return max(grams.count(g) for g in set(grams)) >= 3

def exact_match(answer, expected):
    if expected is None:
        return None

    normalized = re.sub(
        r"[^a-z0-9]+", " ", answer.lower()
    ).strip()
    target = re.sub(
        r"[^a-z0-9]+", " ", expected.lower()
    ).strip()

    # Strict answer matching for short-answer tests
    return normalized == target

# Resume evaluations without duplicating completed questions.
done = set()
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            done.add((item["model"], item["id"]))

for name, path in MODELS.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    print(f"\n===== MODEL: {name} =====", flush=True)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()

    for idx, (category, question, expected) in enumerate(QUESTIONS, 1):
        if (name, idx) in done:
            print(f"[{name}] {idx}/30 already completed", flush=True)
            continue

        start = time.monotonic()
        answer = generate(question)
        elapsed = round(time.monotonic() - start, 2)

        record = {
            "id": idx,
            "model": name,
            "category": category,
            "question": question,
            "expected": expected,
            "answer": answer,
            "exact_match": exact_match(answer, expected),
            "possible_refusal": bool(REFUSAL_PATTERN.search(answer)),
            "possible_repetition": repetition_flag(answer),
            "seconds": elapsed
        }

        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\n[{name}] {idx}/30 | {elapsed}s", flush=True)
        print("Q:", question, flush=True)
        print("A:", repr(answer), flush=True)

# Summarize automatic indicators
records = [
    json.loads(line)
    for line in OUT.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

summary = {}

for name in MODELS:
    rows = [r for r in records if r["model"] == name]
    scored = [r for r in rows if r["exact_match"] is not None]

    summary[name] = {
        "completed": len(rows),
        "strict_exact_matches": sum(r["exact_match"] for r in scored),
        "strict_exact_match_total": len(scored),
        "possible_refusals": sum(r["possible_refusal"] for r in rows),
        "possible_repetitions": sum(r["possible_repetition"] for r in rows),
    }

SUMMARY.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\n===== BENCHMARK SUMMARY =====", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("Saved:", OUT, flush=True)
