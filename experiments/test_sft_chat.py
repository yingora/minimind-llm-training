# Portfolio preparation: replaced AutoDL-specific paths with MINIMIND_ARTIFACT_ROOT.
import os
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "./model",
    local_files_only=True
)

config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    use_moe=False
)

model = MiniMindForCausalLM(config)

weights = torch.load(
    os.path.join(os.environ.get("MINIMIND_ARTIFACT_ROOT", "."), 'out/sft_full/sft_full_768.pth'),
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights, strict=True)
model = model.to(device).eval()

questions = [
    "你好，请介绍一下你自己。",
    "法国的首都是哪里？",
    "请用简单的语言解释什么是机器学习。",
    "What is the difference between AI and machine learning?"
]

@torch.inference_mode()
def chat(question):
    messages = [{"role": "user", "content": question}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

    output_ids = inputs.clone()

    for _ in range(120):
        outputs = model(input_ids=output_ids)
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

        output_ids = torch.cat([output_ids, next_token], dim=1)

    generated = output_ids[0, inputs.shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

for question in questions:
    print("\n" + "=" * 60)
    print("USER:", question)
    print("MINIMIND:", chat(question))

print("\nDevice:", device)
