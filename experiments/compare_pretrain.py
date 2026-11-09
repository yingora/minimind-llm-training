# Portfolio preparation: replaced AutoDL-specific paths with MINIMIND_ARTIFACT_ROOT.
import os
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

torch.manual_seed(42)
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
    "./model", local_files_only=True
)

config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    use_moe=False
)

prompts = [
    "The capital of France is",
    "人工智能的发展",
    "Once upon a time"
]

def load_model(weight_path=None):
    model = MiniMindForCausalLM(config).to(device)

    if weight_path:
        state = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(state)

    model.eval()
    return model

@torch.inference_mode()
def generate(model, prompt, max_new_tokens=40):
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

models = {
    "Random": load_model(),
    "Pretrained 10K": load_model(
        os.path.join(os.environ.get("MINIMIND_ARTIFACT_ROOT", "."), 'out/pretrain_10k/pretrain_10k_768.pth')
    ),
    "Pretrained Full": load_model(
        os.path.join(os.environ.get("MINIMIND_ARTIFACT_ROOT", "."), 'out/pretrain_full/pretrain_full_768.pth')
    )
}

for prompt in prompts:
    print(f"\n===== Prompt: {prompt} =====")
    for name, model in models.items():
        print(f"\n{name}:")
        print(generate(model, prompt))
