import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./model",
    local_files_only=True
)

text = "I love machine learning."
encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
input_ids = encoded["input_ids"].to(device)

print("Original text:", text)
print("Token IDs:", input_ids.tolist())
print("Input shape:", tuple(input_ids.shape))

# 2. Random model initialization
config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    use_moe=False,
    flash_attn=False
)

model = MiniMindForCausalLM(config).to(device)
model.train()

params = sum(p.numel() for p in model.parameters())
print("Model parameters:", f"{params:,}")

# 3. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Save a copy of one parameter for comparison
parameter = model.model.embed_tokens.weight
before = parameter.detach().clone()

# 4. Forward pass and cross-entropy loss
output = model(input_ids=input_ids, labels=input_ids)
loss = output.loss

print("Logits shape:", tuple(output.logits.shape))
print("Loss before update:", loss.item())

# 5. Backward pass
optimizer.zero_grad()
loss.backward()

print("Gradient norm:", parameter.grad.norm().item())

# 6. Update model weights
optimizer.step()

difference = (parameter.detach() - before).abs().max().item()
print("Maximum parameter change:", difference)

# 7. Check loss again on the same example
model.eval()
with torch.no_grad():
    updated_loss = model(input_ids=input_ids, labels=input_ids).loss

print("Loss after update:", updated_loss.item())
print("Device:", device)
