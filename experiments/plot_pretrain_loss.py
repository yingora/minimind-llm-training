# Portfolio preparation: replaced AutoDL-specific paths with MINIMIND_ARTIFACT_ROOT.
import os
import re
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = os.path.join(os.environ.get("MINIMIND_ARTIFACT_ROOT", "."), 'logs/pretrain_10k.log')
output_dir = os.path.join(os.environ.get("MINIMIND_ARTIFACT_ROOT", "."), 'logs')

pattern = re.compile(
    r"Epoch:\[\d+/\d+\]\((\d+)/(\d+)\), "
    r"loss:\s*([\d.]+).*?lr:\s*([\d.]+)"
)

steps, losses, lrs = [], [], []

with open(log_path, encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(3)))
            lrs.append(float(match.group(4)))

assert losses, "No training logs were parsed."

with open(f"{output_dir}/pretrain_10k_metrics.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["batch", "loss", "learning_rate"])
    writer.writerows(zip(steps, losses, lrs))

plt.figure(figsize=(10, 5))
plt.plot(steps, losses, label="Training Loss", alpha=0.65)

window = 5
if len(losses) >= window:
    moving_avg = [
        sum(losses[i-window+1:i+1]) / window
        for i in range(window-1, len(losses))
    ]
    plt.plot(steps[window-1:], moving_avg,
             label="Moving Average", linewidth=2)

plt.xlabel("Training Batch")
plt.ylabel("Cross-Entropy Loss")
plt.title("MiniMind 64M - Pretraining on 10K Samples")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/pretrain_10k_loss.png", dpi=180)

print("Logged points:", len(losses))
print("First 10 average loss:", sum(losses[:10])/len(losses[:10]))
print("Last 10 average loss:", sum(losses[-10:])/len(losses[-10:]))
print("Saved loss curve and CSV.")
