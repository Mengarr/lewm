import torch
import numpy as np
import matplotlib.pyplot as plt


def sample_time(bsize: int, device) -> torch.Tensor:
    m = 0.75
    std = 1.0
    s = m + std * torch.randn(bsize, device=device)
    tau = torch.sigmoid(s)
    return tau.to(dtype=torch.float32, device=device)


device = "cpu"
samples = sample_time(200_000, device).numpy()

# Theoretical logit-normal PDF
x = np.linspace(1e-4, 1 - 1e-4, 1000)
m, std = 0.75, 1.0
logit_x = np.log(x / (1 - x))
pdf = (1.0 / (std * np.sqrt(2 * np.pi) * x * (1 - x))) * np.exp(-((logit_x - m) ** 2) / (2 * std ** 2))

plt.figure(figsize=(8, 5))
plt.hist(samples, bins=100, density=True, alpha=0.6, label="sampled tau")
plt.plot(x, pdf, "r-", lw=2, label="theoretical logit-normal PDF")
plt.xlabel("tau")
plt.ylabel("density")
plt.title(f"Logit-normal time sampling (m={m}, std={std})")
plt.legend()
plt.tight_layout()
plt.savefig("sample_time_dist.png", dpi=150)
print("Saved plot to sample_time_dist.png")
