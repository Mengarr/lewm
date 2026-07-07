import torch
import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt


def sample_time(bsize: int, device) -> torch.Tensor:
    m = 0.75
    std = 1.0
    s = m + std * torch.randn(bsize, device=device)
    tau = torch.sigmoid(s)
    return tau.to(dtype=torch.float32, device=device)


device = "cpu"
samples = sample_time(200_000, device).numpy()

# Logit-normal theoretical PDF
x = np.linspace(1e-4, 1 - 1e-4, 1000)
m, std = 0.75, 1.0
logit_x = np.log(x / (1 - x))
logit_normal_pdf = (1.0 / (std * np.sqrt(2 * np.pi) * x * (1 - x))) * np.exp(-((logit_x - m) ** 2) / (2 * std ** 2))

# Beta-based schedule, flipped so density is largest near tau = s: p(tau) = Beta(tau / s; 1.5, 1)
s = 0.999
tau_grid = np.linspace(1e-6, s - 1e-6, 1000)
u = tau_grid / s
# pdf w.r.t. u, then change of variables du/dtau = 1/s
beta_pdf_u = beta_dist.pdf(u, 1.5, 1)
beta_pdf_tau = beta_pdf_u / s

plt.figure(figsize=(8, 5))
plt.hist(samples, bins=100, density=True, alpha=0.5, label="sampled tau (logit-normal)")
plt.plot(x, logit_normal_pdf, "r-", lw=2, label=f"logit-normal PDF (m={m}, std={std})")
plt.plot(tau_grid, beta_pdf_tau, "g--", lw=2, label=f"Beta(tau/s; 1.5, 1), s={s}")
plt.xlabel("tau")
plt.ylabel("density")
plt.title("Logit-normal vs Beta-based time sampling")
plt.legend()
plt.tight_layout()
plt.savefig("sample_time_overlay.png", dpi=150)
print("Saved plot to sample_time_overlay.png")
