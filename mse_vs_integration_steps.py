"""
Plot MSE (predicted embedding vs true embedding) as a function
of flow-matching integration steps.

Usage:
    python mse_vs_integration_steps.py --checkpoint <ckpt_path> --n_examples 50
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

import lewm  # noqa: F401 — registers FlowJEPA


FRAMESKIP = 5
INTEGRATION_STEPS = [1, 4, 10, 20, 50]


def img_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def _find_row(episode_idx, step_idx, ep, s):
    match = np.nonzero((episode_idx == ep) & (step_idx == s))[0]
    assert len(match) == 1, f"Expected one row for ep={ep} step={s}, got {len(match)}"
    return int(match[0])


def _action_chunk(dataset, episode_idx, step_idx, ep, s):
    """Fetch FRAMESKIP consecutive raw actions starting at step s, flattened."""
    indices = [_find_row(episode_idx, step_idx, ep, s + off) for off in range(FRAMESKIP)]
    raw = dataset.get_row_data(np.array(indices))["action"]  # (FRAMESKIP, raw_dim)
    return raw.flatten()  # (FRAMESKIP * raw_dim,)


def load_examples(dataset, n_examples, history_len, seed=42):
    """Return (ctx_indices, target_indices, ctx_action_chunks).

    ctx_indices:       (n, history_len) — dataset row indices for each context frame
    target_indices:    (n,)             — dataset row index for the target frame (t + FRAMESKIP)
    ctx_action_chunks: (n, history_len, FRAMESKIP * raw_dim)
    """
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")

    ep_ids = np.unique(episode_idx)

    # t=0 frame must have (history_len-1)*FRAMESKIP history and FRAMESKIP future
    min_step = (history_len - 1) * FRAMESKIP
    valid = []
    for ep_id in ep_ids:
        ep_len = int(np.max(step_idx[episode_idx == ep_id])) + 1
        mask = (
            (episode_idx == ep_id)
            & (step_idx >= min_step)
            & (step_idx + FRAMESKIP < ep_len)
        )
        valid.extend(np.nonzero(mask)[0].tolist())

    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid, size=n_examples, replace=False)
    chosen = np.sort(chosen)

    ctx_indices = []
    target_indices = []
    ctx_action_chunks = []

    for idx in chosen:
        ep = episode_idx[idx]
        s = int(step_idx[idx])

        # context frames: t - (H-1)*FS, ..., t - FS, t
        ctx_steps = [s - (history_len - 1 - k) * FRAMESKIP for k in range(history_len)]
        ctx_rows = [_find_row(episode_idx, step_idx, ep, cs) for cs in ctx_steps]
        ctx_indices.append(ctx_rows)

        target_indices.append(_find_row(episode_idx, step_idx, ep, s + FRAMESKIP))

        # action chunk for each context frame (the FRAMESKIP actions starting there)
        chunks = [_action_chunk(dataset, episode_idx, step_idx, ep, cs) for cs in ctx_steps]
        ctx_action_chunks.append(chunks)

    return (
        np.array(ctx_indices),           # (n, H)
        np.array(target_indices),         # (n,)
        np.array(ctx_action_chunks),      # (n, H, FRAMESKIP * raw_dim)
    )


@torch.no_grad()
def compute_mse_for_steps(model, dataset, ctx_indices, target_indices, ctx_action_chunks, n_steps, transform, device):
    """Run flow prediction with n_steps Euler steps; return per-example MSE."""
    model.n_euler_steps = n_steps

    mses = []
    for i in range(len(target_indices)):
        history_len = ctx_indices.shape[1]

        # encode all context frames → (1, H, D)
        ctx_pixels = [
            transform(dataset.get_row_data([ctx_indices[i, k]])["pixels"][0])
            .unsqueeze(0).unsqueeze(0)
            for k in range(history_len)
        ]
        ctx_pixels = torch.cat(ctx_pixels, dim=1).to(device)  # (1, H, C, H_img, W_img)
        info_ctx = model.encode({"pixels": ctx_pixels})
        ctx_emb = info_ctx["emb"]  # (1, H, D)

        # encode context action chunks → (1, H, A)
        ctx_acts = torch.tensor(ctx_action_chunks[i], dtype=torch.float32)  # (H, chunk_dim)
        ctx_acts = ctx_acts.unsqueeze(0).to(device)                          # (1, H, chunk_dim)
        ctx_act_emb = model.action_encoder(ctx_acts)                         # (1, H, A)

        # next_act_emb = last context action (leads to the target frame)
        next_act_emb = ctx_act_emb[:, -1:]  # (1, 1, A)

        # encode target frame
        target_pixels = (
            transform(dataset.get_row_data([target_indices[i]])["pixels"][0])
            .unsqueeze(0).unsqueeze(0).to(device)
        )
        emb_target = model.encode({"pixels": target_pixels})["emb"]  # (1, 1, D)

        pred_emb = model._flow_predict_one_step(
            ctx_emb=ctx_emb,
            ctx_act=ctx_act_emb,
            next_act_emb=next_act_emb,
        )  # (1, 1, D)

        mse = torch.mean((pred_emb - emb_target) ** 2).item()
        mses.append(mse)

    return np.array(mses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="Checkpoint name / path (same as eval.py policy=)")
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--history_len", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output", default="mse_vs_steps.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    assert hasattr(model, "_flow_predict_one_step"), \
        "Checkpoint does not appear to be a FlowJEPA model"

    print("Loading dataset...")
    cache_dir = Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        args.dataset,
        keys_to_cache=["action"],
        cache_dir=cache_dir,
    )

    print(f"Sampling {args.n_examples} examples (frameskip={FRAMESKIP}, history_len={args.history_len})...")
    ctx_indices, target_indices, ctx_action_chunks = load_examples(
        dataset, args.n_examples, args.history_len, seed=args.seed
    )

    transform = img_transform(args.img_size)

    results = {}
    for n_steps in INTEGRATION_STEPS:
        print(f"  integration steps = {n_steps} ...", end=" ", flush=True)
        mses = compute_mse_for_steps(model, dataset, ctx_indices, target_indices, ctx_action_chunks, n_steps, transform, device)
        results[n_steps] = mses
        print(f"mean MSE = {mses.mean():.6f}")

    # --- plot ---
    medians = np.array([np.median(results[s]) for s in INTEGRATION_STEPS])
    q25 = np.array([np.percentile(results[s], 25) for s in INTEGRATION_STEPS])
    q75 = np.array([np.percentile(results[s], 75) for s in INTEGRATION_STEPS])
    x = np.arange(len(INTEGRATION_STEPS))
    labels = [str(s) for s in INTEGRATION_STEPS]
    y_max = max(results[s].max() for s in INTEGRATION_STEPS) * 1.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- bar chart: median ± IQR ---
    ax1.bar(x, medians, color="steelblue", alpha=0.8, zorder=2)
    ax1.errorbar(
        x, medians,
        yerr=[medians - q25, q75 - medians],
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlim(-0.5, len(INTEGRATION_STEPS) - 0.5)
    ax1.set_ylim(0, y_max)
    ax1.set_xlabel("Integration steps (Euler)")
    ax1.set_ylabel("MSE (predicted vs true embedding)")
    ax1.set_title(f"Median ± IQR  (n={args.n_examples})")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # --- strip plot: individual points jittered within each group ---
    rng = np.random.default_rng(0)
    for i, s in enumerate(INTEGRATION_STEPS):
        pts = results[s]
        jitter = rng.uniform(-0.25, 0.25, size=len(pts))
        ax2.scatter(i + jitter, pts, alpha=0.6, s=20, color="steelblue", zorder=2)
        ax2.hlines(np.median(pts), i - 0.3, i + 0.3, colors="black", linewidth=1.5, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlim(-0.5, len(INTEGRATION_STEPS) - 0.5)
    ax2.set_ylim(0, y_max)
    ax2.set_xlabel("Integration steps (Euler)")
    ax2.set_ylabel("MSE (predicted vs true embedding)")
    ax2.set_title(f"Individual samples + median  (n={args.n_examples})")
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    plt.suptitle(f"Flow JEPA: embedding MSE vs integration steps  (history_len={args.history_len})", y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
