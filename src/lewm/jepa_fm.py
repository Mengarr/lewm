"""Flow-matching JEPA for inference."""

import torch
from einops import rearrange

from lewm.jepa import JEPA
from lewm.time_utils import FREQ_DIM, create_sinusoidal_pos_embedding


class FlowJEPA(JEPA):
    """JEPA subclass that adds a flow-matching denoising rollout."""

    def __init__(self, time_proj_mlp, n_euler_steps=10, **jepa_kwargs):
        super().__init__(**jepa_kwargs)
        self.time_proj_mlp = time_proj_mlp
        self.n_euler_steps = n_euler_steps

    def _embed_time(self, batched_tau: torch.Tensor) -> torch.Tensor:
        """Map scalar timesteps → latent dim via sinusoidal + MLP."""
        freq_emb = create_sinusoidal_pos_embedding(batched_tau, FREQ_DIM).to(batched_tau.dtype)
        return self.time_proj_mlp(freq_emb)  # (B, D)

    def _build_inference_mask(self, ctx_len: int, device) -> torch.Tensor:
        """(ctx_len+1, ctx_len+1) mask: ctx causal, x_t attends to all ctx + self."""
        size = ctx_len + 1
        mask = torch.zeros(size, size, dtype=torch.bool, device=device)
        mask[:ctx_len, :ctx_len] = torch.tril(torch.ones(ctx_len, ctx_len, dtype=torch.bool, device=device))
        mask[ctx_len, :] = True
        return mask

    def _flow_predict_one_step(
        self,
        ctx_emb: torch.Tensor,      # (B, ctx_len, D)
        ctx_act: torch.Tensor,      # (B, ctx_len, D)
        next_act_emb: torch.Tensor, # (B, 1, D)
    ) -> torch.Tensor:
        """Euler integration from noise to predicted next embedding."""
        B, ctx_len, D = ctx_emb.shape
        device = ctx_emb.device

        x_t = torch.randn(B, 1, D, device=device, dtype=ctx_emb.dtype)
        mask = self._build_inference_mask(ctx_len, device)
        adarms_base = torch.cat([ctx_act, next_act_emb], dim=1)  # (B, ctx_len+1, D)

        dt = 1.0 / self.n_euler_steps
        for i in range(self.n_euler_steps):
            tau = 1.0 - i * dt  # integrate from τ=1 (noise) → τ=0 (data)
            batched_tau = torch.full((B,), tau, device=device, dtype=torch.float32)
            time_emb = self._embed_time(batched_tau)  # (B, D)

            concat_emb = torch.cat([ctx_emb, x_t], dim=1)          # (B, ctx_len+1, D)
            adarms_cond = adarms_base + time_emb.unsqueeze(1)       # (B, ctx_len+1, D)

            v_t = self.predict(concat_emb, adarms_cond, attn_mask=mask)[:, ctx_len:]  # (B, 1, D)
            x_t = x_t - dt * v_t

        return x_t  # (B, 1, D)

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Flow-matching rollout. Same interface as JEPA.rollout."""
        assert "pixels" in info
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            ctx_emb = emb[:, -HS:]
            ctx_act = act_emb[:, -HS:]
            next_act_emb = self.action_encoder(act_future[:, t: t + 1])

            pred_emb = self._flow_predict_one_step(ctx_emb, ctx_act, next_act_emb)
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t: t + 1]
            act = torch.cat([act, next_act], dim=1)

        # predict the last state
        act_emb = self.action_encoder(act)
        ctx_emb = emb[:, -HS:]
        ctx_act = act_emb[:, -HS:]
        next_act_emb = self.action_encoder(act_future[:, n_steps - 1: n_steps])
        pred_emb = self._flow_predict_one_step(ctx_emb, ctx_act, next_act_emb)
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info
