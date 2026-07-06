"""Interactive visualization of the eval.py inference pipeline for PushT.

After CEM plans an action sequence, step through it one action at a time.
Press Enter to advance each step; 'q' + Enter to quit.

Usage:
    python visualize_eval.py policy=<ckpt_name>   # with model
    python visualize_eval.py                       # random policy (no overlay)
"""

import os
import sys
import termios

os.environ["MUJOCO_GL"] = "egl"

from collections import deque
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stable_worldmodel.policy import WorldModelPolicy

import lewm  # noqa: F401

ACTION_SCALE = 100
WINDOW_SIZE = 512  # pymunk coordinate space


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def planned_agent_positions(current_state, planned_actions_real,
                            k_p=100, k_v=20, dt=0.01, control_hz=10):
    """Simulate PD-controlled agent trajectory given planned actions.

    PushT uses a PD controller that drives the agent towards a target position
    each step, so the agent doesn't instantly reach pos + action * 100.

    Args:
        current_state: [agent_x, agent_y, block_x, block_y, block_angle, vx, vy]
            in pymunk 512px coordinates.
        planned_actions_real: [H, 2] real actions (after inverse transform).

    Returns:
        positions: [H+1, 2] agent positions including current.
    """
    n_substeps = int(1 / (dt * control_hz))
    positions = [current_state[:2].copy()]
    pos = current_state[:2].copy()
    vel = current_state[5:7].copy() if len(current_state) >= 7 else np.zeros(2)

    for action in planned_actions_real:
        target = pos + action * ACTION_SCALE
        for _ in range(n_substeps):
            acc = k_p * (target - pos) + k_v * (-vel)
            vel = vel + acc * dt
            pos = pos + vel * dt
        positions.append(pos.copy())

    return np.array(positions)


def state_to_pixel(positions, img_size):
    """Scale pymunk [0,512] coordinates to pixel [0, img_size]."""
    return positions / WINDOW_SIZE * img_size


def draw_overlay(ax, frame, planned_positions_px, executed_idx, goal_state=None, im_handle=None):
    """Render the frame with planned trajectory overlaid.

    Args:
        ax: matplotlib Axes.
        frame: HxWx3 uint8 image.
        planned_positions_px: [H+1, 2] planned agent positions in pixel space.
        executed_idx: how many steps already executed (greyed out).
        goal_state: optional [7] goal state for goal agent/block positions.
        im_handle: existing AxesImage to update in place (avoids resize flicker).

    Returns:
        The AxesImage handle.
    """
    if im_handle is None:
        im_handle = ax.imshow(frame)
        ax.axis("off")
    else:
        im_handle.set_data(frame)

    # Remove old overlay artists (lines/markers) but keep the image
    for artist in ax.lines + ax.collections:
        artist.remove()
    for legend in [ax.get_legend()]:
        if legend:
            legend.remove()

    img_size = frame.shape[0]
    cmap = plt.cm.plasma

    # Draw planned path with temporal colour coding
    if planned_positions_px is not None and len(planned_positions_px) > 1:
        xs = planned_positions_px[:, 0]
        ys = planned_positions_px[:, 1]
        n_steps = len(xs) - 1  # number of action steps

        # Draw each segment with a colour from the colormap (early=bright, late=dark)
        for i in range(n_steps):
            color = cmap(1.0 - i / max(n_steps - 1, 1))
            ax.plot(xs[i:i+2], ys[i:i+2], "-", color=color, linewidth=2, alpha=0.85)
            ax.plot(xs[i+1], ys[i+1], "o", color=color, markersize=5)

        # Mark current agent position
        ax.plot(xs[0], ys[0], "*", color="yellow", markersize=12,
                markeredgecolor="black", label="agent start pos")

        # Mark goal positions
        if goal_state is not None:
            gx, gy = state_to_pixel(goal_state[:2], img_size)
            ax.plot(gx, gy, "D", color="lime", markersize=10,
                    markeredgecolor="black", label="goal agent")
            bx, by = state_to_pixel(goal_state[2:4], img_size)
            ax.plot(bx, by, "s", color="lime", markersize=10,
                    markeredgecolor="black", label="goal block")

    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    return im_handle


class CapturingWorldModelPolicy(WorldModelPolicy):
    """WorldModelPolicy that captures the last raw planned action sequence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_raw_plan = None  # [n_envs, horizon, action_dim] before buffer split
        self._replanned = False
        self._last_predictor_calls = 0

    def get_action(self, info_dict, **kwargs):
        # Replicate parent logic but capture raw solver output
        from stable_worldmodel.policy import WorldModelPolicy
        import copy

        info_dict_prepared = self._prepare_info(info_dict)
        n_envs = self.env.num_envs
        self._last_predictor_calls = 0

        needs_flush = info_dict_prepared.pop("_needs_flush", None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    if self._next_init is not None:
                        self._next_init[i] = 0

        terminated = info_dict_prepared.get("terminated")
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )

        replan_idx = [
            i for i in range(n_envs)
            if len(self._action_buffer[i]) == 0 and not dead[i]
        ]

        if replan_idx:
            idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
            sliced = {}
            for k, v in info_dict_prepared.items():
                if torch.is_tensor(v):
                    sliced[k] = v[idx_tensor]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan_idx]
                elif isinstance(v, list):
                    sliced[k] = [v[i] for i in replan_idx]
                else:
                    sliced[k] = v

            sliced_init = (
                self._next_init[idx_tensor] if self._next_init is not None else None
            )

            model = self.solver.model
            predict_calls = 0
            orig_predict = model.predict

            def _counting_predict(*p_args, **p_kwargs):
                nonlocal predict_calls
                predict_calls += 1
                return orig_predict(*p_args, **p_kwargs)

            model.predict = _counting_predict
            try:
                outputs = self.solver(sliced, init_action=sliced_init)
            finally:
                model.predict = orig_predict
            self._last_predictor_calls = predict_calls
            print(f"[predictor] {predict_calls} forward passes for this action chunk "
                  f"(replan over {len(replan_idx)} env(s))")

            actions = outputs["actions"]
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]

            # Capture the full planned sequence (normalized action space)
            self._last_raw_plan = actions.detach().cpu().numpy()  # [n_envs_subset, H, D]
            self._replanned = True

            if self.cfg.warm_start and rest.shape[1] > 0:
                if self._next_init is None:
                    self._next_init = torch.zeros(
                        n_envs, rest.shape[1], rest.shape[2], dtype=rest.dtype
                    )
                self._next_init[idx_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None

            plan = plan.reshape(len(replan_idx), self.flatten_receding_horizon, -1)
            for row, env_i in enumerate(replan_idx):
                self._action_buffer[env_i].extend(plan[row])

        action_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, action_dim), float("nan"))
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()

        action = action.reshape(*self.env.action_space.shape)
        action = action.float().numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        return action


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be <= eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    # Single env for interactive stepping
    cfg.world.num_envs = 1
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy_cfg = cfg.get("policy", "random")

    if policy_cfg != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        print(f"JEPA predictor device: {next(model.parameters()).device}")
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        print(f"CEM solver device: {getattr(solver, 'device', 'n/a')}")
        policy = CapturingWorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    world.set_policy(policy)

    all_ep_idx = dataset.get_col_data(col_name)
    all_step_idx = dataset.get_col_data("step_idx")
    episode_len_arr = np.array([
        np.max(all_step_idx[all_ep_idx == ep_id]) + 1
        for ep_id in ep_indices
    ])
    max_start_idx = episode_len_arr - cfg.eval.goal_offset_steps - 1
    valid_ep_mask = max_start_idx > 0
    valid_ep_ids = ep_indices[valid_ep_mask]
    valid_max_starts = max_start_idx[valid_ep_mask]

    import torch as _torch
    def _to_numpy(v):
        return v.numpy() if isinstance(v, _torch.Tensor) else v

    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    from stable_worldmodel.world.world import _apply_callables

    img_size = 224
    has_model = policy_cfg != "random"

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.ion()
    plt.show()

    g = np.random.default_rng(cfg.seed)

    print("\nControls: Enter = step | 'n' + Enter = next episode | 'q' + Enter = quit\n")

    ep_order = g.permutation(len(valid_ep_ids))

    for ep_rank in ep_order:
        chosen_ep = int(valid_ep_ids[ep_rank])
        chosen_start = int(g.integers(0, valid_max_starts[ep_rank]))

        chunk = dataset.load_chunk(
            np.array([chosen_ep]),
            np.array([chosen_start]),
            np.array([chosen_start + cfg.eval.goal_offset_steps + 1]),
        )
        ep_data = next(iter(chunk))
        init_state = _to_numpy(ep_data["state"][0])
        goal_state = _to_numpy(ep_data["state"][-1])

        _, infos = world.envs.reset(seed=None, options=[{"state": init_state, "goal_state": goal_state}])
        world.terminateds = np.zeros(1, dtype=bool)
        world.truncateds = np.zeros(1, dtype=bool)
        world.infos = infos

        if callables:
            _apply_callables(world.envs.envs[0].unwrapped, callables,
                             {"state": init_state, "goal_state": goal_state})

        if has_model:
            for buf in policy._action_buffer:
                buf.clear()
            policy._last_raw_plan = None
            policy._next_init = None

        print(f"\nEpisode {chosen_ep}, start step {chosen_start}")

        quit_all = False
        im_handle = None
        planned_positions_px = None
        for step in range(cfg.eval.eval_budget):
            policy._replanned = False
            action = world._get_actions()

            cur_state = world.infos.get("state")
            cur_state_np = np.array(cur_state[0, 0]) if cur_state is not None else None

            # Only recompute planned positions when CEM just ran
            if has_model and policy._replanned and policy._last_raw_plan is not None and cur_state_np is not None:
                raw = policy._last_raw_plan[0]
                action_dim = world.envs.single_action_space.shape[-1]
                raw = raw.reshape(-1, action_dim)
                real_actions = process["action"].inverse_transform(raw) if "action" in process else raw
                planned_positions_px = state_to_pixel(planned_agent_positions(cur_state_np, real_actions), img_size)

            frame = world.infos.get("pixels")
            if frame is not None:
                frame_np = np.array(frame[0, 0])
                if frame_np.dtype != np.uint8:
                    frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            else:
                frame_np = np.zeros((img_size, img_size, 3), dtype=np.uint8)

            im_handle = draw_overlay(ax, frame_np, planned_positions_px, 0, goal_state, im_handle)
            title = f"Ep {chosen_ep} | Step {step}/{cfg.eval.eval_budget}"
            if cur_state_np is not None:
                title += f"\nAgent: ({cur_state_np[0]:.0f}, {cur_state_np[1]:.0f})  Block: ({cur_state_np[2]:.0f}, {cur_state_np[3]:.0f})"
            ax.set_title(title, fontsize=9)
            fig.canvas.draw()
            fig.canvas.flush_events()

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            user_input = input("Step [Enter] / next episode [n] / quit [q]: ").strip().lower()
            if user_input == "q":
                quit_all = True
                break
            if user_input == "n":
                break

            _, rewards, terminateds, truncateds, infos = world.envs.step(action)
            world.rewards = rewards
            world.terminateds = terminateds
            world.truncateds = truncateds
            world.infos = infos

            if terminateds[0]:
                frame = world.infos.get("pixels")
                if frame is not None:
                    frame_np = np.array(frame[0, 0])
                    if frame_np.dtype != np.uint8:
                        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
                    draw_overlay(ax, frame_np, None, 0, goal_state, im_handle)
                    ax.set_title(f"Ep {chosen_ep} | Success!")
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
                user_input = input("Success! Next episode [Enter] / quit [q]: ").strip().lower()
                if user_input == "q":
                    quit_all = True
                break

        if quit_all:
            break


    plt.close()


if __name__ == "__main__":
    run()
