"""Strip torch.compile's "_orig_mod." prefix from a checkpoint's state_dict keys.

Usage:
    python rename_weights.py --policy pusht/pusht_flow_v2_10
"""

import argparse

import torch
import stable_worldmodel as swm
from stable_worldmodel.data import get_cache_dir
from stable_worldmodel.wm.utils import _resolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="Same value passed as cfg.policy, e.g. pusht/pusht_flow_v2_10")
    args = parser.parse_args()

    cache_dir = get_cache_dir(sub_folder="checkpoints")
    checkpoint_path, _ = _resolve(args.policy, cache_dir)

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    prefix = "_orig_mod."
    n_prefixed = sum(1 for k in state_dict if k.startswith(prefix))
    if n_prefixed == 0:
        print(f"No '{prefix}' keys found in {checkpoint_path}; nothing to do.")
        return

    new_state_dict = {
        (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()
    }
    torch.save(new_state_dict, checkpoint_path)
    print(f"Stripped '{prefix}' from {n_prefixed}/{len(state_dict)} keys in {checkpoint_path}")


if __name__ == "__main__":
    main()
