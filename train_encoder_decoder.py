"""Train a CLS-token decoder on top of a frozen JEPA encoder (encoder-only, no predictor)."""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

import lewm  # noqa: F401 — registers FlowJEPA so load_pretrained can reconstruct it
from lewm import get_img_preprocessor, get_column_normalizer, SaveCkptCallback, GridSaveCallback
from lewm.decoder import CLSDecoder

# get_img_preprocessor ImageNet-normalizes `pixels`; undo it to recover [0, 1].
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _denorm_to_unit(pixels):
    """ImageNet-normalized pixels (B, C, H, W) -> [0, 1]."""
    mean = _IMAGENET_MEAN.to(pixels.device, pixels.dtype)
    std = _IMAGENET_STD.to(pixels.device, pixels.dtype)
    return (pixels * std + mean).clamp(0, 1)


def compute_grid_predictions(model, batch, cfg):
    """Decode the encoder's own [CLS] embedding for a single frame and compare
    it against that same frame's pixels, for the grid-visualization callback."""

    ctx_len = cfg.history_size
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    with torch.no_grad():
        output = model.jepa.encode(batch)
        emb = output["emb"][:, ctx_len - 1]  # (B, D)
        recon = model.decoder(emb)

    tgt_pixels = _denorm_to_unit(batch["pixels"][:, ctx_len - 1].float())
    return {"target": tgt_pixels * 2.0 - 1.0, "recon": recon}


def decoder_forward(self, batch, stage, cfg):
    """Decode the encoder's [CLS] embedding and compare to its own frame."""

    ctx_len = cfg.history_size

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    with torch.no_grad():
        output = self.model.jepa.encode(batch)
        emb = output["emb"]  # (B, T, D)

    cls_emb = emb[:, ctx_len - 1]  # (B, D)
    recon = self.model.decoder(cls_emb)   # (B, 3, H, W)

    # Target: the same frame the [CLS] embedding was encoded from.
    # `pixels` are ImageNet-normalized by the preprocessor; undo that to [0, 1]
    # before mapping to [-1, 1] so it matches the decoder's tanh output range.
    tgt_pixels = _denorm_to_unit(batch["pixels"][:, ctx_len - 1].float())  # (B, C, H, W)
    target = tgt_pixels * 2.0 - 1.0

    loss = F.mse_loss(recon, target)
    output_dict = {"loss": loss}

    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output_dict.items()}, on_step=True, sync_dist=True)

    if stage == "val" and cfg.wandb.enabled:
        _log_images(self, target, recon, stage)

    return output_dict


def _log_images(module, target_pixels, recon, stage, n=4):
    """Log a grid of target vs. reconstructed images to wandb."""
    try:
        import wandb
    except ImportError:
        return

    n = min(n, target_pixels.size(0))
    target_imgs = ((target_pixels[:n].detach().cpu().float() * 0.5 + 0.5).clamp(0, 1) * 255).byte()
    recon_imgs  = ((recon[:n].detach().cpu().float() * 0.5 + 0.5).clamp(0, 1) * 255).byte()

    log_imgs = []
    for t, r in zip(target_imgs, recon_imgs):
        log_imgs.append(wandb.Image(t.permute(1, 2, 0).numpy(), caption="target"))
        log_imgs.append(wandb.Image(r.permute(1, 2, 0).numpy(), caption="recon"))

    wandb.log({f"{stage}/reconstructions": log_imgs})


class DecoderModule(torch.nn.Module):
    """Pairs a frozen JEPA backbone (encoder only) with a trainable decoder."""

    def __init__(self, jepa, decoder):
        super().__init__()
        self.jepa = jepa
        self.decoder = decoder

        for p in self.jepa.parameters():
            p.requires_grad_(False)


@hydra.main(version_base=None, config_path="./config/train", config_name="encoder_decoder")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val   = torch.utils.data.DataLoader(val_set,   **cfg.loader, shuffle=False, drop_last=False)

    sample_gen = torch.Generator().manual_seed(cfg.seed)
    fixed_indices = torch.randperm(len(val_set), generator=sample_gen)[:3].tolist()
    fixed_samples = [val_set[i] for i in fixed_indices]
    fixed_batch = torch.utils.data.default_collate(fixed_samples)

    ##############################
    ##       model / optim      ##
    ##############################

    backbone = swm.wm.utils.load_pretrained(cfg.policy)

    decoder = CLSDecoder(**cfg.decoder)
    model = DecoderModule(jepa=backbone, decoder=decoder)

    optimizers = {
        "decoder_opt": {
            "modules": "model.decoder",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    lit_module = spt.Module(
        model=model,
        forward=partial(decoder_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.decoder,
        epoch_interval=cfg.checkpoint_epoch_interval,
    )

    grid_callback = GridSaveCallback(
        fixed_batch=fixed_batch,
        cfg=cfg,
        predict_fn=compute_grid_predictions,
        output_dir=run_dir,
        epoch_interval=cfg.grid_save.epoch_interval,
        log_wandb=cfg.wandb.enabled,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[ckpt_callback, grid_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=lit_module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()


if __name__ == "__main__":
    run()
