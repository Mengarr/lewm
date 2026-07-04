import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from lewm import SIGReg, get_column_normalizer, get_img_preprocessor, SaveCkptCallback
from lewm.time_utils import FREQ_DIM, create_sinusoidal_pos_embedding


def sample_noise(shape, device):
    """Gaussian Noise Sampler"""
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )

def sample_time(bsize: int, device) -> torch.Tensor:
    # Uses Logit-normal time sampling, setting m<0 pushes probability mass towards low tau
    m = -0.75
    std = 1.0
    s   = m + std * torch.randn(bsize, device=device)
    tau = torch.sigmoid(s)                            # in (0,1)
    return tau.to(dtype=torch.float32, device=device)

def embed_time(self, batched_timestep: torch.Tensor, freq_dim: int = FREQ_DIM) -> torch.Tensor:
    time_emb = create_sinusoidal_pos_embedding(batched_timestep, freq_dim) # (B, freq_dim)
    # time_proj_mlp lives on self.model (FlowJEPA), not on the spt.Module wrapper
    param = next(self.model.time_proj_mlp.parameters())
    return self.model.time_proj_mlp(time_emb.to(dtype=param.dtype))  # (B, latent_dim)
    
"""
In /config/train/data configs, num_steps is constrained to be `num_steps = history_size + num_preds` - this is the key constraint.

The num_preds and history_size are defined in /config/train/lewm.yaml as (by default) history_size = 3, num_preds = 1

Thus, num_steps = 3+1 = 4 = T
This way, ctx_emb/ctx_act contain frames [0,1,2] and tgt_emb contains frames [1,2,3]. Thus they have the same size T.

Usually you pass the noisy target and any other embeddings into the flow model, then adaLN broadcasts the condition. 
However, here, our conditioned adaLN information is positionally tied to our embeddings. 

The method we are using here is called prefix attention.
- context tokens form a prefix that target tokens can attend to - with a controlled attention mask.
- con is that the model produces outputs that are not used.

Other method we can use is cross attention.
- sequence length becomes ctx_len instead of 2 * ctx_len in prefix attention, and no compute is wasted on the discarded ctx emb tokens.
- the complexity is moved to the predictor, it requires cross attention support in the architecture.

attn_mask
1 0 0 0 0 0
1 1 0 0 0 0
1 1 1 0 0 0
1 0 0 1 0 0
1 1 0 0 1 0
1 1 1 0 0 1

           ctx        x_t
ctx    [ causal   | blocked ]
x_t    [ causal   | diagonal  ]

Addition:
Can add another predictor that predicts actions for t+1. Use MoE style shared attention. This can be used during inference as a guidance signal for CEM?
"""

# encoder is passed a history of inputs, then the predictor is fed the first ctx_len steps and must predict 
def lejepa_flow_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
        
    # latent representation z^hat
    emb = output["emb"]  # (batch_size, sequence length from batch, latent dim)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len] # (B, T-1, D)
    ctx_act = act_emb[:, : ctx_len] # (B, T-1, D)

    tgt_emb = emb[:, n_preds:] # (B, T-1, D)
    tgt_act = act_emb[:, n_preds:] # (B, T-1, D)

    device = act_emb.device
    batch_size, _, latent_dim = emb.shape 

    noise = sample_noise((batch_size, ctx_len, latent_dim), device) # (B, T-1, D) 
    """Noise alternative
    We can sample our noise to be a linear interpolation between our noise and the current state embeddings as our current state embeddings should be theoretically close to our goal embeddings.

    noise = beta * ctx_emb + sigma * sample_noise()

    Where:
        - beta [0, 1]
            - beta = 0 -> normal FM
            - beta = 1 -> loss becomes more conditioned on z_t
        - sigma > 0
            - Controls how much noise there is. sigma = 1 -> normal FM

    If beta = 1 and sigma = 0, then it collapses to MSE! 
    """
    batched_time = sample_time(batch_size, device) # (B) scalar time samples [0,1] for each sample in the batch
    time_expanded = batched_time[:, None, None] # expand to (B, 1, 1), pytorch element wise multiplication auto broadcasts to B, T, D

    time_emb = embed_time(self, batched_time) # (B, D)

    # noisy target latent
    x_t =  time_expanded * noise + (1 - time_expanded) * tgt_emb # element wise multiplication, i.e [2,1] * [3,1] = [6, 1]

    concat_emb = torch.cat([ctx_emb, x_t], dim=1) # (B, (T-1)*2, D) concat the context embeddings and the noisy latent so the model sees both

    attn_mask = torch.tril(torch.ones(ctx_len*2, ctx_len*2, dtype=torch.bool, device = device)) # lower triangular mask
    attn_mask[ctx_len:, ctx_len:] = torch.eye(ctx_len, dtype=torch.bool, device=device)
    attn_mask[ctx_len:, :ctx_len] = attn_mask[:ctx_len, :ctx_len] 

    # adaRMS condition, time_emb.unsqueeze transforms it from (B,D) -> (B,1,D) and then pytorch auto broadcasts along dim 1
    adarms_cond = torch.cat([ctx_act, tgt_act], dim=1) + time_emb.unsqueeze(1) # (B, (T-1)*2, D)

    # predicted velocity, here we have to discard the first three
    v_t = self.model.predict(concat_emb, adarms_cond, attn_mask)[:, ctx_len:] # (B, T-1, D)
    # target velocity
    u_t = noise - tgt_emb # (B, T-1, D)

    # Flow matching loss + SIGreg
    output["pred_loss"] = torch.nn.functional.mse_loss(u_t, v_t, reduction="mean") 
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_fm")
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
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_flow_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=10,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
