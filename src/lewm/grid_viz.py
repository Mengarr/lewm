from pathlib import Path

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


def _to_imshow(x):
    """(C,H,W) in [-1,1] -> (H,W,C) float in [0,1]."""
    return (x.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


class GridSaveCallback(Callback):
    """Saves a PNG grid (rows=fixed val samples, cols=[target, ...predict_fn's other keys])
    every epoch_interval epochs."""

    def __init__(self, fixed_batch, cfg, predict_fn, output_dir, epoch_interval: int = 1, log_wandb: bool = False):
        super().__init__()
        self.fixed_batch = fixed_batch
        self.cfg = cfg
        self.predict_fn = predict_fn
        self.output_dir = Path(output_dir)
        self.epoch_interval = epoch_interval
        self.log_wandb = log_wandb

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval != 0 and epoch != trainer.max_epochs:
            return

        self._save_grid(pl_module, epoch)

    def _save_grid(self, pl_module, epoch):
        device = pl_module.device
        batch = {k: v.to(device) for k, v in self.fixed_batch.items()}
        model = getattr(pl_module.model, "_orig_mod", pl_module.model)

        preds = self.predict_fn(model, batch, self.cfg)

        fig = self._build_figure(preds)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"grid_epoch{epoch}.png"
        fig.savefig(out_path, dpi=150)

        if self.log_wandb:
            self._log_to_wandb(fig, epoch)

        plt.close(fig)

    def _build_figure(self, preds):
        other_keys = [k for k in preds if k != "target"]
        col_titles = ["target"] + [f"history={k}" if isinstance(k, int) else str(k) for k in other_keys]
        cols = [preds["target"]] + [preds[k] for k in other_keys]
        n_cols = len(cols)
        n_rows = cols[0].size(0)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_rows == 1:
            axes = axes.reshape(1, n_cols)
        for row in range(n_rows):
            for col_idx, col in enumerate(cols):
                ax = axes[row, col_idx]
                ax.imshow(_to_imshow(col[row]))
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(col_titles[col_idx])
        fig.tight_layout()
        return fig

    def _log_to_wandb(self, fig, epoch):
        try:
            import wandb
        except ImportError:
            return
        wandb.log({"val/grid": wandb.Image(fig)})
