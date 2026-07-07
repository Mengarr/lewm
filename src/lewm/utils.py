import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        model = getattr(model, '_orig_mod', model)
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


class EMACallback(Callback):
    """Maintains an exponential moving average of the model weights and saves
    them alongside the regular checkpoint, for use at eval/rollout time."""

    def __init__(self, run_name, cfg, decay: float = 0.999, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.decay = decay
        self.epoch_interval = epoch_interval
        self.shadow = {}

    def on_fit_start(self, trainer, pl_module):
        if not self.shadow:
            model = getattr(pl_module.model, '_orig_mod', pl_module.model)
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = getattr(pl_module.model, '_orig_mod', pl_module.model)
        for k, v in model.state_dict().items():
            shadow_v = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow_v.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                shadow_v.copy_(v)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        model = getattr(model, '_orig_mod', model)
        live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}_ema.pt',
        )
        model.load_state_dict(live_state)

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]