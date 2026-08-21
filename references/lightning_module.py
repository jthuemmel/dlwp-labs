from dataclasses import replace
from functools import partial

import einops
import lightning as L
import torch
from torch.utils.data import DataLoader, Subset

from utils import loss_fn
from utils.components import ViT
from utils.config import Config
from utils.dataset import WeatherData


class ForecastModule(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.to_dict())

        assert self.cfg.val_time_slice is not None, "the validation years are named in the configuration"
        self.train_dataset = WeatherData(self.data_cfg)
        self.val_dataset = WeatherData(replace(self.data_cfg, time_slice=self.cfg.val_time_slice,
                                               sequence_length=self.cfg.rollout_steps + 1))
        self.model = ViT(self.model_cfg, self.world)
        self.loss_fn = partial(getattr(loss_fn, f"f_{self.objective.loss}"), **self.objective.kwargs)

    @property
    def cfg(self):
        return self.config.trainer

    @property
    def data_cfg(self):
        return self.config.dataset

    @property
    def model_cfg(self):
        return self.config.network

    @property
    def world(self):
        return self.config.world

    @property
    def objective(self):
        return self.config.objective

    @property
    def per_variable_weights(self) -> torch.FloatTensor:
        weights = self.objective.weights or {}
        w = torch.as_tensor([weights.get(var, 1.) for var in self.data_cfg.variables], dtype=torch.float32, device=self.device)
        return einops.repeat(w, f"(v vv) -> {self.world.field_pattern}",
                             **self.world.token_sizes, **self.world.patch_sizes)

    @property
    def area_weights(self) -> torch.FloatTensor:
        latitude = torch.as_tensor(self.train_dataset.dataset.latitude.values, dtype=torch.float32, device=self.device)
        w = torch.cos(torch.deg2rad(latitude))
        return einops.repeat(w / w.mean(), f"(h hh) -> {self.world.field_pattern}",
                             **self.world.token_sizes, **self.world.patch_sizes)

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(state)

    def rollout(self, state: torch.FloatTensor, steps: int) -> torch.FloatTensor:
        out = []
        for _ in range(steps):
            state = self.forward(state)
            out.append(state)
        return torch.stack(out, dim=2)

    def step_loss(self, observation: torch.FloatTensor, prediction: torch.FloatTensor) -> torch.FloatTensor:
        score = self.loss_fn(observation, prediction)
        return score.mul(self.per_variable_weights).mul(self.area_weights).mean()

    def training_step(self, batch, batch_idx):
        states, _ = batch
        steps = 1 if self.global_step < self.cfg.pre_steps else self.cfg.train_rollout_steps
        prediction = self.rollout(states[:, :, 0], steps)
        loss = sum(self.step_loss(states[:, :, k + 1], prediction[:, :, k]) for k in range(steps)) / steps
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, _ = batch
        prediction = self.rollout(states[:, :, 0], self.cfg.rollout_steps)
        for k in range(self.cfg.rollout_steps):
            self.log(f"val/loss_step{k + 1}", self.step_loss(states[:, :, k + 1], prediction[:, :, k]),
                     prog_bar=(k == 0))

    def predict_step(self, batch, batch_idx):
        states, _ = batch
        return self.rollout(states[:, :, 0], self.cfg.predict_steps).cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                                      betas=tuple(self.cfg.betas))
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": self.create_scheduler(optimizer), "interval": "step"}}

    def create_scheduler(self, optimizer):
        schedulers, milestones, total = [], [], 0
        for sch_cfg in self.cfg.schedulers:
            typ, steps = sch_cfg["type"].lower(), sch_cfg["steps"]
            total += steps
            milestones.append(total)
            if typ == "linear":
                sched = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=sch_cfg.get("start_factor", 1.0),
                    end_factor=sch_cfg.get("end_factor", 1.0), total_iters=steps)
            elif typ == "constant":
                sched = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=sch_cfg.get("factor", 1.0), total_iters=steps)
            elif typ == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=steps, eta_min=sch_cfg.get("eta_min", 0.0))
            else:
                raise ValueError(f"Unknown scheduler type: {typ}")
            schedulers.append(sched)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones[:-1])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.subset(self.val_dataset, self.cfg.rollout_steps, self.cfg.val_stride),
                          batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.subset(self.val_dataset, self.cfg.predict_steps, self.cfg.predict_stride),
                          batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    @staticmethod
    def subset(dataset: WeatherData, steps: int, stride: int) -> Subset:
        stop = min(dataset.dataset.sizes["time"] - steps, len(dataset))
        return Subset(dataset, range(0, stop, stride))
