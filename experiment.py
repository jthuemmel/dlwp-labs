"""One run of the project: train a configuration, test it on 2017 to 2019, write the scores and the figures.

    python experiment.py --config configs/baseline.yaml --seed 1

"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from lightning.pytorch.loggers import CSVLogger

from prepare_eval_data import climatology_store, eval_store, stats_store
from utils.components import count_parameters
from utils.config import Config
from utils.dataset import WeatherDataset
from utils.lightning_module import ForecastModule, forecasts_to_xarray, plot_metrics
from utils.metrics import acc, rmse, scores, truth_at

xr.set_options(use_bottleneck=False)

### PROTOCOL
VAL_YEAR = "2016"
TEST_YEARS = ("2017", "2019")
REQUIRED = ("Z500", "T850")


class Experiment:
    def __init__(self, config_path: str, seed: int = None):
        cfg = yaml.safe_load(open(config_path))
        self.settings = cfg.get("experiment", {})
        self.config = Config.from_dict(cfg)
        self.seed = seed if seed is not None else cfg.get("seed", 0)
        self.name = self.settings.get("name", Path(config_path).stem)
        self.out_dir = Path(self.settings.get("out_dir", "runs"))
        self.run_dir = self.out_dir / self.name / f"seed{self.seed}"

    # YOUR CHOICE OF EVALUATION GOES HERE:
    def evaluation_plots(self, forecasts: xr.Dataset, truth: xr.Dataset, climatology: xr.Dataset) -> None:
        # forecasts keep the member axis, truth is ERA5 at the valid times, climatology is indexed by (hour, dayofyear).
        # Figures belong in self.run_dir; the verification of labs 1, 3, and 4 is in utils/metrics.py.
        pass

    # SETUP
    def setup(self) -> None:
        L.seed_everything(self.seed, workers=True)
        data_cfg, trainer_cfg = self.config.dataset, self.config.trainer

        # the reporting protocol
        stop = (data_cfg.time_slice or {}).get("stop")
        assert stop is not None and str(stop) <= "2015", "training data stop with 2015 at the latest"
        assert all(v in data_cfg.variables for v in REQUIRED), f"the report needs {', '.join(REQUIRED)}"
        assert self.settings["eval_path"] not in (data_cfg.path, data_cfg.stats_path), \
            "the evaluation store is separate from the training data and from its statistics"

        self.eval_path = eval_store(self.settings["eval_path"], data_cfg.variables, source=data_cfg.path)
        self.clim_path = climatology_store(self.settings["climatology_path"], data_cfg.variables)
        stats_store(data_cfg.stats_path, data_cfg.path, data_cfg.variables, data_cfg.time_slice)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        resolved = {"seed": self.seed, "experiment": {**self.settings, "name": self.name}, **self.config.to_dict()}
        yaml.safe_dump(resolved, open(self.run_dir / "config.yaml", "w"), sort_keys=False)

        # the module builds a validation dataset from the training store, so it is given the last month there
        last = str(xr.open_zarr(data_cfg.path).time.values[-1])[:7]
        self.module = ForecastModule(replace(self.config, trainer=replace(trainer_cfg, val_time_slice={"start": last, "stop": last})))
        self.module.val_dataset = self.evaluation_dataset(VAL_YEAR, VAL_YEAR)

        train_time = self.module.train_dataset.dataset.time.values
        print(f"training on {str(train_time[0])[:10]} to {str(train_time[-1])[:10]}, {len(self.module.train_dataset)} windows; "
              f"validation {VAL_YEAR}, test {TEST_YEARS[0]} to {TEST_YEARS[1]}")

    def evaluation_dataset(self, start: str, stop: str) -> WeatherDataset:
        # stats_path stays the training statistics, so the evaluation years are standardised with them
        return WeatherDataset(replace(self.config.dataset, path=str(self.eval_path),
                                      time_slice={"start": start, "stop": stop},
                                      sequence_length=self.config.trainer.rollout_steps + 1))

    # TRAIN
    def train(self) -> None:
        cfg = self.config.trainer
        self.trainer = L.Trainer(
            max_steps=cfg.max_steps,
            accelerator="auto",
            logger=CSVLogger(self.run_dir, name="", version=""),
            log_every_n_steps=10,
            val_check_interval=max(1, cfg.max_steps // 10),
            check_val_every_n_epoch=None,
            limit_val_batches=10,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        self.trainer.fit(self.module)
        self.trainer.logger.save()
        if self.settings.get("save_checkpoint", True):
            self.trainer.save_checkpoint(self.run_dir / "model.ckpt")

    # TEST
    def test(self) -> xr.Dataset:
        path = self.run_dir / "forecasts.zarr"
        for i, year in enumerate(range(int(TEST_YEARS[0]), int(TEST_YEARS[1]) + 1)):
            self.module.val_dataset = self.evaluation_dataset(str(year), str(year))
            forecasts = forecasts_to_xarray(self.module, self.trainer.predict(self.module)).chunk({"time": 32})
            forecasts.to_zarr(path, mode="w") if i == 0 else forecasts.to_zarr(path, append_dim="time", align_chunks=True)
        return xr.open_zarr(path)

    # EVAL
    def evaluate(self, forecasts: xr.Dataset) -> None:
        era5 = xr.open_zarr(self.eval_path)
        clim = xr.open_zarr(self.clim_path).load()
        pred = forecasts.mean("number") if "number" in forecasts.dims else forecasts
        truth = truth_at(era5, pred)
        persistence = era5[list(pred.data_vars)].sel(time=pred.time).broadcast_like(pred)

        table = scores(pred, era5, clim)
        extra = xr.concat([rmse(persistence, truth), acc(pred, truth, clim)],
                          dim=xr.DataArray(["rmse_persistence", "acc"], dims="score", name="score")).compute()
        self.write_scores(xr.concat([table, extra], dim="score"))

        plot_forecast(pred, truth, self.run_dir)
        plot_loss(self.run_dir, f"{self.name} seed {self.seed}")
        plot_rmse(self.out_dir)
        self.evaluation_plots(forecasts, truth, clim)

    def write_scores(self, table: xr.Dataset) -> None:
        df = table.reset_coords(drop=True).to_dataframe().reset_index()
        df = df.melt(id_vars=["score", "prediction_timedelta"], var_name="variable", value_name="value")
        df["lead_hours"] = df.pop("prediction_timedelta") / np.timedelta64(1, "h")
        df["run"], df["seed"] = self.name, self.seed
        df = df.rename(columns={"score": "metric"})
        df[["run", "seed", "variable", "lead_hours", "metric", "value"]].to_csv(self.run_dir / "scores.csv", index=False)

    # RUN
    def run(self) -> None:
        self.setup()
        t0 = time.time()
        self.train()
        t1 = time.time()
        forecasts = self.test()
        t2 = time.time()
        self.evaluate(forecasts)
        self.record(train_seconds=t1 - t0, test_seconds=t2 - t1)

    def record(self, train_seconds: float, test_seconds: float) -> None:
        record = {
            "run": self.name,
            "seed": self.seed,
            "device": str(self.trainer.strategy.root_device),
            "parameters": count_parameters(self.module.model),
            "steps": self.config.trainer.max_steps,
            "train_seconds": round(train_seconds, 1),
            "test_seconds": round(test_seconds, 1),
        }
        json.dump(record, open(self.run_dir / "run.json", "w"), indent=2)
        print(f"{self.name} seed {self.seed}: {record['parameters']} parameters, {record['steps']} steps on "
              f"{record['device']}, trained in {train_seconds:.0f} s, tested in {test_seconds:.0f} s, in {self.run_dir}")


### FIGURES
def plot_forecast(forecasts: xr.Dataset, truth: xr.Dataset, run_dir, var: str = "T850", lead: int = 24):
    sel = dict(time=forecasts.time[0], prediction_timedelta=np.timedelta64(lead, "h"))
    pred, obs = forecasts[var].sel(**sel), truth[var].sel(**sel)
    kw = dict(vmin=float(obs.min()), vmax=float(obs.max()), cmap="RdBu_r")
    plt.figure(figsize=(12, 3.2))
    plt.subplot(131)
    pred.plot(**kw)
    plt.title(f"{var} forecast, {lead} h")
    plt.subplot(132)
    obs.plot(**kw)
    plt.title("ERA5")
    plt.subplot(133)
    (pred - obs).plot(cmap="RdBu_r", center=0)
    plt.title("error")
    plt.tight_layout()
    plt.savefig(Path(run_dir) / f"forecast_{var.lower()}_{lead}h.png")
    plt.close()


def plot_loss(run_dir, label: str):
    plot_metrics(run_dir, label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(run_dir) / "loss_curves.png")
    plt.close()


def plot_rmse(out_dir, variables: tuple = REQUIRED, days: int = 5):
    df = pd.concat(map(pd.read_csv, sorted(Path(out_dir).glob("*/*/scores.csv"))))
    df = df[df["lead_hours"] <= days * 24]
    _, axes = plt.subplots(1, len(variables), figsize=(5.5 * len(variables), 4), squeeze=False)
    for ax, var in zip(axes[0], variables):
        d = df[df["variable"] == var]
        for (run, seed), g in d[d["metric"] == "rmse"].groupby(["run", "seed"]):
            ax.plot(g["lead_hours"] / 24, g["value"], marker="o", markersize=3, label=f"{run}, seed {seed}")
        for metric, style in (("rmse_persistence", "k--"), ("rmse_climatology", "k:")):
            baseline = d[d["metric"] == metric].groupby("lead_hours")["value"].mean()
            ax.plot(baseline.index / 24, baseline.values, style, lw=1, label=metric.split("_")[1])
        ax.set_xlabel("lead time / days")
        ax.set_ylabel(f"RMSE {var}")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"rmse_{'_'.join(v.lower() for v in variables)}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train and test one configuration of the ablation study")
    parser.add_argument("--config", type=str, nargs="+", required=True, help="the config files to run, in order")
    parser.add_argument("--seed", type=int, default=None, help="overrides the seed of the config file")
    args = parser.parse_args()

    for config in args.config:
        Experiment(config, seed=args.seed).run()


if __name__ == "__main__":
    main()
